import gradio as gr
import torch
import cv2
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import tempfile
import threading

# ==============================================================================
# 0. Imports / Model modules
# ==============================================================================
try:
    from ViTPose.ViTPose import pose_utils
    from TRANSFORMERS.src.Training.models.LateFusionLogitEncoderWeightedSumNoLinearProjection import LateFusionEncoder
except ImportError:
    print("Adding project root to Python path...")
    project_root = Path(__file__).resolve().parents[1]  # Adjust if your demo is in a different subdir
    sys.path.insert(0, str(project_root))
    from ViTPose.ViTPose import pose_utils
    from TRANSFORMERS.src.Training.models.LateFusionLogitEncoderWeightedSumNoLinearProjection import LateFusionEncoder

# ==============================================================================
# 1. GLOBAL SETUP (Loads models only ONCE when the script starts)
# ==============================================================================
print("--- Initializing Models (this may take a moment)... ---")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load ViTPose Model using pose_utils ---
VITPOSE_BASE_DIR = r"C:\Users\boulosge\Desktop\Acht\ViTPose\ViTPose"
DET_CONFIG = os.path.join(VITPOSE_BASE_DIR, 'demo/mmdetection_cfg/yolox_l_8x8_300e_coco.py')
DET_CHECKPOINT = os.path.join(VITPOSE_BASE_DIR, 'models/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth')
POSE_CONFIG = os.path.join(VITPOSE_BASE_DIR,
                           'configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_large_wholebody_256x192.py')
POSE_CHECKPOINT = os.path.join(VITPOSE_BASE_DIR, 'models/wholebody.pth')

DET_MODEL, POSE_MODEL, DATASET_INFO = pose_utils.load_models(
    DET_CONFIG, DET_CHECKPOINT, POSE_CONFIG, POSE_CHECKPOINT, DEVICE
)
print("... ViTPose models loaded successfully.")

# --- Paths / metadata ---
METADATA_FILE = r"C:\Users\boulosge\Desktop\Acht\AVASAG_feature_processing_vitpose\avasag_metadata.json"

# --- Load Your Late Fusion Model ---
HAND_INPUT_DIM = 84
POSE_INPUT_DIM = 26
FACE_INPUT_DIM = 52
HIDDEN_DIM = 128
NUM_CLASSES = 100

LATE_FUSION_MODEL = LateFusionEncoder(
    hand_input_dim=HAND_INPUT_DIM, pose_input_dim=POSE_INPUT_DIM,
    face_input_dim=FACE_INPUT_DIM, num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM
)
LATE_FUSION_CHECKPOINT = r"C:\Users\boulosge\Desktop\Acht\TRANSFORMERS\src\Training\out-checkpoints\vitpose\LateFusionLogitEncoderWeightedSumNoLinearProjection_avasag\final_model.pth"
if os.path.exists(LATE_FUSION_CHECKPOINT):
    LATE_FUSION_MODEL.load_state_dict(torch.load(LATE_FUSION_CHECKPOINT, map_location=DEVICE))
    print("... Late Fusion model loaded successfully.")
else:
    print(f"[WARNING] Late Fusion checkpoint not found at: {LATE_FUSION_CHECKPOINT}.")
LATE_FUSION_MODEL.to(DEVICE)
LATE_FUSION_MODEL.eval()

print("--- Initialization Complete. Gradio is starting. ---")


# ==============================================================================
# 1. Index-to-gloss mapping from AVASAG metadata
# ==============================================================================
def build_idx2gloss(metadata_path):
    """
    Reads metadata JSON (list of entries with {"gloss": ..., "instances": [...] }),
    sorts by gloss alphabetically, and returns {index(int): gloss(str)}.
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    sorted_meta = sorted(metadata, key=lambda x: x["gloss"])
    return {idx: entry["gloss"] for idx, entry in enumerate(sorted_meta)}


# For faster lookups reuse the mapping (reload only if you change the file)
IDX2GLOSS_FROM_META = build_idx2gloss(METADATA_FILE)


# ==============================================================================
# 2. CORE PROCESSING FUNCTION
# ==============================================================================
def process_video(video_path_or_dict, show_pose, show_hands, show_face,
                  progress=gr.Progress(track_tqdm=True), flip_for_model=False):
    """Process a video file, run inference, and generate annotated output + prediction + weights."""

    def _normalize_video_input(v):
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            return v.get("name")
        return None

    video_path = _normalize_video_input(video_path_or_dict)
    if not video_path or not os.path.exists(video_path):
        return None, "Please provide a valid video file.", None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Could not open video.", None

    # --- Robustly read basic props ---
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(raw_fps) if raw_fps is not None else 0.0
    if (not np.isfinite(fps)) or (fps <= 1.0):  # treat NaN, inf, 0/1 as invalid
        fps = 25.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w <= 0 or h <= 0:
        # Peek a frame to infer size
        ok, first = cap.read()
        if not ok:
            cap.release()
            return None, "Failed to read any frames from the video.", None
        h, w = first.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

    # --- Prepare annotated writer ---
    temp_video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_video_path = temp_video_file.name
    temp_video_file.close()

    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        return None, "Failed to open video writer (check codecs).", None

    all_features_for_model = []

    # Use tqdm without trusting CAP_PROP_FRAME_COUNT (often 0/NaN for uploads)
    pbar = tqdm(desc="Analyzing video...", total=None)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip ONLY IF requested (webcam path); uploads will pass False
        input_frame = cv2.flip(frame, 1) if flip_for_model else frame

        # Run pose on the (maybe flipped) frame
        pose_results = pose_utils.process_single_frame(DET_MODEL, POSE_MODEL, DATASET_INFO, input_frame)

        pose_rows, face_rows, hand_rows = pose_utils.get_pose_data(
            pose_results, "demo", frame_idx, frame_width=w, frame_height=h
        )

        if hand_rows and hand_rows[0].get('person_id', -1) != -1:
            all_features_for_model.append({'pose': pose_rows[0], 'face': face_rows[0], 'hand': hand_rows[0]})
        else:
            all_features_for_model.append({'pose': {}, 'face': {}, 'hand': {}})

        # Draw on the same (maybe flipped) frame so preview matches model input
        vis_frame = pose_utils.draw_landmarks_on_frame(
            input_frame, pose_results,
            draw_pose=show_pose, draw_hands=show_hands, draw_face=show_face
        )
        writer.write(vis_frame)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()

    if not all_features_for_model:
        return output_video_path, "No features were extracted from the video.", None

    # --- Prepare tensors robustly (avoid KeyErrors; fill missing with -2.0) ---
    hand_cols = [f'h{i}_{ax}' for i in range(42) for ax in 'xy']  # 84
    pose_cols = [f'p{i}_{ax}' for i in range(13) for ax in 'xy']  # 26
    face_cols = [f'f{i}_{ax}' for i in range(26) for ax in 'xy']  # 52

    hand_df = pd.DataFrame([d['hand'] for d in all_features_for_model]).reindex(columns=hand_cols)
    pose_df = pd.DataFrame([d['pose'] for d in all_features_for_model]).reindex(columns=pose_cols)
    face_df = pd.DataFrame([d['face'] for d in all_features_for_model]).reindex(columns=face_cols)

    final_df = pd.concat([hand_df, pose_df, face_df], axis=1).fillna(-2.0)
    feature_tensor = torch.tensor(final_df.values, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = LATE_FUSION_MODEL(feature_tensor)
        probabilities = torch.softmax(logits, dim=1)
        prediction_idx = torch.argmax(probabilities, dim=1).item()
        prediction_gloss = IDX2GLOSS_FROM_META.get(prediction_idx, f"CLASS_{prediction_idx}")
        confidence = probabilities[0, prediction_idx].item()
        weights = torch.softmax(LATE_FUSION_MODEL.fusion_weights, dim=0).detach().cpu().numpy()

    prediction_text = f"Prediction: [{prediction_gloss}]\nConfidence: {confidence:.2%}"

    # --- Weights plot ---
    modalities = ['Hands', 'Pose', 'Face']
    fig, ax = plt.subplots()
    ax.bar(modalities, weights, color=['#55a868', '#4c72b0', '#c44e52'])
    ax.set_ylabel('Learned Importance')
    ax.set_title('Adaptive Modality Weights')
    ax.set_ylim(0, 1)
    for i, wv in enumerate(weights):
        if wv > 0.03:
            ax.text(i, wv / 2, f'{wv:.2%}', ha='center', va='center', color='white', fontweight='bold')

    return output_video_path, prediction_text, fig


# ==============================================================================
# 3. Robust OpenCV Recorder (Windows-friendly). Shows raw preview after stop.
# ==============================================================================
class OCVCamRecorder:
    def __init__(self):
        self.stop_evt = threading.Event()
        self.thread = None
        self.path = None
        self.running = False

    def _open_capture(self):
        # Try a few indices + backends to find a working camera
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for be in backends:
            for idx in range(3):
                cap = cv2.VideoCapture(idx, be)
                if cap.isOpened():
                    return cap, idx, be
                cap.release()
        return None, None, None

    def start(self, width=640, height=360, fps=24):
        if self.running:
            return "Already recording."

        cap, idx, be = self._open_capture()
        if cap is None:
            return "❌ Could not open camera. Close Zoom/Teams, or check camera permissions."

        # Best-effort camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # reduce CPU if supported
        except Exception:
            pass

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        path = tmp.name
        tmp.close()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, fps, (int(width), int(height)))
        if not writer.isOpened():
            cap.release()
            return "❌ Failed to open video writer."

        self.path = path
        self.stop_evt.clear()
        self.running = True

        def loop():
            while not self.stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (int(width), int(height)))
                writer.write(frame)
            writer.release()
            cap.release()
            self.running = False

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return f"🎥 Recording… (index={idx})"

    def stop(self):
        if not self.running:
            return None, "Not recording."
        self.stop_evt.set()
        self.thread.join()
        return self.path, "✅ Saved."


recorder = OCVCamRecorder()

# ==============================================================================
# 4. GRADIO UI
# ==============================================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Adaptive Late Fusion for ISLR — Record, Preview, Analyze")

    with gr.Tabs():
        # --- Tab 1: Video File Demo (upload -> annotated preview) ---
        with gr.TabItem("Video File Demo"):
            gr.Markdown("Upload a video, then run the model. The annotated result is previewed below.")

            with gr.Row():
                with gr.Column(scale=1):
                    file_video_input = gr.Video(label="Upload Video")
                    with gr.Row():
                        file_show_pose = gr.Checkbox(label="Show Pose", value=True)
                        file_show_hands = gr.Checkbox(label="Show Hands", value=True)
                        file_show_face = gr.Checkbox(label="Show Face", value=False)
                    with gr.Row():
                        file_analyze_button = gr.Button("Analyze Video", variant="primary")
                        file_clear_button = gr.ClearButton(value="Clear All")

                with gr.Column(scale=2):
                    file_annot_preview = gr.Video(label="Annotated Preview", height=360)
                    with gr.Row():
                        file_prediction_output = gr.Textbox(label="Model Prediction")
                        file_weights_plot = gr.Plot(label="Learned Modality Weights")

            file_analyze_button.click(
                fn=process_video,
                inputs=[file_video_input, file_show_pose, file_show_hands, file_show_face],
                outputs=[file_annot_preview, file_prediction_output, file_weights_plot]
            )

            file_clear_button.click(
                lambda: (None, True, True, False, None, "", None),
                outputs=[file_video_input, file_show_pose, file_show_hands, file_show_face,
                         file_annot_preview, file_prediction_output, file_weights_plot]
            )

        # --- Tab 2: Record (OpenCV) & Analyze (raw preview -> annotated preview) ---
        with gr.TabItem("Record (OpenCV) & Analyze"):
            gr.Markdown(
                "Record locally with OpenCV, **preview the raw clip**, then analyze and preview the **annotated** result.")

            with gr.Row():
                with gr.Column(scale=1):
                    ocv_status = gr.Textbox(label="Status", interactive=False)
                    ocv_start = gr.Button("Start Recording (OpenCV)", variant="primary")
                    ocv_stop = gr.Button("Stop & Show Raw Preview")

                    gr.Markdown("### Visualization Options")
                    ocv_show_pose = gr.Checkbox(label="Show Pose", value=True)
                    ocv_show_hands = gr.Checkbox(label="Show Hands", value=True)
                    ocv_show_face = gr.Checkbox(label="Show Face", value=True)

                    ocv_analyze = gr.Button("Analyze Last Recording", variant="secondary")

                with gr.Column(scale=2):
                    ocv_raw_preview = gr.Video(label="Recorded Preview (raw)", height=360)
                    ocv_annot_preview = gr.Video(label="Annotated Preview", height=360)
                    ocv_pred_out = gr.Textbox(label="Model Prediction")
                    ocv_weights = gr.Plot(label="Learned Modality Weights")

            # Start recording
            def _ocv_start():
                return recorder.start()

            ocv_start.click(fn=_ocv_start, outputs=[ocv_status])

            # Stop recording -> show raw preview
            def _ocv_stop():
                path, msg = recorder.stop()
                return path, msg


            ocv_stop.click(fn=_ocv_stop, outputs=[ocv_raw_preview, ocv_status])

            # Analyze -> annotated preview + prediction + weights
            def _analyze_last(show_pose, show_hands, show_face):
                p = recorder.path
                if not p or not os.path.exists(p):
                    return None, "No recording found.", None, None
                # Flip ONLY for webcam recordings
                ann_path, pred_text, weights_plot = process_video(
                    p, show_pose, show_hands, show_face, flip_for_model=False
                )
                return ann_path, pred_text, weights_plot, f"Analyzed: {os.path.basename(p)}"


            ocv_analyze.click(
                fn=_analyze_last,
                inputs=[ocv_show_pose, ocv_show_hands, ocv_show_face],
                outputs=[ocv_annot_preview, ocv_pred_out, ocv_weights, ocv_status]
            )

if __name__ == "__main__":
    demo.launch(debug=False)

