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
import time
from tqdm import tqdm
import tempfile

try:
    # This works if your project is structured correctly and in the Python path
    from ViTPose.ViTPose import pose_utils
    from TRANSFORMERS.src.Training.models.LateFusionLogitEncoderWeightedSumNoLinearProjection import LateFusionEncoder
except ImportError:
    # Fallback for running the script directly
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

# --- Load Your Winning Late Fusion Model ---
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

# --- Load Class Mappings ---
IDX2GLOSS = {str(i): f"GLOSS_{i}" for i in range(NUM_CLASSES)}
if os.path.exists("idx2gloss.json"):
    with open("idx2gloss.json", "r") as f:
        IDX2GLOSS = json.load(f)
    print("... Index to Gloss mapping loaded.")

print("--- Initialization Complete. Gradio is starting. ---")


# ==============================================================================
# 2. CORE PROCESSING FUNCTION
# ==============================================================================
def process_video(video_path, show_pose, show_hands, show_face, progress=gr.Progress(track_tqdm=True)):
    """Main function to process a video file, run inference, and generate outputs."""
    if video_path is None:
        return None, "Please upload a video first.", None

    # --- Part 1: Run ViTPose and Annotate Video ---
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS);
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- CRITICAL FIX for NAN VIDEO BUG ---
    # Do NOT use a `with` statement. `delete=False` ensures the file persists
    # so Gradio can serve it to the user.
    temp_video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_video_path = temp_video_file.name
    temp_video_file.close()  # Close the handle so VideoWriter can use the file
    # --- END FIX ---

    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    all_features_for_model = []

    for frame_idx in tqdm(range(total_frames), desc="Analyzing video..."):
        ret, frame = cap.read()
        if not ret:
            break

        pose_results = pose_utils.process_single_frame(DET_MODEL, POSE_MODEL, DATASET_INFO, frame)

        pose_rows, face_rows, hand_rows = pose_utils.get_pose_data(
            pose_results, "demo", frame_idx, frame_width=width, frame_height=height
        )

        # Robustly handle frames where no person is detected
        if hand_rows and hand_rows[0].get('person_id', -1) != -1:
            all_features_for_model.append({'pose': pose_rows[0], 'face': face_rows[0], 'hand': hand_rows[0]})
        else:
            # Append a row of NaNs to maintain sequence length
            all_features_for_model.append({'pose': {}, 'face': {}, 'hand': {}})

        vis_frame = pose_utils.draw_landmarks_on_frame(frame, pose_results, draw_pose=show_pose, draw_hands=show_hands,
                                                       draw_face=show_face)
        video_writer.write(vis_frame)

    cap.release();
    video_writer.release()

    # --- Part 2: Run the Late Fusion Recognition Model ---
    if not all_features_for_model:
        return output_video_path, "No features were extracted from the video.", None

    # --- Robustly Prepare the feature tensor ---
    hand_cols_needed = [f'h{i}_{ax}' for i in range(42) for ax in 'xy']
    pose_cols_needed = [f'p{i}_{ax}' for i in range(13) for ax in 'xy']
    face_cols_simulated_needed = [f'f{i}_{ax}' for i in range(26) for ax in 'xy']

    hand_df = pd.DataFrame([d['hand'] for d in all_features_for_model])[hand_cols_needed]
    pose_df = pd.DataFrame([d['pose'] for d in all_features_for_model])[pose_cols_needed]
    face_df = pd.DataFrame([d['face'] for d in all_features_for_model])[face_cols_simulated_needed]

    final_df = pd.concat([hand_df, pose_df, face_df], axis=1).fillna(-2.0)  # Fill NaNs from missed frames

    feature_tensor = torch.tensor(final_df.values, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # --- Run Inference ---
    with torch.no_grad():
        logits = LATE_FUSION_MODEL(feature_tensor)
        probabilities = torch.softmax(logits, dim=1)
        prediction_idx = torch.argmax(probabilities, dim=1).item()
        # Use str(key) for lookup as JSON keys are strings
        prediction_gloss = IDX2GLOSS.get(str(prediction_idx), "Unknown")
        confidence = probabilities[0, prediction_idx].item()
        weights = torch.softmax(LATE_FUSION_MODEL.fusion_weights, dim=0).detach().cpu().numpy()

    prediction_text = f"Prediction: {prediction_gloss}\nConfidence: {confidence:.2%}"

    # --- Part 3: Create the Weights Plot ---
    modalities = ['Hands', 'Pose', 'Face']
    fig, ax = plt.subplots()
    ax.bar(modalities, weights, color=['#55a868', '#4c72b0', '#c44e52'])
    ax.set_ylabel('Learned Importance');
    ax.set_title('Adaptive Modality Weights');
    ax.set_ylim(0, 1)
    for i, w in enumerate(weights):
        if w > 0.03: ax.text(i, w / 2, f'{w:.2%}', ha='center', va='center', color='white', fontweight='bold')

    return output_video_path, prediction_text, fig, video_path


# ==============================================================================
# 4. GRADIO UI DEFINITION
# ==============================================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Adaptive Late Fusion for ISLR Demo")
    gr.Markdown(
        "Drag and drop an AVASAG video to run inference. Use the checkboxes to control which landmarks are visualized in the output video.")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload AVASAG Video")
            with gr.Row():
                show_pose = gr.Checkbox(label="Show Pose", value=True)
                show_hands = gr.Checkbox(label="Show Hands", value=True)
                show_face = gr.Checkbox(label="Show Face", value=False)

            with gr.Row():
                analyze_button = gr.Button("Analyze Video", variant="primary")
                clear_button = gr.ClearButton(value="Clear All")

        with gr.Column(scale=2):
            video_output = gr.Video(label="Annotated Video")
            with gr.Row():
                prediction_output = gr.Textbox(label="Model Prediction")
                weights_plot = gr.Plot(label="Learned Modality Weights")

    analyze_button.click(
        fn=process_video,
        inputs=[video_input, show_pose, show_hands, show_face],
        outputs=[video_output, prediction_output, weights_plot, video_input]
    )

    # Connect the clear button to all inputs and outputs
    clear_button.click(lambda: (None, True, True, False, None, "", None),
                       outputs=[video_input, show_pose, show_hands, show_face, video_output, prediction_output,
                                weights_plot])

if __name__ == "__main__":
    demo.launch(debug=True)