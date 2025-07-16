import mediapipe as mp
import os
import cv2
import numpy as np
import pandas as pd
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from dotenv import load_dotenv
load_dotenv()
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

known_blendshapes = [
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut"
]

# helper to draw avasag_vitpose_extracted_landmarks
def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())
    return annotated_image

def compute_video_facial_blendshapes(video_path: str, gloss: str, show_landmarks: bool = False):
    """
    :param video_path: path to video file
    :param gloss: gloss label for the video
    :param show_landmarks: whether to display avasag_vitpose_extracted_landmarks
    :return: DataFrame of facial blendshapes with one row per frame (padded with -2 where missing), optional video path, and failed to extract avasag_vitpose_extracted_landmarks flag
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return pd.DataFrame(), None, True

    video_id = os.path.basename(video_path)

    # get video resolution
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(f"🖼️ Video resolution: {frame_width} x {frame_height}")

    # face landmarker options
    base_options = python.BaseOptions(
        model_asset_path=r'C:\Users\boulosge\Desktop\Acht\Feature_Processing\utils\face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.25
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # video output if avasag_vitpose_extracted_landmarks shown
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_output_path = None
    if show_landmarks:
        temp_dir = os.getenv("FACIAL_LANDMARKS_TMP_DIR", "facial_landmarks_tmp")
        os.makedirs(temp_dir, exist_ok=True)
        video_output_path = os.path.join(temp_dir, f"{video_id}_landmarked.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 25
        out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

    frames_data = []
    frame_index = 0
    total_frames = 0
    detected_frame_indices = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        total_frames += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        detection_result = detector.detect_for_video(mp_image, timestamp_ms)

        if detection_result.face_blendshapes:
            blendshapes = detection_result.face_blendshapes[0]
            # print(blendshapes)
            row = {"frame": frame_index}
            for cat in blendshapes:
                row[cat.category_name] = cat.score
            frames_data.append(row)
            detected_frame_indices.append(frame_index)

            if show_landmarks:
                annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)
                frame_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

        frame_index += 1

    cap.release()
    detector.close()

    if show_landmarks:
        out.release()

    # print(f"\n🔎 Stats summary:")
    # print(f"1️⃣ Total frames processed: {total_frames}")
    # print(f"2️⃣ Number of frames with blendshapes detected: {len(detected_frame_indices)}")
    # print(f"3️⃣ Indices of frames with detected blendshapes: {detected_frame_indices}")

    # -------------------------
    # if no detections at all
    # -------------------------
    if not frames_data:
        print(f"⚠️ No facial avasag_vitpose_extracted_landmarks detected at all in video {video_id}")
        padded_data = []
        for idx in range(total_frames):
            row = {"frame": idx}
            for key in known_blendshapes:
                row[key] = -2
            row["video_id"] = video_id.split('.')[0]
            row["gloss"] = gloss
            padded_data.append(row)

        final_df = pd.DataFrame(padded_data)
        cols = ["video_id", "gloss"] + [c for c in final_df.columns if c not in ["video_id", "gloss", "frame", "_neutral"]]
        final_df = final_df[cols]
        return final_df, None, True

    # if at least one detection, pad other frames with -2
    blendshape_keys = [k for k in frames_data[0].keys() if k != "frame"]
    padded_data = []
    for idx in range(total_frames):
        row = {"frame": idx}
        for key in blendshape_keys:
            row[key] = -2
        row["video_id"] = video_id.split('.')[0]
        row["gloss"] = gloss
        padded_data.append(row)

    # fill in detected
    for detected in frames_data:
        idx = detected["frame"]
        for key in blendshape_keys:
            padded_data[idx][key] = detected[key]

    final_df = pd.DataFrame(padded_data)
    # move video_id and gloss to the first columns
    cols = ["video_id", "gloss"] + [c for c in final_df.columns if c not in ["video_id", "gloss", "frame", "_neutral"]]
    final_df = final_df[cols]

    return final_df, video_output_path, False

def summarize_single_video_facial_blendshapes(video_facial_blendshapes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes one video's frame-wise facial blendshapes into a single-row DataFrame, ignoring -2 (missing) values.

    :param video_facial_blendshapes_df: Frame-wise blendshapes data for a single video
    :return: DataFrame with one row of video-level statistical top_features
    """
    video_id = video_facial_blendshapes_df['video_id'].iloc[0]
    gloss = video_facial_blendshapes_df['gloss'].iloc[0]

    feature_cols = [col for col in video_facial_blendshapes_df.columns if col not in ['video_id', 'gloss']]
    summary_data = {
        'video_id': video_id,
        'gloss': gloss
    }

    for col in feature_cols:
        values = video_facial_blendshapes_df[col].astype(float).values
        valid_values = values[values != -2]

        if len(valid_values) == 0:
            # All values were missing, pad with NaN
            summary_data[f"{col}_mean"] = -2
            summary_data[f"{col}_std"] = -2
            summary_data[f"{col}_min"] = -2
            summary_data[f"{col}_max"] = -2
            summary_data[f"{col}_var"] = -2
        else:
            summary_data[f"{col}_mean"] = np.mean(valid_values)
            summary_data[f"{col}_std"] = np.std(valid_values)
            summary_data[f"{col}_min"] = np.min(valid_values)
            summary_data[f"{col}_max"] = np.max(valid_values)
            summary_data[f"{col}_var"] = np.var(valid_values)

    return pd.DataFrame([summary_data])


if __name__ == "__main__":
    # TESTING ...
    test_video_path = os.getenv('TEST_VIDEO_PATH')
    # Test avasag_vitpose_extracted_landmarks extraction
    facial_blendshapes_df, vis_path, failed = compute_video_facial_blendshapes(test_video_path, "drink", True)
    if not failed:
        print("✅ Shape of output dataframe:", facial_blendshapes_df.shape)
        print(facial_blendshapes_df.head())
        # Test returned visualization path
        if vis_path and os.path.exists(vis_path):
            print(f"✅ Landmarked video saved at: {vis_path}")
            # Optional: open with default video player (Windows)
            os.system(f'start {vis_path}')
        else:
            print("❌ Landmark visualization video not found.")
    else:
        print("❌ Landmark extraction failed.")

    # Test summarizing avasag_vitpose_extracted_landmarks across frames using stat metrics
    vid_landmarks_summary = summarize_single_video_facial_blendshapes(facial_blendshapes_df)
    print(vid_landmarks_summary)
    print(f"Number of padded metrics with -2: {list(vid_landmarks_summary.iloc[0].values).count(-2)}")
