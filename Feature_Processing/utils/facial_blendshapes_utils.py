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


# helper to draw landmarks
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


def compute_video_facial_landmarks(video_path: str, gloss: str, show_landmarks: bool = False):
    """
    :param video_path: path to video file
    :param gloss: gloss label for the video
    :param show_landmarks:  whether to display landmarks
    :return: DataFrame of facial blendshapes, or empty DataFrame if failed
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return pd.DataFrame()

    video_id = os.path.basename(video_path)



    # face landmarker options
    base_options = python.BaseOptions(model_asset_path=r'C:\Users\boulosge\Desktop\Acht\Feature_Processing\utils\face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # For saving annotated video
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
    video_output_path = None
    if show_landmarks:
        temp_dir = os.getenv("FACIAL_LANDMARKS_TMP_DIR", "facial_landmarks_tmp")
        os.makedirs(temp_dir, exist_ok=True)
        video_output_path = os.path.join(temp_dir, f"{video_id}_landmarked.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 25  # default fallback
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

    cap = cv2.VideoCapture(video_path)
    frames_data = []
    frame_index = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        detection_result = detector.detect_for_video(mp_image, timestamp_ms)

        if detection_result.face_blendshapes:
            blendshapes = detection_result.face_blendshapes[0]
            # row = {"frame": frame_index, "timestamp_ms": timestamp_ms}
            row = {}
            for cat in blendshapes:
                row[cat.category_name] = cat.score
            frames_data.append(row)

        if show_landmarks:
            annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)
            frame_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        frame_index += 1

    # release resources
    cap.release()
    detector.close()
    if show_landmarks:
        out.release()

    if not frames_data:
        print(f"⚠️ No landmarks detected in video {video_id}")
        return pd.DataFrame()

    # after processing, create or append parquet
    new_df = pd.DataFrame(frames_data)
    vid_id_clean = video_id.split(".")[0]
    new_df["video_id"] = vid_id_clean
    new_df["gloss"] = gloss
    return new_df, video_output_path


if __name__ == "__main__":
    # TESTING ...
    test_video_path = os.getenv('TEST_VIDEO_PATH')
    # Test landmarks extraction
    facial_blendshapes_df, vis_path = compute_video_facial_landmarks(test_video_path, "drink", True)
    print("✅ Extracted landmarks. Shape of dataframe:", facial_blendshapes_df.shape)
    print(facial_blendshapes_df.head())
    # Test returned visualization path
    if vis_path and os.path.exists(vis_path):
        print(f"✅ Landmarked video saved at: {vis_path}")
        # Optional: open with default video player (Windows)
        os.system(f'start {vis_path}')
    else:
        print("❌ Landmark visualization video not found.")

