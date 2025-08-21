# extract_blendshapes.py
#
# A generalizable, all-in-one script to extract facial blendshapes from videos
# based on a metadata file. This version uses the superior, more aesthetic drawing function.

import os
import sys
import glob
import cv2
import json
import time
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Constants ---
KNOWN_BLENDSHAPES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft",
    "noseSneerRight", "tongueOut"
]


# --- THIS IS YOUR SUPERIOR DRAWING FUNCTION ---
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


def process_single_video(video_path, gloss, detector, save_annotation=False, output_dir=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        tqdm.write(f"  -> Error: Cannot open video {video_path}")
        return pd.DataFrame()

    video_id_clean = os.path.splitext(os.path.basename(video_path))[0]
    video_writer = None
    if save_annotation:
        if not output_dir: output_dir = "annotated_videos"
        os.makedirs(output_dir, exist_ok=True)
        output_video_path = os.path.join(output_dir, f"{video_id_clean}_blendshapes.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frames_data = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_idx in range(total_frames):
        success, frame = cap.read()
        if not success: break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        frame_row = {"video_id": video_id_clean, "gloss": gloss, "frame": frame_idx}

        if detection_result.face_blendshapes:
            blendshapes = detection_result.face_blendshapes[0]
            for cat in blendshapes:
                frame_row[cat.category_name] = cat.score
            if save_annotation and video_writer:
                annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)
                video_writer.write(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        elif save_annotation and video_writer:
            video_writer.write(frame)

        frames_data.append(frame_row)

    cap.release()
    if video_writer: video_writer.release()

    if not frames_data:
        tqdm.write(f"  -> Warning: No frames processed for video {video_id_clean}")
        return pd.DataFrame()

    final_df = pd.DataFrame(frames_data)
    for shape in KNOWN_BLENDSHAPES:
        if shape not in final_df.columns:
            final_df[shape] = np.nan

    cols = ["video_id", "gloss", "frame"] + [s for s in KNOWN_BLENDSHAPES if
                                             s not in ["_neutral", "frame", "gloss", "video_id"]]
    final_df = final_df[cols]
    return final_df


def main():
    start_time = time.time()
    parser = ArgumentParser(description='Extract facial blendshapes from videos based on metadata or a folder scan.')
    parser.add_argument('--video_folder', type=str, required=True,
                        help='Path to the root folder containing all input videos.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the final Parquet file.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the mediapipe face_landmarker.task model file.')
    parser.add_argument('--metadata_file', type=str, default=None, help='(Optional) Path to the metadata JSON file.')
    parser.add_argument('--n_glosses', type=int, default=None,
                        help='(Optional) Number of glosses to process from metadata.')
    parser.add_argument('--save_video', action='store_true', help='Set this flag to save annotated videos.')
    parser.add_argument('--annotated_video_folder', type=str, default='annotated_videos',
                        help='Folder to save annotated videos.')
    args = parser.parse_args()

    processed_ids, existing_df = set(), None
    if os.path.exists(args.output_path):
        print("Found existing output file. Loading processed video IDs...")
        existing_df = pd.read_parquet(args.output_path)
        processed_ids = set(existing_df['video_id'].unique())
        print(f"{len(processed_ids)} videos already processed. They will be skipped.")

    instances_to_process = []
    if args.metadata_file:
        print(f"Loading metadata from {args.metadata_file}...")
        with open(args.metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        entries = metadata[:args.n_glosses] if args.n_glosses else metadata
        instances_to_process = [(e['gloss'], i['video_id']) for e in entries for i in e['instances'] if
                                i['video_id'] not in processed_ids]
    else:
        print(f"No metadata file provided. Scanning video folder: {args.video_folder}...")
        video_paths = glob.glob(os.path.join(args.video_folder, "*.mp4"))
        for video_path in video_paths:
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            if video_id not in processed_ids:
                instances_to_process.append(("N/A", video_id))

    print("Initializing MediaPipe FaceLandmarker...")
    base_options = python.BaseOptions(model_asset_path=args.model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options, output_face_blendshapes=True, num_faces=1,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    print("Model initialized successfully.")

    all_new_frames, new_videos_processed, missing_videos, failed_videos = [], [], [], []

    if not instances_to_process:
        print("No new videos to process.")
    else:
        for gloss_label, video_id in tqdm(instances_to_process, desc="Processing Videos"):
            video_path = os.path.join(args.video_folder, f"{video_id}.mp4")
            if not os.path.isfile(video_path):
                tqdm.write(f"  -> MISSING: {video_path}")
                missing_videos.append(video_id)
                continue
            df = process_single_video(
                video_path, gloss_label, detector,
                args.save_video, args.annotated_video_folder
            )
            if not df.empty:
                all_new_frames.append(df)
                new_videos_processed.append(video_id)
            else:
                tqdm.write(f"  -> FAILED: Could not extract blendshapes for {video_id}")
                failed_videos.append(video_id)

    detector.close()

    if all_new_frames:
        print("\nMerging new data with existing data...")
        final_df = pd.concat(all_new_frames, ignore_index=True).fillna(-2)
        if existing_df is not None:
            final_df = pd.concat([existing_df, final_df], ignore_index=True)
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        final_df.to_parquet(args.output_path, index=False)
        print(f"SUCCESS: Blendshapes data saved to {args.output_path}")
    else:
        print("\nNo new data to save.")

    end_time = time.time()
    if failed_videos:
        with open("failed_blendshapes_videos.txt", "w") as f: f.writelines([f"{vid}\n" for vid in failed_videos])
    if missing_videos:
        with open("missing_blendshapes_videos.txt", "w") as f: f.writelines([f"{vid}\n" for vid in missing_videos])

    print("\n--- Processing Summary ---")
    print(f"Total time elapsed: {end_time - start_time:.2f} seconds")
    print(f"SUCCESS:  {len(new_videos_processed)} new video(s) processed")
    print(f"SKIPPED:  {len(processed_ids)} video(s) (already done)")
    print(f"MISSING:  {len(missing_videos)} video(s) (file not found)")
    print(f"FAILED:   {len(failed_videos)} video(s) (no landmarks extracted)")
    print("--------------------------")
    if failed_videos: print("A list of failed video IDs was saved to 'failed_blendshapes_videos.txt'")
    if missing_videos: print("A list of missing video IDs was saved to 'missing_blendshapes_videos.txt'")


if __name__ == "__main__":
    main()