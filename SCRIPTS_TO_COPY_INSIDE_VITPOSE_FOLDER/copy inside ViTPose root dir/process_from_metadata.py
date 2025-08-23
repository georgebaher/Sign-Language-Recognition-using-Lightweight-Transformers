# process_from_metadata.py
#
# This script orchestrates batch processing based on a metadata JSON file.
# It uses the functions defined in pose_utils.py to perform the core tasks.

import os
import glob
import cv2
import json
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
import time

# Import our custom utility functions
import pose_utils


def main():
    """
    Main function to run the metadata-driven batch processing pipeline.
    """
    parser = ArgumentParser(description='Run batch processing using a metadata file.')

    # --- Essential Arguments ---
    parser.add_argument('--metadata-file', type=str, required=True, help='Path to the metadata JSON file.')
    parser.add_argument('--n-glosses', type=int, default=100, help='Number of glosses to process.')
    parser.add_argument('--video-folder', type=str, required=True,
                        help='Path to the root folder containing all input videos.')
    parser.add_argument('--output-folder', type=str, required=True,
                        help='Path to the folder where Parquet files will be saved.')

    # --- Optional Arguments (pass-through to utils) ---
    parser.add_argument('--det-config', default='demo/mmdetection_cfg/yolox_l_8x8_300e_coco.py')
    parser.add_argument('--det-checkpoint', default='yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth')
    parser.add_argument('--pose-config',
                        default='configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_large_wholebody_256x192.py')
    parser.add_argument('--pose-checkpoint', default='wholebody.pth')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--bbox-thr', type=float, default=0.5)
    parser.add_argument('--conf-thr', type=float, default=0.5)

    args = parser.parse_args()

    start = time.time()

    # --- 1. Load models once ---
    det_model, pose_model, dataset_info = pose_utils.load_models(
        args.det_config, args.det_checkpoint, args.pose_config, args.pose_checkpoint, args.device
    )

    # --- 2. Load Metadata ---
    print(f"Loading metadata from {args.metadata_file}...")
    with open(args.metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"Found {len(metadata)} glosses to process.")

    glosses = metadata[:args.n_glosses]
    print(f"Processing only {len(glosses)}.")


    os.makedirs(args.output_folder, exist_ok=True)
    all_pose_data, all_face_data, all_hand_data = [], [], []

    processed_video_ids = set()
    new_video_ids = set()
    missing_video_ids = set()
    failed_video_ids = set()
    pose_parquet_path = os.path.join(args.output_folder, 'pose_landmarks.parquet')
    if os.path.exists(pose_parquet_path):
        print("Found existing Parquet file. Checking for previously processed videos...")
        try:
            existing_df = pd.read_parquet(pose_parquet_path)
            if 'video_id' in existing_df.columns:
                processed_video_ids = set(existing_df['video_id'].unique())
                print(f"Found {len(processed_video_ids)} videos already processed. They will be skipped.")
        except Exception as e:
            print(f"Could not read existing Parquet file, will process all videos. Error: {e}")

    # --- 3. Loop through selected number of entries from metadata ---
    for entry in tqdm(glosses, desc="Processing Glosses", unit="gloss", colour="green"):
        gloss = entry['gloss']
        for instance in entry['instances']:
            video_id = instance['video_id']

            if video_id in processed_video_ids:
                # Use tqdm.write to print without disturbing the progress bars
                tqdm.write(f"  -> Video ID '{video_id}' already processed. Skipping.")
                continue

            # Find the video file. This handles various extensions like .mp4, .mov, etc.
            video_pattern = os.path.join(args.video_folder, video_id + '.*')
            video_paths = glob.glob(video_pattern)

            if not video_paths:
                print(f"\n  -> Warning: Video for ID '{video_id}' (gloss: '{gloss}') not found in folder. Skipping.")
                missing_video_ids.add(video_id)
                continue

            video_path = video_paths[0]  # Use the first match

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"\n  -> Error: Could not open video file {video_path}. Skipping.")
                failed_video_ids.add(video_id)
                continue

            new_video_ids.add(video_id)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # --- The INNER progress bar for the current video ---
            with tqdm(
                    total=total_frames,
                    desc=f"  -> {gloss} ({video_id})",
                    unit='frame',
                    leave=False,
                    colour='white'
            ) as pbar:
                frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_height, frame_width, _ = frame.shape

                    pose_results = pose_utils.process_single_frame(
                        det_model, pose_model, dataset_info, frame, args.bbox_thr
                    )

                    pose_rows, face_rows, hand_rows = pose_utils.get_pose_data(
                        pose_results, video_id, frame_idx, gloss,  # Pass the gloss here
                        args.conf_thr, frame_width, frame_height
                    )

                    all_pose_data.extend(pose_rows)
                    all_face_data.extend(face_rows)
                    all_hand_data.extend(hand_rows)
                    frame_idx += 1
                    pbar.update(1)

            cap.release()

    # --- 5. Save aggregated data to Parquet files ---
    print("\nAll videos processed. Converting data to Parquet files...")

    if all_pose_data:
        pose_df = pd.DataFrame(all_pose_data).fillna(-2)
        pose_output_path = os.path.join(args.output_folder, 'pose_landmarks.parquet')
        pose_df.to_parquet(pose_output_path, index=False)
        print(f" Pose landmarks saved to: {pose_output_path}")

    if all_face_data:
        face_df = pd.DataFrame(all_face_data).fillna(-2)
        face_output_path = os.path.join(args.output_folder, 'face_landmarks.parquet')
        face_df.to_parquet(face_output_path, index=False)
        print(f" Face landmarks saved to: {face_output_path}")

    if all_hand_data:
        hand_df = pd.DataFrame(all_hand_data).fillna(-2)
        hand_output_path = os.path.join(args.output_folder, 'hand_landmarks.parquet')
        hand_df.to_parquet(hand_output_path, index=False)
        print(f" Hand landmarks saved to: {hand_output_path}")

    end = time.time()
    print(f"\nSummary")
    print(f"Processed: {len(new_video_ids)}")
    print(f"Skipped: {len(processed_video_ids)}")
    print(f"Missing: {len(missing_video_ids)}")
    print(f"Time elapsed: {round(end - start, 2)} sec")

if __name__ == '__main__':
    main()