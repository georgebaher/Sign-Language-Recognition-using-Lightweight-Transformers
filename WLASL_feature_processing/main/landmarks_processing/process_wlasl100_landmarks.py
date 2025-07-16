import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import sys
sys.path.append('')
sys.path.append('..')
sys.path.append('../../')

# Assuming these are in a utils folder as per your original structure
from WLASL_feature_processing.utils.landmarks_utils import compute_video_landmarks
from WLASL_feature_processing.utils.io_utils import load_existing_ids, save_and_merge

# Start timer
start = time.time()

# Load environment
load_dotenv()
metadata_path = os.getenv("WLASL_METADATA_PATH")
videos_folder_path = os.getenv("WLASL_VIDEOS_PATH")

# Define the three output paths
pose_landmarks_parquet_path = os.getenv("WLASL100_POSE_LANDMARKS_PATH")
face_landmarks_parquet_path = os.getenv("WLASL100_FACE_LANDMARKS_PATH")
hand_landmarks_parquet_path = os.getenv("WLASL100_HAND_LANDMARKS_PATH")

# Load wlasl metadata
with open(metadata_path, 'r', encoding='utf-8') as f:
    glosses = json.load(f)[:100]

# ---  Load existing processed IDs from one of the files (e.g., pose) ---
# We assume if a video is in one file, it's in all of them.
processed_ids, existing_pose_df = load_existing_ids(pose_landmarks_parquet_path)
print(f"{len(processed_ids)} videos already processed. Skipping those.")

# --- Track results for each of the three dataframes ---
all_new_pose_frames = []
all_new_face_frames = []
all_new_hand_frames = []
new_videos_processed = []
missing_videos = []
failed_videos = []

# Loop over glosses
for gloss in tqdm(glosses, desc="Glosses", unit="gloss", colour='green'):
    gloss_label = gloss["gloss"]
    for instance in tqdm(gloss["instances"], desc=f"<{gloss_label}>", unit="video", leave=False, colour='cyan'):
        video_id = instance["video_id"]

        if video_id in processed_ids:
            continue

        video_path = os.path.join(videos_folder_path, f"{video_id}.mp4")
        if not os.path.isfile(video_path):
            print(f"️File not found: {video_path}")
            missing_videos.append(video_id)
            continue

        # --- Handle the three returned DataFrames ---
        pose_df, face_df, hand_df, _ = compute_video_landmarks(video_path, gloss_label)

        # --- Check if all dataframes are empty ---
        if pose_df.empty and face_df.empty and hand_df.empty:
            failed_videos.append(video_id)
            continue

        # --- Append each dataframe to its corresponding list ---
        all_new_pose_frames.append(pose_df)
        all_new_face_frames.append(face_df)
        all_new_hand_frames.append(hand_df)
        new_videos_processed.append(video_id)

# --- Save new data for each of the three feature types ---
# We need to load the other existing DFs if we're merging old and new data
_, existing_face_df = load_existing_ids(face_landmarks_parquet_path)
_, existing_hand_df = load_existing_ids(hand_landmarks_parquet_path)
if new_videos_processed:
    # --- Save POSE data ---
    if all_new_pose_frames:
        final_pose_df = pd.concat(all_new_pose_frames, ignore_index=True)
        save_and_merge(final_pose_df, pose_landmarks_parquet_path, existing_pose_df)

    # --- Save FACE data ---
    if all_new_face_frames:
        final_face_df = pd.concat(all_new_face_frames, ignore_index=True)
        save_and_merge(final_face_df, face_landmarks_parquet_path, existing_face_df)

    # --- Save HAND data ---
    if all_new_hand_frames:
        final_hand_df = pd.concat(all_new_hand_frames, ignore_index=True)
        save_and_merge(final_hand_df, hand_landmarks_parquet_path, existing_hand_df)
else:
    print("No new data to save.")

# --- Log saving and summary remain the same ---
if missing_videos:
    with open("missing_landmarks_videos.txt", "w") as f:
        f.writelines([vid + "\n" for vid in missing_videos])

if failed_videos:
    with open("failed_landmarks_videos.txt", "w") as f:
        f.writelines([vid + "\n" for vid in failed_videos])

end = time.time()
print(f"\nSummary")
print(f"Time elapsed: {round(end - start, 2)} sec")
print(f"Processed: {len(new_videos_processed)}")
print(f"Skipped: {len(processed_ids)}")
print(f"Missing: {len(missing_videos)}")
print(f"Failed: {len(failed_videos)}")
