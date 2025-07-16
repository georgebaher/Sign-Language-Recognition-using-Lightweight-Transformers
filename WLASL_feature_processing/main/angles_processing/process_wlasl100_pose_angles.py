import os
import sys
sys.path.append('')
sys.path.append('..')
sys.path.append('../../')
import json
import time
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from WLASL_feature_processing.utils.pose_angles_utils import compute_pose_angles
from WLASL_feature_processing.utils.io_utils import load_existing_ids, save_and_merge

# Start timer
start = time.time()

# Load env + config
load_dotenv()
metadata_path = os.getenv("WLASL_METADATA_PATH")
landmarks_path = os.getenv("WLASL100_POSE_LANDMARKS_PATH")
output_path = os.getenv("WLASL100_POSE_ANGLES_PATH")

with open(metadata_path) as f:
    glosses = json.load(f)[:100]
landmarks_df = pd.read_parquet(landmarks_path)

# Load already-processed
processed_ids, existing_df = load_existing_ids(output_path)
print(f"{len(processed_ids)} videos already processed. Skipping those.")

# Prepare output and trackers
skipped_videos, failed_videos, processed_videos = [], [], []
all_dfs = []

# Process all glosses
for gloss in tqdm(glosses, desc="Glosses", colour='green'):
    gloss_label = gloss["gloss"]
    for inst in tqdm(gloss["instances"], desc=f"<{gloss_label}>", leave=False, colour='white'):
        video_id = inst["video_id"]

        if video_id in processed_ids:
            skipped_videos.append(video_id)
            continue

        try:
            df = compute_pose_angles(landmarks_df, video_id, gloss_label)
            if df.empty:
                print(f"Skipping empty video_id: {video_id}")
                continue
            all_dfs.append(df)
            processed_videos.append(video_id)
        except Exception as e:
            print(f"Error in video {video_id}: {e}")
            failed_videos.append(video_id)

# Save
if all_dfs:
    final_df = pd.concat(all_dfs, ignore_index=True)
    save_and_merge(final_df, output_path, existing_df)
else:
    print("No new rows to save.")

# Save failed list if needed
if failed_videos:
    with open("failed_pose_videos.txt", "w") as f:
        f.writelines([str(v) + "\n" for v in set(failed_videos)])

# Summary
end = time.time()
print(f"\nSummary")
print(f"Time elapsed: {round(end - start, 2)} sec")
print(f"Processed videos: {len(set(processed_videos))}")
print(f"Skipped (already processed): {len(set(skipped_videos))}")
print(f"Failed videos: {len(set(failed_videos))}")