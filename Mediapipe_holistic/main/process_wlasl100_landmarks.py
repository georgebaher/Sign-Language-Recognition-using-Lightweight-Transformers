import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from Mediapipe_holistic.processing.landmarks_utils import compute_video_landmarks
from Mediapipe_holistic.processing.io_utils import load_existing_ids, save_and_merge

# Start timer
start = time.time()

# Load environment
load_dotenv()
metadata_path = os.getenv("WLASL_METADATA_PATH")
videos_path = os.getenv("WLASL_VIDEOS_PATH")

# data output path
landmark_parquet_path = os.getenv("WLASL100_LANDMARKS_PATH")

# Load metadata
with open(metadata_path, 'r') as f:
    glosses = json.load(f)[:100]

# Load existing processed IDs
processed_ids, existing_df = load_existing_ids(landmark_parquet_path)
print(f"✅ {len(processed_ids)} videos already processed. Skipping those.")

# Track results
all_new_frames = []
new_videos_processed = []
missing_videos = []
failed_videos = []

# Loop over glosses
for gloss in tqdm(glosses, desc="Glosses", unit="gloss", colour='green'):
    gloss_label = gloss["gloss"]
    for instance in tqdm(gloss["instances"], desc=f"<{gloss_label}>", unit="video", leave=False, colour='white'):
        video_id = instance["video_id"]

        if video_id in processed_ids:
            continue

        video_path = os.path.join(videos_path, f"{video_id}.mp4")
        if not os.path.isfile(video_path):
            print(f"⚠️ File not found: {video_path}")
            missing_videos.append(video_id)
            continue

        df = compute_video_landmarks(video_path, gloss_label, video_id)
        if df.empty:
            failed_videos.append(video_id)
            continue

        all_new_frames.append(df)
        new_videos_processed.append(video_id)

# Save new data
if all_new_frames:
    final_df = pd.concat(all_new_frames, ignore_index=True)
    save_and_merge(final_df, landmark_parquet_path, existing_df)
else:
    print("⚠️ No new data to save.")

# Save logs
if missing_videos:
    with open("missing_videos.txt", "w") as f:
        f.writelines([vid + "\n" for vid in missing_videos])

if failed_videos:
    with open("failed_videos.txt", "w") as f:
        f.writelines([vid + "\n" for vid in failed_videos])

# Summary
end = time.time()
print(f"\n📊 Summary")
print(f"🕒 Time elapsed: {round(end - start, 2)} sec")
print(f"🟢 Processed: {len(new_videos_processed)}")
print(f"⏭️ Skipped: {len(processed_ids)}")
print(f"⚠️ Missing: {len(missing_videos)}")
print(f"❌ Failed: {len(failed_videos)}")
