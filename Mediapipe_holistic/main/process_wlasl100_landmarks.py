import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from Mediapipe_holistic.HolisticProcessor import HolisticProcessor
from Mediapipe_holistic.processing.io_utils import load_existing_ids, save_and_merge

# Start timer
start = time.time()

# Load paths
load_dotenv()
metadata_path = os.getenv("WLASL_METADATA_PATH")
videos_path = os.getenv("WLASL_VIDEOS_PATH")
landmark_parquet_path = "wlasl100_landmarks.parquet"

# Load metadata
with open(metadata_path, 'r') as f:
    glosses = json.load(f)[:100]

# Load previously processed IDs
processed_ids, existing_df = load_existing_ids(landmark_parquet_path)
print(f"✅ {len(processed_ids)} videos already processed. Skipping those.")

# Initialize processor
processor = HolisticProcessor(extract=["pose", "hand"])

# Tracking
new_videos_processed = []
missing_videos = []
failed_videos = []
all_new_frames = []
cnt = 0

# Process videos
for gloss in tqdm(glosses, desc="Glosses", unit="gloss", colour='green'):
    gloss_label = gloss['gloss']
    for instance in gloss['instances']:
        video_id = str(instance['video_id']).zfill(5)

        if video_id in processed_ids:
            print(f"⏭️ Skipping already processed video: {video_id}")
            continue

        video_path = os.path.join(videos_path, f"{video_id}.mp4")
        if not os.path.isfile(video_path):
            print(f"⚠️ Skipping missing file: {video_path}")
            missing_videos.append(video_id)
            continue

        try:
            df = processor.process_video(video_path, show_landmarks=False, gloss=gloss_label)
            if df.empty:
                print(f"❌ No frames extracted for {video_id}")
                continue

            df["video_id"] = video_id
            df["gloss"] = gloss_label

            all_new_frames.append(df)
            new_videos_processed.append(video_id)
            cnt += 1
            print(f"✅ Processed and collected video: {video_id} ({cnt} total)")

        except Exception as e:
            print(f"❌ Error processing {video_id}: {e}")
            failed_videos.append(video_id)

# Save combined
if all_new_frames:
    new_df = pd.concat(all_new_frames, ignore_index=True)
    save_and_merge(new_df, landmark_parquet_path, existing_df)
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
