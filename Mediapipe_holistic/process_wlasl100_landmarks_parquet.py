import os
import json
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from HolisticProcessor import HolisticProcessor
import time

start = time.time()

# Load paths from .env
load_dotenv()
wlasl_metadata_path = os.getenv("WLASL_METADATA_PATH")
videos_path = os.getenv("WLASL_VIDEOS_PATH")

# Output Parquet path
landmark_parquet_path = "wlasl100_landmarks.parquet"

# Initialize processor
processor = HolisticProcessor(extract=["pose", "hand"])

# Load WLASL100 metadata
with open(wlasl_metadata_path, 'r') as f:
    glosses = json.load(f)
glosses = glosses[:100]

# Load already processed video_ids
processed_ids = set()
existing_df = None
if os.path.exists(landmark_parquet_path):
    print(f"📄 Found existing Parquet: {landmark_parquet_path}")
    existing_df = pd.read_parquet(landmark_parquet_path)
    if "video_id" in existing_df.columns:
        processed_ids = set(existing_df["video_id"].astype(str).str.zfill(5))
    print(f"✅ {len(processed_ids)} videos already processed. Skipping those.")

# Process videos
new_videos_processed = []
missing_videos = []
failed_videos = []
all_new_frames = []
cnt = 0

for gloss in tqdm(glosses, desc="Glosses", unit="gloss", colour='green'):
    gloss_label = gloss['gloss']
    for instance in gloss['instances']:
        video_id = str(instance['video_id'])
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

# Concatenate and save all new frames
if all_new_frames:
    new_df = pd.concat(all_new_frames, ignore_index=True)
    if existing_df is not None:
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    final_df.to_parquet(landmark_parquet_path, index=False)
    print(f"💾 Saved full dataset to Parquet: {landmark_parquet_path}")
else:
    print("⚠️ No new data to save.")

# Save missing and failed logs
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
