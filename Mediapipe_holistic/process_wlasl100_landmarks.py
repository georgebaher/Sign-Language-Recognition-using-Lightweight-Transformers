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

# Output CSV path
landmark_csv_path = "wlasl100_landmarks.csv"

# Initialize processor
processor = HolisticProcessor(extract=["pose", "hand"])

# Load WLASL100 metadata
with open(wlasl_metadata_path, 'r') as f:
    glosses = json.load(f)

# Only first 100 glosses for WLASL100
glosses = glosses[:100]

# Load already processed video_ids (if CSV exists)
processed_ids = set()
if os.path.exists(landmark_csv_path):
    print(f"📄 Found existing CSV: {landmark_csv_path}")
    existing_df = pd.read_csv(landmark_csv_path)

    if "video_id" in existing_df.columns:
        processed_ids = set(existing_df["video_id"].astype(str).str.zfill(5))

    print(f"✅ {len(processed_ids or [])} videos already processed. Skipping those.")

# Process videos
new_videos_processed = []
missing_videos = []
failed_videos = []
file_exists = os.path.exists(landmark_csv_path)
cnt=0
for gloss in tqdm(glosses, desc="Glosses"):
    gloss_label = gloss['gloss']
    for instance in gloss['instances']:
        video_id = str(instance['video_id'])  # Ensure consistent type
        if video_id in processed_ids:
            print(f"⏭️ Skipping already processed video: {video_id}")
            continue

        video_path = os.path.join(videos_path, f"{video_id}.mp4")

        if not os.path.isfile(video_path):
            print(f"⚠️ Skipping missing file: {video_path}")
            missing_videos.append(video_id)
            continue

        print(f"📹 Processing {video_id} for gloss '{gloss_label}'")
        try:
            df = processor.process_video(video_path, show_landmarks=False, gloss=gloss_label)
            if df.empty:
                print(f"❌ No frames extracted for {video_id}")
                continue
            new_videos_processed.append(video_id)
            df.to_csv(landmark_csv_path, mode='a', header=not file_exists, index=False)
            file_exists = True
            print(f"✅ Saved video <{video_id}.mp4> landmarks to {landmark_csv_path}")
            processed_ids.add(video_id)  # to avoid reprocessing in the same run
        except Exception as e:
            print(f"❌ Error processing {video_id}: {e}")
            failed_videos.append(video_id)
            continue
        cnt += 1
        print(f"🔢 {cnt} videos processed so far")



# Calculate total time
end = time.time()


# Save missing and failed videos
if missing_videos:
    with open("missing_videos.txt", "w") as f:
        f.writelines([vid + "\n" for vid in missing_videos])

if failed_videos:
    with open("failed_videos.txt", "w") as f:
        f.writelines([vid + "\n" for vid in failed_videos])

# Summary of run
print(f"\n       📊 Summary 📊 ")
print(f"🕒 Total time: {round(end - start, 2)} seconds")
print(f"🟢 Processed videos: {len(new_videos_processed)}")
print(f"⚠️ Missing videos: {len(missing_videos)}")
print(f"❌ Failed videos: {len(failed_videos)}")
print(f"⏭️ Skipped (already processed): {len(processed_ids)}")

