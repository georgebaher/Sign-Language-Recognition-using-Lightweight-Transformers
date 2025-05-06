import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from SLR_MediaPipe_DTW.models.pose_model import PoseModel
import time

# Start timer
start = time.time()

# Load paths
load_dotenv()
wlasl_metadata_path = os.getenv("WLASL_METADATA_PATH")
landmarks_parquet_path = "Mediapipe_holistic/wlasl100_landmarks.parquet"
output_parquet_path = "Mediapipe_holistic/wlasl100_pose_angles.parquet"

# Load metadata
with open(wlasl_metadata_path, 'r') as f:
    glosses = json.load(f)
glosses = glosses[:100]

# Load landmarks
df = pd.read_parquet(landmarks_parquet_path)

# Define pose columns
pose_columns = [col for col in df.columns if col.startswith("P#") and col.endswith(("_x", "_y", "_z"))]

# Check already processed video_ids
processed_ids = set()
existing_df = None
if os.path.exists(output_parquet_path) and os.path.getsize(output_parquet_path) > 0:
    print(f"📄 Found existing Parquet: {output_parquet_path}")
    existing_df = pd.read_parquet(output_parquet_path)
    if "video_id" in existing_df.columns:
        processed_ids = set(existing_df["video_id"])
    print(f"⏭️ {len(processed_ids)} video_ids already processed. They will be skipped.")
else:
    print("📁 No existing or non-empty Parquet found — starting fresh.")

# Dummy model to extract angle labels
dummy_pose = np.zeros((33, 3))
dummy_model = PoseModel(dummy_pose)
connection_pairs = list(dummy_model.connections)
angle_labels = [f"Angle{{{a}-{b}}}" for i, a in enumerate(connection_pairs)
                                     for j, b in enumerate(connection_pairs)
                                     if i < j]
header = ["video_id", "gloss"] + [f"P_{label}" for label in angle_labels]

# Stats trackers
skipped_videos, failed_videos, processed_videos = [], [], []
all_rows = []

# Process
for gloss in tqdm(glosses, desc="Glosses", unit="gloss", colour='green'):
    gloss_label = gloss["gloss"]
    for instance in tqdm(gloss["instances"], desc=f"<{gloss_label}>",unit="instance", leave=False, colour='white'):
        try:
            video_id =instance["video_id"]
            if video_id in processed_ids:
                skipped_videos.append(video_id)
                continue

            vid_df = df[df["video_id"] == video_id]
            if vid_df.empty:
                print(f"⚠️ Skipping empty video_id: {video_id}")
                continue

            for _, row in vid_df.iterrows():
                try:
                    pose_landmarks = row[pose_columns].values.astype(float).reshape(-1, 3)
                    pose_model = PoseModel(pose_landmarks)
                    angles = [video_id, gloss_label] + pose_model.feature_vector
                    all_rows.append(angles)
                    processed_videos.append(video_id)
                except Exception as e:
                    print(f"⚠️ Failed on frame in video {video_id}: {e}")
                    failed_videos.append(video_id)
        except Exception as e:
            print(f"❌ Error processing video_id: {e}")
            failed_videos.append(instance["video_id"])
            continue

# Save all results
if all_rows:
    new_df = pd.DataFrame(all_rows, columns=header)
    if existing_df is not None:
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    final_df.to_parquet(output_parquet_path, index=False)
    print(f"✅ Saved all pose angles to: {output_parquet_path}")
else:
    print("⚠️ No new rows to save.")

# Save failed list
if failed_videos:
    with open("failed_pose_videos.txt", "w") as f:
        f.writelines([str(v) + "\n" for v in set(failed_videos)])

# Summary
end = time.time()
print(f"\n📊 Summary")
print(f"🕒 Time elapsed: {round(end - start, 2)} sec")
print(f"🟢 Processed: {len(set(processed_videos))}")
print(f"⏭️ Skipped: {len(set(skipped_videos))}")
print(f"❌ Failed: {len(set(failed_videos))}")
