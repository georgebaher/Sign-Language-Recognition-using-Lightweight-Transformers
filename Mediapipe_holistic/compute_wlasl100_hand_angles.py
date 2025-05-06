import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from SLR_MediaPipe_DTW.models.hand_model import HandModel
import time

start = time.time()

# Load .env paths
load_dotenv()
wlasl_metadata_path = os.getenv("WLASL_METADATA_PATH")
landmarks_csv_path = "Mediapipe_holistic/wlasl100_landmarks.csv"
output_csv_path = "Mediapipe_holistic/wlasl100_hand_angles.csv"

# Load metadata
with open(wlasl_metadata_path, 'r') as f:
    glosses = json.load(f)
glosses = glosses[:100]

# Load landmarks
df = pd.read_csv(landmarks_csv_path)

# Load already processed video_ids
processed_ids = set()
if os.path.exists(output_csv_path):
    existing_df = pd.read_csv(output_csv_path)
    if "video_id" in existing_df.columns:
        processed_ids = set(existing_df["video_id"].astype(str))
    print(f"✅ {len(processed_ids)} videos already processed. Skipping those.")

# Define columns
lh_columns = [col for col in df.columns if col.startswith("LH#") and col.endswith(("_x", "_y", "_z"))]
rh_columns = [col for col in df.columns if col.startswith("RH#") and col.endswith(("_x", "_y", "_z"))]

# Get angle labels from dummy model
dummy_hand = np.zeros((21, 3))
dummy_model = HandModel(dummy_hand)
connection_pairs = list(dummy_model.connections)
angle_labels = [f"Angle{{{a}-{b}}}" for i, a in enumerate(connection_pairs)
                                     for j, b in enumerate(connection_pairs)
                                     if i < j]

# Prepare CSV header
header = ["video_id", "gloss"] + [f"LH_{label}" for label in angle_labels] + [f"RH_{label}" for label in angle_labels]
write_header = not os.path.exists(output_csv_path)

# Tracking stats
new_videos_processed = []
failed_videos = []

cnt = 0
with open(output_csv_path, "a", encoding='utf-8', newline='') as f:
    if write_header:
        pd.DataFrame(columns=header).to_csv(f, index=False)

    for gloss in tqdm(glosses, desc="Glosses"):
        gloss_label = gloss["gloss"]
        for instance in tqdm(gloss["instances"], desc=f"Instances of gloss <{gloss_label}>", leave=False):
            video_id = str(instance["video_id"])
            if video_id in processed_ids:
                print(f"⏭️ Skipping already processed video: {video_id}")
                continue

            try:
                vid_df = df[df["video_id"].astype(str) == video_id]
                if vid_df.empty:
                    raise ValueError("No data found for video_id")

                for _, row in vid_df.iterrows():
                    lh_landmarks = row[lh_columns].values.astype(float).reshape(-1, 3)
                    rh_landmarks = row[rh_columns].values.astype(float).reshape(-1, 3)

                    lh_model = HandModel(lh_landmarks)
                    rh_model = HandModel(rh_landmarks)

                    angles = [video_id, row["gloss"]] + lh_model.feature_vector + rh_model.feature_vector
                    pd.DataFrame([angles]).to_csv(f, header=False, index=False)

                new_videos_processed.append(video_id)
                processed_ids.add(video_id)
                cnt += 1
                print(f"✅ Processed {video_id} ({cnt} total)")

            except Exception as e:
                print(f"❌ Failed on {video_id}: {e}")
                failed_videos.append(video_id)
                continue

# Time summary
end = time.time()

# Save failed list
if failed_videos:
    with open("failed_hand_angle_videos.txt", "w") as f:
        f.writelines([vid + "\n" for vid in failed_videos])

# Final summary
print(f"\n📊 Summary 📊")
print(f"🕒 Total time: {round(end - start, 2)} seconds")
print(f"🟢 New videos processed: {len(new_videos_processed)}")
print(f"❌ Failed videos: {len(failed_videos)}")
print(f"⏭️ Skipped videos: {len(processed_ids) - len(new_videos_processed)}")
