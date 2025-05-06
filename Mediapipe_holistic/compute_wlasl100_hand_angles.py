import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from SLR_MediaPipe_DTW.models.hand_model import HandModel
import time

# Start timer
start = time.time()

# Load paths
load_dotenv()
wlasl_metadata_path = os.getenv("WLASL_METADATA_PATH")
landmarks_csv_path = "Mediapipe_holistic/wlasl100_landmarks.csv"
output_csv_path = "Mediapipe_holistic/wlasl100_hand_angles.csv"

# Load metadata
with open(wlasl_metadata_path, 'r') as f:
    glosses = json.load(f)
glosses = glosses[:100]

# Load landmarks CSV
df = pd.read_csv(landmarks_csv_path)

# Define columns
lh_columns = [col for col in df.columns if col.startswith("LH#") and col.endswith(("_x", "_y", "_z"))]
rh_columns = [col for col in df.columns if col.startswith("RH#") and col.endswith(("_x", "_y", "_z"))]

# Check already processed video_ids
processed_ids = set()
if os.path.exists(output_csv_path):
    print(f"📄 Found existing CSV: {output_csv_path}")
    existing_df = pd.read_csv(output_csv_path)
    if "video_id" in existing_df.columns:
        processed_ids = set(existing_df["video_id"].astype(int))
    print(f"⏭️ {len(processed_ids)} video_ids already processed. They will be skipped.")

# Extract connection labels once using dummy model
dummy_hand = np.zeros((21, 3))
dummy_model = HandModel(dummy_hand)
connection_pairs = list(dummy_model.connections)
angle_labels = [f"Angle{{{a}-{b}}}" for i, a in enumerate(connection_pairs)
                                     for j, b in enumerate(connection_pairs)
                                     if i < j]

# Prepare header
header = ["video_id", "gloss"] + [f"LH_{label}" for label in angle_labels] + [f"RH_{label}" for label in angle_labels]
write_header = not os.path.exists(output_csv_path)

# Stats trackers
skipped_videos, failed_videos, processed_videos = [], [], []

with open(output_csv_path, "a", encoding='utf-8', newline='') as f:
    if write_header:
        pd.DataFrame(columns=header).to_csv(f, index=False)

    for gloss in tqdm(glosses, desc="Glosses", colour='green'):
        gloss_label = gloss["gloss"]
        for instance in tqdm(gloss["instances"], desc=f"<{gloss_label}>", leave=False, colour='white'):
            try:
                video_id = int(instance["video_id"])
                if video_id in processed_ids:
                    skipped_videos.append(video_id)
                    continue

                vid_df = df[df["video_id"] == video_id]
                if vid_df.empty:
                    print(f"⚠️ Skipping empty video_id: {video_id}")
                    continue

                for _, row in vid_df.iterrows():
                    try:
                        lh_landmarks = row[lh_columns].values.astype(float).reshape(-1, 3)
                        rh_landmarks = row[rh_columns].values.astype(float).reshape(-1, 3)

                        lh_model = HandModel(lh_landmarks)
                        rh_model = HandModel(rh_landmarks)

                        angles = [video_id, gloss_label] + lh_model.feature_vector + rh_model.feature_vector
                        pd.DataFrame([angles]).to_csv(f, header=False, index=False)
                        processed_videos.append(video_id)
                    except Exception as e:
                        print(f"⚠️ Failed on frame in video {video_id}: {e}")
                        failed_videos.append(video_id)

            except Exception as e:
                print(f"❌ Error processing video_id: {e}")
                failed_videos.append(instance["video_id"])
                continue

# End timer
end = time.time()

# Save lists if needed
if failed_videos:
    with open("failed_hand_videos.txt", "w") as f:
        f.writelines([str(v) + "\n" for v in set(failed_videos)])


# Report summary
print(f"\n       📊 Summary 📊 ")
print(f"🕒 Total time: {round(end - start, 2)} seconds")
print(f"🟢 Processed videos: {len(set(processed_videos))}")
print(f"⚠️ Skipped (already processed): {len(set(skipped_videos))}")
print(f"❌ Failed videos: {len(set(failed_videos))}")

