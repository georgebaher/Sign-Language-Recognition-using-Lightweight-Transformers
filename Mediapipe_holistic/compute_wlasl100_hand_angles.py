import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from SLR_MediaPipe_DTW.models.hand_model import HandModel

# Load paths from .env
load_dotenv()
wlasl_metadata_path = os.getenv("WLASL_METADATA_PATH")
landmarks_csv_path = "Mediapipe_holistic/wlasl100_landmarks.csv"
output_csv_path = "Mediapipe_holistic/wlasl100_hand_angles.csv"

# Load metadata and CSV
with open(wlasl_metadata_path, 'r') as f:
    glosses = json.load(f)
glosses = glosses[:100]

df = pd.read_csv(landmarks_csv_path)

# Define columns
lh_columns = [col for col in df.columns if col.startswith("LH#") and col.endswith(("_x", "_y", "_z"))]
rh_columns = [col for col in df.columns if col.startswith("RH#") and col.endswith(("_x", "_y", "_z"))]

# Extract connection labels once using HandModel
dummy_hand = np.zeros((21, 3))
dummy_model = HandModel(dummy_hand)
connection_pairs = list(dummy_model.connections)
angle_labels = [f"Angle{{{a}-{b}}}" for i, a in enumerate(connection_pairs)
                                     for j, b in enumerate(connection_pairs)
                                     if i < j]

# Prepare CSV header
header = ["video_id", "gloss"] + [f"LH_{label}" for label in angle_labels] + [f"RH_{label}" for label in angle_labels]
write_header = not os.path.exists(output_csv_path)

with open(output_csv_path, "a", encoding='utf-8', newline='') as f:
    if write_header:
        pd.DataFrame(columns=header).to_csv(f, index=False)

    for gloss in tqdm(glosses, desc="Glosses"):
        gloss_label = gloss["gloss"]
        for instance in tqdm(gloss["instances"], desc=f"Instances of gloss <{gloss_label}>"):
            video_id = int(instance["video_id"]) #Types matter in pandas
            print(video_id)
            vid_df = df[df["video_id"] == video_id]

            for _, row in vid_df.iterrows():
                try:
                    lh_landmarks = row[lh_columns].values.astype(float).reshape(-1, 3)
                    rh_landmarks = row[rh_columns].values.astype(float).reshape(-1, 3)

                    lh_model = HandModel(lh_landmarks)
                    rh_model = HandModel(rh_landmarks)

                    angles = [row["video_id"], row["gloss"]] + lh_model.feature_vector + rh_model.feature_vector
                    pd.DataFrame([angles]).to_csv(f, header=False, index=False)

                except Exception as e:
                    print(f"⚠️ Failed on {video_id}: {e}")
                    continue

print("✅ Done computing and saving hand angles.")
