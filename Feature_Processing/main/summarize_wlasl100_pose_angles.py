import os
import pandas as pd
import json
from Feature_Processing.utils.pose_angles_utils import summarize_single_video_pose_angles
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- Load the Parquet file ---
wlasl100_pose_angles_path = os.getenv("WLASL100_POSE_ANGLES_PATH")
df = pd.read_parquet(wlasl100_pose_angles_path)

# --- Load metadata and filter to first 100 glosses ---
with open(os.getenv("WLASL_METADATA_PATH"), "r") as f:
    metadata = json.load(f)

glosses = metadata[:100]

all_instances_summary = []

# --- Process each instance ---
for gloss in tqdm(glosses, unit="gloss", total=len(glosses), leave=False, colour="green"):
    for instance in gloss['instances']:
        video_id = instance['video_id']
        inst_df = df[df['video_id'] == video_id].reset_index(drop=True)
        summary_df = summarize_single_video_pose_angles(inst_df)
        all_instances_summary.append(summary_df)

# --- Concatenate and save ---
final_summary_df = pd.concat(all_instances_summary, ignore_index=True)

output_path = os.getenv("WLASL100_POSE_ANGLES_SUMMARY_PATH", "wlasl100_pose_angles_summary.old results")
final_summary_df.to_parquet(output_path, index=False)

print(f"✅ Saved summarized pose angles to {output_path}")
