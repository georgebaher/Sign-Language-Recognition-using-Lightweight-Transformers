import os
import pandas as pd
import json
from WLASL_feature_processing.utils.facial_blendshapes_utils import summarize_single_video_facial_blendshapes
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- Load the Parquet file ---
wlasl100_facial_blendshapes_path = os.getenv("WLASL100_FACIAL_BLENDSHAPES_PATH")
df = pd.read_parquet(wlasl100_facial_blendshapes_path)

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
        summary_df = summarize_single_video_facial_blendshapes(inst_df)
        all_instances_summary.append(summary_df)

# --- Concatenate and save ---
final_summary_df = pd.concat(all_instances_summary, ignore_index=True)

output_path = os.getenv("WLASL100_FACIAL_BLENDSHAPES_SUMMARY_PATH")
final_summary_df.to_parquet(output_path, index=False)

print(f"✅ Saved summarized facial blendshapes to {output_path}")
