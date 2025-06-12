import os
from torch.utils.data import DataLoader
from dataloader import WLASLParquetDataset
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

all_features = [
    "HAND_LANDMARKS",
    "POSE_LANDMARKS",
    "HAND_POSE_LANDMARKS",
    "HAND_ANGLES",
    "POSE_ANGLES",
    "HAND_POSE_ANGLES"
]

feature_env_vars = {
    "HAND_LANDMARKS": "WLASL100_HAND_LANDMARKS_PATH",
    "POSE_LANDMARKS": "WLASL100_POSE_LANDMARKS_PATH",
    "HAND_POSE_LANDMARKS": "WLASL100_HAND_POSE_LANDMARKS_PATH",
    "HAND_ANGLES": "WLASL100_HAND_ANGLES_PATH",
    "POSE_ANGLES": "WLASL100_POSE_ANGLES_PATH",
    "HAND_POSE_ANGLES": "WLASL100_HAND_POSE_ANGLES_PATH",
}

metadata_path = os.getenv("WLASL_METADATA_PATH")
assert metadata_path and os.path.exists(metadata_path), f"Missing metadata: {metadata_path}"

summary = []

for feature in all_features:
    parquet_path = os.getenv(feature_env_vars[feature])
    assert parquet_path and os.path.exists(parquet_path), f"Missing parquet for {feature}"

    for fs in [0, 1]:
        dataset = WLASLParquetDataset(
            parquet_path=parquet_path,
            metadata_json_path=metadata_path,
            split="train",
            features=feature,
            fs=fs,
            n_heads=8
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        x, y = next(iter(dataloader))

        summary.append({
            "feature": feature,
            "fs": fs,
            "num_samples": len(dataset),
            "num_classes": len(dataset.gloss2idx),
            "feature_shape": tuple(x.shape),
        })

df = pd.DataFrame(summary)
print(df.to_string(index=False))
