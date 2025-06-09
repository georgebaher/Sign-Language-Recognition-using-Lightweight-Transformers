import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import numpy as np


class WLASLParquetDataset(Dataset):
    def __init__(self, parquet_path, metadata_json_path, split="train", transform=None, max_len=None, axis_mode='xyz'):
        if parquet_path is None:
            raise ValueError("parquet_path cannot be None")
        if metadata_json_path is None:
            raise ValueError("metadata_json_path cannot be None")
        self.df = pd.read_parquet(parquet_path)  # landmarks or angles
        self.transform = transform
        self.split = split
        self.max_len = max_len  # depends on features chosen
        self.axis_mode = axis_mode.lower()  # Select axis to take

        # Load metadata and filter by split
        with open(metadata_json_path, "r") as f:
            metadata_raw = json.load(f)

        # Loaded first 100 glosses
        glosses = metadata_raw[0:100]

        # Flatten instances with correct split
        self.instances = []
        for gloss_entry in glosses:
            for inst in gloss_entry["instances"]:
                if inst["split"] == self.split:
                    self.instances.append({
                        "video_id": str(inst["video_id"]),
                        "gloss": gloss_entry["gloss"],
                        "gloss_idx": None,  # filled below
                        "split": inst["split"],
                    })

        # Assign numeric labels to glosses
        glosses = sorted(set(entry["gloss"] for entry in self.instances))
        self.gloss2idx = {gloss: idx for idx, gloss in enumerate(glosses)}
        self.idx2gloss = {idx: gloss for gloss, idx in self.gloss2idx.items()}
        for entry in self.instances:
            entry["gloss_idx"] = self.gloss2idx[entry["gloss"]]

        # Filter feature DataFrame by available video_ids to get train/val/test instances
        video_ids = [entry["video_id"] for entry in self.instances]
        # Filter only valid video IDs
        self.df = self.df[self.df["video_id"].astype(str).isin(video_ids)]

        # # TODO: Apply or not gaussian noise or other transformation
        # transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])

        # Group all rows by video_id → returns a GroupBy object
        self.feature_map = self.df.groupby("video_id")

        # Find maximum sequence length (number of frames per video)
        self.video_lengths = self.feature_map.size()  # Series: video_id → count
        # print(self.video_lengths)
        self.max_seq_len = self.video_lengths.max()

        if self.max_len is None:
            self.max_len = self.max_seq_len  # Use max frames in dataset if not manually specified

        print(f"[INFO] Max sequence length in dataset is {self.max_seq_len}")


    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        entry = self.instances[idx]
        video_id = entry["video_id"]
        gloss_idx = entry["gloss_idx"]

        # Get group of all frames for this video
        group = self.feature_map.get_group(video_id)
        # print(group)

        features_df = group.drop(columns=["gloss", "video_id"], errors="ignore")

        # Filter columns by axis_mode
        if self.axis_mode == "xy":
            features_df = features_df[[col for col in features_df.columns if not col.endswith('_z')]]
        elif self.axis_mode == "xyz":
            pass  # keep all
        else:
            raise ValueError(f"[ERROR] Invalid axis_mode: {self.axis_mode}")

        # Now convert to NumPy
        features = features_df.values.astype(np.float32)
        feature_tensor = torch.tensor(features)  # shape: (n_frames, n_features)

        # print(f'shape of feature tensor: {feature_tensor.shape}')

        # TODO: Optional transform
        # if self.transform:
        #     feature_tensor = self.transform(feature_tensor)

        # Pad feature dimension to even number if it's odd (for Positional Encoding)
        if feature_tensor.shape[1] % 2 != 0:
            pad_column = torch.full((feature_tensor.shape[0], 3), fill_value=-2.0)
            feature_tensor = torch.cat([feature_tensor, pad_column], dim=1)

        # Pad to max_len
        if self.max_len:
            seq_len, feat_dim = feature_tensor.shape
            if seq_len < self.max_len:
                pad = torch.full((self.max_len - seq_len, feat_dim), fill_value=-2.0)
                feature_tensor = torch.cat([feature_tensor, pad], dim=0)

        # print(feature_tensor)
        label_tensor = torch.tensor(gloss_idx, dtype=torch.long)
        return feature_tensor, label_tensor

