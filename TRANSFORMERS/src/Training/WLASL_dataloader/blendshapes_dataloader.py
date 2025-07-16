import sys

sys.path.append('')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../TRANSFORMER_SAMPLE_2/')
import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import numpy as np


# Note: The feature selection imports are no longer needed for this specific class
# from WLASL_feature_processing.feature_selection.top_features.hand_angles_features import TOP_ANGLE_BASES as hand_angles
# ... etc.


class WLASLBlendshapesDataset(Dataset):
    """
    A PyTorch Dataset class to load facial blendshape top_features for the WLASL dataset.
    This class handles loading data from a parquet file, filtering based on the
    split (train/val/test), and padding sequences and top_features for batching.
    """

    def __init__(self, facial_blendshapes_parquet_path, metadata_json_path,
                 split="train", transform=None, max_len=None,
                 feature_padding_mode="sentinel", n_heads=1):
        """
        Args:
            facial_blendshapes_parquet_path (str): Path to the facial blendshapes parquet file.
            metadata_json_path (str): Path to the WLASL metadata JSON file.
            split (str): The dataset split to load ("train", "val", or "test").
            transform (callable, optional): Optional transform to be applied on a sample.
            max_len (int, optional): Maximum sequence length. If None, uses the max in the dataset.
            feature_padding_mode (str): 'truncate', 'sentinel', or 'repeat'.
                                        Determines how to make the feature dimension divisible by n_heads.
            n_heads (int): Number of attention heads, used for feature dimension padding.
        """
        if facial_blendshapes_parquet_path is None:
            raise ValueError("facial_blendshapes_parquet_path cannot be None")
        if metadata_json_path is None:
            raise ValueError("metadata_json_path cannot be None")

        self.features_df = pd.read_parquet(facial_blendshapes_parquet_path)
        self.transform = transform
        self.split = split
        self.max_len = max_len
        self.feature_padding_mode = feature_padding_mode
        self.n_heads = n_heads

        # ===== Metadata and Instance Loading =====
        with open(metadata_json_path, "r") as f:
            metadata_raw = json.load(f)

        glosses = metadata_raw[0:100]  # Using the first 100 glosses as in original code
        self.instances = []
        for gloss_entry in glosses:
            for inst in gloss_entry["instances"]:
                if inst["split"] == self.split:
                    self.instances.append({
                        "video_id": str(inst["video_id"]),
                        "gloss": gloss_entry["gloss"],
                        "gloss_idx": None,
                        "split": inst["split"],
                    })
        glosses = sorted(set(entry["gloss"] for entry in self.instances))
        self.gloss2idx = {gloss: idx for idx, gloss in enumerate(glosses)}
        self.idx2gloss = {idx: gloss for gloss, idx in self.gloss2idx.items()}
        for entry in self.instances:
            entry["gloss_idx"] = self.gloss2idx[entry["gloss"]]

        # Filter feature DataFrame by available video_ids to get train/val/test instances
        video_ids = [entry["video_id"] for entry in self.instances]
        self.features_df = self.features_df[self.features_df["video_id"].astype(str).isin(video_ids)]

        # Group all rows by video_id
        self.features_map = self.features_df.groupby("video_id")
        self.video_lengths = self.features_map.size()
        self.max_seq_len = self.video_lengths.max()
        if self.max_len is None:
            self.max_len = self.max_seq_len

        # ====== DECIDE THE FINAL FEATURE DIM (padding or truncation) ======
        # Get the initial set of blendshape columns
        sample_id = self.instances[0]["video_id"]
        group = self.features_map.get_group(sample_id)
        keep_cols = group.drop(columns=["gloss", "video_id"], errors="ignore").columns.tolist()

        # Apply truncation or padding logic to make feature dimension divisible by n_heads
        k = len(keep_cols)
        if self.feature_padding_mode == "truncate":
            k = (k // n_heads) * n_heads
            self.final_columns = keep_cols[:k]
        else:  # Handle padding modes ('sentinel' or 'repeat')
            current_dim = len(keep_cols)
            target_dim = ((current_dim + n_heads - 1) // n_heads) * n_heads
            num_to_pad = target_dim - current_dim

            if num_to_pad > 0:
                if self.feature_padding_mode == 'sentinel':
                    pad_cols = [f"pad_{i}" for i in range(num_to_pad)]
                    # Add new padding columns to the entire dataframe
                    for pad_col in pad_cols:
                        self.features_df[pad_col] = -2.0
                    self.final_columns = keep_cols + pad_cols

                elif self.feature_padding_mode == 'repeat':
                    pad_source_cols = [keep_cols[i % len(keep_cols)] for i in range(num_to_pad)]
                    pad_new_names = [f"repeat_cycle_pad_{i}" for i in range(num_to_pad)]
                    # Create copies of source columns with new names in the dataframe
                    for new_name, source_name in zip(pad_new_names, pad_source_cols):
                        self.features_df[new_name] = self.features_df[source_name]
                    self.final_columns = keep_cols + pad_new_names
                else:
                    raise ValueError(f"Unknown feature_padding_mode: '{self.feature_padding_mode}'")
            else:
                self.final_columns = keep_cols

        # After preparing the dataframe, re-group it to include the new columns
        self.features_map = self.features_df.groupby("video_id")
        self.embedding_dim = len(self.final_columns)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        entry = self.instances[idx]
        video_id = entry["video_id"]
        gloss_idx = entry["gloss_idx"]

        # Get group of all frames for this video
        group = self.features_map.get_group(video_id)

        # Select only the final (potentially padded) columns
        features_df = group[self.final_columns]

        features = features_df.values.astype(np.float32)
        feature_tensor = torch.tensor(features)  # shape: (n_frames, n_features)

        # Pad sequence to max_len (temporal padding)
        if self.max_len:
            seq_len, feat_dim = feature_tensor.shape
            if seq_len < self.max_len:
                pad = torch.full((self.max_len - seq_len, feat_dim), fill_value=-2.0)
                feature_tensor = torch.cat([feature_tensor, pad], dim=0)

        label_tensor = torch.tensor(gloss_idx, dtype=torch.long)
        return feature_tensor, label_tensor

    @property
    def feature_dim(self):
        """Returns the dimensionality of the feature vector."""
        return self.embedding_dim


