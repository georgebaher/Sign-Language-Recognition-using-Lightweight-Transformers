# sign_language_dataset.py (Final, Production-Ready Version)
#
# A fully generalizable PyTorch Dataset for loading sign language features.
# It dynamically imports top feature lists for dataset-specific feature selection
# and handles complex data merging scenarios.

import os
import sys
import time
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from functools import reduce
from importlib import import_module
from typing import List


class SignLanguageFeaturesDataset(Dataset):
    """
    A generalizable PyTorch Dataset for loading sign language features from Parquet files.
    """

    def __init__(self,
                 metadata_json_path: str,
                 features: List[str],
                 feature_selection_dir: str = None,
                 pose_landmark_path: str = None, face_landmark_path: str = None, hand_landmark_path: str = None,
                 pose_angle_path: str = None, hand_angle_path: str = None, face_blendshape_path: str = None,
                 split: str = "train",
                 n_glosses: int = None,
                 fs: int = 0,
                 max_len: int = None,
                 feature_padding_mode: str = "sentinel",
                 n_heads: int = 1):

        start_time = time.time()
        print(f"\n--- Initializing Dataset for split: '{split}' | Features: {features} ---")

        # 1. Argument validation and setup
        self.split = split
        self.features = features
        self.fs = fs
        self.max_len = max_len
        self.feature_padding_mode = feature_padding_mode
        self.n_heads = n_heads

        if self.fs == 1 and not (feature_selection_dir and os.path.isdir(feature_selection_dir)):
            raise ValueError("Feature selection (fs=1) requires a valid 'feature_selection_dir'.")

        # 2. Load metadata and filter instances
        with open(metadata_json_path, "r", encoding="utf-8") as f:
            metadata_raw = json.load(f)
        if n_glosses is not None:
            metadata_raw = metadata_raw[:n_glosses]

        self.instances = [
            {"video_id": str(inst["video_id"]), "gloss": entry["gloss"]}
            for entry in metadata_raw for inst in entry["instances"]
            if inst["split"] == self.split
        ]
        glosses = sorted(set(entry["gloss"] for entry in self.instances))
        self.gloss2idx = {gloss: idx for idx, gloss in enumerate(glosses)}
        self.idx2gloss = {idx: gloss for gloss, idx in self.gloss2idx.items()}
        for entry in self.instances:
            entry["gloss_idx"] = self.gloss2idx[entry["gloss"]]
        video_ids_in_split = {entry["video_id"] for entry in self.instances}
        print(f"Loaded metadata for {len(self.instances)} instances from {len(glosses)} glosses.")

        # 3. Load and merge the required Parquet files
        feature_paths = {
            "pose_landmarks": pose_landmark_path, "face_landmarks": face_landmark_path,
            "hand_landmarks": hand_landmark_path, "pose_angles": pose_angle_path,
            "hand_angles": hand_angle_path, "face_blendshapes": face_blendshape_path
        }

        valid_features = list(feature_paths.keys())
        dataframes_to_merge = []
        id_cols_to_drop = ['frame', 'person_id']

        for feature_name in self.features:
            if feature_name not in valid_features:
                raise ValueError(f"Invalid feature '{feature_name}'. Valid options are: {valid_features}")
            path = feature_paths[feature_name]
            if not path or not os.path.exists(path):
                raise FileNotFoundError(
                    f"Path for feature '{feature_name}' not provided or file not found at '{path}'.")

            print(f"Loading {feature_name} from {os.path.basename(path)}...")
            df = pd.read_parquet(path)
            df = df[df["video_id"].astype(str).isin(video_ids_in_split)]
            df.drop(columns=[col for col in id_cols_to_drop if col in df.columns], inplace=True)
            df['frame_idx'] = df.groupby('video_id').cumcount()
            dataframes_to_merge.append(df)

        if not dataframes_to_merge:
            raise ValueError(f"No valid data files could be loaded for feature type: '{self.features_type}'")

        features_df = reduce(
            lambda left, right: pd.merge(left, right, on=['video_id', 'frame_idx', 'gloss'], how='outer'),
            dataframes_to_merge)
        features_df.fillna(-2, inplace=True)

        # 4. Determine the final list of feature columns
        final_id_cols = ['video_id', 'gloss', 'frame_idx', 'hand']
        initial_cols = [col for col in features_df.columns if col not in final_id_cols]

        cols_to_keep = initial_cols
        if self.fs == 1:
            print(f"Applying feature selection from: {feature_selection_dir}")
            ranked_features = {}
            sys.path.insert(0, feature_selection_dir)
            modules_to_delete = []
            try:
                feature_files = {
                    "hand_angles": "hand_angles_features", "pose_angles": "pose_angles_features",
                    "hand_landmarks": "hand_landmarks_features", "pose_landmarks": "pose_landmarks_features",
                    "face_landmarks": "face_landmarks_features", "face_blendshapes": "face_blendshapes_features"
                }
                for feature_name, module_name in feature_files.items():
                    if feature_name in self.features:
                        module = import_module(module_name)
                        ranked_features[feature_name] = module.TOP_FEATURES
                        modules_to_delete.append(module_name)
            finally:
                # --- Clean up both sys.path and sys.modules ---
                if feature_selection_dir in sys.path:
                    sys.path.remove(feature_selection_dir)
                for module_name in modules_to_delete:
                    if module_name in sys.modules:
                        del sys.modules[module_name]

            temp_cols = []
            # Handle landmarks by expanding base names to _x and _y
            for base in ranked_features.get("hand_landmarks", []) + ranked_features.get("pose_landmarks",
                                                                                        []) + ranked_features.get(
                    "face_landmarks", []):
                temp_cols.extend([f"{base}_x", f"{base}_y"])
            # Handle angles and blendshapes, which are single columns
            temp_cols.extend(ranked_features.get("hand_angles", []))
            temp_cols.extend(ranked_features.get("pose_angles", []))
            temp_cols.extend(ranked_features.get("face_blendshapes", []))

            # Filter initial_cols to only keep those selected
            cols_to_keep = [col for col in initial_cols if col in temp_cols]

            if not cols_to_keep:
                raise ValueError(
                    f"Feature selection (fs=1) resulted in an empty feature list. Check your top_features files.")

        # 5. Apply feature dimension padding/truncation
        if self.feature_padding_mode == "truncate":
            final_dim = (len(cols_to_keep) // self.n_heads) * self.n_heads
            self.final_columns = cols_to_keep[:final_dim]
        else:
            current_dim = len(cols_to_keep)
            target_dim = ((current_dim + self.n_heads - 1) // self.n_heads) * self.n_heads
            num_to_pad = target_dim - current_dim
            if num_to_pad > 0:
                pad_cols = [f"pad_{i}" for i in range(num_to_pad)]
                if self.feature_padding_mode == 'sentinel':
                    for pad_col in pad_cols: features_df[pad_col] = -2.0
                elif self.feature_padding_mode == 'repeat':
                    pad_source = [cols_to_keep[i % len(cols_to_keep)] for i in range(num_to_pad)]
                    for new, source in zip(pad_cols, pad_source):
                        features_df[new] = features_df[source]
                self.final_columns = cols_to_keep + pad_cols
            else:
                self.final_columns = cols_to_keep

        self.embedding_dim = len(self.final_columns)

        # 6. Finalize DataFrame for fast lookups
        self.features_map = features_df[["video_id"] + self.final_columns].groupby("video_id")
        self.video_lengths = self.features_map.size()
        if self.max_len is None: self.max_len = self.video_lengths.max()

        end_time = time.time()
        print(
            f"--- Dataset initialized in {end_time - start_time:.2f}s | Instances: {len(self.instances)} | Feat Dim: {self.embedding_dim} | Max Len: {self.max_len} ---")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        entry = self.instances[idx]
        video_id, gloss_idx = entry["video_id"], entry["gloss_idx"]
        try:
            group = self.features_map.get_group(video_id)
        except KeyError:
            print(f"Error: video_id '{video_id}' not in Parquet data.")
            return torch.full((self.max_len, self.embedding_dim), -2.0), torch.tensor(-1, dtype=torch.long)

        features = group[self.final_columns].values.astype(np.float32)
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        seq_len, _ = feature_tensor.shape
        if seq_len < self.max_len:
            pad = torch.full((self.max_len - seq_len, self.embedding_dim), -2.0)
            feature_tensor = torch.cat([feature_tensor, pad], dim=0)
        elif seq_len > self.max_len:
            feature_tensor = feature_tensor[:self.max_len, :]
        return feature_tensor, torch.tensor(gloss_idx, dtype=torch.long)

    @property
    def feature_dim(self):
        return self.embedding_dim