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
from WLASL_feature_processing.feature_selection.top_features.hand_angles_features import TOP_ANGLE_BASES as hand_angles
from WLASL_feature_processing.feature_selection.top_features.hand_landmarks_features import TOP_LANDMARKS_BASES as hand_landmarks
from WLASL_feature_processing.feature_selection.top_features.pose_landmarks_features import TOP_LANDMARKS_BASES as pose_landmarks
from WLASL_feature_processing.feature_selection.top_features.pose_angles_features import TOP_ANGLE_BASES as pose_angles


class WLASLParquetDataset(Dataset):
    def __init__(self, body_features_parquet_path, facial_blendshapes_parquet_path=None, metadata_json_path=None,
                 split="train",
                 transform=None, max_len=None, features='hand_pose_landmarks', include_blendshapes=False,
                 fs=0, feature_padding_mode="sentinel", n_heads=1):

        if body_features_parquet_path is None: raise ValueError("body_features_parquet_path cannot be None")
        if metadata_json_path is None: raise ValueError("metadata_json_path cannot be None")
        if include_blendshapes and not facial_blendshapes_parquet_path:
            raise ValueError("facial_blendshapes_parquet_path cannot be None if include_blendshapes is True")

        self.transform = transform
        self.split = split
        self.max_len = max_len
        self.features_type = features.lower()
        self.include_blendshapes = include_blendshapes
        self.fs = fs
        self.feature_padding_mode = feature_padding_mode
        self.n_heads = n_heads

        # 1. Load metadata and filter instances for the current split
        with open(metadata_json_path, "r") as f:
            metadata_raw = json.load(f)
        glosses = metadata_raw[0:100]
        self.instances = []
        for gloss_entry in glosses:
            for inst in gloss_entry["instances"]:
                if inst["split"] == self.split:
                    self.instances.append(
                        {"video_id": str(inst["video_id"]), "gloss": gloss_entry["gloss"], "gloss_idx": None,
                         "split": inst["split"]})
        glosses = sorted(set(entry["gloss"] for entry in self.instances))
        self.gloss2idx = {gloss: idx for idx, gloss in enumerate(glosses)}
        self.idx2gloss = {idx: gloss for gloss, idx in self.gloss2idx.items()}
        for entry in self.instances:
            entry["gloss_idx"] = self.gloss2idx[entry["gloss"]]
        video_ids = [entry["video_id"] for entry in self.instances]

        # 2. Load DataFrames and get initial, separate column lists
        body_df = pd.read_parquet(body_features_parquet_path)
        body_df = body_df[body_df["video_id"].astype(str).isin(video_ids)]

        # Get body feature columns BEFORE any merge operations
        initial_body_cols = body_df.drop(columns=['video_id', 'gloss'], errors='ignore').columns.tolist()
        initial_blendshape_cols = []

        features_df = body_df  # Start with body_df as the base

        if self.include_blendshapes:
            facial_df = pd.read_parquet(facial_blendshapes_parquet_path)
            facial_df = facial_df[facial_df["video_id"].astype(str).isin(video_ids)]

            # Get blendshape columns BEFORE any merge operations
            initial_blendshape_cols = facial_df.drop(columns=['video_id', 'gloss'], errors='ignore').columns.tolist()

            # Merge the DataFrames to create a unified data source
            body_df['frame_idx'] = body_df.groupby('video_id').cumcount()
            facial_df['frame_idx'] = facial_df.groupby('video_id').cumcount()
            features_df = pd.merge(
                body_df, facial_df, on=['video_id', 'frame_idx'], how='inner',
                suffixes=('_body', '_facial')  # Handles 'gloss' column collision
            ).drop(columns=['frame_idx'])

        # 3. Determine the final list of BODY feature columns to keep
        if self.fs == 1:
            if self.features_type == "hand_angles":
                ranked_features = hand_angles
            elif self.features_type == "pose_angles":
                ranked_features = pose_angles
            elif self.features_type == "hand_pose_angles":
                ranked_features = hand_angles + pose_angles
            elif self.features_type == "hand_landmarks":
                ranked_features = hand_landmarks
            elif self.features_type == "pose_landmarks":
                ranked_features = pose_landmarks
            elif self.features_type == "hand_pose_landmarks":
                ranked_features = hand_landmarks + pose_landmarks
            else:
                raise ValueError(f"[ERROR] Unknown feature selection group: {self.features_type}")

            if "avasag_vitpose_extracted_landmarks" in self.features_type:
                body_cols_to_keep = []
                for base in ranked_features:
                    body_cols_to_keep.extend([f"{base}_x", f"{base}_y"])
            else:
                body_cols_to_keep = [col for col in initial_body_cols if col in ranked_features]
        else:
            body_cols_to_keep = initial_body_cols

        # Apply filters ONLY to the list of body feature names
        def is_forbidden_pose_column(col_name):
            if col_name.startswith("P#"):
                try:
                    idx = int(col_name.split("_")[0].replace("P#", ""));
                    return 25 <= idx <= 32
                except:
                    return False
            elif "P_Angle{" in col_name:
                try:
                    numbers = [int(n) for n in
                               col_name.replace("P_Angle{", "").replace("}", "").replace("(", "").replace(")",
                                                                                                          "").replace(
                                   "-", ",").split(",")];
                    return any(25 <= n <= 32 for n in numbers)
                except:
                    return False
            return False

        body_cols_to_keep = [c for c in body_cols_to_keep if not is_forbidden_pose_column(c)]
        if "avasag_vitpose_extracted_landmarks" in self.features_type:
            body_cols_to_keep = [c for c in body_cols_to_keep if not c.endswith("_z")]

        # 4. Create the final combined list of columns
        combined_cols = body_cols_to_keep + initial_blendshape_cols

        # 5. Apply feature dimension padding/truncation
        if self.feature_padding_mode == "truncate":
            k = len(combined_cols)
            final_dim = (k // n_heads) * n_heads
            self.final_columns = combined_cols[:final_dim]
        else:
            current_dim = len(combined_cols)
            target_dim = ((current_dim + n_heads - 1) // n_heads) * n_heads
            num_to_pad = target_dim - current_dim
            if num_to_pad > 0:
                pad_cols = [f"pad_{i}" for i in range(num_to_pad)]
                if self.feature_padding_mode == 'sentinel':
                    for pad_col in pad_cols:
                        features_df[pad_col] = -2.0
                elif self.feature_padding_mode == 'repeat':
                    pad_source_cols = [combined_cols[i % len(combined_cols)] for i in range(num_to_pad)]
                    for new_name, source_name in zip(pad_cols, pad_source_cols):
                        features_df[new_name] = features_df[source_name]
                else:
                    raise ValueError(f"Unknown feature_padding_mode: '{self.feature_padding_mode}'")
                self.final_columns = combined_cols + pad_cols
            else:
                self.final_columns = combined_cols

        self.embedding_dim = len(self.final_columns)

        # # 6. Finalize the main DataFrame for use in __getitem__
        # # Identify the correct video_id column (could be 'video_id' or 'video_id_body' after merge)
        # vid_col = 'video_id' if 'video_id' in features_df.columns else 'video_id_body'
        # self.features_df = features_df[[vid_col] + self.final_columns]
        # self.features_df = self.features_df.rename(columns={vid_col: 'video_id'})  # Standardize name

        features_df = features_df[["video_id"] + self.final_columns]
        self.features_map = features_df.groupby("video_id")

        self.video_lengths = self.features_map.size()
        self.max_seq_len = self.video_lengths.max()
        if self.max_len is None:
            self.max_len = self.max_seq_len

    # --- (__len__, __getitem__, feature_dim are the same as the previous correct version) ---
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        entry = self.instances[idx]
        video_id = entry["video_id"]
        gloss_idx = entry["gloss_idx"]

        group = self.features_map.get_group(video_id)
        features_df = group.drop(columns=["video_id"])

        features = features_df.values.astype(np.float32)
        feature_tensor = torch.tensor(features)

        if self.max_len:
            seq_len, feat_dim = feature_tensor.shape
            if seq_len < self.max_len:
                pad = torch.full((self.max_len - seq_len, feat_dim), fill_value=-2.0)
                feature_tensor = torch.cat([feature_tensor, pad], dim=0)
            elif seq_len > self.max_len:
                feature_tensor = feature_tensor[:self.max_len, :]

        label_tensor = torch.tensor(gloss_idx, dtype=torch.long)
        return feature_tensor, label_tensor

    @property
    def feature_dim(self):
        return self.embedding_dim
