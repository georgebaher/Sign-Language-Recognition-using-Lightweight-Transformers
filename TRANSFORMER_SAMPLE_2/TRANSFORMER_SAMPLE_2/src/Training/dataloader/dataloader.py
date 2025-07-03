import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import numpy as np
from Feature_Processing.feature_selection.features.hand_angles_features import TOP_ANGLE_BASES as hand_angles
# from Feature_Processing.feature_selection.features.hand_pose_angles_features import TOP_ANGLE_BASES as hand_pose_angles
from Feature_Processing.feature_selection.features.hand_landmarks_features import TOP_LANDMARKS_BASES as hand_landmarks
# from Feature_Processing.feature_selection.features.hand_pose_landmarks_features import TOP_LANDMARKS_BASES as hand_pose_landmarks
from Feature_Processing.feature_selection.features.pose_landmarks_features import TOP_LANDMARKS_BASES as pose_landmarks
from Feature_Processing.feature_selection.features.pose_angles_features import TOP_ANGLE_BASES as pose_angles

class WLASLParquetDataset(Dataset):
    def __init__(self, body_features_parquet_path, facial_blendshapes_parquet_path=None, metadata_json_path=None, split="train",
                 transform=None, max_len=None, features='hand_pose_landmarks', include_blendshapes=False, fs=0):
        if body_features_parquet_path is None:
            raise ValueError("parquet_path cannot be None")
        self.body_features_df = pd.read_parquet(body_features_parquet_path)  # landmarks or angles
        if metadata_json_path is None:
            raise ValueError("metadata_json_path cannot be None")
        if include_blendshapes and not facial_blendshapes_parquet_path:
            raise ValueError("facial_blendshapes_parquet_path cannot be None")
        if include_blendshapes and facial_blendshapes_parquet_path is not None:
            self.facial_blendshapes_df = pd.read_parquet(facial_blendshapes_parquet_path)
            self.facial_features_map = self.facial_blendshapes_df.groupby("video_id")
        self.transform = transform
        self.split = split
        self.max_len = max_len  # depends on features chosen
        self.features = features.lower()  # select features to take, choices=["HAND_LANDMARKS", "POSE_LANDMARKS", "HAND_POSE_LANDMARKS", "HAND_ANGLES", "POSE_ANGLES", "HAND_POSE_ANGLES"]
        self.include_blendshapes = include_blendshapes  # choose whether to include facial blendshapes
        self.fs = fs  # use feature selection or not

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
        self.body_features_df = self.body_features_df[self.body_features_df["video_id"].astype(str).isin(video_ids)]

        # # TODO: Apply or not gaussian noise or other transformation
        # transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])

        # Group all rows by video_id → returns a GroupBy object
        self.body_features_map = self.body_features_df.groupby("video_id")

        # Find maximum sequence length (number of frames per video)
        self.video_lengths = self.body_features_map.size()  # Series: video_id → count
        # print(self.video_lengths)
        self.max_seq_len = self.video_lengths.max()

        if self.max_len is None:
            self.max_len = self.max_seq_len  # Use max frames in dataset if not manually specified

        # print(f"[INFO] Max sequence length in dataset is {self.max_seq_len}")


    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        entry = self.instances[idx]
        video_id = entry["video_id"]
        gloss_idx = entry["gloss_idx"]

        # Get group of all frames for this video
        group = self.body_features_map.get_group(video_id)
        # print(group)

        body_features_df = group.drop(columns=["gloss", "video_id"], errors="ignore")

        # Feature selection logic
        if self.fs == 1:
            if self.features == "hand_angles":
                selected_features = hand_angles
            elif self.features == "pose_angles":
                selected_features = pose_angles
            elif self.features == "hand_pose_angles":
                selected_features = hand_angles + pose_angles
            elif self.features == "hand_landmarks":
                selected_features = hand_landmarks
            elif self.features == "pose_landmarks":
                selected_features = pose_landmarks
            elif self.features == "hand_pose_landmarks":
                selected_features = hand_landmarks + pose_landmarks
            else:
                raise ValueError(f"[ERROR] Unknown feature selection group: {self.features}")
            # Keep only the selected features
            if "landmarks" in self.features:
                body_features_df = body_features_df[[col for col in body_features_df.columns if col.split("_")[0] in selected_features]]
            else:   # if angles
                body_features_df = body_features_df[[col for col in body_features_df.columns if col in selected_features]]

        # Drop any pose landmark or pose angle involving landmarks 25–32 "lower limbs"
        if self.features in ["pose_landmarks", "pose_angles", "hand_pose_landmarks", "hand_pose_angles"]:
            def is_forbidden_pose_column(col_name):
                # Pose landmark format: "P#25", "P#31", etc.
                if col_name.startswith("P#"):
                    try:
                        idx = int(col_name.split("_")[0].replace("P#", ""))
                        return 25 <= idx <= 32
                    except:
                        return False
                # Pose angle format: "P_Angle{(12, 14)-(2, 3)}"
                elif "P_Angle{" in col_name:
                    try:
                        numbers = [int(n) for n in
                                   col_name.replace("P_Angle{", "").replace("}", "").replace("(", "").replace(")",
                                                                                                              "").replace(
                                       "-", ",").split(",")]
                        return any(25 <= n <= 32 for n in numbers)
                    except:
                        return False
                return False

            body_features_df = body_features_df[[col for col in body_features_df.columns if not is_forbidden_pose_column(col)]]

        # Drop z-axis if using landmarks
        if "landmarks" in self.features:
            body_features_df = body_features_df[[col for col in body_features_df.columns if not col.endswith('_z')]]

        if self.include_blendshapes:
            blend_group = self.facial_features_map.get_group(video_id)
            blend_features_df = blend_group.drop(columns=["video_id", "gloss"], errors="ignore")
            if len(body_features_df) != len(blend_features_df):
                raise ValueError(
                    f"Blendshapes frame count ({len(blend_features_df)}) != features frame count ({len(body_features_df)}) for video {video_id}")
            blend_features_df = blend_features_df.reset_index(drop=True)
            body_features_df = pd.concat([body_features_df.reset_index(drop=True), blend_features_df], axis=1)

        # Now convert to NumPy
        features = body_features_df.values.astype(np.float32)
        feature_tensor = torch.tensor(features)  # shape: (n_frames, n_features)

        # print(f'shape of feature tensor: {feature_tensor.shape}')

        # TODO: Optional transform
        # if self.transform:
        #     feature_tensor = self.transform(feature_tensor)

        # Pad to max_len
        if self.max_len:
            seq_len, feat_dim = feature_tensor.shape
            if seq_len < self.max_len:
                pad = torch.full((self.max_len - seq_len, feat_dim), fill_value=-2.0)
                feature_tensor = torch.cat([feature_tensor, pad], dim=0)

        # print(feature_tensor)
        label_tensor = torch.tensor(gloss_idx, dtype=torch.long)
        return feature_tensor, label_tensor

