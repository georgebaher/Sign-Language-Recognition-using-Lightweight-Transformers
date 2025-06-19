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
from Feature_Processing.feature_selection.features.hand_pose_angles_features import TOP_ANGLE_BASES as hand_pose_angles
from Feature_Processing.feature_selection.features.hand_landmarks_features import TOP_LANDMARKS_BASES as hand_landmarks
from Feature_Processing.feature_selection.features.hand_pose_landmarks_features import TOP_LANDMARKS_BASES as hand_pose_landmarks
from Feature_Processing.feature_selection.features.pose_landmarks_features import TOP_LANDMARKS_BASES as pose_landmarks
from Feature_Processing.feature_selection.features.pose_angles_features import TOP_ANGLE_BASES as pose_angles

class WLASLParquetDataset(Dataset):
    def __init__(self, parquet_path, metadata_json_path, split="train",
                 transform=None, max_len=None, features='hand_pose_landmarks', fs=0):
        if parquet_path is None:
            raise ValueError("parquet_path cannot be None")
        if metadata_json_path is None:
            raise ValueError("metadata_json_path cannot be None")
        self.df = pd.read_parquet(parquet_path)  # landmarks or angles
        self.transform = transform
        self.split = split
        self.max_len = max_len  # depends on features chosen
        self.features = features.lower()  # select features to take
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

        # print(f"[INFO] Max sequence length in dataset is {self.max_seq_len}")


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

        # Feature selection logic
        if self.fs == 1:
            if self.features == "hand_angles":
                selected_features = hand_angles
            elif self.features == "pose_angles":
                selected_features = pose_angles
            elif self.features == "hand_pose_angles":
                selected_features = hand_pose_angles
            elif self.features == "hand_landmarks":
                selected_features = hand_landmarks
            elif self.features == "pose_landmarks":
                selected_features = pose_landmarks
            elif self.features == "hand_pose_landmarks":
                selected_features = hand_pose_landmarks
            else:
                raise ValueError(f"[ERROR] Unknown feature selection group: {self.features}")
            # Keep only the selected features
            if "landmarks" in self.features:
                features_df = features_df[[col for col in features_df.columns if col.split("_")[0] in selected_features]]
            else:
                features_df = features_df[[col for col in features_df.columns if col in selected_features]]


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

            features_df = features_df[[col for col in features_df.columns if not is_forbidden_pose_column(col)]]

        # Drop z-axis if using landmarks
        if "landmarks" in self.features:
            features_df = features_df[[col for col in features_df.columns if not col.endswith('_z')]]

        # Now convert to NumPy
        features = features_df.values.astype(np.float32)
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

