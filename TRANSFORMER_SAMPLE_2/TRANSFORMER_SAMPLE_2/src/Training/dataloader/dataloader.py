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


# class WLASLParquetDataset(Dataset):
#     def __init__(self, body_features_parquet_path, facial_blendshapes_parquet_path=None, metadata_json_path=None, split="train",
#                  transform=None, max_len=None, features='hand_pose_landmarks', include_blendshapes=False, fs=0, feature_truncation=False, n_heads=1):
#         if body_features_parquet_path is None:
#             raise ValueError("parquet_path cannot be None")
#         self.body_features_df = pd.read_parquet(body_features_parquet_path)  # landmarks or angles
#         if metadata_json_path is None:
#             raise ValueError("metadata_json_path cannot be None")
#         if include_blendshapes and not facial_blendshapes_parquet_path:
#             raise ValueError("facial_blendshapes_parquet_path cannot be None")
#         if include_blendshapes and facial_blendshapes_parquet_path is not None:
#             self.facial_blendshapes_df = pd.read_parquet(facial_blendshapes_parquet_path)
#             self.facial_features_map = self.facial_blendshapes_df.groupby("video_id")
#         self.transform = transform
#         self.split = split
#         self.max_len = max_len  # depends on features chosen
#         self.features = features.lower()  # select features to take, choices=["HAND_LANDMARKS", "POSE_LANDMARKS", "HAND_POSE_LANDMARKS", "HAND_ANGLES", "POSE_ANGLES", "HAND_POSE_ANGLES"]
#         self.include_blendshapes = include_blendshapes  # choose whether to include facial blendshapes
#         self.fs = fs  # use feature selection or not
#         self.feature_truncation = feature_truncation
#         self.n_heads = n_heads
#
#         # Load metadata and filter by split
#         with open(metadata_json_path, "r") as f:
#             metadata_raw = json.load(f)
#
#         # Loaded first 100 glosses
#         glosses = metadata_raw[0:100]
#
#         # Flatten instances with correct split
#         self.instances = []
#         for gloss_entry in glosses:
#             for inst in gloss_entry["instances"]:
#                 if inst["split"] == self.split:
#                     self.instances.append({
#                         "video_id": str(inst["video_id"]),
#                         "gloss": gloss_entry["gloss"],
#                         "gloss_idx": None,  # filled below
#                         "split": inst["split"],
#                     })
#
#         # Assign numeric labels to glosses
#         glosses = sorted(set(entry["gloss"] for entry in self.instances))
#         self.gloss2idx = {gloss: idx for idx, gloss in enumerate(glosses)}
#         self.idx2gloss = {idx: gloss for gloss, idx in self.gloss2idx.items()}
#         for entry in self.instances:
#             entry["gloss_idx"] = self.gloss2idx[entry["gloss"]]
#
#         # Filter feature DataFrame by available video_ids to get train/val/test instances
#         video_ids = [entry["video_id"] for entry in self.instances]
#         # Filter only valid video IDs
#         self.body_features_df = self.body_features_df[self.body_features_df["video_id"].astype(str).isin(video_ids)]
#
#         # # TODO: Apply or not gaussian noise or other transformation
#         # transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])
#
#         # Group all rows by video_id → returns a GroupBy object
#         self.body_features_map = self.body_features_df.groupby("video_id")
#
#         # Find maximum sequence length (number of frames per video)
#         self.video_lengths = self.body_features_map.size()  # Series: video_id → count
#         # print(self.video_lengths)
#         self.max_seq_len = self.video_lengths.max()
#
#         if self.max_len is None:
#             self.max_len = self.max_seq_len  # Use max frames in dataset if not manually specified
#
#         # print(f"[INFO] Max sequence length in dataset is {self.max_seq_len}")
#
#
#     def __len__(self):
#         return len(self.instances)
#
#     def __pad_to_heads__(self, x):
#         """
#         Pad the tensor x features to be divisible by n_heads.
#         :param x: the input tensor
#         :return: the padded tensor
#         """
#         T, D = x.shape
#         if D % self.n_heads != 0:
#             target_dim = ((D + self.n_heads - 1) // self.n_heads) * self.n_heads
#             pad_width = target_dim - D
#             pad_tensor = torch.full((T, pad_width), fill_value=-2.0, device=x.device)
#             x = torch.cat([x, pad_tensor], dim=-1)
#         return x
#
#     def __getitem__(self, idx):
#         entry = self.instances[idx]
#         video_id = entry["video_id"]
#         gloss_idx = entry["gloss_idx"]
#
#         # Get group of all frames for this video
#         group = self.body_features_map.get_group(video_id)
#         # print(group)
#
#         body_features_df = group.drop(columns=["gloss", "video_id"], errors="ignore")
#
#         # Feature selection logic
#         if self.fs == 1:
#             if self.features == "hand_angles":
#                 ranked_features = hand_angles
#             elif self.features == "pose_angles":
#                 ranked_features = pose_angles
#             elif self.features == "hand_pose_angles":
#                 ranked_features = hand_angles + pose_angles
#             elif self.features == "hand_landmarks":
#                 ranked_features = hand_landmarks
#             elif self.features == "pose_landmarks":
#                 ranked_features = pose_landmarks
#             elif self.features == "hand_pose_landmarks":
#                 ranked_features = hand_landmarks + pose_landmarks
#             else:
#                 raise ValueError(f"[ERROR] Unknown feature selection group: {self.features}")
#
#             # get available columns in the data
#             available_features = body_features_df.columns.tolist()
#
#             if "landmarks" in self.features:
#                 # landmarks columns by decreasing order of rank
#                 keep_cols = []
#                 for base in ranked_features:
#                     keep_cols.append(base+"_x")
#                     keep_cols.append(base+"_y")
#             else:   # angles do not have bases as they do not have _x or _y
#                 keep_cols = [col for col in available_features if col in ranked_features]
#
#             # Determine k:
#             k = len(keep_cols)
#             if self.feature_truncation:
#                 k = (k // self.n_heads) * self.n_heads  # largest divisible by n_heads
#
#             body_features_df = body_features_df[keep_cols[:k]]
#
#         # Drop any pose landmark or pose angle involving landmarks 25–32 "lower limbs"
#         if self.features in ["pose_landmarks", "pose_angles", "hand_pose_landmarks", "hand_pose_angles"]:
#             def is_forbidden_pose_column(col_name):
#                 # Pose landmark format: "P#25", "P#31", etc.
#                 if col_name.startswith("P#"):
#                     try:
#                         idx = int(col_name.split("_")[0].replace("P#", ""))
#                         return 25 <= idx <= 32
#                     except:
#                         return False
#                 # Pose angle format: "P_Angle{(12, 14)-(2, 3)}"
#                 elif "P_Angle{" in col_name:
#                     try:
#                         numbers = [int(n) for n in
#                                    col_name.replace("P_Angle{", "").replace("}", "").replace("(", "").replace(")",
#                                                                                                               "").replace(
#                                        "-", ",").split(",")]
#                         return any(25 <= n <= 32 for n in numbers)
#                     except:
#                         return False
#                 return False
#
#             body_features_df = body_features_df[[col for col in body_features_df.columns if not is_forbidden_pose_column(col)]]
#
#         # Drop z-axis if using landmarks
#         if "landmarks" in self.features:
#             body_features_df = body_features_df[[col for col in body_features_df.columns if not col.endswith('_z')]]
#
#         if self.include_blendshapes:
#             blend_group = self.facial_features_map.get_group(video_id)
#             blend_features_df = blend_group.drop(columns=["video_id", "gloss"], errors="ignore")
#             if len(body_features_df) != len(blend_features_df):
#                 raise ValueError(
#                     f"Blendshapes frame count ({len(blend_features_df)}) != features frame count ({len(body_features_df)}) for video {video_id}")
#             blend_features_df = blend_features_df.reset_index(drop=True)
#             body_features_df = pd.concat([body_features_df.reset_index(drop=True), blend_features_df], axis=1)
#
#         # Now convert to NumPy
#         features = body_features_df.values.astype(np.float32)
#         feature_tensor = torch.tensor(features)  # shape: (n_frames, n_features)
#
#         # TODO: Optional transform
#         # if self.transform:
#         #     feature_tensor = self.transform(feature_tensor)
#
#         # Pad to max_len
#         if self.max_len:
#             seq_len, feat_dim = feature_tensor.shape
#             if seq_len < self.max_len:
#                 pad = torch.full((self.max_len - seq_len, feat_dim), fill_value=-2.0)
#                 feature_tensor = torch.cat([feature_tensor, pad], dim=0)
#
#         # print(feature_tensor)
#         label_tensor = torch.tensor(gloss_idx, dtype=torch.long)
#
#         if not self.feature_truncation:  # pad features if not truncating to be divisible by number of heads
#             feature_tensor = self.__pad_to_heads__(feature_tensor)
#
#         self.embedding_dim = feature_tensor.shape[-1]
#
#         return feature_tensor, label_tensor
#
#     @property
#     def feature_dim(self):
#         return getattr(self, "embedding_dim", None)


class WLASLParquetDataset(Dataset):
    def __init__(self, body_features_parquet_path, facial_blendshapes_parquet_path=None, metadata_json_path=None,
                 split="train",
                 transform=None, max_len=None, features='hand_pose_landmarks', include_blendshapes=False,
                 fs=0, feature_truncation=False, n_heads=1):
        if body_features_parquet_path is None:
            raise ValueError("parquet_path cannot be None")
        self.body_features_df = pd.read_parquet(body_features_parquet_path)
        if metadata_json_path is None:
            raise ValueError("metadata_json_path cannot be None")
        if include_blendshapes and not facial_blendshapes_parquet_path:
            raise ValueError("facial_blendshapes_parquet_path cannot be None")
        if include_blendshapes and facial_blendshapes_parquet_path is not None:
            self.facial_blendshapes_df = pd.read_parquet(facial_blendshapes_parquet_path)
            self.facial_features_map = self.facial_blendshapes_df.groupby("video_id")

        self.transform = transform
        self.split = split
        self.max_len = max_len
        self.features = features.lower()
        self.include_blendshapes = include_blendshapes
        self.fs = fs
        self.feature_truncation = feature_truncation
        self.n_heads = n_heads

        # metadata loading
        with open(metadata_json_path, "r") as f:
            metadata_raw = json.load(f)

        glosses = metadata_raw[0:100]
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

        video_ids = [entry["video_id"] for entry in self.instances]
        self.body_features_df = self.body_features_df[self.body_features_df["video_id"].astype(str).isin(video_ids)]
        self.body_features_map = self.body_features_df.groupby("video_id")
        self.video_lengths = self.body_features_map.size()
        self.max_seq_len = self.video_lengths.max()
        if self.max_len is None:
            self.max_len = self.max_seq_len

        # ====== DECIDE THE FINAL FEATURE DIM (padding or truncation) ======
        sample_id = self.instances[0]["video_id"]
        columns = self.body_features_map.get_group(sample_id).drop(columns=["gloss", "video_id"],
                                                                   errors="ignore").columns
        available_features = columns.tolist()

        # feature selection
        if self.fs == 1:
            if self.features == "hand_angles":
                ranked_features = hand_angles
            elif self.features == "pose_angles":
                ranked_features = pose_angles
            elif self.features == "hand_pose_angles":
                ranked_features = hand_angles + pose_angles
            elif self.features == "hand_landmarks":
                ranked_features = hand_landmarks
            elif self.features == "pose_landmarks":
                ranked_features = pose_landmarks
            elif self.features == "hand_pose_landmarks":
                ranked_features = hand_landmarks + pose_landmarks
            else:
                raise ValueError(f"[ERROR] Unknown feature selection group: {self.features}")

            if "landmarks" in self.features:
                # landmarks columns by decreasing order of importance score
                keep_cols = []
                for base in ranked_features:
                    keep_cols.append(base + "_x")
                    keep_cols.append(base + "_y")
            else:  # angles do not have bases as they do not have _x or _y
                keep_cols = [col for col in available_features if col in ranked_features]
        else:
            keep_cols = available_features

        # remove lower-limb features
        def is_forbidden_pose_column(col_name):
            if col_name.startswith("P#"):
                try:
                    idx = int(col_name.split("_")[0].replace("P#", ""))
                    return 25 <= idx <= 32
                except:
                    return False
            elif "P_Angle{" in col_name:
                try:
                    numbers = [int(n) for n in col_name.replace("P_Angle{", "").replace("}", "")
                    .replace("(", "").replace(")", "").replace("-", ",").split(",")]
                    return any(25 <= n <= 32 for n in numbers)
                except:
                    return False
            return False

        keep_cols = [c for c in keep_cols if not is_forbidden_pose_column(c)]

        # drop z-axis if landmarks
        if "landmarks" in self.features:
            keep_cols = [c for c in keep_cols if not c.endswith("_z")]

        # apply truncation or padding logic
        k = len(keep_cols)
        if feature_truncation:
            k = (k // n_heads) * n_heads
            self.final_columns = keep_cols[:k]
        else:
            current = len(keep_cols)
            target = ((current + n_heads - 1) // n_heads) * n_heads
            pad_cols = [f"pad_{i}" for i in range(target - current)]
            self.final_columns = keep_cols + pad_cols

        self.embedding_dim = len(self.final_columns)
        print(f"[INFO] dataset feature_dim is set to {self.embedding_dim}")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        entry = self.instances[idx]
        video_id = entry["video_id"]
        gloss_idx = entry["gloss_idx"]
        group = self.body_features_map.get_group(video_id).drop(columns=["gloss", "video_id"], errors="ignore")

        # add any pad columns if missing
        for pad_col in [c for c in self.final_columns if c.startswith("pad_")]:
            group[pad_col] = -2.0

        group = group[[col for col in self.final_columns if col in group.columns]]

        features = group.values.astype(np.float32)
        feature_tensor = torch.tensor(features)

        if self.max_len:
            seq_len, feat_dim = feature_tensor.shape
            if seq_len < self.max_len:
                pad = torch.full((self.max_len - seq_len, feat_dim), fill_value=-2.0)
                feature_tensor = torch.cat([feature_tensor, pad], dim=0)

        label_tensor = torch.tensor(gloss_idx, dtype=torch.long)
        return feature_tensor, label_tensor

    @property
    def feature_dim(self):
        return self.embedding_dim
