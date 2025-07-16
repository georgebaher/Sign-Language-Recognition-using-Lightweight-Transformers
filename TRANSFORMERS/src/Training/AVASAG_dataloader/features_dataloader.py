# # avasign100_dataset.py
#
# import torch
# from torch.utils.data import Dataset
# import pandas as pd
# import json
# import numpy as np
# from functools import reduce
#
# # --- Placeholder for your Feature Selection Lists ---
# # Replace these with the actual lists of top top_features you have.
# # These are only used if you set the 'fs=1' argument.
# TOP_HAND_ANGLES = ['left_thumb_flexion', 'right_wrist_flexion', 'left_index_flexion', 'right_index_flexion',
#                    'left_wrist_deviation']  # Example
# TOP_POSE_ANGLES = ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder', 'left_knee']  # Example
# TOP_HAND_LANDMARKS = ['h0', 'h4', 'h8', 'h12', 'h16', 'h21', 'h25', 'h29', 'h33', 'h37']  # Example
# TOP_POSE_LANDMARKS = ['p0', 'p5', 'p6', 'p9', 'p10', 'p11', 'p12']  # Example
#
#
# class AVASign100ParquetDataset(Dataset):
#     """
#     A PyTorch Dataset for loading AVASign100 landmark data stored in Parquet files.
#
#     This class loads and combines data from separate Parquet files for pose, face,
#     and hand avasag_vitpose_extracted_landmarks and/or angles. It handles metadata for splits, feature selection,
#     and padding for use in a Transformer model.
#     """
#
#     def __init__(self,
#                  metadata_json_path,
#                  pose_landmark_path=None, face_landmark_path=None, hand_landmark_path=None,
#                  pose_angle_path=None, hand_angle_path=None,
#                  split="train",
#                  top_features='hand_pose_landmarks',
#                  fs=0,
#                  max_len=None,
#                  feature_padding_mode="sentinel",
#                  n_heads=1):
#         """
#         Args:
#             metadata_json_path (str): Path to the WLASL-style metadata JSON file.
#             pose_landmark_path (str, optional): Path to the POSE_LANDMARKS.parquet file.
#             face_landmark_path (str, optional): Path to the FACE_LANDMARKS.parquet file.
#             hand_landmark_path (str, optional): Path to the HAND_LANDMARKS.parquet file.
#             pose_angle_path (str, optional): Path to the pose_angles.parquet file.
#             hand_angle_path (str, optional): Path to the hand_angles.parquet file.
#             split (str): The dataset split to load ("train", "val", or "test").
#             top_features (str): The type of top_features to use. E.g., "hand_pose_landmarks", "hand_pose_angles".
#             fs (int): If 1, applies feature selection using the predefined top feature lists.
#             max_len (int, optional): The maximum sequence length. Sequences will be padded or truncated.
#             feature_padding_mode (str): 'sentinel' or 'repeat'. How to pad the feature dimension.
#             n_heads (int): Number of attention heads, used for feature dimension padding.
#         """
#         # 1. Argument validation and self.variable setup
#         if metadata_json_path is None: raise ValueError("metadata_json_path cannot be None")
#         self.split = split
#         self.features_type = top_features.lower()
#         self.fs = fs
#         self.max_len = max_len
#         self.feature_padding_mode = feature_padding_mode
#         self.n_heads = n_heads
#
#         # 2. Load metadata and filter instances for the current split
#         with open(metadata_json_path, "r", encoding="utf-8") as f:
#             metadata_raw = json.load(f)
#
#         self.instances = []
#         for gloss_entry in metadata_raw:
#             for inst in gloss_entry["instances"]:
#                 if inst["split"] == self.split:
#                     self.instances.append({
#                         "video_id": str(inst["video_id"]),
#                         "gloss": gloss_entry["gloss"]
#                     })
#
#         glosses = sorted(set(entry["gloss"] for entry in self.instances))
#         self.gloss2idx = {gloss: idx for idx, gloss in enumerate(glosses)}
#         self.idx2gloss = {idx: gloss for gloss, idx in self.gloss2idx.items()}
#         for entry in self.instances:
#             entry["gloss_idx"] = self.gloss2idx[entry["gloss"]]
#
#         video_ids_in_split = {entry["video_id"] for entry in self.instances}
#
#         # 3. Load and merge the required Parquet files
#         feature_paths = {
#             "pose_landmarks": pose_landmark_path, "face_landmarks": face_landmark_path,
#             "hand_landmarks": hand_landmark_path, "pose_angles": pose_angle_path,
#             "hand_angles": hand_angle_path
#         }
#
#         dataframes_to_merge = []
#         for feature_name, path in feature_paths.items():
#             if feature_name in self.features_type and path:
#                 df = pd.read_parquet(path)
#                 # Filter DataFrame to only include videos in the current split for efficiency
#                 df = df[df["video_id"].astype(str).isin(video_ids_in_split)]
#                 dataframes_to_merge.append(df)
#
#         if not dataframes_to_merge:
#             raise ValueError(f"No data files could be loaded for the feature type: '{self.features_type}'")
#
#         # Merge all loaded dataframes into one, aligned by video_id and frame
#         features_df = reduce(
#             lambda left, right: pd.merge(left, right, on=['video_id', 'frame', 'person_id', 'gloss'], how='outer'),
#             dataframes_to_merge)
#         # Fill any NaNs that result from the outer merge with our padding value
#         features_df.fillna(-2, inplace=True)
#
#         # 4. Determine the final list of feature columns to keep
#         initial_cols = [col for col in features_df.columns if col not in ['video_id', 'frame', 'person_id', 'gloss']]
#
#         if self.fs == 1:
#             if self.features_type == "hand_angles":
#                 ranked_features = TOP_HAND_ANGLES
#             elif self.features_type == "pose_angles":
#                 ranked_features = TOP_POSE_ANGLES
#             elif self.features_type == "hand_pose_angles":
#                 ranked_features = TOP_HAND_ANGLES + TOP_POSE_ANGLES
#             elif self.features_type == "hand_landmarks":
#                 ranked_features = TOP_HAND_LANDMARKS
#             elif self.features_type == "pose_landmarks":
#                 ranked_features = TOP_POSE_LANDMARKS
#             elif self.features_type == "hand_pose_landmarks":
#                 ranked_features = TOP_HAND_LANDMARKS + TOP_POSE_LANDMARKS
#             else:
#                 raise ValueError(f"[ERROR] Unknown feature selection group: {self.features_type}")
#
#             if "avasag_vitpose_extracted_landmarks" in self.features_type:
#                 cols_to_keep = []
#                 for base in ranked_features:
#                     cols_to_keep.extend([f"{base}_x", f"{base}_y"])
#             else:  # angles
#                 cols_to_keep = [col for col in initial_cols if col in ranked_features]
#         else:
#             cols_to_keep = initial_cols
#
#         # 5. Apply feature dimension padding/truncation
#         if self.feature_padding_mode == "truncate":
#             k = len(cols_to_keep)
#             final_dim = (k // self.n_heads) * self.n_heads
#             self.final_columns = cols_to_keep[:final_dim]
#         else:  # sentinel or repeat
#             current_dim = len(cols_to_keep)
#             target_dim = ((current_dim + self.n a new, compatible PyTorch `Dataset`
#
#             class called `AVASAGParquetDataset`.
#
# ### Key Design Changes and Adaptations
#
#
#
#
# import os
# import torch
# from torch.utils.data import Dataset
# import pandas as pd
# import json
# import numpy as np
#
#
# class AVASAGParquetDataset(Dataset):
#     def __init__(self, parquet_root_path, metadata_json_path,
#                  split="train",
#                  top_features='hand_pose_landmarks',
#                  max_len=None,
#                  feature_padding_mode="sentinel",
#                  n_heads=1):
#         """
#         PyTorch Dataset for the AVASAG-100 data stored in Parquet files.
#
#         Args:
#             parquet_root_path (str): Path to the folder containing the five Parquet files.
#             metadata_json_path (str): Path to the main metadata JSON file.
#             split (str): The dataset split to use ('train', 'val', or 'test').
#             top_features (str): The feature group to load. Options include:
#                             'pose_landmarks', 'hand_landmarks', 'face_landmarks',
#                             'pose_angles', 'hand_angles', or combinations joined by '_'.
#                             Example: 'hand_pose_landmarks'.
#             max_len (int, optional): The maximum sequence length. Sequences will be
#                                      padded or truncated to this length. Defaults to None.
#             feature_padding_mode (str): How to pad the feature dimension. 'sentinel' or 'repeat'.
#             n_heads (int): Number of attention heads, used to calculate feature padding.
#         """
#         # --- 1. Argument Validation and Setup ---
#         if not os.path.isdir(parquet_root_path):
#             raise ValueError(f"parquet_root_path '{parquet_root_path}' is not a valid directory.")
#         if not os.path.exists(metadata_json_path):
#             raise ValueError(f"metadata_json_path '{metadata_json_path}' not found.")
#
#         self.parquet_root_path = parquet_root_path
#         self.split = split
#         self.features_type = top_features.lower()
#         self.max_len = max_len
#         self.feature_padding_mode = feature_padding_mode
#         self.n_heads = n_heads
#
#         # --- 2. Load metadata and filter instances for the current split ---
#         with open(metadata_json_path, "r", encoding="utf-8") as f:
#             metadata_raw = json.load(f)
#
#         self.instances = []
#         for gloss_entry in metadata_raw:
#             for inst in gloss_entry["instances"]:
#                 if inst["split"] == self.split:
#                     self.instances.append({
#                         "video_id": str(inst["video_id"]),
#                         "gloss": gloss_entry["gloss"]
#                     })
#
#         glosses = sorted(set(entry["gloss"] for entry in self.instances))
#         self.gloss2idx = {gloss: idx for idx, gloss in enumerate(glosses)}
#         self.idx2gloss = {idx: gloss for gloss, idx in self.gloss2idx.items()}
#
#         for entry in self.instances:
#             entry["gloss_idx"] = self.gloss2idx[entry["gloss"]]
#
#         video_ids_in_split = {entry["video_id"] for entry in self.instances}
#
#         # --- 3. Load and Merge Required DataFrames ---
#         print(f"Loading data for '{self.split}' split...")
#
#         # Determine which files to load based on the requested top_features
#         files_to_load = {}
#         if "avasag_vitpose_extracted_landmarks" in self.features_type:
#             if "pose" in self.features_type: files_to_load['pose_landmarks'] = 'POSE_LANDMARKS.parquet'
#             if "hand" in self.features_type: files_to_load['hand_landmarks'] = 'HAND_LANDMARKS.parquet'
#             if "face" in self.features_type: files_to_load['face_landmarks'] = 'FACE_LANDMARKS.parquet'
#         if "angles" in self.features_type:
#             if "pose" in self.features_type: files_to_load['pose_angles'] = 'POSE_ANGLES.parquet'
#             if "hand" in self.features_type: files_to_load['hand_angles'] = 'HAND_ANGLES.parquet'
#
#         # Always include face avasag_vitpose_extracted_landmarks as a base if any landmark type is requested
#         if "avasag_vitpose_extracted_landmarks" in self.features_type:
#             files_to_load['face_landmarks'] = 'FACE_LANDMARKS.parquet'
#
#         if not files_to_load:
#             raise ValueError(f"Invalid 'top_features' argument: '{top_features}'. No data files to load.")
#
#         # Load and merge the dataframes
#         loaded_dfs = {}
#         for key, filename in files_to_load.items():
#             path = os.path.join(self.parquet_root_path, filename)
#             df = pd.read_parquet(path)
#             # Filter for videos only in the current split for memory efficiency
#             df = df[df["video_id"].astype(str).isin(video_ids_in_split)].copy()
#             loaded_dfs[key] = df
#
#         # Merge all loaded dataframes into a single one
#         df_keys = list(loaded_dfs.keys())
#         features_df = loaded_dfs[df_keys[0]]
#         for key in df_keys[1:]:
#             features_df = pd.merge(
#                 features_df, loaded_dfs[key],
#                 on=['video_id', 'frame', 'person_id'],
#                 how='inner',
#                 suffixes=(None, '_y')  # handle duplicate gloss column
#             ).filter(regex='^(?!.*_y$)')  # drop duplicate columns after merge
#
#         # --- 4. Determine the final list of feature columns to keep ---
#         cols_to_keep = []
#         if "pose_landmarks" in self.features_type:
#             cols_to_keep.extend([c for c in features_df.columns if c.startswith('p_')])
#         if "hand_landmarks" in self.features_type:
#             cols_to_keep.extend([c for c in features_df.columns if c.startswith('h_')])
#         if "face_landmarks" in self.features_type:
#             cols_to_keep.extend([c for c in features_df.columns if c.startswith('f_')])
#         if "pose_angles" in self.features_type:
#             # Assuming angle columns are named like 'angle_l_shoulder_...'
#             cols_to_keep.extend([c for c in features_df.columns if 'angle' in c and 'p' in c.lower()])
#         if "hand_angles" in self.features_type:
#             cols_to_keep.extend([c for c in features_df.columns if 'angle' in c and 'h' in c.lower()])
#
#         # --- 5. Apply feature dimension padding/truncation ---
#         if self.feature_padding_mode == "truncate":
#             k = len(cols_to_keep)
#             final_dim = (k // self.n_heads) * self.n_heads
#             self.final_columns = cols_to_keep[:final_dim]
#         else:  # 'sentinel' or 'repeat'
#             current_dim = len(cols_to_keep)
#             target_dim = ((current_dim + self.n_heads - 1) // self.n_heads) * self.n_heads
#             num_to_pad = target_dim - current_dim
#             if num_to_pad > 0:
#                 pad_cols = [f"pad_{i}" for i in range(num_to_pad)]
#                 if self.feature_padding_mode == 'sentinel':
#                     for pad_col in pad_cols:
#                         features_df[pad_col] = -2.0
#                 elif self.feature_padding_mode == 'repeat':
#                     pad_source_cols = [cols_to_keep[i % len(cols_to_keep)] for i in range(num_to_pad)]
#                     for new_name, source_name in zip(pad_cols, pad_source_cols):
#                         features_df[new_name] = features_df[source_name]
#                 else:
#                     raise ValueError(f"Unknown feature_padding_mode: '{self.feature_padding_mode}'")
#                 self.final_columns = cols_to_keep + pad_cols
#             else:
#                 self.final_columns = cols_to_keep
#
#         self.embedding_dim = len(self.final_columns)
#
#         # --- 6. Finalize the main DataFrame for use in __getitem__ ---
#         # Group by video_id for fast lookup in __getitem__
#         self.features_map = features_df[["video_id"] + self.final_columns].groupby("video_id")
#
#         # Determine max sequence length if not provided
#         self.video_lengths = self.features_map.size()
#         if self.max_len is None:
#             self.max_len = self.video_lengths.max()
#
#         print(f"Dataset for '{self.split}' split initialized.")
#         print(f"Number of instances: {len(self.instances)}")
#         print(f"Number of top_features (embedding dim): {self.embedding_dim}")
#         print(f"Max sequence length: {self.max_len}")
#
#     def __len__(self):
#         """Returns the number of instances in the dataset."""
#         return len(self.instances)
#
#     def __getitem__(self, idx):
#         """
#         Retrieves a single data sample.
#
#         Returns:
#             tuple: (feature_tensor, label_tensor)
#         """
#         entry = self.instances[idx]
#         video_id = entry["video_id"]
#         gloss_idx = entry["gloss_idx"]
#
#         # Efficiently get all rows for this video_id
#         group = self.features_map.get_group(video_id)
#
#         # Select only the final feature columns
#         features_df = group[self.final_columns]
#
#         # Convert to tensor
#         top_features = features_df.values.astype(np.float32)
#         feature_tensor = torch.tensor(top_features, dtype=torch.float32)
#
#         # Apply sequence padding or truncation
#         seq_len, _ = feature_tensor.shape
#         if seq_len < self.max_len:
#             pad = torch.full((self.max_len - seq_len, self.embedding_dim), fill_value=-2.0)
#             feature_tensor = torch.cat([feature_tensor, pad], dim=0)
#         elif seq_len > self.max_len:
#             feature_tensor = feature_tensor[:self.max_len, :]
#
#         label_tensor = torch.tensor(gloss_idx, dtype=torch.long)
#
#         return feature_tensor, label_tensor
#
#     @property
#     def feature_dim(self):
#         """Returns the dimensionality of the feature vectors."""
#         return self.embedding_dim