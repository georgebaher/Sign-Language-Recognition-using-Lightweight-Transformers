import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

# Load base landmark + angle files
df_landmarks = pd.read_parquet(os.getenv('WLASL100_HAND_POSE_LANDMARKS_PATH'))
df_hand_angles = pd.read_parquet(os.getenv('WLASL100_HAND_ANGLES_PATH'))
df_pose_angles = pd.read_parquet(os.getenv('WLASL100_POSE_ANGLES_PATH'))
df_hand_angles_summary = pd.read_parquet(os.getenv('WLASL100_HAND_ANGLES_SUMMARY_PATH'))
df_pose_angles_summary = pd.read_parquet(os.getenv('WLASL100_POSE_ANGLES_SUMMARY_PATH'))
df_hand_landmarks_summary = pd.read_parquet(os.getenv('WLASL100_HAND_LANDMARKS_SUMMARY_PATH'))
df_pose_landmarks_summary = pd.read_parquet(os.getenv('WLASL100_POSE_LANDMARKS_SUMMARY_PATH'))
#

# Ensure same number of rows
assert len(df_landmarks) == len(df_hand_angles) == len(df_pose_angles), "Row count mismatch"

# --- Identify columns ---
meta_cols = ['video_id', 'gloss']
hand_cols = [col for col in df_landmarks.columns if col.startswith('LH#') or col.startswith('RH#')]
pose_cols = [col for col in df_landmarks.columns if col.startswith('P#')]

# # --- 1. Hand landmarks only ---
# df_hand_landmarks = df_landmarks[meta_cols + hand_cols]
# df_hand_landmarks.to_parquet(os.getenv('WLASL100_HAND_LANDMARKS_PATH'))
# print(df_hand_landmarks.head())

# # --- 2. Pose landmarks only ---
# df_pose_landmarks = df_landmarks[meta_cols + pose_cols]
# df_pose_landmarks.to_parquet(os.getenv('WLASL100_POSE_LANDMARKS_PATH'))
# print(df_pose_landmarks.head())

# # --- 3. Hand + Pose landmarks ---
# Already there

# # --- 4. Hand angles only ---
# Already there

# # --- 5. Pose angles only ---
# Already there

# # --- 6. Pose + Hand angles ---
# df_hand_pose_angles = pd.concat(
#     [df_landmarks[meta_cols],
#      df_pose_angles.drop(columns=meta_cols, errors='ignore'),
#      df_hand_angles.drop(columns=meta_cols, errors='ignore'),
#      ],
#     axis=1
# )
# df_hand_pose_angles.to_parquet(os.getenv('WLASL100_HAND_POSE_ANGLES_PATH'))
# print(df_hand_pose_angles.head())

# # --- 7. Pose + Hand angles summary ---
# df_hand_pose_angles_summary = pd.concat(
#     [df_hand_angles_summary[meta_cols],
#      df_pose_angles_summary.drop(columns=meta_cols, errors='ignore'),
#      df_hand_angles_summary.drop(columns=meta_cols, errors='ignore'),
#      ],
#     axis=1
# )
# df_hand_pose_angles_summary.to_parquet(os.getenv('WLASL100_HAND_POSE_ANGLES_SUMMARY_PATH'))
# print(df_hand_pose_angles_summary.head())

# # --- 8. Pose + Hand landmarks summary ---
# df_hand_pose_landmarks_summary = pd.concat(
#     [df_hand_landmarks_summary[meta_cols],
#      df_pose_landmarks_summary.drop(columns=meta_cols, errors='ignore'),
#      df_hand_landmarks_summary.drop(columns=meta_cols, errors='ignore'),
#      ],
#     axis=1
# )
# df_hand_pose_landmarks_summary.to_parquet(os.getenv('WLASL100_HAND_POSE_LANDMARKS_SUMMARY_PATH'))
# print(df_hand_pose_landmarks_summary.head())
