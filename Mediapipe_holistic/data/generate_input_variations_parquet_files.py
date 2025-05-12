import pandas as pd

# Load base landmark + angle files
df_landmarks = pd.read_parquet('parquet/wlasl100_hand_pose_landmarks.parquet')
df_hand_angles = pd.read_parquet('parquet/wlasl100_hand_angles.parquet')
df_pose_angles = pd.read_parquet('parquet/wlasl100_pose_angles.parquet')

# Ensure same number of rows
# assert len(df_landmarks) == len(df_hand_angles) == len(df_pose_angles), "Row count mismatch"

# --- Identify columns ---
meta_cols = ['video_id', 'gloss']
hand_cols = [col for col in df_landmarks.columns if col.startswith('LH#') or col.startswith('RH#')]
pose_cols = [col for col in df_landmarks.columns if col.startswith('P#')]

# # --- 1. Hand landmarks only ---
# df_hand_landmarks = df_landmarks[meta_cols + hand_cols]
# df_hand_landmarks.to_parquet('parquet/wlasl100_hand_landmarks.parquet')

# # --- 2. Hand + Pose landmarks ---
# Already there

# # --- 3. Hand angles only ---
# Already there

# # --- 4. Hand + Pose angles ---
# df_hand_pose_angles = pd.concat(
#     [df_landmarks[meta_cols],
#      df_hand_angles.drop(columns=meta_cols, errors='ignore'),
#      df_pose_angles.drop(columns=meta_cols, errors='ignore')],
#     axis=1
# )
#
# df_hand_pose_angles.to_parquet('parquet/wlasl100_hand_pose_angles.parquet')
