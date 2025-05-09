import pandas as pd

# display hand landmarks
df = pd.read_parquet('parquet/wlasl100_hand_landmarks.parquet')
print(df)

# display hand + pose landmarks
df = pd.read_parquet('parquet/wlasl100_hand_pose_landmarks.parquet')
print(df)

# display hand angles
df = pd.read_parquet('parquet/wlasl100_hand_angles.parquet')
print(df)

# display hand + pose angles
df = pd.read_parquet('parquet/wlasl100_hand_pose_angles.parquet')
print(df)