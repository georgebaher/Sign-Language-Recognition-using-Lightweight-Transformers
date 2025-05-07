import pandas as pd
import numpy as np
from SLR_MediaPipe_DTW.models.pose_model import PoseModel

def compute_pose_angles(df: pd.DataFrame, video_id: str, gloss: str) -> pd.DataFrame:
    """
    :param df: landmarks dataframe (can contatin single video frames or multiple video frames)
    :param video_id: video id
    :param gloss: gloss
    :return: dataframe with hand angles without header
    """
    pose_cols = [col for col in df.columns if col.startswith("P#") and col.endswith(("_x", "_y", "_z"))]
    vid_df = df[df["video_id"] == video_id]
    rows = []

    for _, row in vid_df.iterrows():
        try:
            pose = row[pose_cols].values.astype(float).reshape(-1, 3)
            pose_model = PoseModel(pose)
            row_data = [video_id, gloss] + pose_model.feature_vector
            rows.append(row_data)
        except Exception as e:
            print(f"⚠️ Error in video {video_id}: {e}")
    return pd.DataFrame(rows)

def get_pose_header():
    dummy = PoseModel(np.zeros((33, 3)))
    conns = list(dummy.connections)
    angle_labels = [f"Angle{{{a}-{b}}}" for i, a in enumerate(conns) for j, b in enumerate(conns) if i < j]
    return ["video_id", "gloss"] + [f"P_{a}" for a in angle_labels]
