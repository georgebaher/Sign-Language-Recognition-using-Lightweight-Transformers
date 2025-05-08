import pandas as pd
import numpy as np
from SLR_MediaPipe_DTW.models.pose_model import PoseModel

def compute_pose_angles(df: pd.DataFrame, video_id: str, gloss: str) -> pd.DataFrame:
    """
    Compute pose angles from pre-labeled landmark DataFrame.

    :param df: DataFrame containing columns like 'P#0_x', ..., 'P#32_z'
    :param video_id: The ID of the video to process
    :param gloss: The associated gloss
    :return: DataFrame with angle vectors for each frame (with header)
    """
    pose_cols = [f"P#{i}_{axis}" for i in range(33) for axis in ("x", "y", "z")]
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

    # Return DataFrame with proper column headers
    return pd.DataFrame(rows, columns=get_pose_header())

def get_pose_header() -> list:
    dummy = PoseModel(np.zeros((33, 3)))
    conns = list(dummy.connections)
    angle_labels = [f"Angle{{{a}-{b}}}" for i, a in enumerate(conns) for j, b in enumerate(conns) if i < j]
    return ["video_id", "gloss"] + [f"P_{a}" for a in angle_labels]
