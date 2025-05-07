import pandas as pd
import numpy as np
from SLR_MediaPipe_DTW.models.hand_model import HandModel

def compute_hand_angles(df: pd.DataFrame, video_id: str, gloss: str) -> pd.DataFrame:
    """
    :param df: landmarks dataframe (can contatin single video frames or multiple video frames)
    :param video_id: video id
    :param gloss: gloss
    :return: dataframe with hand angles without header
    """
    lh_cols = [col for col in df.columns if col.startswith("LH#") and col.endswith(("_x", "_y", "_z"))]
    rh_cols = [col for col in df.columns if col.startswith("RH#") and col.endswith(("_x", "_y", "_z"))]

    vid_df = df[df["video_id"] == video_id]
    rows = []

    for _, row in vid_df.iterrows():
        try:
            lh = row[lh_cols].values.astype(float).reshape(-1, 3)
            rh = row[rh_cols].values.astype(float).reshape(-1, 3)
            lh_model = HandModel(lh)
            rh_model = HandModel(rh)
            row_data = [video_id, gloss] + lh_model.feature_vector + rh_model.feature_vector
            rows.append(row_data)
        except Exception as e:
            print(f"⚠️ Error in video {video_id}: {e}")
    return pd.DataFrame(rows)

def get_hand_header():
    dummy = HandModel(np.zeros((21, 3)))
    conns = list(dummy.connections)
    angle_labels = [f"Angle{{{a}-{b}}}" for i, a in enumerate(conns) for j, b in enumerate(conns) if i < j]
    return ["video_id", "gloss"] + [f"LH_{a}" for a in angle_labels] + [f"RH_{a}" for a in angle_labels]
