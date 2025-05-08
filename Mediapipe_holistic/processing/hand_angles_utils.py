import pandas as pd
import numpy as np
from SLR_MediaPipe_DTW.models.hand_model import HandModel

def compute_hand_angles(df: pd.DataFrame, video_id: str, gloss: str) -> pd.DataFrame:
    """
       Compute hand angles from pre-labeled landmark DataFrame.

       :param df: DataFrame containing columns like 'LH#0_x', ..., 'RH#20_z'
       :param video_id: The ID of the video to process
       :param gloss: The associated gloss
       :return: DataFrame with angle vectors for each frame
       """
    # Assume LH#*_x, *_y, *_z and RH#*_x, *_y, *_z already exist
    lh_cols = [f"LH#{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")]
    rh_cols = [f"RH#{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")]

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

    # Create DataFrame with header
    return pd.DataFrame(rows, columns=get_hand_header())

def get_hand_header() -> list:
    dummy = HandModel(np.zeros((21, 3)))
    conns = list(dummy.connections)
    angle_labels = [f"Angle{{{a}-{b}}}" for i, a in enumerate(conns) for j, b in enumerate(conns) if i < j]
    return ["video_id", "gloss"] + [f"LH_{a}" for a in angle_labels] + [f"RH_{a}" for a in angle_labels]
