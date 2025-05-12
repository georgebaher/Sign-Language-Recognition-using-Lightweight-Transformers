import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

from SLR_MediaPipe_DTW.models.hand_model import HandModel

def compute_hand_angles(df: pd.DataFrame, video_id: str, gloss: str) -> pd.DataFrame:
    """
       Compute hand angles from pre-labeled landmark DataFrame.

       :param df: DataFrame containing all videos' landmarks with columns like 'P#0_x, ..., LH#0_x', ..., 'RH#20_z'
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
    dummy = HandModel([i for i in range(63)])
    conns = list(dummy.connections)
    angle_labels = [f"Angle{{{a}-{b}}}" for i, a in enumerate(conns) for j, b in enumerate(conns) if i < j]
    return ["video_id", "gloss"] + [f"LH_{a}" for a in angle_labels] + [f"RH_{a}" for a in angle_labels]


def summarize_single_video_hand_angles(video_hand_angles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes one video's frame-wise angle data into a single-row DataFrame, ignoring -2 (missing) values.

    :param video_hand_angles_df: Frame-wise angle data for a single video
    :return: DataFrame with one row of video-level statistical features
    """
    video_id = video_hand_angles_df['video_id'].iloc[0]
    gloss = video_hand_angles_df['gloss'].iloc[0]

    feature_cols = [col for col in video_hand_angles_df.columns if col not in ['video_id', 'gloss']]
    summary_data = {
        'video_id': video_id,
        'gloss': gloss
    }

    for col in feature_cols:
        values = video_hand_angles_df[col].astype(float).values
        valid_values = values[values != -2]

        if len(valid_values) == 0:
            # All values were missing, pad with NaN
            summary_data[f"{col}_mean"] = -2
            summary_data[f"{col}_std"] = -2
            summary_data[f"{col}_min"] = -2
            summary_data[f"{col}_max"] = -2
            summary_data[f"{col}_var"] = -2
        else:
            summary_data[f"{col}_mean"] = np.mean(valid_values)
            summary_data[f"{col}_std"] = np.std(valid_values)
            summary_data[f"{col}_min"] = np.min(valid_values)
            summary_data[f"{col}_max"] = np.max(valid_values)
            summary_data[f"{col}_var"] = np.var(valid_values)

    return pd.DataFrame([summary_data])


if __name__ == "__main__":
    landmarks_df=pd.read_parquet(os.getenv("WLASL100_LANDMARKS_PATH"))
    test_video_id="69302"
    vid_hand_angles_df=compute_hand_angles(landmarks_df, test_video_id, "drink")
    print(vid_hand_angles_df)
    vid_hand_angles_summary = summarize_single_video_hand_angles(vid_hand_angles_df)
    print(vid_hand_angles_summary)
    print(f"Number of padded metrics with -2: {list(vid_hand_angles_summary.iloc[0].values).count(-2)}")
