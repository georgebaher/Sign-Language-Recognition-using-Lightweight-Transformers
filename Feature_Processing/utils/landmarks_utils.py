import os
import gc
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from Feature_Processing.HolisticProcessor import HolisticProcessor

load_dotenv()


def compute_video_landmarks(video_path: str, gloss: str, show_landmarks: bool = False):
    """
    Process a single video and return its landmarks DataFrame with video_id and gloss columns.

    :param video_path: full path to the .mp4 video file
    :param gloss: gloss label for the video
    :param show_landmarks: whether to display landmarks
    :return: DataFrame of landmarks, or empty DataFrame if failed
    """

    processor = HolisticProcessor(extract=["pose", "hand"])

    try:
        df, video_output_path = processor.process_video(video_path, show_landmarks=show_landmarks, gloss=gloss)
        if df.empty:
            print(f"❌ No frames extracted for <{video_path}>")
            return pd.DataFrame()

        # return df
        return df, video_output_path

    except Exception as e:
        print(f"❌ Error processing video <{video_path}>: {e}")
        return pd.DataFrame()
    finally:
        processor.holistic.close()
        del processor
        gc.collect()


def summarize_single_video_landmarks(video_landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes one video's frame-wise landmarks data into a single-row DataFrame, ignoring -2 (missing) values.

    :param video_landmarks_df: Frame-wise landmarks data for a single video
    :return: DataFrame with one row of video-level statistical features
    """
    video_id = video_landmarks_df['video_id'].iloc[0]
    gloss = video_landmarks_df['gloss'].iloc[0]

    feature_cols = [col for col in video_landmarks_df.columns if col not in ['video_id', 'gloss'] and not col.endswith('z')]    # z dropped
    summary_data = {
        'video_id': video_id,
        'gloss': gloss
    }

    for col in feature_cols:
        values = video_landmarks_df[col].astype(float).values
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
    # TESTING ...
    test_video_path = os.getenv('TEST_VIDEO_PATH')
    # Test landmarks extraction
    landmarks_df, vis_path = compute_video_landmarks(test_video_path, "drink", True)
    print("✅ Extracted landmarks. Shape of dataframe:", landmarks_df.shape)
    print(landmarks_df.head())
    # Test returned visualization path
    if vis_path and os.path.exists(vis_path):
        print(f"✅ Landmarked video saved at: {vis_path}")
        # Optional: open with default video player (Windows)
        os.system(f'start {vis_path}')
    else:
        print("❌ Landmark visualization video not found.")

    # Test summarizing landmarks across frames using stat metrics
    vid_landmarks_summary = summarize_single_video_landmarks(landmarks_df)
    print(vid_landmarks_summary)
    print(f"Number of padded metrics with -2: {list(vid_landmarks_summary.iloc[0].values).count(-2)}")
