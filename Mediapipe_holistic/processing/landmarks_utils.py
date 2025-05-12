import os

import pandas as pd
from Mediapipe_holistic.HolisticProcessor import HolisticProcessor

# Initialize once to avoid repeating it
processor = HolisticProcessor(extract=["pose", "hand"])


def compute_video_landmarks(video_path: str, gloss: str, show_landmarks: bool = False) -> pd.DataFrame:
    """
    Process a single video and return its landmarks DataFrame with video_id and gloss columns.

    :param video_path: full path to the .mp4 video file
    :param gloss: gloss label for the video
    :param show_landmarks: whether to display landmarks
    :return: DataFrame of landmarks, or empty DataFrame if failed
    """
    try:
        df = processor.process_video(video_path, show_landmarks=show_landmarks, gloss=gloss)
        if df.empty:
            print(f"❌ No frames extracted for <{video_path}>")
            return pd.DataFrame()

        return df

    except Exception as e:
        print(f"❌ Error processing video <{video_path}>: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # TESTING ...
    res_df = compute_video_landmarks(os.getenv('TEST_VIDEO_PATH'), "drink", True)
    print(res_df)
