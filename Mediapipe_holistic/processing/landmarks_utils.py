import pandas as pd
from Mediapipe_holistic.HolisticProcessor import HolisticProcessor

# Initialize once to avoid repeating it
processor = HolisticProcessor(extract=["pose", "hand"])


def compute_video_landmarks(video_path: str, gloss: str, video_id: str) -> pd.DataFrame:
    """
    Process a single video and return its landmarks DataFrame with video_id and gloss columns.

    :param video_path: full path to the .mp4 video file
    :param gloss: gloss label for the video
    :param video_id: string or int identifier
    :return: DataFrame of landmarks, or empty DataFrame if failed
    """
    try:
        df = processor.process_video(video_path, show_landmarks=False, gloss=gloss)
        if df.empty:
            print(f"❌ No frames extracted for {video_id}")
            return pd.DataFrame()

        df["video_id"] = video_id
        df["gloss"] = gloss
        return df

    except Exception as e:
        print(f"❌ Error processing video {video_id}: {e}")
        return pd.DataFrame()
