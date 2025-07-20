import os
import gc
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from WLASL_feature_processing_mediapipe.HolisticProcessor import HolisticProcessor

load_dotenv()


def compute_video_landmarks(video_path: str, gloss: str, show_landmarks: bool = False):
    """
    Process a single video and return its extracted landmarks DataFrames with video_id and gloss columns.

    :param video_path: full path to the .mp4 video file
    :param gloss: gloss label for the video
    :param show_landmarks: whether to display extracted_landmarks
    :return: 3 DataFrames of extracted pose, hand and face landmarks plus annotated video path
    """

    processor = HolisticProcessor(extract=["pose", "hand", "face"])

    try:
        pose_df, face_df, hand_df, video_output_path = processor.process_video(video_path, save_annotation=show_landmarks, gloss=gloss)
        if pose_df.empty and face_df.empty and hand_df.empty:
            print(f"No landmarks extracted at all for <{video_path}>")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ""

        # return df
        return pose_df, face_df, hand_df, video_output_path

    except Exception as e:
        print(f"Error processing video <{video_path}>: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ""
    finally:
        processor.holistic.close()
        del processor
        gc.collect()


if __name__ == "__main__":
    # testing landmarks extraction
    test_video_path = os.getenv('WLASL_TEST_VIDEO_PATH')
    pose_df, face_df, hand_df, vis_path = compute_video_landmarks(test_video_path, os.getenv("WLASL_TEST_VIDEO_GLOSS"), True)
    if vis_path and os.path.exists(vis_path):
        print(f"annotated video saved at: {vis_path}")
        os.system(f'start {vis_path}')
    else:
        print("annotated video not found.")
