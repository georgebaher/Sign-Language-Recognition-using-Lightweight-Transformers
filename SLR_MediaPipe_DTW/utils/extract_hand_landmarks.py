import os
from typing import List, Tuple
import pandas as pd
from Mediapipe_holistic.HolisticProcessor import HolisticProcessor

def extract_each_hand_landmarks_from_full_dataframe(df: pd.DataFrame, video_id: str) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Extracts left and right hand landmark sequences for a specific video from the given dataframe.

    Args:
        df (pd.DataFrame): The full dataframe containing all frames from all videos.
        video_id (str): The video ID to filter and extract data for.

    Returns:
        Tuple[List[List[float]], List[List[float]]]:
            - Left hand landmarks list: one [x,y,z]*21 vector per frame
            - Right hand landmarks list: one [x,y,z]*21 vector per frame
    """
    # Filter dataframe to the selected video
    video_df = df[df["video_id"] == video_id].reset_index(drop=True)

    # Initialize result lists
    left_hand_list = []
    right_hand_list = []

    for _, row in video_df.iterrows():
        left = []
        right = []

        # Extract left hand (LH#i_x/y/z)
        for i in range(21):
            left.extend([
                row[f"LH#{i}_x"],
                row[f"LH#{i}_y"],
                row[f"LH#{i}_z"]
            ])

        # Extract right hand (RH#i_x/y/z)
        for i in range(21):
            right.extend([
                row[f"RH#{i}_x"],
                row[f"RH#{i}_y"],
                row[f"RH#{i}_z"]
            ])

        left_hand_list.append(left)
        right_hand_list.append(right)

    return left_hand_list, right_hand_list

def extract_hands_landmarks(video_path):
    holistic_processor = HolisticProcessor(["pose","hand"])
    dataframe = holistic_processor.process_video(video_path, True)
    left_hand_list, right_hand_list = extract_each_hand_landmarks_from_full_dataframe(dataframe, os.path.basename(video_path).split(".")[0])
    return left_hand_list, right_hand_list


