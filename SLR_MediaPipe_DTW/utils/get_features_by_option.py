import pandas as pd
import os
from dotenv import load_dotenv

def get_features_by_option(video_id: str,
                           hand_pose_landmarks_df: pd.DataFrame,
                           hand_angles_df: pd.DataFrame,
                           pose_angles_df: pd.DataFrame,
                           option: str) -> pd.DataFrame:
    """
    Return the appropriate feature DataFrame based on video_id and feature option.

    :param video_id: str, the video ID to filter on
    :param hand_pose_landmarks_df: DataFrame of hand + pose landmarks
    :param hand_angles_df: DataFrame of hand angles
    :param pose_angles_df: DataFrame of pose angles
    :param option: One of ["hand_landmarks", "hand+pose_landmarks", "hand_angles", "hand+pose_angles"]
    :return: pd.DataFrame filtered and merged accordingly
    """

    if option not in ["hand_landmarks", "hand+pose_landmarks", "hand_angles", "hand+pose_angles"]:
        raise ValueError(f"❌ Invalid option: {option}")

    # Filter each DataFrame by video_id
    hand_pose_landmarks = hand_pose_landmarks_df[hand_pose_landmarks_df["video_id"] == video_id]
    hand_angles = hand_angles_df[hand_angles_df["video_id"] == video_id]

    # Drop wlasl_csv_exports columns (video_id, gloss)
    def drop_meta(df):
        return df.drop(columns=["video_id", "gloss"], errors="ignore")

    # Option handling
    if option == "hand_landmarks":
        # Keep only hand columns (e.g., LH# and RH#)
        hand_cols = [col for col in hand_pose_landmarks.columns if col.startswith("LH#") or col.startswith("RH#")]
        return drop_meta(hand_pose_landmarks[["video_id"] + hand_cols])

    elif option == "hand+pose_landmarks":
        return drop_meta(hand_pose_landmarks)

    elif option == "hand_angles":
        return drop_meta(hand_angles)

    elif option == "hand+pose_angles":
        hand = hand_angles_df[hand_angles_df["video_id"] == video_id].reset_index(drop=True)
        pose = pose_angles_df[pose_angles_df["video_id"] == video_id].reset_index(drop=True)

        # Drop wlasl_csv_exports if needed
        hand = hand.drop(columns=["video_id", "gloss"], errors="ignore")
        pose = pose.drop(columns=["video_id", "gloss"], errors="ignore")

        merged = pd.concat([hand, pose], axis=1)

        return drop_meta(merged)

    return pd.DataFrame()  # fallback (should never hit)

if __name__ == "__main__":
    # TESTING ...
    load_dotenv()
    landmarks_df = pd.read_parquet(os.getenv("WLASL100_LANDMARKS_PATH"))
    hand_angles_df=pd.read_parquet(os.getenv("WLASL100_HAND_ANGLES_PATH"))
    pose_angles_df =pd.read_parquet(os.getenv("WLASL100_POSE_ANGLES_PATH"))
    vid_id="69241"
    option="hand+pose_angles"
    result_df=get_features_by_option(vid_id, landmarks_df, hand_angles_df, pose_angles_df,option=option)
    print(result_df.values.tolist())