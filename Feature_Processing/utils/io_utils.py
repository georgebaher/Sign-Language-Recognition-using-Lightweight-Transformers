import os
import pandas as pd


def load_existing_ids(parquet_path):
    """
    :param parquet_path: path to old results file containing landmarks
    :return: set of existing ids and all video frames landmarks as dataframe
    """
    if os.path.exists(parquet_path) and os.path.getsize(parquet_path) > 0:
        df = pd.read_parquet(parquet_path)
        return set(df["video_id"]), df
    return set(), None


def save_and_merge(new_df: pd.DataFrame, path: str, existing_df: pd.DataFrame = None):
    """
    :param new_df: pandas dataframe containing landmarks/angles
    :param path: path to save the new dataframe
    :param existing_df: pandas dataframe containing previously existing landmarks/angles
    :return:
    """
    if existing_df is not None:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(path, index=False)
    print(f"✅ Saved to {path}")
