# mediapipe_utils.py
import gc
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.append('C:/Users/boulosge/Desktop/Acht')
from MediaPipe.HolisticProcessor import HolisticProcessor


def process_video_mediapipe(video_path: str, save_annotation: bool = False, annotated_videos_dir: str = None, gloss: str = "UNSPECIFIED"):
    """Processes a single video using MediaPipe and returns landmark dataframes and annotated video path, if there is one"""
    features_to_extract = ['pose', 'face', 'hand']
    processor = HolisticProcessor(extract=features_to_extract)
    try:
        pose_df, face_df, hand_df, annotated_video_path = processor.process_video(
            video_path, save_annotation=save_annotation, annotated_videos_dir=annotated_videos_dir, gloss=gloss
        )

        if pose_df.empty and face_df.empty and hand_df.empty:
            tqdm.write(f"[WARNING] MediaPipe extracted no landmarks for <{video_path}>")

        if save_annotation:
            if annotated_video_path:
                tqdm.write(f"[INFO] Saved annotated video path to <{annotated_video_path}>")
            else:
                tqdm.write(f'[WARNING] Mediapipe generated no annotation for video <{video_path}>')

        return {'pose': pose_df, 'face': face_df, 'hand': hand_df, 'annotated_video_path': annotated_video_path}

    except Exception as e:
        tqdm.write(f"[ERROR] MediaPipe failed on video <{video_path}>. Reason: {e}")
        return {'pose': pd.DataFrame(), 'face': pd.DataFrame(), 'hand': pd.DataFrame(), 'annotated_video_path': video_path}
    finally:
        if 'processor' in locals() and hasattr(processor, 'holistic'):
            processor.holistic.close()
        del processor
        gc.collect()


def save_aggregated_data(output_folder: str, all_data: dict):
    """Saves aggregated landmark data to Parquet files."""
    output_path = Path(output_folder)
    print(f"\nAll videos processed. Saving aggregated data...")

    if not all_data['pose'].empty:
        out_file = output_path / 'pose_landmarks.parquet'
        all_data['pose'].to_parquet(out_file, index=False)
        print(f"[INFO] Pose landmarks saved to: {out_file}")

    if not all_data['face'].empty:
        out_file = output_path / 'face_landmarks.parquet'
        all_data['face'].to_parquet(out_file, index=False)
        print(f"[INFO] Face landmarks saved to: {out_file}")

    if not all_data['hand'].empty:
        out_file = output_path / 'hand_landmarks.parquet'
        all_data['hand'].to_parquet(out_file, index=False)
        print(f"[INFO] Hand landmarks saved to: {out_file}")