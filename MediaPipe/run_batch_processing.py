# run_batch_processing.py
import os
import glob
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
import mediapipe_utils


def main():
    parser = ArgumentParser(description='Run MediaPipe batch processing on a video folder.')
    parser.add_argument('--video-folder', type=str, required=True, help='Path to the folder containing input videos.')
    parser.add_argument('--output-folder', type=str, required=True,
                        help='Path to the folder where Parquet files will be saved.')
    parser.add_argument('--save-video', action='store_true', help='Set this flag to save annotated videos.')
    args = parser.parse_args()

    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    annotated_videos_folder = ""
    if args.save_video:
        annotated_videos_folder = os.path.join(args.output_folder, 'annotated_videos')
        os.makedirs(annotated_videos_folder, exist_ok=True)

    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        video_files.extend(glob.glob(os.path.join(args.video_folder, f"**/{ext}"), recursive=True))

    if not video_files:
        print(f"No videos found in {args.video_folder}. Please check the path.")
        return
    print(f"Found {len(video_files)} videos to process.")

    all_data = {'pose': [], 'face': [], 'hand': []}
    for video_path in tqdm(video_files, desc="Processing Videos", unit="video"):
        video_id = Path(video_path).stem

        # The processing is handled by the utility function
        dfs = mediapipe_utils.process_video_mediapipe(video_path, save_annotation=args.save_video, annotated_videos_dir=annotated_videos_folder, gloss="UNSPECIFIED")
        if not dfs['pose'].empty: all_data['pose'].append(dfs['pose'])
        if not dfs['face'].empty: all_data['face'].append(dfs['face'])
        if not dfs['hand'].empty: all_data['hand'].append(dfs['hand'])

    # Concatenate all collected dataframes and save
    final_dfs = {key: pd.concat(value, ignore_index=True) if value else pd.DataFrame() for key, value in
                 all_data.items()}
    mediapipe_utils.save_aggregated_data(args.output_folder, final_dfs)


if __name__ == '__main__':
    main()