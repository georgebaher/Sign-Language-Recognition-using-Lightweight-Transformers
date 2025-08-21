# process_from_metadata_mediapipe.py
import os
import glob
import json
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
import time
from pathlib import Path
import mediapipe_utils


def main():
    parser = ArgumentParser(description='Run MediaPipe batch processing using a metadata file.')
    parser.add_argument('--metadata-file', type=str, required=True, help='Path to the metadata JSON file.')
    parser.add_argument('--n-glosses', type=int, default=100, help='Number of glosses to process.')
    parser.add_argument('--video-folder', type=str, required=True,
                        help='Path to the root folder containing all input videos.')
    parser.add_argument('--output-folder', type=str, required=True,
                        help='Path to the folder where Parquet files will be saved.')
    args = parser.parse_args()
    start_time = time.time()

    with open(args.metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    glosses_to_process = metadata[:args.n_glosses]

    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    all_data = {'pose': [], 'face': [], 'hand': []}

    processed_video_ids = set()
    primary_parquet_path = output_path / 'pose_landmarks.parquet'
    if primary_parquet_path.exists():
        print(f"Resuming from {primary_parquet_path}...")
        try:
            processed_video_ids = set(pd.read_parquet(primary_parquet_path)['video_id'].unique())
        except Exception as e:
            print(f"Could not read existing Parquet file. Error: {e}")

    videos_to_process = []
    for entry in glosses_to_process:
        for instance in entry['instances']:
            if instance['video_id'] not in processed_video_ids:
                videos_to_process.append({'gloss': entry['gloss'], 'video_id': instance['video_id']})

    print(f"Processing {len(videos_to_process)} new videos.")
    missing_videos = 0

    for item in tqdm(videos_to_process, desc="Processing Videos", unit="video"):
        gloss, video_id = item['gloss'], item['video_id']
        video_paths = glob.glob(os.path.join(args.video_folder, f"**/{video_id}.*"), recursive=True)

        if not video_paths:
            tqdm.write(f"[Warning] Video ID '{video_id}' not found.")
            missing_videos += 1
            continue

        # The processing is handled by the utility function
        dfs = mediapipe_utils.process_video_mediapipe(video_paths[0], gloss=gloss)
        if not dfs['pose'].empty: all_data['pose'].append(dfs['pose'])
        if not dfs['face'].empty: all_data['face'].append(dfs['face'])
        if not dfs['hand'].empty: all_data['hand'].append(dfs['hand'])

    if videos_to_process:
        final_dfs = {key: pd.concat(value, ignore_index=True) if value else pd.DataFrame() for key, value in
                     all_data.items()}
        if processed_video_ids:
            print("Appending new data to existing Parquet files...")
            for landmark_type in ['pose', 'face', 'hand']:
                parquet_file = output_path / f'{landmark_type.upper()}_landmarks.parquet'
                if parquet_file.exists():
                    existing_df = pd.read_parquet(parquet_file)
                    final_dfs[landmark_type] = pd.concat([existing_df, final_dfs[landmark_type]], ignore_index=True)

        mediapipe_utils.save_aggregated_data(str(output_path), final_dfs)

    end_time = time.time()
    print("\n--- Processing Summary ---")
    print(f"New Videos Processed: {len(videos_to_process) - missing_videos}")
    print(f"Videos Skipped (existing): {len(processed_video_ids)}")
    print(f"Videos Not Found: {missing_videos}")
    print(f"Total Runtime: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    main()