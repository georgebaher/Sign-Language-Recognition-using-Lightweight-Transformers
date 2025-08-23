# run_batch_processing.py (Final Version with all fixes)
#
# This script orchestrates the batch processing of a video folder.
# It uses the functions defined in pose_utils.py to perform the core tasks.

import os
import glob
import cv2
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm

# Import our custom utility functions
import pose_utils



def main():
    """
    Main function to run the batch processing pipeline.
    """
    parser = ArgumentParser(description='Run batch processing on a video folder and save landmarks to Parquet files.')

    # --- Essential Arguments ---
    parser.add_argument('--video-folder', type=str,# required=True,
                         default=r'C:\Users\boulosge\Desktop\Acht\ViTPose\ViTPose\test_videos_folder', help='Path to the folder containing input video(s).')
    parser.add_argument('--output-folder', type=str, #required=True,
                        default=r'C:\Users\boulosge\Desktop\Acht\ViTPose\ViTPose\test_videos_folder_annotated', help='Path to the folder where Parquet files and videos will be saved.')

    # --- THIS SECTION IS NEW: Added arguments for video saving ---
    parser.add_argument('--save-video', action='store_true', help='Set this flag to save annotated videos.')
    parser.add_argument('--kpt-thr', type=float, default=0.5, help='Keypoint score threshold for visualization.')

    # --- Model Configs (with defaults for convenience) ---
    parser.add_argument('--det-config', help='Config file for detection model.',
                        default='demo/mmdetection_cfg/yolox_l_8x8_300e_coco.py')
    parser.add_argument('--det-checkpoint', help='Checkpoint file for detection model.',
                        default='yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth')
    parser.add_argument('--pose-config', help='Config file for pose model.',
                        default='configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_large_wholebody_256x192.py')
    parser.add_argument('--pose-checkpoint', help='Checkpoint file for pose model.', default='wholebody.pth')

    # --- Optional Arguments ---
    parser.add_argument('--device', default='cuda:0', help='Device used for inference.')
    parser.add_argument('--bbox-thr', type=float, default=0.5, help='Bounding box score threshold.')
    parser.add_argument('--conf-thr', type=float, default=0.5, help='Keypoint confidence threshold for padding.')

    args = parser.parse_args()

    # --- 1. Load models once using the utility function ---
    det_model, pose_model, dataset_info = pose_utils.load_models(
        args.det_config, args.det_checkpoint, args.pose_config, args.pose_checkpoint, args.device
    )

    # --- 2. Setup Output Directories ---
    os.makedirs(args.output_folder, exist_ok=True)
    if args.save_video:
        annotated_video_folder = os.path.join(args.output_folder, 'annotated_videos')
        os.makedirs(annotated_video_folder, exist_ok=True)

    # --- Data storage lists for aggregating results from all videos ---
    all_pose_data, all_face_data, all_hand_data = [], [], []

    # --- 3. Find and loop through videos ---
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(args.video_folder, ext)))

    if not video_files:
        print(f"No videos found in {args.video_folder}. Please check the path.")
        return

    print(f"Found {len(video_files)} videos to process.")

    for video_path in video_files:
        video_filename = os.path.basename(video_path)
        video_id = os.path.splitext(video_filename)[0]
        print(f"\nProcessing: {video_filename}")

        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f'Failed to load video file {video_path}'

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = tqdm(total=total_frames, desc=f'Analyzing {video_filename}')

        # --- THIS SECTION IS NEW: Initialize video_writer for each video ---
        video_writer = None
        if args.save_video:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_video_path = os.path.join(annotated_video_folder, video_filename)
            video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # --- THIS IS THE NEW PART ---
            # Get frame dimensions for normalization
            frame_height, frame_width, _ = frame.shape

            # --- Get raw landmark data from the models ---
            pose_results = pose_utils.process_single_frame(
                det_model, pose_model, dataset_info, frame,
                args.bbox_thr
            )
            # print(pose_results)

            # --- Convert raw data into structured, NORMALIZED rows ---
            # --- Pass the new arguments here ---
            pose_rows, face_rows, hand_rows = pose_utils.get_pose_data(
                pose_results=pose_results, video_id=video_id, frame_idx=frame_idx,
                frame_width=frame_width, frame_height=frame_height,  # Pass dimensions
                conf_thr=args.conf_thr
            )

            all_pose_data.extend(pose_rows)
            all_face_data.extend(face_rows)
            all_hand_data.extend(hand_rows)

            # --- 5. Draw landmarks on the frame if requested ---
            if args.save_video:
                vis_frame = pose_utils.draw_landmarks_on_frame(
                    frame, pose_results, args.kpt_thr
                )
                if video_writer is not None:
                    video_writer.write(vis_frame)

            frame_idx += 1
            progress_bar.update(1)

        progress_bar.close()
        cap.release()
        # --- THIS SECTION IS NEW: Release the writer for the finished video ---
        if video_writer:
            video_writer.release()
        if args.save_video:
            print(f"  -> Annotated video saved to: {output_video_path}")

    # --- 6. Save aggregated data to Parquet files ---
    print("\nAll videos processed. Converting data to Parquet files...")

    if all_pose_data:
        pose_df = pd.DataFrame(all_pose_data).fillna(-2)
        pose_df.to_parquet(os.path.join(args.output_folder, 'POSE_LANDMARKS.parquet'), index=False)
        print(f"Pose landmarks saved to: {os.path.join(args.output_folder, 'POSE_LANDMARKS.parquet')}")
    if all_face_data:
        face_df = pd.DataFrame(all_face_data).fillna(-2)
        face_df.to_parquet(os.path.join(args.output_folder, 'FACE_LANDMARKS.parquet'), index=False)
        print(f"Face landmarks saved to: {os.path.join(args.output_folder, 'FACE_LANDMARKS.parquet')}")
    if all_hand_data:
        hand_df = pd.DataFrame(all_hand_data).fillna(-2)
        hand_df.to_parquet(os.path.join(args.output_folder, 'HAND_LANDMARKS.parquet'), index=False)
        print(f"Hand landmarks saved to: {os.path.join(args.output_folder, 'HAND_LANDMARKS.parquet')}")


if __name__ == '__main__':
    main()