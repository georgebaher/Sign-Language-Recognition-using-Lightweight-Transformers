# plot_from_parquet.py (Final Corrected Version with De-normalization)
#
# Reads NORMALIZED landmark data from Parquet files, de-normalizes them
# back to pixel coordinates, and generates annotated videos for verification.

import os
import glob
import cv2
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm

# plot_from_parquet.py (Final Version - Correct Label Format)
#
# Reads NORMALIZED landmark data from Parquet files, de-normalizes them,
# and generates annotated videos with connections and the specific label
# format you requested (P0, L0, R0).

import os
import glob
import cv2
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm

# --- Visualization Settings ---
POSE_COLOR = (255, 128, 0)
FACE_COLOR = (0, 255, 0)
HAND_COLOR = (0, 0, 255)
LINE_THICKNESS = 1
POINT_RADIUS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.3
FONT_COLOR = (255, 255, 255)

# --- Landmark Connection Maps ---
POSE_CONNECTIONS = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
                    [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
                    [1, 3], [2, 4], [3, 5], [4, 6]]
HAND_CONNECTIONS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
                    [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],
                    [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]


def plot_landmarks(frame, landmarks, color, width, height):
    """Helper function to de-normalize and draw landmark points."""
    for i in range(0, len(landmarks), 2):
        if landmarks[i] == -2: continue
        x = int(landmarks[i] * width)
        y = int(landmarks[i + 1] * height)
        cv2.circle(frame, (x, y), radius=POINT_RADIUS, color=color, thickness=-1)


def plot_connections(frame, landmarks, connections, color, width, height):
    """Helper function to de-normalize and draw lines connecting landmark points."""
    for conn in connections:
        p1_idx, p2_idx = conn
        if p1_idx * 2 + 1 < len(landmarks) and p2_idx * 2 + 1 < len(landmarks):
            if landmarks[p1_idx * 2] == -2 or landmarks[p2_idx * 2] == -2: continue
            x1 = int(landmarks[p1_idx * 2] * width)
            y1 = int(landmarks[p1_idx * 2 + 1] * height)
            x2 = int(landmarks[p2_idx * 2] * width)
            y2 = int(landmarks[p2_idx * 2 + 1] * height)
            cv2.line(frame, (x1, y1), (x2, y2), color, LINE_THICKNESS)


# --- MODIFIED: The plot_labels function is now updated ---
def plot_labels(frame, landmarks, prefix, width, height):
    """Helper function to de-normalize and draw text labels for landmarks."""
    for i in range(0, len(landmarks), 2):
        if landmarks[i] == -2: continue
        x = int(landmarks[i] * width)
        y = int(landmarks[i + 1] * height)

        # This creates the label format "P0", "L0", "R0", etc.
        label = f"{prefix}{i // 2}"

        # Position the text slightly offset from the point
        cv2.putText(frame, label, (x + 4, y - 4), FONT, FONT_SCALE, FONT_COLOR, 1)


def main():
    parser = ArgumentParser(description='Plot landmarks from Parquet files onto videos.')
    parser.add_argument('--parquet-folder', type=str, required=True,
                        help='Path to the folder containing the Parquet files.')
    parser.add_argument('--video-folder', type=str, required=True,
                        help='Path to the folder containing the original videos.')
    parser.add_argument('--output-folder', type=str, required=True,
                        help='Path to the folder where plotted videos will be saved.')
    args = parser.parse_args()

    print("Loading landmark data from Parquet files...")
    try:
        pose_df = pd.read_parquet(os.path.join(args.parquet_folder, 'POSE_LANDMARKS.parquet'))
        face_df = pd.read_parquet(os.path.join(args.parquet_folder, 'FACE_LANDMARKS.parquet'))
        hand_df = pd.read_parquet(os.path.join(args.parquet_folder, 'HAND_LANDMARKS.parquet'))
    except FileNotFoundError as e:
        print(f"Error: Could not find Parquet file. {e}")
        return

    pose_df.set_index(['video_id', 'frame'], inplace=True)
    face_df.set_index(['video_id', 'frame'], inplace=True)
    hand_df.set_index(['video_id', 'frame'], inplace=True)
    print("Landmark data loaded successfully.")

    os.makedirs(args.output_folder, exist_ok=True)
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        video_files.extend(glob.glob(os.path.join(args.video_folder, ext)))

    if not video_files:
        print(f"No videos found in {args.video_folder}. Please check the path.")
        return

    print(f"Found {len(video_files)} videos to plot.")

    for video_path in video_files:
        video_filename = os.path.basename(video_path)
        video_id = os.path.splitext(video_filename)[0]
        print(f"\nPlotting for: {video_filename}")

        if video_id not in pose_df.index.get_level_values('video_id'):
            print(f"  -> Warning: No landmark data found for {video_id}. Skipping.")
            continue

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_video_path = os.path.join(args.output_folder, f"plotted_landmarks_{video_filename}")
        video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = tqdm(total=total_frames, desc=f'Plotting {video_filename}')

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                frame_pose_data = pose_df.loc[(video_id, frame_idx)]
                frame_face_data = face_df.loc[(video_id, frame_idx)]
                frame_hand_data = hand_df.loc[(video_id, frame_idx)]

                if isinstance(frame_pose_data, pd.Series):
                    frame_pose_data = frame_pose_data.to_frame().T
                    frame_face_data = frame_face_data.to_frame().T
                    frame_hand_data = frame_hand_data.to_frame().T

                for i in range(len(frame_pose_data)):
                    if frame_pose_data.iloc[i]['person_id'] == -1:
                        continue

                    pose_landmarks = frame_pose_data.iloc[i].filter(regex=r'^p\d+_').values
                    face_landmarks = frame_face_data.iloc[i].filter(regex=r'^f\d+_').values
                    all_hand_landmarks = frame_hand_data.iloc[i].filter(regex=r'^h\d+_').values

                    left_hand_landmarks = all_hand_landmarks[:42]
                    right_hand_landmarks = all_hand_landmarks[42:]

                    # --- Draw everything with the correct labels ---
                    # The functions now modify the frame in-place
                    plot_landmarks(frame, pose_landmarks, POSE_COLOR, width, height)
                    plot_connections(frame, pose_landmarks, POSE_CONNECTIONS, POSE_COLOR, width, height)
                    plot_labels(frame, pose_landmarks, "P", width, height)

                    plot_landmarks(frame, face_landmarks, FACE_COLOR, width, height)
                    # No labels for face

                    plot_landmarks(frame, left_hand_landmarks, HAND_COLOR, width, height)
                    plot_connections(frame, left_hand_landmarks, HAND_CONNECTIONS, HAND_COLOR, width, height)
                    plot_labels(frame, left_hand_landmarks, "L", width, height)

                    plot_landmarks(frame, right_hand_landmarks, HAND_COLOR, width, height)
                    plot_connections(frame, right_hand_landmarks, HAND_CONNECTIONS, HAND_COLOR, width, height)
                    plot_labels(frame, right_hand_landmarks, "R", width, height)

            except KeyError:
                pass

            video_writer.write(frame)
            frame_idx += 1
            progress_bar.update(1)

        progress_bar.close()
        cap.release()
        video_writer.release()
        print(f"Finished plotting. Video saved to: {output_video_path}")


if __name__ == '__main__':
    main()

