import torch
from torch.utils.data import DataLoader
from features_dataloader import WLASLParquetDataset  # <- adjust if needed
import cv2
import numpy as np
import os
from dotenv import load_dotenv


# ------------------ PLOT FUNCTION ------------------
def plot_landmarks(feature_tensor: torch.Tensor, save_path="tmp/output.mp4", width=512, height=512, fps=25):
    """
    Render landmarks on a white background and save as video.
    Skips padded frames (-2) and padded landmarks (-2).
    """

    POSE_CONNECTIONS = [
        (0, 2), (1, 2), (2, 3), (2, 7),
        (0, 5), (4, 5), (5, 6), (5, 8),
        (9, 10),
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 12), (12, 24), (23, 24), (11, 23),
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
    ]

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
    ]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    ###### Prepare landmarks ######
    seq_len, _ = feature_tensor.shape

    n_axis = 2  # x and y only

    features_np = feature_tensor.cpu().numpy().reshape(seq_len, -1, n_axis)
    # print(features_np.shape)

    def draw_subset(frame, coords, connections, prefix="", color_points=(0, 0, 255), color_lines=(0, 255, 0),
                    plot_labels=False):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        thickness = 1

        for i, pt in enumerate(coords):
            if np.allclose(pt, -2):
                continue
            x, y = pt[0], pt[1]
            cx, cy = int(x * width), int(y * height)
            cv2.circle(frame, (cx, cy), 2, color_points, -1)
            label = f"{prefix}#{i}" if plot_labels else ''
            cv2.putText(frame, label, (cx + 4, cy - 4), font, font_scale, (255, 0, 255), thickness, cv2.LINE_AA)

        for start_idx, end_idx in connections:
            if start_idx >= coords.shape[0] or end_idx >= coords.shape[0]:
                continue
            pt1 = coords[start_idx]
            pt2 = coords[end_idx]
            if np.allclose(pt1, -2) or np.allclose(pt2, -2):
                continue
            x1, y1 = pt1[0], pt1[1]
            x2, y2 = pt2[0], pt2[1]
            p1 = int(x1 * width), int(y1 * height)
            p2 = int(x2 * width), int(y2 * height)
            cv2.line(frame, p1, p2, color_lines, 1)

    for frame_data in features_np:
        if np.allclose(frame_data, -2):  # skip fully padded frame
            continue

        frame = np.zeros((height, width, 3), dtype=np.uint8) * 255  # black background

        pose = frame_data[:25]
        left_hand = frame_data[25:46]
        right_hand = frame_data[46:67]

        draw_subset(frame, left_hand, HAND_CONNECTIONS, prefix="LH")
        draw_subset(frame, right_hand, HAND_CONNECTIONS, prefix="RH")
        draw_subset(frame, pose, POSE_CONNECTIONS, prefix="P", plot_labels=True)

        writer.write(frame)

    writer.release()
    print(f"    ... Black-background landmark video saved to {save_path}")


# ------------------ MAIN FUNCTION ------------------

load_dotenv()


def main():
    # Set your args to initialize constructor
    args = {
        "body_features_parquet_path": os.getenv('WLASL100_HAND_POSE_LANDMARKS_PATH'),
        "facial_blendshapes_parquet_path": os.getenv('WLASL100_FACIAL_BLENDSHAPES_PATH'),
        "metadata_json_path": os.getenv("WLASL_METADATA_PATH"),
        "split": "train",
        "transform": None,
        "features": "HAND_POSE_LANDMARKS",
        "include_blendshapes": 0,
        "fs": 1,
        "feature_padding_mode": "repeat",
        "n_heads": 8
    }

    # Create dataset and loader
    dataset = WLASLParquetDataset(**args)
    print(f"[INFO] Loaded dataset with {len(dataset)} samples and {len(dataset.gloss2idx)} classes")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"[INFO] Feature dimension is {dataset.feature_dim}")
    print(f'[INFO] loaded {len(dataloader)} batches')

    # Iterate and test
    for i, (features, labels) in enumerate(dataloader):
        print(f"\nBatch {i + 1}")
        print("Features shape:", features.shape)  # Expect (B, seq_len, d_model)
        decoded_labels = [(idx.item(), dataset.idx2gloss[idx.item()]) for idx in labels]
        print("Labels:", decoded_labels)

        # Visualize the first sample from first batch if features are landmarks and feature selection is off
        if "landmarks" in args["features"].lower().split("_") and args["fs"] == 0:
            for idx, feature_tensor in enumerate(features[:3]):
                plot_landmarks(feature_tensor, f'tmp/{decoded_labels[idx][1]}_batch_{i}_idx_{idx}.mp4')
        if i == 0:
            break


if __name__ == "__main__":
    main()
