import os
import cv2
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def convert_frames_to_videos(input_root, output_root, fps=25):
    os.makedirs(output_root, exist_ok=True)

    folders = [f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))]

    processed_count = 0
    skipped_existing = 0
    skipped_empty = 0

    for folder in tqdm(folders, desc="Converting to videos"):
        folder_path = os.path.join(input_root, folder)
        gloss_id, instance_id = folder.split("_")

        video_name = folder + ".mp4"
        video_path = os.path.join(output_root, video_name)

        # --- Skip if video already exists ---
        if os.path.exists(video_path):
            print(f"Skipping {video_path} (already exists)")
            skipped_existing += 1
            continue

        # get sorted images
        frames = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if not frames:
            print(f"No frames found in {folder_path}, skipping.")
            skipped_empty += 1
            continue

        # get frame size
        img0 = cv2.imread(frames[0])
        height, width, _ = img0.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        for frame_file in frames:
            img = cv2.imread(frame_file)
            out.write(img)

        out.release()

        print(f"Saved {video_path}")
        processed_count += 1

    # --- Print summary ---
    print("\nConversion Summary:")
    print(f"Videos processed: {processed_count}")
    print(f"Skipped (already existed): {skipped_existing}")
    print(f"Skipped (no frames found): {skipped_empty}")
    print(f"Total folders scanned: {len(folders)}")

if __name__ == "__main__":
    input_root = os.getenv("AVASAG_FRAMES_PATH")
    output_root = os.getenv("AVASAG_VIDEOS_PATH")
    convert_frames_to_videos(input_root, output_root)
