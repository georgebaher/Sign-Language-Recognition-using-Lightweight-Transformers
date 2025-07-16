import os
import cv2
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

def convert_frames_to_videos(input_root, output_root, fps=25):
    os.makedirs(output_root, exist_ok=True)

    # list all folders
    folders = [f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))]

    for folder in tqdm(folders, desc="Converting to videos"):
        folder_path = os.path.join(input_root, folder)
        gloss_id, instance_id = folder.split("_")

        # get sorted images
        frames = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ])

        if not frames:
            continue

        # get frame size
        img0 = cv2.imread(frames[0])
        height, width, _ = img0.shape

        # video writer
        video_name = folder + ".mp4"
        video_path = os.path.join(output_root, video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        for frame_file in frames:
            img = cv2.imread(frame_file)
            out.write(img)

        out.release()

        print(f" Saved {video_path}")

if __name__ == "__main__":
    input_root = os.getenv("AVASAG_FRAMES_FOLDERS_PATH")
    output_root = os.getenv("AVASAG_VIDEOS_FOLDER_PATH")
    convert_frames_to_videos(input_root, output_root)
