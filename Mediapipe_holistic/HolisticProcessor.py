import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import mediapipe as mp

class HolisticProcessor:
    """
    A class to extract and process holistic landmarks (pose, hands) from videos using MediaPipe Holistic.

    Author: George Elswefy (<georgeelswefy@gmail.com>)

    """

    _landmark_counts = {
        'pose': 33,
        'left_hand': 21,
        'right_hand': 21,
        'face': 478
    }

    def __init__(self, extract=None):
        """
        Initializes the HolisticProcessor with environment configurations and MediaPipe setup.

        Parameters:
        - extract (list): Landmark types to extract. Options include 'pose' and 'hand'.
        """
        if extract is None:
            extract = ["pose", "hand", "face"]
        load_dotenv()
        self.window_size = (int(os.getenv("WINDOW_WIDTH")), int(os.getenv("WINDOW_HEIGHT")))
        self.extract = set(extract)
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )


    def _generate_column_names(self):
        """
        Generates column names for the landmark dataframe with body part and index labels.

        Returns:
        - List[str]: Column names like ['P#0_x', 'P#0_y', ...]
        """
        columns = []
        if "pose" in self.extract:
            for i in range(self._landmark_counts['pose']):
                columns.extend([f'P#{i}_x', f'P#{i}_y', f'P#{i}_z'])
        if "hand" in self.extract:
            for i in range(self._landmark_counts['left_hand']):
                columns.extend([f'LH#{i}_x', f'LH#{i}_y', f'LH#{i}_z'])
            for i in range(self._landmark_counts['right_hand']):
                columns.extend([f'RH#{i}_x', f'RH#{i}_y', f'RH#{i}_z'])
        if "face" in self.extract:
            for i in range(self._landmark_counts['face']):
                columns.extend([f'F#{i}_x', f'F#{i}_y', f'F#{i}_z'])
        return columns

    def _draw_landmarks(self, window_name, frame_rgb, results):
        """
        Draws holistic landmarks on the given frame and displays it.

        Parameters:
        - window_name (str): Name of the display window.
        - frame_rgb (np.ndarray): The input RGB frame.
        - results (object): MediaPipe holistic results.

        Returns:
        - bool: False if ESC key is pressed.
        """
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks and "pose" in self.extract:
            self.mp_drawing.draw_landmarks(frame_bgr, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        if "hand" in self.extract:
            if results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame_bgr, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame_bgr, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        if results.face_landmarks and "face" in self.extract:
            self.mp_drawing.draw_landmarks(frame_bgr, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION)

        cv2.imshow(window_name, frame_bgr)
        return not (cv2.waitKey(1) & 0xFF == 27)

    def _extract_landmarks_as_nparray(self, results, pad_val=-2):
        """
        Extracts landmarks from MediaPipe results into a flat NumPy array.

        Parameters:
        - results (object): MediaPipe holistic result.
        - pad_val (float): Value to fill if landmarks are not detected.

        Returns:
        - np.ndarray: Flattened landmark data for one frame.
        """

        def extract(landmarks, count, use_visibility=False, vis_threshold=0.5):
            if not landmarks:
                return np.full((count, 3), pad_val)

            result = []
            for lm in landmarks.landmark:
                if use_visibility:
                    # Only used for pose landmarks
                    vis = getattr(lm, "visibility", None)
                    if vis is not None and vis < vis_threshold:
                        result.append([pad_val] * 3)
                    else:
                        result.append([lm.x, lm.y, lm.z])
                else:
                    # For hands/face: visibility is meaningless → just return xyz
                    result.append([lm.x, lm.y, lm.z])

            return np.array(result)

        data = []
        if "pose" in self.extract:
            data.append(extract(results.pose_landmarks, self._landmark_counts['pose'], True))
        if "hand" in self.extract:
            data.append(extract(results.left_hand_landmarks, self._landmark_counts['left_hand'],False ))
            data.append(extract(results.right_hand_landmarks, self._landmark_counts['right_hand'], False))
        if "face" in self.extract:
            data.append(extract(results.face_landmarks, self._landmark_counts['face']))
        return np.concatenate(data) if data else np.array([])

    def process_video(self, video_path, show_landmarks=False, gloss=""):
        """
        Processes a single video, extracts landmarks frame-by-frame, and optionally visualizes them.

        Parameters:
        - video_path (str): Full path to the video.
        - show_landmarks (bool): Whether to display landmark visualization.
        - gloss (str): gloss of the video, if provided.

        Returns:
        - pd.DataFrame: A DataFrame of landmark features per frame with header.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video {video_path}")
            return pd.DataFrame()

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_id = os.path.basename(video_path)
        window_name = f"{video_id}"

        if show_landmarks:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, *self.window_size)
            cv2.moveWindow(window_name, 0, 0)

        frames_data = []

        with tqdm(total=frame_count, desc=f"Processing <{video_id}> for gloss '{gloss}'", unit='frame', colour='white', leave=False) as pbar:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(image_rgb)
                frame_features = self._extract_landmarks_as_nparray(results)
                if frame_features.size:
                    frames_data.append(frame_features.flatten())

                if show_landmarks:
                    continue_loop = self._draw_landmarks(window_name, image_rgb, results)
                    if not continue_loop:
                        break

                pbar.update(1)

        cap.release()
        if show_landmarks:
            cv2.destroyWindow(window_name)

        if not frames_data:
            print(f"⚠️ No landmarks detected in video {video_id}")
            return pd.DataFrame()

        column_names = self._generate_column_names()
        df = pd.DataFrame(frames_data, columns=column_names)
        vid_id_clean = video_id.split(".")[0]
        df.insert(0, 'video_id', vid_id_clean)
        if gloss:
            df.insert(1, 'gloss', gloss)
        else:
            df.insert(1, 'gloss', "nil")
        return df




if __name__ == "__main__":
    # TESTING...

    # load wlasl videos folder path
    load_dotenv()
    videos_path = os.getenv("WLASL_VIDEOS_PATH")

    # create video file path for example video 69241.mp4 that is an instance of 'book'
    path=os.path.join(videos_path, f"{69241}.mp4")

    # initiate processor
    processor = HolisticProcessor(extract=["pose", "hand"])

    # process video
    result_df=processor.process_video(path, True, "book")

    # print datafame
    print(result_df)
