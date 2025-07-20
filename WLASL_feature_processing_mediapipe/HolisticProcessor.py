import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import mediapipe as mp


class HolisticProcessor:
    """
    A class to extract and process landmarks from videos using MediaPipe Holistic.
    This version returns separate DataFrames for pose, face, and hand landmarks.

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
        - extract (list): Landmark types to extract. Options include 'pose', 'hand' and 'face'.
        """
        if extract is None:
            extract = ["pose", "hand", "face"]
        load_dotenv()
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

    def close(self):
        self.holistic.close()

    def _generate_column_names(self):
        """
        Generates a dictionary of column name lists for each feature set.

        Returns:
        - dict: A dictionary like {'pose': [...], 'face': [...], 'hand': [...]}.
        """
        columns = {}
        if "pose" in self.extract:
            columns['pose'] = [f'P#{i}_{axis}' for i in range(self._landmark_counts['pose']) for axis in
                               ['x', 'y', 'z']]
        if "hand" in self.extract:
            hand_cols = []
            for i in range(self._landmark_counts['left_hand']):
                hand_cols.extend([f'LH#{i}_x', f'LH#{i}_y', f'LH#{i}_z'])
            for i in range(self._landmark_counts['right_hand']):
                hand_cols.extend([f'RH#{i}_x', f'RH#{i}_y', f'RH#{i}_z'])
            columns['hand'] = hand_cols
        if "face" in self.extract:
            columns['face'] = [f'F#{i}_{axis}' for i in range(self._landmark_counts['face']) for axis in
                               ['x', 'y', 'z']]
        return columns

    def _extract_landmarks_as_dict(self, results, pad_val=-2):
        """
        Extracts landmarks from MediaPipe results into a dictionary of NumPy arrays.

        Returns:
        - dict: A dictionary like {'pose': np.ndarray, 'left_hand': np.ndarray, ...}.
        """

        def extract(landmarks, count, use_visibility=False, vis_threshold=0.5):
            if not landmarks:
                return np.full((count, 3), pad_val)
            result = []
            for lm in landmarks.landmark:
                if use_visibility:
                    # Only used for pose extracted_landmarks
                    vis = getattr(lm, "visibility", None)
                    if vis is not None and vis < vis_threshold:
                        result.append([pad_val] * 3)
                    else:
                        result.append([lm.x, lm.y, lm.z])
                else:
                    # For hands/face: visibility is meaningless → just return xyz
                    result.append([lm.x, lm.y, lm.z])
            return np.array(result)

        extracted_data = {}
        if "pose" in self.extract:
            extracted_data['pose'] = extract(results.pose_landmarks, self._landmark_counts['pose'], True)
        if "hand" in self.extract:
            extracted_data['left_hand'] = extract(results.left_hand_landmarks, self._landmark_counts['left_hand'],
                                                  False)
            extracted_data['right_hand'] = extract(results.right_hand_landmarks, self._landmark_counts['right_hand'],
                                                   False)
        if "face" in self.extract:
            extracted_data['face'] = extract(results.face_landmarks, self._landmark_counts['face'], False)
        return extracted_data

    def process_video(self, video_path, save_annotation=False, gloss=""):
        """
        Processes a single video and returns separate DataFrames for each feature set.

        Returns:
        - tuple: (pose_df, face_df, hand_df, video_output_path)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_id = os.path.basename(video_path)
        clean_video_id = video_id.replace(".mp4", "")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video_output_path = None
        out = None

        if save_annotation:
            temp_dir = os.getenv("WLASL_MEDIAPIPE_ANNOTATED_VIDEOS")
            os.makedirs(temp_dir, exist_ok=True)
            video_output_path = os.path.join(temp_dir, f"{clean_video_id}_annotated.mp4")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

        # Initialize separate lists for each dataframe
        pose_data_list, face_data_list, hand_data_list = [], [], []

        with tqdm(total=frame_count, desc=f"Processing <{video_id}>", unit='frame') as pbar:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(image_rgb)

                landmark_dict = self._extract_landmarks_as_dict(results)

                # Append flattened data to the correct list
                if 'pose' in landmark_dict:
                    pose_data_list.append(landmark_dict['pose'].flatten())
                if 'face' in landmark_dict:
                    face_data_list.append(landmark_dict['face'].flatten())
                if 'left_hand' in landmark_dict and 'right_hand' in landmark_dict:
                    combined_hand = np.concatenate([landmark_dict['left_hand'], landmark_dict['right_hand']])
                    hand_data_list.append(combined_hand.flatten())

                if save_annotation and out:
                    # Drawing logic remains the same
                    frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    if results.pose_landmarks and "pose" in self.extract:
                        self.mp_drawing.draw_landmarks(frame_bgr, results.pose_landmarks,
                                                       self.mp_holistic.POSE_CONNECTIONS)
                    if "hand" in self.extract:
                        if results.left_hand_landmarks: self.mp_drawing.draw_landmarks(frame_bgr,
                                                                                       results.left_hand_landmarks,
                                                                                       self.mp_holistic.HAND_CONNECTIONS)
                        if results.right_hand_landmarks: self.mp_drawing.draw_landmarks(frame_bgr,
                                                                                        results.right_hand_landmarks,
                                                                                        self.mp_holistic.HAND_CONNECTIONS)
                    if results.face_landmarks and "face" in self.extract:
                        self.mp_drawing.draw_landmarks(frame_bgr, results.face_landmarks,
                                                       self.mp_holistic.FACEMESH_TESSELATION)
                    out.write(frame_bgr)

                pbar.update(1)

        cap.release()
        if out:
            out.release()

        # Generate DataFrames from the collected lists
        column_map = self._generate_column_names()
        vid_id_clean = video_id.split(".")[0]

        def create_df(data_list, columns, name):
            if not data_list:
                print(f"No {name} landmarks detected in video {video_id}")
                return pd.DataFrame()
            df = pd.DataFrame(data_list, columns=columns)
            df.insert(0, 'video_id', vid_id_clean)
            df.insert(1, 'gloss', gloss or "nil")
            return df

        pose_df = create_df(pose_data_list, column_map.get('pose', []), 'pose')
        face_df = create_df(face_data_list, column_map.get('face', []), 'face')
        hand_df = create_df(hand_data_list, column_map.get('hand', []), 'hand')

        return pose_df, face_df, hand_df, video_output_path
