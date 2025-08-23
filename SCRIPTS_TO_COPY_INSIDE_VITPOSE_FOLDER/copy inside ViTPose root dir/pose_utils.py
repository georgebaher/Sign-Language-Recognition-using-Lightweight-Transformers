# pose_utils.py
#
# A utility library for ViTPose inference. Contains reusable functions
# for loading models, processing single frames, and visualizing landmarks.
# This version normalizes landmark coordinates to the [0, 1] range.

import numpy as np
import cv2

# Necessary mmpose imports
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results)
from mmpose.datasets import DatasetInfo

# Necessary mmdet imports
try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.3
FONT_COLOR = (255, 255, 255)  # White

# ==============================================================================
# VISUALIZATION SETTINGS (Enhanced for Visibility)
# ==============================================================================
POSE_COLOR = (255, 128, 0)  # Orange
FACE_COLOR = (0, 255, 0)  # Green
HAND_COLOR = (0, 0, 255)  # Red
OUTLINE_COLOR = (0, 0, 0) # Black for contrast


LINE_THICKNESS = 2
FACE_LINE_THICKNESS = 1
FACE_POINT_RADIUS = 2
POINT_RADIUS = 4

# --- Landmark Connection Maps ---
POSE_CONNECTIONS = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
                    [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
                    [1, 3], [2, 4], [3, 5], [4, 6]]
HAND_CONNECTIONS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
                    [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],
                    [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
FACE_CONNECTIONS = [
    # Jawline
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
    # Eyebrows
    [17, 18], [18, 19], [19, 20], [20, 21],  # Left
    [22, 23], [23, 24], [24, 25], [25, 26],  # Right
    # Nose
    [27, 28], [28, 29], [29, 30],
    [31, 32], [32, 33], [33, 34], [34, 35],
    # Eyes
    [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36],  # Left
    [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42],  # Right
    # Mouth
    [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 48],  # Outer
    [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 60]   # Inner
]


def load_models(det_config, det_checkpoint, pose_config, pose_checkpoint, device='cuda:0'):
    """Initializes and loads the detection and pose estimation models."""
    assert has_mmdet, 'Please install mmdet to use this utility.'

    print("Initializing models...")
    detector = init_detector(det_config, det_checkpoint, device=device)
    pose_estimator = init_pose_model(pose_config, pose_checkpoint, device=device)
    dataset_info = DatasetInfo(pose_estimator.cfg.data['test'].get('dataset_info', None))

    print("Models initialized successfully.")
    return detector, pose_estimator, dataset_info


def process_and_flatten_landmarks(keypoints, confidence_threshold, frame_width, frame_height):
    """
    Processes keypoints, normalizes them, pads low-confidence ones with -2,
    and flattens to a list.
    """
    processed_kps = []
    for kp in keypoints:
        x, y, conf = kp
        if conf < confidence_threshold:
            processed_kps.extend([-2, -2])
        else:
            # --- NORMALIZATION STEP ---
            norm_x = x / frame_width
            norm_y = y / frame_height
            processed_kps.extend([norm_x, norm_y])
    return processed_kps


def get_pose_data(pose_results, video_id, frame_idx, gloss="UNSPECIFIED", conf_thr=0.3, frame_width=None, frame_height=None):
    """
    Processes the raw pose_results into structured lists of dictionaries.
    Now includes the 'gloss' label.
    """
    pose_rows, face_rows, hand_rows = [], [], []

    # Get frame dimensions for normalization if they are provided
    if frame_width is None or frame_height is None:
        # This is a fallback, assuming no normalization is needed if dimensions aren't passed
        frame_width, frame_height = 1, 1

    if not pose_results:
        base_row = {'video_id': video_id, 'frame': frame_idx, 'person_id': -1, 'gloss': gloss}
        pose_rows.append(base_row)
        face_rows.append(base_row)
        hand_rows.append(base_row)
    else:
        for person_id, person_data in enumerate(pose_results):
            all_keypoints = person_data['keypoints']
            base_row = {'video_id': video_id, 'frame': frame_idx, 'person_id': person_id, 'gloss': gloss}

            # Pass frame dimensions to the processing function
            pose_kps = all_keypoints[0:23]
            pose_xy = process_and_flatten_landmarks(pose_kps, conf_thr, frame_width, frame_height)
            pose_row = base_row.copy()
            pose_row.update({f'p{i // 2}_{"x" if i % 2 == 0 else "y"}': val for i, val in enumerate(pose_xy)})
            pose_rows.append(pose_row)

            face_kps = all_keypoints[23:91]
            face_xy = process_and_flatten_landmarks(face_kps, conf_thr, frame_width, frame_height)
            face_row = base_row.copy()
            face_row.update({f'f{i // 2}_{"x" if i % 2 == 0 else "y"}': val for i, val in enumerate(face_xy)})
            face_rows.append(face_row)

            hand_kps = all_keypoints[91:133]
            hand_xy = process_and_flatten_landmarks(hand_kps, conf_thr, frame_width, frame_height)
            hand_row = base_row.copy()
            hand_row.update({f'h{i // 2}_{"x" if i % 2 == 0 else "y"}': val for i, val in enumerate(hand_xy)})
            hand_rows.append(hand_row)

    return pose_rows, face_rows, hand_rows


def process_single_frame(detector, pose_estimator, dataset_info, frame, bbox_thr=0.5):
    """Runs inference on a single frame to get raw pose results in pixel coordinates."""
    mmdet_results = inference_detector(detector, frame)
    person_results = process_mmdet_results(mmdet_results, cat_id=1)

    pose_results, _ = inference_top_down_pose_model(
        pose_estimator, frame, person_results,
        bbox_thr=bbox_thr, format='xyxy',
        dataset=pose_estimator.cfg.data['test']['type'],
        dataset_info=dataset_info
    )
    return pose_results


def draw_landmarks_on_frame(frame, pose_results, kpt_thr=0.3, draw_pose=True, draw_hands=True, draw_face=True):
    """
    Draws landmarks and connections on a frame with enhanced visibility and toggles.
    Includes face tessellation.
    """
    for person_data in pose_results:
        keypoints = person_data['keypoints']

        # --- First, draw all connections (lines) with outlines for visibility ---
        if draw_pose:
            pose_kps = keypoints[0:23]
            for conn in POSE_CONNECTIONS:
                if pose_kps[conn[0]][2] > kpt_thr and pose_kps[conn[1]][2] > kpt_thr:
                    p1 = (int(pose_kps[conn[0]][0]), int(pose_kps[conn[0]][1]))
                    p2 = (int(pose_kps[conn[1]][0]), int(pose_kps[conn[1]][1]))
                    cv2.line(frame, p1, p2, OUTLINE_COLOR, LINE_THICKNESS + 2)
                    cv2.line(frame, p1, p2, POSE_COLOR, LINE_THICKNESS)

        if draw_hands:
            left_hand_kps = keypoints[91:112];
            right_hand_kps = keypoints[112:133]
            for hand_kps in [left_hand_kps, right_hand_kps]:
                for conn in HAND_CONNECTIONS:
                    if hand_kps[conn[0]][2] > kpt_thr and hand_kps[conn[1]][2] > kpt_thr:
                        p1 = (int(hand_kps[conn[0]][0]), int(hand_kps[conn[0]][1]))
                        p2 = (int(hand_kps[conn[1]][0]), int(hand_kps[conn[1]][1]))
                        cv2.line(frame, p1, p2, OUTLINE_COLOR, LINE_THICKNESS + 2)
                        cv2.line(frame, p1, p2, HAND_COLOR, LINE_THICKNESS)

        # --- NEW: Draw Face Tessellation (Connections) ---
        if draw_face:
            face_kps = keypoints[23:91]
            for conn in FACE_CONNECTIONS:
                # Indices for FACE_CONNECTIONS are relative to the face block (0-67)
                if face_kps[conn[0]][2] > kpt_thr and face_kps[conn[1]][2] > kpt_thr:
                    p1 = (int(face_kps[conn[0]][0]), int(face_kps[conn[0]][1]))
                    p2 = (int(face_kps[conn[1]][0]), int(face_kps[conn[1]][1]))
                    cv2.line(frame, p1, p2, OUTLINE_COLOR, FACE_LINE_THICKNESS + 1)
                    cv2.line(frame, p1, p2, FACE_COLOR, FACE_LINE_THICKNESS)

        # --- Second, draw all keypoints (circles) on top of the lines ---
        if draw_pose:
            pose_kps = keypoints[0:23]
            for kp in pose_kps:
                if kp[2] > kpt_thr:
                    center = (int(kp[0]), int(kp[1]))
                    cv2.circle(frame, center, POINT_RADIUS + 1, OUTLINE_COLOR, -1)
                    cv2.circle(frame, center, POINT_RADIUS, POSE_COLOR, -1)

        if draw_face:
            face_kps = keypoints[23:91]
            for kp in face_kps:
                if kp[2] > kpt_thr:
                    center = (int(kp[0]), int(kp[1]))
                    cv2.circle(frame, center, FACE_POINT_RADIUS + 1, OUTLINE_COLOR, -1)
                    cv2.circle(frame, center, FACE_POINT_RADIUS, FACE_COLOR, -1)

        if draw_hands:
            left_hand_kps = keypoints[91:112];
            right_hand_kps = keypoints[112:133]
            for hand_kps in [left_hand_kps, right_hand_kps]:
                for kp in hand_kps:
                    if kp[2] > kpt_thr:
                        center = (int(kp[0]), int(kp[1]))
                        cv2.circle(frame, center, POINT_RADIUS + 1, OUTLINE_COLOR, -1)
                        cv2.circle(frame, center, POINT_RADIUS, HAND_COLOR, -1)

    return frame