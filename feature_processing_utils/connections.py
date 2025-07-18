# connections.py
#
# A centralized file to store landmark connection maps for different
# pose estimation models like MediaPipe and ViTPose.

import mediapipe as mp

# --- MediaPipe Official Connection Maps ---
# These are sourced directly from the MediaPipe library for accuracy.
# Note: The POSE_CONNECTIONS from MediaPipe includes connections for face,
# but we will only use the body part for angle calculations.
MEDIAPIPE_POSE_CONNECTIONS = list(mp.solutions.holistic.POSE_CONNECTIONS)
MEDIAPIPE_HAND_CONNECTIONS = list(mp.solutions.holistic.HAND_CONNECTIONS)

# --- ViTPose / AVASAG Custom Connection Maps ---
# As provided from the ViTPose implementation.
VITPOSE_POSE_CONNECTIONS = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
    [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6]
]

# Note: The 21-point hand model and its connections are standard,
# so this is identical to MediaPipe's hand connections.
VITPOSE_HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],
    [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]
]

# --- Master Dictionary for Easy Selection ---
# This allows scripts to select the correct connection map using a simple string.
CONNECTION_MAPS = {
    "mediapipe_pose": MEDIAPIPE_POSE_CONNECTIONS,
    "mediapipe_hand": MEDIAPIPE_HAND_CONNECTIONS,
    "vitpose_pose": VITPOSE_POSE_CONNECTIONS,
    "vitpose_hand": VITPOSE_HAND_CONNECTIONS
}


def get_connections(source_name: str, body_part: str):
    """
    A helper function to safely get the correct connection map.

    Args:
        source_name (str): The name of the model source (e.g., 'mediapipe', 'vitpose').
        body_part (str): The name of the body part (e.g., 'pose', 'hand').

    Returns:
        list: The corresponding list of connections.
    """
    key = f"{source_name.lower()}_{body_part.lower()}"
    if key not in CONNECTION_MAPS:
        raise ValueError(f"Invalid source/part combination: '{source_name}', '{body_part}'. "
                         f"Valid keys are: {list(CONNECTION_MAPS.keys())}")
    return CONNECTION_MAPS[key]