import os
import pandas as pd
from dotenv import load_dotenv
from Feature_Extraction.feature_selection.features.hand_pose_angles_features import TOP_ANGLE_BASES
from Mediapipe_holistic.processing.landmarks_utils import compute_video_landmarks
from Mediapipe_holistic.processing.hand_angles_utils import compute_hand_angles
from Mediapipe_holistic.processing.pose_angles_utils import compute_pose_angles
from SLR_MediaPipe_DTW.dtw_evaluaters.DTWEvaluater import DTWEvaluator

load_dotenv()
# ------------------------------------------------------------------------------
# Step 1: Set your test video path and gloss
# ------------------------------------------------------------------------------
TEST_VIDEO_PATH = os.getenv("TEST_VIDEO_PATH")
TEST_GLOSS = "unknown"  # use actual gloss if known, else 'unknown'
video_id = os.path.splitext(os.path.basename(TEST_VIDEO_PATH))[0]

# ------------------------------------------------------------------------------
# Step 2: Extract landmarks
# ------------------------------------------------------------------------------
landmarks_df, vis_path = compute_video_landmarks(TEST_VIDEO_PATH, gloss=TEST_GLOSS, show_landmarks=True)
if landmarks_df.empty:
    print("❌ Landmark extraction failed.")
    exit()

# ------------------------------------------------------------------------------
# Step 3: Compute + concat hand and pose angles
# ------------------------------------------------------------------------------
hand_angles_df = compute_hand_angles(landmarks_df, video_id, TEST_GLOSS)
pose_angles_df = compute_pose_angles(landmarks_df, video_id, TEST_GLOSS)

if hand_angles_df.empty or pose_angles_df.empty:
    print("❌ Angle extraction failed.")
    exit()

meta_cols = ['video_id', 'gloss']
hand_pose_angles_df = pd.concat(
    [landmarks_df[meta_cols],
     hand_angles_df.drop(columns=meta_cols, errors='ignore'),
     pose_angles_df.drop(columns=meta_cols, errors='ignore')],
    axis=1
)
# ------------------------------------------------------------------------------
# Step 4: Initialize DTWEvaluator and predict gloss
# ------------------------------------------------------------------------------
evaluator = DTWEvaluator(n_glosses=5, angle_bases=TOP_ANGLE_BASES[:300], name="fs_top_300")
result_df = evaluator.evaluate_single(hand_pose_angles_df)
predicted_gloss = result_df["predicted_gloss"]
print(f"🧞 DTW predicted Gloss: {predicted_gloss}")
print(f"📄 Top Matches:\n{result_df['top_k_matches']}")


