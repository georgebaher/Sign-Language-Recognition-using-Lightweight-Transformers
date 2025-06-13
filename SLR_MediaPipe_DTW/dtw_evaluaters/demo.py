import streamlit as st
import tempfile
import sys
import os
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# from SLR_MediaPipe_DTW.feature_selection.features.hand_pose_angles_features import TOP_ANGLE_BASES
from Feature_Extraction.feature_selection.features.hand_angles_features import TOP_ANGLE_BASES
from Mediapipe_holistic.processing.landmarks_utils import compute_video_landmarks
from Mediapipe_holistic.processing.hand_angles_utils import compute_hand_angles
from Mediapipe_holistic.processing.pose_angles_utils import compute_pose_angles
from SLR_MediaPipe_DTW.dtw_evaluaters.DTWEvaluater import DTWEvaluator


load_dotenv()

st.set_page_config(page_title="Sign Gloss Prediction", layout="centered")
st.title("👌 Sign Language Gloss Predictor (DTW)")

uploaded_file = st.file_uploader("Upload a sign video (MP4)", type=["mp4"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(video_path)
    st.info("Processing landmarks and angles... this may take a few seconds")

    # Step 1: Extract landmarks
    video_id = os.path.basename(video_path).split(".")[0]
    landmarks_df, vis_path = compute_video_landmarks(video_path, gloss="unknown", show_landmarks=True)
    if vis_path and os.path.exists(vis_path):
        st.subheader("📹 Visualized Landmarks")
        st.video(vis_path)

    if landmarks_df.empty:
        st.error("❌ Failed to extract landmarks.")
    else:
        # Step 2: Compute hand + pose angles and concatenate
        hand_angles_df = compute_hand_angles(landmarks_df, video_id, "unknown")
        pose_angles_df = compute_pose_angles(landmarks_df, video_id, "unknown")

        if hand_angles_df.empty or pose_angles_df.empty:
            st.error("❌ Failed to compute angles.")
        else:
            meta_cols = ["video_id", "gloss"]
            hand_pose_angles_df = pd.concat([
                landmarks_df[meta_cols],
                hand_angles_df.drop(columns=meta_cols, errors='ignore'),
                pose_angles_df.drop(columns=meta_cols, errors='ignore')
            ], axis=1)

            # Step 3: Run DTW prediction
            evaluator = DTWEvaluator(n_glosses=5, angle_bases=TOP_ANGLE_BASES[:396], name="fs_top_300")
            result = evaluator.evaluate_single(hand_pose_angles_df)

            st.success(f"✅ Predicted Gloss: **{result['predicted_gloss']}**")
            # st.subheader("Top 5 Closest Matches")
            # st.dataframe(result['top_k_matches'].reset_index(drop=True))

    os.remove(video_path)
    os.remove(vis_path)
