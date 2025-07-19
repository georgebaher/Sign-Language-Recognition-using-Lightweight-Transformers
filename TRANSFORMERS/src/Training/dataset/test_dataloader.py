# test_dataloader.py
#
# A comprehensive and professional test script to validate the functionality of the
# SignLanguageFeaturesDataset class across a wide range of scenarios for both
# WLASL and AVASAG datasets.

import os
import torch
import sys
from dotenv import load_dotenv
from argparse import ArgumentParser

# Ensure the dataloader can be imported
sys.path.append('.')
from SLFeaturesDataset import SignLanguageFeaturesDataset


def run_test_scenario(scenario_name, **kwargs):
    """A helper function to run and validate a single dataloader test."""
    print(f"\n{'=' * 25} TESTING SCENARIO: {scenario_name} {'=' * 25}")
    try:
        dataset = SignLanguageFeaturesDataset(**kwargs)

        if len(dataset) == 0:
            print("  -> [SUCCESS]: Dataset initialized, but contains 0 instances for this split.")
            return

        features, label = dataset[0]

        print("\n--- Verifying First Sample ---")
        print(f"  Features tensor shape: {features.shape}")
        print(f"  Label tensor value: {label.item()} (Gloss: '{dataset.idx2gloss.get(label.item(), 'N/A')}')")

        expected_shape = (dataset.max_len, dataset.embedding_dim)
        assert features.shape == expected_shape, \
            f"Dimension mismatch! Expected {expected_shape}, but got {features.shape}"

        print("\n  Sample of feature data (first 5 rows, first 5 columns):")
        print(features[:5, :5])

        print(f"\n[SUCCESS] SCENARIO '{scenario_name}' PASSED")

    except Exception as e:
        print(f"\n[FAILED] SCENARIO '{scenario_name}' FAILED")
        print(f"  -> Error: {e}")
        # Uncomment the next line for a full error traceback during deep debugging
        # raise e


def main():
    """Runs all test scenarios for the SignLanguageFeaturesDataset."""
    load_dotenv()

    parser = ArgumentParser(description="Run test scenarios for the dataloader.")
    parser.add_argument('--n_glosses', type=int, default=5,
                        help="Number of glosses to use for testing (a small number is faster).")
    args = parser.parse_args()

    print(f"--- Running all tests using the first {args.n_glosses} glosses from metadata ---")

    # --- Load all paths from .env and validate them ---
    required_paths = [
        "WLASL_METADATA_PATH", "WLASL_TOP_FEATURES_DIR", "WLASL100_POSE_LANDMARKS_PATH",
        "WLASL100_HAND_LANDMARKS_PATH", "WLASL100_FACE_LANDMARKS_PATH", "WLASL100_FACIAL_BLENDSHAPES_PATH",
        "AVASAG_METADATA_PATH", "AVASAG_TOP_FEATURES_DIR", "AVASAG_POSE_ANGLES_PATH", "AVASAG_HAND_ANGLES_PATH"
    ]
    paths = {key: os.getenv(key) for key in required_paths}
    for key, path in paths.items():
        if not path or not (os.path.exists(path) or os.path.isdir(path)):
            print(
                f"CRITICAL ERROR: Environment variable '{key}' is not set or path '{path}' does not exist. Halting tests.")
            return

    # =================================================================
    #                       WLASL TEST SCENARIOS
    # =================================================================

    # --- WLASL SCENARIO 1: Pose + Hand Landmarks (No FS) ---
    run_test_scenario(
        "WLASL Pose+Hand Landmarks (No FS)",
        metadata_json_path=paths["WLASL_METADATA_PATH"],
        pose_landmark_path=paths["WLASL100_POSE_LANDMARKS_PATH"],
        hand_landmark_path=paths["WLASL100_HAND_LANDMARKS_PATH"],
        split="train", features=['pose_landmarks', 'hand_landmarks'], fs=0, n_glosses=args.n_glosses
    )

    # --- WLASL SCENARIO 2: Pose + Hand Landmarks (With FS) ---
    run_test_scenario(
        "WLASL Pose+Hand Landmarks (With FS)",
        metadata_json_path=paths["WLASL_METADATA_PATH"],
        feature_selection_dir=paths["WLASL_TOP_FEATURES_DIR"],
        pose_landmark_path=paths["WLASL100_POSE_LANDMARKS_PATH"],
        hand_landmark_path=paths["WLASL100_HAND_LANDMARKS_PATH"],
        split="val", features=['pose_landmarks', 'hand_landmarks'], fs=1, n_glosses=args.n_glosses
    )

    # --- WLASL SCENARIO 3: Face Landmarks + Blendshapes (No FS) ---
    run_test_scenario(
        "WLASL Face Landmarks + Blendshapes (No FS)",
        metadata_json_path=paths["WLASL_METADATA_PATH"],
        face_landmark_path=paths["WLASL100_FACE_LANDMARKS_PATH"],
        facial_blendshape_path=paths["WLASL100_FACIAL_BLENDSHAPES_PATH"],
        split="test", features=['face_landmarks', 'facial_blendshapes'], fs=0, n_glosses=args.n_glosses
    )

    # --- WLASL SCENARIO 4: Face Landmarks + Blendshapes (With FS) ---
    run_test_scenario(
        "WLASL Face Landmarks + Blendshapes (With FS)",
        metadata_json_path=paths["WLASL_METADATA_PATH"],
        feature_selection_dir=paths["WLASL_TOP_FEATURES_DIR"],
        face_landmark_path=paths["WLASL100_FACE_LANDMARKS_PATH"],
        facial_blendshape_path=paths["WLASL100_FACIAL_BLENDSHAPES_PATH"],
        split="train", features=['face_landmarks', 'facial_blendshapes'], fs=1, n_glosses=args.n_glosses
    )

    # =================================================================
    #                      AVASAG TEST SCENARIOS
    # =================================================================

    # --- AVASAG SCENARIO 1: Pose + Hand Angles (No FS) ---
    run_test_scenario(
        "AVASAG Pose+Hand Angles (No FS)",
        metadata_json_path=paths["AVASAG_METADATA_PATH"],
        pose_angle_path=paths["AVASAG_POSE_ANGLES_PATH"],
        hand_angle_path=paths["AVASAG_HAND_ANGLES_PATH"],
        split="train", features=['pose_angles', 'hand_angles'], fs=0, n_glosses=args.n_glosses
    )

    # --- AVASAG SCENARIO 2: Pose + Hand Angles (With FS) ---
    run_test_scenario(
        "AVASAG Pose+Hand Angles (With FS)",
        metadata_json_path=paths["AVASAG_METADATA_PATH"],
        feature_selection_dir=paths["AVASAG_TOP_FEATURES_DIR"],
        pose_angle_path=paths["AVASAG_POSE_ANGLES_PATH"],
        hand_angle_path=paths["AVASAG_HAND_ANGLES_PATH"],
        split="val", features=['pose_angles', 'hand_angles'], fs=1, n_glosses=args.n_glosses
    )


if __name__ == "__main__":
    main()