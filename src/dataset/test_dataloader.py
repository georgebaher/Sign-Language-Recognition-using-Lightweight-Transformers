# test_dataloader.py
#
# A comprehensive and professional test script to validate the functionality of the
# SignLanguageFeaturesDataset class across 24 scenarios for both WLASL and AVASAG.

import os
import torch
import sys
from dotenv import load_dotenv
from argparse import ArgumentParser

# Ensure the dataloader can be imported
sys.path.append('.')
from SLFeaturesDataset import SignLanguageFeaturesDataset

# --- Global lists to track the outcome of each test ---
success_scenarios = []
failed_scenarios = []


def run_test_scenario(scenario_name, **kwargs):
    """A helper function to run and validate a single dataloader test."""
    print(f"\n{'=' * 25} TESTING SCENARIO: {scenario_name} {'=' * 25}")
    try:
        dataset = SignLanguageFeaturesDataset(**kwargs)

        if len(dataset) == 0:
            print("  -> [SUCCESS]: Dataset initialized, but contains 0 instances for this split.")
            success_scenarios.append(scenario_name)
            return

        print("\n--- Verifying Feature Columns ---")
        print(f"  Feature columns being used: {dataset.final_columns}")

        features, label = dataset[0]

        print("\n--- Verifying First Sample ---")
        print(f"  Features tensor shape: {features.shape}")
        print(f"  Label tensor value: {label.item()} (Gloss: '{dataset.idx2gloss.get(label.item(), 'N/A')}')")

        expected_shape = (dataset.max_len, dataset.embedding_dim)
        assert features.shape == expected_shape, \
            f"Dimension mismatch! Expected {expected_shape}, but got {features.shape}"

        print(f"\n[SUCCESS] SCENARIO '{scenario_name}' PASSED")
        success_scenarios.append(scenario_name)

    except Exception as e:
        print(f"\n[FAILED] SCENARIO '{scenario_name}' FAILED")
        print(f"  -> Error: {e}")
        failed_scenarios.append(f"{scenario_name} (Error: {e})")
        # raise e # Uncomment for full traceback


def main():
    """Runs all test scenarios for the SignLanguageFeaturesDataset."""
    load_dotenv()

    parser = ArgumentParser(description="Run test scenarios for the dataloader.")
    parser.add_argument('--n_glosses', type=int, default=5, help="Number of glosses to use for testing.")
    args = parser.parse_args()

    print(f"--- Running all tests using the first {args.n_glosses} glosses from metadata ---")

    # --- Load all paths from .env and validate them ---
    required_paths = [
        "WLASL_METADATA_PATH", "WLASL_TOP_FEATURES_DIR", "WLASL_POSE_LANDMARKS_PATH",
        "WLASL_HAND_LANDMARKS_PATH", "WLASL_FACE_LANDMARKS_PATH", "WLASL_POSE_ANGLES_PATH",
        "WLASL_HAND_ANGLES_PATH", "WLASL_FACE_BLENDSHAPES_PATH",
        "AVASAG_METADATA_PATH", "AVASAG_TOP_FEATURES_DIR", "AVASAG_POSE_LANDMARKS_PATH",
        "AVASAG_HAND_LANDMARKS_PATH", "AVASAG_FACE_LANDMARKS_PATH", "AVASAG_POSE_ANGLES_PATH",
        "AVASAG_HAND_ANGLES_PATH", "AVASAG_FACE_BLENDSHAPES_PATH"
    ]
    paths = {key: os.getenv(key) for key in required_paths}
    for key, path in paths.items():
        if not path or not (os.path.exists(path) or os.path.isdir(path)):
            print(f"CRITICAL ERROR: Env variable '{key}' not set or path '{path}' does not exist. Halting.")
            return

    # =================================================================
    #                       WLASL TEST SCENARIOS (12 TOTAL)
    # =================================================================

    run_test_scenario("WLASL Pose Landmarks (No FS)", metadata_json_path=paths["WLASL_METADATA_PATH"],
                      pose_landmark_path=paths["WLASL_POSE_LANDMARKS_PATH"], split="train", features=['pose_landmarks'],
                      fs=0, n_glosses=args.n_glosses)
    run_test_scenario("WLASL Pose Landmarks (With FS)", metadata_json_path=paths["WLASL_METADATA_PATH"],
                      feature_selection_dir=paths["WLASL_TOP_FEATURES_DIR"],
                      pose_landmark_path=paths["WLASL_POSE_LANDMARKS_PATH"], split="train", features=['pose_landmarks'],
                      fs=1, n_glosses=args.n_glosses)
    run_test_scenario("WLASL Pose Angles (No FS)", metadata_json_path=paths["WLASL_METADATA_PATH"],
                      pose_angle_path=paths["WLASL_POSE_ANGLES_PATH"], split="train", features=['pose_angles'], fs=0,
                      n_glosses=args.n_glosses)
    run_test_scenario("WLASL Pose Angles (With FS)", metadata_json_path=paths["WLASL_METADATA_PATH"],
                      feature_selection_dir=paths["WLASL_TOP_FEATURES_DIR"],
                      pose_angle_path=paths["WLASL_POSE_ANGLES_PATH"], split="train", features=['pose_angles'], fs=1,
                      n_glosses=args.n_glosses)
    run_test_scenario("WLASL Hand Landmarks (No FS)", metadata_json_path=paths["WLASL_METADATA_PATH"],
                      hand_landmark_path=paths["WLASL_HAND_LANDMARKS_PATH"], split="train", features=['hand_landmarks'],
                      fs=0, n_glosses=args.n_glosses)
    run_test_scenario("WLASL Hand Landmarks (With FS)", metadata_json_path=paths["WLASL_METADATA_PATH"],
                      feature_selection_dir=paths["WLASL_TOP_FEATURES_DIR"],
                      hand_landmark_path=paths["WLASL_HAND_LANDMARKS_PATH"], split="train", features=['hand_landmarks'],
                      fs=1, n_glosses=args.n_glosses)
    run_test_scenario("WLASL Hand Angles (No FS)", metadata_json_path=paths["WLASL_METADATA_PATH"],
                      hand_angle_path=paths["WLASL_HAND_ANGLES_PATH"], split="train", features=['hand_angles'], fs=0,
                      n_glosses=args.n_glosses)
    run_test_scenario("WLASL Hand Angles (With FS)", metadata_json_path=paths["WLASL_METADATA_PATH"],
                      feature_selection_dir=paths["WLASL_TOP_FEATURES_DIR"],
                      hand_angle_path=paths["WLASL_HAND_ANGLES_PATH"], split="train", features=['hand_angles'], fs=1,
                      n_glosses=args.n_glosses)
    run_test_scenario("WLASL Face Landmarks+Blendshapes (No FS)", metadata_json_path=paths["WLASL_METADATA_PATH"],
                      face_landmark_path=paths["WLASL_FACE_LANDMARKS_PATH"],
                      face_blendshape_path=paths["WLASL_FACE_BLENDSHAPES_PATH"], split="test",
                      features=['face_landmarks', 'face_blendshapes'], fs=0, n_glosses=args.n_glosses)
    run_test_scenario("WLASL Pose+Hand Landmarks & Blendshapes (With FS)",
                      metadata_json_path=paths["WLASL_METADATA_PATH"],
                      feature_selection_dir=paths["WLASL_TOP_FEATURES_DIR"],
                      pose_landmark_path=paths["WLASL_POSE_LANDMARKS_PATH"],
                      hand_landmark_path=paths["WLASL_HAND_LANDMARKS_PATH"],
                      face_blendshape_path=paths["WLASL_FACE_BLENDSHAPES_PATH"], split="val",
                      features=['pose_landmarks', 'hand_landmarks', 'face_blendshapes'], fs=1,
                      n_glosses=args.n_glosses)
    run_test_scenario("WLASL Pose+Hand Angles & Blendshapes (With FS)", metadata_json_path=paths["WLASL_METADATA_PATH"],
                      feature_selection_dir=paths["WLASL_TOP_FEATURES_DIR"],
                      pose_angle_path=paths["WLASL_POSE_ANGLES_PATH"], hand_angle_path=paths["WLASL_HAND_ANGLES_PATH"],
                      face_blendshape_path=paths["WLASL_FACE_BLENDSHAPES_PATH"], split="val",
                      features=['pose_angles', 'hand_angles', 'face_blendshapes'], fs=1, n_glosses=args.n_glosses)
    run_test_scenario("WLASL All Features (No FS)", metadata_json_path=paths["WLASL_METADATA_PATH"],
                      pose_landmark_path=paths["WLASL_POSE_LANDMARKS_PATH"],
                      hand_landmark_path=paths["WLASL_HAND_LANDMARKS_PATH"],
                      face_landmark_path=paths["WLASL_FACE_LANDMARKS_PATH"],
                      pose_angle_path=paths["WLASL_POSE_ANGLES_PATH"], hand_angle_path=paths["WLASL_HAND_ANGLES_PATH"],
                      face_blendshape_path=paths["WLASL_FACE_BLENDSHAPES_PATH"], split="test",
                      features=['pose_landmarks', 'hand_landmarks', 'face_landmarks', 'pose_angles', 'hand_angles',
                                'face_blendshapes'], fs=0, n_glosses=args.n_glosses)

    # =================================================================
    #                      AVASAG TEST SCENARIOS (12 TOTAL)
    # =================================================================

    run_test_scenario("AVASAG Pose Landmarks (No FS)", metadata_json_path=paths["AVASAG_METADATA_PATH"],
                      pose_landmark_path=paths["AVASAG_POSE_LANDMARKS_PATH"], split="train",
                      features=['pose_landmarks'], fs=0, n_glosses=args.n_glosses)
    run_test_scenario("AVASAG Pose Landmarks (With FS)", metadata_json_path=paths["AVASAG_METADATA_PATH"],
                      feature_selection_dir=paths["AVASAG_TOP_FEATURES_DIR"],
                      pose_landmark_path=paths["AVASAG_POSE_LANDMARKS_PATH"], split="train",
                      features=['pose_landmarks'], fs=1, n_glosses=args.n_glosses)
    run_test_scenario("AVASAG Pose Angles (No FS)", metadata_json_path=paths["AVASAG_METADATA_PATH"],
                      pose_angle_path=paths["AVASAG_POSE_ANGLES_PATH"], split="train", features=['pose_angles'], fs=0,
                      n_glosses=args.n_glosses)
    run_test_scenario("AVASAG Pose Angles (With FS)", metadata_json_path=paths["AVASAG_METADATA_PATH"],
                      feature_selection_dir=paths["AVASAG_TOP_FEATURES_DIR"],
                      pose_angle_path=paths["AVASAG_POSE_ANGLES_PATH"], split="train", features=['pose_angles'], fs=1,
                      n_glosses=args.n_glosses)
    run_test_scenario("AVASAG Hand Landmarks (No FS)", metadata_json_path=paths["AVASAG_METADATA_PATH"],
                      hand_landmark_path=paths["AVASAG_HAND_LANDMARKS_PATH"], split="train",
                      features=['hand_landmarks'], fs=0, n_glosses=args.n_glosses)
    run_test_scenario("AVASAG Hand Landmarks (With FS)", metadata_json_path=paths["AVASAG_METADATA_PATH"],
                      feature_selection_dir=paths["AVASAG_TOP_FEATURES_DIR"],
                      hand_landmark_path=paths["AVASAG_HAND_LANDMARKS_PATH"], split="train",
                      features=['hand_landmarks'], fs=1, n_glosses=args.n_glosses)
    run_test_scenario("AVASAG Hand Angles (No FS)", metadata_json_path=paths["AVASAG_METADATA_PATH"],
                      hand_angle_path=paths["AVASAG_HAND_ANGLES_PATH"], split="train", features=['hand_angles'], fs=0,
                      n_glosses=args.n_glosses)
    run_test_scenario("AVASAG Hand Angles (With FS)", metadata_json_path=paths["AVASAG_METADATA_PATH"],
                      feature_selection_dir=paths["AVASAG_TOP_FEATURES_DIR"],
                      hand_angle_path=paths["AVASAG_HAND_ANGLES_PATH"], split="train", features=['hand_angles'], fs=1,
                      n_glosses=args.n_glosses)
    run_test_scenario("AVASAG Face Landmarks+Blendshapes (No FS)", metadata_json_path=paths["AVASAG_METADATA_PATH"],
                      face_landmark_path=paths["AVASAG_FACE_LANDMARKS_PATH"],
                      face_blendshape_path=paths["AVASAG_FACE_BLENDSHAPES_PATH"], split="test",
                      features=['face_landmarks', 'face_blendshapes'], fs=0, n_glosses=args.n_glosses)
    run_test_scenario("AVASAG Pose+Hand Landmarks & Blendshapes (With FS)",
                      metadata_json_path=paths["AVASAG_METADATA_PATH"],
                      feature_selection_dir=paths["AVASAG_TOP_FEATURES_DIR"],
                      pose_landmark_path=paths["AVASAG_POSE_LANDMARKS_PATH"],
                      hand_landmark_path=paths["AVASAG_HAND_LANDMARKS_PATH"],
                      face_blendshape_path=paths["AVASAG_FACE_BLENDSHAPES_PATH"], split="val",
                      features=['pose_landmarks', 'hand_landmarks', 'face_blendshapes'], fs=1,
                      n_glosses=args.n_glosses)
    run_test_scenario("AVASAG Pose+Hand Angles & Blendshapes (With FS)",
                      metadata_json_path=paths["AVASAG_METADATA_PATH"],
                      feature_selection_dir=paths["AVASAG_TOP_FEATURES_DIR"],
                      pose_angle_path=paths["AVASAG_POSE_ANGLES_PATH"],
                      hand_angle_path=paths["AVASAG_HAND_ANGLES_PATH"],
                      face_blendshape_path=paths["AVASAG_FACE_BLENDSHAPES_PATH"], split="val",
                      features=['pose_angles', 'hand_angles', 'face_blendshapes'], fs=1, n_glosses=args.n_glosses)
    run_test_scenario("AVASAG All Features (No FS)", metadata_json_path=paths["AVASAG_METADATA_PATH"],
                      pose_landmark_path=paths["AVASAG_POSE_LANDMARKS_PATH"],
                      hand_landmark_path=paths["AVASAG_HAND_LANDMARKS_PATH"],
                      face_landmark_path=paths["AVASAG_FACE_LANDMARKS_PATH"],
                      pose_angle_path=paths["AVASAG_POSE_ANGLES_PATH"],
                      hand_angle_path=paths["AVASAG_HAND_ANGLES_PATH"],
                      face_blendshape_path=paths["AVASAG_FACE_BLENDSHAPES_PATH"], split="test",
                      features=['pose_landmarks', 'hand_landmarks', 'face_landmarks', 'pose_angles', 'hand_angles',
                                'face_blendshapes'], fs=0, n_glosses=args.n_glosses)

    # --- FINAL SUMMARY REPORT ---
    print("\n\n" + "=" * 30 + " TEST SUMMARY " + "=" * 30)
    total_tests = len(success_scenarios) + len(failed_scenarios)
    print(f"Ran {total_tests} scenarios.")
    print(f"\n[SUCCESS] {len(success_scenarios)} scenarios passed:")
    for name in success_scenarios:
        print(f"  - {name}")

    if failed_scenarios:
        print(f"\n[FAILED] {len(failed_scenarios)} scenarios failed:")
        for name in failed_scenarios:
            print(f"  - {name}")
    print("=" * 74)


if __name__ == "__main__":
    main()