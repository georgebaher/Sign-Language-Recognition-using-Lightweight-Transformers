import os
import time
import pandas as pd
from tqdm import tqdm
from typing import Literal
from dotenv import load_dotenv
from math import ceil
from collections import Counter
from SLR_MediaPipe_DTW.utils.dtw import dtw_distances
from SLR_MediaPipe_DTW.utils.plt_eval_metrics import plot_dtw_evaluation_results
from SLR_MediaPipe_DTW.utils.analyze_wlasl import get_train_val_test_split_per_gloss


def drop_meta(df):
    return df.drop(columns=["video_id", "gloss"], errors="ignore")


class DTWEvaluator:
    def __init__(self, feature_option: Literal[
        "hand_landmarks", "hand+pose_landmarks", "hand_angles", "hand+pose_angles"
    ], n_glosses: int = 10):
        load_dotenv()

        self.feature_option = feature_option
        self.n_glosses = n_glosses

        # Load paths
        self.metadata_path = os.getenv("WLASL_METADATA_PATH")
        self.hand_pose_landmarks_path = os.getenv("WLASL100_HAND_POSE_LANDMARKS_PATH")
        self.hand_landmarks_path = os.getenv("WLASL100_HAND_LANDMARKS_PATH")
        self.hand_angles_path = os.getenv("WLASL100_HAND_ANGLES_PATH")
        self.hand_pose_angles_path = os.getenv("WLASL100_HAND_POSE_ANGLES_PATH")

        # Load data
        self._validate_paths()
        self.hand_pose_landmarks_df = pd.read_parquet(self.hand_pose_landmarks_path)
        self.hand_landmarks_df = pd.read_parquet(self.hand_landmarks_path)
        self.hand_angles_df = pd.read_parquet(self.hand_angles_path)
        self.hand_pose_angles_df = pd.read_parquet(self.hand_pose_angles_path)
        self.gloss_video_ids = get_train_val_test_split_per_gloss(self.metadata_path, self.n_glosses)

        self.total_correct = 0
        self.total_tests = 0
        self.total_time = 0.0
        self.accuracy = 0.0
        self.avg_time = 0.0

    def _validate_paths(self):
        for path in [
            self.metadata_path, self.hand_pose_landmarks_path,
            self.hand_angles_path, self.hand_pose_angles_path
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"❌ Missing required file: {path}")

    def get_flat_embedding(self, video_id: str) -> list[list[float]]:
        df = pd.DataFrame()
        if self.feature_option == "hand_landmarks":
            df = self.hand_landmarks_df[self.hand_landmarks_df["video_id"] == video_id].drop(columns=["video_id", "gloss"], errors="ignore")
        elif self.feature_option == "hand+pose_landmarks":
            df = self.hand_pose_landmarks_df[self.hand_pose_landmarks_df["video_id"] == video_id].drop(
                columns=["video_id", "gloss"], errors="ignore")
        elif self.feature_option == "hand_angles":
            df = self.hand_angles_df[self.hand_angles_df["video_id"] == video_id].drop(columns=["video_id", "gloss"], errors="ignore")
        elif self.feature_option == "hand+pose_angles":
            df = self.hand_pose_angles_df[self.hand_pose_angles_df["video_id"] == video_id].drop(columns=["video_id", "gloss"], errors="ignore")
        return df.values.tolist()

    def evaluate(self):
        reference_entries = []

        for gloss, splits in self.gloss_video_ids.items():
            train_ids = splits["train"][:12]

            for vid in train_ids:
                embedding = self.get_flat_embedding(vid)
                if embedding is not None:
                    reference_entries.append({
                        "name": gloss,
                        "embedding": embedding,
                        "distance": 0.0
                    })

        reference_df_full = pd.DataFrame(reference_entries)

        for gloss, splits in self.gloss_video_ids.items():
            # test_ids = random.sample(splits["test"], min(2, len(splits["test"])))
            test_ids = splits["test"][:2]
            test_embeddings = [
                (vid, self.get_flat_embedding(vid)) for vid in test_ids
            ]

            for test_vid, test_embedding in test_embeddings:
                start = time.time()
                result_df = dtw_distances(test_embedding, reference_df_full.copy())
                end = time.time()

                # predict by top k
                k = max(1, ceil(0.1 * len(result_df)))  # At least one
                top_k_glosses = result_df.iloc[:k]["name"]
                counter = Counter(top_k_glosses)
                predicted_gloss, vote_count = counter.most_common(1)[0]
                confidence = vote_count / k

                # # predict by closest match
                # predicted_gloss = result_df.iloc[0]["name"]

                is_correct = predicted_gloss == gloss

                self.total_correct += int(is_correct)
                self.total_tests += 1
                self.total_time += (end - start)

                print(f"🧪 Gloss: {gloss}, Video_id: {test_vid} → Predicted: {predicted_gloss} "
                      f"| Votes: {vote_count}/{k} ({confidence:.0%}) | {'✅' if is_correct else '❌'}")

        self._report()

    def _report(self):
        if self.total_tests == 0:
            print("⚠️ No valid test cases run.")
            return

        self.accuracy = self.total_correct / self.total_tests * 100
        self.avg_time = self.total_time / self.total_tests
        print(f"📊 Accuracy over {self.total_tests} tests: {self.accuracy:.2f}%")
        print(f"⏱️  Avg DTW time/test: {self.avg_time:.4f} seconds\n")


if __name__ == "__main__":
    results = []

    n_list = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
    feature_options = ["hand_landmarks", "hand+pose_landmarks", "hand_angles", "hand+pose_angles"]

    for feature_option in feature_options:
        for n in tqdm(n_list, desc=f"Evaluating {feature_option}", leave=False):
            print(f"\n🔄 Running DTW Evaluation | Features: {feature_option} | Glosses: {n}")
            evaluator = DTWEvaluator(feature_option = feature_option, n_glosses=n)
            evaluator.evaluate()

            # Store results
            results.append({
                "n_glosses": n,
                "feature_option": feature_option,
                "accuracy": round(evaluator.accuracy, 2),
                "avg_dtw_time_sec": round(evaluator.avg_time, 4),
                "num_tests": evaluator.total_tests
            })

    # Display results
    results_df = pd.DataFrame(results)
    plot_dtw_evaluation_results(results_df)
    print("\n📋 Summary of All Runs:")
    print(results_df.to_string(index=False))
    results_df.to_parquet("dtw_evaluation_results.parquet", index=False)



