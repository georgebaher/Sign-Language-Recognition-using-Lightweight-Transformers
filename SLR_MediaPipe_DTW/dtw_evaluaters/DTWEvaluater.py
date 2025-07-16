import os
import time
import numpy as np
from collections import Counter
import pandas as pd
from dotenv import load_dotenv
from SLR_MediaPipe_DTW.utils.dtw import dtw_distances
from SLR_MediaPipe_DTW.utils.analyze_wlasl import get_train_val_test_split_per_gloss

class DTWEvaluator:
    def __init__(self, n_glosses: int = 5, angle_bases: list[str] = None, name: str = "all_angles"):
        load_dotenv()
        self.n_glosses = n_glosses
        self.name = name
        self.angle_bases = angle_bases

        self.metadata_path = os.getenv("WLASL_METADATA_PATH")
        self.hand_pose_angles_path = os.getenv("WLASL100_HAND_POSE_ANGLES_PATH")

        self.hand_pose_angles_df = pd.read_parquet(self.hand_pose_angles_path)
        self.gloss_video_ids = get_train_val_test_split_per_gloss(self.metadata_path, self.n_glosses)

        if self.angle_bases:
            self._filter_by_angle_bases()

        self.total_correct = 0
        self.total_tests = 0
        self.total_time = 0.0
        self.accuracy = 0.0
        self.avg_time = 0.0

    def _filter_by_angle_bases(self):
        cols = [
            col for col in self.hand_pose_angles_df.columns
            if any(col.startswith(base) for base in self.angle_bases)
        ]
        meta = ["video_id", "gloss"]
        self.hand_pose_angles_df = self.hand_pose_angles_df[meta + cols]

    def get_flat_embedding(self, video_id: str) -> list[list[float]]:
        df = self.hand_pose_angles_df[self.hand_pose_angles_df["video_id"] == video_id]
        angle_columns_to_keep = [col for col in self.hand_pose_angles_df.columns if col not in ["video_id", "gloss"]]
        return df[angle_columns_to_keep].values.tolist()

    def evaluate_single(self, test_df: pd.DataFrame):
        """
        Evaluates a single summarized test video against the reference set using DTW.

        :param test_df: A DataFrame of hand+pose angles
        :return: Dict with predicted gloss, top_k matches, and distance scores
        """
        if test_df.empty:
            raise ValueError("❌ Test DataFrame is empty.")

        # Extract test feature vector
        angle_columns_to_keep = [col for col in self.hand_pose_angles_df.columns if col not in ["video_id", "gloss"]]
        test_vector = test_df[angle_columns_to_keep].values.tolist()

        # Build or reuse reference entries (if not already set)
        reference_entries = []
        for gloss, splits in self.gloss_video_ids.items():
            for vid in splits["train"]:
                emb = self.get_flat_embedding(vid)
                reference_entries.append({
                    "name": gloss,
                    "embedding": emb,
                    "distance": 0.0
                })

        reference_df = pd.DataFrame(reference_entries)
        result_df = dtw_distances(test_vector, reference_df)

        predicted_gloss = result_df.iloc[0]["name"]

        return {
            "predicted_gloss": predicted_gloss,
            "top_k_matches": result_df.iloc[:5][["name", "distance"]],
        }

    def evaluate(self):
        reference_entries = []

        for gloss, splits in self.gloss_video_ids.items():
            train_ids = splits["train"][:]
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
            test_ids = splits["test"][:]
            test_embeddings = [(vid, self.get_flat_embedding(vid)) for vid in test_ids]

            for test_vid, test_embedding in test_embeddings:
                start = time.time()
                result_df = dtw_distances(test_embedding, reference_df_full.copy())
                end = time.time()

                # # Top-k voting (k = 10% of reference set)
                # k = max(1, ceil(0.1 * len(result_df)))
                # top_k_glosses = result_df.iloc[:k]["name"]
                # counter = Counter(top_k_glosses)
                # predicted_gloss, vote_count = counter.most_common(1)[0]
                # confidence = vote_count / k

                predicted_gloss = result_df.iloc[0]["name"]

                is_correct = predicted_gloss == gloss

                print(f"🧪 Gloss: {gloss}, Video_id: {test_vid} → Predicted: {predicted_gloss}| "
                      f"{'✅' if is_correct else '❌'}")

                self.total_correct += int(is_correct)
                self.total_tests += 1
                self.total_time += (end - start)

                # print(f"🧪 Gloss: {gloss}, Video_id: {test_vid} → Predicted: {predicted_gloss} "
                #       f"| Votes: {vote_count}/{k} ({confidence:.0%}) | {'✅' if is_correct else '❌'}")

        self._report()

    def _report(self):
        if self.total_tests == 0:
            print("⚠️ No valid test cases run.")
            return

        self.accuracy = self.total_correct / self.total_tests * 100
        self.avg_time = self.total_time / self.total_tests
        print(f"📊 Accuracy over {self.total_tests} tests: {self.accuracy:.2f}%")
        print(f"⏱️  Avg DTW time/test: {self.avg_time:.4f} seconds\n")
