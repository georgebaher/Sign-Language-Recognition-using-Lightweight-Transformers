from SLR_MediaPipe_DTW.dtw_evaluaters.DTWEvaluater import DTWEvaluator
import pandas as pd
from SLR_MediaPipe_DTW.utils.plt_eval_metrics import plot_dtw_evaluation_results_3
from Feature_Extraction.feature_selection.features.filtered_hand_angles_features import TOP_ANGLE_BASES as filtered_hand_angles_features
from Feature_Extraction.feature_selection.features.hand_angles_features import TOP_ANGLE_BASES as hand_angles_features
from tqdm import tqdm

if __name__ == "__main__":
    angle_sets = {
        "fs_396": hand_angles_features,
        "fs_filtered_356": filtered_hand_angles_features,
    }

    n_list = [2, 3, 4, 5, 10, 15, 20, 25, 30]
    results = []

    for name, angle_bases in angle_sets.items():
        for n in tqdm(n_list, desc=f"Evaluating {name}", leave=False):
            print(f"\n🔄 Running DTW | {name} | Glosses: {n}")
            evaluator = DTWEvaluator(n_glosses=n, angle_bases=angle_bases, name=name)
            evaluator.evaluate()

            results.append({
                "n_glosses": n,
                "accuracy": round(evaluator.accuracy, 2),
                "avg_dtw_time_sec": round(evaluator.avg_time, 4),
                "num_tests": evaluator.total_tests,
                "feature_option": name
            })

    results_df = pd.DataFrame(results)
    results_df.to_parquet("SLR_MediaPipe_DTW/dtw_eval_hand_angles_fs_vs_filtered_fs_top_1_match_up_to_30_glosses", index=False)
    plot_dtw_evaluation_results_3(results_df)
    print("\n📋 Summary of All Runs:")
    print(results_df.to_string(index=False))

    # results_df = pd.read_parquet(r'SLR_MediaPipe_DTW/dtw_eval_hand_angles_fs_vs_filtered_fs_top_1_match_up_to_30_glosses.parquet')
    # plot_dtw_evaluation_results_3(results_df)
