# -*- coding: utf-8 -*-
import argparse
import math
import os
from train import train, get_default_args


def agresti_coull_interval(p_hat, n, z=1.96):
    """
    Agresti-Coull Confidence Interval for a Proportion
    p_hat: observed proportion (accuracy)
    n: sample size
    z: z-score for desired confidence (1.96 for 95%)
    """
    if not 0 <= p_hat <= 1:
        return 0  # Return 0 margin if p_hat is invalid
    n_tilde = n + z ** 2
    p_tilde = (p_hat * n + (z ** 2) / 2) / n_tilde
    margin = z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    return margin


if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    base_args = parser.parse_args([])

    # --- Overwrite base defaults for the ablation study ---
    base_args.n_heads = 8
    base_args.n_layers = 6
    base_args.model2use = 'encoder'
    base_args.pe = 0
    base_args.optimizer = "SGD"
    base_args.sgd_momentum = 0.9
    base_args.dataset_name = "WLASL100"
    base_args.num_classes = 100
    base_args.feature_padding_mode = "sentinel"
    base_args.epochs = 100
    base_args.lr = 1e-3
    base_args.batch_size = 32
    base_args.scheduler_type = 'cosine'

    # --- Experiment Configuration ---
    body_features = [
        "HAND_LANDMARKS",
        "POSE_LANDMARKS",
        "HAND_POSE_LANDMARKS",
        "HAND_ANGLES",
        "POSE_ANGLES",
        "HAND_POSE_ANGLES"
    ]
    # Original dimensions [without FS, with FS] for body features
    original_body_dims = [
        (84, 84),
        (50, 42),
        (134, 126),
        (420, 313),
        (300, 228),
        (720, 541)
    ]
    BLENDSHAPE_DIM = 52

    results = []
    # Only run with feature selection enabled
    all_fs_modes = [1]
    all_bs_modes = [0, 1]
    test_set_size = 258

    # === MAIN ABLATION LOOP FOR BODY FEATURES (+/- BLENDSHAPES) ===
    for i, feature in enumerate(body_features):
        for fs in all_fs_modes:
            for bs in all_bs_modes:
                args = argparse.Namespace(**vars(base_args))

                args.experiment_name = f"{args.model2use}_{feature.lower()}_fs{fs}_bs{bs}"
                args.features = feature
                args.fs = fs
                args.include_blendshapes = bs

                # Use `fs` directly as the index for robustness
                current_original_dim = original_body_dims[i][fs]
                if bs == 1:
                    current_original_dim += BLENDSHAPE_DIM

                print(f"\n[INFO] Running experiment: {args.experiment_name}")
                top_acc, m_f1, w_f1, total_params, elapsed_time, hidden_dim = train(args)

                acc_margin = agresti_coull_interval(top_acc, test_set_size)
                m_f1_margin = agresti_coull_interval(m_f1, test_set_size)
                w_f1_margin = agresti_coull_interval(w_f1, test_set_size)

                results.append({
                    "feature": feature,
                    "fs": fs,
                    "include_blendshapes": bs,
                    "top_acc": top_acc,
                    "acc_margin": acc_margin,
                    "m_f1": m_f1,
                    "m_f1_margin": m_f1_margin,
                    "w_f1": w_f1,
                    "w_f1_margin": w_f1_margin,
                    "hidden_dim": hidden_dim,
                    "padded_features": hidden_dim - current_original_dim,
                    "total_params": total_params,
                    "elapsed_time": elapsed_time
                })

    # === SEPARATE RUN FOR FACIAL BLENDSHAPES ONLY (Unaffected by the change) ===
    print("\n" + "=" * 50)
    print("[INFO] Running special case: FACIAL_BLENDSHAPES only")
    print("=" * 50)

    args = argparse.Namespace(**vars(base_args))
    args.features = "FACIAL_BLENDSHAPES"
    args.fs = 0  # Not applicable
    args.include_blendshapes = 0  # Not applicable
    args.experiment_name = f"{args.model2use}_facial_blendshapes_only"

    current_original_dim = BLENDSHAPE_DIM

    top_acc, m_f1, w_f1, total_params, elapsed_time, hidden_dim = train(args)
    acc_margin = agresti_coull_interval(top_acc, test_set_size)
    m_f1_margin = agresti_coull_interval(m_f1, test_set_size)
    w_f1_margin = agresti_coull_interval(w_f1, test_set_size)

    results.append({
        "feature": "FACIAL_BLENDSHAPES",
        "fs": "N/A",
        "include_blendshapes": "N/A",
        "top_acc": top_acc,
        "acc_margin": acc_margin,
        "m_f1": m_f1,
        "m_f1_margin": m_f1_margin,
        "w_f1": w_f1,
        "w_f1_margin": w_f1_margin,
        "hidden_dim": hidden_dim,
        "padded_features": hidden_dim - current_original_dim,
        "total_params": total_params,
        "elapsed_time": elapsed_time
    })

    # --- WRITING RESULTS TO FILE ---
    results_path = "out-logs/__blendshapes_ablation/_results_fs1_with_blendshapes.txt"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    with open(results_path, "w") as f:
        f.write("==== FEATURE (FS=1) & BLENDSHAPE ABLATION WITH CONFIDENCE INTERVALS ====\n")
        f.write("{:<25} {:<5} {:<5} {:<20} {:<20} {:<20} {:<10} {:<12} {:<15} {:<12}\n".format(
            "Feature", "FS", "+BS", "Top Acc (±)", "Macro F1 (±)", "Weighted F1 (±)",
            "Padded", "HiddenDim", "Total Params", "Time"
        ))
        f.write("-" * 170 + "\n")
        for res in results:
            acc_str = f"{res['top_acc'] * 100:.2f} ± {res['acc_margin'] * 100:.2f}"
            m_f1_str = f"{res['m_f1'] * 100:.2f} ± {res['m_f1_margin'] * 100:.2f}"
            w_f1_str = f"{res['w_f1'] * 100:.2f} ± {res['w_f1_margin'] * 100:.2f}"
            padded = res["padded_features"]
            hidden_dim = res["hidden_dim"]
            params_str = f"{res['total_params']:,}"
            time_str = f"{res['elapsed_time']:.1f}s"

            f.write("{:<25} {:<5} {:<5} {:<20} {:<20} {:<20} {:<10} {:<12} {:<15} {:<12}\n".format(
                res["feature"], str(res["fs"]), str(res["include_blendshapes"]), acc_str, m_f1_str, w_f1_str,
                padded, hidden_dim, params_str, time_str
            ))

    print(f"\n[INFO] Saved FS=1 ablation results to {results_path}")