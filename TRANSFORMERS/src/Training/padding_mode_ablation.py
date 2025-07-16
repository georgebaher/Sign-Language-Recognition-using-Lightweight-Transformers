# -*- coding: utf-8 -*-
import argparse
import math
import os
from train import train, get_default_args

def agresti_coull_interval(p_hat, n, z=1.96):
    """Confidence interval for accuracy/f1 using Agresti-Coull method."""
    n_tilde = n + 4
    X_tilde = p_hat * n + 2
    p_tilde = X_tilde / n_tilde
    margin = z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    return margin

def run_experiments():
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    base_args = parser.parse_args([])

    # Fixed experiment parameters
    base_args.model2use = "encoder"
    base_args.features = "POSE_ANGLES"  # "HAND_LANDMARKS", "POSE_LANDMARKS", "HAND_POSE_LANDMARKS", "HAND_ANGLES", "POSE_ANGLES", "HAND_POSE_ANGLES"
    base_args.fs = 1
    base_args.include_blendshapes = 0
    base_args.num_classes = 100
    base_args.n_heads = 8
    base_args.n_layers = 6
    base_args.pe = 0
    base_args.optimizer = "SGD"
    base_args.sgd_momentum = 0.9
    base_args.scheduler_type = "cosine"
    base_args.epochs = 100
    base_args.batch_size = 32
    base_args.lr = 1e-3

    test_set_size = 258
    padding_modes = ["truncate", "sentinel", "repeat"]
    results = []

    for mode in padding_modes:
        args = argparse.Namespace(**vars(base_args))  # Deep copy
        args.feature_padding_mode = mode
        args.experiment_name = f"{args.model2use}_{args.features.lower()}_fs1_pad_{mode}"

        print(f"\n Running: {args.experiment_name}")
        acc, m_f1, w_f1, total_params, elapsed_time, dim = train(args)

        results.append({
            "padding": mode,
            "top_acc": acc,
            "acc_margin": agresti_coull_interval(acc, test_set_size),
            "m_f1": m_f1,
            "m_f1_margin": agresti_coull_interval(m_f1, test_set_size),
            "w_f1": w_f1,
            "w_f1_margin": agresti_coull_interval(w_f1, test_set_size),
            "hidden_dim": dim,
            "total_params": total_params,
            "elapsed_time": elapsed_time,
        })

    log_path = "out-logs/__feature_padding_ablation/pose_angles/__padding_ablation_results_pose_angles.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w") as f:
        f.write("==== FEATURE PADDING ABLATION ====\n")
        f.write("{:<10} {:<20} {:<20} {:<20} {:<12} {:<15} {:<10}\n".format(
            "Padding", "Top Acc (±)", "Macro F1 (±)", "Weighted F1 (±)",
            "Dim", "Total Params", "Time"
        ))
        f.write("-" * 120 + "\n")
        for r in results:
            acc_str = f"{r['top_acc']*100:.2f} ± {r['acc_margin']*100:.2f}"
            m_f1_str = f"{r['m_f1']*100:.2f} ± {r['m_f1_margin']*100:.2f}"
            w_f1_str = f"{r['w_f1']*100:.2f} ± {r['w_f1_margin']*100:.2f}"
            params_str = f"{r['total_params']:,}"
            time_str = f"{r['elapsed_time']:.1f}s"
            f.write("{:<10} {:<20} {:<20} {:<20} {:<12} {:<15} {:<10}\n".format(
                r["padding"], acc_str, m_f1_str, w_f1_str, r["hidden_dim"],
                params_str, time_str
            ))

    print(f"\n Saved ablation results to: {log_path}")

if __name__ == "__main__":
    run_experiments()
