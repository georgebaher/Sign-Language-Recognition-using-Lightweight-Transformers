import argparse
import os
import math
from train import train, get_default_args

def agresti_coull_interval(p_hat, n, z=1.96):
    """
    Agresti-Coull Confidence Interval for a Proportion
    p_hat: observed proportion (accuracy)
    n: sample size
    z: z-score for desired confidence (1.96 for 95%)
    """
    n_tilde = n + 4
    X_tilde = p_hat * n + 2
    p_tilde = X_tilde / n_tilde
    margin = z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    return margin

if __name__ == "__main__":
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    base_args = parser.parse_args([])

    # Overwrite base defaults
    base_args.n_heads = 8
    base_args.n_layers = 6
    base_args.model2use = 'encoder'
    base_args.pe = 0
    base_args.optimizer = "SGD"
    base_args.sgd_momentum = 0.9
    base_args.dataset_name = "WLASL100"
    base_args.include_blendshapes = 0
    base_args.fs = 1
    base_args.num_classes = 100
    base_args.epochs = 100
    base_args.lr = 1e-3
    base_args.batch_size = 32
    base_args.scheduler_type = 'cosine'

    all_features = [
        "HAND_LANDMARKS",
        "POSE_LANDMARKS",
        "HAND_POSE_LANDMARKS",
        "HAND_ANGLES",
        "POSE_ANGLES",
        "HAND_POSE_ANGLES",
    ]

    original_dims = [
        (84, 84),
        (50, 42),
        (134, 126),
        (420, 313),
        (300, 228),
        (720, 541),
    ]
    hidden_dims = [
        (88, 88),
        (56, 48),
        (136, 128),
        (424, 320),
        (304, 232),
        (720, 544),
    ]

    results = []
    test_set_size = 258

    for feature_index, feature in enumerate(all_features):
        for trunc in [0, 1]:
            args = argparse.Namespace(**vars(base_args))
            args.experiment_name = f"encoder_{feature.lower()}_fs{1}_{'trunc' if trunc else 'pad'}"
            args.hidden_dim = hidden_dims[feature_index][1]
            args.features = feature

            args.feature_truncation = trunc
            args.model2use = "encoder"

            print(f"\n[INFO] Running experiment: {args.experiment_name}")
            top_acc, m_f1, w_f1, total_params, elapsed_time = train(args)

            # CI
            acc_margin = agresti_coull_interval(top_acc, test_set_size)
            m_f1_margin = agresti_coull_interval(m_f1, test_set_size)
            w_f1_margin = agresti_coull_interval(w_f1, test_set_size)

            results.append({
                "feature": feature,
                "fs": fs,
                "method": "truncation" if trunc else "padding",
                "top_acc": top_acc,
                "acc_margin": acc_margin,
                "m_f1": m_f1,
                "m_f1_margin": m_f1_margin,
                "w_f1": w_f1,
                "w_f1_margin": w_f1_margin,
                "hidden_dim": hidden_dims[feature_index][fs],
                "padded_features": hidden_dims[feature_index][fs] - original_dims[feature_index][fs],
                "total_params": total_params,
                "elapsed_time": elapsed_time
            })

    # Write results
    results_path = "out-logs/__padding_vs_truncating/_results.txt"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    with open(results_path, "w") as f:
        f.write("==== PADDING VS TRUNCATION COMPARISON ====\n")
        f.write("{:<25} {:<5} {:<10} {:<20} {:<20} {:<20} {:<10} {:<12} {:<15} {:<12}\n".format(
            "Feature", "FS", "Method", "Top Acc (±)", "Macro F1 (±)", "Weighted F1 (±)",
            "Padded", "HiddenDim", "Total Params", "Time"
        ))
        f.write("-" * 180 + "\n")
        for res in results:
            acc_str = f"{res['top_acc']*100:.2f} ± {res['acc_margin']*100:.2f}"
            m_f1_str = f"{res['m_f1']*100:.2f} ± {res['m_f1_margin']*100:.2f}"
            w_f1_str = f"{res['w_f1']*100:.2f} ± {res['w_f1_margin']*100:.2f}"
            param_str = f"{res['total_params']:,}"
            time_str = f"{res['elapsed_time']:.1f}s"

            f.write("{:<25} {:<5} {:<10} {:<20} {:<20} {:<20} {:<10} {:<12} {:<15} {:<12}\n".format(
                res["feature"],
                res["fs"],
                res["method"],
                acc_str,
                m_f1_str,
                w_f1_str,
                res["padded_features"],
                res["hidden_dim"],
                param_str,
                time_str
            ))

    print(f"\n[INFO] Saved comparison results to {results_path}")
