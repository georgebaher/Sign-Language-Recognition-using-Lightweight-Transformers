import argparse
from train import train, get_default_args
import math

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
    lower = max(0.0, p_tilde - margin)
    upper = min(1.0, p_tilde + margin)
    return lower, upper

if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    base_args = parser.parse_args([])

    # Common hyperparameters
    base_args.fs = 1
    base_args.n_heads = 8
    base_args.pe = 0
    base_args.optimizer = "SGD"
    base_args.sgd_momentum = 0.9
    base_args.lr = 1e-3
    base_args.epochs = 150
    base_args.batch_size = 32
    base_args.dataset_name = "WLASL100"
    base_args.scheduler_type = "cosine"

    all_features = [
        "HAND_LANDMARKS",
        # "POSE_LANDMARKS",
        "HAND_POSE_LANDMARKS",
        "HAND_ANGLES",
        # "POSE_ANGLES",
        "HAND_POSE_ANGLES"
    ]
    hidden_dims = [(88, 88), (56, 48), (136, 104), (424, 320), (304, 232), (720, 520)]

    results = []
    test_set_size = 258

    for i, feature in enumerate(all_features):
        for model_name in ["baseline_transformer", "spoter"]:
            args = argparse.Namespace(**vars(base_args))
            args.features = feature
            args.model2use = model_name
            args.hidden_dim = hidden_dims[i][1]
            args.experiment_name = f"{feature.lower()}_{model_name}"

            print(f"\n[INFO] Running experiment: {args.experiment_name}")
            top_acc = train(args)

            # Confidence interval
            ci_low, ci_high = agresti_coull_interval(top_acc, test_set_size)

            results.append({
                "feature": feature,
                "model": model_name,
                "top_acc": top_acc,
                "ci_low": ci_low,
                "ci_high": ci_high
            })

    results_path = "out-logs/model_comparison_results.txt"
    with open(results_path, "w") as f:
        f.write("==== MODEL COMPARISON WITH CONFIDENCE INTERVALS ====\n")
        f.write("{:<25} {:<20} {:<10} {:<10} {:<10}\n".format("Feature", "Model", "Top Acc", "CI Low", "CI High"))
        f.write("-" * 75 + "\n")
        for res in results:
            f.write("{:<25} {:<20} {:<10.2f} {:<10.2f} {:<10.2f}\n".format(
                res["feature"], res["model"], res["top_acc"] * 100, res["ci_low"] * 100, res["ci_high"] * 100
            ))

    print(f"\n[INFO] Saved results with confidence intervals to {results_path}")
