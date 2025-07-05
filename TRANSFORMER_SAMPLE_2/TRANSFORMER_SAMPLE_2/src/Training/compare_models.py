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
    return margin

if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    base_args = parser.parse_args([])

    # Overwrite base defaults
    base_args.n_heads = 8
    base_args.n_layers = 6
    base_args.pe = 0
    base_args.optimizer = "SGD"
    base_args.sgd_momentum = 0.9
    base_args.dataset_name = "WLASL100"
    base_args.features = 'HAND_POSE_LANDMARKS'
    base_args.include_blendshapes = 0
    base_args.fs = 0
    base_args.num_classes = 100
    base_args.epochs = 100
    base_args.lr = 1e-3
    base_args.batch_size = 32
    base_args.scheduler_type = 'cosine'

    results = []
    test_set_size = 258

    for model_name in ["baseline_transformer", "spoter", "encoder", "lstm"]:
        args = argparse.Namespace(**vars(base_args))
        args.model2use = model_name
        args.hidden_dim = 136
        args.experiment_name = f"{model_name}"

        # override specific parameters for LSTM
        if model_name == "lstm":
            args.batch_size = 1
            args.n_layers = 1
        else:
            args.batch_size = 32
            args.n_layers = 6

        print(f"\n[INFO] Running experiment: {args.experiment_name}")
        top_acc, m_f1, w_f1, total_params, elapsed_time = train(args)

        # Confidence intervals
        acc_margin = agresti_coull_interval(top_acc, test_set_size)
        m_f1_margin = agresti_coull_interval(m_f1, test_set_size)
        w_f1_margin = agresti_coull_interval(w_f1, test_set_size)

        results.append({
            "feature": "HAND_POSE_LANDMARKS",
            "model": model_name,
            "top_acc": top_acc,
            "acc_margin": acc_margin,
            "m_f1": m_f1,
            "m_f1_margin": m_f1_margin,
            "w_f1": w_f1,
            "w_f1_margin": w_f1_margin,
            "total_params": total_params,
            "elapsed_time": elapsed_time
        })

    results_path = "out-logs/__models_comparison/_results.txt"
    with open(results_path, "w") as f:
        f.write("==== MODEL COMPARISON WITH CONFIDENCE INTERVALS ====\n")
        f.write("{:<25} {:<20} {:<20} {:<20} {:<20} {:<15} {:<12}\n".format(
            "Feature", "Model", "Top Acc (±)", "Macro F1 (±)", "Weighted F1 (±)", "Total Params", "Elapsed Time"
        ))
        f.write("-" * 145 + "\n")
        for res in results:
            acc_str = f"{res['top_acc']*100:.2f} ± {res['acc_margin']*100:.2f}"
            m_f1_str = f"{res['m_f1']*100:.2f} ± {res['m_f1_margin']*100:.2f}"
            w_f1_str = f"{res['w_f1']*100:.2f} ± {res['w_f1_margin']*100:.2f}"
            param_str = f"{res['total_params']:,}"
            time_str = f"{res['elapsed_time']:.1f}s"

            f.write("{:<25} {:<20} {:<20} {:<20} {:<20} {:<15} {:<12}\n".format(
                res["feature"], res["model"], acc_str, m_f1_str, w_f1_str, param_str, time_str
            ))

    print(f"\n[INFO] Saved results with confidence intervals to {results_path}")
