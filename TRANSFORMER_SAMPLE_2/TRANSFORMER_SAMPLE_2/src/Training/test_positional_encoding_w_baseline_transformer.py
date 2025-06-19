import argparse
from train import train, get_default_args

if __name__ == '__main__':
    # Use the same argument structure as train.py
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    base_args = parser.parse_args([])  # Don't read from CLI

    # Overwrite base defaults
    base_args.fs = 1
    base_args.n_heads = 8
    base_args.pe = 1
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
        # "HAND_POSE_LANDMARKS",
        # "HAND_ANGLES",
        # "POSE_ANGLES",
        # "HAND_POSE_ANGLES"
    ]
    hidden_dims = [
        (88, 88), (56, 48), (136, 104),
        (424, 320), (304, 232), (720, 520)  # (fs0, fs1)
    ]

    results = []

    for i, feature in enumerate(all_features):
        args = argparse.Namespace(**vars(base_args))  # Clone base_args
        args.features = feature
        args.hidden_dim = hidden_dims[i][1]  # Use fs=1 dim
        args.experiment_name = f"{feature.lower()}_pe1"

        print(f"\n[INFO] Running experiment: {args.experiment_name}")
        top_acc = train(args)
        results.append({
            "feature": feature,
            "pe": 1,
            "top_acc": top_acc
        })

    results_path = "out-logs/pe_experiment_results.txt"
    with open(results_path, "w") as f:
        f.write("==== EXPERIMENT COMPARISON WITH PE ====\n")
        f.write("{:<25} {:<10}\n".format("Feature", "Top Acc (%)"))
        f.write("-" * 40 + "\n")
        for res in results:
            f.write("{:<25} {:<10.2f}\n".format(res["feature"], res["top_acc"] * 100))

    print(f"\n[INFO] Saved experiment results to {results_path}")
    print("[INFO] Done.")
