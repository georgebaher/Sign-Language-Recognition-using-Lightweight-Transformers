import argparse
from train import train, get_default_args

if __name__ == '__main__':
    # Use the same argument structure as train.py
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    base_args = parser.parse_args([])  # Don't read from CLI

    # Overwrite base defaults
    base_args.n_heads = 8
    base_args.pe = 0
    base_args.optimizer = "SGD"
    base_args.sgd_momentum = 0.9
    base_args.lr = 1e-3
    base_args.epochs = 100
    base_args.batch_size = 32
    base_args.dataset_name = "WLASL100"
    base_args.scheduler_type = "cosine"

    all_features = [
        "HAND_LANDMARKS",
        "POSE_LANDMARKS",
        "HAND_POSE_LANDMARKS",
        "HAND_ANGLES",
        "POSE_ANGLES",
        "HAND_POSE_ANGLES"
    ]
    hidden_dims = [(88, 88), (56, 48), (136, 104), (424, 320), (304, 232), (720, 520)]  # fs0, fs1

    results = []
    all_fs = [0, 1]

    for i, feature in enumerate(all_features):
        for j, fs in enumerate(all_fs):
            args = argparse.Namespace(**vars(base_args))  # deep copy base args
            args.features = feature
            args.fs = fs
            args.experiment_name = f"{feature.lower()}_fs{fs}"
            args.hidden_dim = hidden_dims[i][j]

            print(f"\n[INFO] Running experiment: {args.experiment_name}")
            top_acc = train(args)
            results.append({
                "feature": feature,
                "fs": fs,
                "top_acc": top_acc
            })

    # Save results to a text file
    results_path = "out-logs/fs_experiment/fs_experiment_results.txt"
    with open(results_path, "w") as f:
        f.write("==== EXPERIMENT COMPARISON ====\n")
        f.write("{:<25} {:<5} {:<10}\n".format("Feature", "FS", "Top Accuracy"))
        f.write("-" * 45 + "\n")
        for res in results:
            f.write("{:<25} {:<5} {:.2f}\n".format(res["feature"], res["fs"], res["top_acc"] * 100))

    print(f"\n[INFO] Saved experiment results to {results_path}")
    print(f"\n[INFO] Done.")