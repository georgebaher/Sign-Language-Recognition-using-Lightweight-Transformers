import argparse
from train import train, get_default_args

if __name__ == '__main__':
    # Use the same argument structure as train.py
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    base_args = parser.parse_args([])  # don't read from CLI

    # Overwrite defaults
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
    base_args.model2use = "spoter"  # Make sure your train.py supports this model

    # All feature types and their hidden dims (fs0, fs1)
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
        args = argparse.Namespace(**vars(base_args))  # clone base_args
        args.features = feature
        args.hidden_dim = hidden_dims[i][1]  # fs=1
        args.experiment_name = f"spoter_{feature.lower()}"

        print(f"\n[INFO] Running experiment: {args.experiment_name}")
        top_acc = train(args)
        results.append({"feature": feature, "top_acc": top_acc})

    results_path = "out-logs/spoter_hand_landmarks_150_epochs/spoter_experiment_results.txt"
    with open(results_path, "w") as f:
        f.write("==== SPOTER EXPERIMENT COMPARISON ====\n")
        f.write("{:<25}  {:<10}\n".format("Feature", "Top Accuracy"))
        f.write("-" * 55 + "\n")
        for res in results:
            f.write("{:<25} {:.2f}\n".format(res["feature"], res["top_acc"] * 100))

    print(f"\n[INFO] Saved experiment results to {results_path}")
    print(f"[INFO] Done.")
