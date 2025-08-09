import argparse
import sys

sys.path.append('')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../TRANSFORMER_SAMPLE_2/')
import os
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from src.utils.args_utils import *
from src.Training.models.BaselineTransformerClassification import BaselineTransformerClassification
from src.Training.models.SPOTER import SPOTERTransformer
from src.Training.models.EncoderOnlyTransformer import EncoderOnly
from src.Training.models.LSTM import LSTMClassifier
from src.Training.trainingUtils.utils import evaluate_batch
from src.Training.dataloader.features_dataloader import WLASLParquetDataset
from src.Training.train import get_default_args
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def get_model(args):
    if args.model2use == 'baseline_transformer':
        return BaselineTransformerClassification(num_classes=args.num_classes,
                                                 hidden_dim=args.hidden_dim,
                                                 n_heads=args.n_heads,
                                                 w_pe=args.pe)
    elif args.model2use == 'spoter':
        return SPOTERTransformer(num_classes=args.num_classes,
                                 hidden_dim=args.hidden_dim,
                                 n_heads=args.n_heads,
                                 w_pe=args.pe)
    elif args.model2use == 'encoder':
        return EncoderOnly(
            num_classes=args.num_classes,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            w_pe=args.pe
        )
    elif args.model2use == 'lstm':
        return LSTMClassifier(
            input_dim=args.hidden_dim,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
            num_layers=args.num_layers,
        )
    else:
        raise ValueError(f"Unrecognized model2use: {args.model2use}")


def load_test_data(args):
    feature_parquet_map = {
        "HAND_LANDMARKS": "WLASL100_HAND_LANDMARKS_PATH",
        "POSE_LANDMARKS": "WLASL100_POSE_LANDMARKS_PATH",
        "HAND_POSE_LANDMARKS": "WLASL100_HAND_POSE_LANDMARKS_PATH",
        "HAND_ANGLES": "WLASL100_HAND_ANGLES_PATH",
        "POSE_ANGLES": "WLASL100_POSE_ANGLES_PATH",
        "HAND_POSE_ANGLES": "WLASL100_HAND_POSE_ANGLES_PATH",
    }
    parquet_env_var = feature_parquet_map.get(args.features.upper())
    parquet_path = os.getenv(parquet_env_var)
    metadata_path = os.getenv("WLASL_METADATA_PATH")

    return WLASLParquetDataset(body_features_parquet_path=parquet_path,
                               metadata_json_path=metadata_path,
                               split='test',
                               transform=args.transform,
                               features=args.features,
                               fs=args.fs,
                               )


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")



def group_glosses_by_accuracy(stats, idx2gloss, precision=2):
    acc_to_glosses = defaultdict(list)
    for class_id, acc in stats.items():
        gloss = idx2gloss.get(class_id, f"gloss_{class_id}")
        rounded_acc = round(acc, precision)
        acc_to_glosses[rounded_acc].append(gloss)
    return dict(acc_to_glosses)

def plot_grouped_accuracy_distribution(stats_dict, save_path=None, title="Grouped Per-Class Accuracy Distribution"):
    accuracies = list(stats_dict.values())
    bins = np.arange(0, 1.1, 0.1)  # [0.0, 0.1, ..., 1.0]
    bin_labels = [f"{int(b * 100)}–{int((b + 0.1) * 100)}%" for b in bins[:-1]]

    hist, _ = np.histogram(accuracies, bins=bins)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(bin_labels, hist, color="mediumseagreen")

    # Annotate values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{height}", ha="center", va="bottom")

    plt.xlabel("Accuracy Range")
    plt.ylabel("Number of Classes")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved plot to {save_path}")
    else:
        plt.show()

def test(args):
    load_dotenv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join("out-checkpoints", args.experiment_name,
                                   f"checkpoint_{args.checkpoint_tag}{args.checkpoint_index}.pth")

    print(f"[INFO] Loading model from: {checkpoint_path}")
    model = torch.load(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)

    count_params(model)

    test_dataset = load_test_data(args)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)
    loss_fn = torch.nn.CrossEntropyLoss()

    print("[INFO] Evaluating...")
    _, avg_acc, stats = evaluate_batch(model, loss_fn, test_loader, device, print_stats=True)
    idx2gloss = test_dataset.idx2gloss  # ← assuming you're using WLASLParquetDataset
    acc_gloss_dict = group_glosses_by_accuracy(stats, idx2gloss)

    # Optional: sort by accuracy
    for acc in sorted(acc_gloss_dict):
        print(f"{acc:.2f}: {acc_gloss_dict[acc]}")

    print(f"[RESULT] Test Accuracy: {avg_acc:.2%}")
    plot_grouped_accuracy_distribution(stats, title=f"Per-Class Accuracy ({args.experiment_name})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    parser.add_argument("--checkpoint_tag", type=str, default="v", help="Checkpoint prefix tag: 'v' or 't'")
    parser.add_argument("--checkpoint_index", type=int, default=0, help="Checkpoint index to test")
    args = parser.parse_args()
    test(args)
    print("Done")


#     python test_checkpointed_model.py --experiment_name encoder_hand_landmarks --hidden_dim 88  --model2use encoder  --optimizer adam --fs 0  --scheduler_type cosine --dataset_name WLASL100 --top_features HAND_LANDMARKS
# --batch_size 32 --num_classes 100 --n_heads 8 --pe 0 --checkpoint_tag t --checkpoint_index 34
