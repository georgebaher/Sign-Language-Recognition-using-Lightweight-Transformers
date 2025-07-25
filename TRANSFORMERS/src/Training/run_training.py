# run_training.py
#
# A fully generalizable and clean script for training sign language recognition models
# using the SignLanguageFeaturesDataset.

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
import torch.optim as optim
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from sklearn.metrics import f1_score

# --- Add the project's root directory to the Python path ---
# This allows the script to find the 'src' and 'dataset' modules.
# It assumes the script is in a directory like 'project_root/src/Training/'.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.Training.dataset.SLFeaturesDataset import SignLanguageFeaturesDataset
from src.Training.trainingUtils.utils import train_epoch_batch, evaluate_batch
from src.Training.models.BaselineTransformerClassification import BaselineTransformerClassification
from src.Training.models.SPOTER import SPOTERTransformer
from src.Training.models.LSTM import LSTMClassifier
from src.Training.models.EncoderOnlyTransformer import SPOTEREncoderOnly


def setup_logging(log_dir, experiment_name):
    """Configures a logger to write to a file and the console."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{experiment_name}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]: logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_path, mode='w')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def get_args_parser():
    """Defines command-line arguments for the training script."""
    parser = argparse.ArgumentParser("Sign Language Transformer Training", add_help=False)
    parser.add_argument("--experiment_name", type=str, default='SLR_Experiment')
    parser.add_argument("--seed", type=int, default=379)
    parser.add_argument("--dataset_name", type=str, default="wlasl", choices=["wlasl", "avasag"])
    parser.add_argument("--feature_extraction_model", type=str, default="vitpose", choices=["vitpose", "mediapipe"])
    parser.add_argument("--n_glosses", type=int, default=100)
    parser.add_argument("--features", nargs='+', required=True)
    parser.add_argument("--fs", type=int, default=0)
    parser.add_argument("--feature_padding_mode", type=str, default="sentinel",
                        choices=["sentinel", "repeat", "truncate"])
    parser.add_argument("--model", type=str, default="baseline_transformer",
                        choices=["baseline_transformer", "spoter", "lstm", "encoder"])
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--pe", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--sgd_momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["warmup_linear", "warmup_cosine", "warmup_constant", "none"])
    parser.add_argument("--clip_gradients", type=float, default=0.0)
    parser.add_argument("--save_checkpoints", action='store_true')
    parser.add_argument("--log_freq", type=int, default=1)
    parser.add_argument("--plot_stats", action='store_true')
    return parser


def fix_randomisation(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(args):
    start_time = time.time()
    fix_randomisation(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = setup_logging("out-logs", args.experiment_name)

    logger.info("--- Starting Experiment ---")
    logger.info(f"Experiment Name: {args.experiment_name}")
    logger.info(f"Using device: {device}")
    logger.info("Arguments:")
    for key, value in sorted(vars(args).items()): logger.info(f"  > {key}: {value}")

    load_dotenv()
    base_path = os.getenv(f"{args.dataset_name.upper()}_{args.feature_extraction_model.upper()}_BASE_PATH")
    fs_dir = os.getenv(f"{args.dataset_name.upper()}_{args.feature_extraction_model.upper()}_TOP_FEATURES_DIR")

    dataloader_args = {
        "metadata_json_path": os.getenv(f"{args.dataset_name.upper()}_METADATA_PATH"),
        "features": args.features, "n_glosses": args.n_glosses,
        "fs": args.fs, "feature_selection_dir": fs_dir if args.fs else None,
        "n_heads": args.n_heads, "feature_padding_mode": args.feature_padding_mode,
        "pose_landmark_path": os.path.join(base_path, "POSE_LANDMARKS.parquet"),
        "hand_landmark_path": os.path.join(base_path, "HAND_LANDMARKS.parquet"),
        "face_landmark_path": os.path.join(base_path, "FACE_LANDMARKS.parquet"),
        "pose_angle_path": os.path.join(base_path, "POSE_ANGLES.parquet"),
        "hand_angle_path": os.path.join(base_path, "HAND_ANGLES.parquet"),
        "face_blendshape_path": os.path.join(base_path, "FACE_BLENDSHAPES.parquet"),
    }

    logger.info("\n--- Loading Datasets ---")
    train_set = SignLanguageFeaturesDataset(**dataloader_args, split="train")
    val_set = SignLanguageFeaturesDataset(**dataloader_args, split="val")
    test_set = SignLanguageFeaturesDataset(**dataloader_args, split="test")

    logger.info(f"Dataset Sizes: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    logger.info(f"Number of Classes: {len(train_set.gloss2idx)}")
    logger.info(f"Feature Dimension: {train_set.feature_dim} | Max Sequence Length: {train_set.max_len}")

    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=args.batch_size)

    input_dim, hidden_dim, num_classes = train_set.feature_dim, train_set.feature_dim, len(train_set.gloss2idx)

    logger.info(f"\n--- Building Model: {args.model} ---")
    if args.model == 'baseline_transformer':
        model = BaselineTransformerClassification(num_classes=num_classes, hidden_dim=hidden_dim,
                                                  num_layers=args.n_layers, n_heads=args.n_heads, w_pe=bool(args.pe))
    elif args.model == 'spoter':
        model = SPOTERTransformer(num_classes=num_classes, hidden_dim=hidden_dim, num_layers=args.n_layers,
                                  n_heads=args.n_heads, w_pe=bool(args.pe))
    elif args.model == 'lstm':
        model = LSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    elif args.model == 'encoder':
        model = SPOTEREncoderOnly(num_classes=num_classes, hidden_dim=hidden_dim, num_layers=args.n_layers,
                                  n_heads=args.n_heads, w_pe=bool(args.pe))
    # Send model to GPU
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Trainable Parameters: {total_params:,}")
    if device.type == 'cuda': logger.info(
        f"Initial GPU Memory used: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

    loss_fn = nn.CrossEntropyLoss()
    optimizer_map = {"sgd": optim.SGD(model.parameters(), lr=args.lr, momentum=args.sgd_momentum),
                     "adam": optim.Adam(model.parameters(), lr=args.lr),
                     "adamw": optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)}
    optimizer = optimizer_map.get(args.optimizer.lower())
    if optimizer is None: raise ValueError(f"Invalid optimizer name: {args.optimizer}")

    scheduler = None
    if args.scheduler != "none":
        total_steps = len(train_loader) * args.epochs
        warmup_steps = int(0.1 * total_steps)
        scheduler_map = {"cosine": get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                                   num_training_steps=total_steps),
                         "linear": get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                                   num_training_steps=total_steps),
                         "constant": get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)}
        scheduler = scheduler_map.get(args.scheduler.lower())

    train_losses, train_accs, val_losses, val_accs, lr_progress = [], [], [], [], []
    best_val_acc = 0.0
    checkpoint_dir = Path("out-checkpoints") / args.experiment_name
    if args.save_checkpoints: checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n--- Starting Training for {args.epochs} epochs ---")
    for epoch in range(args.epochs):
        avg_train_loss, avg_train_acc = train_epoch_batch(model, train_loader, loss_fn, optimizer, device, scheduler,
                                                          args.clip_gradients)
        train_losses.append(avg_train_loss);
        train_accs.append(avg_train_acc)
        avg_val_loss, avg_val_acc, _ = evaluate_batch(model, loss_fn, val_loader, device)
        val_losses.append(avg_val_loss);
        val_accs.append(avg_val_acc)

        if (epoch + 1) % args.log_freq == 0:
            logger.info(
                f"[Epoch {epoch + 1:03d}] Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")

        if args.save_checkpoints and avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_model_path = checkpoint_dir / "best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  -> [Checkpoint] Saved new best model (Val Acc: {best_val_acc:.4f})")

        lr_progress.append(optimizer.param_groups[0]['lr'])

    if args.save_checkpoints:
        final_model_path = checkpoint_dir / "final_model.pth"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"--- Saved final model to {final_model_path} ---")

    logger.info("\n--- Starting Final Evaluation on Test Set ---")
    best_model_path = checkpoint_dir / "best_model.pth"
    if args.save_checkpoints and best_model_path.exists():
        logger.info(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    _, test_acc, (y_true, y_pred) = evaluate_batch(model, loss_fn, test_loader, device, return_preds=True)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    end_time = time.time()
    total_time = end_time - start_time
    logger.info("\n" + "=" * 30 + " EXPERIMENT SUMMARY " + "=" * 30)
    logger.info(f"Experiment Name: {args.experiment_name}")
    logger.info(f"Final Test Accuracy: {test_acc:.4f}")
    logger.info(f"Final Macro F1 Score: {macro_f1:.4f}")
    logger.info(f"Final Weighted F1 Score: {weighted_f1:.4f}")
    logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
    logger.info(f"Total Trainable Parameters: {total_params:,}")
    logger.info(f"Total Runtime: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    logger.info("=" * 82)

    if args.plot_stats:
        plot_dir = Path("out-img") / args.experiment_name
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 5), ncols=2)
        fig.suptitle(f"Test Acc: {test_acc:.3f} | Wighted F1: {weighted_f1:.3f}")
        ax[0].plot(train_losses, label="Training loss")
        ax[0].plot(val_losses, label="Validation loss")
        ax[0].plot(train_accs, label="Training accuracy")
        ax[0].plot(val_accs, label="Validation accuracy")
        ax[0].set_xlabel("Epoch");
        ax[0].set_ylabel("Value");
        ax[0].set_title("Training Metrics");
        ax[0].legend();
        ax[0].grid(True)
        ax[1].plot(lr_progress, label="Learning Rate")
        ax[1].set_xlabel("Epoch");
        ax[1].set_ylabel("LR");
        ax[1].set_title("Learning Rate Schedule");
        ax[1].legend();
        ax[1].grid(True)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(plot_dir / "training_stats.png")
        logger.info(f"Saved training plot to {plot_dir / 'training_stats.png'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Sign Language Transformer Training", parents=[get_args_parser()])
    args = parser.parse_args()
    train(args)