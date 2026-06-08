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
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns


# --- Add the project's root directory to the Python path ---
# Script lives at <repo_root>/src/run_training.py.
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.dataset.SLFeaturesDataset import SignLanguageFeaturesDataset
from src.utils import train_epoch_batch, evaluate_batch
from src.ci_calculator import calculate_approximate_ci
from src.models.BaselineTransformerClassification import BaselineTransformerClassification
from src.models.SPOTER import SPOTERTransformer
from src.models.LSTM import LSTMClassifier
from src.models.BiLSTM import BiLSTMClassifier
from src.models.EncoderOnlyTransformer import EncoderOnly

from src.models.LateFusionLogitEncoderWeightedSum import LateFusionEncoder
# from src.models.LateFusionLogitEncoderConcat import LateFusionEncoder
from src.models.LateFusionModelUsingPretrainedEncodersWeightedSum import LateFusionPET
# from src.models.LateFusionModelUsingPretrainedEncodersConcat import LateFusionPET

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


def load_expert_encoder(ckpt_path, input_dim, num_classes, hidden_dim, n_heads, num_layers, device):
    expert_model = EncoderOnly(input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim,
                               n_heads=n_heads, num_layers=num_layers)
    state_dict = torch.load(ckpt_path, map_location=device)
    expert_model.load_state_dict(state_dict)
    expert_model.to(device)
    return expert_model


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

    ############################## late fusion encoder specific args ###############################################
    parser.add_argument("--model", type=str, default="baseline_transformer",
                        choices=["baseline_transformer", "spoter", "lstm", "bilstm", "encoder", "latefusion_encoder", "latefusion_pet"])
    parser.add_argument("--hand_input_dim", type=int, default=84)
    parser.add_argument("--pose_input_dim", type=int, default=50)
    parser.add_argument("--face_input_dim", type=int, default=52)
    parser.add_argument("--debug", action="store_true")
    ###############################################################################################################


    ############################## late fusion encoder PET specific args ###############################################
    parser.add_argument("--hand_ckpt_path", type=str, help="Path to the pre-trained hand expert model.")
    parser.add_argument("--pose_ckpt_path", type=str, help="Path to the pre-trained pose expert model.")
    parser.add_argument("--face_ckpt_path", type=str, help="Path to the pre-trained face expert model.")
    #####################################################################################################################

    parser.add_argument("--hidden_dim", type=int, default=256, help="Fixed model hidden dimension (input embedding output, transformer d_model).")
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--pe", type=str, default="sincos", choices=["sincos", "learnable", "none"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                        help="L2 weight-decay for AdamW (decoupled). Ignored if optimizer is not AdamW.")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--sgd_momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "linear", "constant", "none"])
    parser.add_argument("--clip_gradients", type=float, default=0.0)
    parser.add_argument("--save_checkpoints", action='store_true')
    parser.add_argument("--log_freq", type=int, default=1)
    parser.add_argument("--plot_stats", action='store_true')

    # --- NEW: Evaluation-Only Arguments ---
    eval_group = parser.add_argument_group('Evaluation-Only Mode')
    eval_group.add_argument("--evaluate_only", action='store_true',
                            help="If set, skips training and only runs evaluation.")
    eval_group.add_argument("--checkpoint_path", type=str,
                            help="Path to the trained model checkpoint (.pth) for evaluation-only mode.")
    eval_group.add_argument("--output_dir", type=str, default="evaluation_results",
                            help="Directory to save evaluation artifacts like plots.")

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


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Generates and saves a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))), normalize='true')

    fig, ax = plt.subplots(figsize=(40, 40))  # Large figure size for many classes
    sns.heatmap(cm, annot=False, fmt=".2f", cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)

    ax.set_title('Normalized Confusion Matrix', fontsize=20)
    ax.set_ylabel('True Label', fontsize=16)
    ax.set_xlabel('Predicted Label', fontsize=16)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    fig.savefig(output_path, dpi=300)
    logging.info(f"Confusion matrix saved to: {output_path}")


def train(args):
    start_time = time.time()
    fix_randomisation(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = setup_logging(f"out-logs/{args.feature_extraction_model.lower()}", args.experiment_name)

    logger.info(f"--- Experiment: {args.experiment_name} | device={device} ---")
    logger.info("Arguments:")
    logger.info(f"  > data     : dataset={args.dataset_name} backend={args.feature_extraction_model} "
                f"n_glosses={args.n_glosses} features={args.features} fs={args.fs} seed={args.seed}")
    logger.info(f"  > model    : {args.model} | hidden_dim={args.hidden_dim} n_heads={args.n_heads} "
                f"n_layers={args.n_layers} pe={args.pe}")
    logger.info(f"  > optim    : optimizer={args.optimizer} lr={args.lr} weight_decay={args.weight_decay} "
                f"sgd_momentum={args.sgd_momentum} scheduler={args.scheduler} clip_gradients={args.clip_gradients}")
    logger.info(f"  > training : epochs={args.epochs} batch_size={args.batch_size} "
                f"save_checkpoints={args.save_checkpoints} log_freq={args.log_freq} plot_stats={args.plot_stats}")
    if args.model in ("latefusion_encoder", "latefusion_pet"):
        logger.info(f"  > fusion   : hand_input_dim={args.hand_input_dim} pose_input_dim={args.pose_input_dim} "
                    f"face_input_dim={args.face_input_dim} debug={args.debug}")
    if args.model == "latefusion_pet":
        logger.info(f"  > experts  : hand_ckpt={args.hand_ckpt_path} pose_ckpt={args.pose_ckpt_path} "
                    f"face_ckpt={args.face_ckpt_path}")
    if args.evaluate_only:
        logger.info(f"  > eval     : evaluate_only={args.evaluate_only} checkpoint_path={args.checkpoint_path} "
                    f"output_dir={args.output_dir}")

    load_dotenv()
    base_path = os.getenv(f"{args.dataset_name.upper()}_{args.feature_extraction_model.upper()}_BASE_PATH")
    fs_dir = os.getenv(f"{args.dataset_name.upper()}_{args.feature_extraction_model.upper()}_TOP_FEATURES_DIR")

    dataloader_args = {
        "metadata_json_path": os.getenv(f"{args.dataset_name.upper()}_METADATA_PATH"),
        "features": args.features, "n_glosses": args.n_glosses,
        "fs": args.fs, "feature_selection_dir": fs_dir if args.fs else None,
        "pose_landmark_path": os.path.join(base_path, "pose_landmarks.parquet"),
        "hand_landmark_path": os.path.join(base_path, "hand_landmarks.parquet"),
        "face_landmark_path": os.path.join(base_path, "face_landmarks.parquet"),
        "pose_angle_path": os.path.join(base_path, "pose_angles.parquet"),
        "hand_angle_path": os.path.join(base_path, "hand_angles.parquet"),
        "face_blendshape_path": os.path.join(base_path, "face_blendshapes.parquet"),
    }

    # Load the dataset(s) for this run up front; model dimensions come straight
    # from the data (feature_dim and the gloss vocabulary are split-independent).
    logger.info("--- Loading datasets ---")
    if args.evaluate_only:
        test_set = SignLanguageFeaturesDataset(**dataloader_args, split="test")
        train_set = val_set = None
        dim_source = test_set
        logger.info(f"Datasets ready | Test={len(test_set)} | classes={len(test_set.gloss2idx)} "
                    f"feat_dim={test_set.feature_dim} max_len={test_set.max_len}")
    else:
        train_set = SignLanguageFeaturesDataset(**dataloader_args, split="train", verbose=False)
        val_set = SignLanguageFeaturesDataset(**dataloader_args, split="val", verbose=False)
        test_set = SignLanguageFeaturesDataset(**dataloader_args, split="test")
        dim_source = train_set
        logger.info(f"Datasets ready | Train={len(train_set)} Val={len(val_set)} Test={len(test_set)} | "
                    f"classes={len(train_set.gloss2idx)} feat_dim={train_set.feature_dim} max_len={train_set.max_len}")

    input_dim, num_classes = dim_source.feature_dim, len(dim_source.gloss2idx)

    logger.info(f"--- Building model: {args.model} ---")
    if args.model == 'baseline_transformer':
        model = BaselineTransformerClassification(input_dim=input_dim, num_classes=num_classes,
                                                  hidden_dim=args.hidden_dim, num_layers=args.n_layers,
                                                  n_heads=args.n_heads, pe=args.pe)
    elif args.model == 'spoter':
        model = SPOTERTransformer(input_dim=input_dim, num_classes=num_classes, hidden_dim=args.hidden_dim,
                                  num_layers=args.n_layers, n_heads=args.n_heads, pe=args.pe)
    elif args.model == 'lstm':
        model = LSTMClassifier(input_dim=input_dim, hidden_dim=args.hidden_dim, num_classes=num_classes)
    elif args.model == "bilstm":
        model = BiLSTMClassifier(input_dim=input_dim, hidden_dim=args.hidden_dim, num_classes=num_classes)
    elif args.model == 'encoder':
        model = EncoderOnly(input_dim=input_dim, num_classes=num_classes, hidden_dim=args.hidden_dim,
                            num_layers=args.n_layers, n_heads=args.n_heads, pe=args.pe)
    elif args.model == 'latefusion_encoder':
        if not all([args.hand_input_dim, args.pose_input_dim, args.face_input_dim]):
            raise ValueError("For --model latefusion_encoder, you must provide all input dims .")
        model = LateFusionEncoder(hand_input_dim=args.hand_input_dim, pose_input_dim=args.pose_input_dim,
                                  face_input_dim=args.face_input_dim,
                                  num_classes=num_classes, hidden_dim=args.hidden_dim,
                                  n_heads=args.n_heads, num_layers=args.n_layers,
                                  debug=args.debug)
    elif args.model == 'latefusion_pet':
        if not all([args.hand_ckpt_path, args.pose_ckpt_path, args.face_ckpt_path,
                    args.hand_input_dim, args.pose_input_dim, args.face_input_dim]):
            raise ValueError("For --model latefusion_pet, you must provide checkpoint paths and per-modality input dims for all three experts.")
        hand_expert = load_expert_encoder(args.hand_ckpt_path, args.hand_input_dim, args.n_glosses,
                                          args.hidden_dim, args.n_heads, args.n_layers, device)
        pose_expert = load_expert_encoder(args.pose_ckpt_path, args.pose_input_dim, args.n_glosses,
                                          args.hidden_dim, args.n_heads, args.n_layers, device)
        face_expert = load_expert_encoder(args.face_ckpt_path, args.face_input_dim, args.n_glosses,
                                          args.hidden_dim, args.n_heads, args.n_layers, device)

        model = LateFusionPET(
            hand_expert=hand_expert,
            pose_expert=pose_expert,
            face_expert=face_expert,
            num_classes=args.n_glosses,
            hand_input_dim=args.hand_input_dim,
            pose_input_dim=args.pose_input_dim,
            face_input_dim=args.face_input_dim,
            debug=args.debug,
        )

    # Send model to GPU
    model.to(device)

    # ==============================================================================
    # MODE SWITCH: Run Evaluation-Only or Full Training
    # ==============================================================================
    if args.evaluate_only:
        if not args.checkpoint_path or not os.path.exists(args.checkpoint_path):
            raise ValueError("--evaluate_only mode requires a valid --checkpoint_path.")

        logger.info(f"--- Evaluation-only | checkpoint: {args.checkpoint_path} ---")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))

        test_loader = DataLoader(test_set, shuffle=False, batch_size=args.batch_size)

        loss_fn = nn.CrossEntropyLoss()
        _, test_acc, (y_true, y_pred) = evaluate_batch(model, loss_fn, test_loader, device, return_preds=True)

        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # 95% CI margins (Agresti-Coull approximation, scaled back to [0, 1])
        n_test = len(test_set)
        acc_ci = calculate_approximate_ci(test_acc * 100, n_test) / 100
        macro_f1_ci = calculate_approximate_ci(macro_f1 * 100, n_test) / 100
        weighted_f1_ci = calculate_approximate_ci(weighted_f1 * 100, n_test) / 100

        logger.info("\n" + "=" * 30 + " EVALUATION SUMMARY " + "=" * 30)
        logger.info(f"Checkpoint: {os.path.basename(args.checkpoint_path)}")
        logger.info(f"Test Accuracy: {test_acc:.4f} ± {acc_ci:.4f} (95% CI, n={n_test})")
        logger.info(f"Test Macro F1-Score: {macro_f1:.4f} ± {macro_f1_ci:.4f} (95% CI)")
        logger.info(f"Test Weighted F1-Score: {weighted_f1:.4f} ± {weighted_f1_ci:.4f} (95% CI)")
        logger.info("=" * 82)

        os.makedirs(args.output_dir, exist_ok=True)
        class_names = [test_set.idx2gloss[i] for i in range(len(test_set.idx2gloss))]
        plot_filename = f"{args.experiment_name}_confusion_matrix.png"
        plot_path = os.path.join(args.output_dir, plot_filename)
        plot_confusion_matrix(y_true, y_pred, class_names, plot_path)
        return  # Exit the function cleanly

    # --- If not evaluate_only, proceed with the full training pipeline ---
    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=args.batch_size)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gpu_mem = f" | GPU mem: {torch.cuda.memory_allocated() / 1024 ** 2:.0f} MB" if device.type == 'cuda' else ""
    logger.info(f"Trainable params: {total_params:,}{gpu_mem}")

    loss_fn = nn.CrossEntropyLoss()

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))

    optimizer_map = {"sgd": optim.SGD(trainable_params, lr=args.lr, momentum=args.sgd_momentum),
                     "adam": optim.Adam(trainable_params, lr=args.lr),
                     "adamw": optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)}
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
        if scheduler is None: raise ValueError(f"Invalid scheduler name: {args.scheduler}")

    train_losses, train_accs, val_losses, val_accs, lr_progress = [], [], [], [], []
    best_val_acc = 0.0
    checkpoint_dir = Path("out-checkpoints") / f"{args.feature_extraction_model.lower()}" / args.experiment_name
    if args.save_checkpoints: checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- Training for {args.epochs} epochs ---")
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

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            if args.save_checkpoints:
                best_model_path = checkpoint_dir / "best_model.pth"
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"  -> Saved new best model (Val Acc: {best_val_acc:.4f})")

        lr_progress.append(optimizer.param_groups[0]['lr'])

    if args.save_checkpoints:
        final_model_path = checkpoint_dir / "final_model.pth"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"--- Saved final model to {final_model_path} ---")

    logger.info("--- Final evaluation on test set ---")
    best_model_path = checkpoint_dir / "best_model.pth"
    if args.save_checkpoints and best_model_path.exists():
        logger.info(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    _, test_acc, (y_true, y_pred) = evaluate_batch(model, loss_fn, test_loader, device, return_preds=True)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # 95% CI margins (Agresti-Coull approximation, scaled back to [0, 1])
    n_test = len(test_set)
    acc_ci = calculate_approximate_ci(test_acc * 100, n_test) / 100
    macro_f1_ci = calculate_approximate_ci(macro_f1 * 100, n_test) / 100
    weighted_f1_ci = calculate_approximate_ci(weighted_f1 * 100, n_test) / 100

    end_time = time.time()
    total_time = end_time - start_time
    logger.info("\n" + "=" * 30 + " EXPERIMENT SUMMARY " + "=" * 30)
    logger.info(f"Experiment Name: {args.experiment_name}")
    logger.info(f"Final Test Accuracy: {test_acc:.4f} ± {acc_ci:.4f} (95% CI, n={n_test})")
    logger.info(f"Final Macro F1 Score: {macro_f1:.4f} ± {macro_f1_ci:.4f} (95% CI)")
    logger.info(f"Final Weighted F1 Score: {weighted_f1:.4f} ± {weighted_f1_ci:.4f} (95% CI)")
    logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
    logger.info(f"Total Trainable Parameters: {total_params:,}")
    logger.info(f"Total Runtime: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    logger.info("=" * 82)

    if args.plot_stats:
        plot_dir = Path("out-img") / f"{args.feature_extraction_model.lower()}" / args.experiment_name
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
