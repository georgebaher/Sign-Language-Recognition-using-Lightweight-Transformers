import logging
import os
import random
import sys
import time
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
load_dotenv()

sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
from src.utils.args_utils import *
from src.Training.trainingUtils.utils import train_epoch_batch, evaluate_batch
from src.Training.dataloader.dataloader import WLASLParquetDataset
from src.Training.models.BaselineTransformerClassification import BaselineTransformerClassification


def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experiment_name", type=str, default='EXPERIMENT',
                        help="Specify experment name")

    parser.add_argument("--hidden_dim", type=int, default=150,
                        help="Hidden dimension of the underlying Transformer model")
    parser.add_argument("--n_heads", type=int, default=9,
                        help="Hidden dimension of the underlying Transformer model")
    parser.add_argument("--seed", type=int, default=379,
                        help="Seed with which to initialize all the random components of the training")
    parser.add_argument("--clip_weights", type=int, default=0, help="Clip weights during training")
    parser.add_argument("--clip_gradients", type=int, default=0, help="Clip gradients during training")
    parser.add_argument("--model2use", type=str,
                        choices=["baselineTransformer"],
                        default="baselineTransformer",
                        help='Type of model to select for the training. choices=["baselineTransformer"]')
    parser.add_argument("--pe", type=int, default=1,
                        help="Determines whether positional encoding is used or not")
    parser.add_argument("--optimizer", type=str,
                        choices=["SGD", "adam", "adamW"],
                        default="adamW",
                        help='Type of optimizer. choices=["SGD", "adam", "adamW"]')
    parser.add_argument("--sgd_momentum", type=float, default=0.0,
                        help="Momentum value for SGD optimizer. Set to 0.0 for no momentum.")
    parser.add_argument("--dataset_name", type=str,
                        choices=["WLASL100", "AVASAG100"],
                        default="WLASL100",
                        help='Dataset used. choices=["WLASL100", "AVASAG100"]')
    parser.add_argument("--features", type=str,
                        default="XYZ",
                        help='features used')
    parser.add_argument("--ignore_lower_body", type=int,
                        default=0,
                        help='ignore or not lower body features')
    parser.add_argument("--num_classes", type=int,
                        default=100,
                        help='number of classes recognized')

    # Landmarks library
    parser.add_argument("--mediapipe_holistic", type=str, default='True',
                        help="Determines whether the landmarks were generated using MediaPipe or other")

    parser.add_argument("--padding", type=str, default='True',
                        help="Determines whether the missing features were padded in features extractions or not")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model for")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the model training")
    parser.add_argument("--log_freq", type=int, default=1,
                        help="Log frequency (frequency of printing all the training info)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--scheduler_type", type=str,
                        choices=["linear", "cosine", "constant", "none"],
                        default="none",
                        help="Learning rate scheduler: 'linear' for warmup + linear decay, 'cosine' for warmup + cosine decay, 'constant' for warmup only, 'none' for no scheduler")

    # Checkpointing
    parser.add_argument("--save_checkpoints", type=bool, default=True,
                        help="Determines whether to save weights checkpoints")

    # # TODO: Gaussian noise normalization (Not yet)
    parser.add_argument("--transform", type=int, default=0, help="Apply gaussian noise transformation")
    # parser.add_argument("--gaussian_mean", type=int, default=0, help="Mean parameter for Gaussian noise layer")
    # parser.add_argument("--gaussian_std", type=int, default=0.001,
    #                     help="Standard deviation parameter for Gaussian noise layer")

    # Visualization (Not yet)
    parser.add_argument("--plot_stats", type=bool, default=True,
                        help="Determines whether continuous statistics should be plotted at the end")
    parser.add_argument("--plot_lr", type=bool, default=True,
                        help="Determines whether the LR should be plotted at the end")

    return parser


def fix_randomisation():
    # Set all random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    # TRAINING PREPARATION AND MODULES

    ##########  Initialize all the random seeds and set device ############
    # init seeds
    fix_randomisation()
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'[INFO] Running using {device} on <{torch.cuda.get_device_name(0)}>')
    #_____________________________________________________________________#

    ##########  PARAMETERS ############
    mediapipe_holistic = args.mediapipe_holistic
    dataset_name = args.dataset_name
    padding = args.padding
    n_heads = args.n_heads
    hidden_dim = args.hidden_dim
    model2use = args.model2use
    batch_size = args.batch_size
    num_classes = args.num_classes
    optimizer_name = args.optimizer
    sgd_momentum = args.sgd_momentum
    scheduler_type = args.scheduler_type
    clip_weights = args.clip_weights
    clip_gradients = args.clip_gradients
    transform = args.transform  # TODO: implement transformations (Not yet)
    features = args.features
    ignore_lower_body = args.ignore_lower_body
    pe = args.pe
    epochs = args.epochs
    lr = args.lr
    log_freq = args.log_freq
    save_checkpoints = args.save_checkpoints
    experiment_args = ", ".join([
        f"dataset={dataset_name}",
        f"features={features}",
        f"ignore_lower_body={ignore_lower_body}",
        f"num_classes={num_classes}",
        f"model={model2use}",
        f"mediapipe_holistic={True if mediapipe_holistic else False}",
        f"dim={hidden_dim}",
        f"heads={n_heads}",
        f"PE={pe}",
        f"pad={padding}",
        f"clip_weights={clip_weights}",
        f"clip_gradients={clip_gradients}",
        f"opt={optimizer_name}",
        f"sgd_momentum={sgd_momentum}",
        f"scheduler={scheduler_type}",
        f"bs={batch_size}",
        f"epochs={epochs}",
        f"lr={lr:.0e}",  # scientific notation (e.g., 1e-03)
        f"logfreq={log_freq}",
        f"checkpoints={save_checkpoints}",
    ])
    experiment_name = args.experiment_name

    # Set the output format to print into the console and save into LOG file
    log_dir = "out-logs"  # or "out-logs", "results/logs", etc.
    os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist
    log_path = os.path.join(log_dir, f"{experiment_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path)
        ]
    )
    # _____________________________________________________________________#

    ##################### MODELS #########################
    # Construct the model
    # FOR NOW, ONLY ONE MODEL VARIATION

    model = BaselineTransformerClassification(num_classes=num_classes, hidden_dim=hidden_dim,
                                              n_heads=n_heads, w_pe=pe)
    model.train(True)
    model.to(device)  # moves model to cpu or gpu if available

    print(f"[INFO] Using GPU memory: {torch.cuda.memory_allocated() / 1024 / 1024} MB")

    # Ensure that the path for checkpointing and for images both exist
    Path("out-checkpoints/" + experiment_name + "/").mkdir(parents=True, exist_ok=True)
    Path("out-img/").mkdir(parents=True, exist_ok=True)

    ############### DATA LOADERS #####################
    train_set = val_set = test_set = None
    if dataset_name == "WLASL100":
        print("[INFO] Processing WLASL100 dataset...")

        # Training set
        train_set = WLASLParquetDataset(parquet_path=os.getenv('WLASL100_HAND_POSE_LANDMARKS_PATH'),
                                        metadata_json_path=os.getenv('WLASL_METADATA_PATH'),
                                        split='train',
                                        transform=transform,
                                        axis_mode=features,
                                        ignore_lower_body=ignore_lower_body
                                        )
        print(f"       loaded train dataset with {len(train_set)} samples, {len(train_set.gloss2idx)} classes and shape {train_set.__getitem__(0)[0].shape}")

        # Validation set
        val_set = WLASLParquetDataset(parquet_path=os.getenv('WLASL100_HAND_POSE_LANDMARKS_PATH'),
                                      metadata_json_path=os.getenv('WLASL_METADATA_PATH'),
                                      split='val',
                                      transform=transform,
                                      axis_mode=features,
                                      ignore_lower_body=ignore_lower_body
                                      )
        print(
            f"       loaded val dataset with {len(val_set)} samples, {len(val_set.gloss2idx)} classes and shape {val_set.__getitem__(0)[0].shape}")

        # Test
        test_set = WLASLParquetDataset(parquet_path=os.getenv('WLASL100_HAND_POSE_LANDMARKS_PATH'),
                                       metadata_json_path=os.getenv('WLASL_METADATA_PATH'),
                                       split='test',
                                       transform=transform,
                                       axis_mode=features,
                                       ignore_lower_body=ignore_lower_body
                                       )
        print(
            f"       loaded test dataset with {len(test_set)} samples, {len(test_set.gloss2idx)} classes and shape {test_set.__getitem__(0)[0].shape}")

    elif dataset_name == "AVASAG":
        print("[INFO] Processing AVASAG100 dataset...")
        # TODO: load AVASAG100 dataset (Not yet)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=1)

    ####################### LOSS FUNCTION, OPTIMIZER & SCHEDULER #####################
    loss_fn = nn.CrossEntropyLoss()
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=sgd_momentum)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "adamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # Setup scheduler for warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warm-up

    if scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    elif scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    elif scheduler_type == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps
        )
    elif scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    ####################### TRAINING / VALIDATION #####################
    average_train_acc, average_val_acc = 0, 0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    lr_progress = []
    top_train_acc, top_val_acc = 0, 0
    checkpoint_index = 0

    print("[INFO] Starting experiment " + experiment_name + "...\n\n")
    logging.info("[INFO] Starting experiment " + experiment_name + "...\n\n")

    start = time.time()

    for epoch in range(args.epochs):
        average_train_loss, average_train_acc = train_epoch_batch(model, train_loader, loss_fn, optimizer, device, scheduler=scheduler,
                                                                  batch_size=batch_size, clip_gradients=clip_gradients, clip_weights=clip_weights)
        train_losses.append(average_train_loss)
        train_accs.append(average_train_acc)

        if val_loader:
            model.train(False)
            average_val_loss, average_val_acc, _ = evaluate_batch(model, loss_fn, val_loader, device, False)
            model.train(True)
            val_losses.append(average_val_loss)
            val_accs.append(average_val_acc)

        # Save checkpoints if they are best in the current subset
        if args.save_checkpoints:
            if average_train_acc > top_train_acc:
                top_train_acc = average_train_acc
                print("... Saving checkpoint: out-checkpoints/" + experiment_name + "/checkpoint_t" + str(
                    checkpoint_index) + ".pth")
                torch.save(model,
                           "out-checkpoints/" + experiment_name + "/checkpoint_t" + str(checkpoint_index) + ".pth")

            if average_val_acc > top_val_acc:
                top_val_acc = average_val_acc
                print("... Saving checkpoint: out-checkpoints/" + experiment_name + "/checkpoint_v" + str(
                    checkpoint_index) + ".pth")
                torch.save(model,
                           "out-checkpoints/" + experiment_name + "/checkpoint_v" + str(checkpoint_index) + ".pth")

        if epoch % args.log_freq == 0:
            print("[" + str(epoch + 1) + "] TRAIN | loss: " + str(average_train_loss) + ", acc: " + str(
                average_train_acc))
            logging.info(
                "[" + str(epoch + 1) + "] TRAIN | loss: " + str(average_train_loss) + ", acc: " + str(
                    average_train_acc))

            if val_loader:
                print("[" + str(epoch + 1) + "] VALIDATION | acc: " + str(average_val_acc))
                logging.info("[" + str(epoch + 1) + "] VALIDATION | acc: " + str(average_val_acc))


        # Reset the top accuracies on static subsets
        if epoch % 10 == 0:
            top_train_acc, top_val_acc = 0, 0
            checkpoint_index += 1

        lr_progress.append(optimizer.param_groups[0]["lr"])

    ####################### TESTING #####################
    print("\n[INFO] Testing checkpointed models starting...\n")
    logging.info("Testing checkpointed models starting...\n")
    top_result, top_result_name = 0, ""

    if test_loader:
        for i in range(checkpoint_index + 1):
            for checkpoint_id in ["t", "v"]:
                # tested_model = VisionTransformer(dim=2, mlp_dim=108, num_classes=100, depth=12, heads=8)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    tested_model = torch.load(
                        "out-checkpoints/" + experiment_name + "/checkpoint_" + checkpoint_id + str(i) + ".pth"
                    )
                tested_model.eval()
                tested_model.train(False)
                _, eval_acc, _ = evaluate_batch(tested_model, loss_fn, test_loader, device, print_stats=False)

                if eval_acc > top_result:
                    top_result = eval_acc
                    top_result_name = experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i)

                print("checkpoint_" + checkpoint_id + str(i) + "  ->  " + str(eval_acc))
                logging.info("checkpoint_" + checkpoint_id + str(i) + "  ->  " + str(eval_acc))

        end = time.time()
        elapsed_time = end - start
        print("\nThe top result was recorded at " + str(
            top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")
        logging.info("The top result was recorded at " + str(
            top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")
        logging.info("----------------Config information----------------------- ")
        logging.info(" - Elapsed Time training (seconds): " + str(elapsed_time))
        logging.info("- Experiment Args: " + str(experiment_args))

    # PLOT 0: Performance (loss, accuracies) chart plotting
    if args.plot_stats:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(train_losses) + 1), train_losses, c="#D64436", label="Training loss")
        ax.plot(range(1, len(train_accs) + 1), train_accs, c="#00B09B", label="Training accuracy")

        if val_loader:
            ax.plot(range(1, len(val_losses) + 1), val_losses, c="#002213", label="Validation loss")
            ax.plot(range(1, len(val_accs) + 1), val_accs, c="#E0A938", label="Validation accuracy")

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.set(xlabel="Epoch", ylabel="Accuracy / Loss", title="")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True,
                   fontsize="xx-small")
        ax.grid()

        fig.savefig("out-img/" + experiment_name + "_loss.png")

    # PLOT 1: Learning rate progress
    if args.plot_lr:
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, len(lr_progress) + 1), lr_progress, label="LR")
        ax1.set(xlabel="Epoch", ylabel="LR", title="")
        ax1.grid()

        fig1.savefig("out-img/" + experiment_name + "_lr.png")

    print("\nAny desired statistics have been plotted.\nThe experiment is finished.")
    logging.info("\nAny desired statistics have been plotted.\nThe experiment is finished.")
    print("\nThe experiment is finished.")
    logging.info("The experiment is finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()

    # loop experiments:
    # for batch_size in [1, 10, 50, 100]: # 1, 10, 50, 100
    #     for complete_block in [True, False]:
    #         for PE in [True, False]:
    train(args)
