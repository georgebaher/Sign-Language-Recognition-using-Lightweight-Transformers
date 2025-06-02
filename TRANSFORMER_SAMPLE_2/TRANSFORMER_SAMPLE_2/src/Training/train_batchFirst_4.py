
import os, sys
import argparse
import random
import logging
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
import time

sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')

from src.utils.args_utils import *
from src.Training.trainingUtils.utils import train_epoch_batch, evaluate_batch
from src.Training.dataloader.landmarks_dataloader import simpleLandmarksDataLoader, simpleLandmarksDataLoader_memory, simpleLandmarksDataLoaderFace, AVASAGsimpleLandmarksDataLoader_memory
from src.Training.models.BaselineTransformerClassification import BaselineTransformerClassification, BaselineTransformerClassificationNoPE



def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="AVASAG100_originalSpoterPE_land-75",
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--num_classes", type=int, default=100, help="Number of classes to be recognized by the model")
    parser.add_argument("--hidden_dim", type=int, default=108,
                        help="Hidden dimension of the underlying Transformer model")
    parser.add_argument("--n_heads", type=int, default=9,
                        help="Hidden dimension of the underlying Transformer model")
    parser.add_argument("--seed", type=int, default=379,
                        help="Seed with which to initialize all the random components of the training")
    parser.add_argument("--model2use", type=str, choices=["originalSpoterPE", "originalSpoterNOPE", "baselineTransformer",
                                                          "ownModelwquery", "ownModelwseq"],
                        default="originalSpoterPE",
                        help='Type of model to select for the training. choices=["originalSpoterPE", '
                             '"originalSpoterNOPE", "baselineTransformer","ownModelwquery", "ownModelwseq"]')

    # data
    parser.add_argument("--training_set_path", type=str, default="", help="Path to the training dataset CSV file")
    parser.add_argument("--testing_set_path", type=str, default="", help="Path to the testing dataset CSV file")
    parser.add_argument("--experimental_train_split", type=float, default=None,
                        help="Determines how big a portion of the training set should be employed (intended for the "
                             "gradually enlarging training set experiment from the paper)")

    parser.add_argument("--validation_set", type=str, choices=["from-file", "split-from-train", "none"],
                        default="from-file", help="Type of validation set construction. See README for further rederence")
    parser.add_argument("--validation_set_size", type=float,
                        help="Proportion of the training set to be split as validation set, if 'validation_size' is set"
                             " to 'split-from-train'")
    parser.add_argument("--validation_set_path", type=str, default="", help="Path to the validation dataset CSV file")
    # Landmarks library:
    parser.add_argument("--mediaPipe", type=str, default='True',
                        help="Determines whether the landmarks were generated using MediaPipe[True] or using VisionAPI[False]")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model for")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the model training")
    parser.add_argument("--log_freq", type=int, default=1,
                        help="Log frequency (frequency of printing all the training info)")

    # Checkpointing
    parser.add_argument("--save_checkpoints", type=bool, default=True,
                        help="Determines whether to save weights checkpoints")


    # Gaussian noise normalization
    parser.add_argument("--gaussian_mean", type=int, default=0, help="Mean parameter for Gaussian noise layer")
    parser.add_argument("--gaussian_std", type=int, default=0.001,
                        help="Standard deviation parameter for Gaussian noise layer")

    # Visualization
    parser.add_argument("--plot_stats", type=bool, default=True,
                        help="Determines whether continuous statistics should be plotted at the end")
    parser.add_argument("--plot_lr", type=bool, default=True,
                        help="Determines whether the LR should be plotted at the end")
    parser.add_argument("--PE", type=str2bool, default="True",
                        help="USE PE or not")
    parser.add_argument("--complete_block", type=str2bool, default="True",
                        help="USE completeBlock or not")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch size")



    return parser


def train(args):

    # MARK: TRAINING PREPARATION AND MODULES

    ##########  Initialize all the random seeds // DEVICES ############
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    # Set device to CUDA only if applicable
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    ##### PARAMETERS ######
    # n_landmarks = int(args.hidden_dim/2) # 54 21 42 21 75
    mediaPipe = True
    ### WLASL100 ###
    dataset_name = "WLASL100"
    path_distribution = "../TRANSFORMER_SAMPLE_2/data/WLASLS100/WLASL100_v0.3.csv"
    path_features = "datasets/wlasl/wlasl/features/pose_features" 

    ### AVASAG ###
    # dataset_name = "AVASAG"
    # path_features = "/run/user/1000/gvfs/smb-share:server=137.250.171.20,share=datasets/AVASAG_dictionary/videos_sentences/features/videos_body/mediaPipe"
    # path_distribution = "/home/cristinalunaj/PycharmProjects/AVASAG_processing/data/AVASAG/AVASAG_100_v0.0.csv"

    extra_name_features = "_poses_landmarks.csv" #_poses_landmarks.csv" #_blenderShapes

    max_len_videos = 210 # max-len WLASL100:210, max-len AVASAG:
    padding = True
    n_landmarks = 75 # 75 pose landmarks (21x2 RHand, 21x2 LHand, 33x2 Pose) ; 52 blendershapes
    n_heads = 10
    hidden_dim = 75*2
    model2use = "regularTransformer" #"Transf_NoPE" # SPOTER_PE_Q1_NoPE_batchFirst   TransfPE "TransfPEQ1s" #"originalSpoterPE" regularTransformer regularTransformer
    batch_size = args.batch_size
    name_optimizer = "SGD" # SGD, adam, adamW, RMSprop
    clipWeights = False


    experiment_name = dataset_name+"_"+model2use+"_land-"+str(n_landmarks)+"_v"+str(1)+"_cblock-"+str(int(complete_block))+"_PE-"+str(int(PE))+"_bs-"+str(batch_size)
    print(">>>> Experiment name: "+experiment_name+ " <<<<")


    # Set the output format to print into the console and save into LOG file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + ".log")
        ]
    )


    ##################### MODELS #########################
    # Construct the model
    if (model2use == "regularTransformer"): # Transformer GAP - Yes PE
        print("Using model regular Transformer - Transformer GAP - with PE")
        slrt_model = BaselineTransformerClassification(num_classes=args.num_classes, hidden_dim=hidden_dim, n_heads=n_heads)
    elif (model2use == "regularTransformerNoPE"):  # Transformer GAP - No PE
        print("Using model regular Transformer NO PE - Transformer GAP - withOUT PE")
        slrt_model = BaselineTransformerClassificationNoPE(num_classes=args.num_classes, hidden_dim=hidden_dim,n_heads=n_heads)


    slrt_model.train(True)
    slrt_model.to(device)

    #print(f"Used GPU memory: {torch.cuda.memory_allocated() / 1024 / 1024} MB")
    # Construct the other modules
    cel_criterion = nn.CrossEntropyLoss()

    if(name_optimizer=="SGD"):
        optimizer = optim.SGD(slrt_model.parameters(), lr=args.lr)
    elif (name_optimizer == "adam"):
        optimizer = torch.optim.Adam(slrt_model.parameters(), lr=args.lr)
    elif (name_optimizer == "adamW"):
        optimizer = torch.optim.AdamW(slrt_model.parameters(), lr=args.lr)
    elif (name_optimizer == "RMSprop"):
        optimizer = torch.optim.RMSprop(slrt_model.parameters(), lr=args.lr)



    # Ensure that the path for checkpointing and for images both exist
    Path("out-checkpoints/" + experiment_name + "/").mkdir(parents=True, exist_ok=True)
    Path("out-img/").mkdir(parents=True, exist_ok=True)

    ############### MARK: DATA LOADERS:: #####################
    # Training set
    # transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])
    transform = None
    if (dataset_name == "WLASL100"):
        print("Processing WLASL100")
        train_set = simpleLandmarksDataLoader_memory(path_distribution, path_features, transform=transform,
                                                     mediapipe=mediaPipe,
                                                     n_landmarks=n_landmarks, split2load="train",
                                                     extra_name_file=extra_name_features, max_len_videos=max_len_videos,
                                                     padding=padding)  # simpleLandmarksDataLoader_memory
        # Validation set
        val_set = simpleLandmarksDataLoader_memory(path_distribution, path_features, transform=transform,
                                                   mediapipe=mediaPipe,
                                                   n_landmarks=n_landmarks, split2load="val",
                                                   extra_name_file=extra_name_features, max_len_videos=max_len_videos,
                                                   padding=padding)  # simpleLandmarksDataLoader_memory
        # # Test
        test_set = simpleLandmarksDataLoader_memory(path_distribution, path_features, transform=transform,
                                                    mediapipe=mediaPipe,
                                                    n_landmarks=n_landmarks, split2load="test",
                                                    extra_name_file=extra_name_features, max_len_videos=max_len_videos,
                                                    padding=padding)  # simpleLandmarksDataLoader_memory

        # train_set = val_set = test_set # for debug purposes

    elif(dataset_name == "AVASAG"):
        print("Processing AVASAG")
        extra_name_features = "_poses_landmarks.csv"
        train_set = AVASAGsimpleLandmarksDataLoader_memory(path_distribution, path_features, transform=transform,
                                                     mediapipe=mediaPipe,
                                                     n_landmarks=n_landmarks, split2load="train",
                                                     extra_name_file=extra_name_features, max_len_videos=max_len_videos,
                                                     padding=padding)  # simpleLandmarksDataLoader_memory

        # Validation set
        val_set = AVASAGsimpleLandmarksDataLoader_memory(path_distribution, path_features, transform=transform,
                                                   mediapipe=mediaPipe,
                                                   n_landmarks=n_landmarks, split2load="val",
                                                   extra_name_file=extra_name_features, max_len_videos=max_len_videos,
                                                   padding=padding)  # simpleLandmarksDataLoader_memory
        # Test
        test_set = AVASAGsimpleLandmarksDataLoader_memory(path_distribution, path_features, transform=transform,
                                                    mediapipe=mediaPipe,
                                                    n_landmarks=n_landmarks, split2load="test",
                                                    extra_name_file=extra_name_features, max_len_videos=max_len_videos,
                                                    padding=padding)  # simpleLandmarksDataLoader_memory


    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=1)


    ####################### TRAINING / VALIDATION #####################
    # MARK: TRAINING
    train_acc, val_acc = 0, 0
    losses, train_accs, val_accs = [], [], []
    lr_progress = []
    top_train_acc, top_val_acc = 0, 0
    checkpoint_index = 0

    if args.experimental_train_split:
        print("Starting " + experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")
        logging.info("Starting " + experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")

    else:
        print("Starting " + experiment_name + "...\n\n")
        logging.info("Starting " + experiment_name + "...\n\n")

    start = time.time()

    for epoch in range(args.epochs):
        train_loss, _, _, train_acc = train_epoch_batch(slrt_model, train_loader, cel_criterion, optimizer, device, batch_size=batch_size, clip_weights=clipWeights)
        losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)

        if val_loader:
            slrt_model.train(False)
            _, _, val_acc, _ = evaluate_batch(slrt_model, val_loader, device) #num_classes=args.num_classes
            slrt_model.train(True)
            val_accs.append(val_acc)

        # Save checkpoints if they are best in the current subset
        if args.save_checkpoints:
            if train_acc > top_train_acc:
                top_train_acc = train_acc
                torch.save(slrt_model, "out-checkpoints/" + experiment_name + "/checkpoint_t_" + str(checkpoint_index) + ".pth")

            if val_acc > top_val_acc:
                top_val_acc = val_acc
                print("... Saving checkpoint: out-checkpoints/" + experiment_name + "/checkpoint_v_" + str(checkpoint_index) + ".pth")
                torch.save(slrt_model, "out-checkpoints/" + experiment_name + "/checkpoint_v_" + str(checkpoint_index) + ".pth")

        if epoch % args.log_freq == 0:
            print("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss / len(train_loader)) + " acc: " + str(train_acc))
            logging.info("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss / len(train_loader)) + " acc: " + str(train_acc))

            if val_loader:
                print("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(val_acc))
                logging.info("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(val_acc))

            print("")
            logging.info("")

        # Reset the top accuracies on static subsets
        if epoch % 10 == 0:
            top_train_acc, top_val_acc = 0, 0
            checkpoint_index += 1

        lr_progress.append(optimizer.param_groups[0]["lr"])

    # MARK: TESTING
    print("\nTesting checkpointed models starting...\n")
    logging.info("\nTesting checkpointed models starting...\n")
    top_result, top_result_name = 0, ""

    if test_loader:
        for i in range(checkpoint_index+1):
            for checkpoint_id in ["t", "v"]:
                # tested_model = VisionTransformer(dim=2, mlp_dim=108, num_classes=100, depth=12, heads=8)
                tested_model = torch.load("out-checkpoints/" + experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i) + ".pth")
                tested_model.eval()
                tested_model.train(False)
                _, _, eval_acc, _ = evaluate_batch(tested_model, test_loader, device, print_stats=True)

                if eval_acc > top_result:
                    top_result = eval_acc
                    top_result_name = experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i)

                print("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))
                logging.info("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))

        end = time.time()
        elapsed_time = end-start
        print("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")
        logging.info("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")
        logging.info("\n ----------------Config information----------------------- \n")
        logging.info("- Features:" + str(extra_name_features) + "\n")
        logging.info("- Model:" + str(model2use) + "\n")
        logging.info("- Batch size:" + str(batch_size) + "\n")
        logging.info("- Opt: " + str(optimizer) + "\n")
        logging.info("- Max. Len:" + str(max_len_videos) + "\n")
        logging.info("- N landm.:" + str(n_landmarks) + "\n")
        logging.info("- N heads:" + str(n_heads) + "\n")
        logging.info("- Transform (data Augm.):" + str(transform) + "\n")
        logging.info(" - Elapsed Time training (seconds): "+str(elapsed_time)+"\n")



    # PLOT 0: Performance (loss, accuracies) chart plotting
    if args.plot_stats:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(losses) + 1), losses, c="#D64436", label="Training loss")
        ax.plot(range(1, len(train_accs) + 1), train_accs, c="#00B09B", label="Training accuracy")

        if val_loader:
            ax.plot(range(1, len(val_accs) + 1), val_accs, c="#E0A938", label="Validation accuracy")

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.set(xlabel="Epoch", ylabel="Accuracy / Loss", title="")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True, fontsize="xx-small")
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()

    # loop experiments:
    # for batch_size in [1, 10, 50, 100]: # 1, 10, 50, 100
    #     for complete_block in [True, False]:
    #         for PE in [True, False]:
    train(args)


