
import os
import argparse
import random
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import transform
from torch.utils.data import DataLoader
from src.Training.trainingUtils.utils import evaluate_batch_savePred
from src.Training.dataloader.landmarks_dataloader import simpleLandmarksDataLoader, simpleLandmarksDataLoader_memory, simpleLandmarksDataLoaderFace, AVASAGsimpleLandmarksDataLoader_memory
from src.Training.models.spoter_model_original import SPOTER_batchFirst, SPOTER_PE_Q1_batchFirst, SPOTER_PE_batchFirst, SPOTER_PE_Q1_NoPE_batchFirst, SPOTER_NoPE_batchFirst
from src.Training.models.transformer_attentive_pooler import TransformerClassificationAttentivePooler
from src.Training.models.BaselineTransformerClassification import BaselineTransformerClassification, BaselineTransformerClassificationNoPE




def getNparams(model):
    print(" #### NETWORK SIZE ####")
    total_params = sum(
        param.numel() for param in model.parameters()
    )
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("The model has in total : ", total_params, " parameters (trainable & non-trainable)")
    print("The model has in total : ", trainable_params, " parameters (trainable)")
    print(" ############ ")


def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="lsa_64_spoter",
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--num_classes", type=int, default=100, help="Number of classes to be recognized by the model")
    parser.add_argument("--hidden_dim", type=int, default=108,
                        help="Hidden dimension of the underlying Transformer model")
    parser.add_argument("--n_heads", type=int, default=9,
                        help="Hidden dimension of the underlying Transformer model")
    parser.add_argument("--seed", type=int, default=379,
                        help="Seed with which to initialize all the random components of the training")

    # data
    parser.add_argument("--testing_set_path", type=str, default="", help="Path to the testing dataset CSV file")

    # Landmarks library:
    parser.add_argument("--mediaPipe", type=str, default='True',
                        help="Determines whether the landmarks were generated using MediaPipe[True] or using VisionAPI[False]")
    parser.add_argument("--model2use", type=str,
                        choices=["originalSpoterPE", "originalSpoterNOPE", "baselineTransformer",
                                 "ownModelwquery", "ownModelwseq"],
                        default="originalSpoterPE",
                        help='Type of model to select for the training. choices=["originalSpoterPE", '
                             '"originalSpoterNOPE", "baselineTransformer","ownModelwquery", "ownModelwseq"]')

    parser.add_argument("--namePE", type=str, default=None,
                        help="name of the positional Encodign layer (For the Query-Class version the name is: 'wEnc', for spoter is 'pos')")

    # Checkpointing
    parser.add_argument("--load_checkpoint", type=str, default="True",
                        help="Determines the path to load weights checkpoints")
    return parser




def test(args):
    # Initialize all the random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    #namePE = args.namePE  # pos wEnc

    # Set device to CUDA only if applicable
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    mediaPipe = True

    ##### PARAMETERS ######
    ### WLASL100 ###
    dataset_name = "WLASL100"
    path_distribution = "TRANSFORMER_SAMPLE_2/data/WLASLS100/WLASL100_v0.3.csv"
    path_features = "datasets/wlasl/wlasl/features/pose_features"#

    ### AVASAG ###
    # dataset_name = "AVASAG"
    # path_features = "datasets/AVASAG_dictionary/videos_sentences/features/videos_body/mediaPipe"
    # path_distribution = "TRANSFORMER_SAMPLE_2/data/AVASAG/AVASAG_100_v0.0.csv"

    model2use = "regularTransformer"  # TransfPE "TransfPEQ1s" #"originalSpoterPE" regularTransformer regularTransformer TransAttentivePooler
    PE = False
    complete_block= False
    posteriors_folder =  dataset_name + "_"+model2use+"_land-75_v1_cblock-"+str(int(complete_block))+"_PE-"+str(int(PE))+"_bs-1" #"AVASAG_regularTransformer_land-75_v1" #
    checkpoint_folder = "checkpoint_t_10"


    save_posteriors = "TRANSFORMER_SAMPLE_2/src/Training/out-checkpoints/"+posteriors_folder+"/posteriors/"+checkpoint_folder
    load_checkpoint = "TRANSFORMER_SAMPLE_2/src/Training/out-checkpoints/"+posteriors_folder+"/"+checkpoint_folder+".pth"
        #"TRANSFORMER_SAMPLE_2/src/Training/out-checkpoints/AVASAG100_originalSpoterPE_land-75/checkpoint_t_8.pth"
    # n_landmarks = int(args.hidden_dim/2) # 54 21 42 21 75

    set2test = "test"
    extra_name_features = "_poses_landmarks.csv" #_poses_landmarks.csv" #_blenderShapes extra_name_features = "_poses_landmarks.csv"
    batch_size = 1
    max_len_videos = 210
    padding = True
    n_landmarks = 75
    n_heads = 10
    hidden_dim = int(75 * 2)
    save_posteriors_set = os.path.join(save_posteriors, set2test)

    # Construct the model
    if (model2use == "regularTransformer"):
        print("Using model regular Transformer")
        slrt_model = BaselineTransformerClassification(num_classes=args.num_classes, hidden_dim=hidden_dim,
                                                       n_heads=n_heads)
    elif (model2use == "regularTransformerNoPE"):
        print("Using model regular Transformer NO PE")
        slrt_model = BaselineTransformerClassificationNoPE(num_classes=args.num_classes, hidden_dim=hidden_dim,
                                                           n_heads=n_heads)

    #Load weigths:
    slrt_model.load_state_dict(torch.load(load_checkpoint).state_dict())
    slrt_model.eval()
    slrt_model.train(False)
    getNparams(slrt_model)


    slrt_model.to(device)

    ######### PREPARE DATA FOR BEING EVALUATED BY THE NW ################
    print( " #### EVALUATING ... ####")
    transform = None
    if (dataset_name == "WLASL100"):
        test_set = simpleLandmarksDataLoader_memory(path_distribution, path_features, transform=transform,
                                                    mediapipe=mediaPipe,
                                                    n_landmarks=n_landmarks, split2load=set2test,
                                                    extra_name_file=extra_name_features, max_len_videos=max_len_videos,
                                                    padding=padding)
    elif (dataset_name == "AVASAG"):
        test_set = AVASAGsimpleLandmarksDataLoader_memory(path_distribution, path_features, transform=transform,
                                                    mediapipe=mediaPipe,
                                                    n_landmarks=n_landmarks, split2load=set2test,
                                                    extra_name_file=extra_name_features, max_len_videos=max_len_videos,
                                                    padding=padding)

    test_loader = DataLoader(test_set, shuffle=False, batch_size=1)
    os.makedirs(save_posteriors_set, exist_ok=True)
    pred_correct, pred_all, eval_acc, stats = evaluate_batch_savePred(slrt_model, test_loader, device, save_path=save_posteriors_set, print_stats=True)
    print("ACC: ", str(eval_acc), " (", str(pred_correct), "/", str(pred_all), ")")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    test(args)
