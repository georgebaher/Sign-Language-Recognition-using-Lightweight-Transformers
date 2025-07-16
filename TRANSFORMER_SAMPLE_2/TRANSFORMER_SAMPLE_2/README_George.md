# AVASAG_processing
repo for working with AVASAG dataset



## TRAINING


* Train models: TRANSFORMER_SAMPLE/src/Training/train_batchFirst4.py
Modify and adapt the parameters:
 ##### PARAMETERS ######
    # n_landmarks = int(args.hidden_dim/2) # 54 21 42 21 75
    mediaPipe = True
    ### WLASL ###
    dataset_name = "WLASL"
    path_distribution = "../TRANSFORMER_SAMPLE/data/WLASLS100/WLASL100_v0.3.csv"
    path_features = "datasets/wlasl/wlasl/features/pose_features" 

    ### AVASAG ###
    # dataset_name = "AVASAG"
    # path_features = "datasets/AVASAG_dictionary/videos_sentences/features/videos_body/mediaPipe"
    # path_distribution = "TRANSFORMER_SAMPLE/data/AVASAG/AVASAG_100_v0.0.csv"

    extra_name_features = "_poses_landmarks.csv" #_poses_landmarks.csv" #_blenderShapes

    max_len_videos = 210 # max-len WLASL:210, max-len AVASAG:
    padding = True
    n_landmarks = 75 # 75 pose landmarks (21x2 RHand, 21x2 LHand, 33x2 Pose) ; 52 blendershapes
    n_heads = 10
    hidden_dim = 75*2
    model2use = "regularTransformer" #"Transf_NoPE" # SPOTER_PE_Q1_NoPE_batchFirst   TransfPE "TransfPEQ1s" #"originalSpoterPE" regularTransformer regularTransformer
    batch_size = args.batch_size
    name_optimizer = "SGD" # SGD, adam, adamW, RMSprop
    clipWeights = False

The code will load the features and start a training process using one transformer model

## TEST

* Test models: TRANSFORMER_SAMPLE/src/Training/test_batchFirst.py
As previously, for testing you need to change the parameters to indicate where is your data and chich ckeckpoint of the previous model you want to load for evauate it

    ### WLASL ###
    dataset_name = "WLASL"
    path_distribution = "TRANSFORMER_SAMPLE/data/WLASLS100/WLASL100_v0.3.csv"
    path_features = "datasets/wlasl/wlasl/features/pose_features"#

    posteriors_folder =  dataset_name + "_"+model2use+"_land-75_v1_cblock-"+str(int(complete_block))+"_PE-"+str(int(PE))+"_bs-1" #"AVASAG_regularTransformer_land-75_v1" #
    checkpoint_folder = "checkpoint_t_10"

  



# For extracting additional metrics:
TRANSFORMER_SAMPLE/src/Training/process_outputs/extract_extraMetrics.py



