# Expressive: The Role of Pose and Facial Cues in Transformer-Based Sign Language Recognition

This repository contains the official implementation for the Bachelor Thesis "Expressive: The Role of Pose and Facial Cues in Transformer-Based Sign Language Recognition." It provides a complete pipeline for data processing, feature engineering, and training/evaluating advanced Transformer models for Isolated Sign Language Recognition (ISLR).

## 1. Setup and Installation

### Step 1: Clone This Repository
Clone this project to your local machine:
```bash
git clone https://github.com/georgebaher/SLR.git
cd SLR
```

### Step 2: Configure The Environment File
This project is configured using a **`.env`** file located in the root directory of the repository. You only need to make one simple change to this file for the project to work correctly.

1.  **Open the `.env` file** in your code editor.
2.  **Edit the `BASE_DIR` variable** to match the absolute path of the project folder on your local machine.

### Step 3: Create Conda Environment
A fully specified `environment.yml` file is provided. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate slrenv
```

### Step 4: Set Up ViTPose Backend
This project uses ViTPose (beside MediaPipe) as a high-accuracy feature extraction backend.

1.  Clone the official ViTPose repository:
    ```bash
    git clone https://github.com/ViTAE-Transformer/ViTPose.git
    ```
2.  Install ViTPose as a package:
    ```bash
    cd ViTPose
    pip install -v -e .
    ```
3.  Copy the necessary scripts into the `ViTPose` directory. From the root of *this* project:
    *   Copy contents of `SCRIPTS_TO_COPY_INSIDE_VITPOSE_FOLDER/ViTPose_root_dir/` into `ViTPose/`.
    *   Copy contents of `SCRIPTS_TO_COPY_INSIDE_VITPOSE_FOLDER/mmdetection_cfg_dir/` into `ViTPose/demo/mmdetection_cfg/`.

4.  Download model checkpoints:
    *   **YOLOX-L Detector:** [yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth)
    *   **ViTPose-L Whole-Body:** [vitpose _large.pth](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBaW1CZ1lWN0pqVGxnY2NzMVNORlVHU1RzbVJKOHc%5FZT1hOXpLd1o&cid=E534267B85818129&id=E534267B85818129%2125516&parId=E534267B85818129%21158&o=OneUp)

5.  Place both `.pth` files into the `ViTPose/` root directory.

6.  Split the ViTPose model for inference:
    ```bash
    cd ViTPose
    python tools/model_split.py --source <PATH TO THE DOWNLOADED (e.g. vitpose _large.pth)>
    cd ..
    ```

## 2. Data Preparation

### Option A: Use Pre-processed Data (Recommended)
Pre-processed Parquet files for all features and datasets are available. This is the fastest way to get started.

1.  Download the data from: [link-to-preprocessed-datasets](https://drive.google.com/drive/folders/1tUoqJawvh0-hSxrIrfU3NUFPcQ8Xhusr?usp=drive_link)
2.  Unzip the file. You will see folders like `AVASAG_feature_processing_mediapipe`, etc.
3.  Place these folders inside the directory you specified as `BASE_DIR` in your `.env` file. The training script will now find them automatically.

### Option B: Process Data from Scratch (Advanced)
If you have your own video dataset, you can use the provided scripts to generate all feature sets. The pipeline is as follows:

**1. Primary Feature Extraction**

*   **Extract Landmarks (Hands, Pose, Face):** Use `run_batch_processing.py` (for a folder of videos) or `process_from_metadata.py` (for a subset defined in a JSON file). These scripts are located in the `MediaPipe/` or `ViTPose/` directories.
    ```bash
    # Example 1: Process all videos in a folder using ViTPose
    python ViTPose/run_batch_processing.py \
        --video-folder "path/to/your/videos" \
        --output-folder "path/to/save/parquets" \
        --save-video

    # Example 2: Process a subset from a metadata file using MediaPipe
    python MediaPipe/process_from_metadata.py \
        --metadata-file "path/to/your/metadata.json" \
        --n-glosses 50 \
        --video-folder "path/to/your/videos" \
        --output-folder "path/to/save/parquets"
    ```

*   **Extract Facial Blendshapes (MediaPipe only):** This requires a separate script.
    ```bash
    python feature_processing_utils/extract_blendshapes.py \
        --metadata_file "path/to/your/metadata.json" \
        --video_folder "path/to/your/videos" \
        --model_path "MediaPipe/face_model/face_landmarker.task" \
        --n_glosses 50
    ```

**2. Feature Cleaning & Pre-processing**

*   **Drop Z-Columns (for MediaPipe):** To ensure consistency with ViTPose, remove the estimated z-coordinate.
    ```bash
    python feature_processing_utils/drop_z_columns.py \
        --input-path "path/to/your/parquets/HAND_LANDMARKS.parquet"
    ```
*   **Drop Lower-Body Landmarks:** Remove non-linguistic lower-body keypoints.
    ```bash
    python feature_processing_utils/drop_landmarks_range.py \
        --input_path "path/to/your/parquets/POSE_LANDMARKS.parquet" \
        --start_index 25 --end_index 33 # Example for MediaPipe
    ```

**3. Feature Engineering**

*   **Calculate Inter-Joint Angles:** Generate angle features from the cleaned landmark files.
    ```bash
    python feature_processing_utils/calculate_angles.py \
        --input_path "path/to/cleaned/parquets/POSE_LANDMARKS.parquet" \
        --output_path "path/to/save/POSE_ANGLES.parquet" \
        --source "MediaPipe" \
        --body_part "pose" \
        --num_landmarks 25
    ```

**4. Feature Selection (Optional)**

*   **Aggregate Features:** Convert time-series data into a fixed-length summary vector for the selection model.
    ```bash
    python feature_processing_utils/aggregate_features.py \
        --input_path "path/to/POSE_ANGLES.parquet" \
        --out_path "path/to/save/POSE_ANGLES_SUMMARY.parquet"
    ```
*   **Run Feature Selection:** Use the summary file to find and save the most discriminative features.
    ```bash
    python feature_processing_utils/run_feature_selction.py \
        --feature_type "pose" \
        --input_type "angles" \
        --summary_path "path/to/save/POSE_ANGLES_SUMMARY.parquet" \
        --metadata_path "path/to/your/metadata.json" \
        --output_dir "path/to/save/top_features/"
    ```
    This will create a `.py` file with the list of top features and a `.json` file with the tuned model hyperparameters in the specified output directory

## 3. Training the Models

The `run_training.py` script is the main entry point for all experiments. It is highly configurable via the command-line arguments detailed below.

### **Important: Choosing the Late Fusion Strategy (Manual Step)**

The `latefusion_encoder` and `latefusion_pet` models have two variants, determined by which class is imported in the `run_training.py` script:

1.  **Weighted Sum:** The adaptive strategy that learns weights for each modality's logits.
2.  **Logit Concatenation:** A simpler fusion method that concatenates the logits.

Before running any late fusion experiment, you **must** edit `src/Training/run_training.py` to ensure the correct model is imported. The last import wins.

**To use the `WeightedSum` strategy (for the main adaptive model):**
Ensure the `LateFusionLogitEncoderWeightedSum` import is active (uncommented) and the `Concat` version is commented out.

```python
# In src/Training/run_training.py

# ...
from src.Training.models.LateFusionLogitEncoderWeightedSum import LateFusionEncoder
# from src.Training.models.LateFusionLogitEncoderConcat import LateFusionEncoder # This one is commented out
# ...
```

**To use the `LogitConcat` strategy:**
You must comment out the `WeightedSum` line and uncomment the `Concat` line.

```python
# In src/Training/run_training.py

# ...
# from src.Training.models.LateFusionLogitEncoderWeightedSum import LateFusionEncoder # This one is commented out
from src.Training.models.LateFusionLogitEncoderConcat import LateFusionEncoder
# ...
```

---

### Command-Line Arguments

All aspects of training can be controlled through the following flags:

#### **Experiment & Data Selection**
*   `--experiment_name`: A unique name for your run (e.g., `WLASL_EncoderOnly_EarlyFusion`). Outputs will be saved under this name.
*   `--seed`: Random seed for reproducibility. `(default: 379)`
*   `--dataset_name`: The dataset to use. `[choices: wlasl, avasag]`.
*   `--feature_extraction_model`: The backend whose features you want to use. `[choices: mediapipe, vitpose]`
*   `--n_glosses`: Number of glosses (classes) to use from the dataset. `(default: 100)`
*   `--features`: A space-separated list of feature types to load (e.g., `hand_landmarks` `pose_landmarks` `face_blendshapes`).
*   `--fs`: Use feature selection. `1` to use the pre-selected top features, `0` to use all features. `(default: 0)`
*   `--feature_padding_mode`: Method to pad feature dimension for Transformers. `[choices: sentinel, repeat, truncate] (default: sentinel)`

#### **Model Architecture**
*   `--model`: The architecture to train. `[choices: baseline_transformer, spoter, lstm, bilstm, encoder, latefusion_encoder, latefusion_pet]`
*   `--n_heads`: Number of attention heads for Transformer models. `(default: 8)`
*   `--n_layers`: Number of layers for Transformer models. `(default: 6)`
*   `--pe`: Enable Positional Encoding for Transformers. `0` for disabled, `1` for enabled. `(default: 0)`
*   `--debug`: A flag to enable verbose debug prints in the model's forward pass.

#### **Late Fusion Specific Arguments**
*   `--hand_input_dim`, `--pose_input_dim`, `--face_input_dim`: **Required for `latefusion_encoder` and `latefusion_pet`**. These tell the model how to slice the input tensor. `(defaults: 84, 50, 52)`
*   `--hand_ckpt_path`, `--pose_ckpt_path`, `--face_ckpt_path`: **Required for `latefusion_pet`**. Paths to the pre-trained expert encoder models.
*   `--pet_hidden_dim_hand`, `--pet_hidden_dim_pose`, `--pet_hidden_dim_face`: **Required for `latefusion_pet`**. The hidden dimension (including padding) that the expert models were trained with.

#### **Training Hyperparameters**
*   `--epochs`: Number of epochs to train for. `(default: 100)`
*   `--batch_size`: The batch size. `(default: 32)`
*   `--lr`: The learning rate for the optimizer. `(default: 1e-3)`
*   `--optimizer`: The optimization algorithm. `[choices: sgd, adam, adamw] (default: sgd)`
*   `--sgd_momentum`: Momentum for the SGD optimizer. `(default: 0.9)`
*   `--scheduler`: Learning rate scheduler. `[choices: warmup_linear, warmup_cosine, warmup_constant, none] (default: none)`

#### **Logging & Saving**
*   `--save_checkpoints`: A flag to save the best model (based on validation accuracy) and the final model.
*   `--log_freq`: How often to print logs (every N epochs). `(default: 1)`
*   `--plot_stats`: A flag to generate and save training statistics plots.

---
### Example Training Scenarios

**A. Train an Encoder-Only Model (Early Fusion Baseline)**
This command trains the `encoder` model on the AVASAG dataset.

```bash
python src/Training/run_training.py \
    --experiment_name "AVASAG_EarlyFusion_Baseline" \
    --dataset_name "avasag" \
    --feature_extraction_model "vitpose" \
    --model "encoder" \
    --features "hand_landmarks" "pose_landmarks" "face_blendshapes" \
    --fs 0 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3 \
    --optimizer "sgd" \
    --sgd_momentum 0.9 \
    --scheduler "none" \
    --save_checkpoints \
    --plot_stats
```

**B. Train the Adaptive Late Fusion Model (Weighted Sum, From Scratch)**
This command trains the proposed `latefusion_encoder` model on the WLASL dataset. **Note: For this command to work, ensure the `WeightedSum` import is active in `run_training.py` as described above.**

```bash
python src/Training/run_training.py \
    --experiment_name "WLASL_LateFusion_Adaptive" \
    --dataset_name "wlasl" \
    --feature_extraction_model "mediapipe" \
    --model "latefusion_encoder" \
    --features "hand_landmarks" "pose_landmarks" "face_blendshapes" \
    --fs 0 \
    --hand_input_dim 84 \
    --pose_input_dim 50 \
    --face_input_dim 52 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3 \
    --optimizer "sgd" \
    --sgd_momentum 0.9 \
    --scheduler "cosine" \
    --save_checkpoints \
    --plot_stats
```

**C. Train the Late Fusion Model with Pre-Trained Experts (PET) using Weighted Sum for fusion**
This trains the `latefusion_pet` model, freezing the expert encoders and only training the final fusion layers. You must first train the individual expert models and provide the paths to their saved checkpoints.

**Note: For this command to work, ensure the `WeightedSum` import is active in `run_training.py` as described above.**

```bash
python src/Training/run_training.py \
    --experiment_name "WLASL_LateFusion_PET" \
    --dataset_name "wlasl" \
    --feature_extraction_model "mediapipe" \
    --model "latefusion_pet" \
    --features "hand_landmarks" "pose_landmarks" "face_blendshapes" \
    --fs 1 \
    --hand_input_dim 84 \
    --pose_input_dim 50 \
    --face_input_dim 52 \
    --pet_hidden_dim_hand 88 \
    --pet_hidden_dim_pose 52 \
    --pet_hidden_dim_face 52 \
    --hand_ckpt_path "out-checkpoints/mediapipe/WLASL_Hand_Expert/best_model.pth" \
    --pose_ckpt_path "out-checkpoints/mediapipe/WLASL_Pose_Expert/best_model.pth" \
    --face_ckpt_path "out-checkpoints/mediapipe/WLASL_Face_Expert/best_model.pth" \
    --epochs 100 \
    --lr 1e-3 \
    --optimizer "sgd" \
    --sgd_momentum 0.9 \
    --scheduler "none" \
    --save_checkpoints
```

## 4. Evaluation
Evaluate a pre-trained model checkpoint on the test set without retraining.

```bash
python src/Training/run_training.py \
    --evaluate_only \
    --checkpoint_path "out-checkpoints/mediapipe/WLASL_LateFusion_Adaptive/best_model.pth" \
    --experiment_name "EVAL_WLASL_LateFusion" \
    --dataset_name "wlasl" \
    --feature_extraction_model "mediapipe" \
    --model "latefusion_encoder" \
    --features "hand_landmarks" "pose_landmarks" "face_blendshapes" \
    --fs 1 \
    --hand_input_dim 84 \
    --pose_input_dim 50 \
    --face_input_dim 52
```

## 5. Project Outputs
*   **Logs:** `out-logs/`
*   **Model Checkpoints:** `out-checkpoints/`
*   **Training Plots:** `out-img/`
*   **Evaluation Artifacts:** `evaluation_results/`