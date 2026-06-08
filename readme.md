# Expressive: The Role of Pose and Facial Cues in Transformer-Based Sign Language Recognition

This repository contains the official implementation for the Bachelor Thesis "Expressive: The Role of Pose and Facial Cues in Transformer-Based Sign Language Recognition."

> **Branch note (`iberspeech26`).** This branch is a slimmed-down, training-only version of the project used for additional experiments. It assumes you are using the pre-processed Parquet datasets linked below — the MediaPipe / ViTPose feature-extraction and feature-engineering pipelines have been removed. For the full preprocessing pipeline see the [`master`](../../tree/master) branch.

## 1. Setup

### 1.1 Clone the repository
```bash
git clone https://github.com/georgebaher/Sign-Language-Recognition-using-Lightweight-Transformers.git
cd Sign-Language-Recognition-using-Lightweight-Transformers
git checkout iberspeech26
```

### 1.2 Create the conda environment
```bash
conda env create -f environment.yml
conda activate slrenv
```

### 1.3 Download the pre-processed datasets
1. Download the pre-processed Parquet files from:
   [link-to-preprocessed-datasets](https://drive.google.com/drive/folders/1tUoqJawvh0-hSxrIrfU3NUFPcQ8Xhusr?usp=drive_link)
2. Unzip. You should see folders like:
   ```
   WLASL_feature_processing_mediapipe/
   WLASL_feature_processing_vitpose/
   AVASAG_feature_processing_mediapipe/
   AVASAG_feature_processing_vitpose/
   ```
   Each contains a `data_parquet/` directory with the feature files and a `top_features/` directory used when `--fs 1` is set.
3. Place all four folders inside a single directory of your choice — this will be your `BASE_DIR`.

### 1.4 Configure the `.env` file
Open [.env](.env) and set `BASE_DIR` to the absolute path of the directory that contains the four dataset folders above. All other variables are derived from `BASE_DIR` and should not need editing.

## 2. Training

`src/run_training.py` is the single entry point for all experiments.

### 2.1 Choosing the late-fusion variant (manual step)

`latefusion_encoder` and `latefusion_pet` each have two variants: **Weighted Sum** (adaptive per-modality logit weights) and **Logit Concatenation**. The variant is selected by which class is imported in [src/run_training.py](src/run_training.py#L42-L45). The last import wins — uncomment the line you want and comment the other:

```python
# Weighted Sum (default)
from src.Training.models.LateFusionLogitEncoderWeightedSum import LateFusionEncoder
# from src.Training.models.LateFusionLogitEncoderConcat import LateFusionEncoder

from src.Training.models.LateFusionModelUsingPretrainedEncodersWeightedSum import LateFusionPET
# from src.Training.models.LateFusionModelUsingPretrainedEncodersConcat import LateFusionPET
```

### 2.2 Command-line arguments

#### Experiment & data selection
| Flag | Description | Default |
|---|---|---|
| `--experiment_name` | Unique name for the run; used in output paths. | `SLR_Experiment` |
| `--seed` | Random seed. | `379` |
| `--dataset_name` | `wlasl` or `avasag`. | `wlasl` |
| `--feature_extraction_model` | Backend whose features to load: `mediapipe` or `vitpose`. | `vitpose` |
| `--n_glosses` | Number of glosses (classes). | `100` |
| `--features` | Space-separated list, e.g. `hand_landmarks pose_landmarks face_blendshapes`. | required |
| `--fs` | `1` to use the bundled top-features lists, `0` for all features. | `0` |

#### Model architecture
| Flag | Description | Default |
|---|---|---|
| `--model` | `baseline_transformer` \| `spoter` \| `lstm` \| `bilstm` \| `encoder` \| `latefusion_encoder` \| `latefusion_pet` | `baseline_transformer` |
| `--hidden_dim` | Model hidden dimension (output of the input `nn.Linear` embedding, transformer `d_model`). | `256` |
| `--n_heads` | Attention heads. Must divide `--hidden_dim`. | `8` |
| `--n_layers` | Transformer layers. | `6` |
| `--pe` | Positional encoding kind: `sincos` \| `learnable` \| `none`. | `sincos` |
| `--debug` | Verbose forward-pass prints. | off |

#### Late-fusion-specific
| Flag | Description |
|---|---|
| `--hand_input_dim`, `--pose_input_dim`, `--face_input_dim` | **Required for `latefusion_encoder` / `latefusion_pet`.** Tell the model how to slice the early-fused input tensor. Defaults: `84`, `50`, `52`. |
| `--hand_ckpt_path`, `--pose_ckpt_path`, `--face_ckpt_path` | **Required for `latefusion_pet`.** Paths to the pre-trained expert encoder checkpoints. The experts must have been trained with the same `--hidden_dim` you pass here. |

#### Training hyperparameters
| Flag | Description | Default |
|---|---|---|
| `--epochs` | Number of epochs. | `100` |
| `--batch_size` | Batch size. | `32` |
| `--lr` | Learning rate. | `1e-4` |
| `--optimizer` | `sgd` \| `adam` \| `adamw`. | `adamw` |
| `--sgd_momentum` | SGD momentum. | `0.9` |
| `--scheduler` | `warmup_linear` \| `warmup_cosine` \| `warmup_constant` \| `none`. | `cosine` |
| `--clip_gradients` | Max-norm for grad clipping (`0` disables). | `0.0` |

#### Logging & saving
| Flag | Description |
|---|---|
| `--save_checkpoints` | Save best (by val acc) and final model. |
| `--log_freq` | Log every N epochs. Default `1`. |
| `--plot_stats` | Save training-curve plots. |

#### Evaluation-only mode
| Flag | Description |
|---|---|
| `--evaluate_only` | Skip training, just evaluate a checkpoint. |
| `--checkpoint_path` | Path to the `.pth` checkpoint to evaluate. |
| `--output_dir` | Where to save evaluation artifacts. Default `evaluation_results`. |

### 2.3 Example training commands

**A. Encoder-only (early fusion baseline) on AVASAG**
```bash
python src/run_training.py \
    --experiment_name "AVASAG_EarlyFusion_Baseline" \
    --dataset_name "avasag" \
    --feature_extraction_model "vitpose" \
    --model "encoder" \
    --features "hand_landmarks" "pose_landmarks" "face_blendshapes" \
    --fs 0 \
    --epochs 100 --batch_size 32 --lr 1e-3 \
    --optimizer "sgd" --sgd_momentum 0.9 --scheduler "none" \
    --save_checkpoints --plot_stats
```

**B. Adaptive late-fusion (Weighted Sum) on WLASL**
> Ensure the `WeightedSum` import is active in [run_training.py](src/run_training.py#L42-L45).
```bash
python src/run_training.py \
    --experiment_name "WLASL_LateFusion_Adaptive" \
    --dataset_name "wlasl" \
    --feature_extraction_model "mediapipe" \
    --model "latefusion_encoder" \
    --features "hand_landmarks" "pose_landmarks" "face_blendshapes" \
    --fs 0 \
    --hand_input_dim 84 --pose_input_dim 50 --face_input_dim 52 \
    --epochs 100 --batch_size 32 --lr 1e-3 \
    --optimizer "sgd" --sgd_momentum 0.9 --scheduler "warmup_cosine" \
    --save_checkpoints --plot_stats
```

**C. Late-fusion with Pre-trained Experts (PET)**
First train one expert per modality (using `--model encoder` on a single feature each), then:
```bash
python src/run_training.py \
    --experiment_name "WLASL_LateFusion_PET" \
    --dataset_name "wlasl" \
    --feature_extraction_model "mediapipe" \
    --model "latefusion_pet" \
    --features "hand_landmarks" "pose_landmarks" "face_blendshapes" \
    --fs 1 \
    --hand_input_dim 84 --pose_input_dim 50 --face_input_dim 52 \
    --hand_ckpt_path "out-checkpoints/mediapipe/WLASL_Hand_Expert/best_model.pth" \
    --pose_ckpt_path "out-checkpoints/mediapipe/WLASL_Pose_Expert/best_model.pth" \
    --face_ckpt_path "out-checkpoints/mediapipe/WLASL_Face_Expert/best_model.pth" \
    --epochs 100 --lr 1e-3 \
    --optimizer "sgd" --sgd_momentum 0.9 --scheduler "none" \
    --save_checkpoints
```

## 3. Evaluation

Evaluate an existing checkpoint without retraining:
```bash
python src/run_training.py \
    --evaluate_only \
    --checkpoint_path "out-checkpoints/mediapipe/WLASL_LateFusion_Adaptive/best_model.pth" \
    --experiment_name "EVAL_WLASL_LateFusion" \
    --dataset_name "wlasl" \
    --feature_extraction_model "mediapipe" \
    --model "latefusion_encoder" \
    --features "hand_landmarks" "pose_landmarks" "face_blendshapes" \
    --fs 1 \
    --hand_input_dim 84 --pose_input_dim 50 --face_input_dim 52
```

A 95% confidence-interval helper for percentage metrics is provided in [src/ci_calculator.py](src/ci_calculator.py):
```bash
python src/ci_calculator.py --n 1000 --score_pct 60.62
```

## 4. Outputs

| Directory | Contents |
|---|---|
| `out-logs/<backend>/` | Per-experiment training logs. |
| `out-checkpoints/<backend>/<experiment>/` | `best_model.pth` and `final_model.pth`. |
| `out-img/<backend>/<experiment>/` | Training-curve plots (with `--plot_stats`). |
| `evaluation_results/` | Confusion matrices from `--evaluate_only` runs. |

## 5. Running on Kaggle (no local GPU)

These models are small (~8-24M params) and fit comfortably on Kaggle's free P100 / T4. Quota: 30 GPU-hrs/week. A 100-epoch run on a single-modality `encoder` takes ~30-60 minutes.

### 5.1 One-time setup

**(a)** Upload the four pre-processed dataset folders as **one Kaggle Dataset** (e.g. `sl-features`). Keep the folder layout intact:

```
sl-features/
├── WLASL_feature_processing_mediapipe/   (data_parquet/, top_features/, WLASL_v0.3.json)
├── WLASL_feature_processing_vitpose/     (data_parquet/, top_features/)
├── AVASAG_feature_processing_mediapipe/  (data_parquet/, top_features/)
└── AVASAG_feature_processing_vitpose/    (data_parquet/, top_features/, avasag_metadata.json)
```

You can upload via the Kaggle web UI ("Create → New Dataset") or `kaggle datasets create`. Zip the folders first to upload faster — Kaggle auto-extracts.

**(b)** Make sure this branch is pushed to a Git remote Kaggle can clone over HTTPS (public GitHub works; for a private repo use a Kaggle Secret with a token).

### 5.2 Per-experiment notebook

In Kaggle: **New Notebook** → right sidebar:
- Accelerator: **GPU T4 ×2** (or **GPU P100** if available)
- Internet: **On**
- Add data: attach your `sl-features` dataset

Paste these four cells:

```python
# Cell 1 — clone the iberspeech26 branch into the writable working dir
!git clone -b iberspeech26 https://github.com/georgebaher/Sign-Language-Recognition-using-Lightweight-Transformers.git /kaggle/working/SLR
%cd /kaggle/working/SLR
```

```python
# Cell 2 — install missing deps. Kaggle already has torch (2.x), sklearn,
# pandas, pyarrow, matplotlib, seaborn. Only python-dotenv and transformers
# need to be added.
!pip install -q python-dotenv transformers
```

```python
# Cell 3 — point BASE_DIR at the attached dataset
# (replace the slug after /kaggle/input/ with whatever you named yours)
!sed -i 's|^BASE_DIR=.*|BASE_DIR=/kaggle/input/sl-features|' .env
!grep '^BASE_DIR' .env
```

```python
# Cell 4 — smoke test: load one configuration and print shapes / columns / sample
# Prints split sizes, classes, feature columns, sample tensor shape, and a
# DataLoader batch. Run this once after Cell 3 to confirm the data is mounted
# and readable before launching a long training run.
!python src/dataset/test_dataloader.py \
    --dataset_name wlasl \
    --feature_extraction_model mediapipe \
    --features hand_landmarks pose_landmarks \
    --n_glosses 10
```

```python
# Cell 5 — run a training experiment
!python src/run_training.py \
    --experiment_name "WLASL_Encoder_Hand_Sincos" \
    --dataset_name wlasl \
    --feature_extraction_model mediapipe \
    --model encoder \
    --features hand_landmarks \
    --hidden_dim 256 --n_heads 8 --n_layers 6 --pe sincos \
    --epochs 100 --batch_size 32 --lr 1e-3 \
    --optimizer adamw --scheduler warmup_cosine \
    --save_checkpoints --plot_stats
```

Outputs land in `/kaggle/working/SLR/out-logs/`, `out-checkpoints/`, `out-img/` — Kaggle preserves `/kaggle/working/` as the notebook's downloadable output zip.

### 5.3 Practical tips

- **Interactive vs offline runs.** "Save & Run All (Commit)" reruns the whole notebook on a fresh container, with **up to 12 hours** of GPU per run. The interactive session is also capped at ~12 hours of idle, ~9 hours of GPU time. For sweeps, prefer Commit — you can queue several without babysitting.
- **Persisting checkpoints between runs.** `/kaggle/working/` only lives as long as the notebook output. To carry a trained expert into a follow-up PET run, either (a) download `best_model.pth`, re-upload as its own dataset, attach, point `--hand_ckpt_path` at `/kaggle/input/...`, or (b) use the notebook's "Add Output to Dataset" feature to push artifacts into a versioned Kaggle Dataset.
- **Reusing the same notebook for many configs.** Parametrise Cell 5 — define a small dict of args at the top, then loop or branch on it. Kaggle notebooks support widgets if you want a quick dropdown.
- **Torch version.** Kaggle ships PyTorch 2.x + CUDA 12.x. Nothing in this codebase relies on torch 1.10 specifics, so no pin is needed — just don't run `pip install torch==1.10...` in Cell 2.
- **`%cd` matters.** Every cell starts in `/kaggle/working/` by default. The `%cd /kaggle/working/SLR` in Cell 1 sticks across subsequent cells so the relative output paths (`out-logs/...`) work as expected.

## 6. Previous Experiments

Logs, checkpoints and plots for the prior experiments are archived at:
https://drive.google.com/drive/folders/1RJlSvbJw0QXVB49blhyAC616sA-ItzYT?usp=sharing
