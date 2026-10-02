# FG2027 — modalities and fusion in pose-based sign language recognition

This branch (`FG2027`) contains the experiments for the FG2027 paper. It compares how much
hands, body pose and face contribute to isolated sign recognition, and whether combining them
early or late matters, using ViTPose landmarks from **WLASL100** and **AVASAG100**.

Earlier work lives on other branches: the IberSPEECH training pipeline (`run_training.py`) on
[`iberspeech26`](../../tree/iberspeech26), and the original thesis pipeline, including feature
extraction, on [`master`](../../tree/master).

- [FG2027.md](FG2027.md) — study plan and agreed protocol.
- [PAPER_NOTES.md](PAPER_NOTES.md) — data facts and decisions to carry into the paper.

## Experiments

The first batch has **22 runs**: 11 model/input combinations × 2 datasets, seed 379.

| Model | Inputs | Fusion |
| --- | --- | --- |
| BiLSTM | H / HP / HPF | Inputs concatenated |
| SPOTER-inspired Transformer | H / HP / HPF | Inputs concatenated |
| Encoder-only Transformer | H / HP / HPF | Early: inputs concatenated before one encoder |
| Encoder-only Transformer | HP / HPF | Late: one encoder per modality, learned weighted sum of predictions |

H = hands (42 points), P = body pose (13 points), F = face (68 points); x/y only, so input
widths are 84 / 110 / 246.

Training settings are fixed in [configs/fg2027.json](configs/fg2027.json): 100 epochs, batch 32,
AdamW (learning rate 1e-4, weight decay 0.01), constant learning rate, no dropout, no positional
encoding. The checkpoint with the best validation accuracy is evaluated once on the test set.

## Running

Runs locally on a machine with a CUDA GPU, from the repository root. Install PyTorch with CUDA
support for your GPU, then [requirements-fg2027.txt](requirements-fg2027.txt). Put the eight data
files (metadata + hand/pose/face landmarks per dataset) under `data/`:

```
data/<wlasl|avasag>/metadata.json
data/<wlasl|avasag>/vitpose/{hand,pose,face}_landmarks.parquet
```

Then, in order:

```bash
python -m src.fg2027 audit      # check the data; writes data_audit.json per dataset. Review it.
python -m src.fg2027 smoke      # after setting protocol_reviewed=true in the config
python -m src.fg2027 run --dataset wlasl --model encoder --modalities H   # pilot; review curves
python -m src.fg2027 run --dataset wlasl                                  # remaining runs, one dataset at a time
python -m src.fg2027 run --dataset avasag
python -m src.fg2027 summarize  # results.csv and results.md
```

All commands default to [configs/fg2027.json](configs/fg2027.json); pass `--config` to use another.

### Safeguards

- Training refuses to start unless the smoke tests passed with the same data, code and config,
  and `protocol_reviewed` is `true` in the config.
- Each run folder records a fingerprint of its data, code and config. Rerunning with anything
  changed stops with an error instead of mixing results.
- **No resuming:** completed runs are skipped; a run that was interrupted starts again from
  epoch 1 when the same command is rerun.

## Data handling

The loader ([src/dataset/FGFeaturesDataset.py](src/dataset/FGFeaturesDataset.py)):

- uses the first 100 glosses of each metadata file and its train/val/test splits as given;
- aligns hands, pose and face by video ID and frame number, never by row order;
- keeps full sequences; missing landmarks are stored as −2 and kept separate from batch padding;
- in frames with more than one detected person, keeps the person closest to the horizontal image
  centre, for all modalities;
- keeps coordinates slightly outside [0, 1] unchanged;
- stops with an error on missing clips, duplicate rows, labels that disagree with the metadata,
  coordinates far outside [0, 1], or invalid numbers.

Every input is loaded for every run, so all 22 runs use exactly the same clips and frames.

## Outputs

Written to `runs/seed379/` (git-ignored):

| Path | Contents |
| --- | --- |
| `<dataset>/data_audit.json` | Split counts, missing-data rates, person-selection choices, file checksums |
| `<dataset>/smoke_pass.json` | Proof the smoke tests passed for this data, code and config |
| `<dataset>/<model>_<inputs>/seed_379/` | One run: `metrics.json`, `predictions.csv`, `history.json`, `best.pt`, `manifest.json`, `environment.json`, `pip-freeze.txt`, `status.json` |
| `results.csv`, `results.md` | Summary of all runs, including parameter counts |

`metrics.json` holds accuracy, macro-F1, weighted-F1, the best validation epoch, parameter count,
training time and peak GPU memory; for late fusion, also the learned weight of each modality.
`predictions.csv` keeps per-video predictions for the planned bootstrap confidence intervals.

## Repository layout

| Path | Purpose |
| --- | --- |
| [src/fg2027.py](src/fg2027.py) | Entry point: audit, smoke, run, summarize |
| [src/dataset/FGFeaturesDataset.py](src/dataset/FGFeaturesDataset.py) | Data loading, alignment and checks |
| [src/utils.py](src/utils.py) | Training and evaluation loops |
| [src/models/](src/models/) | Models. Used here: `BiLSTM.py`, `SPOTER.py`, `EncoderOnlyTransformer.py`, `FGLateFusion.py`, `positional_encoding.py`. The others are kept from earlier work and are not used by FG2027. |
| [tests/fg_smoke.py](tests/fg_smoke.py) | Smoke tests (need a GPU) |
| [configs/fg2027.json](configs/fg2027.json) | Experiment configuration |
| [experiments/matrix.csv](experiments/matrix.csv) | The 22 planned runs |
