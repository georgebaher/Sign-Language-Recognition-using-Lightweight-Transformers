# FG2027 — landmark groups and fusion

ViTPose-based isolated sign recognition on WLASL100 and AVASAG100.
The first batch is complete: **22 experiments, seed 379**.

- [Results](runs/seed379/results.md)
- [Design decisions](FG2027.md)
- [Paper notes](PAPER_NOTES.md)
- [Configuration](configs/fg2027.json) and [experiment matrix](experiments/matrix.csv)

Earlier pipelines remain on [`iberspeech26`](../../tree/iberspeech26) and
[`master`](../../tree/master).

## Run

Use Python 3.12+ and a CUDA-compatible PyTorch installation, then:

```bash
pip install -r requirements-fg2027.txt
```

Set data paths in the configuration. Each dataset needs metadata and the three
ViTPose landmark parquet files. WLASL metadata comes from its MediaPipe folder;
this does not introduce MediaPipe features.

For new experiments, choose a fresh `output_dir` to preserve the committed results.
Run from the repository root:

```bash
python -m src.fg2027 audit
# Review the audit, then set protocol_reviewed=true before smoke tests.
python -m src.fg2027 smoke
python -m src.fg2027 run --dataset wlasl --model encoder --modalities H
# Check the pilot before launching the remaining runs.
python -m src.fg2027 run --dataset wlasl
python -m src.fg2027 run --dataset avasag
python -m src.fg2027 summarize
```

Pass `--config <path>` to every command when using a different configuration.

Completed runs are skipped. Interrupted runs **restart from epoch 1**.
Changed data, code or configuration require new smoke tests and a separate output
directory. Checkpoints stay local; committed predictions and run records support
result review without retraining.
