# FG2027 — paper notes

Facts and decisions to carry into the paper. Numbers come from the Colab data audit
(`data_audit.json`) unless marked otherwise. AVASAG figures are pending until its audit passes.

## Data

### Missing landmarks

Share of coordinates with no detection, WLASL100 (ViTPose):

| Modality | Missing coordinates | Clips with no detection at all |
| --- | --- | --- |
| Hands | 45.3% | 3 |
| Pose (13 points) | 10.3% | 0 |
| Face | 1.3% | 0 |

- Hands are missing far more often than pose or face, so hands-only models train on sparse input.
  This belongs in the data description.
- Not yet checked: whether this is mostly one hand at a time (e.g. the non-dominant hand
  out of frame) or both hands.
- AVASAG: pending.

### Frames with more than one detected person

- WLASL: 21 frames. AVASAG: 2,176 frames in 59 videos; in 31 of those videos every frame has a
  second detection.
- The second detection is a small figure at the image edge (shoulder width ≈ 0.00–0.05 vs
  ≈ 0.19–0.34 for the signer) or a duplicate detection of the signer.
- Rule: in such frames, keep the person whose visible pose points are on average closest to
  the horizontal image centre, and use that person for hands, pose and face. The rule uses no
  labels or model outputs. Every choice is saved in `data_audit.json`.
- Suggested wording: *"In frames with more than one detected person (WLASL: 21; AVASAG: 2,176),
  we kept the detection closest to the horizontal image centre, consistently across modalities."*

### Coordinates slightly outside [0, 1]

- WLASL: hands range from −0.023 to 1.080 (1.3% of values outside [0, 1]); pose from 0.074 to
  1.232 (0.6%); face is fully inside (0.066–0.771).
- These are points the detector placed just past the frame edge, typically hands and hips below
  the bottom of the frame. They are kept unchanged, not clipped, to preserve the stored
  normalization. Values outside [−0.5, 1.5] would stop the loader as a scaling error.

## Models

### Parameter counts

Measured at the configuration used (hidden size 256, 6 layers, 8 heads):

| Model | Hands | Hands + Pose | Hands + Pose + Face |
| --- | --- | --- | --- |
| BiLSTM | 0.75M | 0.80M | 1.08M |
| Encoder-only (early fusion) | 7.94M | 7.94M | 7.98M |
| SPOTER-inspired | 17.41M | 17.42M | 17.45M |
| Late fusion | — | 15.86M | 23.81M |

- Sizes differ by more than 20×. No size-matched runs are planned; report parameter counts
  alongside scores (the results table includes them) and acknowledge the difference.
- Late fusion uses one full encoder per modality, so any gain over early fusion cannot be
  credited to fusion alone.
- Most Transformer parameters sit in the feed-forward layers (PyTorch's default width of 2048).

### Missing values as model input

- Missing coordinates are stored as −2 and replaced with 0 inside the models; there is no separate
  "missing" flag. A real coordinate of 0 would look the same.
- In practice this rarely matters: observed values never get close to 0 in WLASL (the lowest
  is −0.023, and 99.9% of hand values are above 0.07). Mention as a minor limitation at most.

## Comparison with IberSPEECH

FG2027 numbers may not match the IberSPEECH numbers exactly, even for the same model and inputs.
The earlier loader (removed from this branch; see the `iberspeech26` branch):

- lined up hands, pose and face by row order within each video, not by frame number, so a
  missing row in one modality shifted all later frames;
- did not handle frames with more than one detected person. If IberSPEECH used these same
  ViTPose files, the 31 AVASAG videos with a second detection in every frame would have
  contained both people's rows, interleaved.

The FG2027 loader aligns by frame number and keeps one person per frame. Note this if the paper
compares against IberSPEECH results.
