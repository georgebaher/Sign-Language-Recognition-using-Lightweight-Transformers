# test_dataloader.py
#
# Smoke test for SignLanguageFeaturesDataset. Loads one configuration end-to-end
# and prints what was read (split sizes, classes, feature columns, sample shape,
# one DataLoader batch). Run it before launching real experiments to confirm
# the .env paths, parquet files, and feature combo all wire up correctly.

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from torch.utils.data import DataLoader

# Repo root for src.* imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.dataset.SLFeaturesDataset import SignLanguageFeaturesDataset


def get_parser():
    p = argparse.ArgumentParser("SignLanguageFeaturesDataset smoke test")
    p.add_argument("--dataset_name", default="wlasl", choices=["wlasl", "avasag"])
    p.add_argument("--feature_extraction_model", default="mediapipe", choices=["mediapipe", "vitpose"])
    p.add_argument("--features", nargs="+", required=True,
                   help="e.g. hand_landmarks pose_landmarks face_blendshapes")
    p.add_argument("--n_glosses", type=int, default=10)
    p.add_argument("--fs", type=int, default=0, help="1 = use bundled top-features lists, 0 = all features")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_cols_shown", type=int, default=12,
                   help="Show this many feature columns before truncating the list.")
    return p


def _format_columns(cols, limit):
    """Pretty-print a (possibly long) list of feature-column names."""
    if len(cols) <= limit:
        return [f"  [{i:4d}] {c}" for i, c in enumerate(cols)]
    head, tail = limit // 2, limit - limit // 2
    lines = [f"  [{i:4d}] {cols[i]}" for i in range(head)]
    lines.append(f"  ... ({len(cols) - limit} more columns) ...")
    lines += [f"  [{i:4d}] {cols[i]}" for i in range(len(cols) - tail, len(cols))]
    return lines


def main():
    args = get_parser().parse_args()
    load_dotenv()

    dataset_key = args.dataset_name.upper()
    backend_key = args.feature_extraction_model.upper()
    base_path = os.getenv(f"{dataset_key}_{backend_key}_BASE_PATH")
    fs_dir = os.getenv(f"{dataset_key}_{backend_key}_TOP_FEATURES_DIR")
    metadata_path = os.getenv(f"{dataset_key}_METADATA_PATH")

    if base_path is None:
        raise RuntimeError(f"Env var {dataset_key}_{backend_key}_BASE_PATH is not set. Check .env / BASE_DIR.")
    if metadata_path is None:
        raise RuntimeError(f"Env var {dataset_key}_METADATA_PATH is not set. Check .env / BASE_DIR.")

    # Map each feature name to its parquet path and the dataloader kwarg key
    FEATURE_PATHS = {
        "hand_landmarks":   ("hand_landmark_path",    os.path.join(base_path, "hand_landmarks.parquet")),
        "pose_landmarks":   ("pose_landmark_path",    os.path.join(base_path, "pose_landmarks.parquet")),
        "face_landmarks":   ("face_landmark_path",    os.path.join(base_path, "face_landmarks.parquet")),
        "hand_angles":      ("hand_angle_path",       os.path.join(base_path, "hand_angles.parquet")),
        "pose_angles":      ("pose_angle_path",       os.path.join(base_path, "pose_angles.parquet")),
        "face_blendshapes": ("face_blendshape_path",  os.path.join(base_path, "face_blendshapes.parquet")),
    }

    common = {
        "metadata_json_path": metadata_path,
        "features": args.features,
        "n_glosses": args.n_glosses,
        "fs": args.fs,
        "feature_selection_dir": fs_dir if args.fs else None,
    }
    for feat, (kwarg, path) in FEATURE_PATHS.items():
        common[kwarg] = path

    bar = "=" * 80
    print(bar)
    print(f"Dataset : {args.dataset_name} | Backend: {args.feature_extraction_model}")
    print(f"Features: {args.features}")
    print(f"fs={bool(args.fs)} | n_glosses={args.n_glosses}")
    print(f"BASE    : {base_path}")
    print(bar)

    # Resolve and print every path the dataset will actually read; aggregate any misses.
    print("\nResolved paths:")
    checks = [("metadata", metadata_path, True)]
    if args.fs:
        checks.append(("top_features_dir", fs_dir, True))
    for feat in args.features:
        if feat not in FEATURE_PATHS:
            raise ValueError(f"Unknown feature '{feat}'. Valid: {list(FEATURE_PATHS)}")
        _, path = FEATURE_PATHS[feat]
        checks.append((feat, path, True))

    missing = []
    for label, path, required in checks:
        if path is None:
            status = "UNSET"
        elif os.path.isdir(path) or os.path.isfile(path):
            status = "OK"
        else:
            status = "MISS"
        print(f"  [{status:5s}]  {label:18s}  {path}")
        if required and status != "OK":
            missing.append((label, path, status))

    if missing:
        print()
        msg = ["One or more required paths are missing:"]
        for label, path, status in missing:
            msg.append(f"  - [{status}] {label}: {path}")
        msg.append("Check the BASE_DIR in .env and the uploaded dataset layout.")
        raise FileNotFoundError("\n".join(msg))

    sizes = {}
    train_set = None
    for split in ["train", "val", "test"]:
        ds = SignLanguageFeaturesDataset(**common, split=split)
        sizes[split] = len(ds)
        if split == "train":
            train_set = ds

    print()
    print(bar)
    print("DATASET SUMMARY")
    print("-" * 80)
    print(f"Splits          : train={sizes['train']}, val={sizes['val']}, test={sizes['test']}")
    print(f"# classes       : {len(train_set.gloss2idx)}")
    print(f"feature_dim     : {train_set.feature_dim}")
    print(f"max_len (frames): {train_set.max_len}")

    print()
    print(f"First {min(10, len(train_set.idx2gloss))} gloss labels:")
    for i in range(min(10, len(train_set.idx2gloss))):
        print(f"  [{i:2d}] {train_set.idx2gloss[i]}")

    print()
    print(f"Feature columns ({len(train_set.final_columns)} total):")
    for line in _format_columns(train_set.final_columns, args.max_cols_shown):
        print(line)

    print()
    x, y = train_set[0]
    label_idx = int(y)
    print("Sample at idx=0:")
    print(f"  features.shape : {tuple(x.shape)}")
    print(f"  features.dtype : {x.dtype}")
    print(f"  label          : {label_idx}  ('{train_set.idx2gloss[label_idx]}')")
    preview = x[0, : min(8, x.shape[1])].tolist()
    preview_str = ", ".join(f"{v:.3f}" for v in preview)
    suffix = " ..." if x.shape[1] > 8 else ""
    print(f"  features[0,:8] : [{preview_str}{suffix}]")

    print()
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    xb, yb = next(iter(loader))
    print(f"DataLoader (batch_size={args.batch_size}):")
    print(f"  batch.shape    : {tuple(xb.shape)}")
    print(f"  labels.shape   : {tuple(yb.shape)}")

    print(bar)
    print("[OK] Dataset loads cleanly.")


if __name__ == "__main__":
    main()
