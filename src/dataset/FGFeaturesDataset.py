"""Strict ViTPose loading: explicit frame alignment and identical samples for all ablations."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

MODALITIES = {'H': ['hand_landmarks'], 'HP': ['hand_landmarks', 'pose_landmarks'],
              'HPF': ['hand_landmarks', 'pose_landmarks', 'face_landmarks']}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def keep_central_person(tables, pose_columns):
    """Keep one person per frame across all modalities.

    In frames with several detected people, keep the person whose visible pose
    points are on average closest to the horizontal image centre (ties: lowest
    person ID). The choice comes from pose and applies to every modality, so
    hands, pose and face always come from the same person. If no candidate has
    a visible pose point, no detection is kept and the frame becomes missing.
    Returns the filtered tables and one [video_id, frame, person_id or None]
    record per affected frame.
    """
    keys = ['video_id', 'frame']
    people = pd.concat([t[keys + ['person_id']] for t in tables.values()]).drop_duplicates()
    multi = people[people.duplicated(keys, keep=False)][keys].drop_duplicates()
    if multi.empty:
        return tables, []
    pose = tables['pose_landmarks'].merge(multi, on=keys)
    x = pose[pose_columns[0::2]].to_numpy(dtype=float)  # columns alternate _x, _y
    seen = x != -2
    centre = np.where(seen, x, 0).sum(1) / np.maximum(seen.sum(1), 1)
    pose = pose.assign(offset=np.where(seen.any(1), np.abs(centre - .5), np.nan))
    chosen = (pose.dropna(subset=['offset']).sort_values(keys + ['offset', 'person_id'])
              .drop_duplicates(keys)[keys + ['person_id']])
    multi_index, chosen_index = pd.MultiIndex.from_frame(multi), pd.MultiIndex.from_frame(chosen)
    selected = {}
    for modality, df in tables.items():
        in_multi = pd.MultiIndex.from_frame(df[keys]).isin(multi_index)
        is_chosen = pd.MultiIndex.from_frame(df[keys + ['person_id']]).isin(chosen_index)
        selected[modality] = df[~in_multi | is_chosen]
    decisions = multi.merge(chosen, on=keys, how='left')
    records = [[v, int(f), None if pd.isna(p) else int(p)] for v, f, p in decisions.itertuples(index=False)]
    return selected, records


class FeatureStore:
    def __init__(self, config):
        """Load features using metadata, features_dir, and pose_indices keys.

        The caller combines config['datasets'][dataset] from the experiment
        configuration with its top-level pose_indices before passing it here.
        """
        self.config = config
        self.columns = {
            'hand_landmarks': [f'h{i}_{axis}' for i in range(42) for axis in 'xy'],
            'pose_landmarks': [f'p{i}_{axis}' for i in config['pose_indices'] for axis in 'xy'],
            'face_landmarks': [f'f{i}_{axis}' for i in range(68) for axis in 'xy'],
        }
        metadata = json.loads(Path(config['metadata']).read_text())[:100]
        if len(metadata) != 100 or len({e['gloss'] for e in metadata}) != 100:
            raise ValueError('Expected 100 unique glosses in the selected metadata.')
        self.gloss2idx = {name: i for i, name in enumerate(sorted(e['gloss'] for e in metadata))}
        self.splits = {s: [] for s in ('train', 'val', 'test')}
        labels = {}
        for entry in metadata:
            for inst in entry['instances']:
                video_id, split = str(inst['video_id']), inst['split']
                if video_id in labels:
                    raise ValueError(f'Duplicate video ID across/within splits: {video_id}')
                if split not in self.splits:
                    raise ValueError(f'Unexpected split: {split}')
                labels[video_id] = entry['gloss']
                self.splits[split].append(video_id)
        for split, ids in self.splits.items():
            if not ids:
                raise ValueError(f'Empty split: {split}')
        train_labels = {labels[v] for v in self.splits['train']}
        if train_labels != set(self.gloss2idx):
            raise ValueError('Every class must have a training example.')
        self.labels = labels
        self.hashes = {'metadata': sha256(config['metadata'])}
        tables = {}
        self.missing = {}
        self.outside_unit_range = {}
        # Load every modality even for H: timeline and cohort stay identical across ablations.
        for modality, columns in self.columns.items():
            path = Path(config['features_dir']) / f'{modality}.parquet'
            self.hashes[modality] = sha256(path)
            df = pd.read_parquet(path)
            required = ['video_id', 'frame', 'gloss'] + columns
            absent = set(required) - set(df.columns)
            if absent:
                raise ValueError(f'{path.name}: missing columns {sorted(absent)}. Do not infer frame IDs from row order.')
            df['video_id'] = df.video_id.astype(str)
            # Keep only rows for video IDs in the selected metadata.
            df = df[df.video_id.isin(labels)].copy()
            if df.empty:
                raise ValueError(f'{path.name}: no matching video IDs')
            if df.frame.isna().any() or not np.isfinite(df.frame.to_numpy(dtype=float)).all():
                raise ValueError('Invalid frame IDs')
            if (df.frame < 0).any() or (df.frame % 1 != 0).any():
                raise ValueError('Frame IDs must be non-negative integers')
            if 'person_id' not in df:
                df['person_id'] = 0
            if (df.person_id.isna().any() or not np.isfinite(df.person_id.to_numpy(dtype=float)).all() or
                    (df.person_id < -1).any() or (df.person_id % 1 != 0).any()):
                raise ValueError(f'{path.name}: person IDs must be integers >= -1')
            df['frame'], df['person_id'] = df.frame.astype('int64'), df.person_id.astype('int64')
            if df.duplicated(['video_id', 'frame', 'person_id']).any():
                raise ValueError(f'{path.name}: duplicate rows for the same video/frame/person')
            if not (df.gloss == df.video_id.map(labels)).all():
                raise ValueError(f'{path.name}: gloss/metadata mismatch')
            values = df[columns].to_numpy(dtype=np.float32)
            values = np.where(np.isnan(values), -2., values)
            if not np.isfinite(values).all():
                raise ValueError(f'{path.name}: infinite feature values')
            observed = values[values != -2]
            # ViTPose can place points slightly past the frame edge; keep them unchanged.
            # The wider limit still catches coordinates that were never scaled to [0,1].
            if ((observed < -0.5) | (observed > 1.5)).any():
                raise ValueError(f'{path.name}: coordinates far outside [0,1]; verify normalization before training')
            self.outside_unit_range[modality] = float(((observed < 0) | (observed > 1)).mean())
            df[columns] = values
            tables[modality] = df[['video_id', 'frame', 'person_id'] + columns]
        keys = ['video_id', 'frame']
        all_frames = pd.MultiIndex.from_frame(pd.concat([t[keys] for t in tables.values()]).drop_duplicates())
        tables, self.person_choices = keep_central_person(tables, self.columns['pose_landmarks'])
        merged = None
        for modality, columns in self.columns.items():
            # Index by video ID and frame ID to align modalities on the same frames.
            part = tables[modality].set_index(keys)[columns]
            merged = part if merged is None else merged.join(part, how='outer', validate='one_to_one')
        # Keep every original frame, sorted by video ID then frame ID; mark missing coordinates as -2.
        merged = merged.reindex(all_frames).sort_index().fillna(-2.)
        self.frames = {}
        all_columns = sum(self.columns.values(), [])
        for video_id, rows in merged.groupby(level='video_id', sort=False):
            values = rows[all_columns].to_numpy(dtype=np.float32)
            if (values == -2).all():
                raise ValueError(f'Video {video_id} has feature rows, but all coordinates are missing (-2).')
            self.frames[video_id] = values
        absent_ids = set(labels) - set(self.frames)
        if absent_ids:
            raise ValueError(f'{len(absent_ids)} videos listed in metadata have no feature rows in any modality. Examples: {sorted(absent_ids)[:5]}')
        self.indices = {}
        offset = 0
        for modality, columns in self.columns.items():
            self.indices[modality] = list(range(offset, offset + len(columns)))
            offset += len(columns)
        for modality, idx in self.indices.items():
            self.missing[modality] = {
                'missing_coordinate_fraction': float(sum((v[:, idx] == -2).sum() for v in self.frames.values()) /
                    sum(v[:, idx].size for v in self.frames.values())),
                'fully_missing_clips': sum(bool((v[:, idx] == -2).all()) for v in self.frames.values())}

    def report(self):
        return {'sha256': self.hashes, 'split_ids': self.splits, 'gloss2idx': self.gloss2idx,
                'columns': self.columns, 'missingness': self.missing,
                'person_selection': {
                    'rule': 'in frames with several people, keep the one whose visible pose points '
                            'are on average closest to the horizontal image centre; same person for all modalities',
                    'frames': len(self.person_choices),
                    'chosen_person_ids': dict(Counter(str(p) for _, _, p in self.person_choices)),
                    'choices': self.person_choices},
                'fraction_outside_0_1': self.outside_unit_range,
                'counts': {s: len(v) for s, v in self.splits.items()},
                'length_min': min(map(len, self.frames.values())),
                'length_max': max(map(len, self.frames.values()))}


class FGFeaturesDataset(Dataset):
    def __init__(self, store, split, modalities):
        """Select a FeatureStore split ('train', 'val', 'test') and inputs ('H', 'HP', 'HPF')."""
        self.store, self.ids = store, store.splits[split]
        # Concatenate the column-index lists for the requested modalities.
        self.indices = sum((store.indices[m] for m in MODALITIES[modalities]), [])
        self.feature_dim = len(self.indices)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        video_id = self.ids[index]
        # Keep missing observations as -2; collate records lengths before padding.
        x = torch.from_numpy(self.store.frames[video_id][:, self.indices].copy())
        # x: tensor shaped [frames, selected_features], e.g. [30, 110] for HP with 13 pose landmarks.
        # The second return value is the integer class ID for this video's gloss.
        return x, self.store.gloss2idx[self.store.labels[video_id]]


def collate(batch):
    """Batch (x, label) pairs: x is [frames, features], label is an integer class ID."""
    # Unpack the pairs into a tuple of video tensors and a tuple of labels.
    x, labels = zip(*batch)
    # Record each video's frame count before adding batch padding.
    lengths = torch.tensor([len(v) for v in x], dtype=torch.long)
    # Return padded videos [B, T, D], labels [B], and lengths [B].
    # B = batch size, T = longest video in this batch, D = selected feature count.
    return pad_sequence(x, batch_first=True, padding_value=-2.), torch.tensor(labels), lengths
