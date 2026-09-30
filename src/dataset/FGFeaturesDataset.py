"""Strict ViTPose loading: explicit frame alignment and identical samples for all ablations."""
import hashlib
import json
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


class FeatureStore:
    def __init__(self, config):
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
        merged = None
        self.missing = {}
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
            df = df[df.video_id.isin(labels)].copy()
            if df.empty:
                raise ValueError(f'{path.name}: no matching video IDs')
            if df.frame.isna().any() or not np.isfinite(df.frame.to_numpy(dtype=float)).all():
                raise ValueError('Invalid frame IDs')
            if (df.frame < 0).any() or (df.frame % 1 != 0).any():
                raise ValueError('Frame IDs must be non-negative integers')
            if df.duplicated(['video_id', 'frame']).any():
                raise ValueError(f'{path.name}: multiple people/rows in a frame; signer selection must be resolved first.')
            if 'person_id' in df:
                if df.person_id.isna().any() or not df.person_id.isin([-1, 0]).all():
                    raise ValueError('Unexpected person IDs; inspect extraction/signer selection.')
            if not (df.gloss == df.video_id.map(labels)).all():
                raise ValueError(f'{path.name}: gloss/metadata mismatch')
            values = df[columns].to_numpy(dtype=np.float32)
            values = np.where(np.isnan(values), -2., values)
            if not np.isfinite(values).all():
                raise ValueError(f'{path.name}: infinite feature values')
            observed = values != -2
            if ((values[observed] < 0) | (values[observed] > 1)).any():
                raise ValueError(f'{path.name}: coordinates outside [0,1]; verify normalization before training')
            df[columns] = values
            part = df.set_index(['video_id', 'frame'])[columns]
            merged = part if merged is None else merged.join(part, how='outer', validate='one_to_one')
        merged = merged.sort_index().fillna(-2.)
        self.frames = {}
        all_columns = sum(self.columns.values(), [])
        for video_id, rows in merged.groupby(level='video_id', sort=False):
            values = rows[all_columns].to_numpy(dtype=np.float32)
            if (values == -2).all():
                raise ValueError(f'Completely missing clip: {video_id}; resolve cohort policy before training')
            self.frames[video_id] = values
        absent_ids = set(labels) - set(self.frames)
        if absent_ids:
            raise ValueError(f'{len(absent_ids)} metadata clips lack all features; examples: {sorted(absent_ids)[:5]}')
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
                'counts': {s: len(v) for s, v in self.splits.items()},
                'length_min': min(map(len, self.frames.values())),
                'length_max': max(map(len, self.frames.values()))}


class FGFeaturesDataset(Dataset):
    def __init__(self, store, split, modalities):
        self.store, self.ids = store, store.splits[split]
        self.indices = sum((store.indices[m] for m in MODALITIES[modalities]), [])
        self.feature_dim = len(self.indices)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        video_id = self.ids[index]
        # Missing observations stay -2; true sequence lengths are passed separately.
        x = torch.from_numpy(self.store.frames[video_id][:, self.indices].copy())
        return x, self.store.gloss2idx[self.store.labels[video_id]]


def collate(batch):
    x, labels = zip(*batch)
    lengths = torch.tensor([len(v) for v in x], dtype=torch.long)
    return pad_sequence(x, batch_first=True, padding_value=-2.), torch.tensor(labels), lengths
