"""Smoke tests (need a CUDA GPU): meaningful masking, padding, alignment, and leakage checks."""
import torch
from src import console
from src.models.EncoderOnlyTransformer import EncoderOnly
from src.models.SPOTER import SPOTERTransformer
from src.models.BiLSTM import BiLSTMClassifier
from src.models.FGLateFusion import FGLateFusion


def check_reporting():
    """A completed single-seed run must show its score, not Pending or a fake SD."""
    import csv
    import json
    import tempfile
    from pathlib import Path
    from src.fg2027 import matrix, run_name, summarize
    config = {'datasets': {'wlasl': {}, 'avasag': {}}, 'seeds': [379]}
    specs = list(matrix(config))
    assert len(specs) == 22 and len({run_name(s) for s in specs}) == 22
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        spec = specs[0]
        path = out / run_name(spec) / 'metrics.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({**spec, 'accuracy': .5, 'macro_f1': .4, 'weighted_f1': .45,
                                    'parameters': 7_940_000}))
        summarize(config, out)
        rows = list(csv.DictReader((out / 'results.csv').open()))
        assert len(rows) == 22 and sum(r['status'] == 'completed' for r in rows) == 1
        assert '| 1/1 | 50.00 | 40.00 | 45.00 | 7.94M |' in (out / 'results.md').read_text()
    console.line(f"  {console.green('✓')} Single-seed matrix and reporting")


def check_models():
    torch.manual_seed(379)
    for dim, parts in [(84, [84]), (110, [84, 26]), (246, [84, 26, 136])]:
        args = dict(hidden_dim=16, n_heads=2, num_layers=1, dropout=0., pe='none')
        models = [EncoderOnly(dim, 100, **args), SPOTERTransformer(dim, 100, **args),
                  BiLSTMClassifier(dim, 16, 100)]
        if len(parts) > 1:
            models.append(FGLateFusion(parts, 100, **args))
        x = torch.rand(2, 7, dim)
        x[0, 3:] = -2
        x[1, 2] = -2  # Actual missing frame INSIDE sequence; not padding.
        lengths = torch.tensor([3, 7])
        for model in models:
            model = model.cuda().eval()
            values = x.cuda()
            with torch.no_grad():
                output = model(values, lengths)
                padded = torch.cat([values, values.new_full((2, 5, dim), -2)], dim=1)
                assert output.shape == (2, 100) and torch.isfinite(output).all()
                torch.testing.assert_close(output, model(padded, lengths), atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(output[:1], model(values[:1, :3], lengths[:1]), atol=2e-5, rtol=2e-5)
                missing = values.new_full((2, 7, dim), -2.)
                assert torch.isfinite(model(missing, lengths)).all()
                if isinstance(model, EncoderOnly):
                    order = torch.tensor([6, 2, 0, 4, 1, 5, 3], device='cuda')
                    torch.testing.assert_close(output[1:], model(values[1:, order], lengths[1:]), atol=2e-5, rtol=2e-5)
            model.train()
            loss = torch.nn.functional.cross_entropy(model(values, lengths), torch.tensor([1, 2], device='cuda'))
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    console.line(f"  {console.green('✓')} Model masking, missing-frame, padding invariance, gradients")


def check_data():
    import json
    import tempfile
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from src.dataset.FGFeaturesDataset import FeatureStore, FGFeaturesDataset, collate
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        metadata = [{'gloss': f'g{i:03}', 'instances': [
            {'video_id': f'{i}_{s}', 'split': s} for s in ('train', 'val', 'test')]} for i in range(100)]
        (root / 'metadata.json').write_text(json.dumps(metadata))
        originals = {}
        for modality, prefix, count in [('hand_landmarks','h',42), ('pose_landmarks','p',13), ('face_landmarks','f',68)]:
            rows = []
            for entry in metadata:
                for inst in entry['instances']:
                    for frame in [2, 0, 1]:
                        row = {'video_id': inst['video_id'], 'frame': frame, 'gloss': entry['gloss'], 'person_id': 0}
                        row.update({f'{prefix}{i}_{axis}': frame / 10 for i in range(count) for axis in 'xy'})
                        rows.append(row)
            df = pd.DataFrame(rows)
            # Missing face row must not shift the remaining frames.
            if prefix == 'f':
                df = df[~((df.video_id == '0_train') & (df.frame == 1))]
            originals[modality] = df
            df.to_parquet(root / f'{modality}.parquet')
        cfg = {'metadata': str(root / 'metadata.json'), 'features_dir': str(root), 'pose_indices': list(range(13))}
        store = FeatureStore(cfg)
        ds = FGFeaturesDataset(store, 'train', 'HPF')
        values, label = ds[0]
        assert values.shape == (3, 246) and label == 0
        np.testing.assert_allclose(values[:, 0], [0, .1, .2])
        assert (values[1, 110:] == -2).all() and (values[2, 110:] == .2).all()
        batch, labels, lengths = collate([ds[0], ds[1]])
        assert batch.shape == (2, 3, 246) and lengths.tolist() == [3, 3]
        for modality, dim in [('H',84), ('HP',110), ('HPF',246)]:
            selected = FGFeaturesDataset(store, 'train', modality)
            assert selected.ids == ds.ids and selected.feature_dim == dim
        path = root / 'hand_landmarks.parquet'
        duplicate = pd.concat([originals['hand_landmarks'], originals['hand_landmarks'].iloc[:1]])
        duplicate.to_parquet(path)
        try:
            FeatureStore(cfg)
        except ValueError as error:
            assert 'duplicate rows' in str(error)
        else:
            raise AssertionError('Duplicate frames were accepted')
        # A second person in one frame of every modality. The person nearer the horizontal
        # image centre is kept for all modalities, whatever its ID; no frame is lost.
        frame = 2  # the signer's coordinates are 0.2 here, i.e. 0.3 from the centre

        def add_person(x):
            for modality, original in originals.items():
                extra = original[(original.video_id == '0_train') & (original.frame == frame)].assign(person_id=1)
                extra[[c for c in extra if c.endswith(('_x', '_y'))]] = x
                pd.concat([original, extra]).sample(frac=1, random_state=1).to_parquet(root / f'{modality}.parquet')
            return FeatureStore(cfg)
        central = add_person(.5)
        assert central.frames['0_train'].shape == (3, 246) and (central.frames['0_train'][frame] == .5).all()
        assert central.person_choices == [['0_train', frame, 1]]
        at_edge = add_person(.95)
        assert (at_edge.frames['0_train'][frame] == np.float32(.2)).all()
        assert at_edge.person_choices == [['0_train', frame, 0]] and at_edge.splits == store.splits
        for modality, original in originals.items():
            original.to_parquet(root / f'{modality}.parquet')
        # Points slightly past the frame edge are kept unchanged; unscaled pixels are rejected.
        edge = originals['hand_landmarks'].copy()
        edge.loc[edge.index[0], 'h0_y'] = 1.08
        edge.to_parquet(path)
        assert FeatureStore(cfg).frames['0_train'][frame, 1] == np.float32(1.08)
        edge.loc[edge.index[0], 'h0_y'] = 540.
        edge.to_parquet(path)
        try:
            FeatureStore(cfg)
        except ValueError as error:
            assert 'far outside' in str(error)
        else:
            raise AssertionError('Unscaled pixel coordinates were accepted')
        originals['hand_landmarks'].to_parquet(path)
        metadata[0]['instances'][2]['video_id'] = metadata[0]['instances'][0]['video_id']
        (root / 'metadata.json').write_text(json.dumps(metadata))
        try:
            FeatureStore(cfg)
        except ValueError as error:
            assert 'Duplicate video ID' in str(error)
        else:
            raise AssertionError('Split overlap was accepted')
    console.line(f"  {console.green('✓')} Frame alignment, modality widths, common cohort, duplicate/leakage rejection")
