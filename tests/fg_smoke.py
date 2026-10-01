"""Run on Colab: meaningful masking, padding, alignment, and leakage checks."""
import torch
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
        path.write_text(json.dumps({**spec, 'accuracy': .5, 'macro_f1': .4, 'weighted_f1': .45}))
        summarize(config, out)
        rows = list(csv.DictReader((out / 'results.csv').open()))
        assert len(rows) == 22 and sum(r['status'] == 'completed' for r in rows) == 1
        assert '| 1/1 | 50.00 | 40.00 | 45.00 |' in (out / 'results.md').read_text()
    print('Single-seed matrix and reporting: PASS')


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
    print('Model masking, missing-frame, padding invariance, gradients: PASS')


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
            assert 'multiple people/rows' in str(error)
        else:
            raise AssertionError('Duplicate frames were accepted')
        originals['hand_landmarks'].to_parquet(path)
        metadata[0]['instances'][2]['video_id'] = metadata[0]['instances'][0]['video_id']
        (root / 'metadata.json').write_text(json.dumps(metadata))
        try:
            FeatureStore(cfg)
        except ValueError as error:
            assert 'Duplicate video ID' in str(error)
        else:
            raise AssertionError('Split overlap was accepted')
    print('Frame alignment, modality widths, common cohort, duplicate/leakage rejection: PASS')
