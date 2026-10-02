"""FG2027 orchestration; reuse the existing models and train/evaluate loops."""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

from src.dataset.FGFeaturesDataset import FeatureStore, FGFeaturesDataset, MODALITIES, collate, sha256
from src.models.BiLSTM import BiLSTMClassifier
from src.models.EncoderOnlyTransformer import EncoderOnly
from src.models.SPOTER import SPOTERTransformer
from src.models.FGLateFusion import FGLateFusion
from src.utils import train_epoch_batch, evaluate_batch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def code_hash():
    digest = hashlib.sha256()
    paths = list(Path('src').rglob('*.py')) + list(Path('tests').rglob('*.py')) + [Path('requirements-fg2027.txt')]
    for path in sorted(paths):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def matrix(config):
    for dataset, model, modality, seed in itertools.product(config['datasets'],
            ['bilstm', 'spoter', 'encoder', 'latefusion'], ['H', 'HP', 'HPF'], config['seeds']):
        if model == 'latefusion' and modality == 'H':
            continue
        yield {'dataset': dataset, 'model': model, 'modalities': modality, 'seed': seed}


def run_name(spec):
    return f"{spec['dataset']}/{spec['model']}_{spec['modalities']}/seed_{spec['seed']}"


def build_model(spec, store, config):
    dims = [len(store.columns[m]) for m in MODALITIES[spec['modalities']]]
    args = dict(hidden_dim=config['hidden_dim'], n_heads=config['n_heads'],
                num_layers=config['n_layers'], dropout=config['dropout'], pe=config['pe'],
                max_len=max(map(len, store.frames.values())))
    if spec['model'] == 'bilstm':
        return BiLSTMClassifier(sum(dims), config['hidden_dim'], 100,
                                num_layers=config['bilstm_layers'], dropout=config['dropout'])
    if spec['model'] == 'latefusion':
        return FGLateFusion(dims, 100, **args)
    cls = {'encoder': EncoderOnly, 'spoter': SPOTERTransformer}[spec['model']]
    return cls(input_dim=sum(dims), num_classes=100, **args)


def environment():
    return {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
            'source_sha256': code_hash()}


def smoke(store, config, dataset, out):
    from tests.fg_smoke import check_models, check_data, check_reporting
    seed_all(config['seeds'][0])
    check_data()
    check_reporting()
    check_models()
    device = 'cuda'
    for spec in matrix(config):
        if spec['dataset'] != dataset or spec['seed'] != config['seeds'][0]:
            continue
        data = FGFeaturesDataset(store, 'train', spec['modalities'])
        loader = DataLoader(Subset(data, range(min(2, len(data)))), batch_size=2, collate_fn=collate)
        batch, labels, lengths = next(iter(loader))
        # Diagnostic-only short sequences; production runs never truncate.
        batch, lengths = batch[:, :32], lengths.clamp_max(32)
        model = build_model(spec, store, config).to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        train_epoch_batch(model, [(batch, labels, lengths)], loss_fn, optimizer, device)
        valid = FGFeaturesDataset(store, 'val', spec['modalities'])
        vx, vy, vl = next(iter(DataLoader(Subset(valid, range(min(2, len(valid)))), batch_size=2, collate_fn=collate)))
        evaluate_batch(model, loss_fn, [(vx[:, :32], vy, vl.clamp_max(32))], device)
        print('SMOKE PASS', spec['model'], spec['modalities'], flush=True)
        del model, optimizer
        torch.cuda.empty_cache()
    write_json(out / dataset / 'smoke_pass.json', {'inputs': store.hashes,
               'source_sha256': code_hash(), 'config': config, 'status': 'passed'})


def train_run(spec, store, config, out, resume=False):
    run_dir = out / run_name(spec)
    signature = {'experiment': spec, 'config': config, 'input_sha256': store.hashes,
                 'source_sha256': code_hash()}
    if (run_dir / 'manifest.json').exists():
        previous = json.loads((run_dir / 'manifest.json').read_text())
        if previous != signature:
            raise ValueError(f'Run directory has a different config/data/code: {run_dir}')
        if (run_dir / 'metrics.json').exists():
            print('SKIP completed', run_dir, flush=True)
            return
        if not resume:
            raise ValueError(f'Incomplete run exists; inspect then pass --resume: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / 'manifest.json', signature)
    current_env = environment()
    if (run_dir / 'environment.json').exists():
        previous_env = json.loads((run_dir / 'environment.json').read_text())
        if any(previous_env[k] != current_env[k] for k in ('python', 'torch', 'cuda', 'gpu')):
            raise ValueError('Runtime changed; use the recorded environment before resuming')
    else:
        write_json(run_dir / 'environment.json', current_env)
    (run_dir / 'pip-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    seed_all(spec['seed'])
    device = 'cuda'
    train_set = FGFeaturesDataset(store, 'train', spec['modalities'])
    val_set = FGFeaturesDataset(store, 'val', spec['modalities'])
    shuffle_rng = torch.Generator().manual_seed(spec['seed'])
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, collate_fn=collate, generator=shuffle_rng)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], collate_fn=collate)
    model = build_model(spec, store, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    loss_fn = torch.nn.CrossEntropyLoss()
    start_epoch, best_accuracy, best_epoch, history, elapsed = 0, -1., -1, [], 0.
    last_path = run_dir / 'last.pt'
    if resume and last_path.exists():
        # Only load checkpoints produced by this pipeline in the user's run directory.
        state = torch.load(last_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(device)
        start_epoch, best_accuracy, best_epoch = state['epoch'] + 1, state['best_accuracy'], state['best_epoch']
        history, elapsed = state['history'], state['elapsed']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        shuffle_rng.set_state(state['shuffle_rng'])
    torch.cuda.reset_peak_memory_stats()
    write_json(run_dir / 'status.json', {'status': 'running', 'next_epoch': start_epoch})
    for epoch in range(start_epoch, config['epochs']):
        started = time.time()
        train_loss, train_acc = train_epoch_batch(model, train_loader, loss_fn, optimizer, device,
                                                 clip_gradients=config['clip_gradients'])
        val_loss, val_acc, _ = evaluate_batch(model, loss_fn, val_loader, device)
        elapsed += time.time() - started
        row = {'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc,
               'val_loss': val_loss, 'val_acc': val_acc, 'lr': optimizer.param_groups[0]['lr']}
        history.append(row)
        print(run_name(spec), json.dumps(row), flush=True)
        if val_acc > best_accuracy:
            best_accuracy, best_epoch = val_acc, epoch + 1
            torch.save(model.state_dict(), run_dir / 'best.tmp')
            (run_dir / 'best.tmp').replace(run_dir / 'best.pt')
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'epoch': epoch, 'history': history,
                 'best_accuracy': best_accuracy, 'best_epoch': best_epoch, 'elapsed': elapsed,
                 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                 'shuffle_rng': shuffle_rng.get_state()}
        torch.save(state, run_dir / 'last.tmp')
        (run_dir / 'last.tmp').replace(last_path)
        write_json(run_dir / 'history.json', history)
    # Test is evaluated only after the validation-selected checkpoint is loaded.
    model.load_state_dict(torch.load(run_dir / 'best.pt', map_location=device, weights_only=True))
    test_set = FGFeaturesDataset(store, 'test', spec['modalities'])
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], collate_fn=collate)
    _, _, (truth, pred) = evaluate_batch(model, loss_fn, test_loader, device, return_preds=True)
    metrics = {**spec, 'accuracy': accuracy_score(truth, pred),
               'macro_f1': f1_score(truth, pred, labels=list(range(100)), average='macro', zero_division=0),
               'weighted_f1': f1_score(truth, pred, labels=list(range(100)), average='weighted', zero_division=0),
               'best_val_accuracy': best_accuracy, 'best_epoch': best_epoch,
               'parameters': sum(p.numel() for p in model.parameters()), 'training_seconds': elapsed,
               'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(), 'n_test': len(test_set)}
    if isinstance(model, FGLateFusion):
        # Learned contribution of each modality in the selected checkpoint (softmax, sums to 1).
        metrics['fusion_weights'] = dict(zip(MODALITIES[spec['modalities']],
                                             model.fusion_weights.softmax(0).tolist()))
    with (run_dir / 'predictions.csv').open('w') as stream:
        writer = csv.writer(stream); writer.writerow(['video_id', 'y_true', 'y_pred'])
        writer.writerows(zip(test_set.ids, truth, pred))
    write_json(run_dir / 'metrics.json', metrics)
    write_json(run_dir / 'status.json', {'status': 'completed'})


def summarize(config, out):
    rows = []
    for spec in matrix(config):
        path = out / run_name(spec) / 'metrics.json'
        rows.append({**spec, 'status': 'completed' if path.exists() else (json.loads((path.parent / 'status.json').read_text())['status'] if (path.parent / 'status.json').exists() else 'pending'),
                     **(json.loads(path.read_text()) if path.exists() else {})})
    fields = list(dict.fromkeys(k for row in rows for k in row))
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    grouped = []
    for dataset, model, modality in dict.fromkeys((r['dataset'], r['model'], r['modalities']) for r in rows):
        available = [r for r in rows if (r['dataset'], r['model'], r['modalities']) == (dataset, model, modality)
                     and r['status'] == 'completed']
        line = [dataset, model, modality, str(len(available)) + '/' + str(len(config['seeds']))]
        for metric in ('accuracy', 'macro_f1', 'weighted_f1'):
            values = [r[metric] * 100 for r in available]
            if len(values) >= 2:
                line.append(f'{np.mean(values):.2f} ± {np.std(values, ddof=1):.2f}')
            else:
                line.append(f'{values[0]:.2f}' if values else 'Pending')
        parameters = [r['parameters'] for r in available if 'parameters' in r]
        line.append(f'{parameters[0] / 1e6:.2f}M' if parameters else 'Pending')
        grouped.append('| ' + ' | '.join(line) + ' |')
    (out / 'results.md').write_text('| Dataset | Model | Input | Seeds | Accuracy | Macro F1 | Weighted F1 | Parameters |\n'
        '|---|---|---|---|---|---|---|---|\n' + '\n'.join(grouped) +
        '\n\nOne completed seed: score only. Multiple completed seeds: mean ± sample SD, not confidence intervals.\n')
    print(out / 'results.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['audit', 'smoke', 'run', 'summarize'])
    parser.add_argument('--config', default='configs/fg2027.json')
    parser.add_argument('--dataset'); parser.add_argument('--model'); parser.add_argument('--modalities')
    parser.add_argument('--seed', type=int); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(config['output_dir'])
    if args.action == 'summarize':
        summarize(config, out); return
    if 'COLAB_RELEASE_TAG' not in os.environ:
        raise RuntimeError('This project is configured for Google Colab only. Run the notebook there.')
    if args.action != 'audit' and not torch.cuda.is_available():
        raise RuntimeError('Select a Colab GPU runtime before smoke tests or training.')
    specs = [s for s in matrix(config) if all(getattr(args, k) is None or s[k] == getattr(args, k)
                                            for k in ('dataset', 'model', 'modalities', 'seed'))]
    if not specs:
        raise ValueError('No matching experiment configurations')
    for dataset in dict.fromkeys(s['dataset'] for s in specs):
        # Combine this dataset's paths with the shared pose selection.
        store_config = {
            **config['datasets'][dataset],
            'pose_indices': config['pose_indices'],
        }
        store = FeatureStore(store_config)
        report = store.report()
        write_json(out / dataset / 'data_audit.json', report)
        print(dataset, report['counts'], report['missingness'], flush=True)
        print('Person selection:', {k: v for k, v in report['person_selection'].items() if k != 'choices'}, flush=True)
        if args.action == 'audit':
            continue
        if args.action == 'smoke':
            smoke(store, config, dataset, out); continue
        gate = out / dataset / 'smoke_pass.json'
        if not gate.exists():
            raise RuntimeError('Run smoke tests first')
        checked = json.loads(gate.read_text())
        if checked['inputs'] != store.hashes or checked['source_sha256'] != code_hash() or checked['config'] != config:
            raise RuntimeError('Data/code/config changed since smoke tests; rerun them')
        if not config['protocol_reviewed']:
            raise RuntimeError('Review data audit and provisional protocol decisions, then set protocol_reviewed=true and rerun smoke')
        for spec in (s for s in specs if s['dataset'] == dataset):
            try:
                train_run(spec, store, config, out, args.resume)
            except Exception as error:
                error_dir = out / run_name(spec)
                if not (error_dir / 'metrics.json').exists():
                    write_json(error_dir / 'status.json', {'status': 'failed', 'error': repr(error)})
                raise
    if args.action == 'run':
        summarize(config, out)


if __name__ == '__main__':
    main()
