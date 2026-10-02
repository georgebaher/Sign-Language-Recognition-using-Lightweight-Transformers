"""FG2027 orchestration; reuse the existing models and train/evaluate loops."""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

from src import console
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
        console.line(f"  {console.green('✓')} smoke  {spec['model']:<10} {spec['modalities']}")
        del model, optimizer
        torch.cuda.empty_cache()
    write_json(out / dataset / 'smoke_pass.json', {'inputs': store.hashes,
               'source_sha256': code_hash(), 'config': config, 'status': 'passed'})


def train_run(spec, store, config, out):
    run_dir = out / run_name(spec)
    signature = {'experiment': spec, 'config': config, 'input_sha256': store.hashes,
                 'source_sha256': code_hash()}
    if (run_dir / 'manifest.json').exists():
        previous = json.loads((run_dir / 'manifest.json').read_text())
        if previous != signature:
            raise ValueError(f'Run directory has a different config/data/code: {run_dir}')
        if (run_dir / 'metrics.json').exists():
            done = json.loads((run_dir / 'metrics.json').read_text())
            console.line(console.yellow(f"  ↷ already completed (test acc {done['accuracy']:.1%}); skipped"))
            return 'skipped'
        # No resuming: clear the interrupted run so nothing stale survives the restart.
        console.line(console.yellow('  ↻ incomplete run found; restarting from epoch 1'))
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / 'manifest.json', signature)
    write_json(run_dir / 'environment.json', environment())
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
    best_accuracy, best_epoch, history, elapsed = -1., -1, [], 0.
    torch.cuda.reset_peak_memory_stats()
    write_json(run_dir / 'status.json', {'status': 'running'})
    epochs = config['epochs']
    console.line(console.dim(f"  {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters · "
                             f"{len(train_set)} train / {len(val_set)} val clips"))
    console.line(console.dim(console.EPOCH_HEADER))
    for epoch in range(epochs):
        started = time.time()
        train_loss, train_acc = train_epoch_batch(model, train_loader, loss_fn, optimizer, device,
                                                 clip_gradients=config['clip_gradients'])
        val_loss, val_acc, _ = evaluate_batch(model, loss_fn, val_loader, device)
        elapsed += time.time() - started
        row = {'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc,
               'val_loss': val_loss, 'val_acc': val_acc, 'lr': optimizer.param_groups[0]['lr']}
        history.append(row)
        if val_acc > best_accuracy:
            best_accuracy, best_epoch = val_acc, epoch + 1
            torch.save(model.state_dict(), run_dir / 'best.tmp')
            (run_dir / 'best.tmp').replace(run_dir / 'best.pt')
        write_json(run_dir / 'history.json', history)
        # A permanent row every 10 epochs (and the first/last); a live progress line in between.
        progress = (row, epochs, best_accuracy, best_epoch, elapsed)
        if row['epoch'] in (1, epochs) or row['epoch'] % 10 == 0:
            console.line(console.epoch_row(*progress))
        console.live(console.epoch_live(*progress))
    # Test is evaluated only after the validation-selected checkpoint is loaded.
    model.load_state_dict(torch.load(run_dir / 'best.pt', map_location=device, weights_only=True))
    test_set = FGFeaturesDataset(store, 'test', spec['modalities'])
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], collate_fn=collate)
    _, _, (truth, pred) = evaluate_batch(model, loss_fn, test_loader, device, return_preds=True)
    # Average F1 over the classes present in the test set; absent classes would otherwise score 0.
    labels = np.unique(truth)
    metrics = {**spec, 'accuracy': accuracy_score(truth, pred),
               'macro_f1': f1_score(truth, pred, labels=labels, average='macro', zero_division=0),
               'weighted_f1': f1_score(truth, pred, labels=labels, average='weighted', zero_division=0),
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
    console.line(f"  {console.green('✓')} test acc {console.bold(f'{metrics['accuracy']:.1%}')}"
                 f"  macro-F1 {metrics['macro_f1']:.1%}  weighted-F1 {metrics['weighted_f1']:.1%}"
                 + console.dim(f"  ·  best val {best_accuracy:.1%} @{best_epoch}  ·  {console.duration(elapsed)}"
                               f"  ·  peak {metrics['peak_cuda_memory_bytes'] / 1e9:.2f} GB"))
    return 'completed'


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
        grouped.append(line)
    header = ['Dataset', 'Model', 'Input', 'Seeds', 'Accuracy', 'Macro F1', 'Weighted F1', 'Parameters']
    (out / 'results.md').write_text('| ' + ' | '.join(header) + ' |\n' + '|---' * len(header) + '|\n' +
        '\n'.join('| ' + ' | '.join(line) + ' |' for line in grouped) +
        '\n\nOne completed seed: score only. Multiple completed seeds: mean ± sample SD, not confidence intervals.\n')
    return header, grouped


def show_results(config, out):
    header, grouped = summarize(config, out)
    console.rule('Results')
    console.table(header, grouped)
    console.line(console.dim(f"\n  Saved to {out / 'results.md'} and {out / 'results.csv'}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['audit', 'smoke', 'run', 'summarize'])
    parser.add_argument('--config', default='configs/fg2027.json')
    parser.add_argument('--dataset'); parser.add_argument('--model'); parser.add_argument('--modalities')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(config['output_dir'])
    if args.action == 'summarize':
        show_results(config, out); return
    if args.action != 'audit' and not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU available; smoke tests and training need one.')
    specs = [s for s in matrix(config) if all(getattr(args, k) is None or s[k] == getattr(args, k)
                                            for k in ('dataset', 'model', 'modalities', 'seed'))]
    if not specs:
        raise ValueError('No matching experiment configurations')
    started, outcomes = time.time(), []
    title = f"FG2027 · {args.action}" + (f" · {len(specs)} run{'s' * (len(specs) != 1)} · seed {', '.join(map(str, config['seeds']))}"
                                          if args.action == 'run' else '')
    console.rule(title)
    for dataset in dict.fromkeys(s['dataset'] for s in specs):
        # Combine this dataset's paths with the shared pose selection.
        store_config = {
            **config['datasets'][dataset],
            'pose_indices': config['pose_indices'],
        }
        store = FeatureStore(store_config)
        report = store.report()
        write_json(out / dataset / 'data_audit.json', report)
        console.dataset_summary(dataset, report)
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
            console.line(console.bold(f"[{len(outcomes) + 1:>2}/{len(specs)}] {spec['dataset']} · {spec['model']} · "
                                      f"{spec['modalities']}") + console.dim(f"   (elapsed {console.duration(time.time() - started)})"))
            try:
                outcomes.append(train_run(spec, store, config, out))
            except BaseException as error:
                console.line(console.red('  ✗ stopped (Ctrl-C)' if isinstance(error, KeyboardInterrupt)
                                         else f'  ✗ failed: {error!r}'))
                error_dir = out / run_name(spec)
                if not (error_dir / 'metrics.json').exists():
                    write_json(error_dir / 'status.json', {'status': 'failed', 'error': repr(error)})
                raise
            console.line()
    if args.action == 'run':
        show_results(config, out)
        console.line(console.green(f"  Done: {outcomes.count('completed')} trained, {outcomes.count('skipped')} skipped"
                                   f" in {console.duration(time.time() - started)}"))


if __name__ == '__main__':
    main()
