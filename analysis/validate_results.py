"""Validate seed-379 outputs and generate the paper's tables and error analysis.

Usage: python analysis/validate_results.py REPOSITORY [OUTPUT_DIRECTORY]
No training or checkpoint loading. Intervals reproduce the IberSPEECH calculator.
Its proportion-based F1 margins are heuristic, not calibrated F1 confidence intervals.
"""
import csv
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('repository', type=Path, help='FG2027 repository containing runs/seed379')
parser.add_argument('output_directory', type=Path, nargs='?', default=Path(__file__).resolve().parent/'output')
args = parser.parse_args()
if not __debug__:
    raise RuntimeError('Run without -O: validation requires assertions.')
repo = args.repository.resolve()
root = repo / 'runs/seed379'
dest = args.output_directory.resolve()
dest.mkdir(parents=True, exist_ok=True)
figdir = dest / 'figures'
figdir.mkdir(exist_ok=True)
read = lambda p: json.loads(p.read_text())
metrics, histories, configs, hashes, envs, freezes = {}, {}, [], [], [], []
predictions, audits = {}, {}
checks = []
expected = [('bilstm','H'),('bilstm','HP'),('bilstm','HPF'),
            ('spoter','H'),('spoter','HP'),('spoter','HPF'),
            ('encoder','H'),('encoder','HP'),('encoder','HPF'),
            ('latefusion','HP'),('latefusion','HPF')]
for dataset in ['wlasl','avasag']:
    audit = read(root/dataset/'data_audit.json')
    audits[dataset] = audit
    splits = [set(audit['split_ids'][s]) for s in ['train','val','test']]
    assert all(not (a & b) for i,a in enumerate(splits) for b in splits[i+1:])
    assert audit['counts'] == dict(zip(['train','val','test'],
                                      [1442,338,258] if dataset=='wlasl' else [2596,849,890]))
    assert [len(audit['columns'][k]) for k in ['hand_landmarks','pose_landmarks','face_landmarks']] == [84,26,136]
    smoke = read(root/dataset/'smoke_pass.json')
    assert smoke['status']=='passed' and smoke['inputs']==audit['sha256']
    reference_truth = None
    for model, modality in expected:
        folder = root/dataset/f'{model}_{modality}'/'seed_379'
        m, manifest, env, history = [read(folder/f'{f}.json') for f in ['metrics','manifest','environment','history']]
        rows = list(csv.DictReader((folder/'predictions.csv').open()))
        assert read(folder/'status.json')['status']=='completed'
        spec = {'dataset':dataset,'model':model,'modalities':modality,'seed':379}
        assert manifest['experiment']==spec and all(m[k]==v for k,v in spec.items())
        assert manifest['input_sha256']==audit['sha256']
        assert manifest['source_sha256']==env['source_sha256']==smoke['source_sha256']
        assert manifest['config']==smoke['config']
        assert env['git_dirty'] is False
        assert len(rows)==m['n_test']==audit['counts']['test']
        ids = [r['video_id'] for r in rows]
        assert ids==audit['split_ids']['test'] and len(set(ids))==len(ids)
        truth, pred = [np.array([int(r[k]) for r in rows]) for k in ['y_true','y_pred']]
        predictions[(dataset,model,modality)] = (ids, truth, pred)
        assert np.array_equal(np.unique(truth),np.arange(100))
        assert ((pred>=0)&(pred<100)).all()
        if reference_truth is None:
            reference_truth=truth.copy()
        assert np.array_equal(truth,reference_truth)
        recomputed = {'accuracy':accuracy_score(truth,pred),
                      'macro_f1':f1_score(truth,pred,labels=np.unique(truth),average='macro',zero_division=0),
                      'weighted_f1':f1_score(truth,pred,labels=np.unique(truth),average='weighted',zero_division=0)}
        assert all(np.isclose(m[k],v,atol=1e-12,rtol=0) for k,v in recomputed.items())
        assert len(history)==100 and [r['epoch'] for r in history]==list(range(1,101))
        assert all(np.isfinite(list(r.values())).all() for r in history)
        assert all(r['lr']==1e-4 and 0<=r['train_acc']<=1 and 0<=r['val_acc']<=1
                   and r['train_loss']>=0 and r['val_loss']>=0 for r in history)
        best=max(history,key=lambda r:r['val_acc'])
        assert m['best_epoch']==best['epoch'] and m['best_val_accuracy']==best['val_acc']
        if model=='latefusion':
            weights=m['fusion_weights']
            assert set(weights)==set(['hand_landmarks','pose_landmarks']+(['face_landmarks'] if modality=='HPF' else []))
            assert all(0<=w<=1 for w in weights.values()) and abs(sum(weights.values())-1)<1e-6
        configs.append(manifest['config']); hashes.append(manifest['source_sha256']); envs.append(env)
        freezes.append((folder/'pip-freeze.txt').read_text())
        metrics[(dataset,model,modality)] = m
        histories[(dataset,model,modality)] = history
        checks.append({'dataset':dataset,'model':model,'modalities':modality,
                       'status':'passed','prediction_rows':len(rows),'recomputed_metrics':recomputed})
assert len(metrics)==22 and all(c==configs[0] for c in configs)
assert len(set(hashes))==1 and all(e==envs[0] for e in envs) and len(set(freezes))==1
digest=hashlib.sha256()
paths=list((repo/'src').rglob('*.py'))+list((repo/'tests').rglob('*.py'))+[repo/'requirements-fg2027.txt']
for path in sorted(paths):
    digest.update(str(path.relative_to(repo)).encode()); digest.update(path.read_bytes())
assert digest.hexdigest()==hashes[0]
json_report={'source_results_commit':'7a791f6','training_environment':envs[0],
             'runs':checks,'limitations':['Raw feature extraction is not independently verified by this artifact audit.',
             'Disjoint clip identifiers do not establish source/signer independence.',
             'Best model checkpoints are not committed; their contents were not independently inspected.',
             'Single seed; intervals do not measure training-seed variability.',
             'IberSPEECH proportion-based margins are reproduced for all metrics; F1 coverage is not calibrated.']}
(dest/'RESULTS_VALIDATION.json').write_text(json.dumps(json_report,indent=2))


def approximate_ci(score_pct, n):
    """Exact numerical rule from iberspeech26/src/ci_calculator.py.

    Commit 518a7f2f3934bca087305d7d04f6d3b71bcf6b69, blob 7a37a1d.
    Returns a percentage-point margin, centered on the observed score as in
    the original reporting code. This is not bootstrapping or seed variance.
    """
    if n == 0 or score_pct < 0 or score_pct > 100:
        return 0
    adjusted_p = ((score_pct/100)*n + 2)/(n + 4)
    return 1.96*math.sqrt(adjusted_p*(1-adjusted_p)/(n+4))*100


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


names = {'bilstm':'BiLSTM', 'spoter':'SPOTER-inspired',
         'encoder':'Encoder-only', 'latefusion':'Encoder-only'}
metric_keys = ['accuracy', 'macro_f1', 'weighted_f1']
table_rows = []
for (dataset, model, modality), m in metrics.items():
    row = {k:m[k] for k in ['dataset','model','modalities','seed', *metric_keys,
                            'parameters','best_epoch','best_val_accuracy',
                            'training_seconds','peak_cuda_memory_bytes','n_test']}
    row['correct_predictions'] = int(round(m['accuracy']*m['n_test']))
    row.update({k+'_approx95_margin_pp':approximate_ci(m[k]*100,m['n_test']) for k in metric_keys})
    table_rows.append(row)
write_csv(dest/'results_verified.csv', table_rows)

maxima = {(d,k):max(m[k] for (ds,_,_),m in metrics.items() if ds==d)
          for d in audits for k in metric_keys}
lines = []
for model, modality in expected:
    cells = []
    for dataset in ['wlasl','avasag']:
        m = metrics[(dataset,model,modality)]
        for key in metric_keys:
            score = f'{m[key]*100:.2f}'
            if m[key] == maxima[(dataset,key)]:
                score = r'\mathbf{'+score+'}'
            margin = approximate_ci(m[key]*100,m['n_test'])
            cells.append('$'+score+r'{\pm}'+f'{margin:.2f}'+'$')
    fusion = modality + (' / late' if model=='latefusion' else ' / early' if modality!='H' else '')
    m = metrics[('wlasl',model,modality)]
    assert m['parameters'] == metrics[('avasag',model,modality)]['parameters']
    lines.append(' & '.join([names[model],fusion,*cells,f"{m['parameters']/1e6:.2f}"])+r'\\')
    if (model,modality) in [('bilstm','HPF'),('spoter','HPF'),('encoder','HPF')]:
        lines.append(r'\midrule')
(dest/'results_rows.tex').write_text('\n'.join(lines)+'\n')

# Descriptive analysis of the highest observed test-accuracy model per dataset.
# This post-hoc choice does not replace validation-based checkpoint selection.
best_models = {d:max((k for k in metrics if k[0]==d),key=lambda k:metrics[k]['accuracy']) for d in audits}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
fig, axes = plt.subplots(1,2,figsize=(7.0,2.95),layout='constrained')
summary = {}
for ax, (dataset,key) in zip(axes,best_models.items()):
    ids, truth, pred = predictions[key]
    class_names = [None]*100
    for name, idx in audits[dataset]['gloss2idx'].items():
        class_names[idx] = name
    cm = confusion_matrix(truth,pred,labels=np.arange(100))
    assert cm.sum()==len(truth) and cm.trace()==round(metrics[key]['accuracy']*len(truth))
    support = cm.sum(axis=1)
    normalized = cm/support[:,None]
    assert np.allclose(normalized.sum(axis=1),1)
    write_csv(dest/f'{dataset}_confusion_counts.csv',
              [{'true_gloss':name,**dict(zip(class_names,map(int,cm[i])))} for i,name in enumerate(class_names)])
    write_csv(dest/f'{dataset}_confusion_normalized.csv',
              [{'true_gloss':name,**dict(zip(class_names,map(float,normalized[i])))} for i,name in enumerate(class_names)])
    errors = [{'video_id':v,'true_gloss':class_names[t],'predicted_gloss':class_names[p],
               'true_index':int(t),'predicted_index':int(p)} for v,t,p in zip(ids,truth,pred) if t!=p]
    write_csv(dest/f'{dataset}_errors.csv',errors)
    # Rank directed confusions by count; break ties by gloss names, not manual selection.
    pairs = sorted([(int(cm[t,p]),t,p) for t in range(100) for p in range(100) if t!=p and cm[t,p]],
                   key=lambda x:(-x[0],class_names[x[1]],class_names[x[2]]))
    pair_rows = [{'true_gloss':class_names[t],'predicted_gloss':class_names[p],
                 'count':n,'true_class_test_count':int(support[t]),'fraction_of_true_class':n/int(support[t])}
                for n,t,p in pairs]
    write_csv(dest/f'{dataset}_confused_pairs.csv',pair_rows)
    # Main-paper excerpts: classes involved in the three largest directed confusions.
    # Normalize with ALL test examples of each class, never the cropped submatrix.
    selected = sorted({i for _,t,p in pairs[:3] for i in (t,p)},key=lambda i:class_names[i])
    sub = normalized[np.ix_(selected,selected)]
    image = ax.imshow(sub,vmin=0,vmax=1,cmap='Blues',aspect='equal')
    labels = [class_names[i] for i in selected]
    ax.set_xticks(range(len(selected)),labels,rotation=45,ha='right',rotation_mode='anchor',fontsize=8)
    ax.set_yticks(range(len(selected)),[f'{class_names[i]} ({support[i]})' for i in selected],fontsize=8)
    for i,t in enumerate(selected):
        for j,p in enumerate(selected):
            ax.text(j,i,str(cm[t,p]),ha='center',va='center',fontsize=8,
                    color='white' if normalized[t,p]>.5 else '#222222')
    ax.set_xlabel('Predicted gloss',labelpad=1)
    ax.set_ylabel('True gloss (test count)',labelpad=2)
    model_label = 'Encoder, HPF late' if key[1]=='latefusion' else 'SPOTER-inspired, HPF'
    ax.set_title(f'{dataset.upper()}100: {model_label}',fontsize=9,pad=6)
    full, full_ax = plt.subplots(figsize=(16,16),layout='constrained')
    im = full_ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues')
    full_ax.set_xticks(range(100),class_names,rotation=90,fontsize=6)
    full_ax.set_yticks(range(100),class_names,fontsize=6)
    full_ax.set_xlabel('Predicted gloss'); full_ax.set_ylabel('True gloss')
    full_ax.set_title(f'{dataset.upper()}100 - {model_label}; all 100 classes; row-normalized')
    full.colorbar(im,ax=full_ax,fraction=.035,pad=.02,label='Fraction of true-class test clips')
    full.savefig(figdir/f'{dataset}_confusion_full.pdf'); plt.close(full)
    summary[dataset] = {'model':key[1],'modalities':key[2],'seed':379,
                        'selection':'highest observed test accuracy; descriptive post-hoc analysis',
                        'n_test':len(truth),'correct':int(cm.trace()),'errors':len(errors),
                        'top_pairs':pair_rows[:3],'figure_classes':labels,
                        'classes_with_errors':int(np.count_nonzero(cm.diagonal()<support)),
                        'classes_with_zero_recall':int(np.count_nonzero(cm.diagonal()==0)),
                        'directed_error_pairs':len(pairs),
                        'singleton_error_pairs':sum(n==1 for n,_,_ in pairs),
                        'top_three_error_fraction':sum(n for n,_,_ in pairs[:3])/len(errors)}
fig.colorbar(image,ax=list(axes),fraction=.025,pad=.02,label='Fraction of true-class clips',shrink=.78)
fig.savefig(figdir/'confusion_excerpts.pdf')
fig.savefig(figdir/'confusion_excerpts.png',dpi=220)
plt.close(fig)
(dest/'ERROR_ANALYSIS.json').write_text(json.dumps(summary,indent=2))
comparisons = []
for dataset, first, second in [('wlasl',('encoder','HPF'),('latefusion','HPF')),
                               ('avasag',('spoter','HP'),('spoter','HPF')),
                               ('avasag',('encoder','HPF'),('latefusion','HPF'))]:
    ids_a, truth_a, pred_a = predictions[(dataset,*first)]
    ids_b, truth_b, pred_b = predictions[(dataset,*second)]
    assert ids_a==ids_b and np.array_equal(truth_a,truth_b)
    correct_a, correct_b = pred_a==truth_a, pred_b==truth_b
    comparisons.append({'dataset':dataset,'from_model':'_'.join(first),'to_model':'_'.join(second),
                        'corrected_errors':int(np.count_nonzero(~correct_a & correct_b)),
                        'new_errors':int(np.count_nonzero(correct_a & ~correct_b))})
write_csv(dest/'paired_changes.csv', comparisons)
print('PASS: all 22 runs, 66 scores, source hash, splits, predictions, histories and manifests verified.')
print('Wrote approximate margins, results table, two confusion matrices and error clip lists.')

