"""Plots and paired long-minus-short contrasts; never pick epochs using test."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from .io import digest, read_json, write_json


def mean_ci(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None, None
    mean = float(v.mean())
    half = float(t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None
    return mean, half


def analyze(root, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    groups, all_rows = defaultdict(list), []
    for path in sorted(Path(root).rglob('completion.json')):
        run = path.parent
        c = read_json(run / 'config.json')
        common = {k: v for k, v in c.items() if k not in ('method', 'seed')}
        common['manifest_sha256'] = digest(run / 'manifest.json')
        keyhash = hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()[:10]
        key = f"{c['dataset']}_{c['augmentation']['family']}_{keyhash}"
        entry = {'path': run, 'config': c, 'validation': {}, 'test': {}}
        for epoch in c['checkpoints']:
            e = read_json(run / f'epoch_{epoch:04d}' / 'evaluation.json')
            for split in ['validation']:
                row = {'group': key, 'dataset': c['dataset'], 'stage': c['stage'], 'method': c['method'],
                       'seed': c['seed'], 'split': split, 'epoch': epoch,
                       **{f'{probe}_{metric}': e['probes'][probe][split][metric]
                          for probe in ['logreg', 'catboost'] for metric in ['ap', 'roc_auc']},
                       **{k: v for k, v in e['spectral'][split].items() if k != 'eigenvalues'}}
                entry[split][epoch] = row
                all_rows.append(row)
        if (run / 'test.json').exists():
            for e in read_json(run / 'test.json')['rows']:
                row = {'group': key, 'dataset': c['dataset'], 'stage': c['stage'], 'method': c['method'],
                       'seed': c['seed'], 'split': 'test', 'epoch': e['epoch'],
                       **{f'{probe}_{metric}': e['probes'][probe][metric]
                          for probe in ['logreg', 'catboost'] for metric in ['ap', 'roc_auc']},
                       **{k: v for k, v in e['spectral'].items() if k != 'eigenvalues'}}
                entry['test'][e['epoch']] = row
                all_rows.append(row)
        if any(e['config']['seed'] == c['seed'] and e['config']['method'] == c['method'] for e in groups[key]):
            raise ValueError(f'Duplicate run for the same configuration and seed: {run}')
        groups[key].append(entry)
    if not all_rows:
        raise ValueError('No completed runs found.')
    with (out / 'trajectory.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(all_rows)
    contrasts = []
    metrics = ['logreg_ap', 'catboost_ap', 'logreg_roc_auc', 'catboost_roc_auc',
               'effective_rank', 'singular_effective_rank', 'participation_ratio', 'total_variance']
    for group, entries in groups.items():
        for split in ['validation', 'test']:
            selected = [e for e in entries if e[split]]
            if not selected:
                continue
            fig, axes = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)
            methods = sorted({e['config']['method'] for e in selected})
            for ax, metric in zip(axes.flat, metrics + ['ssl_loss']):
                for method in methods:
                    subset = [e for e in selected if e['config']['method'] == method]
                    values = defaultdict(list)
                    for e in subset:
                        if metric == 'ssl_loss':
                            for r in read_json(e['path'] / 'loss.json'):
                                values[r['epoch']].append(r['ssl_loss'])
                        else:
                            for epoch, row in e[split].items():
                                if row[metric] is not None:
                                    values[epoch].append(row[metric])
                    epochs = sorted(values)
                    means, halves = [], []
                    for epoch in epochs:
                        mean, half = mean_ci(values[epoch])
                        means.append(mean)
                        halves.append(0 if half is None else half)
                    means, halves = np.asarray(means), np.asarray(halves)
                    ax.plot(epochs, means, label=method, marker='o' if len(epochs) <= 20 else None, markersize=3)
                    ax.fill_between(epochs, means - halves, means + halves, alpha=.15)
                if metric.startswith(('logreg_', 'catboost_')):
                    probe, measure = metric.split('_', 1)
                    baseline = []
                    seen = set()
                    for e in selected:
                        if e['config']['seed'] in seen:
                            continue
                        raw_path = e['path'] / ('raw_baseline.json' if split == 'validation' else 'test.json')
                        if raw_path.exists():
                            raw = read_json(raw_path)
                            raw = raw if split == 'validation' else raw.get('raw_baseline')
                            if raw:
                                baseline.append(raw[probe][split][measure] if split == 'validation' else raw[probe][measure])
                                seen.add(e['config']['seed'])
                    if baseline:
                        ax.axhline(np.mean(baseline), color='black', linestyle='--', label='raw features')
                ax.set(title=metric, xlabel='Epoch')
                ax.grid(alpha=.2)
            axes.flat[0].legend(fontsize=8)
            counts = ', '.join(f"{m}: n={sum(e['config']['method'] == m for e in selected)}" for m in methods)
            stage = selected[0]['config']['stage']
            fig.suptitle(f'{group} | {split} | {stage.upper()} | {counts}\n'
                         'Mean; 95% t intervals only for n>1. SSL losses have different scales and objectives')
            fig.savefig(out / f'{group}_{split}.png', dpi=160)
            svg = out / f'{group}_{split}.svg'
            fig.savefig(svg)
            svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
            plt.close(fig)
            for method in methods:
                subset = [e for e in selected if e['config']['method'] == method]
                for metric in metrics:
                    pairs = []
                    for e in subset:
                        short, long = e['config']['short_epoch'], e['config']['epochs']
                        a, b = e[split][short][metric], e[split][long][metric]
                        if a is not None and b is not None:
                            pairs.append({'seed': e['config']['seed'], 'short': a, 'long': b, 'delta': b - a})
                    mean, half = mean_ci([p['delta'] for p in pairs])
                    contrasts.append({'group': group, 'method': method, 'split': split, 'metric': metric,
                                      'short_epoch': short, 'long_epoch': long, 'n_seeds': len(pairs),
                                      'mean_delta': mean, 'ci95_half_width': half, 'pairs': pairs,
                                      'stage': subset[0]['config']['stage']})
    write_json(out / 'paired_contrasts.json', contrasts)
    write_json(out / 'inputs.json', {k: [str(e['path']) for e in v] for k, v in groups.items()})
