"""Descriptive trajectories and paired contrasts; no test-driven selection."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUALITY = ['precision_anomaly', 'recall_anomaly', 'f1_anomaly', 'average_precision',
           'roc_auc', 'precision_normal', 'recall_normal', 'f1_normal']
RANKS = ['RankMe', 'CER', 'CovER', 'NESum']


def plot_mean_sd(ax, frame, value, label, x='epoch'):
    table = frame.groupby(x)[value].agg(['mean', 'std'])
    line, = ax.plot(table.index, table['mean'], marker='o', ms=3, label=label)
    sd = table['std'].fillna(0)
    ax.fill_between(table.index, table['mean'] - sd, table['mean'] + sd, alpha=.15, color=line.get_color())
    ax.set_xlabel(x)
    ax.set_ylabel(value)
    ax.grid(alpha=.2)


def analyze(root):
    root = Path(root)
    out = root / 'analysis'
    out.mkdir(exist_ok=True)
    quality, ranks, losses = [], [], []
    for directory in sorted(root.iterdir()):
        if not (directory / 'identity.json').exists():
            continue
        meta = json.loads((directory / 'identity.json').read_text())
        common = {k: meta[k] for k in ('category', 'seed', 'method', 'fraction')}
        common['fraction'] = -1 if common['fraction'] is None else common['fraction']
        for path in sorted(directory.glob('epoch_*/metrics.json')):
            row = json.loads(path.read_text())
            base = {**common, 'epoch': row['epoch']}
            for classifier, data in row['classifiers'].items():
                for split, metrics in data['splits'].items():
                    quality.append({**base, 'classifier': classifier, 'split': split,
                                    **{k: v for k, v in metrics.items() if k != 'confusion_matrix'}})
            for split, variants in row['spectral'].items():
                for normalization, metrics in variants.items():
                    ranks.append({**base, 'split': split, 'normalization': normalization,
                                  **{k: metrics[k] for k in RANKS}})
        if (directory / 'loss.json').exists():
            losses.extend([{**common, **row} for row in json.loads((directory / 'loss.json').read_text())])
    if not quality:
        return
    q, r, loss = pd.DataFrame(quality), pd.DataFrame(ranks), pd.DataFrame(losses)
    q.to_csv(out / 'quality.csv', index=False)
    r.to_csv(out / 'ranks.csv', index=False)
    loss.to_csv(out / 'loss.csv', index=False)
    baseline = q[(q.method == 'baseline') & (q.split == 'test')]
    summary = baseline.groupby(['category', 'classifier'])[QUALITY].agg(['mean', 'std', 'min', 'max'])
    summary.to_csv(out / 'baseline_summary.csv')
    lines = ['# VisA: random ResNet-18 baseline', '',
             'Mean ± sample standard deviation across initialization seeds. Test is the complete official high-shot test.',
             'Precision uses a validation-selected threshold with validation recall >= 0.5; test recall is unrestricted.', '',
             '| Category | Classifier | Seeds | Precision | Recall | Average precision | ROC-AUC |',
             '|---|---|---:|---:|---:|---:|---:|']
    for (cat, clf), g in baseline.groupby(['category', 'classifier']):
        cells = [f'{g[k].mean():.3f} ± {g[k].std():.3f}' if len(g) > 1 else f'{g[k].mean():.3f}'
                 for k in ('precision_anomaly', 'recall_anomaly', 'average_precision', 'roc_auc')]
        lines.append(f'| {cat} | {clf} | {g.seed.nunique()} | ' + ' | '.join(cells) + ' |')
    (out / 'BASELINE.md').write_text('\n'.join(lines) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, metric in zip(axes, ['precision_anomaly', 'average_precision']):
        for j, (clf, group) in enumerate(baseline.groupby('classifier')):
            cats = sorted(group.category.unique())
            stats = group.groupby('category')[metric].agg(['mean', 'std']).reindex(cats)
            positions = np.arange(len(cats)) + (j - .5) * .18
            ax.errorbar(positions, stats['mean'], yerr=stats['std'].fillna(0), fmt='o', capsize=4, label=clf)
            for i, cat in enumerate(cats):
                vals = group[group.category == cat][metric]
                ax.scatter(np.full(len(vals), positions[i]), vals, alpha=.4, s=14)
            ax.set_xticks(np.arange(len(cats)), cats)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
    fig.savefig(out / 'baseline.png', dpi=160)
    plt.close(fig)
    # Each category/imbalance has its own panel: category means are not extra seeds.
    for (cat, frac, split), group in q[q.method != 'baseline'].groupby(['category', 'fraction', 'split']):
        if split == 'probe_train':
            continue
        fig, axes = plt.subplots(2, 4, figsize=(17, 7), constrained_layout=True)
        for ax, metric in zip(axes.flat, QUALITY):
            for (method, clf), values in group.groupby(['method', 'classifier']):
                plot_mean_sd(ax, values, metric, f'{method}/{clf}')
            ax.set_ylim(-.03, 1.03)
        axes.flat[0].legend(fontsize=6)
        fig.suptitle(f'{cat} | SSL anomaly fraction {frac:g} | {split} | mean ± SD across seeds')
        fig.savefig(out / f'{cat}_p{frac:g}_{split}_quality.png', dpi=140)
        plt.close(fig)
    for (cat, frac, split, norm), group in r[r.method != 'baseline'].groupby(['category', 'fraction', 'split', 'normalization']):
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), constrained_layout=True)
        for ax, metric in zip(axes, RANKS):
            for method, values in group.groupby('method'):
                plot_mean_sd(ax, values, metric, method)
        axes[0].legend(fontsize=7)
        fig.suptitle(f'{cat} | p={frac:g} | {split} | {norm} | mean ± SD')
        fig.savefig(out / f'{cat}_p{frac:g}_{split}_{norm}_ranks.png', dpi=140)
        plt.close(fig)
    if len(loss):
        for (cat, frac), group in loss.groupby(['category', 'fraction']):
            methods = sorted(group.method.unique())
            fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 3.5), squeeze=False, constrained_layout=True)
            for ax, method in zip(axes.flat, methods):
                plot_mean_sd(ax, group[group.method == method], 'loss', method)
                ax.set_title(method)
            fig.suptitle(f'{cat} | p={frac:g} | losses use different scales')
            fig.savefig(out / f'{cat}_p{frac:g}_loss.png', dpi=140)
            plt.close(fig)
    merged = q.merge(r, on=['category', 'seed', 'method', 'fraction', 'epoch', 'split'])
    merged.to_csv(out / 'quality_vs_ranks.csv', index=False)
    # Scatter: preserve seed trajectories; plot precision and AP against every rank.
    for (cat, frac, norm, clf), group in merged[(merged.method != 'baseline') & (merged.split == 'test')].groupby(
            ['category', 'fraction', 'normalization', 'classifier']):
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
        for row, metric in enumerate(('precision_anomaly', 'average_precision')):
            for ax, rank in zip(axes[row], RANKS):
                for method, values in group.groupby('method'):
                    color = None
                    for i, (_, trajectory) in enumerate(values.groupby('seed')):
                        trajectory = trajectory.sort_values('epoch')
                        line, = ax.plot(trajectory[rank], trajectory[metric], 'o-', ms=3, alpha=.65,
                                        color=color, label=method if i == 0 else None)
                        color = line.get_color()
                    ax.set_xlabel(rank)
                    ax.set_ylabel(metric)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f'{cat} | p={frac:g} | {norm} | {clf} | individual seed trajectories')
        fig.savefig(out / f'{cat}_p{frac:g}_{norm}_{clf}_scatter.png', dpi=140)
        plt.close(fig)
    keys = ['category', 'fraction', 'method', 'seed', 'classifier', 'split']
    contrasts = []
    for key, group in q[q.method != 'baseline'].groupby(keys):
        indexed = group.set_index('epoch')
        for early, late in [(0, 1), (0, 5), (0, 30), (1, 30), (5, 30), (10, 30)]:
            if early in indexed.index and late in indexed.index:
                contrasts.append({**dict(zip(keys, key)), 'early_epoch': early, 'late_epoch': late,
                                  **{k: indexed.loc[late, k] - indexed.loc[early, k] for k in QUALITY}})
    paired = pd.DataFrame(contrasts)
    paired.to_csv(out / 'paired_deltas.csv', index=False)
    if len(paired):
        paired.groupby([k for k in keys if k != 'seed'] + ['early_epoch', 'late_epoch'])[QUALITY].agg(
            ['mean', 'std', 'count', 'min', 'max']).to_csv(out / 'paired_deltas_summary.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root)
