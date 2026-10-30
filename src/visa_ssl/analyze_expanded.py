"""V2 reports keep architecture, augmentation and category separate."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from .analyze import QUALITY, RANKS, plot_mean_sd

KEYS = ['category', 'architecture', 'augmentation', 'method', 'seed', 'fraction']


def analyze(root, plots=False):
    root = Path(root)
    out = root / 'analysis'
    out.mkdir(exist_ok=True)
    quality, ranks, loss = [], [], []
    for identity_path in sorted(root.glob('*/identity.json')):
        meta = json.loads(identity_path.read_text())
        common = {k: meta[k] for k in KEYS}
        common['fraction'] = -1 if common['fraction'] is None else common['fraction']
        for path in sorted(identity_path.parent.glob('epoch_*/metrics.json')):
            metrics = json.loads(path.read_text())
            base = {**common, 'epoch': metrics['epoch']}
            for classifier, data in metrics['classifiers'].items():
                for split, values in data['splits'].items():
                    quality.append({**base, 'classifier': classifier, 'split': split,
                                    **{k: v for k, v in values.items() if k != 'confusion_matrix'}})
            for split, variants in metrics['spectral'].items():
                for normalization, values in variants.items():
                    ranks.append({**base, 'split': split, 'normalization': normalization,
                                  **{k: values[k] for k in RANKS + ['n', 'dimension']}})
        path = identity_path.parent / 'loss.json'
        if path.exists():
            loss.extend({**common, **r} for r in json.loads(path.read_text()))
    if not quality:
        return
    q, r, losses = pd.DataFrame(quality), pd.DataFrame(ranks), pd.DataFrame(loss)
    assert not q.duplicated(KEYS + ['epoch', 'split', 'classifier']).any()
    q.to_csv(out / 'quality.csv', index=False)
    r.to_csv(out / 'ranks.csv', index=False)
    losses.to_csv(out / 'loss.csv', index=False)
    merged = q.merge(r, on=KEYS + ['epoch', 'split'], validate='many_to_many')
    merged.to_csv(out / 'quality_vs_ranks.csv', index=False)
    baseline = q[(q.method == 'baseline') & (q.split == 'test')]
    baseline.groupby(['architecture', 'category', 'classifier'])[QUALITY].agg(
        ['mean', 'std', 'count', 'min', 'max']).to_csv(out / 'baseline_summary.csv')
    q.groupby([k for k in KEYS if k != 'seed'] + ['epoch', 'split', 'classifier'])[QUALITY].agg(
        ['mean', 'std', 'count']).to_csv(out / 'quality_summary.csv')
    lines = ['# VisA v2 random encoder baseline', '',
             'Mean ± sample SD across SSL seeds; fixed official test. Thresholds selected on validation.', '',
             '| Architecture | Category | Classifier | n | Precision | ROC-AUC |',
             '|---|---|---|---:|---:|---:|']
    for (arch, cat, clf), group in baseline.groupby(['architecture', 'category', 'classifier']):
        stats = [f'{group[m].mean():.4f} ± {group[m].std():.4f}' for m in ['precision_anomaly', 'roc_auc']]
        lines.append(f'| {arch} | {cat} | {clf} | {group.seed.nunique()} | ' + ' | '.join(stats) + ' |')
    (out / 'BASELINE.md').write_text('\n'.join(lines) + '\n')
    contrast_keys = KEYS + ['classifier', 'split']
    contrasts = []
    for values, group in q[q.method != 'baseline'].groupby(contrast_keys):
        indexed = group.set_index('epoch')
        for early, late in [(0, 30), (5, 30), (0, 100), (5, 100), (10, 100), (30, 100)]:
            if early in indexed.index and late in indexed.index:
                contrasts.append({**dict(zip(contrast_keys, values)), 'early_epoch': early, 'late_epoch': late,
                                  **{k: indexed.loc[late, k] - indexed.loc[early, k] for k in QUALITY}})
    paired = pd.DataFrame(contrasts)
    paired.to_csv(out / 'paired_deltas.csv', index=False)
    if len(paired):
        paired.groupby([k for k in contrast_keys if k != 'seed'] + ['early_epoch', 'late_epoch'])[QUALITY].agg(
            ['mean', 'std', 'count', 'min', 'max']).to_csv(out / 'paired_deltas_summary.csv')
    if not plots:
        return
    for (cat, arch, aug), group in q[(q.method != 'baseline') & (q.split == 'test')].groupby(
            ['category', 'architecture', 'augmentation']):
        name = f'{cat}_{arch}_{aug}'
        fig, axes = plt.subplots(3, 4, figsize=(17, 11), constrained_layout=True)
        for ax, (metric, clf) in zip(axes[0], [(m, c) for m in ['precision_anomaly', 'roc_auc']
                                            for c in ['logistic_regression', 'lightgbm']]):
            for method, values in group[group.classifier == clf].groupby('method'):
                plot_mean_sd(ax, values, metric, method)
            ax.set_title(clf)
            ax.set_ylim(-.03, 1.03)
        rank_group = r[(r.category == cat) & (r.architecture == arch) & (r.augmentation == aug)
                       & (r.split == 'test') & (r.method != 'baseline')]
        for row, norm in enumerate(['raw', 'l2'], 1):
            for ax, metric in zip(axes[row], RANKS):
                for method, values in rank_group[rank_group.normalization == norm].groupby('method'):
                    plot_mean_sd(ax, values, metric, method)
                ax.set_title(norm)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f'{name} | test | mean ± sample SD; n varies until all seeds finish')
        fig.savefig(out / f'{name}_quality_ranks.png', dpi=130)
        plt.close(fig)
        fig, axes = plt.subplots(2, 4, figsize=(17, 7), constrained_layout=True)
        for ax, metric in zip(axes.flat, QUALITY):
            for (method, clf), values in group.groupby(['method', 'classifier']):
                plot_mean_sd(ax, values, metric, f'{method}/{clf}')
            ax.set_ylim(-.03, 1.03)
        axes[0, 0].legend(fontsize=5)
        fig.suptitle(name)
        fig.savefig(out / f'{name}_all_quality.png', dpi=130)
        plt.close(fig)
    if len(losses):
        for (cat, arch, aug), group in losses.groupby(['category', 'architecture', 'augmentation']):
            fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
            for ax, (method, values) in zip(axes, group.groupby('method')):
                plot_mean_sd(ax, values, 'loss', method)
                ax.set_title(method)
            fig.savefig(out / f'{cat}_{arch}_{aug}_loss.png', dpi=130)
            plt.close(fig)
    for values, group in merged[(merged.method != 'baseline') & (merged.split == 'test')].groupby(
            ['category', 'architecture', 'augmentation', 'classifier', 'normalization']):
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
        for row, metric in enumerate(['precision_anomaly', 'roc_auc']):
            for ax, rank in zip(axes[row], RANKS):
                for method, data in group.groupby('method'):
                    color = None
                    for i, (_, trajectory) in enumerate(data.groupby('seed')):
                        trajectory = trajectory.sort_values('epoch')
                        line, = ax.plot(trajectory[rank], trajectory[metric], 'o-', ms=3, alpha=.6,
                                        color=color, label=method if i == 0 else None)
                        color = line.get_color()
                ax.set_xlabel(rank)
                ax.set_ylabel(metric)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(' | '.join(values))
        fig.savefig(out / ('_'.join(values) + '_scatter.png'), dpi=130)
        plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--plots', action='store_true')
    args = p.parse_args()
    analyze(args.root, args.plots)
