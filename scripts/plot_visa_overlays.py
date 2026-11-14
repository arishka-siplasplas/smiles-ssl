"""Overlay downstream quality and spectral rank from the audited Kaggle report.

Usage: python scripts/plot_visa_overlays.py reports/visa-duration-v1
Outputs are separate from the source report; training and probe selection are unchanged.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import itertools
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_visa_results import (CATEGORIES, CLASSIFIERS, CLF_LABELS, EPOCHS,
                               FRACTIONS, LABELS, METHODS, RANKS, SEEDS)

QUALITY = ['precision_anomaly', 'roc_auc']
Q_LABELS = ['Precision · anomaly', 'ROC-AUC']
KEYS = ['category', 'fraction', 'method', 'seed', 'epoch', 'split']
GROUPS = ['category', 'fraction', 'method', 'classifier', 'normalization']
BLACK = '#252525'
ORANGE = '#ed8427'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(source):
    audit = json.loads((source / 'audit.json').read_text())
    for name, digest in audit['csv_sha256'].items():
        assert sha256(source / name) == digest, f'Source changed: {name}'
    for name, digest in audit.get('metadata_sha256', {}).items():
        assert sha256(source / name) == digest, f'Metadata changed: {name}'
    status = json.loads((source / 'status.json').read_text())
    assert status['status'] == 'complete'
    assert all(j['status'] == 'complete' for j in status['baseline_jobs'] + status['train_jobs'])
    q = pd.read_csv(source / 'quality.csv')
    r = pd.read_csv(source / 'ranks.csv')
    q = q[(q.split == 'test') & q.method.isin(METHODS)].copy()
    r = r[(r.split == 'test') & r.method.isin(METHODS)].copy()
    expected = set(itertools.product(CATEGORIES, FRACTIONS, METHODS, SEEDS, EPOCHS, ['test']))
    for frame, variant, variants in [(q, 'classifier', CLASSIFIERS), (r, 'normalization', ['raw', 'l2'])]:
        assert not frame.duplicated(KEYS + [variant]).any()
        for value in variants:
            actual = set(frame.loc[frame[variant] == value, KEYS].itertuples(index=False, name=None))
            assert actual == expected, f'Incomplete {variant}={value}'
    joined = q[KEYS + ['classifier'] + QUALITY].merge(
        r[KEYS + ['normalization'] + RANKS], on=KEYS, validate='many_to_many')
    assert len(joined) == len(expected) * len(CLASSIFIERS) * 2
    assert not joined.duplicated(GROUPS + ['epoch', 'seed']).any()
    assert np.isfinite(joined[QUALITY + RANKS]).all().all()
    assert joined[QUALITY].ge(0).all().all() and joined[QUALITY].le(1).all().all()
    count = joined.groupby(GROUPS + ['epoch']).seed.nunique()
    assert count.eq(3).all()
    stats = joined.groupby(GROUPS + ['epoch'])[QUALITY + RANKS].agg(['mean', 'std'])
    stats.columns = ['_'.join(col) for col in stats.columns]
    stats = stats.reset_index()
    stats['n_seeds'] = 3
    scales = []
    for group_key, values in stats.groupby(GROUPS):
        for metric in QUALITY + RANKS:
            mean = values[f'{metric}_mean']
            scales.append({**dict(zip(GROUPS, group_key)), 'metric': metric,
                           'mean_min': mean.min(), 'mean_max': mean.max(),
                           'constant': bool(mean.max() - mean.min() <= 1e-12)})
    return joined, stats, pd.DataFrame(scales), audit


def minmax(values):
    """Scale the seven seed-averaged checkpoint values, without reversing sign."""
    values = np.asarray(values, dtype=float)
    span = np.ptp(values)
    return np.full_like(values, .5) if span <= 1e-12 else (values - values.min()) / span


def draw(stats, category, fraction, norm, classifier, rank, mode, output):
    selected = stats[(stats.category == category) & (stats.fraction == fraction)
                     & (stats.normalization == norm) & (stats.classifier == classifier)]
    fig, axes = plt.subplots(3, 2, figsize=(10.6, 11.9))
    fig.subplots_adjust(left=.13, right=.90, bottom=.16, top=.82, wspace=.38, hspace=.43)
    fig.suptitle(f'VisA · {category} · {fraction:.0%} anomalies in SSL train',
                 y=.975, fontsize=18, weight='semibold')
    fig.text(.5, .939, f'{CLF_LABELS[classifier]}  |  ranks: {norm.upper()} embeddings',
             ha='center', fontsize=12)
    normalized = mode == 'minmax'
    subtitle = ('Each mean trajectory independently min–max scaled to [0, 1]'
                if normalized else 'Original values · left axis: quality · right axis: rank')
    fig.text(.5, .909, subtitle, ha='center', fontsize=10.5, color='#515151')
    handles = [Line2D([], [], color=BLACK, marker='s', lw=1.7, ms=4, label='Downstream metric'),
               Line2D([], [], color=ORANGE, marker='o', lw=1.7, ms=4, label=rank)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .89),
               ncol=2, frameon=False, fontsize=11)
    rank_high = float((selected[f'{rank}_mean'] + selected[f'{rank}_std']).max()) * 1.08
    quality_low = min(-.035, min((selected[f'{m}_mean'] - selected[f'{m}_std']).min()
                               for m in QUALITY) - .025)
    quality_high = max(1.035, max((selected[f'{m}_mean'] + selected[f'{m}_std']).max()
                                 for m in QUALITY) + .025)
    for row, method in enumerate(METHODS):
        data = selected[selected.method == method].sort_values('epoch')
        np.testing.assert_array_equal(data.epoch, EPOCHS)
        epoch = data.epoch.to_numpy()
        axes[row, 0].text(-.24, .5, LABELS[method].replace('Masked reconstruction', 'Masked\nreconstruction'),
                          transform=axes[row, 0].transAxes, rotation=90,
                          ha='center', va='center', fontsize=11, weight='semibold')
        for col, quality in enumerate(QUALITY):
            ax = axes[row, col]
            q_mean, q_sd = data[f'{quality}_mean'].to_numpy(), data[f'{quality}_std'].to_numpy()
            r_mean, r_sd = data[f'{rank}_mean'].to_numpy(), data[f'{rank}_std'].to_numpy()
            if normalized:
                ax.plot(epoch, minmax(q_mean), color=BLACK, marker='s', ms=4, lw=1.5, zorder=3)
                ax.plot(epoch, minmax(r_mean), color=ORANGE, marker='o', ms=4, lw=1.7, zorder=4)
                ax.set_ylim(-.06, 1.06)
                ax.set_yticks(np.linspace(0, 1, 6))
                ax.set_ylabel('Min–max scaled value', fontsize=9)
                if np.ptp(q_mean) <= 1e-12 or np.ptp(r_mean) <= 1e-12:
                    ax.text(.02, .03, 'Constant series shown at 0.5', transform=ax.transAxes, fontsize=8)
            else:
                ax.plot(epoch, q_mean, color=BLACK, marker='s', ms=4, lw=1.5, zorder=3)
                ax.fill_between(epoch, q_mean - q_sd, q_mean + q_sd, color=BLACK, alpha=.10, lw=0)
                ax.set_ylim(quality_low, quality_high)
                ax.set_yticks(np.linspace(0, 1, 6))
                ax.set_ylabel('Precision' if col == 0 else 'ROC-AUC', fontsize=9)
                other = ax.twinx()
                other.plot(epoch, r_mean, color=ORANGE, marker='o', ms=4, lw=1.7, zorder=4)
                other.fill_between(epoch, r_mean - r_sd, r_mean + r_sd, color=ORANGE, alpha=.15, lw=0)
                other.set_ylim(0, rank_high)
                other.set_ylabel(rank, color=ORANGE, fontsize=9)
                other.tick_params(axis='y', colors=ORANGE, labelsize=8)
                other.spines['right'].set_color(ORANGE)
                other.spines['top'].set_visible(False)
            ax.set_title(Q_LABELS[col], fontsize=11, pad=9)
            ax.set_xlim(-.6, 30.6)
            ax.set_xticks([0, 5, 10, 20, 30])
            ax.set_xlabel('SSL epoch', fontsize=10)
            ax.tick_params(labelsize=9)
            ax.grid(color='#d8d8d8', lw=.65, alpha=.8)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            if not normalized:
                ax.spines['right'].set_visible(False)
    footer = [
        'Fixed official test · mean across seeds 6101, 6102, 6103 · epoch 0: untrained encoder.',
        ('Scaling: (mean − min mean) / (max mean − min mean), separately over epochs 0–30.'
         if normalized else 'Shading: ± sample SD across 3 seeds, not a confidence interval; axes use different units.'),
        ('This view compares trajectory shapes; equal heights do not imply equal metric values.'
         if normalized else 'The rank-axis range is identical across all methods and both columns in this figure.'),
        'Precision threshold: selected on validation with recall ≥ 0.5; test recall is unconstrained.',
        'Both classifiers use raw features. Raw/L2 switches only the spectral-rank calculation.',
    ]
    for i, line in enumerate(footer):
        fig.text(.08, .110 - .019 * i, line, fontsize=8.5, color='#525252')
    short_clf = 'lr' if classifier == CLASSIFIERS[0] else 'lgbm'
    stem = f'{category}_p{round(fraction * 100):02d}_{norm}_{short_clf}_{rank}_{mode}'
    for ext in ['png', 'svg']:
        fig.savefig(output / f'{stem}.{ext}', dpi=160, facecolor='white')
    plt.close(fig)
    return stem


def write_gallery(output, plots):
    template = Path(__file__).with_name('visa_overlays.template.html').read_text()
    # Compress the embedded SVG collection; all downloads also work from a lone HTML file.
    data = {key: {'stem': stem, 'image': 'data:image/svg+xml;base64,' +
                 base64.b64encode((output / f'{stem}.svg').read_bytes()).decode()}
            for key, stem in plots.items()}
    packed = base64.b64encode(gzip.compress(json.dumps(data).encode(), mtime=0)).decode()
    default = data['fryum|0.05|raw|logistic_regression|RankMe|minmax']['image']
    page = template.replace('__PLOTS_GZIP__', packed).replace('__DEFAULT_IMAGE__', default)
    for name, mime in [('paired_test_values.csv', 'text/csv'), ('mean_sd.csv', 'text/csv'),
                       ('minmax_bounds.csv', 'text/csv'), ('audit.json', 'application/json')]:
        embedded = base64.b64encode((output / name).read_bytes()).decode()
        page = page.replace(f'href="{name}"', f'download="{name}" href="data:{mime};base64,{embedded}"')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name in ['plot_visa_overlays.py', 'visa_overlays.template.html', 'plot_visa_results.py']:
            bundle.writestr(name, Path(__file__).with_name(name).read_bytes())
    embedded = base64.b64encode(buffer.getvalue()).decode()
    page = page.replace('href="reproduction.zip"',
                        f'download="visa_plot_scripts.zip" href="data:application/zip;base64,{embedded}"')
    (output / 'index.html').write_text(page)
    # Keep previously opened HTML paths usable; the build template has a distinct name.
    (output / 'visa_overlays.html').write_text(page)
    Path(__file__).with_name('visa_overlays.html').write_text(page)
    share_path = output.parent / 'VisA_downstream_rank_share.html'
    share_path.write_text(page)
    return share_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = (args.output or source.with_name(source.name + '-overlays')).resolve()
    assert source != output and source not in output.parents, 'Use a separate output directory'
    output.mkdir(parents=True, exist_ok=True)
    original_files = {p: sha256(p) for p in source.iterdir() if p.is_file()}
    joined, stats, scales, audit = load_data(source)
    joined.to_csv(output / 'paired_test_values.csv', index=False)
    stats.to_csv(output / 'mean_sd.csv', index=False)
    scales.to_csv(output / 'minmax_bounds.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'axes.edgecolor': '#777777', 'axes.linewidth': .7})
    plots = {}
    for cat, fraction, norm, clf, rank, mode in itertools.product(
            CATEGORIES, FRACTIONS, ['raw', 'l2'], CLASSIFIERS, RANKS, ['minmax', 'native']):
        key = '|'.join([cat, f'{fraction:.2f}', norm, clf, rank, mode])
        plots[key] = draw(stats, cat, fraction, norm, clf, rank, mode, output)
        if len(plots) % 16 == 0:
            print(f'Rendered {len(plots)}/192 figures', flush=True)
    assert all(sha256(path) == digest for path, digest in original_files.items())
    checks = {'source': str(source), 'source_audit': audit, 'paired_rows': len(joined),
              'summary_rows': len(stats), 'figures': len(plots), 'seeds_per_checkpoint': 3,
              'mean_then_minmax': True, 'source_report_unchanged': True,
              'constant_mean_trajectories': int(scales.constant.sum()),
              'normalization': '(mean - min(mean)) / (max(mean) - min(mean)), per trajectory over epochs 0–30',
              'outputs': {p.name: sha256(p) for p in output.glob('*.csv')}}
    (output / 'audit.json').write_text(json.dumps(checks, indent=2, ensure_ascii=False) + '\n')
    share_path = write_gallery(output, plots)
    for name in ['plot_visa_overlays.py', 'visa_overlays.template.html', 'plot_visa_results.py']:
        shutil.copy2(Path(__file__).with_name(name), output / name)
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.iterdir()):
            if path.is_file():
                bundle.write(path, f'{output.name}/{path.name}')
    print(json.dumps({'output': str(output), 'archive': str(archive), 'share_html': str(share_path),
                      'figures': len(plots)}, indent=2))


if __name__ == '__main__':
    main()
