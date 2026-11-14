"""Independent paired augmentation audit from original VisA v2 quality.csv.

Compares A0+Rotate / A0+CutPaste against A0 at the SAME epoch and seed.
Macro SD is across three seed-specific means on a fixed condition grid, never
across pooled conditions treated as independent repeats. No hypothesis testing.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/visa-v2-statistics-2026-11-14/summary'
KEYS = ['category', 'architecture', 'method', 'seed', 'classifier', 'epoch']
METRICS = ['roc_auc', 'average_precision']
EPOCHS = [5, 10, 30, 100]
EPS = 1e-10


def summarize(paired, dimensions):
    base = dimensions + ['augmentation', 'classifier', 'metric', 'epoch']
    per_seed = paired.groupby(base + ['seed']).delta_pp.mean()
    macro = per_seed.groupby(base).agg(mean_delta_pp='mean', seed_sd_of_macro_pp='std', n_seeds='size',
                                     macro_seed_positive=lambda a: (a > EPS).sum(),
                                     macro_seed_negative=lambda a: (a < -EPS).sum()).reset_index()
    half_width = t.ppf(.975, macro.n_seeds - 1) * macro.seed_sd_of_macro_pp / np.sqrt(macro.n_seeds)
    macro['macro_seed_ci95_low_pp'] = macro.mean_delta_pp - half_width
    macro['macro_seed_ci95_high_pp'] = macro.mean_delta_pp + half_width
    cells = paired.groupby(['category', 'architecture', 'method', 'augmentation',
                            'classifier', 'metric', 'epoch']).delta_pp.agg(
        mean='mean', sd='std', n='size', positive=lambda a: (a > EPS).sum(),
        negative=lambda a: (a < -EPS).sum()).reset_index()
    counts = cells.groupby(base).agg(
        n_conditions=('n', 'size'), mean_positive=('mean', lambda a: (a > EPS).sum()),
        mean_negative=('mean', lambda a: (a < -EPS).sum()),
        all3_positive=('positive', lambda a: (a == 3).sum()),
        all3_negative=('negative', lambda a: (a == 3).sum()),
        median_cell_sd_pp=('sd', 'median'), min_cell_mean_pp=('mean', 'min'),
        max_cell_mean_pp=('mean', 'max')).reset_index()
    out = macro.merge(counts, on=base, validate='one_to_one')
    return out, cells


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'reports/visa-v2-block4/quality.csv'
    q = pd.read_csv(source)
    q = q[q.split.eq('test') & q.method.ne('baseline') & q.epoch.isin(EPOCHS)].copy()
    assert len(q) == 324 * 2 * len(EPOCHS)
    assert not q.duplicated(KEYS + ['augmentation']).any()
    baseline = q[q.augmentation.eq('baseline')].set_index(KEYS)
    parts = []
    for augmentation in ['rotate', 'cutpaste']:
        changed = q[q.augmentation.eq(augmentation)].set_index(KEYS).reindex(baseline.index)
        assert changed[METRICS].notna().all().all()
        differences = (changed[METRICS] - baseline[METRICS]) * 100
        part = differences.reset_index().melt(id_vars=KEYS, value_vars=METRICS,
                                             var_name='metric', value_name='delta_pp')
        part['augmentation'] = augmentation
        parts.append(part)
    paired = pd.concat(parts, ignore_index=True)
    assert len(paired) == 3456
    macro, cells = summarize(paired, ['architecture'])
    by_method, _ = summarize(paired, ['architecture', 'method'])
    by_category, _ = summarize(paired, ['architecture', 'category'])
    checks = pd.read_csv(ROOT / 'reports/visa-v2-statistics-2026-11-14/paired_statistics.csv')
    checks = checks[checks.comparison.eq('augmentation_vs_A0') & checks.metric.isin(METRICS)]
    join = ['category', 'architecture', 'method', 'augmentation', 'classifier', 'metric']
    comparison = cells[cells.epoch.eq(100)].merge(checks, on=join, validate='one_to_one')
    assert len(comparison) == 288
    assert comparison.n_x.eq(3).all() and comparison.n_y.eq(3).all()
    error = float(np.max(np.abs(comparison['mean'] - comparison.mean_delta_pp)))
    sd_error = float(np.max(np.abs(comparison.sd - comparison.sd_delta_pp)))
    assert error < 1e-10 and sd_error < 1e-10
    assert comparison.positive_x.eq(comparison.positive_y).all()
    assert comparison.negative_x.eq(comparison.negative_y).all()
    for filename, data in [('aug_review_paired.csv', paired), ('aug_review_macro.csv', macro),
                           ('aug_review_by_method.csv', by_method), ('aug_review_by_category.csv', by_category),
                           ('aug_review_cells.csv', cells)]:
        data.to_csv(OUT / filename, index=False)
    audit = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
             'comparison': 'A0+Rotate or A0+CutPaste minus A0, matched category/architecture/method/seed/classifier/epoch',
             'epochs': EPOCHS, 'metrics': METRICS, 'paired_rows': len(paired),
             'final_cells_verified': len(comparison), 'max_mean_error_vs_existing_statistics': error,
             'max_sd_error_vs_existing_statistics': sd_error,
             'all_seed_sign_counts_match_existing_statistics': True,
             'macro_sd_definition': 'Sample SD of three seed-specific equal-weight means on the fixed condition grid.',
             'scope': 'Descriptive fixed-grid effects; conditions are not treated as independent replications.',
             'protocol': 'A0 already contains RandomResizedCrop and ColorJitter; Rotate and CutPaste are additions, not no-augmentation comparisons.'}
    (OUT / 'aug_review_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(audit, indent=2))
    print(macro[macro.epoch.eq(100)].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
