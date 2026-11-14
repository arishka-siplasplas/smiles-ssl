"""Recompute the requested VisA v2 comparisons for a portable table report.

All changes are paired by training seed, conditional on the fixed data split.
No practical-equivalence threshold or checkpoint selection is imposed.
"""
from pathlib import Path
import hashlib
import itertools
import json

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from analyze_visa_v2_statistics import holm

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reports/visa-v2-statistics-2026-11-14'
OUT = BASE / 'tables'
KEYS = ['category', 'architecture', 'method', 'augmentation', 'classifier']
METRICS = ['roc_auc', 'average_precision', 'precision_anomaly', 'recall_anomaly', 'f1_anomaly']
SEEDS = [6101, 6102, 6103]
EARLY = [0, 1, 2, 5, 10, 20, 30]
CONTRASTS = sorted(set(zip(EARLY[:-1], EARLY[1:])) | {(0, e) for e in EARLY[1:]} |
                   {(a, 100) for a in (0, 5, 10, 30)})
IDENTITY = ['kind'] + KEYS + ['from_epoch', 'to_epoch', 'metric']
DISPLAY_KEYS = ['kind', 'category', 'architecture', 'method', 'augmentation',
                'from_epoch', 'to_epoch', 'classifier', 'metric']
NUMERIC = ['delta6101', 'delta6102', 'delta6103', 'mean', 'sd', 'ci_low', 'ci_high',
           'positive', 'negative', 'tied']
SUMMARY_EXTRA = ['n_conditions', 'median_cell_sd', 'mean_positive', 'mean_negative',
                 'unanimous_positive', 'unanimous_negative']


def describe(frame):
    values = frame[[f'delta{s}' for s in SEEDS]].to_numpy()
    frame['mean'] = values.mean(axis=1)
    frame['sd'] = values.std(axis=1, ddof=1)
    se = frame.sd / np.sqrt(3)
    valid = se > 1e-10
    half = stats.t.ppf(.975, 2) * se
    frame['ci_low'] = np.where(valid, frame['mean'] - half, np.nan)
    frame['ci_high'] = np.where(valid, frame['mean'] + half, np.nan)
    frame['positive'] = (values > 1e-10).sum(axis=1)
    frame['negative'] = (values < -1e-10).sum(axis=1)
    frame['tied'] = (abs(values) <= 1e-10).sum(axis=1)
    return frame, values, valid


def row_label(r):
    direction = 'рост' if r['mean'] > 1e-10 else 'снижение' if r['mean'] < -1e-10 else 'без изменения'
    if r.positive == 3 or r.negative == 3:
        agreement = 'одинаковый знак в 3/3'
    elif r.positive and r.negative:
        agreement = 'знак меняется между seed'
    elif r.tied == 3:
        agreement = 'совпадение трёх наблюдений'
    else:
        agreement = 'есть нулевые изменения'
    return f'В среднем {direction}; {agreement}.'


def summarize_cells(cells):
    chunks = []
    for kind in ('duration', 'augmentation'):
        part = cells[cells.kind.eq(kind)]
        grouping = ['kind', 'architecture', 'from_epoch', 'to_epoch', 'classifier', 'metric']
        grouping += ['method'] if kind == 'duration' else ['augmentation']
        # Average the fixed grid within each seed first; only three seed means
        # are used for the uncertainty of that grid's mean effect.
        macro = part.groupby(grouping)[[f'delta{s}' for s in SEEDS]].mean().reset_index()
        macro, _, _ = describe(macro)
        info = part.groupby(grouping).agg(
            n_conditions=('mean', 'size'), median_cell_sd=('sd', 'median'),
            mean_positive=('mean', lambda s: int((s > 1e-10).sum())),
            mean_negative=('mean', lambda s: int((s < -1e-10).sum())),
            unanimous_positive=('positive', lambda s: int((s == 3).sum())),
            unanimous_negative=('negative', lambda s: int((s == 3).sum()))).reset_index()
        macro = macro.merge(info, on=grouping, validate='one_to_one')
        macro['category'] = 'all'
        macro['augmentation' if kind == 'duration' else 'method'] = 'all'
        chunks.append(macro)
    return pd.concat(chunks, ignore_index=True)[DISPLAY_KEYS + NUMERIC + SUMMARY_EXTRA]


def validate_previous(cells):
    index = cells.set_index(IDENTITY)
    errors = {}
    for name, path in [('original', BASE / 'paired_statistics.csv'),
                       ('early', BASE / 'early/early_paired_statistics.csv')]:
        old = pd.read_csv(path).rename(columns={
            'comparison': 'kind', 'early_epoch': 'from_epoch', 'late_epoch': 'to_epoch',
            'mean_delta_pp': 'mean', 'sd_delta_pp': 'sd',
            'ci95_low_pp': 'ci_low', 'ci95_high_pp': 'ci_high',
            **{f'delta_seed_{s}_pp': f'delta{s}' for s in SEEDS}})
        if name == 'early':
            old['kind'] = 'duration'
        old['kind'] = old.kind.replace({'augmentation_vs_A0': 'augmentation'})
        old = old.set_index(IDENTITY).sort_index()
        current = index.reindex(old.index)
        assert current['mean'].notna().all()
        cols = ['mean', 'sd', 'ci_low', 'ci_high', 'p_t'] + [f'delta{s}' for s in SEEDS]
        np.testing.assert_allclose(current[cols], old[cols], rtol=1e-10, atol=1e-11, equal_nan=True)
        errors[name] = {'rows': len(old), 'max_absolute_error': float(np.nanmax(abs(current[cols] - old[cols])))}
    aug = pd.read_csv(BASE / 'summary/aug_review_cells.csv').rename(columns={
        'epoch': 'from_epoch', 'mean_delta_pp': 'mean', 'sd_delta_pp': 'sd'})
    # That earlier augmentation review uses short column names for its moments.
    if 'mean' not in aug:
        aug = aug.rename(columns={'mean_delta': 'mean', 'sd_delta': 'sd'})
    aug['kind'], aug['to_epoch'] = 'augmentation', aug.from_epoch
    old = aug.set_index(IDENTITY).sort_index()
    current = index.reindex(old.index)
    np.testing.assert_allclose(current[['mean', 'sd']], old[['mean', 'sd']], atol=1e-11)
    errors['augmentation_review'] = {'rows': len(old), 'max_absolute_error': float(np.nanmax(abs(current[['mean', 'sd']] - old[['mean', 'sd']])))}
    return errors


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'reports/visa-v2-block4/quality.csv'
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert sha == '16397c62662ef686c1ebb61fccda137775a7318db64b7533171176d81c196f0b'
    q = pd.read_csv(source)
    q = q[q.method.ne('baseline') & q.split.eq('test')].copy()
    assert len(q) == 6480 and not q.duplicated(KEYS + ['seed', 'epoch']).any()
    assert np.isfinite(q[METRICS]).all().all() and q.support_anomaly.eq(40).all()
    chunks = []
    for a, b in CONTRASTS:
        before = q[q.epoch.eq(a)].set_index(KEYS + ['seed']).sort_index()
        after = q[q.epoch.eq(b)].set_index(KEYS + ['seed']).reindex(before.index)
        delta = (100 * (after[METRICS] - before[METRICS])).reset_index().melt(
            id_vars=KEYS + ['seed'], value_vars=METRICS, var_name='metric', value_name='delta')
        delta['kind'], delta['from_epoch'], delta['to_epoch'] = 'duration', a, b
        chunks.append(delta)
    ak = [k for k in KEYS if k != 'augmentation'] + ['seed']
    for epoch in (5, 10, 30, 100):
        at = q[q.epoch.eq(epoch)]
        baseline = at[at.augmentation.eq('baseline')].set_index(ak).sort_index()
        for augmentation in ('rotate', 'cutpaste'):
            other = at[at.augmentation.eq(augmentation)].set_index(ak).reindex(baseline.index)
            delta = (100 * (other[METRICS] - baseline[METRICS])).reset_index().melt(
                id_vars=ak, value_vars=METRICS, var_name='metric', value_name='delta')
            delta['kind'], delta['augmentation'] = 'augmentation', augmentation
            delta['from_epoch'] = delta['to_epoch'] = epoch
            chunks.append(delta)
    pairs = pd.concat(chunks, ignore_index=True)
    assert len(pairs) == 57240 and np.isfinite(pairs.delta).all()
    pivot = pairs.pivot(index=IDENTITY, columns='seed', values='delta')
    assert pivot.columns.tolist() == SEEDS and not pivot.isna().any().any()
    cells = pivot.rename(columns={s: f'delta{s}' for s in SEEDS}).reset_index()
    cells, values, valid = describe(cells)
    with np.errstate(divide='ignore', invalid='ignore'):
        cells['p_t'] = np.where(valid, 2 * stats.t.sf(abs(cells['mean'] / (cells.sd / np.sqrt(3))), 2), np.nan)
    cells['p_holm'] = holm(cells.p_t)
    flips = np.asarray(list(itertools.product((-1, 1), repeat=3)))
    cells['p_signflip'] = (abs(values @ flips.T / 3) >= abs(values.mean(axis=1))[:, None] - 1e-12).mean(axis=1)
    assert len(cells) == 19080 and len(CONTRASTS) == 15
    assert cells.groupby('kind').size().to_dict() == {'augmentation': 2880, 'duration': 16200}
    check = stats.ttest_1samp(values[valid], 0, axis=1)
    np.testing.assert_allclose(check.pvalue, cells.loc[valid, 'p_t'], atol=1e-13)
    np.testing.assert_allclose(check.confidence_interval().low, cells.loc[valid, 'ci_low'], atol=1e-11)
    np.testing.assert_allclose(check.confidence_interval().high, cells.loc[valid, 'ci_high'], atol=1e-11)
    np.testing.assert_allclose(multipletests(cells.p_t.fillna(1), method='holm')[1], cells.p_holm, atol=1e-14)
    previous = validate_previous(cells)
    summaries = summarize_cells(cells)
    assert len(summaries) == 1060 and summaries.n_conditions.eq(18).all()
    cells['short_interpretation'] = cells.apply(row_label, axis=1)
    cells[DISPLAY_KEYS + NUMERIC + ['p_t', 'p_holm', 'p_signflip', 'short_interpretation']].to_csv(OUT / 'all_paired_statistics.csv', index=False)
    summaries.to_csv(OUT / 'fixed_grid_summaries.csv', index=False)
    cols = DISPLAY_KEYS + NUMERIC + ['p_t', 'p_holm', 'p_signflip']
    summary_cols = DISPLAY_KEYS + NUMERIC + SUMMARY_EXTRA
    # pandas JSON conversion maps missing/undefined inferential values to null.
    records = lambda df: json.loads(df.to_json(orient='values', double_precision=10, force_ascii=False))
    meta = dict(title='VisA v2 · статистики', generated='2026-11-14',
                source=str(source.relative_to(ROOT)), source_sha256=sha,
                seeds=SEEDS, unit='п.п.', family_tests=len(cells),
                duration_contrasts=CONTRASTS, augmentation_epochs=[5, 10, 30, 100],
                independent_seeds=3, test_anomalies_per_category=40,
                nominal_significant=int(cells.p_t.lt(.05).sum()),
                holm_significant=int(cells.p_holm.lt(.05).sum()),
                degenerate_rows=int((~valid).sum()))
    payload = dict(meta=meta, columns=cols, rows=records(cells[cols]),
                   summary_columns=summary_cols, summary_rows=records(summaries[summary_cols]),
                   labels={'resnet18': 'ResNet-18', 'deit_tiny': 'ViT-Tiny',
                           'masked_reconstruction': 'Reconstruction', 'simclr': 'SimCLR', 'vicreg': 'VICReg',
                           'baseline': 'A0', 'rotate': 'Rotate', 'cutpaste': 'CutPaste',
                           'logistic_regression': 'Logistic Regression', 'lightgbm': 'LightGBM',
                           'roc_auc': 'AUC', 'average_precision': 'AP',
                           'precision_anomaly': 'Precision', 'recall_anomaly': 'Recall', 'f1_anomaly': 'F1',
                           'all': 'Все'})
    (OUT / 'data_table_payload.json').write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    audit = dict(meta, paired_seed_rows=len(pairs), summary_rows=len(summaries),
                 prior_results=previous, independent_scipy_checks=int(valid.sum()),
                 statsmodels_holm_match=True, source_unchanged=True)
    (OUT / 'table_data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
