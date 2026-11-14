"""Exploratory paired-seed inference and replication planning for VisA v2.

All inference is conditional on the fixed data and training protocol. Three
seeds do not support checking the normality assumption of Student intervals.
This script never starts training or chooses an epoch on test performance.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
KEYS = ['category', 'architecture', 'augmentation', 'method', 'classifier']
RUN_KEYS = KEYS[:-1]
METRICS = ['roc_auc', 'average_precision', 'precision_anomaly', 'recall_anomaly', 'f1_anomaly']
SEEDS = [6101, 6102, 6103]
EPOCHS = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100]


def holm(values):
    """Holm family-wise correction, valid for arbitrary test dependence."""
    p = np.nan_to_num(np.asarray(values, dtype=float), nan=1.)
    order = np.argsort(p)
    adjusted = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty(len(p))
    result[order] = adjusted
    return result


def summarize(values):
    a = np.asarray(values, dtype=float)
    n = len(a)
    mean, sd = float(a.mean()), float(a.std(ddof=1))
    se = sd / math.sqrt(n)
    out = dict(n=n, mean_delta_pp=mean, sd_delta_pp=sd, se_delta_pp=se,
               positive=int((a > 1e-10).sum()), negative=int((a < -1e-10).sum()),
               zero=int((abs(a) <= 1e-10).sum()), degenerate_sd=sd < 1e-10)
    if sd < 1e-10:
        # Coincident discrete outcomes are not evidence of zero population noise.
        out.update(ci95_low_pp=np.nan, ci95_high_pp=np.nan, ci90_low_pp=np.nan,
                   ci90_high_pp=np.nan, p_t=np.nan, p_tost_1pp=np.nan, p_tost_2pp=np.nan)
    else:
        for coverage in (90, 95):
            half = float(stats.t.ppf((1 + coverage / 100) / 2, n - 1) * se)
            out[f'ci{coverage}_low_pp'], out[f'ci{coverage}_high_pp'] = mean-half, mean+half
        out['p_t'] = float(2 * stats.t.sf(abs(mean / se), n - 1))
        for margin in (1, 2):
            out[f'p_tost_{margin}pp'] = float(max(stats.t.sf((mean + margin)/se, n-1),
                                                       stats.t.cdf((mean - margin)/se, n-1)))
    # Exhaust all sign flips of paired differences: requires null symmetry.
    flips = np.asarray(list(itertools.product((-1, 1), repeat=n)))
    out['p_signflip'] = float(np.mean(abs((flips * a).mean(axis=1)) >= abs(mean) - 1e-12))
    return out


def classify(row, margin):
    lo, hi = row.ci95_low_pp, row.ci95_high_pp
    if not np.isfinite(lo):
        return 'degenerate_unresolved'
    if lo > -margin and hi < margin:
        return 'small_conditional'
    if lo > margin:
        return 'material_increase_conditional'
    if hi < -margin:
        return 'material_decrease_conditional'
    if lo < -margin and hi > margin:
        return 'both_material_directions'
    return 'boundary_unresolved'


def precision_n(sd, halfwidth=2., max_n=10000):
    """Plug-in total n for a nominal t95 halfwidth, NOT observed power."""
    if sd < 1e-10:
        return np.nan
    ns = np.arange(3, max_n+1)
    widths = stats.t.ppf(.975, ns-1) * sd / np.sqrt(ns)
    valid = ns[widths <= halfwidth]
    return int(valid[0]) if len(valid) else max_n + 1


def validate_and_pair(source):
    q = pd.read_csv(source / 'quality.csv')
    d = q[q.method.ne('baseline') & q.split.eq('test')].copy()
    assert len(q) == 19656 and len(d) == 6480
    assert not d.duplicated(KEYS + ['seed', 'epoch']).any()
    assert np.isfinite(d[METRICS]).all().all()
    assert d.groupby(KEYS).size().eq(30).all()
    for _, g in d.groupby(KEYS):
        assert set(zip(g.seed, g.epoch)) == set(itertools.product(SEEDS, EPOCHS))
    assert d.support_anomaly.eq(40).all()
    # The shared baseline must be identical, not counted as nine observations.
    assert d[d.epoch.eq(0)].groupby(['category', 'architecture', 'seed', 'classifier'])[METRICS].nunique().eq(1).all().all()
    pairs = []
    final = d[d.epoch.eq(100)].set_index(KEYS + ['seed']).sort_index()
    for early in (0, 5, 10, 30):
        before = d[d.epoch.eq(early)].set_index(KEYS + ['seed']).reindex(final.index)
        for metric in METRICS:
            frame = (100*(final[metric] - before[metric])).rename('delta_pp').reset_index()
            frame['comparison'] = 'duration'
            frame['early_epoch'], frame['late_epoch'], frame['metric'] = early, 100, metric
            pairs.append(frame)
    ak = [k for k in KEYS if k != 'augmentation'] + ['seed']
    f = d[d.epoch.eq(100)]
    base = f[f.augmentation.eq('baseline')].set_index(ak).sort_index()
    for aug in ('rotate', 'cutpaste'):
        other = f[f.augmentation.eq(aug)].set_index(ak).reindex(base.index)
        for metric in METRICS:
            frame = (100*(other[metric]-base[metric])).rename('delta_pp').reset_index()
            frame['augmentation'], frame['comparison'] = aug, 'augmentation_vs_A0'
            frame['early_epoch'], frame['late_epoch'], frame['metric'] = 100, 100, metric
            pairs.append(frame)
    paired = pd.concat(pairs, ignore_index=True)
    assert len(paired) == 15120 and np.isfinite(paired.delta_pp).all()
    # Independent reconciliation with previously published descriptive deltas.
    old = pd.read_csv(ROOT / 'reports/visa-v2-final/data/quality_paired_seed_deltas.csv')
    index = KEYS + ['seed', 'early_epoch', 'late_epoch', 'metric']
    new = paired[paired.comparison.eq('duration')].set_index(index).sort_index().delta_pp
    np.testing.assert_allclose(new, old.set_index(index).sort_index().delta_pp, atol=1e-11, rtol=1e-12)
    return paired


def main(source, output):
    output.mkdir(parents=True, exist_ok=True)
    audit = json.loads((ROOT / 'reports/visa-v2-final/data/report_audit.json').read_text())
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
              for name in ('quality.csv', 'loss.csv', 'ranks.csv')}
    assert all(hashes[k] == audit['source_sha256'][k] for k in hashes)
    paired = validate_and_pair(source)
    groupkeys = ['comparison'] + KEYS + ['early_epoch', 'late_epoch', 'metric']
    records = []
    for key, group in paired.groupby(groupkeys, sort=True):
        assert sorted(group.seed) == SEEDS
        row = dict(zip(groupkeys, key))
        row.update(summarize(group.sort_values('seed').delta_pp))
        row.update({f'delta_seed_{r.seed}_pp': r.delta_pp for r in group.itertuples()})
        records.append(row)
    table = pd.DataFrame(records)
    assert len(table) == 5040
    # Families declared before examining the correction results: all 5040
    # requested duration/augmentation outcomes, separately for each hypothesis.
    for pcol in ('p_t', 'p_tost_1pp', 'p_tost_2pp'):
        table[pcol + '_holm_all5040'] = holm(table[pcol])
        table[pcol + '_holm_panel'] = table.groupby(
            ['comparison', 'early_epoch', 'classifier', 'metric'])[pcol].transform(holm)
    for margin in (1, 2):
        table[f'decision_{margin}pp'] = table.apply(classify, margin=margin, axis=1)
    table.to_csv(output / 'paired_statistics.csv', index=False)
    paired.to_csv(output / 'paired_seed_deltas.csv', index=False)

    focus = table[table.comparison.eq('duration') & table.early_epoch.eq(30)].copy()
    counts = []
    for (classifier, metric), g in focus.groupby(['classifier', 'metric']):
        counts.append(dict(classifier=classifier, metric=metric, n_conditions=len(g),
            nominal_t_significant=int(g.p_t.lt(.05).sum()),
            panel_holm_significant=int(g.p_t_holm_panel.lt(.05).sum()),
            global_holm_significant=int(g.p_t_holm_all5040.lt(.05).sum()),
            small_ci95_2pp=int(g.decision_2pp.eq('small_conditional').sum()),
            small_ci95_1pp=int(g.decision_1pp.eq('small_conditional').sum()),
            material_ci95_2pp=int(g.decision_2pp.str.startswith('material_').sum()),
            both_directions_ci95_2pp=int(g.decision_2pp.eq('both_material_directions').sum()),
            nominal_tost_2pp=int(g.p_tost_2pp.lt(.05).sum()),
            global_holm_tost_2pp=int(g.p_tost_2pp_holm_all5040.lt(.05).sum())))
    pd.DataFrame(counts).to_csv(output / 'late_training_summary.csv', index=False)

    loss = pd.read_csv(source / 'loss.csv')
    assert len(loss) == 32400
    cost = loss.groupby(RUN_KEYS + ['seed']).seconds.sum().groupby(RUN_KEYS).mean()/3600
    jobs = []
    for key, g in focus.groupby(RUN_KEYS):
        row = dict(zip(RUN_KEYS, key))
        main_g = g[g.classifier.eq('logistic_regression') & g.metric.isin(['roc_auc', 'average_precision'])]
        guard = g[g.metric.isin(['roc_auc', 'average_precision'])]
        for metric, prefix in [('roc_auc', 'auc'), ('average_precision', 'ap')]:
            r = main_g[main_g.metric.eq(metric)].iloc[0]
            for name in ('mean_delta_pp','sd_delta_pp','ci95_low_pp','ci95_high_pp','decision_2pp'):
                row[prefix+'_'+name] = r[name]
            row[prefix+'_positive_seeds'] = int(r.positive)
            row[prefix+'_negative_seeds'] = int(r.negative)
            row[prefix+'_unanimous_direction'] = bool(r.positive == 3 or r.negative == 3)
            row[prefix+'_ci95_width_pp'] = float(r.ci95_high_pp-r.ci95_low_pp)
            row[prefix+'_n_total_for_2pp_halfwidth_plugin'] = precision_n(r.sd_delta_pp)
            row[prefix+'_n_total_for_2pp_halfwidth_sd_x1_5'] = precision_n(1.5*r.sd_delta_pp)
        row['both_lr_metrics_small_2pp'] = bool(main_g.decision_2pp.eq('small_conditional').all())
        row['auc_ap_both_probes_small_2pp'] = bool(guard.decision_2pp.eq('small_conditional').all())
        row['ambiguous_lr_metrics'] = int(main_g.decision_2pp.isin(['both_material_directions','boundary_unresolved','degenerate_unresolved']).sum())
        row['sd_max_auc_ap_lr_pp'] = float(main_g.sd_delta_pp.max())
        row['mean_small_but_uncertain'] = bool(((main_g.mean_delta_pp.abs() <= 2) & main_g.decision_2pp.ne('small_conditional')).any())
        row['training_gpu_hours_per_seed_observed'] = float(cost.loc[key])
        row['training_gpu_hours_plus3_observed'] = float(3*cost.loc[key])
        jobs.append(row)
    # User's criterion is repeatability, not a fixed two-point equivalence bound.
    # Rank by observed spread only; margin-based flags remain a sensitivity view.
    jobs = pd.DataFrame(jobs).sort_values('sd_max_auc_ap_lr_pp', ascending=False)
    jobs['queue_order'] = np.arange(1,len(jobs)+1)
    jobs.to_csv(output / 'replication_candidates.csv', index=False)
    repeatability = jobs.sort_values('sd_max_auc_ap_lr_pp').copy()
    repeatability['stability_rank_within_108'] = np.arange(1,len(repeatability)+1)
    repeatability.to_csv(output / 'repeatability_by_condition.csv', index=False)
    # One finite planning scenario, explicitly not a unique optimal allocation.
    batch = jobs.head(8).copy()
    batch['new_seeds_proposed'] = '6104;6105;6106'
    batch['new_runs'] = 3
    batch['scope'] = 'duration_100_minus_30; retain_all_prespecified_checkpoints'
    batch.to_csv(output / 'proposed_batch_24_runs.csv', index=False)

    # Per-seed macro summaries keep all shared-initialization dependencies.
    macro = paired[paired.comparison.eq('duration')].groupby(
        ['architecture','method','classifier','early_epoch','metric','seed']).delta_pp.mean().reset_index()
    macros = []
    mk = ['architecture','method','classifier','early_epoch','metric']
    for key, g in macro.groupby(mk):
        macros.append({**dict(zip(mk,key)), **summarize(g.delta_pp), 'fixed_conditions_per_seed':18})
    macro_table = pd.DataFrame(macros)
    # Macro intervals are descriptive; do not create an uncorrected second
    # collection of significance claims outside the declared cell family.
    macro_table.drop(columns=[c for c in macro_table if c.startswith('p_')]).to_csv(
        output / 'macro_fixed_grid_seed_intervals.csv', index=False)
    summary = dict(date='2026-11-14', source=str(source.resolve()), source_sha256=hashes,
        audit=dict(training_runs=324, complete_training_runs=324, missing_training_runs=0,
                   training_configurations=108, seeds=SEEDS, probes=2, test_anomalies_per_category=40,
                   validation_anomalies_per_category=12, comparisons_total=len(table),
                   duplicated_or_missing_pairs=0, existing_deltas_reconciled=True,
                   degenerate_sd_rows=int(table.degenerate_sd.sum())),
        family='Holm across all 5040 duration and augmentation comparisons, separately for zero-null and each equivalence margin; narrower panel Holm is sensitivity only.',
        significance=dict(nominal_t=int(table.p_t.lt(.05).sum()),
                          global_holm_t=int(table.p_t_holm_all5040.lt(.05).sum()),
                          global_holm_tost_1pp=int(table.p_tost_1pp_holm_all5040.lt(.05).sum()),
                          global_holm_tost_2pp=int(table.p_tost_2pp_holm_all5040.lt(.05).sum())),
        late_training=counts,
        joint_small=dict(auc_ap_lr=int(jobs.both_lr_metrics_small_2pp.sum()),
                         auc_ap_both_probes=int(jobs.auc_ap_both_probes_small_2pp.sum())),
        user_priority='Small spread / repeatability. Margins of 1 and 2 pp are illustrative sensitivity scenarios, not user-specified stopping rules.',
        repeatability=[dict(classifier=classifier,metric=metric,
                            sd_min_pp=float(g.sd_delta_pp.min()),
                            sd_q25_pp=float(g.sd_delta_pp.quantile(.25)),
                            sd_median_pp=float(g.sd_delta_pp.median()),
                            sd_q75_pp=float(g.sd_delta_pp.quantile(.75)),
                            sd_max_pp=float(g.sd_delta_pp.max()),
                            all_three_same_direction=int((g.positive.eq(3)|g.negative.eq(3)).sum()))
                       for (classifier,metric),g in focus.groupby(['classifier','metric'])],
        proposed_batch=dict(configurations=8,new_seeds_per_configuration=3,new_runs=24,
                            estimated_training_gpu_hours=float(batch.training_gpu_hours_plus3_observed.sum()),
                            excludes='feature extraction, probing, I/O, orchestration; not a launched batch'),
        limitations=['Exploratory, conditional on fixed train/validation/test objects.',
                     'n=3: Student assumptions cannot be checked; no robust per-cell significance.',
                     'Nominal intervals are not simultaneous; small_conditional is a planning flag, not corrected proof of equivalence.',
                     'New seed IDs include probe random_state. Classifiers and epochs are not new replicates.',
                     '100−30 is a current planning focus, not a newly claimed preregistered primary endpoint.',
                     'Precision must be read with recall and AP. Same numerical margins need not imply same scientific utility.',
                     'Precision sample size is a plug-in scenario from uncertain SD, not observed power or a guarantee.',
                     'Targeted confirmations selected after test inspection require independent data for generalization claims.'])
    object_path = output / 'object_bootstrap.csv'
    if object_path.exists():
        objects = pd.read_csv(object_path)
        merged = focus.merge(objects, on=KEYS+['metric','early_epoch','late_epoch'],
                             how='left', suffixes=('', '_object'), validate='one_to_one')
        available = merged.n_fitted_seed_models.notna()
        np.testing.assert_allclose(merged.loc[available,'mean_delta_pp'],
                                   merged.loc[available,'mean_delta_pp_object'], atol=1e-11)
        merged['object_bootstrap_available'] = available
        merged['next_information_needed'] = 'retrieve_predictions_or_metric_not_bootstrapped'
        for idx, r in merged[available].iterrows():
            seed_small = r.decision_2pp == 'small_conditional'
            object_small = bool(r.object_ci95_within_2pp)
            if seed_small and object_small:
                action = 'lower_priority_conditional_two_separate_intervals'
            elif seed_small:
                action = 'test_object_uncertainty_cannot_be_resolved_by_seeds_alone'
            elif object_small:
                action = 'seed_variability_or_seed_interval_boundary'
            elif ((r.ci95_high_pp < -2 and r.object_ci95_high_pp < -2) or
                  (r.ci95_low_pp > 2 and r.object_ci95_low_pp > 2)):
                action = 'coherent_material_signal_targeted_confirmation'
            else:
                action = 'both_uncertainties_or_practical_boundary'
            merged.loc[idx,'next_information_needed'] = action
        merged.to_csv(output / 'late_training_uncertainty_sources.csv', index=False)
        summary['object_uncertainty'] = dict(
            conditions_covered=int(objects[RUN_KEYS].drop_duplicates().shape[0]),
            total_conditions=108, replicates=int(objects.bootstrap_replicates.iloc[0]),
            scope='Two separate conditional intervals, not a joint seed-and-object interval.',
            rows_by_next_information=merged[available].next_information_needed.value_counts().to_dict())
    (output / 'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'reports/visa-v2-block4')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/visa-v2-statistics-2026-11-14')
    args=parser.parse_args()
    main(args.source,args.output)
