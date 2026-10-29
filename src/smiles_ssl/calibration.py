"""Class-blind corruption screening and downstream validation-based selection.

Screening predictors use only outer-train labels and group-held-out predictions.
Their scores on corrupted objects are proxies, not true labels of new objects.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import warnings

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from .augment import Corruptor
from .config import SEEDS, default_config, run_id
from .data import load_table, validate_split_integrity
from .io import digest, read_json, write_json
from .metrics import binary_scores

METHODS = ['simclr', 'vicreg', 'masked_reconstruction']
PILOT_SEEDS = [3101, 3102, 3103]


def screen_spec():
    return {'version': 'tabular-augmentation-calibration-v1', 'folds': 3, 'fold_seed': 20261029,
            'draw_seeds': [7101, 7102, 7103, 7104, 7105], 'fit_normal_fpr': .05,
            'ap_drop_tolerance': .02, 'tpr_drop_tolerance': .05, 'fpr_rise_tolerance': .02,
            'logreg_C': 1., 'catboost_iterations': 300, 'catboost_depth': 4,
            'pilot_epochs': 100, 'schedule_epochs': 1000, 'pilot_seeds': PILOT_SEEDS,
            'validation_near_tie_ap': .005, 'donor_policy': 'uniform_all_fit_rows_no_labels',
            'mean_policy': 'all_fit_rows_no_labels'}


def candidates():
    rows = []
    for family in ['zero', 'mean', 'donor']:
        for p in [.01, .025, .05, .1, .2]:
            rows.append({'family': family, 'fraction': p, 'sigma': 0.,
                         'mask_sampling': 'bernoulli', 'patch_size': 16})
    for p in [.05, .1]:
        for sigma in [.01, .03, .1]:
            rows.append({'family': 'noise', 'fraction': p, 'sigma': sigma,
                         'mask_sampling': 'bernoulli', 'patch_size': 16})
    return rows


def candidate_id(a):
    return f"{a['family']}_p{a['fraction']:g}_s{a['sigma']:g}"


def strength(a):
    return a['fraction'] * (a['sigma'] ** 2 if a['family'] == 'noise' else 1)


def eligible_columns(x):
    return [j for j in range(x.shape[1]) if len(np.unique(x[:, j])) > 20
            and x[:, j].std() > 0
            and not np.allclose(x[:, j], np.rint(x[:, j]), atol=1e-7, rtol=0)]


def corrupt_queries(queries, fit_bank, prep, augmentation, seed):
    """No labels accepted; held-out queries can only draw donors from fit_bank."""
    donor = None
    if augmentation['family'] == 'donor':
        generator = torch.Generator().manual_seed(seed + 100000)
        rows = torch.randint(len(fit_bank), queries.shape, generator=generator)
        donor = fit_bank[rows, torch.arange(queries.shape[1])[None, :]]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        fn = Corruptor('table', augmentation, prep['mean'], prep['scale'], prep['noise_columns'])
        return fn(queries, donor)


def _rate(mask, event):
    return float(np.mean(event[mask])) if np.any(mask) else None


def proxy_metrics(y, clean, corrupted, thresholds, changed):
    before, after = clean >= thresholds, corrupted >= thresholds
    metrics = binary_scores(y, corrupted)
    metrics.update({
        'tpr': _rate(y == 1, after), 'fpr': _rate(y == 0, after),
        'anomaly_score_mean_shift': float(np.mean(corrupted[y == 1] - clean[y == 1])),
        'normal_score_mean_shift': float(np.mean(corrupted[y == 0] - clean[y == 0])),
        'detected_anomaly_to_below_threshold': _rate((y == 1) & before, ~after),
        'normal_to_above_threshold': _rate((y == 0) & ~before, after),
        'changed_detected_anomaly_to_below_threshold': _rate((y == 1) & before & changed, ~after),
        'changed_normal_to_above_threshold': _rate((y == 0) & ~before & changed, after),
    })
    return metrics


def classify_risk(per_probe, spec):
    flags = {}
    for name, r in per_probe.items():
        flags[name] = (r['mean_ap_drop'] > spec['ap_drop_tolerance']
                       or r['mean_tpr_drop'] > spec['tpr_drop_tolerance']
                       or r['mean_fpr_rise'] > spec['fpr_rise_tolerance'])
    return 'flagged' if all(flags.values()) else 'review' if any(flags.values()) else 'not_flagged'


def shortlist(rows):
    """Retain weakest and strongest unflagged candidates within each family."""
    selected = []
    for family in ['zero', 'mean', 'donor', 'noise']:
        pool = sorted([r for r in rows if r['augmentation']['family'] == family and r['risk'] == 'not_flagged'],
                      key=lambda r: (strength(r['augmentation']), candidate_id(r['augmentation'])))
        if pool:
            selected.append(pool[0]['id'])
            if pool[-1]['id'] != pool[0]['id']:
                selected.append(pool[-1]['id'])
    return selected


def screen(manifest_path, data_root, output, spec=None):
    from catboost import CatBoostClassifier
    spec = screen_spec() if spec is None else spec
    manifest = read_json(manifest_path)
    if manifest['modality'] != 'table':
        raise ValueError('This calibration protocol is tabular only.')
    path = Path(data_root) / manifest['source_filename']
    if digest(path) != manifest['source_sha256']:
        raise ValueError('Dataset differs from the prepared version.')
    x_all, y_all = load_table(path)
    all_groups = [hashlib.sha256(row.tobytes()).hexdigest() for row in x_all]
    validate_split_integrity(manifest, y_all, all_groups)
    ids = np.asarray(manifest['splits']['train'])
    x, y, groups = x_all[ids], y_all[ids], np.asarray(all_groups)[ids]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    augments = candidates()
    n, repeats = len(x), len(spec['draw_seeds'])
    clean = {name: np.zeros(n) for name in ['logreg', 'catboost']}
    thresholds = {name: np.zeros(n) for name in clean}
    predictions = {candidate_id(a): {name: np.zeros((repeats, n)) for name in clean} for a in augments}
    changed = {candidate_id(a): np.zeros((repeats, n), dtype=bool) for a in augments}
    selected = {candidate_id(a): np.zeros((repeats, n)) for a in augments}
    displacement = {candidate_id(a): np.zeros((repeats, n)) for a in augments}
    fold_records, fold_id = [], np.zeros(n, dtype=int)
    cv = StratifiedGroupKFold(n_splits=spec['folds'], shuffle=True, random_state=spec['fold_seed'])
    unavailable = set()
    for fold, (fit, audit) in enumerate(cv.split(x, y, groups)):
        if set(y[fit]) != {0, 1} or set(y[audit]) != {0, 1}:
            raise ValueError('Each internal fold needs both classes.')
        assert not set(groups[fit]) & set(groups[audit])
        fold_id[audit] = fold
        mean, scale = x[fit].mean(0), x[fit].std(0)
        scale = np.where(scale > 0, scale, 1.)
        noise = eligible_columns(x[fit])
        if manifest['preprocessing']['noise_policy'].startswith('explicit'):
            noise = sorted(set(noise) & set(manifest['preprocessing']['noise_columns']))
        prep = {'mean': mean.tolist(), 'scale': scale.tolist(), 'noise_columns': noise}
        bank = torch.from_numpy(((x[fit] - mean) / scale).astype(np.float32))
        query = torch.from_numpy(((x[audit] - mean) / scale).astype(np.float32))
        models = {
            'logreg': make_pipeline(StandardScaler(), LogisticRegression(C=spec['logreg_C'],
                        class_weight='balanced', max_iter=10000, random_state=spec['fold_seed'])),
            'catboost': CatBoostClassifier(iterations=spec['catboost_iterations'], depth=spec['catboost_depth'],
                        l2_leaf_reg=3., learning_rate=.05, auto_class_weights='Balanced', loss_function='Logloss',
                        random_seed=spec['fold_seed'], thread_count=1, allow_writing_files=False, verbose=False)}
        for name, model in models.items():
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                model.fit(bank.numpy(), y[fit])
            clean[name][audit] = model.predict_proba(query.numpy())[:, 1]
            fit_scores = model.predict_proba(bank.numpy())[:, 1]
            thresholds[name][audit] = np.nextafter(
                np.quantile(fit_scores[y[fit] == 0], 1 - spec['fit_normal_fpr'], method='higher'), np.inf)
        fold_records.append({'fold': fold, 'fit_source_indices': ids[fit].tolist(),
                             'audit_source_indices': ids[audit].tolist(), 'preprocessing': prep,
                             'audit_anomalies': int(y[audit].sum())})
        for a in augments:
            key = candidate_id(a)
            if a['family'] == 'noise' and not noise:
                unavailable.add(key)
                continue
            for repeat, seed in enumerate(spec['draw_seeds']):
                z, mask = corrupt_queries(query, bank, prep, a, seed + fold * 1000)
                changed[key][repeat, audit] = (z != query).any(1).numpy()
                selected[key][repeat, audit] = mask.float().mean(1).numpy()
                displacement[key][repeat, audit] = (z - query).square().mean(1).sqrt().numpy()
                for name, model in models.items():
                    predictions[key][name][repeat, audit] = model.predict_proba(z.numpy())[:, 1]
        print(f"{manifest['dataset']}: screen fold {fold + 1}/{spec['folds']} done", flush=True)
    baseline = {name: proxy_metrics(y, values, values, thresholds[name], np.zeros(n, dtype=bool))
                for name, values in clean.items()}
    rows = []
    for a in augments:
        key = candidate_id(a)
        if key in unavailable:
            rows.append({'id': key, 'augmentation': a, 'risk': 'unavailable', 'reason': 'No eligible noise columns in an inner fit fold.'})
            continue
        per_probe = {}
        for name in clean:
            draws = [proxy_metrics(y, clean[name], predictions[key][name][r], thresholds[name], changed[key][r])
                     for r in range(repeats)]
            per_probe[name] = {'draws': draws,
                              'mean_ap_drop': baseline[name]['ap'] - float(np.mean([r['ap'] for r in draws])),
                              'mean_tpr_drop': baseline[name]['tpr'] - float(np.mean([r['tpr'] for r in draws])),
                              'mean_fpr_rise': float(np.mean([r['fpr'] for r in draws])) - baseline[name]['fpr']}
        rows.append({'id': key, 'augmentation': a, 'risk': classify_risk(per_probe, spec), 'probes': per_probe,
                     'selected_cell_fraction': float(selected[key].mean()),
                     'modified_row_fraction': float(changed[key].mean()),
                     'modified_normal_fraction': float(changed[key][:, y == 0].mean()),
                     'modified_anomaly_fraction': float(changed[key][:, y == 1].mean()),
                     'mean_standardized_rms': float(displacement[key].mean()),
                     'p95_standardized_rms': float(np.quantile(displacement[key], .95))})
    from .run import runtime_info
    report = {'dataset': manifest['dataset'], 'manifest_sha256': digest(manifest_path),
              'environment': runtime_info('cpu'),
              'source_hashes': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
              'spec': spec, 'split_usage': 'outer train only, duplicate-group-held-out inner folds',
              'augmentation_label_policy': 'class_blind', 'baseline_proxies': baseline,
              'candidates': rows, 'shortlist': shortlist(rows),
              'interpretation': 'Proxy risk screen; not proof that anomaly identity is preserved. Repeated draws are not independent datasets.'}
    write_json(out / 'screen.json', report)
    write_json(out / 'inner_folds.json', fold_records)
    return report


def make_pilot_plan(screen_paths, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    jobs = []
    for path in screen_paths:
        report = read_json(path)
        chosen = [r for r in report['candidates'] if r['id'] in report['shortlist']]
        identity = {'id': 'identity', 'augmentation': {'family': 'mean', 'fraction': 0., 'sigma': 0.,
                                                      'mask_sampling': 'bernoulli', 'patch_size': 16}}
        for row in chosen + [identity]:
            for method in METHODS:
                if row['id'] == 'identity' and method == 'masked_reconstruction':
                    continue  # Zero masked targets is not a reconstruction objective.
                for seed in report['spec']['pilot_seeds']:
                    c = default_config('table', report['dataset'], method, seed)
                    c.update(stage='augmentation_calibration', epochs=report['spec']['pilot_epochs'],
                             schedule_epochs=report['spec']['schedule_epochs'], checkpoints=[0, 20, report['spec']['pilot_epochs']],
                             short_epoch=20, augmentation=row['augmentation'], calibration_candidate_id=row['id'],
                             calibration_screen_sha256=digest(path))
                    filename = run_id(c) + '.json'
                    write_json(out / filename, c)
                    jobs.append({'config': filename, 'id': run_id(c), 'dataset': report['dataset'], 'method': method,
                                 'seed': seed, 'candidate_id': row['id'], 'screen_sha256': digest(path),
                                 'manifest_sha256': report['manifest_sha256'], 'status': 'planned'})
    write_json(out / 'plan.json', {'stage': 'augmentation_calibration', 'n_runs': len(jobs), 'jobs': jobs,
                                  'selection': 'fixed final pilot epoch, clean validation AP; test sealed'})
    return jobs


def select_and_promote(plan_path, runs_root, output, near_tie=.005):
    """Require the entire planned study; choose a common strength for all methods."""
    plan = read_json(plan_path)
    planned = {j['id']: j for j in plan['jobs']}
    found = {}
    for path in Path(runs_root).rglob('completion.json'):
        folder = path.parent
        c = read_json(folder / 'config.json')
        if c.get('stage') != 'augmentation_calibration' or run_id(c) not in planned:
            continue
        key = run_id(c)
        if key in found:
            raise ValueError(f'Duplicate pilot run: {key}')
        job = planned[key]
        expected = read_json(Path(plan_path).parent / job['config'])
        if c != expected or digest(folder / 'manifest.json') != job['manifest_sha256']:
            raise ValueError(f'Pilot config/manifest mismatch: {folder}')
        ev = read_json(folder / f"epoch_{c['epochs']:04d}" / 'evaluation.json')
        found[key] = (c, ev)
    missing = sorted(set(planned) - set(found))
    if missing:
        raise ValueError(f'{len(missing)} planned pilots are incomplete; no selection made.')
    pooled = defaultdict(lambda: defaultdict(list))
    for key, (c, ev) in found.items():
        if c['calibration_candidate_id'] == 'identity':
            continue
        pooled[(c['dataset'], c['augmentation']['family'])][c['calibration_candidate_id']].append((c, ev))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    selections, jobs = [], []
    for (dataset, family), choices in sorted(pooled.items()):
        summaries = []
        for candidate, results in choices.items():
            per_probe = {name: float(np.mean([ev['probes'][name]['validation']['ap'] for _, ev in results]))
                         for name in ['logreg', 'catboost']}
            summaries.append({'candidate': candidate, 'augmentation': results[0][0]['augmentation'],
                              'score': float(np.mean(list(per_probe.values()))), 'per_probe_ap': per_probe})
        best = max(s['score'] for s in summaries)
        winner = min([s for s in summaries if s['score'] >= best - near_tie], key=lambda s: strength(s['augmentation']))
        controls, comparisons = {}, {}
        winner_runs = choices[winner['candidate']]
        for method in ['simclr', 'vicreg']:
            control_evals = [ev for c, ev in found.values() if c['dataset'] == dataset
                             and c['method'] == method and c['calibration_candidate_id'] == 'identity']
            controls[method] = {name: float(np.mean([ev['probes'][name]['validation']['ap'] for ev in control_evals]))
                                for name in ['logreg', 'catboost']}
            comparisons[method] = {name: float(np.mean([ev['probes'][name]['validation']['ap']
                                   for c, ev in winner_runs if c['method'] == method])) - controls[method][name]
                                   for name in ['logreg', 'catboost']}
        selections.append({'dataset': dataset, 'family': family, 'chosen': winner, 'all_candidates': summaries,
                           'identity_control_ap': controls, 'chosen_minus_identity_ap': comparisons,
                           'review_before_long_runs': any(v < -.02 for row in comparisons.values() for v in row.values())})
        for method in METHODS:
            for seed in SEEDS:
                c = default_config('table', dataset, method, seed)
                c.update(augmentation=winner['augmentation'], schedule_epochs=1000,
                         augmentation_selection_plan_sha256=digest(plan_path))
                filename = run_id(c) + '.json'
                write_json(out / filename, c)
                jobs.append({'id': run_id(c), 'config': filename, 'dataset': dataset, 'method': method, 'seed': seed})
    write_json(out / 'selection.json', {'near_tie_absolute_ap': near_tie, 'selections': selections,
                                       'selection_data': 'clean validation only', 'common_strength_across_methods': True})
    write_json(out / 'plan.json', {'stage': 'full_after_augmentation_calibration', 'n_runs': len(jobs), 'jobs': jobs})
