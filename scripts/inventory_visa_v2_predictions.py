"""Read-only inventory and score verification for locally saved VisA v2 predictions.

Run from any directory with the project venv. Only writes predictions_coverage.json
in reports/visa-v2-statistics-2026-11-14/. No downloads, training, or image bootstrap are performed.
"""
from collections import Counter
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/visa-v2-statistics-2026-11-14'
IDENTITY = ['category', 'architecture', 'augmentation', 'method', 'seed']
EARLY = [0, 5, 10, 30]
PROBES = ['logistic_regression', 'lightgbm']


def describe(key):
    return dict(zip(IDENTITY, key))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'manifests/visa-v2/fixed.json').read_text())
    quality = pd.read_csv(ROOT / 'reports/visa-v2-final/data/quality_test.csv')
    quality = quality.set_index(IDENTITY + ['epoch', 'classifier'])
    runs = {}
    source_counts = {}
    for source in sorted((ROOT / 'runs').glob('visa-ssl-v2-*')):
        prediction_count = 0
        for identity_path in source.rglob('identity.json'):
            identity = json.loads(identity_path.read_text())
            files = list(identity_path.parent.glob('epoch_*/predictions.npz'))
            prediction_count += len(files)
            if identity['method'] == 'baseline':
                continue
            key = tuple(identity[k] for k in IDENTITY)
            for path in files:
                epoch = int(path.parent.name.split('_')[-1])
                runs.setdefault(key, {})[epoch] = path
        if prediction_count:
            source_counts[str(source.relative_to(ROOT))] = prediction_count

    duration = {}
    verified = set()
    maximum_error = 0.0
    score_pairs = 0
    for early in EARLY:
        eligible = {k: v for k, v in runs.items() if early in v and 100 in v}
        cells = Counter(k[:-1] for k in eligible)
        duration[f'100_minus_{early}'] = {
            'paired_training_runs': len(eligible),
            'complete_three_seed_training_cells': sum(n == 3 for n in cells.values()),
            'complete_three_seed_probe_cells': 2 * sum(n == 3 for n in cells.values()),
            'complete_cells': [dict(zip(IDENTITY[:-1], k)) for k, n in sorted(cells.items()) if n == 3],
            'partial_cells': [dict(zip(IDENTITY[:-1], k), n_seeds=n) for k, n in sorted(cells.items()) if n != 3],
            'paired_run_counts_by_architecture_augmentation': [
                {'architecture': k[0], 'augmentation': k[1], 'n': n}
                for k, n in sorted(Counter((k[1], k[2]) for k in eligible).items())],
        }
        for key, files in eligible.items():
            records = manifest['categories'][key[0]]['test']
            for epoch in [early, 100]:
                if (key, epoch) in verified:
                    continue
                with np.load(files[epoch].with_name('embeddings.npz')) as emb:
                    labels, paths = emb['test_labels'], emb['test_paths']
                assert paths.tolist() == [r['path'] for r in records]
                assert labels.tolist() == [r['label'] for r in records]
                with np.load(files[epoch]) as scores:
                    for probe in PROBES:
                        score = scores[probe + '_test']
                        reference = quality.loc[key + (epoch, probe)]
                        error = max(abs(roc_auc_score(labels, score) - reference.roc_auc),
                                    abs(average_precision_score(labels, score) - reference.average_precision))
                        assert error < 1e-12
                        maximum_error = max(maximum_error, error)
                        score_pairs += 1
                verified.add((key, epoch))

    augmentation = {}
    for aug in ['rotate', 'cutpaste']:
        paired = []
        for key, files in runs.items():
            if key[2] != aug or 100 not in files:
                continue
            baseline_key = key[:2] + ('baseline',) + key[3:]
            if 100 in runs.get(baseline_key, {}):
                paired.append(key)
        cells = Counter(k[:-1] for k in paired)
        augmentation[aug + '_minus_A0_epoch100'] = {
            'paired_training_runs': len(paired),
            'complete_three_seed_training_cells': sum(n == 3 for n in cells.values()),
            'complete_three_seed_probe_cells': 2 * sum(n == 3 for n in cells.values()),
            'complete_cells': [dict(zip(IDENTITY[:-1], k)) for k, n in sorted(cells.items()) if n == 3],
            'partial_cells': [dict(zip(IDENTITY[:-1], k), n_seeds=n) for k, n in sorted(cells.items()) if n != 3],
        }

    archives = []
    for top in ['runs', 'tmp', 'reports']:
        for archive in sorted((ROOT / top).rglob('*.zip')):
            with zipfile.ZipFile(archive) as handle:
                names = handle.namelist()
            selected = [n for n in names if 'prediction' in n.lower() or n.endswith('.npz')]
            archives.append({'path': str(archive.relative_to(ROOT)), 'member_count': len(names),
                             'prediction_or_npz_members': selected})

    output = {
        'analysis_date_moscow': '2026-11-14',
        'scope': 'Local files only; no remote status query or downloads. Final tables verified against block2 predictions.',
        'local_prediction_counts_including_baselines_and_partial_runs': source_counts,
        'local_training_runs_with_any_predictions': len(runs),
        'duration_contrast_coverage': duration,
        'augmentation_contrast_coverage': augmentation,
        'test_manifest': str((ROOT / 'manifests/visa-v2/fixed.json').resolve()),
        'test_counts': {category: {'normal': sum(r['label'] == 0 for r in groups['test']),
                                  'anomaly': sum(r['label'] == 1 for r in groups['test']),
                                  'total': len(groups['test'])}
                        for category, groups in manifest['categories'].items()},
        'verification': {'unique_completed_run_epoch_pairs': len(verified),
                         'probe_score_vectors_checked': score_pairs,
                         'metrics_recomputed_per_vector': ['roc_auc', 'average_precision'],
                         'max_absolute_difference_from_final_table': maximum_error,
                         'test_labels_and_ordered_paths_match_manifest': True},
        'archives': archives,
        'limits': ['No CutPaste predictions local.',
                   'Only 67 of 108 training configuration triples complete locally; 134 of 216 probe cells.',
                   'Image bootstrap feasible on covered cells without SSL retraining, conditional on existing seed models.',
                   'Do not stack block2 results with final cumulative tables as independent observations.'],
    }
    (OUT / 'predictions_coverage.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'duration': {k: {a: b for a, b in v.items() if not isinstance(b, list)} for k, v in duration.items()},
                      'augmentation': {k: {a: b for a, b in v.items() if not isinstance(b, list)} for k, v in augmentation.items()},
                      'verification': output['verification']}, indent=2))


if __name__ == '__main__':
    main()
