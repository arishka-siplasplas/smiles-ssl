"""Refit both probes from scratch; choose parameters using validation AP only."""
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import binary_scores


def fit_probes(train, y_train, validation, y_validation, config, seed):
    from catboost import CatBoostClassifier
    results, fitted = {}, {}
    grids = {
        'logreg': [{'C': c, 'class_weight': w} for c in config['logreg_C'] for w in config['class_weights']],
        'catboost': [{'depth': d, 'l2_leaf_reg': l, 'class_weight': w}
                     for d in config['catboost_depth'] for l in config['catboost_l2'] for w in config['class_weights']]}
    for name, grid in grids.items():
        best_score, best_model, candidates = -np.inf, None, []
        for params in grid:
            if name == 'logreg':
                model = make_pipeline(StandardScaler(), LogisticRegression(
                    C=params['C'], class_weight=params['class_weight'], solver='lbfgs',
                    max_iter=config['logreg_max_iter'], random_state=seed))
            elif np.all(np.ptp(train, axis=0) == 0):
                # CatBoost rejects an entirely constant matrix. No classifier can
                # rank these train representations; record a prior-only control.
                model = DummyClassifier(strategy='prior')
            else:
                model = CatBoostClassifier(
                    depth=params['depth'], l2_leaf_reg=params['l2_leaf_reg'],
                    auto_class_weights='Balanced' if params['class_weight'] == 'balanced' else None,
                    iterations=config['catboost_iterations'], learning_rate=.05,
                    loss_function='Logloss', random_seed=seed, thread_count=1,
                    verbose=False, allow_writing_files=False, task_type='CPU')
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                model.fit(train, y_train)
            metrics = binary_scores(y_validation, model.predict_proba(validation)[:, 1])
            candidates.append({'params': params, 'constant_feature_fallback': isinstance(model, DummyClassifier), **metrics})
            if metrics['ap'] > best_score:  # deterministic tie-break: first candidate
                best_score, best_model, selected = metrics['ap'], model, candidates[-1]
        fitted[name] = best_model
        results[name] = {'selected': selected, 'candidates': candidates,
                         'train': binary_scores(y_train, best_model.predict_proba(train)[:, 1]),
                         'validation': binary_scores(y_validation, best_model.predict_proba(validation)[:, 1])}
    return results, fitted
