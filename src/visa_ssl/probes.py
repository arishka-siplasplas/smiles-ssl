"""Fit on probe_train only; select hyperparameters and threshold on validation."""
import warnings
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.exceptions import ConvergenceWarning
from .metrics import choose_threshold, binary_metrics


def evaluate(features, labels, config, seed):
    x, y = features['probe_train'], labels['probe_train']
    validation, vy = features['validation'], labels['validation']
    factories = {
        'logistic_regression': [(dict(C=c), lambda c=c: make_pipeline(StandardScaler(),
            LogisticRegression(C=c, max_iter=3000, solver='lbfgs', random_state=seed)))
            for c in config['logreg_C']],
        'lightgbm': [(params, lambda params=params: LGBMClassifier(**params,
            learning_rate=0.05, reg_lambda=1., verbosity=-1, n_jobs=1,
            deterministic=True, force_col_wise=True, random_state=seed))
            for params in config['lightgbm_grid']]
    }
    results, predictions, models = {}, {}, {}
    for name, candidates in factories.items():
        best = None
        search = []
        for params, factory in candidates:
            model = factory()
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                model.fit(x, y)
            probability = model.predict_proba(validation)[:, 1]
            ap = float(average_precision_score(vy, probability))
            search.append({'parameters': params, 'validation_average_precision': ap})
            if best is None or ap > best[0]:
                best = ap, model, params, probability
        _, model, params, probability = best
        threshold = choose_threshold(vy, probability, config['minimum_validation_recall'])
        scores = {split: model.predict_proba(z)[:, 1] for split, z in features.items()}
        results[name] = {'selected_parameters': params, 'search': search,
                         'threshold_selection': 'max precision with validation recall >= configured floor',
                         'splits': {split: binary_metrics(labels[split], score, threshold)
                                    for split, score in scores.items()}}
        predictions.update({f'{name}_{split}': score for split, score in scores.items()})
        models[name] = model
    return results, predictions, models
