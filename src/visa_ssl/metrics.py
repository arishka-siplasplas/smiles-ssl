"""Definitions copied mathematically from experiments/ssl/src/common.py."""
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_fscore_support, confusion_matrix,
                             precision_recall_curve)


def spectral(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite N x D, N >= 2')
    centered = x - x.mean(0)
    eig = np.maximum(np.linalg.eigvalsh(centered.T @ centered / len(x)), 0)
    raw = np.maximum(np.linalg.eigvalsh(x.T @ x / len(x)), 0)
    def erank(w):
        if w.sum() <= 1e-20:
            return None
        p = w[w > 0] / w.sum()
        return float(np.exp(-(p * np.log(p)).sum()))
    return {'RankMe': erank(np.sqrt(raw)), 'CER': erank(np.sqrt(eig)),
            'CovER': erank(eig), 'NESum': float(eig.sum() / eig[-1]) if eig[-1] > 1e-20 else None,
            'centered_collapse': bool(eig.sum() <= 1e-20),
            'covariance_eigenvalues': eig.tolist(), 'n': len(x), 'dimension': x.shape[1]}


def choose_threshold(y, scores, minimum_recall):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    candidates = np.flatnonzero(recall[:-1] >= minimum_recall)
    # Precision first; among ties favor recall, then lower threshold.
    index = max(candidates, key=lambda i: (precision[i], recall[i], -thresholds[i]))
    return float(thresholds[index])


def binary_metrics(y, scores, threshold):
    prediction = np.asarray(scores) >= threshold
    p, r, f, support = precision_recall_fscore_support(y, prediction, labels=[0, 1], zero_division=0)
    result = {'average_precision': float(average_precision_score(y, scores)),
              'roc_auc': float(roc_auc_score(y, scores)), 'threshold': threshold,
              'prevalence': float(np.mean(y)), 'confusion_matrix': confusion_matrix(y, prediction, labels=[0, 1]).tolist()}
    for i, name in enumerate(('normal', 'anomaly')):
        result.update({f'precision_{name}': float(p[i]), f'recall_{name}': float(r[i]),
                       f'f1_{name}': float(f[i]), f'support_{name}': int(support[i])})
    return result
