import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_scores(y, score):
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('AP and ROC-AUC require both classes in this protocol.')
    return {'ap': float(average_precision_score(y, score)),
            'roc_auc': float(roc_auc_score(y, score)), 'prevalence': float(np.mean(y))}


def spectral(z):
    """Covariance effective rank, centered singular-value rank and diagnostics."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or len(z) < 2 or not np.isfinite(z).all():
        raise ValueError('Need a finite N x D matrix, N>=2.')
    centered = z - z.mean(0)
    s = np.linalg.svd(centered, compute_uv=False)
    eigen = s ** 2 / (len(z) - 1)
    total = float(eigen.sum())
    def er(v):
        if v.sum() == 0:
            return None
        p = v[v > 0] / v.sum()
        return float(np.exp(-(p * np.log(p)).sum()))
    return {'n': len(z), 'dimension': z.shape[1], 'collapse': total == 0,
            'effective_rank': er(eigen), 'singular_effective_rank': er(s),
            'participation_ratio': float(total ** 2 / (eigen @ eigen)) if total > 0 else None,
            'stable_rank': float(total / eigen[0]) if total > 0 else None,
            'top1_variance_share': float(eigen[0] / total) if total > 0 else None,
            'total_variance': total, 'mean_embedding_norm': float(np.linalg.norm(z, axis=1).mean()),
            'eigenvalues': eigen.tolist()}
