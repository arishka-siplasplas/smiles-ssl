"""Paired stratified object bootstrap for locally available VisA v2 predictions.

This conditions on the three fitted seed models and on the observed class counts.
It resamples test objects, with common weights across epochs, seeds, classifiers
and conditions within a category. It does NOT resample training seeds, refit
probes, select epochs, reselect thresholds, or estimate a joint uncertainty CI.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CELL_KEYS = ["category", "architecture", "augmentation", "method"]
SEEDS = [6101, 6102, 6103]
CLASSIFIERS = ["logistic_regression", "lightgbm"]
METRICS = ["roc_auc", "average_precision"]
EPOCHS = [30, 100]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted_metrics(labels, scores, weights):
    """Return replicate x (AUC, AP), grouping exactly equal scores together.

    Integer object multiplicities equal explicit bootstrap sampling. ROC uses
    trapezoids across tied-score groups; AP uses unsmoothed step integration,
    matching sklearn.average_precision_score, including exact score ties.
    """
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.atleast_2d(weights)
    order = np.argsort(-scores, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(scores[order]) != 0), len(scores) - 1]
    ordered = weights[:, order]
    tp = np.cumsum(ordered * labels[order], axis=1, dtype=np.float64)[:, ends]
    fp = np.cumsum(ordered * (1 - labels[order]), axis=1, dtype=np.float64)[:, ends]
    dtp = np.diff(tp, axis=1, prepend=0.0)
    dfp = np.diff(fp, axis=1, prepend=0.0)
    positive, negative = tp[:, -1], fp[:, -1]
    if np.any(positive <= 0) or np.any(negative <= 0):
        raise ValueError("Both classes must have positive resampled weight")
    auc = (dfp * (tp - 0.5 * dtp)).sum(axis=1) / (positive * negative)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    ap = (dtp * precision).sum(axis=1) / positive
    return np.column_stack([auc, ap])


def validate_metric_implementation():
    """Check ties, zero-weight score groups and unequal object multiplicities."""
    rng = np.random.default_rng(927104)
    labels = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    count = 0
    error = 0.0
    for scores in [np.zeros(8), np.arange(8),
                   np.array([0.1, 0.1, 0.1, 0.4, 0.4, 0.9, 0.9, 0.9]),
                   np.array([0.6, 0.4, 0.9, 0.2, 0.6, 0.4, 0.9, 0.2])]:
        weights = rng.multinomial(8, np.full(8, 1 / 8), size=30)
        weights = weights[(weights[:, labels == 1].sum(axis=1) > 0) &
                          (weights[:, labels == 0].sum(axis=1) > 0)]
        actual = weighted_metrics(labels, scores, weights)
        expected = np.array([[roc_auc_score(labels, scores, sample_weight=w),
                              average_precision_score(labels, scores, sample_weight=w)]
                             for w in weights])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        error = max(error, float(np.max(np.abs(actual - expected))))
        count += len(weights) * 2
    return {"weighted_tie_checks": count, "max_absolute_error": error}


def main(output: Path, replicates: int, bootstrap_seed: int):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "manifests/visa-v2/fixed.json"
    quality_path = ROOT / "reports/visa-v2-final/data/quality_test.csv"
    manifest = json.loads(manifest_path.read_text())
    quality = pd.read_csv(quality_path)
    quality = quality[quality.method.ne("baseline")]
    index_keys = CELL_KEYS + ["seed", "classifier", "epoch"]
    assert not quality.duplicated(index_keys).any()
    reference = quality.set_index(index_keys)
    candidates = defaultdict(dict)
    for architecture_slug in ("resnet18", "vit-tiny"):
        source = ROOT / f"runs/visa-ssl-v2-{architecture_slug}-block-2/visa-experiment/results"
        for identity_path in sorted(source.glob("*/identity.json")):
            identity = json.loads(identity_path.read_text())
            if identity["method"] == "baseline":
                continue
            directory = identity_path.parent
            if not all((directory / f"epoch_{epoch:03d}" / filename).exists()
                       for epoch in EPOCHS for filename in ("predictions.npz", "embeddings.npz")):
                continue
            key = tuple(identity[k] for k in CELL_KEYS)
            seed = identity["seed"]
            assert seed not in candidates[key]
            candidates[key][seed] = directory
    complete = {k: v for k, v in candidates.items() if sorted(v) == SEEDS}
    assert len(complete) == 67, f"Local input coverage changed: found {len(complete)} full triples"
    categories = sorted({k[0] for k in complete})
    labels_by_category, paths_by_category, weights_by_category = {}, {}, {}
    category_audit = {}
    for category in categories:
        records = manifest["categories"][category]["test"]
        paths = np.array([r["path"] for r in records])
        labels = np.array([r["label"] for r in records])
        assert len(set(paths)) == len(paths) and set(labels) == {0, 1}
        # Stable per-category streams: same draws even if source coverage changes.
        category_seed = int.from_bytes(hashlib.sha256(category.encode()).digest()[:4], "little")
        rng = np.random.default_rng(np.random.SeedSequence([bootstrap_seed, category_seed]))
        weights = np.zeros((replicates, len(labels)), dtype=np.int64)
        for label in (0, 1):
            indices = np.flatnonzero(labels == label)
            weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)),
                                                 size=replicates)
        labels_by_category[category], paths_by_category[category] = labels, paths
        weights_by_category[category] = weights
        category_audit[category] = {"normal": int((labels == 0).sum()),
                                    "anomaly": int((labels == 1).sum()),
                                    "paths_sha256": hashlib.sha256("\n".join(paths).encode()).hexdigest(),
                                    "weight_matrix_sha256": hashlib.sha256(weights.tobytes()).hexdigest()}

    validation = validate_metric_implementation()
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in (manifest_path, quality_path)}
    all_rows, source_metric_errors, actual_weighted_errors = [], [], []
    scored_checkpoints = 0
    checked_actual_by_classifier = set()
    for cell_number, (key, seed_directories) in enumerate(sorted(complete.items()), start=1):
        meta = dict(zip(CELL_KEYS, key))
        category = meta["category"]
        labels, paths = labels_by_category[category], paths_by_category[category]
        weights = weights_by_category[category]
        # classifier -> seed x epoch x metric and replicate x metric sums.
        point_values = {c: [] for c in CLASSIFIERS}
        boot_deltas = {c: np.zeros((replicates, len(METRICS))) for c in CLASSIFIERS}
        for seed in SEEDS:
            directory = seed_directories[seed]
            seed_points = {c: [] for c in CLASSIFIERS}
            for epoch in EPOCHS:
                checkpoint = directory / f"epoch_{epoch:03d}"
                with np.load(checkpoint / "embeddings.npz", allow_pickle=False) as emb:
                    np.testing.assert_array_equal(emb["test_labels"], labels)
                    np.testing.assert_array_equal(emb["test_paths"], paths)
                prediction_path = checkpoint / "predictions.npz"
                source_hashes[str(prediction_path.relative_to(ROOT))] = sha256(prediction_path)
                with np.load(prediction_path, allow_pickle=False) as predictions:
                    for classifier in CLASSIFIERS:
                        scores = predictions[classifier + "_test"]
                        assert scores.shape == labels.shape and np.isfinite(scores).all()
                        direct = np.array([roc_auc_score(labels, scores), average_precision_score(labels, scores)])
                        csv_values = reference.loc[key + (seed, classifier, epoch), METRICS].to_numpy(float)
                        np.testing.assert_allclose(direct, csv_values, rtol=0, atol=1e-12)
                        source_metric_errors.extend(np.abs(direct - csv_values).tolist())
                        weighted_point = weighted_metrics(labels, scores, np.ones((1, len(labels))))[0]
                        np.testing.assert_allclose(weighted_point, direct, rtol=0, atol=1e-12)
                        seed_points[classifier].append(direct)
                        boot = weighted_metrics(labels, scores, weights)
                        if classifier not in checked_actual_by_classifier:
                            expected = np.array([[roc_auc_score(labels, scores, sample_weight=w),
                                                  average_precision_score(labels, scores, sample_weight=w)]
                                                 for w in weights[:10]])
                            np.testing.assert_allclose(boot[:10], expected, rtol=0, atol=1e-12)
                            actual_weighted_errors.extend(np.abs(boot[:10] - expected).ravel().tolist())
                            checked_actual_by_classifier.add(classifier)
                        boot_deltas[classifier] += (-1 if epoch == EPOCHS[0] else 1) * boot / len(SEEDS)
                scored_checkpoints += 1
            for classifier in CLASSIFIERS:
                point_values[classifier].append(seed_points[classifier])
        for classifier in CLASSIFIERS:
            points = np.asarray(point_values[classifier]).mean(axis=0)
            for metric_index, metric in enumerate(METRICS):
                delta_pp = 100 * boot_deltas[classifier][:, metric_index]
                lower, upper = np.quantile(delta_pp, [0.025, 0.975], method="linear")
                row = {**meta, "classifier": classifier, "metric": metric,
                       "early_epoch": EPOCHS[0], "late_epoch": EPOCHS[1],
                       "n_fitted_seed_models": len(SEEDS), "n_test_objects": len(labels),
                       "n_test_anomalies": int((labels == 1).sum()),
                       "n_test_normals": int((labels == 0).sum()), "bootstrap_replicates": replicates,
                       "early_mean": points[0, metric_index], "late_mean": points[1, metric_index],
                       "mean_delta_pp": 100 * (points[1, metric_index] - points[0, metric_index]),
                       "object_ci95_low_pp": lower, "object_ci95_high_pp": upper,
                       "object_bootstrap_sd_pp": float(delta_pp.std(ddof=1)),
                       "object_bootstrap_mean_pp": float(delta_pp.mean()),
                       "object_ci95_excludes_zero": bool(lower > 0 or upper < 0),
                       "object_ci95_within_1pp": bool(lower > -1 and upper < 1),
                       "object_ci95_within_2pp": bool(lower > -2 and upper < 2),
                       "object_ci95_above_2pp": bool(lower > 2),
                       "object_ci95_below_minus2pp": bool(upper < -2),
                       "uncertainty_scope": "test_objects_conditional_on_3_fixed_seed_models_and_class_counts"}
                all_rows.append(row)
        if cell_number % 10 == 0 or cell_number == len(complete):
            print(f"Completed {cell_number}/{len(complete)} cells ({time.perf_counter()-started:.1f}s)", flush=True)
    result = pd.DataFrame(all_rows)
    assert len(result) == 67 * 2 * 2
    assert not result.duplicated(CELL_KEYS + ["classifier", "metric"]).any()
    result.to_csv(output / "object_bootstrap.csv", index=False)
    expected_cells = set(tuple(row) for row in quality[CELL_KEYS].drop_duplicates().to_numpy())
    missing = sorted(expected_cells - set(complete))
    audit = {
        "analysis": "paired stratified percentile bootstrap of test images",
        "contrast": "100_minus_30", "replicates": replicates, "bootstrap_seed": bootstrap_seed,
        "input_roots": [f"runs/visa-ssl-v2-{a}-block-2/visa-experiment/results" for a in ("resnet18", "vit-tiny")],
        "complete_training_cells_available": len(complete), "total_training_cells": len(expected_cells),
        "training_seeds_per_cell": SEEDS, "training_runs_included": len(complete) * len(SEEDS),
        "scored_checkpoints": scored_checkpoints, "output_rows": len(result),
        "coverage_by_augmentation": result[CELL_KEYS].drop_duplicates().augmentation.value_counts().to_dict(),
        "cells_unavailable": [dict(zip(CELL_KEYS, k)) for k in missing],
        "partial_local_cells_excluded": [{**dict(zip(CELL_KEYS, k)), "available_seeds": sorted(v)}
                                         for k, v in sorted(candidates.items()) if k not in complete],
        "categories": category_audit,
        "checks": {**validation, "scores_vs_final_csv_checks": len(source_metric_errors),
                   "scores_vs_final_csv_max_absolute_error": max(source_metric_errors),
                   "weighted_real_prediction_checks": len(actual_weighted_errors),
                   "weighted_real_prediction_max_absolute_error": max(actual_weighted_errors),
                   "test_paths_and_labels_match_manifest_every_checkpoint": True,
                   "common_weights_across_epochs_seeds_classifiers_and_conditions": True},
        "scope": "Mean of three per-model metric differences, conditional on these fitted models and observed class counts.",
        "limitations": [
            "No seed resampling: this interval is not a joint training-and-test uncertainty interval.",
            "No model/probe refitting or hyperparameter/threshold reselection; validation/training sample uncertainty is excluded.",
            "Image-level resampling assumes independent test images within class; unknown subject/batch dependence is not modelled.",
            "Stratification holds prevalence fixed; AP uncertainty under changing prevalence is excluded.",
            "Marginal percentile intervals are exploratory and not multiplicity-corrected; excluding zero is not a familywise significance claim.",
            "Coverage is nonrandom: all 36 A0 cells and 31 Rotate cells; CutPaste is unavailable locally.",
            "Bootstrap percentiles may have finite-sample coverage error, particularly with only 40 distinct anomalies per category.",
            "Resampling existing images does not create new independent observations."],
        "source_sha256": source_hashes, "script_sha256": sha256(Path(__file__)),
        "runtime_seconds": time.perf_counter() - started,
        "summary": result.groupby(["classifier", "metric"])[[
            "object_ci95_excludes_zero", "object_ci95_within_1pp", "object_ci95_within_2pp",
            "object_ci95_above_2pp", "object_ci95_below_minus2pp"]].sum().reset_index().to_dict("records"),
    }
    (output / "object_bootstrap_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: audit[k] for k in ("complete_training_cells_available", "output_rows", "runtime_seconds", "summary")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/visa-v2-statistics-2026-11-14")
    parser.add_argument("--replicates", type=int, default=3000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261114)
    args = parser.parse_args()
    if args.replicates < 100:
        parser.error("Use at least 100 replicates")
    main(args.output, args.replicates, args.bootstrap_seed)
