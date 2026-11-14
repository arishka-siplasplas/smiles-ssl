"""Descriptive VisA v2 overviews from full-precision paired statistics.

Average the fixed condition grid within each of the three seeds first, then
summarize those three means. No hypothesis tests or new multiplicity family.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "reports/visa-v2-statistics-2026-11-14/tables"
SOURCE = DIRECTORY / "all_paired_statistics.csv"
PREVIOUS = DIRECTORY / "fixed_grid_summaries.csv"
OUTPUT = DIRECTORY / "overview_payload.json"
SEEDS = [6101, 6102, 6103]
DELTAS = [f"delta{seed}" for seed in SEEDS]
METRICS = ["roc_auc", "average_precision", "precision_anomaly", "recall_anomaly", "f1_anomaly"]
CLASSIFIERS = ["logistic_regression", "lightgbm"]
IDENTITY = ["kind", "category", "architecture", "method", "augmentation",
            "from_epoch", "to_epoch", "classifier", "metric"]
STAGES = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 100)]
METHOD_CONTRASTS = [(0, 10), (10, 100), (0, 100)]
AUGMENTATION_EPOCHS = [10, 100]
EPS = 1e-10


def aggregate(frame, labels, expected_conditions):
    """Return one row; n_conditions is a fixed grid size, never inferential n."""
    assert len(frame) == expected_conditions
    assert not frame.duplicated(["category", "architecture", "method", "augmentation"]).any()
    values = frame[DELTAS].to_numpy(dtype=float)
    assert values.shape == (expected_conditions, 3) and np.isfinite(values).all()
    per_seed = values.mean(axis=0)
    mean = float(per_seed.mean())
    sd = float(per_seed.std(ddof=1))
    se = sd / np.sqrt(3)
    half = float(t.ppf(.975, 2) * se)
    # Match source policy: identical observed values do not imply certainty.
    valid = se > EPS
    positive = int(frame.positive.eq(3).sum())
    negative = int(frame.negative.eq(3).sum())
    row = dict(labels, mean=mean, macro_sd=sd,
               ci_low=mean - half if valid else None,
               ci_high=mean + half if valid else None,
               median_cell_sd=float(frame.sd.median()),
               n_conditions=expected_conditions, unanimous=positive + negative,
               mean_positive=int(frame["mean"].gt(EPS).sum()),
               mean_negative=int(frame["mean"].lt(-EPS).sum()),
               unanimous_positive=positive, unanimous_negative=negative,
               **{column: float(value) for column, value in zip(DELTAS, per_seed)})
    # Independent equivalent expressions guard against unequal weights or wrong
    # axes. Source cell means/SD/sign counts are also reconciled in main().
    np.testing.assert_allclose(mean, values.mean(), rtol=0, atol=1e-11)
    np.testing.assert_allclose(mean, frame["mean"].mean(), rtol=0, atol=1e-11)
    np.testing.assert_allclose(per_seed, frame[DELTAS].sum().to_numpy() / len(frame),
                               rtol=0, atol=1e-11)
    return row


def reconcile_previous(payload, previous):
    max_error = 0.0
    checked = 0
    cols = DELTAS + ["mean", "sd", "ci_low", "ci_high", "median_cell_sd"]
    for section in ("method", "augmentation"):
        for row in payload[section]:
            selected = previous
            for column in ("kind", "architecture", "method", "augmentation",
                           "from_epoch", "to_epoch", "classifier", "metric"):
                selected = selected[selected[column].eq(row[column])]
            assert len(selected) == 1
            expected = selected.iloc[0]
            current = np.array([row["macro_sd" if column == "sd" else column]
                                for column in cols], dtype=float)
            old = expected[cols].to_numpy(dtype=float)
            np.testing.assert_allclose(current, old, rtol=1e-12, atol=1e-11, equal_nan=True)
            max_error = max(max_error, float(np.nanmax(np.abs(current - old))))
            for column in ("n_conditions", "mean_positive", "mean_negative",
                           "unanimous_positive", "unanimous_negative"):
                assert row[column] == expected[column]
            checked += 1
    # Overall stages must equal the six equally weighted architecture-method
    # grid means; their CI still uses only three resulting seed means.
    for row in payload["stage"]:
        selected = previous[previous.kind.eq("duration") &
                            previous.from_epoch.eq(row["from_epoch"]) &
                            previous.to_epoch.eq(row["to_epoch"]) &
                            previous.classifier.eq(row["classifier"]) &
                            previous.metric.eq(row["metric"])]
        assert len(selected) == 6 and selected.n_conditions.eq(18).all()
        expected_seeds = selected[DELTAS].mean().to_numpy(dtype=float)
        current_seeds = np.array([row[column] for column in DELTAS])
        np.testing.assert_allclose(current_seeds, expected_seeds, rtol=0, atol=1e-11)
        np.testing.assert_allclose(row["macro_sd"], expected_seeds.std(ddof=1),
                                   rtol=0, atol=1e-11)
        max_error = max(max_error, float(np.max(np.abs(current_seeds - expected_seeds))))
        checked += 1
    return {"overview_rows_reconciled": checked,
            "max_absolute_error_vs_existing_fixed_grid_summaries": max_error}


def main():
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    previous_hash = hashlib.sha256(PREVIOUS.read_bytes()).hexdigest()
    cells = pd.read_csv(SOURCE)
    assert len(cells) == 19080 and not cells.duplicated(IDENTITY).any()
    assert set(cells.metric) == set(METRICS) and set(cells.classifier) == set(CLASSIFIERS)
    assert cells.groupby("kind").size().to_dict() == {"augmentation": 2880, "duration": 16200}
    source_values = cells[DELTAS].to_numpy(dtype=float)
    assert np.isfinite(source_values).all()
    np.testing.assert_allclose(cells["mean"], source_values.mean(axis=1), rtol=0, atol=1e-11)
    np.testing.assert_allclose(cells.sd, source_values.std(axis=1, ddof=1), rtol=0, atol=1e-11)
    np.testing.assert_array_equal(cells.positive, (source_values > EPS).sum(axis=1))
    np.testing.assert_array_equal(cells.negative, (source_values < -EPS).sum(axis=1))
    payload = {"stage": [], "method": [], "augmentation": []}
    duration = cells[cells.kind.eq("duration")]
    augmentation = cells[cells.kind.eq("augmentation")]
    for before, after in STAGES:
        part = duration[duration.from_epoch.eq(before) & duration.to_epoch.eq(after)]
        for (classifier, metric), frame in part.groupby(["classifier", "metric"], sort=True):
            labels = dict(kind="duration", architecture="all", method="all", augmentation="all",
                          from_epoch=before, to_epoch=after, classifier=classifier, metric=metric)
            payload["stage"].append(aggregate(frame, labels, 108))
    for before, after in METHOD_CONTRASTS:
        part = duration[duration.from_epoch.eq(before) & duration.to_epoch.eq(after)]
        for values, frame in part.groupby(["architecture", "method", "classifier", "metric"], sort=True):
            architecture, method, classifier, metric = values
            labels = dict(kind="duration", architecture=architecture, method=method, augmentation="all",
                          from_epoch=before, to_epoch=after, classifier=classifier, metric=metric)
            payload["method"].append(aggregate(frame, labels, 18))
    for epoch in AUGMENTATION_EPOCHS:
        part = augmentation[augmentation.from_epoch.eq(epoch) & augmentation.to_epoch.eq(epoch)]
        for values, frame in part.groupby(["architecture", "augmentation", "classifier", "metric"], sort=True):
            architecture, variant, classifier, metric = values
            labels = dict(kind="augmentation", architecture=architecture, method="all", augmentation=variant,
                          from_epoch=epoch, to_epoch=epoch, classifier=classifier, metric=metric)
            payload["augmentation"].append(aggregate(frame, labels, 18))
    assert {k: len(v) for k, v in payload.items()} == {"stage": 50, "method": 180, "augmentation": 80}
    previous = pd.read_csv(PREVIOUS)
    reconciliation = reconcile_previous(payload, previous)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
    assert hashlib.sha256(PREVIOUS.read_bytes()).hexdigest() == previous_hash
    meta = dict(title="VisA v2 · обобщённые результаты", generated="2026-11-14", unit="п.п.",
                source=str(SOURCE.relative_to(ROOT)), source_sha256=source_hash,
                reconciliation_source=str(PREVIOUS.relative_to(ROOT)),
                reconciliation_source_sha256=previous_hash, source_rows=len(cells),
                family_tests=19080, new_hypothesis_tests=0, inferential_seeds=3, seeds=SEEDS,
                stage_contrasts=STAGES, method_contrasts=METHOD_CONTRASTS,
                augmentation_epochs=AUGMENTATION_EPOCHS,
                section_rows={k: len(v) for k, v in payload.items()},
                uncertainty="95% marginal t interval across three seed-specific fixed-grid means; conditional on the fixed data split.",
                macro_sd_definition="Sample SD of three seed-specific fixed-grid means, not the SD across conditions.",
                median_cell_sd_definition="Median of per-condition paired-difference sample SDs across the included fixed grid.",
                unanimous_definition="Number of conditions where all three seed differences have the same strictly positive or strictly negative sign.",
                condition_count_definition="Fixed grid size, not the number of independent repetitions.",
                limitations=["No test-object, validation-selection or dataset-sampling uncertainty is included.",
                             "Intervals are marginal and not adjusted for multiple comparisons.",
                             "Three seeds do not permit a meaningful check of normality of seed differences.",
                             "Stage intervals span unequal numbers of epochs; changes are not rates per epoch.",
                             "Small macro means can hide opposing condition-specific changes.",
                             "A0 already includes crop and color jitter; augmentation rows compare added Rotate/CutPaste against A0 at the same epoch."],
                checks=dict(source_cell_means_sd_and_sign_counts_reconciled=True,
                            source_files_unchanged=True, **reconciliation))
    result = dict(meta=meta, **payload)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    rows = pd.DataFrame(payload["stage"])
    print(rows[rows.classifier.eq("logistic_regression") & rows.metric.isin(["roc_auc", "average_precision"])][
        ["from_epoch", "to_epoch", "metric", "mean", "macro_sd", "ci_low", "ci_high", "unanimous", "n_conditions"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
