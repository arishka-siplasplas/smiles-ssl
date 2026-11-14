"""Augmentation-stratified duration and paired late interaction statistics.

Interaction = (100-10)_augmentation - (100-10)_A0 within category and seed.
Add 720 distinct interaction tests to the previous 19080-test Holm family.
Early 0-10 interactions duplicate the existing epoch10 augmentation contrasts
and are validated but not counted twice. All input reports remain unchanged.
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
DIRECTORY = ROOT / "reports/visa-v2-statistics-2026-11-14/tables"
SOURCE = DIRECTORY / "all_paired_statistics.csv"
OUTPUT = DIRECTORY / "augmentation_interaction_summary.json"
OLD_PAYLOAD = DIRECTORY / "data_table_payload.json"
NEW_PAYLOAD = DIRECTORY / "data_table_payload_with_interactions.json"
DELTAS = ["delta6101", "delta6102", "delta6103"]
SEEDS = [6101, 6102, 6103]
METRICS = ["roc_auc", "average_precision", "precision_anomaly", "recall_anomaly", "f1_anomaly"]
GROUP = ["architecture", "method", "augmentation", "classifier", "metric"]
MATCH = ["category", "architecture", "method", "classifier", "metric"]
EPS = 1e-10


def describe(frame, inferential=False):
    values = frame[DELTAS].to_numpy(dtype=float)
    frame = frame.copy()
    frame["mean"] = values.mean(axis=1)
    frame["sd"] = values.std(axis=1, ddof=1)
    se = frame.sd / np.sqrt(3)
    valid = se > EPS
    half = stats.t.ppf(.975, 2) * se
    frame["ci_low"] = np.where(valid, frame["mean"] - half, np.nan)
    frame["ci_high"] = np.where(valid, frame["mean"] + half, np.nan)
    frame["positive"] = (values > EPS).sum(axis=1)
    frame["negative"] = (values < -EPS).sum(axis=1)
    frame["tied"] = (np.abs(values) <= EPS).sum(axis=1)
    if inferential:
        with np.errstate(divide="ignore", invalid="ignore"):
            frame["p_t"] = np.where(valid, 2 * stats.t.sf(np.abs(frame["mean"] / se), 2), np.nan)
        flips = np.array(list(itertools.product((-1, 1), repeat=3)))
        frame["p_signflip"] = (np.abs(values @ flips.T / 3) >=
                               np.abs(values.mean(axis=1))[:, None] - 1e-12).mean(axis=1)
        check = stats.ttest_1samp(values[valid], 0, axis=1)
        np.testing.assert_allclose(frame.loc[valid, "p_t"], check.pvalue, rtol=0, atol=1e-13)
        np.testing.assert_allclose(frame.loc[valid, "ci_low"], check.confidence_interval().low,
                                   rtol=0, atol=1e-11)
        np.testing.assert_allclose(frame.loc[valid, "ci_high"], check.confidence_interval().high,
                                   rtol=0, atol=1e-11)
    return frame


def fixed_grid_summary(cells, kind):
    grouping = GROUP + ["from_epoch", "to_epoch"]
    means = cells.groupby(grouping)[DELTAS].mean().reset_index()
    means = describe(means)
    info = cells.groupby(grouping).agg(
        n_conditions=("mean", "size"), median_cell_sd=("sd", "median"),
        mean_positive=("mean", lambda s: int(s.gt(EPS).sum())),
        mean_negative=("mean", lambda s: int(s.lt(-EPS).sum())),
        unanimous_positive=("positive", lambda s: int(s.eq(3).sum())),
        unanimous_negative=("negative", lambda s: int(s.eq(3).sum()))).reset_index()
    out = means.merge(info, on=grouping, validate="one_to_one")
    assert out.n_conditions.eq(6).all()
    out["kind"], out["category"] = kind, "all"
    return out


def summarize(frame, labels):
    frame = frame.sort_values("category")
    assert len(frame) == 6 and frame.category.nunique() == 6
    values = frame[DELTAS].to_numpy(dtype=float)
    assert values.shape == (6, 3) and np.isfinite(values).all()
    seed_means = values.mean(axis=0)
    means = values.mean(axis=1)
    sd = values.std(axis=1, ddof=1)
    positive = (values > EPS).sum(axis=1)
    negative = (values < -EPS).sum(axis=1)
    categories = []
    for i, category in enumerate(frame.category):
        categories.append(dict(category=category, mean=float(means[i]), sd=float(sd[i]),
                               positive=int(positive[i]), negative=int(negative[i]),
                               **{k: float(v) for k, v in zip(DELTAS, values[i])}))
    return dict(labels, mean=float(seed_means.mean()), macro_sd=float(seed_means.std(ddof=1)),
                **{k: float(v) for k, v in zip(DELTAS, seed_means)},
                seed_positive=int((seed_means > EPS).sum()),
                seed_negative=int((seed_means < -EPS).sum()),
                n_categories=6, category_positive=int((means > EPS).sum()),
                category_negative=int((means < -EPS).sum()),
                category_unanimous_positive=int((positive == 3).sum()),
                category_unanimous_negative=int((negative == 3).sum()),
                median_category_sd=float(np.median(sd)), categories=categories)


def main():
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    payload_hash = hashlib.sha256(OLD_PAYLOAD.read_bytes()).hexdigest()
    data = pd.read_csv(SOURCE)
    assert len(data) == 19080
    assert set(data.metric) == set(METRICS)
    old_payload = json.loads(OLD_PAYLOAD.read_text())
    assert old_payload["meta"]["family_tests"] == 19080
    duration, interactions = [], []
    stratified_parts = []
    for before, after in [(0, 10), (10, 100)]:
        selected = data[data.kind.eq("duration") & data.from_epoch.eq(before) & data.to_epoch.eq(after)]
        assert len(selected) == 1080
        stratified_parts.append(selected)
        for key, group in selected.groupby(GROUP):
            duration.append(summarize(group, dict(zip(GROUP, key)) |
                                      dict(kind="duration", from_epoch=before, to_epoch=after)))
    late = data[data.kind.eq("duration") & data.from_epoch.eq(10) & data.to_epoch.eq(100)]
    control = late[late.augmentation.eq("baseline")].set_index(MATCH)[DELTAS].sort_index()
    assert len(control) == 360 and not control.index.duplicated().any()
    identity_error = 0.0
    identity_checks = 0
    early_identity_error = 0.0
    early_identity_checks = 0
    interaction_parts = []
    for augmentation in ["rotate", "cutpaste"]:
        active = late[late.augmentation.eq(augmentation)].set_index(MATCH)[DELTAS].reindex(control.index)
        assert active.notna().all().all()
        difference = active - control
        # Equivalent identity: same-epoch augmentation contrast at100 minus at10.
        at = data[data.kind.eq("augmentation") & data.augmentation.eq(augmentation)]
        start = at[at.from_epoch.eq(10) & at.to_epoch.eq(10)].set_index(MATCH)[DELTAS].reindex(control.index)
        end = at[at.from_epoch.eq(100) & at.to_epoch.eq(100)].set_index(MATCH)[DELTAS].reindex(control.index)
        equivalent = end - start
        np.testing.assert_allclose(difference, equivalent, rtol=0, atol=1e-11)
        identity_error = max(identity_error, float(np.max(np.abs(difference.to_numpy() - equivalent.to_numpy()))))
        identity_checks += difference.size
        # The shared epoch0 baseline makes the early interaction exactly equal
        # to an already-tested epoch10 augmentation contrast: do not duplicate.
        early = data[data.kind.eq("duration") & data.from_epoch.eq(0) & data.to_epoch.eq(10)]
        early_control = early[early.augmentation.eq("baseline")].set_index(MATCH)[DELTAS].reindex(control.index)
        early_active = early[early.augmentation.eq(augmentation)].set_index(MATCH)[DELTAS].reindex(control.index)
        early_interaction = early_active - early_control
        np.testing.assert_allclose(early_interaction, start, rtol=0, atol=1e-11)
        early_identity_error = max(early_identity_error, float(np.max(np.abs(early_interaction.to_numpy() - start.to_numpy()))))
        early_identity_checks += early_interaction.size
        frame = difference.reset_index()
        frame["augmentation"] = augmentation
        frame["kind"], frame["from_epoch"], frame["to_epoch"] = "interaction", 10, 100
        interaction_parts.append(frame)
        for key, group in frame.groupby(GROUP):
            interactions.append(summarize(group, dict(zip(GROUP, key)) |
                                          dict(kind="late_augmentation_interaction", from_epoch=10, to_epoch=100)))
    assert len(duration) == 360 and len(interactions) == 120
    interaction_cells = describe(pd.concat(interaction_parts, ignore_index=True), inferential=True)
    assert len(interaction_cells) == 720
    id_columns = ["kind"] + MATCH + ["augmentation", "from_epoch", "to_epoch"]
    assert not interaction_cells.duplicated(id_columns).any()
    all_cells = pd.concat([data, interaction_cells], ignore_index=True)
    assert len(all_cells) == 19800 and not all_cells.duplicated(id_columns).any()
    all_cells["p_holm"] = holm(all_cells.p_t)
    np.testing.assert_allclose(all_cells.p_holm,
                               multipletests(all_cells.p_t.fillna(1), method="holm")[1],
                               rtol=0, atol=1e-14)
    # Adding a family cannot decrease an existing adjusted p-value.
    assert (all_cells.iloc[:len(data)].p_holm.to_numpy() + 1e-14 >= data.p_holm.to_numpy()).all()
    unchanged_columns = [c for c in old_payload["columns"] if c != "p_holm"]
    pd.testing.assert_frame_equal(all_cells.iloc[:len(data)][unchanged_columns].reset_index(drop=True),
                                  data[unchanged_columns].reset_index(drop=True), check_dtype=False)
    interaction_cells = all_cells[all_cells.kind.eq("interaction")].copy()
    stratified_summary = fixed_grid_summary(pd.concat(stratified_parts), "duration_by_augmentation")
    interaction_summary = fixed_grid_summary(interaction_cells, "interaction")
    assert len(stratified_summary) == 360 and len(interaction_summary) == 120
    columns, summary_columns = old_payload["columns"], old_payload["summary_columns"]
    interaction_cells[columns].to_csv(DIRECTORY / "interaction_cells.csv", index=False)
    stratified_summary[summary_columns].to_csv(DIRECTORY / "duration_by_augmentation_summary.csv", index=False)
    interaction_summary[summary_columns].to_csv(DIRECTORY / "interaction_fixed_grid_summary.csv", index=False)
    records = lambda frame: json.loads(frame.to_json(orient="values", double_precision=10, force_ascii=False))
    output_rows = records(all_cells[columns])
    # CSV round-tripping can move a value across a 10-decimal JSON rounding
    # boundary. Preserve already-rendered old numbers exactly, replacing only
    # their Holm adjustment; full-precision calculations were checked above.
    keep = [i for i, c in enumerate(columns) if c != "p_holm"]
    adjustment_index = columns.index("p_holm")
    for index, original in enumerate(old_payload["rows"]):
        for j in keep:
            a, b = output_rows[index][j], original[j]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                assert abs(a - b) <= 2e-10
            else:
                assert a == b
        adjustment = output_rows[index][adjustment_index]
        output_rows[index] = list(original)
        output_rows[index][adjustment_index] = adjustment
    assert [[r[i] for i in keep] for r in output_rows[:19080]] == [[r[i] for i in keep] for r in old_payload["rows"]]
    new_payload = dict(old_payload)
    new_payload["rows"] = output_rows
    new_payload["summary_rows"] = old_payload["summary_rows"] + records(stratified_summary[summary_columns]) + records(interaction_summary[summary_columns])
    assert len(new_payload["summary_rows"]) == 1540
    new_payload["meta"] = dict(old_payload["meta"], family_tests=19800,
        interaction_phase=[10, 100], interaction_tests=720,
        interaction_definition="(100−10)augmentation − (100−10)A0, paired within category and seed.",
        nominal_significant=int(all_cells.p_t.lt(.05).sum()),
        holm_significant=int(all_cells.p_holm.lt(.05).sum()),
        degenerate_rows=int(all_cells.p_t.isna().sum()),
        source_previous_family=19080, source_previous_payload=str(OLD_PAYLOAD.relative_to(ROOT)),
        added_summary_rows={"duration_by_augmentation": 360, "interaction": 120})
    NEW_PAYLOAD.write_text(json.dumps(new_payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    # Recombine the three augmentation-specific six-category means to recover
    # each existing 18-condition architecture-method duration overview exactly.
    reconciled = 0
    reconciliation_error = 0.0
    for before, after in [(0, 10), (10, 100)]:
        flat = data[data.kind.eq("duration") & data.from_epoch.eq(before) & data.to_epoch.eq(after)]
        for key, group in flat.groupby(["architecture", "method", "classifier", "metric"]):
            expected = group[DELTAS].mean().to_numpy(dtype=float)
            selected = [r for r in duration if r["from_epoch"] == before and r["to_epoch"] == after and
                        tuple(r[k] for k in ["architecture", "method", "classifier", "metric"]) == key]
            assert len(selected) == 3
            actual = np.array([[r[k] for k in DELTAS] for r in selected]).mean(axis=0)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-11)
            reconciliation_error = max(reconciliation_error, float(np.max(np.abs(actual - expected))))
            reconciled += 1
    meta = dict(title="VisA v2 · длительность и аугментации", generated="2026-11-14", unit="п.п.",
                source=str(SOURCE.relative_to(ROOT)), source_sha256=source_hash,
                seeds=SEEDS, inferential_seeds=3, fixed_categories=6,
                duration_rows=len(duration), interaction_rows=len(interactions),
                family_tests=19800, previous_family_tests=19080, new_hypothesis_tests=720,
                interaction_phase=[10, 100], metrics=METRICS,
                interaction_definition="(100−10)augmentation − (100−10)A0, matched within category and seed.",
                positive_interaction="The augmentation makes the late change more positive than A0; it does not by itself imply a positive late change or higher final quality.",
                macro_sd_definition="Sample SD of three seed-specific means across the six fixed categories.",
                augmentation_definitions={"baseline": "A0: random resized crop + color jitter, already active augmentations.",
                                          "rotate": "A0 plus rotation by 0/90/180/270 degrees.",
                                          "cutpaste": "A0 plus copying a patch within the image."},
                limitations=["Cell-level paired t intervals/p-values assume normally distributed seed differences, uncheckable with only three seeds.",
                             "Macro summaries do not create additional tests or independent observations.",
                             "95% t intervals are marginal; Holm p-values cover the 19800 distinct cell-level hypotheses.",
                             "Results condition on the fixed train/validation/test objects.",
                             "Category means can hide different signs across seeds.",
                             "There is no no-augmentation control, so an A0 decline does not rule out a role of augmentation.",
                             "There is no direct test here of whether an augmentation preserves the anomaly cue.",
                             "An interaction in downstream metrics does not identify a defect-erasure mechanism."],
                checks={"paired_difference_in_differences_identity_checks": identity_checks,
                        "max_identity_absolute_error": identity_error,
                        "early_interaction_duplicate_identity_checks": early_identity_checks,
                        "max_early_duplicate_absolute_error": early_identity_error,
                        "new_interaction_cells": len(interaction_cells),
                        "new_cells_with_valid_scipy_t_and_ci_checks": int(interaction_cells.p_t.notna().sum()),
                        "new_degenerate_cells": int(interaction_cells.p_t.isna().sum()),
                        "statsmodels_holm_19800_match": True,
                        "old_statistics_unchanged_except_adjustment": True,
                        "old_payload_values_unchanged_except_adjustment": True,
                        "old_summary_rows_preserved": len(old_payload["summary_rows"]),
                        "new_summary_rows": len(new_payload["summary_rows"]),
                        "new_interaction_nominal_significant": int(interaction_cells.p_t.lt(.05).sum()),
                        "new_interaction_holm_significant": int(interaction_cells.p_holm.lt(.05).sum()),
                        "new_interaction_min_signflip_p": float(interaction_cells.p_signflip.min()),
                        "architecture_method_overviews_reconciled": reconciled,
                        "max_reconciliation_absolute_error": reconciliation_error,
                        "source_unchanged": hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash,
                        "old_payload_unchanged": hashlib.sha256(OLD_PAYLOAD.read_bytes()).hexdigest() == payload_hash})
    assert meta["checks"]["source_unchanged"]
    assert meta["checks"]["old_payload_unchanged"]
    OUTPUT.write_text(json.dumps(dict(meta=meta, duration=duration, interactions=interactions),
                                 ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (DIRECTORY / "interaction_audit.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    display = pd.DataFrame(duration)
    print(display[display.from_epoch.eq(10) & display.classifier.eq("logistic_regression")][
        GROUP + ["mean", "macro_sd", "seed_negative", "category_negative"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
