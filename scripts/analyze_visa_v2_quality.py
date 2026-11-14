"""Reproduce descriptive paired downstream analyses for the completed VisA v2.

No test-based epoch or hyperparameter selection is performed. All contrasts use
the prespecified final epoch 100 and fixed earlier checkpoints. Differences are
percentage points; macro averages give each experimental condition equal weight.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["roc_auc", "average_precision", "precision_anomaly", "recall_anomaly", "f1_anomaly"]
CELLS = ["category", "architecture", "augmentation", "method", "classifier"]
EPOCHS = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100]
SEEDS = [6101, 6102, 6103]
EPS = 1e-10


def sign_counts(values):
    a = np.asarray(values)
    return dict(positive=int((a > EPS).sum()), negative=int((a < -EPS).sum()), tied=int((np.abs(a) <= EPS).sum()))


def main(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    q = pd.read_csv(source / "quality.csv")
    ssl = q[(q.method != "baseline") & (q.split == "test")].copy()
    assert len(ssl) == 6480
    assert not ssl.duplicated(CELLS + ["seed", "epoch"]).any()
    assert ssl.groupby(CELLS).size().eq(30).all()
    assert np.isfinite(ssl[METRICS]).all().all()
    for _, group in ssl.groupby(CELLS):
        assert set(zip(group.seed, group.epoch)) == set(itertools.product(SEEDS, EPOCHS))

    trajectories = ssl.groupby(CELLS + ["epoch"])[METRICS].agg(["mean", "std", "count"])
    trajectories.columns = ["_".join(x) for x in trajectories.columns]
    trajectories.reset_index().to_csv(output / "quality_cell_trajectories.csv", index=False)
    final = ssl[ssl.epoch == 100]
    baseline = q[(q.method == "baseline") & (q.split == "test")]
    baseline.groupby(["architecture", "category", "classifier"])[METRICS].agg(["mean", "std"]).to_csv(output / "quality_baseline_summary.csv")

    pairs = []
    late = final.set_index(CELLS + ["seed"])
    for early in [0, 5, 10, 30]:
        before = ssl[ssl.epoch == early].set_index(CELLS + ["seed"])
        assert before.index.equals(late.index)
        for metric in METRICS:
            d = late[metric] - before[metric]
            t = d.rename("delta_pp").reset_index()
            t["delta_pp"] *= 100
            t["early_value"] = before[metric].to_numpy()
            t["late_value"] = late[metric].to_numpy()
            t["early_epoch"], t["late_epoch"], t["metric"] = early, 100, metric
            pairs.append(t)
    paired = pd.concat(pairs, ignore_index=True)
    paired.to_csv(output / "quality_paired_seed_deltas.csv", index=False)
    rows = []
    for key, group in paired.groupby(CELLS + ["early_epoch", "metric"]):
        row = dict(zip(CELLS + ["early_epoch", "metric"], key))
        row.update(late_epoch=100, n=3, mean_delta_pp=group.delta_pp.mean(), sd_delta_pp=group.delta_pp.std(),
                   early_mean=group.early_value.mean(), final_mean=group.late_value.mean(),
                   final_sd=group.late_value.std(), **sign_counts(group.delta_pp))
        rows.append(row)
    cells = pd.DataFrame(rows)
    cells.to_csv(output / "quality_paired_cell_summary.csv", index=False)

    macro_rows = []
    views = [["classifier"], ["architecture", "classifier"], ["architecture", "method", "classifier"],
             ["architecture", "augmentation", "classifier"], ["architecture", "category", "classifier"],
             ["architecture", "method", "augmentation", "classifier"]]
    for dimensions in views:
        for key, group in cells.groupby(dimensions + ["early_epoch", "metric"]):
            row = dict(zip(dimensions + ["early_epoch", "metric"], key))
            row.update(view="/".join(dimensions), n_cells=len(group), macro_delta_pp=group.mean_delta_pp.mean(),
                       macro_initial=group.early_mean.mean(), macro_final=group.final_mean.mean(),
                       all_three_positive=int(group.positive.eq(3).sum()), all_three_negative=int(group.negative.eq(3).sum()),
                       **sign_counts(group.mean_delta_pp))
            macro_rows.append(row)
    macro = pd.DataFrame(macro_rows)
    macro.to_csv(output / "quality_macro_summary.csv", index=False)

    # Within-condition augmentation contrasts at the fixed final checkpoint.
    aug_keys = [x for x in CELLS if x != "augmentation"] + ["seed"]
    aug_rows = []
    base = final[final.augmentation == "baseline"].set_index(aug_keys)
    for augmentation in ["rotate", "cutpaste"]:
        other = final[final.augmentation == augmentation].set_index(aug_keys).reindex(base.index)
        for metric in METRICS:
            d = (100 * (other[metric] - base[metric])).rename("delta_pp").reset_index()
            d["augmentation"], d["metric"] = augmentation, metric
            aug_rows.append(d)
    aug = pd.concat(aug_rows, ignore_index=True)
    aug.to_csv(output / "quality_augmentation_seed_deltas.csv", index=False)
    aug_cells = aug.groupby(CELLS + ["metric"]).delta_pp.agg(["mean", "std", "count"]).reset_index()
    aug_cells.to_csv(output / "quality_augmentation_cell_summary.csv", index=False)
    aug_macro = aug_cells.groupby(["architecture", "augmentation", "classifier", "metric"]).agg(
        macro_delta_pp=("mean", "mean"), n_cells=("mean", "size"), positive=("mean", lambda s: int((s > EPS).sum())),
        negative=("mean", lambda s: int((s < -EPS).sum()))).reset_index()
    aug_macro.to_csv(output / "quality_augmentation_macro_summary.csv", index=False)

    # Architecture contrast is descriptive, not an identical-initialization pair.
    arch_keys = [x for x in CELLS if x != "architecture"] + ["seed"]
    rn = final[final.architecture == "resnet18"].set_index(arch_keys)
    vit = final[final.architecture == "deit_tiny"].set_index(arch_keys).reindex(rn.index)
    arch = (100 * (rn[METRICS] - vit[METRICS])).reset_index()
    arch.to_csv(output / "quality_architecture_final_resnet_minus_vit.csv", index=False)

    summary = {
        "input": str(source.resolve()), "sha256": hashlib.sha256((source / "quality.csv").read_bytes()).hexdigest(),
        "audit": {"test_ssl_rows": len(ssl), "cells": len(cells[(cells.early_epoch == 0) & (cells.metric == "roc_auc")]),
                  "cells_per_classifier": 108, "ssl_runs": 324, "seeds_per_cell": 3, "epochs": EPOCHS,
                  "baseline_test_rows": len(baseline)},
        "notes": ["Все контрасты используют эпоху 100; лучшая эпоха по test не выбирается.",
                  "Macro означает равный вес условий; это описательная сводка, не объединённая метрика предсказаний.",
                  "Три seed не являются независимыми test-выборками; клетки с общим baseline также не независимы.",
                  "Положительное изменение во всех seed — проверка согласованности знака, не тест статистической значимости.",
                  "Стандартное отклонение по seed не является доверительным интервалом.",
                  "Precision зависит от порога, выбранного только на validation; test recall может быть ниже 0.5."],
        "macro_by_architecture": macro[macro.view.eq("architecture/classifier")].replace({np.nan: None}).to_dict("records"),
        "macro_by_method": macro[macro.view.eq("architecture/method/classifier") & macro.early_epoch.eq(0)].replace({np.nan: None}).to_dict("records"),
        "augmentation_macro": aug_macro.to_dict("records"),
        "test_recall_below_half": {"all_checkpoints": int(ssl.recall_anomaly.lt(.5-EPS).sum()),
                                   "all_checkpoints_denominator": len(ssl),
                                   "final": int(final.recall_anomaly.lt(.5-EPS).sum()), "final_denominator": len(final)},
    }
    (output / "quality_findings.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(summary["audit"], indent=2))
    print("\nMacro final-minus-baseline, pp:")
    print(macro[(macro.view == "architecture/method/classifier") & (macro.early_epoch == 0)].pivot(
        index=["architecture", "method", "classifier"], columns="metric", values="macro_delta_pp").round(3).to_string())


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "reports/visa-v2-block4")
    parser.add_argument("--output", type=Path, default=root / "reports/visa-v2-final/data")
    args = parser.parse_args()
    main(args.source, args.output)
