"""Build a standalone, Russian-language VisA v2 statistics report.

Read-only with respect to the original experiment and original final report.
The scientific calculations live in analyze_visa_v2_statistics.py; this file
only presents its outputs. All charts are embedded as data URLs.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
from statistics import median
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "reports/visa-v2-statistics-2026-11-14"
LABELS = {
    "resnet18": "ResNet-18", "deit_tiny": "ViT-Tiny", "baseline": "A0",
    "rotate": "Rotate", "cutpaste": "CutPaste", "simclr": "SimCLR",
    "vicreg": "VICReg", "masked_reconstruction": "Masked reconstruction",
    "logistic_regression": "Logistic regression", "lightgbm": "LightGBM",
    "roc_auc": "ROC-AUC", "average_precision": "AP",
    "precision_anomaly": "Precision", "recall_anomaly": "Recall",
    "f1_anomaly": "F1",
}


def load_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def first(row, names, default=None):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def canon(row):
    """The aliases keep presentation independent of harmless CSV renamings."""
    r = dict(row)
    aliases = {
        "mean": ["mean_delta_pp", "mean_pp", "mean"],
        "sd": ["sd_delta_pp", "sd_pp", "sd", "std"],
        "lo": ["ci95_low", "ci95_low_pp", "ci95_lower_pp", "ci95_lo", "ci95_lo_pp", "ci_low_pp", "ci_lower_pp"],
        "hi": ["ci95_high", "ci95_high_pp", "ci95_upper_pp", "ci95_hi", "ci95_hi_pp", "ci_high_pp", "ci_upper_pp"],
        "p": ["p_t", "p_ttest", "p_ttest_two_sided", "p_value", "p_nominal", "p_two_sided"],
        "holm": ["p_t_holm_all5040", "p_holm", "p_holm_global", "p_t_holm", "p_t_holm_global", "p_holm_all"],
        "tost1": ["p_tost_1", "tost_p_1pp", "p_equivalence_1pp", "p_tost_1pp"],
        "tost2": ["p_tost_2", "tost_p_2pp", "p_equivalence_2pp", "p_tost_2pp"],
        "tost1_holm": ["p_tost_1pp_holm_all5040", "p_tost_1_holm", "tost_p_holm_1pp", "p_equivalence_1pp_holm", "p_tost_1pp_holm"],
        "tost2_holm": ["p_tost_2pp_holm_all5040", "p_tost_2_holm", "tost_p_holm_2pp", "p_equivalence_2pp_holm", "p_tost_2pp_holm"],
    }
    for key, names in aliases.items():
        r[key] = number(first(row, names))
    r["n"] = int(number(first(row, ["n", "n_seeds"], 3)) or 3)
    r["early_epoch"] = number(row.get("early_epoch"))
    r["late_epoch"] = number(row.get("late_epoch"))
    r["contrast"] = str(first(row, ["contrast", "contrast_id"], ""))
    if not r["contrast"]:
        r["contrast"] = (f"{LABELS.get(r.get('augmentation'), r.get('augmentation', ''))} − A0 · эпоха 100"
                         if row.get("comparison") == "augmentation_vs_A0"
                         else f"100−{int(r['early_epoch'])}")
    if r["mean"] is None or r["sd"] is None or r["lo"] is None or r["hi"] is None:
        raise ValueError(f"Unrecognized required statistics columns: {list(row)}")
    return r


def fmt(value, signed=False, digits=2):
    value = number(value)
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def pformat(value):
    value = number(value)
    return "—" if value is None else (f"{value:.2g}" if value < .001 else f"{value:.3f}")


def label(value):
    return LABELS.get(value, value)


def cell_label(row):
    return " / ".join(label(row.get(key, "")) for key in
                      ["category", "architecture", "augmentation", "method"])


def esc(value):
    return html.escape(str(value))


def status(row, margin=2):
    if row["sd"] < 1e-12:
        return "degenerate"
    if row["lo"] > -margin and row["hi"] < margin:
        return "small"
    if row["lo"] > margin or row["hi"] < -margin:
        return "material"
    return "uncertain"


def table(headers, rows, classes=""):
    return '<div class="table-wrap"><table class="' + classes + '"><thead><tr>' + "".join(
        f"<th>{esc(h)}</th>" for h in headers) + '</tr></thead><tbody>' + "".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def focused_rows(rows):
    return [r for r in rows if r["early_epoch"] == 30 and "augmentation" not in str(r.get("contrast_type", ""))]


def summary_table(rows):
    focused = focused_rows(rows)
    content = []
    for classifier in ["logistic_regression", "lightgbm"]:
        for metric in ["roc_auc", "average_precision", "precision_anomaly"]:
            group = [r for r in focused if r.get("classifier") == classifier and r.get("metric") == metric]
            if not group:
                continue
            content.append([esc(label(classifier)), esc(label(metric)), str(len(group)),
                fmt(min(r["sd"] for r in group), digits=3), fmt(median(r["sd"] for r in group), digits=3),
                fmt(max(r["sd"] for r in group), digits=3),
                fmt(median(r["hi"] - r["lo"] for r in group)),
                str(sum(number(r.get("positive")) == 3 or number(r.get("negative")) == 3 for r in group))])
    return table(["Пробинг", "Метрика", "Условий", "Мин. SD, п.п.", "Медиана SD, п.п.", "Макс. SD, п.п.", "Медиана ширины 95% ДИ, п.п.", "Одинаковый знак во всех 3 seed"], content)


def repeatability_tables(rows):
    groups = {}
    for r in focused_rows(rows):
        if r.get("classifier") == "logistic_regression" and r.get("metric") in {"roc_auc", "average_precision"}:
            groups.setdefault(tuple(r[k] for k in ["category", "architecture", "augmentation", "method"]), {})[r["metric"]] = r
    ordered = sorted(groups.values(), key=lambda g: max(g["roc_auc"]["sd"], g["average_precision"]["sd"]))
    def content(selected):
        result = []
        for group in selected:
            auc, ap = group["roc_auc"], group["average_precision"]
            def sign(r):
                return "3/3 ↑" if number(r.get("positive")) == 3 else "3/3 ↓" if number(r.get("negative")) == 3 else "Знак меняется / есть нули"
            result.append([esc(cell_label(auc)), fmt(auc["mean"], True), fmt(auc["sd"], digits=3), esc(sign(auc)),
                           fmt(ap["mean"], True), fmt(ap["sd"], digits=3), esc(sign(ap))])
        return table(["Условие · Logistic regression", "Δ AUC, п.п.", "SD AUC, п.п.", "Знак AUC", "Δ AP, п.п.", "SD AP, п.п.", "Знак AP"], result)
    return '<h3>Наиболее повторяемые по текущим трём seed</h3>' + content(ordered[:6]) + '<h3>Наибольший разброс</h3>' + content(list(reversed(ordered[-6:])))


def examples_table(rows):
    targets = [
        ("cashew", "deit_tiny", "rotate", "vicreg", "roc_auc"),
        ("fryum", "deit_tiny", "rotate", "simclr", "roc_auc"),
        ("capsules", "deit_tiny", "baseline", "masked_reconstruction", "roc_auc"),
        ("candle", "deit_tiny", "rotate", "vicreg", "average_precision"),
    ]
    meanings = ["Крошечное устойчивое изменение по seed", "Крупное ухудшение: кандидат для подтверждения", "Большой разброс: среднее не определяет решение", "Среднее почти нулевое, но наблюдения очень шумные"]
    body = []
    focused = focused_rows(rows)
    for target, meaning in zip(targets, meanings):
        match = [r for r in focused if tuple(r.get(k) for k in
                 ["category", "architecture", "augmentation", "method", "metric"]) == target
                 and r.get("classifier") == "logistic_regression"]
        if not match:
            continue
        r = match[0]
        body.append([esc(cell_label(r)), esc(label(r["metric"])), fmt(r["mean"], True),
                     fmt(r["sd"]), f'[{fmt(r["lo"], True)}; {fmt(r["hi"], True)}]', esc(meaning)])
    return table(["Условие · Logistic regression", "Метрика", "Среднее Δ, п.п.", "SD, п.п.", "95% ДИ по seed, п.п.", "Как читать"], body)


def bootstrap_section(directory, rows):
    path = directory / "object_bootstrap.csv"
    if not path.exists():
        return '<p>Парный bootstrap объектов не включён в эту сборку. Интервалы по seed условны на фиксированном test-наборе.</p>'
    data = load_csv(path)
    cells = {tuple(r.get(k) for k in ["category", "architecture", "augmentation", "method"]) for r in data}
    examples = []
    keys = ["category", "architecture", "augmentation", "method", "classifier", "metric"]
    targets = [
        ("cashew", "deit_tiny", "baseline", "vicreg", "logistic_regression", "average_precision"),
        ("fryum", "deit_tiny", "rotate", "simclr", "logistic_regression", "roc_auc"),
        ("fryum", "deit_tiny", "rotate", "simclr", "logistic_regression", "average_precision"),
    ]
    for target in targets:
        obj = next((r for r in data if tuple(r[k] for k in keys) == target), None)
        seed = next((r for r in focused_rows(rows) if tuple(r[k] for k in keys) == target), None)
        if obj is not None and seed is not None:
            examples.append([esc(cell_label(seed)), esc(label(seed["metric"])), fmt(seed["mean"], True),
                f'[{fmt(seed["lo"], True)}; {fmt(seed["hi"], True)}]',
                f'[{fmt(obj["object_ci95_low_pp"], True)}; {fmt(obj["object_ci95_high_pp"], True)}]'])
    examples_html = table(["Условие · Logistic regression", "Метрика", "Δ, п.п.", "95% ДИ по seed, п.п.", "95% ДИ по test-объектам, п.п."], examples)
    return f'''<p>Дополнительно выполнен парный стратифицированный bootstrap test-объектов для <b>{len(cells)} из 108 конфигураций</b>,
    для которых локально доступны предсказания всех трёх seed на эпохах 30 и 100. Это A0 и часть Rotate; отсутствие CutPaste
    в этом расчёте означает отсутствие локальных массивов, а не отсутствие эффекта.</p>
    <p>В каждой перевыборке одни и те же ID используются для обеих эпох и всех seed; нормальные и аномальные объекты
    пересэмплируются отдельно. Сначала считаются разности метрик каждого seed, затем их среднее.
    Модели и настроенные пороги фиксированы. Такой интервал показывает неопределённость <b>состава test</b>,
    а интервал по seed — неопределённость <b>повторного обучения</b>. Они не складываются и не заменяют друг друга.
    Для каждого условия выполнено 3000 перевыборок.</p>
    {examples_html}
    <p><b>Cashew / ViT / A0 / VICReg:</b> seed-интервал AP узкий, а интервал по объектам существенно шире и включает
    оба знака. Результат повторяется при переобучении на текущих данных, но состав test остаётся источником неопределённости.</p>
    <p><b>Fryum / ViT / Rotate / SimCLR:</b> ухудшение AUC и AP сохраняется в обоих типах интервалов.
    Это содержательно полезный кандидат для независимого подтверждения; приведённые интервалы номинальные, без поправки на выбор примера.</p>
    <p class="note">Bootstrap не переобучает модель и не повторяет выбор порога на validation. Он также не создаёт новые
    типы дефектов: в каждой категории остаётся исходная информация всего о 40 test-аномалиях.
    <a href="object_bootstrap.csv">Все результаты bootstrap объектов →</a></p>'''


def read_candidate_file(directory):
    for name in ["proposed_batch_24_runs.csv", "replication_queue.csv", "replication_candidates.csv", "next_wave_candidates.csv", "training_priorities.csv", "triage_by_training_configuration.csv"]:
        path = directory / name
        if path.exists():
            return path, load_csv(path)
    return None, []


def priority_section(directory, rows):
    path, candidates = read_candidate_file(directory)
    if not candidates:
        return '<p>Сначала выбрать содержательно важные условия с неопределённым результатом; высокий SD сам по себе не оправдывает неограниченную серию запусков.</p>'
    names = candidates[0].keys()
    columns = [k for k in ["category", "architecture", "augmentation", "method", "auc_mean_delta_pp", "auc_sd_delta_pp", "ap_mean_delta_pp", "ap_sd_delta_pp", "training_gpu_hours_plus3_observed"] if k in names]
    if len(columns) < 4:
        columns = list(names)[:8]
    heads = {"category": "Категория", "architecture": "Энкодер", "augmentation": "Аугментация", "method": "SSL",
             "priority": "Приоритет", "priority_group": "Группа", "reason": "Причина", "action": "Действие",
             "suggested_action": "Действие", "auc_mean_delta_pp": "Δ AUC, п.п.", "auc_sd_delta_pp": "SD AUC, п.п.",
             "ap_mean_delta_pp": "Δ AP, п.п.", "ap_sd_delta_pp": "SD AP, п.п.", "training_hours_per_seed": "SSL ч/seed",
             "additional_seeds": "Новых seed", "new_seeds": "Новые seed", "training_gpu_hours_plus3_observed": "GPU·ч на +3 seed, только SSL"}
    content = []
    for r in candidates[:8]:
        content.append([esc(label(r.get(c, ""))) if number(r.get(c)) is None else fmt(r[c]) for c in columns])
    cost = sum(number(r.get("training_gpu_hours_plus3_observed")) or 0 for r in candidates[:8])
    return '<p>Ниже пример ограниченной волны: первые 8 конфигураций с наибольшим SD по AUC или AP для Logistic regression. Фиксированный порог эффекта при сортировке не используется. Такая сортировка не оптимизирует научную ценность или бюджет; приоритет содержательной гипотезы может изменить порядок. Все числа — для 100−30.</p>' + table([heads.get(c, c) for c in columns], content) + f'<p><b>8 конфигураций × 3 новых seed = 24 обучения.</b> По наблюдавшемуся времени SSL — около {cost:.1f} GPU·ч, без пробинга, извлечения признаков, загрузки и накладных расходов. Это предложение для выбора бюджета, не запущенная очередь.</p><p><a href="{esc(path.name)}">Эта волна в CSV →</a> · <a href="replication_candidates.csv">Все 108 конфигураций и сценарии точности →</a></p>'


def build(directory, statistics_path):
    rows = [canon(r) for r in load_csv(statistics_path)]
    if not rows:
        raise ValueError("Statistics file is empty")
    focused = focused_rows(rows)
    lr = [r for r in focused if r.get("classifier") == "logistic_regression" and r.get("metric") == "roc_auc"]
    ap_lr = [r for r in focused if r.get("classifier") == "logistic_regression" and r.get("metric") == "average_precision"]
    total_sig = sum(r["holm"] is not None and r["holm"] < .05 for r in rows)
    equivalent = sum(r["tost2_holm"] is not None and r["tost2_holm"] < .05 for r in rows)
    joint = {}
    for r in focused:
        if r.get("metric") in {"roc_auc", "average_precision"}:
            joint.setdefault(tuple(r.get(k) for k in ["category", "architecture", "augmentation", "method"]), []).append(status(r) == "small")
    joint_count = sum(len(v) == 4 and all(v) for v in joint.values())
    charts = []
    for path in sorted(directory.glob("*.png")):
        if any(word in path.stem.lower() for word in ["forest", "overview", "triage", "uncertainty", "intervals"]):
            encoded = base64.b64encode(path.read_bytes()).decode()
            charts.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="Изменение качества и интервалы неопределённости"></figure>')
    payload = json.dumps(rows, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    labels = json.dumps(LABELS, ensure_ascii=False)
    assets = [p for p in directory.iterdir() if p.suffix in {".csv", ".json"} and not p.name.startswith("inventory")]
    links = "".join(f'<li><a href="{esc(p.name)}">{esc(p.name)}</a></li>' for p in sorted(assets))
    page = TEMPLATE.replace("@@ROWS@@", payload).replace("@@LABELS@@", labels)
    replacements = {
        "@@COUNT@@": str(len(rows)), "@@TOTAL_SIG@@": str(total_sig), "@@EQUIVALENT@@": str(equivalent),
        "@@SMALL_LR@@": str(sum(status(r) == "small" for r in lr)),
        "@@N_LR@@": str(len(lr)), "@@JOINT_SMALL@@": str(joint_count),
        "@@AUC_SD_MEDIAN@@": fmt(median(r["sd"] for r in lr)),
        "@@AUC_SD_MIN@@": fmt(min(r["sd"] for r in lr), digits=3),
        "@@AUC_SD_MAX@@": fmt(max(r["sd"] for r in lr)),
        "@@AP_SD_MEDIAN@@": fmt(median(r["sd"] for r in ap_lr)),
        "@@AP_SD_MIN@@": fmt(min(r["sd"] for r in ap_lr), digits=3),
        "@@AP_SD_MAX@@": fmt(max(r["sd"] for r in ap_lr)),
        "@@SUMMARY_TABLE@@": summary_table(rows), "@@EXAMPLES@@": examples_table(rows),
        "@@REPEATABILITY@@": repeatability_tables(rows),
        "@@PRIORITY@@": priority_section(directory, rows), "@@BOOTSTRAP@@": bootstrap_section(directory, rows),
        "@@CHARTS@@": "".join(charts), "@@DOWNLOADS@@": links,
    }
    for before, after in replacements.items():
        page = page.replace(before, after)
    target = directory / "index.html"
    target.write_text(page, encoding="utf-8")
    assert "@@" not in page, "Unresolved template marker"
    print(f"Wrote {target} ({len(rows)} comparisons)")


TEMPLATE = r'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VisA v2 · Какие эксперименты повторять</title>
<style>
:root{--ink:#172b3a;--muted:#536778;--line:#dbe4e9;--blue:#225d81;--light:#f3f7fa;--teal:#146655;--amber:#875c06;--rose:#9e3a53}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:#f5f7f8;font:16px/1.65 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1420px;margin:auto;padding:48px 36px 80px}header{padding-bottom:26px}.eyebrow{text-transform:uppercase;letter-spacing:.11em;font-size:12px;font-weight:750;color:var(--blue)}h1{font-size:42px;line-height:1.12;letter-spacing:-.025em;margin:12px 0 18px;max-width:1000px}h2{font-size:25px;line-height:1.3;margin:0 0 18px}h3{font-size:18px;margin:24px 0 10px}p{margin:10px 0 16px}.lead{font-size:19px;max-width:1020px}.muted,.note{color:var(--muted)}.note{font-size:14px}a{color:var(--blue);text-decoration-thickness:1px;text-underline-offset:3px}section{background:white;border:1px solid var(--line);border-radius:14px;padding:28px;margin:22px 0}.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:26px 0 4px}.stat{padding:18px 20px;background:white;border:1px solid var(--line);border-radius:12px}.stat b{display:block;font-size:30px;line-height:1.1;margin-bottom:7px}.stat span{font-size:13px;line-height:1.45;display:block;color:var(--muted)}.callout{border-left:4px solid var(--blue);padding:12px 18px;background:var(--light);border-radius:0 9px 9px 0}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:8px;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.45}th{background:var(--light);text-align:left;font-weight:650;padding:12px 13px;vertical-align:bottom;border-bottom:1px solid var(--line)}td{padding:12px 13px;border-bottom:1px solid #e9eef1;vertical-align:top}tbody tr:last-child td{border-bottom:0}tbody tr:hover{background:#fafcfd}.methods{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.method{padding:18px;border:1px solid var(--line);border-radius:10px}.method h3{margin:0 0 8px}.method p{font-size:14px;margin:0}.small{color:var(--teal)}.material{color:var(--rose)}.uncertain,.degenerate{color:var(--amber)}.badge{display:inline-block;font-size:12px;font-weight:650;line-height:1.3;border-radius:6px;background:#f1f4f5;padding:5px 7px}figure{margin:20px 0}figure img{display:block;width:100%;height:auto}label{display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:650;color:var(--muted)}select,input,button{font:14px system-ui;color:var(--ink);background:white;border:1px solid #bdccd5;border-radius:7px;padding:9px 10px;min-height:40px}button{cursor:pointer}button:disabled{opacity:.45;cursor:default}input{min-width:240px}.filters{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}.filters label{flex:1;min-width:135px}.controls{display:flex;justify-content:space-between;align-items:center;gap:14px}.controls p{margin:0;font-size:13px;color:var(--muted)}.num{font-variant-numeric:tabular-nums;white-space:nowrap}.data-table{min-width:1200px}.data-table th{position:sticky;top:0}.data-table td{font-size:12px}details{margin:16px 0}summary{font-weight:650;cursor:pointer}.sources{font-size:14px}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:28px}ul,ol{padding-left:22px}li{margin:8px 0}footer{font-size:13px;color:var(--muted);padding-top:10px}@media(max-width:800px){main{padding:25px 16px 50px}section{padding:20px}h1{font-size:33px}.stat-grid{grid-template-columns:1fr 1fr}.methods,.two-col{grid-template-columns:1fr}.lead{font-size:17px}}@media print{body{background:white}main{padding:0}section{break-inside:avoid;border-radius:0}.filters,.controls{display:none}h1{font-size:32px}}
</style></head><body><main>
<header><div class="eyebrow">VisA v2 · статистический разбор · 14 ноября 2026</div>
<h1>Какие эксперименты повторять</h1>
<p class="lead">Главный критерий — небольшой разброс между повторениями. Смотрим, насколько повторяются изменения качества, сохраняется ли их знак и насколько точно оценено среднее. Все запланированные обучения завершены; ниже речь о дополнительных независимых повторениях.</p>
<div class="stat-grid"><div class="stat"><b>324 / 324</b><span>SSL-обучения завершены<br>недостающих запусков нет</span></div><div class="stat"><b>108</b><span>конфигураций обучения<br>6 категорий × 2 энкодера × 3 SSL × 3 аугментации</span></div><div class="stat"><b>3 seed</b><span>на каждую конфигурацию<br>6101, 6102, 6103</span></div><div class="stat"><b>40 / 12</b><span>аномалий на категорию<br>в test / validation</span></div></div></header>
<section><h2>Разброс сильно зависит от условия</h2>
<p>Для каждого seed считается изменение одной и той же модели между эпохами: Δ = качество на эпохе 100 − качество на эпохе 30. SD этих трёх разностей показывает, насколько результат меняется от повторного обучения. Среднее и SD нужно читать вместе.</p>
<div class="callout"><b>Logistic regression, 100−30:</b> медианный SD изменения ROC-AUC — <b>@@AUC_SD_MEDIAN@@ п.п.</b>, диапазон по условиям @@AUC_SD_MIN@@–@@AUC_SD_MAX@@ п.п.
Для AP медианный SD — <b>@@AP_SD_MEDIAN@@ п.п.</b>, диапазон @@AP_SD_MIN@@–@@AP_SD_MAX@@ п.п. Одни части сетки уже хорошо повторяются, другие значительно шумнее.</div>
<p>Поэтому повторять всю сетку одинаково не требуется. Приоритет можно снизить у условий с небольшим SD и узким интервалом для интересующего вывода, а новые seed сосредоточить там, где большой разброс или меняется направление эффекта. При этом устойчивость обучения не устраняет неопределённость маленького test-набора.</p>
<p class="note">Здесь не задан универсальный числовой порог «достаточно малого» SD. Три повторения дают предварительную оценку разброса; она сама может измениться после новых запусков. 1 п.п. соответствует изменению метрики на 0,01.</p>
</section>
<section><h2>Что смотреть рядом со средним</h2><div class="methods">
<div class="method"><h3 class="small">SD: повторяемость</h3><p>Малый SD означает близкие изменения качества при повторных обучениях. Он может сопровождать и нулевое, и положительное, и отрицательное среднее изменение.</p></div>
<div class="method"><h3 class="material">Знак: устойчивость вывода</h3><p>Если направление повторяется во всех seed и среднее велико относительно разброса, вывод уже выглядит устойчивее. Но три совпавших знака сами по себе ещё не доказывают статистическую значимость.</p></div>
<div class="method"><h3 class="uncertain">Интервал: точность среднего</h3><p>Доверительный интервал показывает неопределённость среднего эффекта. При n = 3 он широк даже при умеренном SD. Отдельный bootstrap test-объектов проверяет другой источник шума.</p></div>
</div></section>
<section><h2>Продолжение обучения: эпоха 100 против эпохи 30</h2>
<p>Этот контраст выбран сейчас как удобный фокус планирования. Он входил в исходный набор сравнений вместе с 100−0, 100−5 и 100−10; «лучшую» раннюю эпоху по test мы не ищем. В строке ниже — 108 условий одного пробинга и одной метрики.</p>
@@SUMMARY_TABLE@@
<p class="note">SD рассчитан по парным изменениям, а не по смешанным значениям разных эпох, категорий или методов. Ширина 95% интервала — верхняя граница минус нижняя; при трёх seed она равна примерно 4,968 × SD.</p>
@@CHARTS@@</section>
<section><h2>Где повторения уже близки, а где расходятся</h2><p>Для обзорной сортировки берём больший из двух SD — ROC-AUC и AP для Logistic regression. Так низкий разброс одной метрики не скрывает высокий разброс другой. Это относительный порядок внутри текущей сетки, без фиксированного порога эффекта.</p>
@@REPEATABILITY@@
<p class="note">Оба SD показаны отдельно. Для важных условий нужно также смотреть LightGBM и интервалы по test-объектам. <a href="repeatability_by_condition.csv">Все условия по возрастанию разброса →</a></p></section>
<section><h2>Почему средних недостаточно</h2>@@EXAMPLES@@
<p class="note">Примеры выбраны после просмотра результатов. Все интервалы в таблице оценивают среднюю парную разность по seed; они не являются интервалами для отдельного будущего запуска.</p></section>
<section><h2>Что поставить в следующую волну</h2>
<p>Сначала имеет смысл ограничить список важными гипотезами и добавить одинаковые новые seed в сравниваемые условия. Для длительности один новый запуск до 100 эпох даст сразу обе точки 30 и 100. Для эффекта Rotate или CutPaste нужен также соответствующий A0 с тем же новым seed.</p>
@@PRIORITY@@
<div class="callout"><b>Небольшая следующая волна:</b> довести выбранные условия с 3 до 6 seed, затем заново оценить разброс и бюджет.
Шесть seed не гарантируют достаточную точность. Число повторов, рассчитанное по SD всего трёх наблюдений, — только сценарий планирования.</div>
<p class="note">Запуски в рамках этого анализа не отправлялись. Условия, метрики, границы практической важности и правило остановки следующей волны следует зафиксировать до новых результатов; повторное добавление seed до получения p &lt; 0,05 не является корректным правилом.</p></section>
<section><h2>Шум обучения и шум test-набора</h2>@@BOOTSTRAP@@
<p>Если основной источник неопределённости — seed, нужны новые обучения. Если результат чувствителен к нескольким test-аномалиям, полезнее новые уникальные оценочные объекты или независимая проверка. Новые seed на тех же 40 аномалиях не заменяют новые данные.</p></section>
<section id="explorer"><h2>Все сравнения</h2><p class="note">По умолчанию сначала показан наибольший SD. Фильтры меняют только отображение. Поправка Holm остаётся общей и не пересчитывается по отфильтрованным строкам. Пороги ±1 и ±2 п.п. — необязательный сценарий чувствительности, а не критерий, заданный пользователем, и не автоматическое правило остановки.</p>
<div class="filters"><label>Сравнение<select id="contrast"></select></label><label>Метрика<select id="metric"></select></label><label>Пробинг<select id="classifier"></select></label><label>Энкодер<select id="architecture"></select></label><label>Иллюстративный порог эффекта<select id="margin"><option value="2">±2 п.п.</option><option value="1">±1 п.п.</option></select></label><label>Поиск условия<input id="search" placeholder="Например, capsules simclr"></label></div>
<div class="filters"><label>Статус по 95% ДИ<select id="state"><option value="all">Все статусы</option><option value="uncertain">Неопределённо</option><option value="small">Небольшое и точное</option><option value="material">Крупное одного знака</option><option value="degenerate">Нулевой SD</option></select></label><label>Порядок<select id="sort"><option value="sd">Больше разброс по seed</option><option value="abs">Больше абсолютное изменение</option><option value="mean">Больше среднее изменение</option><option value="name">По условию</option></select></label></div>
<div class="controls"><p id="counter" aria-live="polite"></p><div><button id="previous">← Назад</button> <button id="next">Далее →</button></div></div>
<div class="table-wrap"><table class="data-table"><thead><tr><th>Условие / пробинг</th><th>Сравнение / метрика</th><th>Δ по трём seed, п.п.</th><th>Среднее Δ<br>п.п.</th><th>SD<br>п.п.</th><th>95% ДИ<br>п.п.</th><th>p без поправки</th><th>p Holm</th><th>TOST: p / p Holm</th><th>Статус по ДИ</th></tr></thead><tbody id="data"></tbody></table></div></section>
<section><h2>Статистическая значимость и дополнительные сценарии</h2>
<p>В общей семье из @@COUNT@@ сравнений после поправки Holm <b>@@TOTAL_SIG@@ тестов отличия от нуля</b> проходят уровень 0,05.
Это не означает отсутствия эффекта. При трёх seed строгие множественные проверки имеют мало возможностей подтвердить вывод, а маленькое p без поправки не заменяет оценку разброса.</p>
<details><summary>Чувствительность к условным границам ±1 и ±2 п.п.</summary>
<p>Эти границы сохранены только как иллюстрация различия между «средний эффект мал» и «эффект оценён точно». Для ROC-AUC и Logistic regression при 100−30 у @@SMALL_LR@@ из @@N_LR@@ условий 95% интервал по seed лежит внутри ±2 п.п. У обоих пробингов одновременно по AUC и AP — @@JOINT_SMALL@@ из 108. После общей поправки Holm тестов эквивалентности ±2 п.п. с p &lt; 0,05 — @@EQUIVALENT@@.</p>
<p>Эти числа не определяют, какие эксперименты обязательно повторять: для текущей задачи главным остаётся фактический разброс. В интерактивной таблице можно посмотреть оба сценария.</p></details>
</section>
<section><h2>Как считали и где заканчиваются выводы</h2><div class="two-col"><div>
<h3>Единица анализа</h3><p>Внутри условия разность считается между парными результатами одного seed. Эпохи, два классификатора, raw/L2 и разные метрики не увеличивают число повторений. Все блоки были продолжением одной серии; используется последний полный блок без складывания копий.</p>
<h3>Интервалы и тест отличия</h3><p>Среднее ± t<sub>0,975; n−1</sub> × SD / √n. При n = 3 половина 95% интервала равна 2,484 × SD. Парный t-тест проверяет нулевую среднюю разность. Его выводы требуют модели нормальных независимых разностей seed и условны на фиксированном разбиении.</p>
<p>При трёх разностях точный двусторонний тест со сменой знака имеет лишь восемь комбинаций, а минимальное достижимое p — 0,25. Согласованный знак во всех трёх запусках не является достаточным непараметрическим подтверждением на уровне 0,05.</p>
</div><div><h3>Эквивалентность и множество сравнений</h3><p>TOST проверяет, помещается ли средний эффект внутрь заданных границ. Без поправки уровень 0,05 соответствует 90% t-интервалу внутри ±порога. Более строгий 95% интервал используется отдельно для сортировки практического приоритета. В таблице есть p TOST и его поправка Holm для каждого порога.</p>
<p>Общая семья включает все @@COUNT@@ сравнений длительности и аугментаций по пяти метрикам и двум пробингам. Отличие от нуля и эквивалентность при каждом пороге корректируются как отдельные семейства. Разделение на фильтры не делает семью меньше.</p>
<h3>Что не проверено</h3><p>Это не доказательство переноса на новые категории или новые распределения данных. Seed меняет инициализацию, порядок обучения и случайность полного протокола; разбиения фиксированы. Precision читается вместе с recall и AP, поскольку порог выбран по маленькому validation.</p></div></div>
<details><summary>Методические источники и протокол</summary><ul class="sources"><li><a href="https://itl.nist.gov/div898/handbook/prc/section3/prc312.htm">NIST: доверительные интервалы для парных разностей</a>.</li><li><a href="https://www.statsmodels.org/v0.14.4/generated/statsmodels.stats.weightstats.ttost_paired.html">Statsmodels: парный TOST и границы эквивалентности</a>.</li><li><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html">SciPy: точные перестановочные тесты</a>.</li><li><a href="../../docs/VISA_EXPANDED_PLAN_RU.md">Исходный протокол VisA v2</a>. Итоговые таблицы: <a href="../visa-v2-block4/quality.csv">quality.csv</a>, <a href="../visa-v2-block4/loss.csv">loss.csv</a>; контрольные суммы сохранены в summary.json.</li></ul></details></section>
<section><h2>Данные и воспроизведение</h2><p>Исходные результаты не изменялись. CSV содержат полные значения без округления, JSON — сводки и аудит. Для передачи всего набора сохраняйте эту папку целиком; сама HTML-страница открывается автономно, включая таблицу и графики.</p><details><summary>Скачать таблицы и аудит</summary><ul>@@DOWNLOADS@@</ul></details></section>
<footer>VisA v2 · статистика для планирования дополнительных наблюдений · 14.11.2026. HTML построен из рассчитанных таблиц, не из изображений прежнего отчёта.</footer>
</main><script>
const rows=@@ROWS@@;const labels=@@LABELS@@;const el=id=>document.getElementById(id);const lab=x=>labels[x]||x||'';const safe=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const num=(x,s=false)=>x===null||x===undefined?'—':(s&&x>=0?'+':'')+Number(x).toFixed(2);const pv=x=>x===null||x===undefined?'—':Number(x)<.001?Number(x).toExponential(2):Number(x).toFixed(3);
const stat=(r,m)=>r.sd<1e-12?'degenerate':r.lo>-m&&r.hi<m?'small':r.lo>m||r.hi< -m?'material':'uncertain';const statLabel={small:'Небольшое и точное',material:'Крупное одного знака',uncertain:'Неопределённо',degenerate:'Нулевой SD: осторожно'};let page=0;const pageSize=40;
for(const key of ['contrast','metric','classifier','architecture']){const values=[...new Set(rows.map(r=>String(r[key])))];el(key).innerHTML='<option value="all">Все</option>'+values.map(v=>'<option value="'+safe(v)+'">'+safe(lab(v))+'</option>').join('');}
const firstFocused=rows.find(r=>r.early_epoch===30);if(firstFocused)el('contrast').value=firstFocused.contrast;el('metric').value='roc_auc';el('classifier').value='logistic_regression';
function seedDeltas(r){let result=[];for(const seed of ['6101','6102','6103']){const v=r['delta_seed_'+seed+'_pp']??r['delta_seed_'+seed]??r['delta_s'+seed+'_pp']??r['delta_'+seed+'_pp']??r['seed_'+seed+'_delta_pp']??r['delta_'+seed]??r['seed_'+seed];if(v!==undefined&&v!=='')result.push(num(Number(v),true));}if(!result.length){const raw=r.seed_deltas_pp??r.deltas_pp??r.seed_deltas;if(raw){try{const d=typeof raw==='string'?JSON.parse(raw):raw;result=(Array.isArray(d)?d:Object.values(d)).map(v=>num(Number(v),true));}catch{result=[safe(raw)];}}}return result.join(' / ')||'см. CSV';}
function render(){const margin=Number(el('margin').value);const terms=el('search').value.toLowerCase().trim().split(/\s+/).filter(Boolean);let view=rows.filter(r=>['contrast','metric','classifier','architecture'].every(k=>el(k).value==='all'||String(r[k])===el(k).value));view=view.filter(r=>terms.every(t=>Object.values(r).map(lab).join(' ').toLowerCase().includes(t))).filter(r=>el('state').value==='all'||stat(r,margin)===el('state').value);const sort=el('sort').value;view.sort((a,b)=>sort==='sd'?b.sd-a.sd:sort==='abs'?Math.abs(b.mean)-Math.abs(a.mean):sort==='mean'?b.mean-a.mean:[a.category,a.architecture,a.augmentation,a.method].join().localeCompare([b.category,b.architecture,b.augmentation,b.method].join()));page=Math.max(0,Math.min(page,Math.ceil(view.length/pageSize)-1));const slice=view.slice(page*pageSize,(page+1)*pageSize);el('counter').textContent=view.length?`${page*pageSize+1}–${Math.min((page+1)*pageSize,view.length)} из ${view.length} сравнений`:'По выбранным фильтрам сравнений нет';el('previous').disabled=page===0;el('next').disabled=(page+1)*pageSize>=view.length;el('data').innerHTML=slice.map(r=>{const s=stat(r,margin);return '<tr><td><b>'+safe([r.category,lab(r.architecture),lab(r.augmentation),lab(r.method)].join(' / '))+'</b><br>'+safe(lab(r.classifier))+'</td><td>'+safe(r.contrast)+'<br>'+safe(lab(r.metric))+'</td><td class="num">'+seedDeltas(r)+'</td><td class="num">'+num(r.mean,true)+'</td><td class="num">'+num(r.sd)+'</td><td class="num">['+num(r.lo,true)+'; '+num(r.hi,true)+']</td><td class="num">'+pv(r.p)+'</td><td class="num">'+pv(r.holm)+'</td><td class="num">'+pv(r['tost'+margin])+' / '+pv(r['tost'+margin+'_holm'])+'</td><td><span class="badge '+s+'">'+statLabel[s]+'</span></td></tr>';}).join('');}
for(const id of ['contrast','metric','classifier','architecture','margin','search','state','sort'])el(id).addEventListener('input',()=>{page=0;render();});el('previous').addEventListener('click',()=>{page--;render();});el('next').addEventListener('click',()=>{page++;render();});render();
</script></body></html>'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--statistics", type=Path)
    args = parser.parse_args()
    source = args.statistics
    if source is None:
        options = [args.directory / name for name in ["all_statistics.csv", "statistics.csv", "paired_statistics.csv", "seed_statistics.csv"]]
        source = next((p for p in options if p.exists()), None)
    if source is None:
        raise SystemExit("No statistics table found; pass --statistics PATH")
    build(args.directory, source)
