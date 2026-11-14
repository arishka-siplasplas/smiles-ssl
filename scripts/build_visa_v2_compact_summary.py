"""Present prepared VisA v2 conclusions as a compact standalone HTML report.

This builder performs formatting only; all statistical values are supplied by
the accompanying summary tables/JSON. It does not read model predictions, fit
tests, or modify the original experiment and report.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "reports/visa-v2-statistics-2026-11-14/summary"
NAMES = {
    "resnet18": "ResNet-18", "deit_tiny": "ViT-Tiny", "baseline": "A0",
    "rotate": "Rotate", "cutpaste": "CutPaste", "simclr": "SimCLR",
    "vicreg": "VICReg", "masked_reconstruction": "Masked reconstruction",
    "logistic_regression": "Logistic regression", "lightgbm": "LightGBM",
    "roc_auc": "ROC-AUC", "average_precision": "AP",
    "precision_anomaly": "Precision", "recall_anomaly": "Recall", "f1_anomaly": "F1",
}


def esc(value):
    return html.escape(str(value))


def label(value):
    return NAMES.get(value, value)


def number(value, signed=False, digits=2):
    if value is None or value == "":
        return "—"
    value = float(value)
    if not math.isfinite(value):
        return "—"
    result = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return result.replace("-", "−")


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def table(headers, rows, max_rows=None):
    if max_rows is not None and len(rows) > max_rows:
        raise ValueError(f"Compact table has {len(rows)} rows; maximum is {max_rows}")
    head = "".join(f"<th>{esc(value)}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>" for row in rows)
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def prose_cards(records):
    return "".join(f'<article><h3>{esc(r["title"])}</h3><p>{esc(r["text"])}</p></article>' for r in records)


def render_prepared_table(spec, max_rows):
    """A small generic interface for already-formatted report table cells."""
    return table(spec["headers"], [[esc(v) for v in row] for row in spec["rows"]], max_rows=max_rows)


def build(directory, data):
    content = TEMPLATE
    replacements = {
        "@@HEADLINES@@": prose_cards(data["headlines"]),
        "@@DURATION@@": render_prepared_table(data["duration_table"], 4),
        "@@METHODS@@": render_prepared_table(data["method_table"], 6),
        "@@AUGMENTATIONS@@": render_prepared_table(data["augmentation_table"], 4),
        "@@DURATION_NOTES@@": "".join(f"<p>{esc(v)}</p>" for v in data.get("duration_notes", [])),
        "@@METHOD_NOTES@@": "".join(f"<p>{esc(v)}</p>" for v in data.get("method_notes", [])),
        "@@AUGMENTATION_NOTES@@": "".join(f"<p>{esc(v)}</p>" for v in data.get("augmentation_notes", [])),
        "@@SIGNIFICANCE@@": "".join(f"<p>{esc(v)}</p>" for v in data["significance_notes"]),
        "@@NOISE@@": "".join(f"<p>{esc(v)}</p>" for v in data["noise_notes"]),
        "@@CASES@@": prose_cards(data["cases"]),
        "@@NEXT@@": "".join(f"<li>{esc(v)}</li>" for v in data["next_steps"]),
        "@@DOWNLOADS@@": "".join(f'<li><a href="{esc(r["path"])}">{esc(r["label"])}</a></li>' for r in data.get("downloads", [])),
    }
    for key, value in replacements.items():
        content = content.replace(key, value)
    if "@@" in content:
        raise ValueError("Unresolved report placeholder")
    output = directory / "index.html"
    output.write_text(content, encoding="utf-8")
    print(f"Wrote {output} ({output.stat().st_size:,} bytes)")


TEMPLATE = '''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VisA v2 — выводы и статистика</title>
<style>
:root{--ink:#203140;--muted:#607080;--line:#dce5ea;--blue:#275e7c;--soft:#f3f7f9}
*{box-sizing:border-box}body{margin:0;background:#f5f7f8;color:var(--ink);font:16px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1160px;margin:auto;padding:45px 32px 65px}.eyebrow{font-size:12px;color:var(--blue);font-weight:750;letter-spacing:.12em;text-transform:uppercase}h1{font-size:39px;line-height:1.15;letter-spacing:-.025em;margin:12px 0 17px}h2{font-size:24px;line-height:1.3;margin:0 0 18px}h3{font-size:17px;line-height:1.4;margin:0 0 10px}p{margin:10px 0 16px}.lead{font-size:18px;max-width:940px}.metadata{display:flex;gap:8px;flex-wrap:wrap;margin:23px 0 29px}.metadata span{background:#e8f0f4;border-radius:7px;padding:7px 11px;font-size:13px}.section{padding:25px 27px;background:white;border:1px solid var(--line);border-radius:13px;margin:19px 0}.cards{display:grid;grid-template-columns:1fr 1fr;gap:16px}.cards article{padding:19px;background:var(--soft);border:1px solid var(--line);border-radius:10px}.cards p{font-size:14px;margin:0}.conclusions article{border-left:3px solid var(--blue)}.note{font-size:13px;color:var(--muted)}.table-scroll{overflow:auto;border:1px solid var(--line);border-radius:8px;margin:15px 0 18px}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.45}th{text-align:left;background:var(--soft);padding:12px 11px;vertical-align:bottom;font-weight:650}td{padding:12px 11px;border-top:1px solid var(--line);vertical-align:top;font-variant-numeric:tabular-nums}td:first-child{font-weight:600}td small{display:block;color:var(--muted)}.two{display:grid;grid-template-columns:1fr 1fr;gap:28px}.callout{padding:12px 16px;background:var(--soft);border-left:3px solid var(--blue)}a{color:var(--blue);text-underline-offset:3px}ol,ul{padding-left:23px}li{margin:10px 0}summary{cursor:pointer;font-weight:650}footer{color:var(--muted);font-size:12px;margin-top:22px}strong{font-weight:700}@media(max-width:760px){main{padding:25px 14px 45px}.section{padding:20px}.cards,.two{grid-template-columns:1fr}h1{font-size:32px}table{font-size:12px}td,th{padding:10px 8px}}@media print{body{background:white}main{padding:0}.section{border-radius:0;break-inside:avoid}a{color:inherit;text-decoration:none}details{display:none}}
</style></head><body><main>
<header><div class="eyebrow">VisA v2 · краткий разбор · 14 ноября 2026</div>
<h1>Что показал большой прогон</h1>
<p class="lead">Выводы о длительности SSL, методах и аугментациях — вместе с величиной изменений и их разбросом. Главный вопрос для следующих запусков: какие наблюдения уже повторяются, а какие ещё зависят от seed или test-набора.</p>
<div class="metadata"><span>324 / 324 обучения завершены</span><span>108 конфигураций × 3 seed</span><span>6 категорий · 2 энкодера · 2 пробинга</span><span>40 test-аномалий на категорию</span></div></header>
<section class="section"><h2>Основные выводы</h2><div class="cards conclusions">@@HEADLINES@@</div></section>
<section class="section"><h2>1. Длительность обучения</h2>
<p>Сравниваем финальную эпоху 100 с четырьмя заранее заданными точками: 0, 5, 10 и 30. Это парные изменения одного seed; лучшую раннюю эпоху по test не выбираем.</p>
@@DURATION@@
@@DURATION_NOTES@@
<p class="note">Все изменения ниже выражены в процентных пунктах: AUC 0,80 → 0,82 соответствует +2 п.п. Здесь нет правила «прекратить, если меньше 2%».</p></section>
<section class="section"><h2>2. Метод и архитектура меняют картину</h2>
@@METHODS@@
@@METHOD_NOTES@@</section>
<section class="section"><h2>3. Что дали Rotate и CutPaste</h2>
<p>Аугментации сравниваются с A0 при одинаковых категории, архитектуре, SSL-методе и seed. В таблице — различия на фиксированной эпохе 100; A0 — исходный pipeline без добавленного Rotate или CutPaste.</p>
@@AUGMENTATIONS@@
@@AUGMENTATION_NOTES@@</section>
<section class="section"><h2>4. Насколько устойчивы выводы</h2><div class="two"><div><h3>Разброс и точность</h3>@@NOISE@@</div><div><h3>Статистическая значимость</h3>@@SIGNIFICANCE@@</div></div>
<p class="note">Два классификатора, разные эпохи и метрики не превращают три обучения в большее число независимых повторений. Среднее по сетке характеризует только этот фиксированный набор категорий и условий.</p></section>
<section class="section"><h2>Четыре случая, которые помогают решить, что догонять</h2><div class="cards">@@CASES@@</div>
<p class="note">Примеры выбраны после просмотра результатов. Они помогают планировать независимое подтверждение, но сами не являются формальным подтверждением гипотезы.</p></section>
<section class="section"><h2>Следующие действия</h2><ol>@@NEXT@@</ol>
<div class="callout">Повторять всю сетку одинаково не требуется. Выбор следующей волны должен сочетать фактический разброс и важность проверяемого вывода; отсутствие значимости само по себе не задаёт приоритет.</div></section>
<section class="section"><details><summary>Таблицы, определения и источники</summary><ul>@@DOWNLOADS@@</ul>
<p class="note">SD — выборочное стандартное отклонение парных изменений по seed. 95% t-интервал среднего: mean ± t × SD / √3; при трёх seed половина интервала равна 2,484 × SD. Для интервального вывода нужна модель нормальных разностей, которую по n = 3 проверить нельзя. <a href="https://itl.nist.gov/div898/handbook/prc/section3/prc312.htm">NIST: парные доверительные интервалы</a>.</p>
<p class="note">Все исходные модели, метрики и прежние отчёты сохранены. Новые обучения этим анализом не запускались.</p></details></section>
<footer>Краткая сводка подготовлена из проверенных таблиц VisA v2. Самодостаточный HTML без внешних скриптов, сетевых запросов и встроенной полной таблицы экспериментов.</footer>
</main></body></html>'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    source = args.data or args.directory / "compact_summary.json"
    build(args.directory, json.loads(source.read_text()))
