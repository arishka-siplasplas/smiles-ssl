"""Format prepared statistics for VisA v2 epochs 0 through 30.

This builder only presents supplied table cells and conclusions. Statistical
calculations and selection of comparisons belong to the companion analysis.
It creates a small static HTML document without embedding the complete data.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "reports/visa-v2-statistics-2026-11-14/early"


def esc(value):
    return html.escape(str(value))


def paragraphs(values):
    return "".join(f"<p>{esc(value)}</p>" for value in values)


def cards(records):
    return "".join(f'<article><h3>{esc(row["title"])}</h3><p>{esc(row["text"])}</p></article>' for row in records)


def table(spec, maximum=6):
    headers, rows = spec["headers"], spec["rows"]
    if len(rows) > maximum:
        raise ValueError(f"Compact table has {len(rows)} rows; maximum is {maximum}")
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("Prepared table row does not match its headers")
    head = "".join(f"<th>{esc(value)}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(value)}</td>" for value in row) + "</tr>" for row in rows)
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def build(directory, data):
    if len(data["stage_table"]["rows"]) != 6 or len(data["method_table"]["rows"]) != 6:
        raise ValueError("The early report requires six stage rows and six method rows")
    cumulative = ""
    if data.get("cumulative_table"):
        cumulative = '<section class="section"><details><summary>Накопленные изменения относительно эпохи 0</summary>' + table(data["cumulative_table"]) + paragraphs(data.get("cumulative_notes", [])) + '</details></section>'
    examples = ""
    if data.get("cases"):
        examples = '<section class="section"><h2>Примеры разных ранних изменений</h2><div class="cards">' + cards(data["cases"]) + '</div><p class="note">Примеры выбраны после просмотра данных. Они иллюстрируют величину и разброс парных изменений, а не подтверждают выбранную гипотезу независимой проверкой.</p></section>'
    substitutions = {
        "@@HEADLINES@@": cards(data["headlines"]),
        "@@STAGES@@": table(data["stage_table"]),
        "@@STAGE_NOTES@@": paragraphs(data.get("stage_notes", [])),
        "@@METHODS@@": table(data["method_table"]),
        "@@METHOD_NOTES@@": paragraphs(data.get("method_notes", [])),
        "@@NOTES@@": paragraphs(data.get("notes", [])),
        "@@SIGNIFICANCE@@": paragraphs(data["significance_notes"]),
        "@@CUMULATIVE@@": cumulative,
        "@@CASES@@": examples,
        "@@DOWNLOADS@@": "".join(f'<li><a href="{esc(row["path"])}">{esc(row["label"])}</a></li>' for row in data.get("downloads", [])),
    }
    page = TEMPLATE
    for marker, value in substitutions.items():
        page = page.replace(marker, value)
    if "@@" in page:
        raise ValueError("Unresolved HTML placeholder")
    output = directory / "index.html"
    output.write_text(page, encoding="utf-8")
    print(f"Wrote {output} ({output.stat().st_size:,} bytes)")


TEMPLATE = '''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VisA v2 — первые 30 эпох</title>
<style>
:root{--ink:#203140;--muted:#607080;--line:#dce5ea;--blue:#275e7c;--soft:#f3f7f9}
*{box-sizing:border-box}body{margin:0;background:#f5f7f8;color:var(--ink);font:16px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1160px;margin:auto;padding:45px 32px 65px}.eyebrow{font-size:12px;color:var(--blue);font-weight:750;letter-spacing:.12em;text-transform:uppercase}h1{font-size:39px;line-height:1.15;letter-spacing:-.025em;margin:12px 0 17px}h2{font-size:24px;line-height:1.3;margin:0 0 18px}h3{font-size:17px;line-height:1.4;margin:0 0 10px}p{margin:10px 0 16px}.lead{font-size:18px;max-width:940px}.metadata{display:flex;gap:8px;flex-wrap:wrap;margin:23px 0 29px}.metadata span{background:#e8f0f4;border-radius:7px;padding:7px 11px;font-size:13px}.section{padding:25px 27px;background:white;border:1px solid var(--line);border-radius:13px;margin:19px 0}.cards{display:grid;grid-template-columns:1fr 1fr;gap:16px}.cards article{padding:19px;background:var(--soft);border:1px solid var(--line);border-left:3px solid var(--blue);border-radius:10px}.cards p{font-size:14px;margin:0}.note{font-size:13px;color:var(--muted)}.table-scroll{overflow:auto;border:1px solid var(--line);border-radius:8px;margin:15px 0 18px}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.45}th{text-align:left;background:var(--soft);padding:12px 11px;vertical-align:bottom;font-weight:650}td{padding:12px 11px;border-top:1px solid var(--line);vertical-align:top;font-variant-numeric:tabular-nums}td:first-child{font-weight:600}.callout{padding:12px 16px;background:var(--soft);border-left:3px solid var(--blue)}a{color:var(--blue);text-underline-offset:3px}ul{padding-left:23px}li{margin:10px 0}summary{cursor:pointer;font-weight:650}footer{color:var(--muted);font-size:12px;margin-top:22px}@media(max-width:760px){main{padding:25px 14px 45px}.section{padding:20px}.cards{grid-template-columns:1fr}h1{font-size:32px}table{font-size:12px}td,th{padding:10px 8px}}@media print{body{background:white}main{padding:0}.section{border-radius:0;break-inside:avoid}a{color:inherit;text-decoration:none}}
</style></head><body><main>
<header><div class="eyebrow">VisA v2 · ранняя динамика · 14 ноября 2026</div>
<h1>Что происходит в первые 30 эпох</h1>
<p class="lead">Отдельный разбор раннего обучения: что меняется на каждом соседнем интервале, где направление повторяется между seed и какие методы дают разные результаты к эпохам 5 и 30.</p>
<div class="metadata"><span>Эпохи 0, 1, 2, 5, 10, 20, 30</span><span>6 соседних интервалов</span><span>108 конфигураций × 3 seed</span><span>2 классификатора</span></div></header>
<section class="section"><h2>Основные наблюдения</h2><div class="cards">@@HEADLINES@@</div></section>
<section class="section"><h2>1. Изменения между соседними ранними точками</h2>
<p>Для каждого условия и seed берётся качество в конце интервала минус качество в его начале. Так видно, когда именно меняется направление, без сравнения с финальной эпохой.</p>
@@STAGES@@
@@STAGE_NOTES@@
<p class="note">Значения — изменения за весь интервал, а не в расчёте на одну эпоху. Например, 2→5 и 10→20 имеют разную длительность. П.п. — процентные пункты: изменение AUC на 0,01 равно 1 п.п.</p></section>
<section class="section"><h2>2. Метод и архитектура: от начала к эпохам 5 и 30</h2>
@@METHODS@@
@@METHOD_NOTES@@
<p class="note">Среднее группы не означает одинакового эффекта во всех категориях и режимах аугментации. Каждая исходная конфигурация по-прежнему имеет только три повторения.</p></section>
@@CASES@@
<section class="section"><h2>Как читать разброс и статистическую значимость</h2>
@@NOTES@@
@@SIGNIFICANCE@@
<div class="callout">Одинаковый знак показывает повторяемость направления; небольшой SD — близкую величину изменений. Ни совпавший знак, ни среднее по многим условиям сами по себе не дают дополнительных независимых повторений.</div>
<p class="note">Соседние интервалы используют общие checkpoints и зависят друг от друга. Два классификатора и несколько метрик также не увеличивают число seed.</p></section>
@@CUMULATIVE@@
<section class="section"><details><summary>Исходные таблицы и определения</summary><ul>@@DOWNLOADS@@</ul>
<p class="note">SD — выборочное стандартное отклонение парных изменений по трём seed. 95% t-интервал среднего равен mean ± t × SD / √3; при n = 3 его половина составляет 2,484 × SD. Нормальность разностей по трём наблюдениям проверить нельзя. <a href="https://itl.nist.gov/div898/handbook/prc/section3/prc312.htm">NIST: парные доверительные интервалы</a>.</p>
<p class="note">Это анализ сохранённых результатов. Исходные обучения, данные и предыдущие отчёты не менялись; новые обучения не запускались.</p></details></section>
<footer>VisA v2 · отдельная сводка ранних этапов обучения. Статический автономный HTML без полного набора строк экспериментов.</footer>
</main></body></html>'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    source = args.data or args.directory / "early_summary.json"
    build(args.directory, json.loads(source.read_text()))
