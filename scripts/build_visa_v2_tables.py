"""Build a single offline HTML file containing VisA v2 calculation tables.

The opening overview includes the user's requested short findings summary.
Training recommendations remain outside the file.
Prepared statistical records are compacted into column/row arrays and filtered
in the browser. This builder does not run statistical tests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reports/visa-v2-statistics-2026-11-14/tables/data_table_payload_with_interactions.json"
DEFAULT_OUTPUT = ROOT / "outputs/visa-v2-stats-20261114/visa_v2_statistics.html"


def pack(records):
    if isinstance(records, dict) and "columns" in records and "rows" in records:
        return records
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty list of records or a columns/rows object")
    columns = list(records[0])
    if not all(set(row) == set(columns) for row in records):
        raise ValueError("Record columns are inconsistent")
    return {"columns": columns, "rows": [[row[key] for key in columns] for row in records]}


def normalize(data):
    if "columns" in data and "rows" in data and "summary_columns" in data:
        kind_index = data["columns"].index("kind")
        datasets = {
            kind: {"columns": data["columns"], "rows": [row for row in data["rows"] if row[kind_index] == kind]}
            for kind in ["duration", "augmentation", "interaction"]
        }
        datasets["summary"] = {"columns": data["summary_columns"], "rows": data["summary_rows"]}
        if sum(len(datasets[k]["rows"]) for k in ["duration", "augmentation", "interaction"]) != len(data["rows"]):
            raise ValueError("Unknown comparison kind")
        return {"meta": data.get("meta", {}), "datasets": datasets, "labels": data.get("labels", {})}
    source = data.get("datasets", data)
    datasets = {}
    aliases = {"duration": ["duration", "durations"], "augmentation": ["augmentation", "augmentations"], "interaction": ["interaction", "interactions"], "summary": ["summary", "summaries"]}
    for name, options in aliases.items():
        found = next((source[key] for key in options if key in source), None)
        if found is None:
            raise ValueError(f"Missing dataset: {name}")
        datasets[name] = pack(found)
    return {"meta": data.get("meta", {}), "datasets": datasets}


def build(source, output):
    data = normalize(json.loads(source.read_text()))
    overview_path = source.parent / "overview_payload.json"
    data["overview"] = json.loads(overview_path.read_text())
    data["overview"].setdefault("meta", {})["family_tests"] = data["meta"]["family_tests"]
    packed = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    page = TEMPLATE.replace("@@DATA@@", packed)
    if "@@" in page:
        raise ValueError("Unresolved template placeholder")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size,
                      "rows": {key: len(value["rows"]) for key, value in data["datasets"].items()}}, ensure_ascii=False))


TEMPLATE = r'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VisA v2 — обновлённые выводы и статистика — 14.11.2026</title>
<style>
:root{--ink:#243443;--muted:#5b6f7d;--line:#dbe3e8;--blue:#245d7c;--soft:#f2f6f8;--green:#216553;--red:#953a50;--amber:#875e19}*{box-sizing:border-box}body{margin:0;background:#f5f7f8;color:var(--ink);font:14px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1540px;margin:auto;padding:28px 26px 42px}h1{font-size:29px;line-height:1.2;letter-spacing:-.02em;margin:0 0 9px}.definition{font-size:13px;color:var(--muted);max-width:1160px;margin:0 0 19px}.tabs{display:flex;gap:6px;border-bottom:1px solid var(--line);margin:20px 0 0}button,select,input{font:inherit;color:var(--ink);border:1px solid #bdcbd5;background:white;border-radius:7px;padding:8px 10px;min-height:37px}button{cursor:pointer}button:hover{background:var(--soft)}button:disabled{opacity:.42;cursor:default}.tab{border:0;background:transparent;border-radius:7px 7px 0 0;padding:11px 17px}.tab.active{background:var(--blue);color:white}.panel{background:white;border:1px solid var(--line);border-top:0;border-radius:0 0 12px 12px;padding:19px}.filters{display:flex;flex-wrap:wrap;gap:11px;margin:0 0 15px}.filters label{display:flex;flex:1;min-width:145px;max-width:245px;flex-direction:column;gap:5px;font-size:12px;color:var(--muted);font-weight:650}.filters select,.filters input{width:100%;font-weight:400}.filters .wide{max-width:320px}.toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:17px 0 12px}.toolbar .actions{display:flex;align-items:center;gap:7px}.count{font-size:13px;color:var(--muted);margin:0}.table-scroll{overflow:auto;border:1px solid var(--line);border-radius:8px;max-height:680px}table{border-collapse:collapse;width:100%;font-size:13px}th{position:sticky;top:0;z-index:1;background:var(--soft);text-align:left;font-weight:650;padding:11px 10px;vertical-align:bottom;border-bottom:1px solid var(--line)}td{padding:10px;border-bottom:1px solid #e8eef2;vertical-align:middle}tbody tr:last-child td{border-bottom:0}tbody tr:hover{background:#fbfcfd}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.category{font-weight:600}.sub{display:block;font-size:11px;color:var(--muted);margin-top:2px}.tag{font-size:12px;line-height:1.35;white-space:nowrap}.tag.positive{color:var(--green)}.tag.negative{color:var(--red)}.tag.mixed{color:var(--amber)}.details-button{min-height:29px;padding:3px 8px;font-size:12px}.detail-row td{background:#f5f8fa;padding:16px}.detail-title{font-weight:650;margin-bottom:8px}.detail-grid{width:100%;font-size:12px}.detail-grid th{position:static;background:#eaf0f4}.detail-grid td{padding:8px;border-bottom:1px solid #dae4eb}.small-note{font-size:12px;color:var(--muted);margin:12px 0 0}.source{font-size:12px;color:var(--muted);margin-top:17px}details.legend{margin:14px 0 0;font-size:12px;color:var(--muted)}summary{cursor:pointer;font-weight:650}details.legend p{margin:7px 0}.empty{padding:35px;color:var(--muted);text-align:center}a{color:var(--blue)}@media(max-width:800px){main{padding:19px 12px 30px}.panel{padding:13px}.toolbar{align-items:flex-start;flex-direction:column}.filters label{min-width:135px;max-width:none}.tab{padding:10px 13px}h1{font-size:25px}.table-scroll{max-height:70vh}}@media print{body{background:white}main{padding:0}.filters,.actions,.tabs,.details-button{display:none}.panel{padding:0;border:0}.table-scroll{max-height:none;overflow:visible}th{position:static}}
.overview-heading{font-size:19px;line-height:1.3;margin:24px 0 9px}.overview-heading:first-of-type{margin-top:18px}.overview-table{max-height:none}.overview-table td:last-child{min-width:170px;max-width:300px}.overview-table td{padding:11px 12px}.overview-table th{position:static}.overview-intro{margin:0 0 15px;color:var(--muted);font-size:13px;max-width:1050px}.overview-probe{display:flex;align-items:center;gap:12px;font-size:13px}.overview-table .group{font-weight:600;white-space:nowrap}.overview-note{font-size:12px;color:var(--muted);max-width:1150px;margin:10px 0 0}
.overview-findings{max-width:1160px;margin:0 0 26px;padding:18px;background:#f4f8fa;border:1px solid var(--line);border-radius:10px}.overview-findings h2{font-size:21px;margin:0 0 8px}.overview-findings th{position:static}.overview-findings td:first-child{width:28%;min-width:170px;font-weight:600}.overview-findings .table-scroll{max-height:none;background:white}.overview-findings p{font-size:13px;margin:10px 0 0}.overview-findings .scope-note{color:var(--muted);margin:0 0 12px}
.comparison-controls{display:flex;flex-wrap:wrap;gap:18px;align-items:center;margin:0 0 16px;font-size:13px}.comparison-controls label{align-items:center;gap:8px}.comparison-controls label:not([hidden]){display:flex}.comparison-controls input[type=checkbox]{width:17px;height:17px;min-height:0;margin:0;padding:0}
</style></head><body><main>
<p style="display:inline-block;margin:0 0 14px;padding:7px 12px;border:1px solid #b6d8c9;border-radius:7px;background:#e8f4ee;color:#205840;font-size:13px;font-weight:650">Обновлённый отчёт · 14.11.2026 · 19 800 сравнений</p>
<h1>VisA v2. Выводы и статистика</h1>
<p class="definition">Δ — парное изменение метрики, п.п. SD — стандартное отклонение трёх Δ по seed. ДИ — доверительный интервал среднего. Полные значения каждой строки доступны по кнопке «Детали».</p>
<nav class="tabs" aria-label="Таблицы"><button class="tab active" data-tab="overview" aria-selected="true">Общий обзор</button><button class="tab" data-tab="summary" aria-selected="false">Сводки</button><button class="tab" data-tab="duration" aria-selected="false">Длительность</button><button class="tab" data-tab="augmentation" aria-selected="false">Аугментации</button></nav>
<section class="panel" id="overview-panel">
<section class="overview-findings" id="overview-findings" aria-labelledby="overview-findings-title">
<h2 id="overview-findings-title">Выводы с учётом аугментаций</h2>
<p class="scope-note">Выводы проверены отдельно для A0, Rotate и CutPaste. Сравнение динамики: ΔΔ = (Q100−Q10)добавка − (Q100−Q10)A0.</p>
<div class="table-scroll"><table><thead><tr><th>Вывод</th><th>Основание и границы вывода</th></tr></thead><tbody>
<tr><td>Поздний спад не объясняется только добавлением Rotate/CutPaste.</td><td>У SimCLR обеих архитектур и у ResNet + VICReg средние AUC/AP за 10→100 отрицательны во всех трёх режимах у обоих пробингов. Спад есть уже на A0. При этом A0 содержит crop и color jitter: контроля без преобразований здесь нет.</td></tr>
<tr><td>Добавки могут усиливать или ослаблять позднее ухудшение.</td><td>Для ViT + SimCLR Rotate и CutPaste усиливают среднее падение AP относительно A0; для ResNet + SimCLR CutPaste его ослабляет. Это совпадает у обоих пробингов. У ViT + VICReg знак AUC для Logistic Regression меняется между A0 (+0,54) и Rotate/CutPaste (−1,11/−1,57 п.п.). Поэтому вывод о длительности нужно формулировать с указанием режима аугментации.</td></tr>
<tr><td>Положительный вывод о reconstruction нельзя одинаково переносить на обе архитектуры.</td><td>У ResNet итог 0→100, усреднённый по трём режимам, положителен по AUC/AP у обоих пробингов и всех seed, охватывает 4 из 6 категорий; на A0 дополнительный прирост после эпохи 10 невелик. У ViT при усреднении режимов положительный поздний эффект Logistic Regression зависит от capsules; без неё AUC/AP меняются на −0,70/−1,05 п.п. У LightGBM такого переворота нет.</td></tr>
</tbody></table></div>
<p><strong>Небольшое среднее не позволяет признать эксперимент устойчивым:</strong> оно может получиться из взаимного погашения улучшений и ухудшений. В показанных ниже этапах типичный SD изменений AP составляет <strong>4,3–5,3 п.п.</strong> Для решения о повторах нужны согласие направления и разброс внутри выбранной конфигурации.</p>
<p class="overview-note">Это описание наблюдений по трём seed. После общей поправки Holm значимость различий не подтверждена; отсутствие значимости не доказывает равенство результатов.</p>
<details class="legend"><summary>Гипотеза: преобразования и признаки дефектов</summary>
<p>SimCLR/VICReg сближают представления двух независимо преобразованных версий. Crop может убрать дефект из одного вида, CutPaste — закрыть его или создать похожую на дефект структуру; для Rotate возможна потеря полезной информации об ориентации. Если эти различия несут метку дефекта, обучение инвариантности может снижать чувствительность к ним. Это гипотеза, а не установленная причина падения AUC/AP.</p>
<p>Reconstruction получает одну аугментированную картинку и восстанавливает её же до маскирования, включая CutPaste. Это другая задача. В оригинальной работе CutPaste модель различает исходные и изменённые изображения; у нас вставки используются внутри положительных пар. Для проверки механизма нужны контроль сохранности дефектов и сравнение с более щадящими crop/jitter, а не только дополнительные seed.</p>
<p>Основания гипотезы: <a href="https://openaccess.thecvf.com/content/CVPR2021/html/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.html">CutPaste (CVPR 2021)</a>; <a href="https://arxiv.org/abs/2208.07734">Yoo et al. (2022)</a>.</p></details>
</section>
<p class="overview-intro">Вся серия в трёх небольших таблицах. В числовых парах сначала AUC, затем AP; изменения и SD — в процентных пунктах. Плюс означает рост качества, минус — снижение.</p>
<label class="overview-probe">Пробинг <select id="overview-classifier"><option value="logistic_regression">Logistic Regression</option><option value="lightgbm">LightGBM</option></select></label>
<h2 class="overview-heading">1. Общая динамика обучения</h2>
<div class="table-scroll overview-table"><table><thead><tr><th>Этап</th><th class="num">Среднее Δ<br>AUC / AP</th><th class="num">Типичный SD<br>AUC / AP</th><th class="num">Направление повторилось 3/3<br>AUC / AP</th><th>Кратко</th></tr></thead><tbody id="overview-stages"></tbody></table></div>
<p class="overview-note">Каждая строка усредняет все 108 конфигураций. Этапы имеют разную длину: показано изменение за весь отрезок, а не скорость за эпоху.</p>
<h2 class="overview-heading">2. Архитектуры и методы: раннее обучение, продолжение и итог</h2>
<div class="table-scroll overview-table"><table><thead><tr><th>Группа</th><th class="num">0→10<br>AUC / AP</th><th class="num">10→100<br>AUC / AP</th><th class="num">0→100<br>AUC / AP</th><th class="num">Типичный SD 0→100<br>AUC / AP</th><th>Кратко</th></tr></thead><tbody id="overview-methods"></tbody></table></div>
<p class="overview-note">Каждая группа объединяет шесть категорий и три режима аугментации — 18 конфигураций с равными весами.</p>
<h2 class="overview-heading">3. Что дают дополнительные аугментации</h2>
<div class="table-scroll overview-table"><table><thead><tr><th>Добавка к A0</th><th class="num">Эпоха 10<br>Δ AUC / AP</th><th class="num">Эпоха 100<br>Δ AUC / AP</th><th class="num">Типичный SD на 100<br>AUC / AP</th><th>Кратко</th></tr></thead><tbody id="overview-augmentations"></tbody></table></div>
<p class="overview-note">Δ здесь — качество с Rotate/CutPaste минус A0 на той же эпохе. Каждая строка усредняет шесть категорий и три SSL-метода. A0 уже включает crop и color jitter.</p>
<details class="legend"><summary>Как читать обобщение</summary><p>Типичный SD — медиана разброса трёх парных изменений внутри отдельных конфигураций. Он показывает, насколько расходятся повторы обычной конфигурации, и не равен разбросу общего среднего.</p><p>«Направление повторилось 3/3» — доля конфигураций, где все три seed дали один знак: либо рост, либо снижение. Это не доля улучшений и не проверка значимости.</p><p>Средние рассчитаны по фиксированной сетке с равными весами. Категории и методы не считаются независимыми seed: повторов по-прежнему три. Общий эффект может скрывать противоположные изменения отдельных конфигураций.</p><p>Макросводки описательные. <span id="overview-family"></span> Тесты для отдельных конфигураций и подробные расчёты, включая влияние аугментации на динамику, доступны в остальных вкладках.</p></details>
</section>
<section class="panel" id="detail-panel" hidden>
<div class="comparison-controls"><label id="summary-kind-label">Сравнение <select id="summary-kind"><option value="duration">Обучение</option><option value="augmentation">Аугментации</option><option value="duration_by_augmentation">Обучение по аугментациям</option><option value="interaction">Влияние добавки на динамику</option></select></label><label id="augmentation-kind-label">Сравнение <select id="augmentation-kind"><option value="augmentation">Эффект на эпохе</option><option value="interaction">Влияние на продолжение 10→100</option></select></label><label id="all-periods-label"><input id="all-periods" type="checkbox">Все интервалы (15)</label></div>
<div class="filters"><label id="period-label">Интервал<select id="period"></select></label><label>Пробинг<select id="classifier"></select></label><label>Метрика<select id="metric"><option value="pair">ROC-AUC и AP</option><option value="roc_auc">ROC-AUC</option><option value="average_precision">AP</option><option value="precision_anomaly">Precision</option><option value="recall_anomaly">Recall</option><option value="f1_anomaly">F1</option></select></label><label>Категория<select id="category"></select></label><label>Энкодер<select id="architecture"></select></label><label>SSL-метод<select id="method"></select></label><label>Аугментация<select id="augmentation"></select></label></div>
<div class="filters"><label class="wide">Поиск<input id="search" placeholder="Название условия"></label><label>Сортировка<select id="sort"><option value="name">По условию</option><option value="sd_desc">SD: сначала больший</option><option value="sd_asc">SD: сначала меньший</option><option value="delta_desc">Δ: сначала больший</option><option value="delta_asc">Δ: сначала меньший</option></select></label><label>Строк на странице<select id="size"><option>20</option><option>50</option><option>100</option></select></label></div>
<div class="toolbar"><p class="count" id="counter" aria-live="polite"></p><div class="actions"><button id="csv">CSV выбранных строк</button><button id="previous">← Назад</button><button id="next">Далее →</button></div></div>
<div class="table-scroll"><table id="table"><thead id="head"></thead><tbody id="body"></tbody></table></div>
<p class="small-note" id="scope"></p>
<details class="legend"><summary>Обозначения и расчёт</summary><p>«Рост 3/3» и «Снижение 3/3» означают одинаковый знак трёх парных изменений. «Разный знак / нули» означает, что ненулевое направление не совпало во всех seed. Эти метки не являются тестом значимости.</p><p>95% t-ДИ: mean ± t × SD / √3. При n = 3 предположение нормальности разностей проверить нельзя. Нулевой SD в трёх наблюдениях не доказывает отсутствие неопределённости; недоступные интервалы и p показаны знаком «—».</p><p>A0 содержит crop и color jitter. «Эффект на эпохе»: A0 + Rotate/CutPaste минус A0 на одной эпохе. «Влияние на продолжение»: ΔΔ = (Q100 − Q10) с добавкой − (Q100 − Q10) A0. Положительное ΔΔ означает более благоприятную динамику с добавкой, а не обязательно более высокое конечное качество.</p><p id="family"></p><p>Фильтры меняют только отображение. Поправка Holm не пересчитывается по выбранным строкам. Для пары AUC/AP сортировка по SD использует больший SD двух метрик; сортировка по Δ использует ROC-AUC. Все расчётные значения сохранены в файле как снимок.</p></details>
</section><p class="source" id="source"></p>
</main><script>
const payload=@@DATA@@;
const names={resnet18:'ResNet-18',deit_tiny:'ViT-Tiny',baseline:'A0',rotate:'Rotate',cutpaste:'CutPaste',simclr:'SimCLR',vicreg:'VICReg',masked_reconstruction:'Reconstruction',logistic_regression:'Logistic regression',lightgbm:'LightGBM',roc_auc:'ROC-AUC',average_precision:'AP',precision_anomaly:'Precision',recall_anomaly:'Recall',f1_anomaly:'F1',all:'Все'};
const el=id=>document.getElementById(id);const label=x=>names[x]||x||'';const escape=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const n=x=>x===null||x===undefined||x===''?null:Number(x);const first=(r,keys,otherwise=null)=>{for(const k of keys)if(r[k]!==undefined&&r[k]!==null&&r[k]!=='')return r[k];return otherwise;};
const fmt=(x,signed=false)=>x===null||x===undefined||!Number.isFinite(Number(x))?'—':(signed&&Number(x)>=0?'+':'')+Number(x).toFixed(2).replace('-', '−');const pformat=x=>x===null||x===undefined||!Number.isFinite(Number(x))?'—':Number(x)<.001?Number(x).toExponential(2):Number(x).toFixed(3);
function canonical(r,type){const result={...r};result.mean=n(first(r,['mean_delta_pp','macro_delta_pp','mean_pp','mean']));result.sd=n(first(r,['sd_delta_pp','seed_sd_of_macro_pp','sd_macro_pp','sd_pp','sd','std']));result.lo=n(first(r,['ci95_low_pp','ci95_low','ci95_lower_pp','ci_low']));result.hi=n(first(r,['ci95_high_pp','ci95_high','ci95_upper_pp','ci_high']));result.p=n(first(r,['p_t','p_nominal','p_value']));result.holm=n(first(r,['p_holm_all19800','p_t_holm_all19800','p_holm_all19080','p_t_holm_all19080','p_holm_expanded','p_t_holm_expanded','p_t_holm_all16920','p_holm_all16920','p_holm_global','p_t_holm_all5040','p_holm']));result.n=n(first(r,['n','n_seeds','seeds'],3));result.positive=n(first(r,['positive','n_positive']));result.negative=n(first(r,['negative','n_negative']));result.zero=n(first(r,['zero','n_zero','tied']));result.early=n(first(r,['early_epoch','start_epoch','epoch_start','from_epoch']));result.late=n(first(r,['late_epoch','end_epoch','epoch_end','epoch','to_epoch']));result.period=r.kind==='augmentation'?String(result.late):`${result.early}→${result.late}`;result.annotation=first(r,['interpretation','annotation','row_tag','short_interpretation','sign_note'],'');result.seedValues=[6101,6102,6103].map(seed=>n(first(r,[`delta_seed_${seed}_pp`,`delta_${seed}_pp`,`seed_${seed}_delta_pp`,`seed_${seed}`,`delta${seed}`])));return result;}
const datasets=Object.fromEntries(Object.entries(payload.datasets).map(([type,d])=>[type,d.rows.map(values=>canonical(Object.fromEntries(d.columns.map((key,i)=>[key,values[i]])),type))]));
let tab='overview',page=0,visible=[],filteredGroups=[];const tabs=Array.from(document.querySelectorAll('[data-tab]'));
function options(id,values,all=true,preferred=null){const select=el(id);const old=preferred??select.value;select.innerHTML=(all?'<option value="all">Все</option>':'')+values.map(v=>`<option value="${escape(v)}">${escape(label(v))}</option>`).join('');select.value=values.includes(old)?old:all?'all':values[0]??'';}
const mainPeriods=['0→10','10→100','30→100','0→100'];
const periodLabels={'0→10':'0→10 — раннее обучение','10→100':'10→100 — продолжение обучения','30→100':'30→100 — поздний участок','0→100':'0→100 — итог'};
function comparisonKind(){return tab==='summary'?el('summary-kind').value:tab==='augmentation'?el('augmentation-kind').value:'duration';}
function comparisonData(){const kind=comparisonKind();return datasets[tab==='summary'?'summary':kind].filter(r=>r.kind===kind);}
function syncFilters(){const data=comparisonData().filter(r=>r.period===el('period').value);for(const key of ['classifier','category','architecture','method','augmentation']){const values=[...new Set(data.map(r=>r[key]).filter(x=>x!==undefined&&x!==null&&x!==''))];options(key,values.filter(x=>x!=='all'),key!=='classifier',key==='classifier'&&(el(key).value===''||el(key).value==='all')?'logistic_regression':null);el(key).disabled=values.length===0||values.length===1&&values[0]==='all';}}
function setup(preferredPeriod=null){
 el('overview-panel').hidden=tab!=='overview';el('detail-panel').hidden=tab==='overview';if(tab==='overview'){renderOverview();return;}
 const kind=comparisonKind(),data=comparisonData(),isAugmentation=kind==='augmentation';
 el('summary-kind-label').hidden=tab!=='summary';el('augmentation-kind-label').hidden=tab!=='augmentation';el('all-periods-label').hidden=kind!=='duration';
 const available=[...new Set(data.map(r=>r.period))];
 const periods=isAugmentation?available.sort((a,b)=>Number(a)-Number(b)):kind==='interaction'?['10→100']:kind==='duration_by_augmentation'?['0→10','10→100']:el('all-periods').checked?available.sort((a,b)=>{const A=a.split('→').map(Number),B=b.split('→').map(Number);return A[1]-B[1]||A[0]-B[0];}):mainPeriods.filter(p=>available.includes(p));
 const defaultPeriod=isAugmentation?'100':kind==='interaction'?'10→100':'0→10',selection=preferredPeriod??defaultPeriod;
 el('period').innerHTML=periods.map(p=>`<option value="${escape(p)}">${escape(!isAugmentation&&kind!=='interaction'&&!el('all-periods').checked?(periodLabels[p]??p):p)}</option>`).join('');
 el('period').value=periods.includes(selection)?selection:periods.includes(defaultPeriod)?defaultPeriod:periods[0];
 syncFilters();el('period-label').firstChild.textContent=isAugmentation?'Эпоха':'Интервал';
 const meaning=kind==='interaction'?'ΔΔ = (Q100 − Q10) с добавкой − (Q100 − Q10) A0, для того же seed. Плюс означает более благоприятную динамику с добавкой; он не гарантирует более высокое качество на эпохе 100.':isAugmentation?'Δ = A0 с добавкой минус A0 на одной эпохе.':'Δ = качество в конце интервала минус качество в начале, для того же seed.';
 el('scope').textContent=meaning+(tab==='summary'?' Mean — среднее по фиксированной сетке. В основной таблице — медиана SD отдельных условий; в деталях — SD и ДИ трёх средних по seed. Число условий не является числом независимых повторов.':'');
 page=0;render();
}
el('summary-kind').addEventListener('input',()=>setup());
el('augmentation-kind').addEventListener('input',()=>setup());
el('all-periods').addEventListener('input',()=>setup(el('period').value));
function overviewPair(rows,field='mean',signed=true){return ['roc_auc','average_precision'].map(metric=>{const row=rows.find(r=>r.metric===metric);return fmt(row?.[field],signed);}).join(' / ');}
function overviewDirection(rows){const values=['roc_auc','average_precision'].map(metric=>rows.find(r=>r.metric===metric)?.mean);if(values.some(v=>v===undefined))throw Error('Incomplete overview metric pair');if(values.every(v=>v>1e-10))return 'Рост обеих метрик';if(values.every(v=>v< -1e-10))return 'Снижение обеих метрик';if(values.every(v=>Math.abs(v)<=1e-10))return 'Средние изменения нулевые';return 'Метрики меняются по-разному';}
function overviewAugNote(early,late){const start=overviewDirection(early),end=overviewDirection(late);if(start==='Рост обеих метрик'&&end==='Снижение обеих метрик')return 'Ранний плюс сменяется минусом';return 'На эпохе 100: '+end.toLowerCase();}
function renderOverview(){
 const probe=el('overview-classifier').value;
 const pick=(section,conditions)=>payload.overview[section].filter(r=>r.classifier===probe&&Object.entries(conditions).every(([k,v])=>r[k]===v));
 const numberCell=(rows,field='mean',signed=true)=>`<td class="num">${overviewPair(rows,field,signed)}</td>`;
 const stages=[[0,5,'Старт'],[5,10,'Раннее обучение'],[10,20,'Продолжение'],[20,30,'До эпохи 30'],[30,100,'Длительное обучение']];
 el('overview-stages').innerHTML=stages.map(([from,to,title])=>{const rows=pick('stage',{from_epoch:from,to_epoch:to});const shares=['roc_auc','average_precision'].map(metric=>{const r=rows.find(r=>r.metric===metric);return `${Math.round(100*r.unanimous/r.n_conditions)}%`;}).join(' / ');return `<tr><td class="group">${title}<span class="sub">${from}→${to}</span></td>${numberCell(rows)}${numberCell(rows,'median_cell_sd',false)}<td class="num">${shares}</td><td>В среднем: ${overviewDirection(rows).toLowerCase()}</td></tr>`;}).join('');
 el('overview-methods').innerHTML=['resnet18','deit_tiny'].flatMap(architecture=>['simclr','vicreg','masked_reconstruction'].map(method=>{const group={architecture,method},early=pick('method',{...group,from_epoch:0,to_epoch:10}),late=pick('method',{...group,from_epoch:10,to_epoch:100}),total=pick('method',{...group,from_epoch:0,to_epoch:100});return `<tr><td class="group">${label(architecture)}<span class="sub">${label(method)}</span></td>${numberCell(early)}${numberCell(late)}${numberCell(total)}${numberCell(total,'median_cell_sd',false)}<td>До 10: ${overviewDirection(early).toLowerCase()}.<br>После 10: ${overviewDirection(late).toLowerCase()}.</td></tr>`;})).join('');
 el('overview-augmentations').innerHTML=['resnet18','deit_tiny'].flatMap(architecture=>['rotate','cutpaste'].map(augmentation=>{const group={architecture,augmentation},early=pick('augmentation',{...group,to_epoch:10}),late=pick('augmentation',{...group,to_epoch:100});return `<tr><td class="group">${label(architecture)}<span class="sub">${label(augmentation)} − A0</span></td>${numberCell(early)}${numberCell(late)}${numberCell(late,'median_cell_sd',false)}<td>${overviewAugNote(early,late)}</td></tr>`;})).join('');
}
el('overview-classifier').addEventListener('input',renderOverview);
function keyOf(r){return [r.category??'',r.architecture??'',r.method??'',r.augmentation??'',r.classifier??'',r.period,r.view??r.level??''].join('|');}
function tag(r){if(!r)return '';if(r.annotation)return escape(r.annotation);const interaction=r.kind==='interaction',positive=interaction?'Динамика лучше':'Рост',negative=interaction?'Динамика хуже':'Снижение',direction=r.mean>0?(interaction?'В среднем динамика лучше':'В среднем рост'):r.mean<0?(interaction?'В среднем динамика хуже':'В среднем снижение'):`Среднее ${interaction?'ΔΔ':'Δ'} = 0`;if(tab==='summary'&&r.n_conditions!==undefined)return `<span class="tag">${direction}<br>${interaction?'Лучше':'Рост'}: ${r.mean_positive}/${r.n_conditions}; ${interaction?'хуже':'снижение'}: ${r.mean_negative}/${r.n_conditions}</span>`;if(r.positive===3)return `<span class="tag positive">${positive} 3/3</span>`;if(r.negative===3)return `<span class="tag negative">${negative} 3/3</span>`;if(r.zero===3)return `<span class="tag">Нулевые ${interaction?'ΔΔ':'Δ'}</span>`;return `<span class="tag mixed">${direction}<br>Разный знак / нули</span>`;}
function inputFilter(r){if(r.kind!==comparisonKind()||r.period!==el('period').value)return false;if(!['classifier','category','architecture','method','augmentation'].every(k=>el(k).disabled||el(k).value==='all'||String(r[k])===el(k).value))return false;const metric=el('metric').value;if(metric==='pair'?!['roc_auc','average_precision'].includes(r.metric):r.metric!==metric)return false;const q=el('search').value.trim().toLowerCase().split(/\s+/).filter(Boolean);const content=[r.category,r.architecture,r.method,r.augmentation,label(r.architecture),label(r.method),label(r.augmentation)].join(' ').toLowerCase();return q.every(t=>content.includes(t));}
function valueCell(value,signed=false){return `<td class="num" title="${escape(value??'Недоступно')}">${fmt(value,signed)}</td>`;}
function idCells(r){return `<td class="category">${escape(!r.category||r.category==='all'?'Все категории':r.category)}</td><td>${escape(label(r.architecture))}</td><td>${escape(r.method==='all'?'Все методы':label(r.method))}</td><td>${escape(!r.augmentation||r.augmentation==='all'?'Все режимы':label(r.augmentation))}</td>`;}
function columnCount(){return 10;}
function interval(r){return r.lo===null||r.hi===null?'—':`[${fmt(r.lo,true)}; ${fmt(r.hi,true)}]`;}
function render(){
 const metric=el('metric').value,aggregate=tab==='summary',symbol=comparisonKind()==='interaction'?'ΔΔ':'Δ';
 const records=comparisonData().filter(inputFilter),grouped=new Map();
 for(const r of records){const key=keyOf(r);if(!grouped.has(key))grouped.set(key,{base:r,metrics:{}});if(grouped.get(key).metrics[r.metric])throw Error('Duplicate metric in presentation group');grouped.get(key).metrics[r.metric]=r;}
 filteredGroups=[...grouped.values()];
 const spread=r=>aggregate?r.median_cell_sd:r.sd;const measure=g=>metric==='pair'?Math.max(...Object.values(g.metrics).map(r=>spread(r)??-Infinity)):spread(g.base)??-Infinity;
 const change=g=>(g.metrics.roc_auc??g.base).mean??-Infinity,order=el('sort').value;
 filteredGroups.sort((a,b)=>order==='sd_desc'?measure(b)-measure(a):order==='sd_asc'?measure(a)-measure(b):order==='delta_desc'?change(b)-change(a):order==='delta_asc'?change(a)-change(b):keyOf(a.base).localeCompare(keyOf(b.base)));
 const size=Number(el('size').value)||20;page=Math.max(0,Math.min(page,Math.ceil(filteredGroups.length/size)-1));visible=filteredGroups.slice(page*size,(page+1)*size);
 el('counter').textContent=filteredGroups.length?`${page*size+1}–${Math.min((page+1)*size,filteredGroups.length)} из ${filteredGroups.length} строк`:'Нет строк по выбранным фильтрам';el('previous').disabled=page===0;el('next').disabled=(page+1)*size>=filteredGroups.length;
 const common='<th>Категория</th><th>Энкодер</th><th>SSL</th><th>Аугментация</th>';
 const pairHead='<th class="num">'+symbol+' AUC, п.п.</th><th class="num">'+(aggregate?'Медиана SD<br>AUC по условиям':'SD AUC')+'</th><th class="num">'+symbol+' AP, п.п.</th><th class="num">'+(aggregate?'Медиана SD<br>AP по условиям':'SD AP')+'</th><th>Кратко</th>';
 const singleHead='<th class="num">'+symbol+', п.п.</th><th class="num">SD '+(aggregate?'среднего':symbol)+', п.п.</th>'+(aggregate?'<th class="num">Медиана SD<br>по условиям</th>':'')+'<th class="num">95% ДИ, п.п.</th>'+(aggregate?'':'<th class="num">p Holm</th>')+'<th>Кратко</th>';
 el('head').innerHTML='<tr>'+common+(metric==='pair'?pairHead:singleHead)+'<th></th></tr>';
 el('body').innerHTML=visible.map((g,i)=>{const r=g.base,a=g.metrics.roc_auc,b=g.metrics.average_precision;let cells;
  if(metric==='pair')cells=valueCell(a?.mean,true)+valueCell(aggregate?a?.median_cell_sd:a?.sd)+valueCell(b?.mean,true)+valueCell(aggregate?b?.median_cell_sd:b?.sd)+`<td><span class="sub">AUC: ${tag(a)}</span><span class="sub">AP: ${tag(b)}</span></td>`;
  else cells=valueCell(r.mean,true)+valueCell(r.sd)+(aggregate?valueCell(r.median_cell_sd):'')+`<td class="num">${interval(r)}</td>`+(aggregate?'':`<td class="num">${pformat(r.holm)}</td>`)+`<td>${tag(r)}</td>`;
  return `<tr>${idCells(r)}${cells}<td><button class="details-button" data-detail="${i}" aria-expanded="false">Детали</button></td></tr>`;
 }).join('')||`<tr><td colspan="${columnCount()}" class="empty">Нет строк</td></tr>`;
}
function detailHtml(index){
 const group=visible[index],rows=Object.values(group.metrics),aggregate=tab==='summary',symbol=comparisonKind()==='interaction'?'ΔΔ':'Δ';
 const tail=aggregate?'<th>Условий</th><th>Медиана SD</th><th>Среднее +/−</th><th>Все seed +/−</th>':'<th>p t</th><th>p Holm</th><th>p sign-flip</th>';
 return `<div class="detail-title">${escape([group.base.category,label(group.base.architecture),label(group.base.method),label(group.base.augmentation),group.base.period].filter(Boolean).join(' / '))}</div><table class="detail-grid"><thead><tr><th>Метрика</th><th>n seed</th><th>${symbol} 6101</th><th>${symbol} 6102</th><th>${symbol} 6103</th><th>Mean</th><th>SD</th><th>95% ДИ</th><th>Seed +/−/0</th>${tail}</tr></thead><tbody>${rows.map(r=>`<tr><td>${escape(label(r.metric))}</td><td class="num">${r.n??'—'}</td>${r.seedValues.map(v=>valueCell(v,true)).join('')}${valueCell(r.mean,true)}${valueCell(r.sd)}<td class="num">${interval(r)}</td><td class="num">${r.positive??'—'}/${r.negative??'—'}/${r.zero??'—'}</td>`+(aggregate?`<td class="num">${r.n_conditions}</td>${valueCell(r.median_cell_sd)}<td class="num">${r.mean_positive}/${r.mean_negative}</td><td class="num">${r.unanimous_positive}/${r.unanimous_negative}</td>`:`<td class="num">${pformat(r.p)}</td><td class="num">${pformat(r.holm)}</td><td class="num">${pformat(n(r.p_signflip))}</td>`)+`</tr>`).join('')}</tbody></table>`;
}
el('body').addEventListener('click',event=>{const button=event.target.closest('[data-detail]');if(!button)return;const row=button.closest('tr');if(button.getAttribute('aria-expanded')==='true'){row.nextElementSibling.remove();button.setAttribute('aria-expanded','false');return;}const detail=document.createElement('tr');detail.className='detail-row';detail.innerHTML=`<td colspan="${columnCount()}">${detailHtml(Number(button.dataset.detail))}</td>`;row.after(detail);button.setAttribute('aria-expanded','true');});
for(const id of ['period','classifier','metric','category','architecture','method','augmentation','search','sort','size'])el(id).addEventListener('input',()=>{if(id==='period')syncFilters();page=0;render();});el('previous').addEventListener('click',()=>{page--;render();});el('next').addEventListener('click',()=>{page++;render();});for(const button of tabs)button.addEventListener('click',()=>{tab=button.dataset.tab;for(const other of tabs){other.classList.toggle('active',other===button);other.setAttribute('aria-selected',String(other===button));}setup();});
el('csv').addEventListener('click',()=>{const records=filteredGroups.flatMap(g=>Object.values(g.metrics));if(!records.length)return;const cols=['kind','category','architecture','method','augmentation','classifier','period','metric','n','delta6101','delta6102','delta6103','mean','sd','lo','hi','positive','negative','zero','p','holm','p_signflip','n_conditions','median_cell_sd','mean_positive','mean_negative','unanimous_positive','unanimous_negative'];const cell=x=>'"'+String(x??'').replace(/"/g,'""')+'"';const text='\uFEFF'+[cols.map(cell).join(','),...records.map(r=>cols.map(k=>cell(r[k])).join(','))].join('\r\n');const url=URL.createObjectURL(new Blob([text],{type:'text/csv;charset=utf-8'}));const anchor=document.createElement('a');anchor.href=url;anchor.download=`visa_v2_${tab}_${el('period').value.replace('→','-')}.csv`;anchor.click();URL.revokeObjectURL(url);});
el('overview-family').textContent=`Общая поправка Holm охватывает ${payload.meta.family_tests.toLocaleString('ru-RU')} тестов.`;el('family').textContent=payload.meta.holm_family??payload.meta.family??`Поправка Holm для единой семьи из ${payload.meta.family_tests??19800} тестов. p — двусторонний парный t-тест; p sign-flip в деталях — точная смена знака.`;el('source').textContent=payload.meta.source_note??'Источник: сохранённые результаты VisA v2. Дата расчётов: 14.11.2026.';setup();
el('family').textContent+=' ДИ номинальные, рассчитаны по seed на фиксированной выборке; bootstrap объектов здесь не используется. В сводках сортировка по SD использует медиану SD отдельных условий.';
</script></body></html>'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.input, args.output)
