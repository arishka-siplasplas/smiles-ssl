"""Audit frozen Kaggle tables; export combined precision/AUC/rank trajectories."""
from __future__ import annotations

import argparse
import base64
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

METHODS = ['simclr', 'vicreg', 'masked_reconstruction']
LABELS = {'simclr': 'SimCLR', 'vicreg': 'VICReg', 'masked_reconstruction': 'Masked reconstruction'}
COLORS = {'simclr': '#2767B1', 'vicreg': '#D27621', 'masked_reconstruction': '#218370'}
CLASSIFIERS = ['logistic_regression', 'lightgbm']
CLF_LABELS = {'logistic_regression': 'Logistic regression', 'lightgbm': 'LightGBM'}
RANKS = ['RankMe', 'CER', 'CovER', 'NESum']
EPOCHS = [0, 1, 2, 5, 10, 20, 30]
SEEDS = [6101, 6102, 6103]
CATEGORIES = ['cashew', 'chewinggum', 'fryum']
FRACTIONS = [.05, .2]


def audit(q, r, loss, root):
    keys = ['category', 'fraction', 'method', 'seed', 'epoch', 'split']
    assert not q.duplicated(keys + ['classifier']).any()
    assert not r.duplicated(keys + ['normalization']).any()
    expected = set(itertools.product(CATEGORIES, FRACTIONS, METHODS, SEEDS, EPOCHS,
                                     ['probe_train', 'validation', 'test'], CLASSIFIERS))
    actual = set(q.loc[q.method != 'baseline', keys + ['classifier']].itertuples(index=False, name=None))
    assert actual == expected, f'Incomplete quality table: missing={len(expected-actual)} extra={len(actual-expected)}'
    expected_r = set(itertools.product(CATEGORIES, FRACTIONS, METHODS, SEEDS, EPOCHS,
                                      ['probe_train', 'validation', 'test'], ['raw', 'l2']))
    actual_r = set(r.loc[r.method != 'baseline', keys + ['normalization']].itertuples(index=False, name=None))
    assert actual_r == expected_r, 'Incomplete rank table'
    expected_loss = set(itertools.product(CATEGORIES, FRACTIONS, METHODS, SEEDS, range(1, 31)))
    actual_loss = set(loss[['category', 'fraction', 'method', 'seed', 'epoch']].itertuples(index=False, name=None))
    assert expected_loss == actual_loss, 'Incomplete loss table'
    assert loss.amp_skipped_steps.sum() == 0
    quality_columns = ['precision_anomaly', 'recall_anomaly', 'f1_anomaly', 'roc_auc', 'average_precision',
                       'precision_normal', 'recall_normal', 'f1_normal', 'threshold', 'prevalence']
    assert np.isfinite(q[quality_columns]).all().all()
    assert q[quality_columns].ge(-1e-12).all().all() and q[quality_columns].le(1 + 1e-12).all().all()
    assert np.isfinite(r[RANKS]).all().all()
    assert np.isfinite(loss.loss).all()
    baseline_checks = 0
    for frame, variant, values in [(q, 'classifier', quality_columns), (r, 'normalization', RANKS)]:
        for key, group in frame.groupby(['category', 'seed', 'split', variant]):
            original = group[group.method == 'baseline']
            assert len(original) == 1
            copies = group[(group.method != 'baseline') & (group.epoch == 0)]
            assert len(copies) == 6
            np.testing.assert_array_equal(copies[values].values,
                                          np.repeat(original[values].values, 6, axis=0))
            baseline_checks += 1
    val = q[q.split == 'validation']
    assert (val.recall_anomaly >= .5 - 1e-12).all()
    expected_support = {'cashew': (200, 40), 'chewinggum': (201, 40), 'fryum': (200, 40)}
    for cat, group in q[q.split == 'test'].groupby('category'):
        assert set(zip(group.support_normal, group.support_anomaly)) == {expected_support[cat]}
    checks = {'ssl_runs': 54, 'baseline_runs': 9, 'seeds': SEEDS, 'epochs': EPOCHS,
              'quality_rows': len(q), 'rank_rows': len(r), 'loss_rows': len(loss),
              'paired_baseline_groups_verified': baseline_checks,
              'amp_skipped_steps': int(loss.amp_skipped_steps.sum()),
              'test_support_fixed': True, 'all_validation_recall_at_least_0_5': True,
              'csv_sha256': {name: hashlib.sha256((root / 'analysis' / name).read_bytes()).hexdigest()
                             for name in ['quality.csv', 'ranks.csv', 'loss.csv']}}
    if (root / 'status.json').exists():
        status = json.loads((root / 'status.json').read_text())
        assert status['status'] == 'complete'
        assert all(j['status'] == 'complete' for j in status['baseline_jobs'] + status['train_jobs'])
        checks['kaggle_job_status_verified'] = True
    if (root / 'source_snapshot.json').exists():
        checks['training_commit'] = json.loads((root / 'source_snapshot.json').read_text())['git_commit']
    for name in ['environment.json', 'config.json', 'source_snapshot.json', 'status.json']:
        if (root / name).exists():
            checks.setdefault('metadata_sha256', {})[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return checks


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
                         'axes.labelsize': 10, 'axes.spines.right': False, 'axes.spines.top': False,
                         'axes.edgecolor': '#AFBAC5', 'axes.labelcolor': '#354456',
                         'xtick.color': '#576779', 'ytick.color': '#576779',
                         'axes.titleweight': 'semibold', 'figure.facecolor': '#FFFFFF',
                         'axes.facecolor': '#FFFFFF', 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})


def curves(ax, frame, metric, early=False, percent=False, baseline=True, legend=False):
    factor = 100 if percent else 1
    plot_frame = frame[frame.epoch <= 10] if early else frame
    for method in METHODS:
        group = plot_frame[plot_frame.method == method]
        stats = group.groupby('epoch')[metric].agg(['mean', 'std', 'count'])
        assert (stats['count'] == 3).all()
        mean, sd = stats['mean'].to_numpy() * factor, stats['std'].to_numpy() * factor
        ax.plot(stats.index, mean, color=COLORS[method], lw=2, marker='o', ms=3.8,
                label=LABELS[method], zorder=3)
        ax.fill_between(stats.index, mean - sd, mean + sd, color=COLORS[method], alpha=.13, lw=0)
    if baseline:
        # All six epoch-zero copies were audited; use exactly one method.
        initial = plot_frame[(plot_frame.method == METHODS[0]) & (plot_frame.epoch == 0)][metric].mean() * factor
        ax.axhline(initial, color='#596879', ls=(0, (4, 3)), lw=.9, alpha=.7, zorder=1)
    ax.set_xlim(-.4 if early else -.7, 10.4 if early else 30.7)
    ax.set_xticks([0, 1, 2, 5, 10] if early else [0, 5, 10, 20, 30])
    ax.set_xlabel('Эпоха SSL')
    ax.grid(axis='y', color='#D9E0E7', lw=.65, alpha=.85)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    if legend:
        ax.legend(frameon=False, fontsize=8)


def legend(fig, y=.9, include_baseline=True):
    handles = [Line2D([0], [0], color=COLORS[m], lw=2.5, label=LABELS[m]) for m in METHODS]
    if include_baseline:
        handles.append(Line2D([0], [0], color='#596879', ls=(0, (4, 3)), lw=1, label='Эпоха 0'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, y), frameon=False,
               ncol=len(handles), fontsize=10)


def save(fig, out, stem):
    paths = {}
    for extension in ['png', 'svg']:
        p = out / f'{stem}.{extension}'
        fig.savefig(p, dpi=180)
        paths[extension] = p.name
    plt.close(fig)
    return paths


def combined(q, r, cat, fraction, norm, early, out):
    quality = q[(q.category == cat) & (q.fraction == fraction)]
    ranks = r[(r.category == cat) & (r.fraction == fraction) & (r.normalization == norm)]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.3))
    fig.subplots_adjust(left=.055, right=.98, top=.79, bottom=.12, wspace=.30, hspace=.48)
    name = 'raw' if norm == 'raw' else 'L2-нормированные'
    window = 'эпохи 0–10' if early else 'эпохи 0–30'
    fig.suptitle(f'{cat} · {fraction:.0%} аномалий в SSL-train · {window}', y=.975, fontsize=20, weight='bold')
    fig.text(.5, .927, f'Фиксированный test · 3 seed · среднее ± SD · ранговые метрики: {name}',
             ha='center', fontsize=11, color='#546477')
    legend(fig, y=.895)
    for ax, (metric, clf) in zip(axes[0], itertools.product(['precision_anomaly', 'roc_auc'], CLASSIFIERS)):
        curves(ax, quality[quality.classifier == clf], metric, early, percent=True)
        ax.set_title(f'{"Precision" if metric == "precision_anomaly" else "ROC-AUC"} · {CLF_LABELS[clf]}', loc='left')
        ax.set_ylabel('%' if metric == 'precision_anomaly' else 'ROC-AUC × 100')
        # Honest full scale for precision; an explicitly labelled zoom for ROC-AUC.
        if metric == 'precision_anomaly':
            ax.set_ylim(-2, 102)
        else:
            ax.set_ylim({'cashew': 89, 'chewinggum': 48, 'fryum': 80}[cat], 100.5)
    for ax, metric in zip(axes[1], RANKS):
        curves(ax, ranks, metric, early)
        ax.set_title(metric, loc='left')
        ax.set_ylabel('Значение метрики')
        lo, hi = ax.get_ylim()
        ax.set_ylim(max(0, lo), hi)
    fig.text(.055, .046,
             'Precision: аномальный класс; порог выбран на validation при recall ≥ 0.5. '
             'Серый пунктир — среднее необученного энкодера.', fontsize=9, color='#546477')
    fig.text(.055, .022,
             'Полосы — стандартное отклонение по 3 seed, не доверительный интервал. '
             'Классификаторы всегда используют raw-признаки; raw/L2 меняет только нижний ряд.',
             fontsize=9, color='#546477')
    stem = f'{cat}_p{round(fraction*100):02d}_{norm}_{"early" if early else "full"}'
    return save(fig, out, stem)


def overview_quality(q, metric, out):
    fig, axes = plt.subplots(4, 3, figsize=(15.5, 13.2))
    fig.subplots_adjust(left=.07, right=.98, top=.87, bottom=.07, wspace=.25, hspace=.5)
    name = 'Precision аномального класса' if metric == 'precision_anomaly' else 'ROC-AUC'
    fig.suptitle(f'{name} по эпохам · все категории и оба классификатора', y=.985, fontsize=19, weight='bold')
    fig.text(.5, .953, 'Фиксированный test · среднее ± SD по 3 seed · каждая категория анализируется отдельно',
             ha='center', fontsize=10.5, color='#546477')
    legend(fig, y=.935)
    for row, (fraction, clf) in enumerate(itertools.product(FRACTIONS, CLASSIFIERS)):
        for col, cat in enumerate(CATEGORIES):
            ax = axes[row, col]
            values = q[(q.category == cat) & (q.fraction == fraction) & (q.classifier == clf)]
            curves(ax, values, metric, percent=True)
            ax.set_title(f'{cat} · {fraction:.0%} · {CLF_LABELS[clf]}', loc='left', fontsize=10.5)
            ax.set_ylim(-2, 102) if metric == 'precision_anomaly' else ax.set_ylim(45, 100.5)
            ax.set_ylabel('%' if metric == 'precision_anomaly' else 'ROC-AUC × 100')
    fig.text(.07, .025, 'Пунктир — эпоха 0. Порог precision выбран только на validation при recall ≥ 0.5. '
             'У ROC-AUC шкала начинается с 45.', fontsize=9, color='#546477')
    return save(fig, out, f'overview_{metric}')


def overview_ranks(r, norm, out):
    fig, axes = plt.subplots(4, 6, figsize=(23, 12))
    fig.subplots_adjust(left=.05, right=.985, top=.86, bottom=.07, wspace=.35, hspace=.45)
    fig.suptitle(f'Спектр эмбеддингов по эпохам · {"raw" if norm == "raw" else "L2-нормированные признаки"}',
                 y=.98, fontsize=22, weight='bold')
    fig.text(.5, .94, 'Фиксированный test · среднее ± SD по 3 seed · шкалы внутри каждого ряда одинаковые',
             ha='center', color='#546477', fontsize=11)
    legend(fig, y=.917)
    for col, (cat, frac) in enumerate(itertools.product(CATEGORIES, FRACTIONS)):
        data = r[(r.category == cat) & (r.fraction == frac) & (r.normalization == norm)]
        for row, rank in enumerate(RANKS):
            ax = axes[row, col]
            curves(ax, data, rank)
            if row == 0:
                ax.set_title(f'{cat} · {frac:.0%}', fontsize=11)
            if col == 0:
                ax.set_ylabel(rank, fontsize=12, weight='bold')
    for row in range(4):
        low = max(0, min(ax.get_ylim()[0] for ax in axes[row]))
        high = max(ax.get_ylim()[1] for ax in axes[row])
        for ax in axes[row]:
            ax.set_ylim(low, high)
    fig.text(.05, .022, 'RankMe — нецентрированный спектр; CER — центрированный; CovER — ковариационный; '
             'NESum = сумма собственных значений / максимальное. L2 выполняется до центрирования.',
             fontsize=10, color='#546477')
    return save(fig, out, f'overview_ranks_{norm}')


def pair_tables(q, r, out):
    keys = ['category', 'fraction', 'method', 'classifier', 'seed']
    metrics = ['precision_anomaly', 'roc_auc', 'recall_anomaly', 'average_precision']
    records = []
    for key, values in q.groupby(keys):
        values = values.set_index('epoch')
        for early in [0, 1, 5, 10]:
            for metric in metrics:
                records.append({**dict(zip(keys, key)), 'early_epoch': early, 'late_epoch': 30,
                                'metric': metric, 'early': values.loc[early, metric], 'late': values.loc[30, metric],
                                'delta_pp': 100 * (values.loc[30, metric] - values.loc[early, metric])})
    paired = pd.DataFrame(records)
    paired.to_csv(out / 'paired_quality_deltas.csv', index=False)
    summary = paired.groupby([k for k in keys if k != 'seed'] + ['early_epoch', 'late_epoch', 'metric']).agg(
        early_mean=('early', 'mean'), late_mean=('late', 'mean'), delta_mean_pp=('delta_pp', 'mean'),
        delta_sd_pp=('delta_pp', 'std'), seeds_declined=('delta_pp', lambda x: int((x < 0).sum())),
        n=('seed', 'nunique')).reset_index()
    summary.to_csv(out / 'paired_quality_summary.csv', index=False)
    for frame, variant, values in [(q, 'classifier', metrics), (r, 'normalization', RANKS)]:
        group = ['category', 'fraction', 'method', variant, 'epoch']
        frame.groupby(group)[values].agg(['mean', 'std', 'count']).to_csv(out / f'{variant}_mean_sd.csv')
    return summary


def html_report(out, plots, overview, q, r, checks, summary):
    # SVG assets are embedded so the HTML remains usable as one standalone file.
    data = {key: {**item, 'image': 'data:image/svg+xml;base64,' +
                 base64.b64encode((out / item['svg']).read_bytes()).decode()} for key, item in plots.items()}
    overview_links = ''.join(f'<a href="{item["png"]}">{label} · PNG</a><a href="{item["svg"]}">SVG</a>'
                             for label, item in overview)
    evidence = []
    for cat, method, clf, fraction in [('chewinggum', 'simclr', 'lightgbm', .05),
                                      ('fryum', 'masked_reconstruction', 'logistic_regression', .05)]:
        subset = summary[(summary.category == cat) & (summary.method == method) & (summary.classifier == clf)
                         & (summary.fraction == fraction) & (summary.early_epoch == 5)]
        values = {row.metric: row for row in subset.itertuples()}
        p, auc = values['precision_anomaly'], values['roc_auc']
        evidence.append(f'<li><b>{cat} · {LABELS[method]} · {CLF_LABELS[clf]} · 5%</b>: '
                        f'эпохи 5 → 30, precision {p.early_mean:.3f} → {p.late_mean:.3f}, '
                        f'ROC-AUC {auc.early_mean:.3f} → {auc.late_mean:.3f}. '
                        f'ROC-AUC снизился в {auc.seeds_declined} из 3 seed.</li>')
    example = r[(r.category == 'fryum') & (r.method == 'masked_reconstruction') &
                (r.fraction == .05) & (r.normalization == 'raw')].set_index(['seed', 'epoch'])
    assert all(example.loc[(seed, 30), rank] > example.loc[(seed, 5), rank] for seed in SEEDS for rank in RANKS)
    cer_early = example.xs(5, level='epoch').CER.mean()
    cer_late = example.xs(30, level='epoch').CER.mean()
    evidence.append(f'<li>В том же условии <b>fryum / masked reconstruction / 5%</b> все четыре raw-ранга '
                    f'увеличиваются с эпохи 5 до 30 в каждом из трёх seed. Например, средний CER: '
                    f'{cer_early:.1f} → {cer_late:.1f}. Рост рангов здесь сопровождается снижением качества.</li>')
    page = '''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>VisA · SSL, precision, ROC-AUC и спектр</title><style>
body{margin:0;background:#f3f6f9;color:#213146;font:16px/1.5 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:34px 24px}
h1{font-size:34px;line-height:1.15;margin:8px 0 16px}h2{font-size:23px}p{max-width:1100px}.tag{font-size:12px;letter-spacing:.15em;text-transform:uppercase;color:#426481;font-weight:700}
.card{background:white;border:1px solid #dde5ed;border-radius:16px;padding:22px;margin-top:20px}.controls{display:flex;flex-wrap:wrap;gap:16px}label{display:flex;flex-direction:column;font-size:13px;font-weight:650;gap:6px}
select{font:inherit;padding:9px 12px;border:1px solid #c5d1df;border-radius:7px;background:white;color:#213146}
.figure{width:100%;display:block;margin-top:14px}a{color:#2767B1;text-decoration:none}a:hover{text-decoration:underline}.links{display:flex;gap:14px;flex-wrap:wrap;font-size:14px}
.sub{color:#5b6c80;font-size:14px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px 12px;border-bottom:1px solid #e6ebf0;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}
li{margin:10px 0}.note{border-left:3px solid #d27621;padding-left:14px}details{margin-top:14px}summary{cursor:pointer;font-weight:650}
</style><main><div class="tag">VisA / длительность SSL / завершённая серия</div>
<h1>Качество классификации и спектр по эпохам</h1>
<p>54 SSL-запуска × 30 эпох, 9 baseline необученного ResNet-18. Три фиксированных seed: 6101, 6102, 6103.
Во всех графиках используется один и тот же официальный test для каждой категории. Эпоха 0 общая для методов и дисбалансов.</p>
<div class="card"><div class="controls">
<label>Категория<select id="category"><option>cashew</option><option>chewinggum</option><option selected>fryum</option></select></label>
<label>Аномалии в SSL-train<select id="fraction"><option value="05">5%</option><option value="20">20%</option></select></label>
<label>Признаки для рангов<select id="norm"><option value="raw">Raw</option><option value="l2">L2</option></select></label>
<label>Диапазон эпох<select id="window"><option value="full">0–30</option><option value="early">0–10, ранние изменения</option></select></label>
</div><img id="figure" class="figure" alt="Precision, ROC-AUC и четыре ранговые метрики по эпохам">
<div class="links"><a id="png" download>Скачать выбранный PNG</a><a id="svg" download>Скачать выбранный SVG</a></div>
<p class="sub">Полосы: среднее ± стандартное отклонение по 3 seed, не доверительный интервал. Пунктир: среднее на эпохе 0.
Выбор Raw/L2 меняет только ранговые метрики; оба классификатора обучались на raw-признаках.
Ось ROC-AUC в совмещённых графиках приближена и явно подписана; precision показан от 0 до 100%.</p>
<details><summary>Числа по каждому seed и recall выбранного условия</summary><div id="numbers" class="scroll"></div></details></div>
<div class="card"><h2>Обзор всех условий</h2><div class="links">OVERVIEW_LINKS</div></div>
<div class="card"><h2>Что видно в этой серии</h2><ul>EVIDENCE</ul>
<p>Падение качества при продолжении SSL наблюдается в отдельных условиях. Единой монотонной связи между ранговыми метриками и качеством для всех методов и категорий нет.
Эти примеры описательные: эпохи 5 и 30 были включены в протокол заранее, но категории в примерах выбраны после просмотра результатов.</p>
<p class="note">Precision зависит от порога: на каждой оценочной эпохе он выбирался заново только на validation, с ограничением validation recall ≥ 0.5.
Это не гарантирует recall ≥ 0.5 на test. ROC-AUC не зависит от выбранного порога. Для интерпретации precision рядом доступны recall и AP.</p></div>
<div class="card"><h2>Определения и данные</h2><p>RankMe — энтропийный ранг нецентрированных сингулярных значений;
CER — центрированных; CovER — спектра ковариации; NESum — сумма собственных значений / максимальное.
Для L2-нормировки каждая строка нормируется до центрирования. Значения не делятся на размерность.</p>
<div class="links"><a href="quality.csv">Качество по seed / CSV</a><a href="ranks.csv">Ранги по seed / CSV</a>
<a href="loss.csv">SSL-loss / CSV</a><a href="paired_quality_summary.csv">Парные изменения / CSV</a><a href="audit.json">Проверка полноты</a></div>
<p class="sub">Проверено: все 54 × 7 оценочных точек, отсутствие дубликатов, совпадение эпохи 0, постоянный состав классов test,
все 30 эпох SSL и отсутствие пропущенных AMP-шагов. Test не использовался для настройки моделей или порогов.</p></div>
</main><script>const plots=__PLOTS_JSON__; const quality=__QUALITY_JSON__; const ranks=__RANKS_JSON__;
const controls=['category','fraction','norm','window'];const el=id=>document.getElementById(id);
function update(){const c=el('category').value,f=el('fraction').value,n=el('norm').value,w=el('window').value;
const plot=plots[`${c}_${f}_${n}_${w}`];el('figure').src=plot.image;
el('png').href=plot.png;el('svg').href=plot.image;el('svg').download=plot.svg;
const rows=quality.filter(x=>x.category===c&&Math.round(x.fraction*100)===+f&&(w==='full'||x.epoch<=10));
let content='<table><thead><tr><th>Метод / классификатор</th><th>Seed</th><th>Эпоха</th><th>Precision</th><th>Recall</th><th>ROC-AUC</th><th>AP</th><th>RankMe</th><th>CER</th><th>CovER</th><th>NESum</th></tr></thead><tbody>';
for(const x of rows){const r=ranks.find(r=>r.category===c&&r.fraction===x.fraction&&r.method===x.method&&r.seed===x.seed&&r.epoch===x.epoch&&r.normalization===n);
content+=`<tr><td>${x.method} / ${x.classifier}</td><td>${x.seed}</td><td>${x.epoch}</td>`;
for(const z of [x.precision_anomaly,x.recall_anomaly,x.roc_auc,x.average_precision,r.RankMe,r.CER,r.CovER,r.NESum])content+=`<td>${z.toFixed(3)}</td>`;content+='</tr>';}el('numbers').innerHTML=content+'</tbody></table>';}
for(const id of controls)el(id).addEventListener('change',update);update();</script></html>'''
    page = page.replace('OVERVIEW_LINKS', overview_links).replace('EVIDENCE', ''.join(evidence))
    page = page.replace('__PLOTS_JSON__', json.dumps(data)).replace('__QUALITY_JSON__', q.to_json(orient='records'))
    page = page.replace('__RANKS_JSON__', r.to_json(orient='records'))
    (out / 'index.html').write_text(page)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    q, r, loss = [pd.read_csv(args.root / 'analysis' / name) for name in ['quality.csv', 'ranks.csv', 'loss.csv']]
    checks = audit(q, r, loss, args.root)
    (args.output / 'audit.json').write_text(json.dumps(checks, indent=2) + '\n')
    for name in ['quality.csv', 'ranks.csv', 'loss.csv', 'baseline_summary.csv']:
        shutil.copy2(args.root / 'analysis' / name, args.output / name)
    for name in ['environment.json', 'config.json', 'source_snapshot.json', 'status.json']:
        if (args.root / name).exists():
            shutil.copy2(args.root / name, args.output / name)
    q = q[(q.split == 'test') & (q.method != 'baseline')].copy()
    r = r[(r.split == 'test') & (r.method != 'baseline')].copy()
    summary = pair_tables(q, r, args.output)
    style()
    plots = {}
    for cat, fraction, norm, early in itertools.product(CATEGORIES, FRACTIONS, ['raw', 'l2'], [False, True]):
        key = f'{cat}_{round(fraction*100):02d}_{norm}_{"early" if early else "full"}'
        plots[key] = combined(q, r, cat, fraction, norm, early, args.output)
        print(f'Plotted {key}', flush=True)
    overview = [('Precision', overview_quality(q, 'precision_anomaly', args.output)),
                ('ROC-AUC', overview_quality(q, 'roc_auc', args.output)),
                ('Ранги raw', overview_ranks(r, 'raw', args.output)),
                ('Ранги L2', overview_ranks(r, 'l2', args.output))]
    html_report(args.output, plots, overview, q, r, checks, summary)
    archive_path = args.output.parent / (args.output.name + '.zip')
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(args.output.iterdir()):
            if file.is_file():
                archive.write(file, arcname=f'{args.output.name}/{file.name}')
    print(json.dumps({'output': str(args.output), 'archive': str(archive_path), 'figures': len(plots) + len(overview),
                      'audit': checks}, indent=2), flush=True)


if __name__ == '__main__':
    main()
