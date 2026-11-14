"""Build an offline VisA v2 report from the final cumulative block, without training.

Run from smiles-ssl: ../.venv/bin/python scripts/build_visa_v2_report.py
The two analyze_visa_v2_*.py scripts produce additional descriptive tables.
"""
from __future__ import annotations

import base64
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import zipfile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/visa-v2-report-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/visa-v2-block4'
OUT = ROOT / 'reports/visa-v2-final'
CATS = ['cashew', 'chewinggum', 'fryum', 'candle', 'capsules', 'pcb1']
ARCHS = ['resnet18', 'deit_tiny']
AUGS = ['baseline', 'rotate', 'cutpaste']
METHODS = ['simclr', 'vicreg', 'masked_reconstruction']
CLFS = ['logistic_regression', 'lightgbm']
SEEDS = [6101, 6102, 6103]
EPOCHS = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100]
RANKS = ['RankMe', 'CER', 'CovER', 'NESum']
QUALITY = ['precision_anomaly', 'roc_auc', 'average_precision', 'recall_anomaly']
KEYS = ['category', 'architecture', 'augmentation', 'method', 'seed', 'epoch', 'split']
MLABEL = {'simclr': 'SimCLR', 'vicreg': 'VICReg', 'masked_reconstruction': 'Masked reconstruction'}
ALABEL = {'resnet18': 'ResNet-18', 'deit_tiny': 'ViT-Tiny'}
ULABEL = {'baseline': 'A0', 'rotate': 'A1 Rotate', 'cutpaste': 'A2 CutPaste'}
CLABEL = {'logistic_regression': 'Logistic regression', 'lightgbm': 'LightGBM'}
BLACK, ORANGE = '#252525', '#ed8427'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(q, r, loss):
    expected = set(itertools.product(CATS, ARCHS, AUGS, METHODS, SEEDS, EPOCHS,
                                     ['probe_train', 'validation', 'test']))
    for frame, column, variants, metrics in [(q, 'classifier', CLFS, QUALITY),
                                            (r, 'normalization', ['raw', 'l2'], RANKS)]:
        assert not frame.duplicated(KEYS + [column]).any()
        assert np.isfinite(frame[metrics]).all().all()
        for variant in variants:
            actual = set(frame.loc[frame.method.isin(METHODS) & frame[column].eq(variant), KEYS]
                         .itertuples(index=False, name=None))
            assert actual == expected, (column, variant, len(expected-actual), len(actual-expected))
        for _, group in frame.groupby(['category', 'architecture', 'seed', 'split', column]):
            b = group[group.method.eq('baseline')]
            z = group[group.method.isin(METHODS) & group.epoch.eq(0)]
            assert len(b) == 1 and len(z) == 9
            np.testing.assert_array_equal(z[metrics].values, np.repeat(b[metrics].values, 9, axis=0))
    assert q[QUALITY].ge(-1e-12).all().all() and q[QUALITY].le(1+1e-12).all().all()
    lkeys = KEYS[:5] + ['epoch']
    assert not loss.duplicated(lkeys).any()
    lexpected = set(itertools.product(CATS, ARCHS, AUGS, METHODS, SEEDS, range(1,101)))
    assert set(loss[lkeys].itertuples(index=False, name=None)) == lexpected
    assert np.isfinite(loss.loss).all() and loss.amp_skipped_steps.sum() == 0
    assert (q[q.split.eq('validation')].recall_anomaly >= .5).all()
    source_audit = json.loads((SOURCE/'audit.json').read_text())
    assert source_audit['all_training_complete'] and source_audit['completed_ssl_runs'] == 324
    return {'report_date_moscow': '2026-11-04', 'source': str(SOURCE),
            'ssl_runs':324, 'unique_baselines':36, 'quality_rows':len(q), 'rank_rows':len(r),
            'training_epochs':len(loss), 'epochs':EPOCHS, 'seeds':SEEDS,
            'full_factorial_grid_verified':True, 'epoch_zero_copies_exact':True,
            'duplicates':0, 'amp_skipped_updates':0, 'finite_metrics':True,
            'source_sha256':{n:digest(SOURCE/n) for n in ['quality.csv','ranks.csv','loss.csv','audit.json']},
            'scope':'Final cumulative block only; earlier blocks are not concatenated.',
            'verification_limit':'CSV and status audit; model weights and embeddings not reloaded.',
            'statistics':'Mean and sample SD across 3 seeds; paired deltas, descriptive macro means; no significance claims.'}


def payload(q, r):
    data = {'epochs':EPOCHS, 'seeds':SEEDS, 'q':{}, 'r':{},
            'fractions':q.groupby('category').fraction.first().to_dict()}
    for frame, suffix, metrics, dest in [(q,'classifier',QUALITY,'q'),(r,'normalization',RANKS,'r')]:
        for key, group in frame.groupby(['category','architecture','augmentation','method',suffix]):
            values = {}
            for metric in metrics:
                pivot = group.pivot(index='seed',columns='epoch',values=metric).loc[SEEDS,EPOCHS]
                assert pivot.shape == (3,10) and pivot.notna().all().all()
                values[metric] = pivot.values.tolist()
            data[dest]['|'.join(key)] = values
    return data


def save(fig, stem):
    for ext in ['png','svg']:
        fig.savefig(OUT/'figures'/f'{stem}.{ext}', dpi=170, facecolor='white')
    plt.close(fig)


def minmax(x):
    return (x-x.min())/np.ptp(x) if np.ptp(x)>1e-12 else np.full_like(x,.5)


def overlay(q, r, cat, arch, aug, clf, rank='RankMe', norm='raw', mode='minmax'):
    gq = q[(q.category==cat)&(q.architecture==arch)&(q.augmentation==aug)&(q.classifier==clf)]
    gr = r[(r.category==cat)&(r.architecture==arch)&(r.augmentation==aug)&(r.normalization==norm)]
    fig, axes = plt.subplots(3,2,figsize=(11.8,11.9))
    fig.subplots_adjust(left=.13,right=.90,top=.82,bottom=.14,wspace=.38,hspace=.45)
    fig.suptitle(f'VisA · {cat} · {ALABEL[arch]} · {ULABEL[aug]}',y=.975,fontsize=18,weight='semibold')
    fig.text(.5,.94,f'{CLABEL[clf]}  |  ranks: {norm.upper()}  |  seeds 6101 / 6102 / 6103',ha='center',fontsize=11)
    fig.text(.5,.908,'Each mean trajectory independently scaled to [0, 1]' if mode=='minmax'
             else 'Original values · left: quality · right: rank · mean ± sample SD',ha='center',fontsize=10.5,color='#515151')
    fig.legend(handles=[Line2D([],[],color=BLACK,marker='s',lw=1.5,ms=4,label='Downstream metric'),
                        Line2D([],[],color=ORANGE,marker='o',lw=1.7,ms=4,label=rank)],
               loc='upper center',bbox_to_anchor=(.5,.89),ncol=2,frameon=False,fontsize=11)
    rs = gr.groupby(['method','epoch'])[rank].agg(['mean','std'])
    rankhigh = float((rs['mean']+rs['std']).max())*1.08
    for row,method in enumerate(METHODS):
        axes[row,0].text(-.25,.5,MLABEL[method].replace('Masked reconstruction','Masked\nreconstruction'),
                         transform=axes[row,0].transAxes,rotation=90,ha='center',va='center',fontsize=11,weight='semibold')
        ranks=rs.loc[method].loc[EPOCHS]
        for col,metric in enumerate(QUALITY[:2]):
            ax=axes[row,col]
            qs=gq[gq.method==method].groupby('epoch')[metric].agg(['mean','std']).loc[EPOCHS]
            qm,rm=qs['mean'].values,ranks['mean'].values
            if mode=='minmax':
                ax.plot(EPOCHS,minmax(qm),color=BLACK,marker='s',ms=4,lw=1.5,zorder=3)
                ax.plot(EPOCHS,minmax(rm),color=ORANGE,marker='o',ms=4,lw=1.7,zorder=4)
                ax.set_ylim(-.06,1.06);ax.set_ylabel('Min–max scaled value',fontsize=9)
            else:
                ax.plot(EPOCHS,qm,color=BLACK,marker='s',ms=4,lw=1.5,zorder=3)
                ax.fill_between(EPOCHS,qm-qs['std'].values,qm+qs['std'].values,color=BLACK,alpha=.1,lw=0)
                ax.set_ylim(-.10,1.10);ax.set_ylabel('Quality',fontsize=9)
                other=ax.twinx();other.plot(EPOCHS,rm,color=ORANGE,marker='o',ms=4,lw=1.7,zorder=4)
                other.fill_between(EPOCHS,rm-ranks['std'].values,rm+ranks['std'].values,color=ORANGE,alpha=.15,lw=0)
                other.set_ylim(0,rankhigh);other.set_ylabel(rank,color=ORANGE,fontsize=9)
                other.tick_params(axis='y',colors=ORANGE,labelsize=8);other.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            ax.set_title(['Precision · anomaly','ROC-AUC'][col],fontsize=11,pad=9)
            ax.set_xlim(-2,102);ax.set_xticks([0,10,30,50,75,100]);ax.set_yticks(np.linspace(0,1,6))
            ax.set_xlabel('SSL epoch',fontsize=10);ax.tick_params(labelsize=9)
            ax.grid(color='#d8d8d8',lw=.65,alpha=.8);ax.set_axisbelow(True);ax.spines['top'].set_visible(False)
    for i,line in enumerate(['Fixed official test · mean across 3 seeds · epoch 0: untrained encoder.',
                             'Black squares: quality; orange circles: rank. Straight segments, no smoothing.',
                             'Min–max compares shapes only; native shading is sample SD, not a confidence interval.',
                             'Precision threshold selected on validation with recall ≥ 0.5; test recall unconstrained.',
                             'Classifiers use raw features. Raw/L2 changes only the rank computation.']):
        fig.text(.08,.103-.018*i,line,fontsize=8.5,color='#525252')
    stem=f'{cat}_{arch}_{aug}_{"lr" if clf==CLFS[0] else "lgbm"}_{rank}_{norm}_{mode}'
    save(fig,stem)
    return stem


def heatmap(q, early, metric='roc_auc'):
    keys=['category','architecture','augmentation','method','seed','classifier']
    paired=q[q.epoch==100].set_index(keys)[metric]-q[q.epoch==early].set_index(keys)[metric]
    means=paired.groupby([k for k in keys if k!='seed']).mean()*100
    fig,axes=plt.subplots(1,2,figsize=(17,8.4))
    fig.subplots_adjust(left=.14,right=.93,bottom=.19,top=.83,wspace=.13)
    rows=list(itertools.product(ARCHS,CATS));cols=list(itertools.product(METHODS,AUGS))
    mats=[np.array([[means.loc[(c,a,u,m,clf)] for m,u in cols] for a,c in rows]) for clf in CLFS]
    vmax=max(abs(x).max() for x in mats)
    title={'roc_auc':'ROC-AUC','average_precision':'AP'}[metric]
    fig.suptitle(f'{title}: эпоха 100 − эпоха {early}',fontsize=20,weight='semibold',y=.97)
    fig.text(.5,.92,'Средняя парная разность по 3 seed, процентные пункты · каждая категория отдельно',ha='center',fontsize=11)
    for ax,mat,clf in zip(axes,mats,CLFS):
        im=ax.imshow(mat,cmap='RdBu',vmin=-vmax,vmax=vmax,aspect='auto')
        ax.set_title(CLABEL[clf],fontsize=12,pad=12)
        ax.set_xticks(range(len(cols)),[MLABEL[m].replace('Masked reconstruction','Masked')+'\n'+ULABEL[u] for m,u in cols],rotation=45,ha='right',fontsize=9)
        ax.set_yticks(range(len(rows)),[ALABEL[a]+' · '+c for a,c in rows],fontsize=10)
        if ax is axes[1]: ax.set_yticklabels([])
        for i in range(len(rows)):
            for j in range(len(cols)):
                ax.text(j,i,f'{mat[i,j]:+.1f}',ha='center',va='center',fontsize=8.5,
                        color='white' if abs(mat[i,j])>vmax*.54 else '#252525')
        ax.axhline(5.5,color='white',lw=3)
    cax=fig.add_axes([.95,.19,.012,.64]);fig.colorbar(im,cax=cax,label='Δ, п.п.')
    fig.text(.14,.025,'Синий: качество выросло; красный: снизилось. Это описательные оценки на фиксированном test, без тестов значимости.',fontsize=10,color='#525252')
    stem=f'{metric}_delta_100_minus_{early}';save(fig,stem);return stem


def image_html(stem,caption):
    encoded=base64.b64encode((OUT/'figures'/f'{stem}.svg').read_bytes()).decode()
    return f'<figure class="summary-figure"><img src="data:image/svg+xml;base64,{encoded}" alt="{caption}" loading="lazy"><figcaption>{caption}</figcaption></figure>'


def counterexample(q,r):
    sel=dict(category='fryum',architecture='deit_tiny',augmentation='rotate',method='simclr')
    q=q[(q[list(sel)]==pd.Series(sel)).all(axis=1)&q.classifier.eq(CLFS[0])&q.epoch.ge(30)]
    r=r[(r[list(sel)]==pd.Series(sel)).all(axis=1)&r.normalization.eq('raw')&r.epoch.ge(30)]
    qs=q.groupby('epoch').roc_auc.agg(['mean','std'])*100
    fig,axes=plt.subplots(2,2,figsize=(12,8.2))
    fig.subplots_adjust(left=.085,right=.91,bottom=.14,top=.78,wspace=.39,hspace=.38)
    fig.suptitle('Ранг растёт, качество снижается',y=.97,fontsize=21,weight='semibold')
    fig.text(.5,.919,'fryum · ViT-Tiny · SimCLR · A1 Rotate · Logistic regression · Raw',ha='center',fontsize=11)
    fig.legend(handles=[Line2D([],[],color=BLACK,marker='s',lw=1.5,label='ROC-AUC, левая ось'),
                        Line2D([],[],color=ORANGE,marker='o',lw=1.7,label='Ранг, правая ось')],
               loc='upper center',bbox_to_anchor=(.5,.885),ncol=2,frameon=False)
    for ax,rank in zip(axes.flat,RANKS):
        rs=r.groupby('epoch')[rank].agg(['mean','std']);ep=rs.index.values
        ax.plot(ep,qs['mean'],color=BLACK,marker='s',lw=1.5,ms=4)
        ax.fill_between(ep,qs['mean']-qs['std'],qs['mean']+qs['std'],color=BLACK,alpha=.10,lw=0)
        ax.set_ylim(82,100);ax.set_yticks([85,90,95,100]);ax.set_ylabel('ROC-AUC, %');ax.set_xlabel('Эпоха SSL')
        ax.set_xticks(ep);ax.set_xlim(28,102);ax.grid(color='#d8d8d8',lw=.65,alpha=.8);ax.set_axisbelow(True)
        ax.set_title(rank,fontsize=12,pad=9);ax.spines['top'].set_visible(False)
        other=ax.twinx();other.plot(ep,rs['mean'],color=ORANGE,marker='o',lw=1.7,ms=4)
        other.fill_between(ep,rs['mean']-rs['std'],rs['mean']+rs['std'],color=ORANGE,alpha=.15,lw=0)
        other.set_ylim(0,float((rs['mean']+rs['std']).max())*1.10)
        other.set_ylabel(rank,color=ORANGE);other.tick_params(axis='y',colors=ORANGE);other.spines['top'].set_visible(False)
    fig.text(.085,.072,'Среднее ± выборочное SD по 3 seed; исходные значения, оси имеют разные единицы. Шкала ROC-AUC начинается с 82%.',fontsize=9,color='#525252')
    fig.text(.085,.044,'Во всех 3 seed: каждый ранг на эпохе 100 выше, а ROC-AUC, AP и precision ниже, чем на эпохе 30.',fontsize=9,color='#525252')
    fig.text(.085,.016,'Пример выбран после анализа для иллюстрации; не является отдельным подтверждающим экспериментом.',fontsize=9,color='#525252')
    save(fig,'rank_quality_counterexample')
    return 'rank_quality_counterexample'


def findings_html():
    macro=pd.read_csv(OUT/'data/quality_macro_summary.csv')
    sel=macro[macro['view'].eq('architecture/method/classifier')&macro.early_epoch.eq(0)]
    table=['<div class="table-wrap"><table><thead><tr><th>Энкодер / SSL</th><th>Δ ROC-AUC · LR</th><th>Δ ROC-AUC · LGBM</th><th>Δ AP · LR</th><th>Δ AP · LGBM</th></tr></thead><tbody>']
    for a,m in itertools.product(ARCHS,METHODS):
        values=[]
        for metric,clf in itertools.product(['roc_auc','average_precision'],CLFS):
            v=sel[(sel.architecture==a)&(sel.method==m)&(sel.metric==metric)&(sel.classifier==clf)].macro_delta_pp.item()
            values.append(f'<td>{v:+.2f}</td>')
        table.append(f'<tr><th>{ALABEL[a]} · {MLABEL[m]}</th>'+''.join(values)+'</tr>')
    table.append('</tbody></table></div>')
    correlations=pd.read_csv(OUT/'data/rank_correlation_summary.csv')
    cor=correlations[(correlations.level=='overall')&(correlations.window=='trained_1_100')&
                     (correlations.classifier==CLFS[0])&(correlations.metric=='roc_auc')]
    rtable=['<div class="table-wrap"><table><thead><tr><th>Ранг</th><th>Raw: медиана ρ</th><th>L2: медиана ρ</th><th>Raw: 25–75%</th></tr></thead><tbody>']
    for rank in RANKS:
        raw=cor[(cor['rank']==rank)&(cor.normalization=='raw')].iloc[0]
        l2=cor[(cor['rank']==rank)&(cor.normalization=='l2')].iloc[0]
        rtable.append(f'<tr><th>{rank}</th><td>{raw.median_spearman:+.3f}</td><td>{l2.median_spearman:+.3f}</td><td>{raw.q25_spearman:+.3f} … {raw.q75_spearman:+.3f}</td></tr>')
    rtable.append('</tbody></table></div>')
    architecture_table=['<div class="table-wrap"><table><thead><tr><th>Архитектура / probe</th><th>ROC-AUC: 0 → 100</th><th>AP: 0 → 100</th></tr></thead><tbody>']
    for a,clf in itertools.product(ARCHS,CLFS):
        group=macro[(macro['view']=='architecture/classifier')&(macro.early_epoch==0)&(macro.architecture==a)&(macro.classifier==clf)]
        cells=[]
        for metric in ['roc_auc','average_precision']:
            row=group[group.metric==metric].iloc[0]
            cells.append(f'<td>{row.macro_initial:.4f} → {row.macro_final:.4f}</td>')
        architecture_table.append(f'<tr><th>{ALABEL[a]} · {CLABEL[clf]}</th>'+''.join(cells)+'</tr>')
    architecture_table.append('</tbody></table></div>')
    return '''<p class="lead"><b>Главный результат:</b> универсальной связи «выше ранг → лучше downstream» в этой сетке нет.
    Продолжать SSL до 100 эпох полезно не для всех методов и категорий.</p>
    <div class="finding"><h3>1. Позднее обучение чаще ухудшает качество</h3>
    <p>С эпохи 30 до 100 средний ROC-AUC снижается в <b>142 из 216 условий (65.7%)</b>, AP — в <b>139 из 216 (64.4%)</b>.
    Для LR среднее изменение составляет −1.01 п.п. ROC-AUC и −2.38 п.п. AP; для LightGBM — −1.41 и −3.84 п.п.
    Отрицательный знак ROC-AUC повторяется во всех трёх seed в 30/108 условиях LR и 36/108 LightGBM.
    Особенно заметен спад у SimCLR + ViT: −3.09 / −4.89 п.п. ROC-AUC (LR / LightGBM).</p></div>
    <div class="finding"><h3>2. Masked reconstruction даёт наиболее устойчивый средний выигрыш</h3>
    <p>Это единственный из трёх методов с положительным средним изменением <b>и ROC-AUC, и AP</b> относительно необученного энкодера
    для обеих архитектур и обоих классификаторов. Таблица показывает Δ100−0 в п.п., среднее по 6 категориям × 3 аугментациям.</p>'''+''.join(table)+'''
    <p>Средний выигрыш не означает улучшения в каждой задаче: для ResNet + reconstruction ROC-AUC растёт в 11/18 условиях LR и 12/18 LightGBM;
    для ViT — в 11/18 и 9/18. На ResNet продолжение reconstruction после 30 эпох в среднем ещё даёт +1.18 / +0.59 п.п. ROC-AUC.</p></div>
    <div class="finding"><h3>3. Ранг не подходит как универсальное правило выбора checkpoint</h3>
    <p>Spearman рассчитан <b>внутри каждого отдельного запуска</b> между рангом и ROC-AUC по девяти checkpoint эпох 1–100.
    Ниже — медианы по 324 запускам для LR. Медианы около нуля, а межквартильные интервалы пересекают ноль.
    По всем четырём рангам, двум нормировкам и двум классификаторам медианы лежат от −0.100 до +0.063.
    Это не исключает связи внутри отдельных условий.</p>'''+''.join(rtable)+'''
    <p>В позднем сравнении 100−30 для RankMe / Raw / LR направления средних изменений ранга и ROC-AUC противоположны
    в 67/108 условий. Ранг растёт, AUC падает в 59/108, но оба знака сохраняются во всех трёх seed только в 12/108.</p>
    <p><b>Иллюстрация:</b> fryum / ViT-Tiny / SimCLR / Rotate / LR, 30→100.
    Все четыре Raw-ранга растут во всех seed, тогда как ROC-AUC падает <b>95.76% → 90.32%</b>,
    AP <b>76.74% → 60.30%</b>, precision <b>80.02% → 66.93%</b>. Пример выбран после просмотра данных.</p></div>
    <div class="finding"><h3>4. Категории и аугментации меняют знак эффекта</h3>
    <p>При одинаковых ResNet + SimCLR + A0 + LR ROC-AUC для chewinggum растёт 0.6796→0.8026,
    а для capsules падает 0.8279→0.6716. Оба направления наблюдаются во всех трёх seed.
    CutPaste относительно A0 на эпохе 100 повышает средний ROC-AUC ResNet на +0.21 / +0.80 п.п.,
    но снижает его у ViT на −1.30 / −0.72 п.п. (LR / LightGBM). Rotate снижает средний ROC-AUC
    для обеих архитектур и обоих классификаторов. Универсального улучшения от дополнительных аугментаций нет.</p></div>
    <div class="finding"><h3>5. Precision следует читать вместе с recall и AP</h3>
    <p>На эпохе 100 test recall ниже 0.5 у <b>324 из 648 оценок (50%)</b>. Ограничение recall ≥ 0.5 действовало
    на validation при выборе порога и не гарантируется на test. Поэтому во вкладке графиков доступна пара AP / recall.</p></div>
    <div class="finding"><h3>6. Случайный энкодер — сильная исходная точка</h3>
    <p>Если дать равный вес всем методам, аугментациям и категориям, итоговый ROC-AUC остаётся близок к исходному
    или снижается. На эпохе 100 средняя AP у ResNet выше, чем у ViT, для обоих probes; у LR средний ROC-AUC почти одинаков.</p>'''+''.join(architecture_table)+'''
    <p>Это сводка по 54 условиям на строку; методы и категории в ней смешаны с равным весом. Она не заменяет
    разбор отдельных задач и не является изолированным сравнением размерностей или архитектур.</p></div>
    <p class="note">Условие = категория × архитектура × аугментация × SSL-метод × классификатор; всего 216, каждое усреднено по 3 seed.
    Macro — равный вес условий, а не объединение предсказаний. Здесь описательные сравнения на общих фиксированных данных,
    без заявлений о статистической значимости и причинности. Эпоха 30 — заранее заданная точка сравнения, а не найденный оптимум.</p>'''


def main(template_path=None,share_path=None):
    (OUT/'data').mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','axes.edgecolor':'#777777','axes.linewidth':.7})
    qa=pd.read_csv(SOURCE/'quality.csv');ra=pd.read_csv(SOURCE/'ranks.csv');loss=pd.read_csv(SOURCE/'loss.csv')
    checks=audit(qa,ra,loss)
    q=qa[qa.split.eq('test')&qa.method.isin(METHODS)].copy()
    r=ra[ra.split.eq('test')&ra.method.isin(METHODS)].copy()
    for name,frame in [('quality_test.csv',q),('ranks_test.csv',r)]:frame.to_csv(OUT/'data'/name,index=False)
    for frame,variant,metrics,name in [(q,'classifier',QUALITY,'quality_mean_sd.csv'),(r,'normalization',RANKS,'ranks_mean_sd.csv')]:
        g=frame.groupby(['category','architecture','augmentation','method',variant,'epoch'])[metrics].agg(['mean','std','count'])
        g.columns=['_'.join(c) for c in g.columns];g.to_csv(OUT/'data'/name)
    data=payload(q,r)
    (OUT/'data/plot_data.json').write_text(json.dumps(data,separators=(',',':'),allow_nan=False))
    (OUT/'data/report_audit.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    figs=[image_html(counterexample(q,r),'Рост всех четырёх рангов при падении качества на позднем отрезке. Пример для иллюстрации расхождения.')]
    for early in [0,30]:
        stem=heatmap(q,early);figs.append(image_html(stem,f'ROC-AUC: 100−{early}, средняя парная разность по трём seed.'))
    stem=heatmap(q,0,'average_precision');figs.append(image_html(stem,'Average precision: 100−0. AP дополняет ROC-AUC при редком аномальном классе.'))
    # Illustrations only, selected for discussion; full matrix remains in the explorer.
    for cat,arch,aug,clf in [('fryum','resnet18','baseline',CLFS[0]),
                             ('fryum','deit_tiny','rotate',CLFS[0]),
                             ('capsules','resnet18','baseline',CLFS[0]),
                             ('chewinggum','resnet18','baseline',CLFS[0])]:
        for mode in ['minmax','native']:
            overlay(q,r,cat,arch,aug,clf,mode=mode)
    findings=findings_html();(OUT/'findings.html').write_text(findings)
    downloads=[]
    for p in sorted((OUT/'data').glob('*')):
        if p.suffix not in {'.csv','.json'}:continue
        if p.stat().st_size>2_000_000:continue
        mime='text/csv' if p.suffix=='.csv' else 'application/json'
        encoded=base64.b64encode(p.read_bytes()).decode()
        downloads.append(f'<a download="{p.name}" href="data:{mime};base64,{encoded}">{p.name}</a>')
    template=(template_path or Path(__file__).with_name('visa_v2_report.template.html')).read_text()
    page=template.replace('__PAYLOAD__',json.dumps(data,separators=(',',':'),allow_nan=False))
    page=page.replace('__FINDINGS__',findings).replace('__SUMMARY_FIGURES__','\n'.join(figs))
    page=page.replace('__DOWNLOADS__','\n'.join(downloads)).replace('__AUDIT__',
         '<p>Проверены все 324 запуска × 10 checkpoint, 36 уникальных baseline, отсутствие дубликатов, конечность метрик и точное совпадение копий эпохи 0. '
         'Всего 19 656 строк качества, столько же строк рангов и 32 400 обучающих эпох. Пропущенных AMP updates — 0. '
         'В отчёт входят итоговые накопленные таблицы block4; промежуточные блоки не добавлялись повторно. '
         'Хеши источников сохранены в report_audit.json. Веса и эмбеддинги в этом анализе повторно не загружались.</p>')
    assert not any(token in page for token in ['__PAYLOAD__','__FINDINGS__','__SUMMARY_FIGURES__','__DOWNLOADS__','__AUDIT__'])
    (OUT/'index.html').write_text(page)
    share=share_path or ROOT/'reports/VisA_v2_final_report.html';share.write_text(page)
    code=OUT/'reproduce';code.mkdir(exist_ok=True)
    for name in ['build_visa_v2_report.py','visa_v2_report.template.html','analyze_visa_v2_quality.py','analyze_visa_v2_ranks.py']:
        p=Path(__file__).with_name(name)
        if p.exists() and p.resolve()!=(code/name).resolve():shutil.copy2(p,code/name)
    for n,h in checks['source_sha256'].items():assert digest(SOURCE/n)==h
    print(json.dumps({'html':str(share),'bytes':share.stat().st_size,'figures':len(list((OUT/'figures').glob('*.png'))),'data_files':len(downloads)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--template',type=Path)
    parser.add_argument('--share',type=Path)
    args=parser.parse_args();SOURCE=args.source.resolve();OUT=args.output.resolve()
    main(args.template,args.share)
