"""Paired early-stage changes: no endpoint after epoch 30.

Six adjacent intervals and six cumulative 0->t intervals have 11 unique pairs.
Inference is exploratory: n=3 seeds, conditional fixed data. Holm is recomputed
over the union of the previous 5040 hypotheses and 11880 new early hypotheses.
"""
from pathlib import Path
import hashlib
import json
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from analyze_visa_v2_statistics import holm

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'reports/visa-v2-statistics-2026-11-14'
OUT=PARENT/'early'
RUN=['category','architecture','augmentation','method']
KEYS=RUN+['classifier']
METRICS=['roc_auc','average_precision','precision_anomaly','recall_anomaly','f1_anomaly']
PRIMARY=['roc_auc','average_precision']
EPOCHS=[0,1,2,5,10,20,30]
ADJACENT=list(zip(EPOCHS[:-1],EPOCHS[1:]))
CONTRASTS=sorted(set(ADJACENT+[(0,e) for e in EPOCHS[1:]]))
NAMES={'resnet18':'ResNet','deit_tiny':'ViT','simclr':'SimCLR','vicreg':'VICReg',
       'masked_reconstruction':'reconstruction'}


def fmt(v,signed=True):
    return (f'{v:+.2f}' if signed else f'{v:.2f}').replace('.',',')


def aggregate(pairs,cells,keys):
    seed=pairs.groupby(keys+['seed']).delta_pp.mean().reset_index()
    assert seed.groupby(keys).size().eq(3).all()
    macro=seed.groupby(keys).delta_pp.agg(mean_delta_pp='mean',macro_seed_sd_pp='std').reset_index()
    half=stats.t.ppf(.975,2)*macro.macro_seed_sd_pp/np.sqrt(3)
    macro['macro_ci95_low_pp']=macro.mean_delta_pp-half
    macro['macro_ci95_high_pp']=macro.mean_delta_pp+half
    info=cells.groupby(keys).agg(n_conditions=('n','size'),median_cell_sd_pp=('sd_delta_pp','median'),
        mean_positive=('mean_delta_pp',lambda s:int((s>1e-10).sum())),
        mean_negative=('mean_delta_pp',lambda s:int((s < -1e-10).sum())),
        unanimous_positive=('positive',lambda s:int((s==3).sum())),
        unanimous_negative=('negative',lambda s:int((s==3).sum())),
        nominal_p_under_05=('p_t',lambda s:int((s<.05).sum())),
        holm_union16920_under_05=('p_t_holm_union16920',lambda s:int((s<.05).sum()))).reset_index()
    return macro.merge(info,on=keys,validate='one_to_one')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'reports/visa-v2-block4/quality.csv'
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha==json.loads((PARENT/'summary.json').read_text())['source_sha256']['quality.csv']
    q=pd.read_csv(path)
    q=q[q.method.ne('baseline') & q.split.eq('test') & q.epoch.isin(EPOCHS)]
    assert len(q)==4536 and not q.duplicated(KEYS+['seed','epoch']).any()
    assert q.groupby(KEYS+['seed']).epoch.nunique().eq(7).all()
    assert np.isfinite(q[METRICS]).all().all()
    chunks=[]
    for a,b in CONTRASTS:
        left=q[q.epoch.eq(a)].set_index(KEYS+['seed']).sort_index()
        right=q[q.epoch.eq(b)].set_index(KEYS+['seed']).reindex(left.index)
        part=(100*(right[METRICS]-left[METRICS])).reset_index().melt(
            id_vars=KEYS+['seed'],value_vars=METRICS,var_name='metric',value_name='delta_pp')
        part['from_epoch'],part['to_epoch']=a,b
        chunks.append(part)
    pairs=pd.concat(chunks,ignore_index=True)
    assert len(pairs)==35640 and np.isfinite(pairs.delta_pp).all()
    group=KEYS+['metric','from_epoch','to_epoch']
    pivot=pairs.pivot(index=group,columns='seed',values='delta_pp')
    assert pivot.columns.tolist()==[6101,6102,6103]
    values=pivot.to_numpy()
    cells=pivot.rename(columns={s:f'delta_seed_{s}_pp' for s in pivot.columns}).reset_index()
    cells['n']=3
    cells['mean_delta_pp']=values.mean(axis=1)
    cells['sd_delta_pp']=values.std(axis=1,ddof=1)
    se=cells.sd_delta_pp/np.sqrt(3)
    valid=se>1e-10
    half=stats.t.ppf(.975,2)*se
    cells['ci95_low_pp']=np.where(valid,cells.mean_delta_pp-half,np.nan)
    cells['ci95_high_pp']=np.where(valid,cells.mean_delta_pp+half,np.nan)
    with np.errstate(divide='ignore',invalid='ignore'):
        cells['p_t']=np.where(valid,2*stats.t.sf(abs(cells.mean_delta_pp/se),2),np.nan)
    cells['positive']=(values>1e-10).sum(axis=1)
    cells['negative']=(values < -1e-10).sum(axis=1)
    cells['zero']=(abs(values)<=1e-10).sum(axis=1)
    cells['degenerate_sd']=~valid
    flips=np.asarray(list(itertools.product((-1,1),repeat=3)))
    nullmeans=values@flips.T/3
    cells['p_signflip']=(abs(nullmeans)>=abs(values.mean(axis=1))[:,None]-1e-12).mean(axis=1)
    previous=pd.read_csv(PARENT/'paired_statistics.csv')
    assert len(previous)==5040 and len(cells)==11880
    union=holm(pd.concat([previous.p_t,cells.p_t],ignore_index=True))
    cells['p_t_holm_union16920']=union[len(previous):]
    cells['p_t_holm_early11880']=holm(cells.p_t)
    oldcopy=previous[['comparison']+KEYS+['metric','early_epoch','late_epoch','p_t']].copy()
    oldcopy['p_t_holm_union16920']=union[:len(previous)]
    oldcopy.to_csv(OUT/'previous_tests_with_expanded_correction.csv',index=False)
    cells['adjacent_interval']=[(a,b) in ADJACENT for a,b in zip(cells.from_epoch,cells.to_epoch)]
    cells['cumulative_from_zero']=cells.from_epoch.eq(0)
    cells.to_csv(OUT/'early_paired_statistics.csv',index=False)
    pairs.to_csv(OUT/'early_paired_seed_deltas.csv',index=False)
    keys=['from_epoch','to_epoch','classifier','metric']
    overall=aggregate(pairs,cells,keys)
    method=aggregate(pairs,cells,['architecture','method']+keys)
    augmethod=aggregate(pairs,cells,['architecture','method','augmentation']+keys)
    overall.to_csv(OUT/'early_overall_summary.csv',index=False)
    method.to_csv(OUT/'early_method_summary.csv',index=False)
    augmethod.to_csv(OUT/'early_method_augmentation_summary.csv',index=False)
    # Per-checkpoint quality spread is not the spread of paired changes.
    long=q.melt(id_vars=KEYS+['seed','epoch'],value_vars=METRICS,var_name='metric',value_name='quality')
    long['quality_pp']=100*long.quality
    checkpoint=long.groupby(KEYS+['epoch','metric']).quality_pp.agg(mean_quality_pp='mean',sd_quality_pp='std').reset_index()
    checkpoint.to_csv(OUT/'early_checkpoint_quality_spread.csv',index=False)
    # Telescoping checks: adjacent deltas must sum to each cumulative delta,
    # matching identifiers, seed, probe and metric (not just overall means).
    pi=pairs.set_index(KEYS+['seed','metric','from_epoch','to_epoch']).delta_pp
    max_telescoping=0.
    for end in EPOCHS[2:]:
        stages=[(a,b) for a,b in ADJACENT if b<=end]
        summed=sum(pi.xs((a,b),level=['from_epoch','to_epoch']).sort_index() for a,b in stages)
        actual=pi.xs((0,end),level=['from_epoch','to_epoch']).sort_index()
        max_telescoping=max(max_telescoping,float(abs(summed-actual).max()))
        np.testing.assert_allclose(summed,actual,atol=2e-12)
    # Independent library validation of representative intervals/tests.
    ix=np.random.default_rng(141130).choice(np.flatnonzero(valid),30,replace=False)
    test=stats.ttest_1samp(values[ix],0,axis=1)
    np.testing.assert_allclose(test.pvalue,cells.iloc[ix].p_t,atol=1e-13)
    np.testing.assert_allclose(test.confidence_interval().low,cells.iloc[ix].ci95_low_pp,atol=1e-12)

    def pair(g,column='mean_delta_pp'):
        g=g.set_index('metric')
        return fmt(g.loc['roc_auc',column],column=='mean_delta_pp')+' / '+fmt(g.loc['average_precision',column],column=='mean_delta_pp')
    def stage_rows(contrasts):
        rows=[]
        for a,b in contrasts:
            g=overall[overall.from_epoch.eq(a)&overall.to_epoch.eq(b)&overall.classifier.eq('logistic_regression')&overall.metric.isin(PRIMARY)]
            gi=g.set_index('metric')
            count=' / '.join(f"{int(gi.loc[m,'unanimous_positive']+gi.loc[m,'unanimous_negative'])}/108" for m in PRIMARY)
            rows.append([f'{a} → {b}',pair(g),pair(g,'median_cell_sd_pp'),count])
        return rows
    headers=['Эпохи','Среднее Δ AUC / AP, п.п.','Типичный SD изменений AUC / AP','Одинаковый знак всех seed: AUC / AP']
    narrative={
        ('resnet18','simclr'):'Ранний рост примерно до 5–10 эпох; затем на 10→20 средние AUC/AP снижаются.',
        ('resnet18','vicreg'):'Рост средних до эпохи 5; далее часть выигрыша теряется. Это не непрерывное ухудшение с начала обучения.',
        ('resnet18','masked_reconstruction'):'Рост более постепенный; заметный подъём AUC на 5→10, затем снижение на 10→20.',
        ('deit_tiny','simclr'):'У LR средний AUC растёт на всех ранних checkpoints. Для AP и LightGBM есть колебания.',
        ('deit_tiny','vicreg'):'У LR после первой эпохи рост до 20, затем откат на 20→30; LightGBM меняется иначе.',
        ('deit_tiny','masked_reconstruction'):'Неровная ранняя динамика: падение средних на 5→10 и частичное восстановление на 10→20.'}
    mrows=[]
    for arch in ['resnet18','deit_tiny']:
        for met in ['simclr','vicreg','masked_reconstruction']:
            g=method[method.architecture.eq(arch)&method.method.eq(met)&method.classifier.eq('logistic_regression')&method.metric.isin(PRIMARY)&method.from_epoch.eq(0)]
            mrows.append([NAMES[arch]+' + '+NAMES[met],pair(g[g.to_epoch.eq(5)]),pair(g[g.to_epoch.eq(30)]),narrative[(arch,met)]])
    # Three examples distinguish checkpoint spread, delta spread and sign.
    case_specs=[
        ('capsules','deit_tiny','baseline','masked_reconstruction','logistic_regression','roc_auc',5,10,'Провал качества между ранними checkpoints'),
        ('capsules','deit_tiny','cutpaste','masked_reconstruction','lightgbm','average_precision',10,20,'Устойчивое изменение при различающихся уровнях качества'),
        ('candle','deit_tiny','rotate','vicreg','logistic_regression','average_precision',20,30,'Большое среднее снижение с разбросом и сменой знака')]
    case_notes=[]
    ci=cells.set_index(group)
    for cat,arch,aug,met,clf,metric,a,b,title in case_specs:
        r=ci.loc[(cat,arch,aug,met,clf,metric,a,b)]
        ds=', '.join(fmt(r[f'delta_seed_{s}_pp']) for s in [6101,6102,6103])
        case_notes.append(dict(title=title,text=f"{cat} / {NAMES[arch]} / {met} / {aug} / {clf}, {a}→{b}, {metric}: Δ = {fmt(r.mean_delta_pp)} п.п., SD = {fmt(r.sd_delta_pp,False)} п.п.; три изменения: {ds}."))
    significant=int(cells.p_t_holm_union16920.lt(.05).sum())
    summary={
        'headlines':[
            {'title':'Раннее обучение имеет другую динамику','text':'Для всех трёх методов ResNet к эпохе 10 средние AUC и AP выше исходных у обоих классификаторов; затем на 10→20 средние обеих метрик снижаются во всех трёх группах. Это не утверждение о монотонном росте на каждом шаге.'},
            {'title':'ViT + SimCLR на старте улучшается','text':'Средний AUC Logistic Regression возрастает на каждом сохранённом checkpoint от 0 до 30: 88,83% → 90,20%. Это средняя кривая фиксированной сетки, не утверждение о каждом запуске.'},
            {'title':'Разброс изменения нужно считать парно','text':'Два запуска могут иметь разные абсолютные качества, но одинаковое улучшение. Поэтому SD качества на эпохе и SD разности между эпохами — разные показатели.'}],
        'stage_table':{'headers':headers,'rows':stage_rows(ADJACENT)},
        'cumulative_table':{'headers':headers,'rows':stage_rows([(0,e) for e in EPOCHS[1:]])},
        'method_table':{'headers':['Группа','0→5: Δ AUC / AP','0→30: Δ AUC / AP','Ранняя динамика'],'rows':mrows},
        'cases':case_notes,
        'notes':['Все основные таблицы — Logistic Regression, единицы — процентные пункты. Методические выводы сверены с LightGBM; оба пробинга есть в CSV.',
                 'Типичный SD — медиана SD трёх парных изменений внутри 108 условий. Это не SD среднего всей сетки и не разброс абсолютного качества.',
                 'Соседние интервалы имеют разную длину. Показаны изменения за весь отрезок, без деления на число эпох; эти числа не являются скоростью обучения.',
                 '0→1 входит и в соседние, и в накопленные сравнения, но в расчётах учитывается один раз. Всего 11 уникальных пар эпох, ни одна не использует эпоху 100.',
                 'Средние смешивают шесть категорий и три аугментации. Совпадение их направления не означает одинакового эффекта у каждой конфигурации. Ранние точки описываются, а не выбираются как оптимальный checkpoint по test.'],
        'significance_notes':[f'Новые ранние сравнения: 11880 тестов (11 пар × 216 условий с пробингом × 5 метрик). Вместе с прежними 5040 тестами полная расширенная семья Holm содержит 16920 гипотез; новых значимых после этой поправки: {significant}.',
            'Ранний анализ добавлен после просмотра результатов и является исследовательским. Интервалы — номинальные Student при n=3; нормальность разностей по трём наблюдениям проверить нельзя.',
            'Три seed остаются тремя независимыми повторениями. Эпохи, метрики и классификаторы не увеличивают n. Существующие результаты прежнего отчёта сохранены; расширенная поправка записана отдельным файлом.'],
        'downloads':[
            {'path':'early_overall_summary.csv','label':'Все 11 ранних сравнений: средние, SD, знаки'},
            {'path':'early_method_summary.csv','label':'Ранние сравнения по архитектурам и методам'},
            {'path':'early_method_augmentation_summary.csv','label':'Ранние сравнения с разделением аугментаций'},
            {'path':'early_paired_statistics.csv','label':'Парные изменения, интервалы, p и расширенная поправка'},
            {'path':'early_checkpoint_quality_spread.csv','label':'Отдельно: разброс качества на каждой ранней эпохе'},
            {'path':'../summary/index.html','label':'Сводка длительного обучения и аугментаций'}]}
    (OUT/'early_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    audit={'source_sha256':sha,'quality_rows':len(q),'epochs':EPOCHS,'contrasts':CONTRASTS,
        'independent_seeds_per_condition':3,'new_tests':len(cells),'expanded_family_tests':len(union),
        'nominal_p_under_05':int(cells.p_t.lt(.05).sum()),'holm_early11880_significant':int(cells.p_t_holm_early11880.lt(.05).sum()),
        'holm_union16920_new_significant':significant,'degenerate_sd_rows':int(cells.degenerate_sd.sum()),
        'telescoping_max_error':max_telescoping,'independent_scipy_test_checks':len(ix),
        'no_epoch_100_in_new_comparisons':bool(cells.to_epoch.le(30).all()),
        'scope':'Exploratory seed inference on fixed data; no object bootstrap or optimal checkpoint selection.'}
    (OUT/'early_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    print(overall[overall.classifier.eq('logistic_regression')&overall.metric.isin(PRIMARY)].round(3).to_string(index=False))


if __name__=='__main__':
    main()
