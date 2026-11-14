"""Compact, auditable synthesis of duration, repeatability and augmentation.

Descriptive grouping only. Existing cell-wise t/Holm results retain their full
5040-test family. Shared conditions/epochs/probes never become extra seeds.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/visa-v2-statistics-2026-11-14'
OUT = SOURCE / 'summary'
RUN = ['category', 'architecture', 'augmentation', 'method']
METRICS = ['roc_auc', 'average_precision']
EPS = 1e-10
NAMES = {'resnet18':'ResNet', 'deit_tiny':'ViT', 'simclr':'SimCLR',
         'vicreg':'VICReg', 'masked_reconstruction':'reconstruction',
         'baseline':'A0', 'rotate':'Rotate', 'cutpaste':'CutPaste'}


def aggregate(pairs, cells, keys):
    seed = pairs.groupby(keys+['seed']).delta_pp.mean().reset_index()
    assert seed.groupby(keys).seed.nunique().eq(3).all()
    macro = seed.groupby(keys).delta_pp.agg(mean_delta_pp='mean', macro_seed_sd_pp='std').reset_index()
    half = t.ppf(.975, 2) * macro.macro_seed_sd_pp / np.sqrt(3)
    macro['macro_ci95_low_pp'], macro['macro_ci95_high_pp'] = macro.mean_delta_pp-half, macro.mean_delta_pp+half
    counts = cells.groupby(keys).agg(
        n_conditions=('n','size'), median_cell_sd_pp=('sd_delta_pp','median'),
        q25_cell_sd_pp=('sd_delta_pp',lambda x:x.quantile(.25)),
        q75_cell_sd_pp=('sd_delta_pp',lambda x:x.quantile(.75)),
        mean_positive=('mean_delta_pp',lambda x:int((x>EPS).sum())),
        mean_negative=('mean_delta_pp',lambda x:int((x < -EPS).sum())),
        unanimous_positive=('positive',lambda x:int((x==3).sum())),
        unanimous_negative=('negative',lambda x:int((x==3).sum())),
        nominal_t_p_under_05=('p_t',lambda x:int((x<.05).sum())),
        holm_all5040_under_05=('p_t_holm_all5040',lambda x:int((x<.05).sum()))).reset_index()
    return macro.merge(counts,on=keys,validate='one_to_one')


def fmt(x):
    return f'{x:+.2f}'.replace('.',',')


def spread(x):
    return f'{x:.2f}'.replace('.',',')


def mean_pair(frame, classifier='logistic_regression'):
    frame=frame[frame.classifier.eq(classifier)].set_index('metric')
    return f"{fmt(frame.loc['roc_auc','mean_delta_pp'])} / {fmt(frame.loc['average_precision','mean_delta_pp'])}"


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cells=pd.read_csv(SOURCE/'paired_statistics.csv')
    pairs=pd.read_csv(SOURCE/'paired_seed_deltas.csv')
    assert len(cells)==5040 and len(pairs)==15120
    d=cells[cells.comparison.eq('duration')]
    p=pairs[pairs.comparison.eq('duration')]
    keys=['architecture','method','classifier','metric','early_epoch']
    method=aggregate(p,d,keys)
    assert len(method)==240 and method.n_conditions.eq(18).all()
    method.to_csv(OUT/'method_duration_statistics.csv',index=False)
    by_aug=aggregate(p,d,['architecture','method','augmentation','classifier','metric','early_epoch'])
    assert by_aug.n_conditions.eq(6).all()
    by_aug.to_csv(OUT/'duration_by_augmentation.csv',index=False)
    # Joint sign counts describe consistency, not extra independent samples.
    selected=d[d.metric.isin(METRICS)]
    joint=[]
    for key,g in selected.groupby(RUN+['early_epoch']):
        assert len(g)==4
        joint.append({**dict(zip(RUN+['early_epoch'],key)),
            'all_four_means_positive':bool(g.mean_delta_pp.gt(EPS).all()),
            'all_four_means_negative':bool(g.mean_delta_pp.lt(-EPS).all()),
            'all_three_seeds_positive_for_both_metrics_and_probes':bool(g.positive.eq(3).all()),
            'all_three_seeds_negative_for_both_metrics_and_probes':bool(g.negative.eq(3).all())})
    joint=pd.DataFrame(joint)
    joint.to_csv(OUT/'joint_direction_by_condition.csv',index=False)
    flags=[c for c in joint if c.startswith('all_')]
    jsummary=joint.groupby(['architecture','method','early_epoch'])[flags].sum().reset_index()
    jsummary['conditions_denominator']=18
    jsummary.to_csv(OUT/'joint_direction_summary.csv',index=False)
    # Keep precision of the change separate from consistency of its sign.
    repeat=[]
    for key,g in selected.groupby(RUN):
        assert len(g)==16
        row={**dict(zip(RUN,key)), 'n_dependent_summaries':16,
             'positive_means':int(g.mean_delta_pp.gt(EPS).sum()),
             'negative_means':int(g.mean_delta_pp.lt(-EPS).sum()),
             'unanimous_positive_summaries':int(g.positive.eq(3).sum()),
             'unanimous_negative_summaries':int(g.negative.eq(3).sum())}
        for metric,prefix in [('roc_auc','auc'),('average_precision','ap')]:
            mg=g[g.metric.eq(metric)]
            row[prefix+'_median_sd_pp']=float(mg.sd_delta_pp.median())
            row[prefix+'_max_sd_pp']=float(mg.sd_delta_pp.max())
        row['max_metric_median_sd_pp']=max(row['auc_median_sd_pp'],row['ap_median_sd_pp'])
        repeat.append(row)
    repeat=pd.DataFrame(repeat).sort_values('max_metric_median_sd_pp')
    repeat.to_csv(OUT/'repeatability_all_four_contrasts.csv',index=False)
    # The actual quality at every saved checkpoint also has a seed spread.
    quality_path=ROOT/'reports/visa-v2-block4/quality.csv'
    q=pd.read_csv(quality_path)
    q=q[q.method.ne('baseline') & q.split.eq('test')]
    per_epoch=q.melt(id_vars=RUN+['classifier','epoch','seed'],value_vars=[
        'roc_auc','average_precision','precision_anomaly','recall_anomaly','f1_anomaly'],
        var_name='metric',value_name='quality')
    per_epoch['quality_pp']=100*per_epoch.quality
    per_cell=per_epoch.groupby(RUN+['classifier','epoch','metric']).quality_pp.agg(mean='mean',sd='std').reset_index()
    per_cell.groupby(['architecture','method','classifier','epoch','metric']).sd.agg(
        median_sd_pp='median',min_sd_pp='min',max_sd_pp='max').reset_index().to_csv(
        OUT/'quality_spread_by_checkpoint.csv',index=False)
    aug=pd.read_csv(OUT/'aug_review_macro.csv')
    aug100=aug[aug.epoch.eq(100)]
    ac=cells[cells.comparison.eq('augmentation_vs_A0') & cells.metric.isin(METRICS)]
    aug_joint=[]
    for key,g in ac.groupby(['architecture','augmentation','category','method']):
        lr=g[g.classifier.eq('logistic_regression')]
        aug_joint.append(dict(zip(['architecture','augmentation','category','method'],key)) | {
            'both_lr_metrics_all3_positive':bool(lr.positive.eq(3).all()),
            'both_lr_metrics_all3_negative':bool(lr.negative.eq(3).all()),
            'both_probes_metrics_all3_positive':bool(g.positive.eq(3).all()),
            'both_probes_metrics_all3_negative':bool(g.negative.eq(3).all())})
    aj=pd.DataFrame(aug_joint)
    aj.to_csv(OUT/'augmentation_joint_direction.csv',index=False)
    aj_counts=aj.groupby(['architecture','augmentation'])[[c for c in aj if c.startswith('both_')]].sum()

    method_rows=[]
    descriptions={
        ('resnet18','masked_reconstruction'):'Наиболее ясный средний выигрыш; продолжение обычно даёт меньший плюс. Не все категории одинаковы.',
        ('deit_tiny','masked_reconstruction'):'Общий плюс небольшой; эффект продолжения зависит от точки сравнения и классификатора.',
        ('resnet18','simclr'):'Общий эффект смешанный; во всех трёх сравнениях продолжения средние AUC/AP снижаются у обоих классификаторов.',
        ('deit_tiny','simclr'):'Наиболее выраженное среднее ухудшение при продолжении; это не означает малый разброс каждого условия.',
        ('resnet18','vicreg'):'Среднее ухудшение, но аугментации могут менять знак поздней динамики.',
        ('deit_tiny','vicreg'):'Отрицательные средние скрывают как устойчивые улучшения, так и ухудшения отдельных категорий.'}
    for arch,met in [('resnet18','masked_reconstruction'),('deit_tiny','masked_reconstruction'),
                     ('resnet18','simclr'),('deit_tiny','simclr'),('resnet18','vicreg'),('deit_tiny','vicreg')]:
        g=method[method.architecture.eq(arch)&method.method.eq(met)&method.metric.isin(METRICS)]
        overall=g[g.early_epoch.eq(0)]
        continuation=g[g.early_epoch.isin([5,10,30]) & g.classifier.eq('logistic_regression')]
        ranges=[]
        for metric in METRICS:
            mg=continuation[continuation.metric.eq(metric)].mean_delta_pp
            ranges.append(f'{fmt(mg.min())}…{fmt(mg.max())}')
        method_rows.append([NAMES[arch]+' + '+NAMES[met],mean_pair(overall),
                            ' / '.join(ranges),descriptions[(arch,met)]])
    duration_rows=[]
    overall=aggregate(p,d,['classifier','metric','early_epoch'])
    overall.to_csv(OUT/'duration_overall_statistics.csv',index=False)
    for early in [0,5,10,30]:
        g=overall[overall.early_epoch.eq(early)&overall.classifier.eq('logistic_regression')]
        means=mean_pair(g)
        mg=g.set_index('metric')
        duration_rows.append([f'{early} → 100',means,
            spread(mg.loc['roc_auc','median_cell_sd_pp'])+' / '+spread(mg.loc['average_precision','median_cell_sd_pp']),
            f"{int(mg.loc['roc_auc','unanimous_positive']+mg.loc['roc_auc','unanimous_negative'])}/108 / "+
            f"{int(mg.loc['average_precision','unanimous_positive']+mg.loc['average_precision','unanimous_negative'])}/108"])
    aug_rows=[]
    for arch in ['resnet18','deit_tiny']:
        for a in ['rotate','cutpaste']:
            g=aug100[aug100.architecture.eq(arch)&aug100.augmentation.eq(a)]
            lr=g[g.classifier.eq('logistic_regression')].set_index('metric')
            counts=aj_counts.loc[(arch,a)]
            aug_rows.append([NAMES[arch]+' + '+NAMES[a],mean_pair(g),mean_pair(g,'lightgbm'),
                spread(lr.loc['roc_auc','median_cell_sd_pp'])+' / '+spread(lr.loc['average_precision','median_cell_sd_pp']),
                f"{int(counts.both_lr_metrics_all3_positive)}/18 ↑; {int(counts.both_lr_metrics_all3_negative)}/18 ↓"])

    target_cases=[
        ('fryum','deit_tiny','rotate','simclr','Ухудшение повторяется во всех четырёх сравнениях, у обоих классификаторов и по обеим метрикам.'),
        ('cashew','deit_tiny','cutpaste','vicreg','Положительное исключение внутри группы ViT + VICReg: все средние положительны, большинство знаков повторяется между seed. Это эффект обучения в режиме CutPaste, не доказательство преимущества CutPaste над A0.'),
        ('fryum','deit_tiny','rotate','vicreg','Средние отрицательны, но величина эффекта сильно колеблется; высокий приоритет уточнения разброса.'),
        ('capsules','deit_tiny','baseline','masked_reconstruction','Направление и величина зависят от запуска и ранней точки сравнения; вывод о пользе продолжения пока неустойчив.')]
    case_rows=[]
    for cat,arch,augname,met,reading in target_cases:
        r=repeat.set_index(RUN).loc[(cat,arch,augname,met)]
        title=' / '.join(NAMES.get(x,x) for x in [cat,arch,augname,met])
        text=(f"Медианный SD изменений: AUC {spread(r.auc_median_sd_pp)} п.п., AP {spread(r.ap_median_sd_pp)} п.п. "
              f"Одинаковый знак во всех трёх seed в {int(r.unanimous_positive_summaries+r.unanimous_negative_summaries)}/16 сводок. "+reading)
        case_rows.append(dict(title=title,text=text))
    neg_all=int(repeat.unanimous_negative_summaries.eq(16).sum())
    proof={
        'headlines':[
            {'title':'Общая картина уже читается','text':'ResNet + reconstruction в среднем полезнее; продолжение SimCLR/VICReg часто ухудшает качество. ViT + reconstruction не даёт ясного общего выигрыша от продолжения.'},
            {'title':'Аугментации зависят от времени обучения','text':'На ViT обе добавки улучшают средние AUC/AP на эпохе 10, но к эпохе 100 средние эффекты обеих добавок отрицательны. Это сравнения с A0 на той же эпохе.'},
            {'title':'Направление и разброс нужно читать отдельно','text':f'Только у {neg_all}/108 условий ухудшение AUC/AP повторяется во всех seed, классификаторах и четырёх временных сравнениях. Даже у таких условий величина ухудшения может заметно колебаться.'}],
        'duration_table':{'headers':['Сравнение','Среднее Δ AUC / AP, п.п.','Типичный SD Δ AUC / AP, п.п.','Знак совпал во всех seed: AUC / AP'],'rows':duration_rows},
        'duration_notes':['Здесь Logistic Regression и все 108 условий. Типичный SD — медиана SD парных изменений внутри условий; это не SD среднего всей сетки.',
            'Проверены четыре заданных сравнения 100−0/5/10/30. Они не являются независимыми повторениями и не описывают каждый переход траектории. Разброс самого качества на всех 10 checkpoints сохранён отдельно.'],
        'method_table':{'headers':['Архитектура + метод','0→100: Δ AUC / AP','Продолжение: диапазон Δ AUC / AP','Как читать'],'rows':method_rows},
        'method_notes':['Все числа — п.п. и Logistic Regression. Для каждой строки усреднены шесть категорий × три аугментации. Диапазон продолжения охватывает три сравнения 100−5, 100−10 и 100−30, а не доверительный интервал.',
            'Выводы сверены с LightGBM. У ResNet + reconstruction небольшое исключение — AP LightGBM 100−5: −0,24 п.п.; у ViT + reconstruction знак зависит от классификатора и сравнения.'],
        'augmentation_table':{'headers':['Добавка к A0','Δ AUC / AP · LR','Δ AUC / AP · LightGBM','Типичный SD Δ AUC / AP · LR','Оба показателя во всех seed · LR'],'rows':aug_rows},
        'augmentation_notes':[
            'Сравнивается A0 + Rotate/CutPaste минус A0 при одинаковых категории, методе, seed и эпохе 100. A0 уже содержит crop и color jitter; это не сравнение с полным отсутствием аугментаций.',
            'Каждая строка охватывает 18 условий: шесть категорий × три метода. Последний столбец требует одинакового знака у AUC и AP во всех трёх seed. Малые средние при большом SD не доказывают отсутствия эффекта.',
            'Все 16 номинальных 95% интервалов общего эффекта добавок (две архитектуры × две добавки × два классификатора × две метрики) на эпохе 100 включают ноль. Они рассчитаны по трём средним каждого seed на фиксированной сетке и не являются интервалами отдельного условия.',
            'CutPaste на ResNet даёт небольшой средний плюс, но не универсальный: на chewinggum среднее Δ AUC составляет −3,58 п.п. для LR и −2,55 для LightGBM; на cashew +1,47 и +4,94 п.п.',
            'На ViT на эпохе 10 обе добавки улучшают средние AUC/AP у обоих классификаторов (AUC +0,91…+1,51 п.п.; AP +1,35…+2,09), а на эпохе 100 все соответствующие средние отрицательны. Ранний выигрыш не сохраняется.',
            'Аугментация меняет и позднюю динамику. Например, ResNet + VICReg: Δ AUC 100−30 для LR равно +0,45 с A0, −2,44 с Rotate, −1,97 с CutPaste. Это динамика внутри режима, не эффект добавки на фиксированной эпохе.'],
        'cases':case_rows,
        'noise_notes':['Согласованный знак означает повторяемое направление. Небольшой SD означает близкую величину изменений. Одно не гарантирует другого.',
            'Число независимых повторов остаётся три. 16 сводок примера — четыре контраста × две метрики × два классификатора, а не 16 независимых экспериментов.',
            'Известные интервалы bootstrap по test-объектам охватывают 67/108 условий для сравнения 100−30. Они отдельны от seed-интервалов и не покрывают эффект CutPaste относительно A0. Во всех категориях только 40 test-аномалий.'],
        'significance_notes':[
            f"В существующей полной семье из 5040 сравнений значимых после Holm: {int(cells.p_t_holm_all5040.lt(.05).sum())}. Для 288 сравнений AUC/AP добавок на эпохе 100 номинальное p<0,05 у {int(ac.p_t.lt(.05).sum())}, после полной поправки — {int(ac.p_t_holm_all5040.lt(.05).sum())}.",
            'Это отсутствие достаточного подтверждения при данной процедуре, а не доказательство отсутствия эффектов. Student-интервалы при n=3 зависят от непроверяемого здесь предположения нормальности разностей.',
            'Новые сводки по архитектуре, методу и аугментации описательные. Их средние и SD считаются сначала внутри каждого seed по фиксированной сетке; условия и эпохи не используются как дополнительные независимые наблюдения.'],
        'next_steps':['Новые seed прежде направлять в важные условия с большим разбросом или меняющимся знаком, например fryum / ViT / Rotate / VICReg и capsules / ViT / A0 / reconstruction.',
            'Для проверки добавки повторять парой A0 и соответствующую аугментацию с одинаковыми новыми seed. Новый одиночный запуск Rotate или CutPaste не даст нового парного наблюдения эффекта добавки.',
            'Старый пример очереди из 24 запусков был выбран по 100−30; он не является итоговым приоритетом для всей динамики и аугментаций. Новых обучений не запускалось.'],
        'downloads':[
            {'path':'../early/index.html','label':'Отдельный разбор ранних эпох 0–30, без финальной эпохи'},
            {'path':'method_duration_statistics.csv','label':'Полная статистика шести групп по четырём сравнениям'},
            {'path':'duration_by_augmentation.csv','label':'Как аугментация меняет динамику обучения'},
            {'path':'aug_review_macro.csv','label':'Эффект добавок на эпохах 5, 10, 30 и 100'},
            {'path':'aug_review_by_method.csv','label':'Эффект добавок отдельно для каждого SSL-метода'},
            {'path':'aug_review_by_category.csv','label':'Эффект добавок отдельно для каждой категории'},
            {'path':'repeatability_all_four_contrasts.csv','label':'Повторяемость 108 условий по всем четырём сравнениям'},
            {'path':'quality_spread_by_checkpoint.csv','label':'Разброс качества на каждом из десяти checkpoints'},
            {'path':'../index.html','label':'Подробный прежний отчёт с индивидуальными сравнениями'}]}
    (OUT/'compact_summary.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    # Reconcile headline aggregates with prior independently built macro file.
    old=pd.read_csv(SOURCE/'macro_fixed_grid_seed_intervals.csv')
    check=method.merge(old,on=keys,validate='one_to_one',suffixes=('','_old'))
    np.testing.assert_allclose(check.mean_delta_pp,check.mean_delta_pp_old,atol=1e-12)
    np.testing.assert_allclose(check.macro_seed_sd_pp,check.sd_delta_pp,atol=1e-12)
    assert neg_all==4
    audit=dict(source_sha256=hashlib.sha256(quality_path.read_bytes()).hexdigest(),
        duration_group_rows=len(method),prior_macro_means_sd_reconciled=len(check),
        paired_rows=len(pairs),cell_summaries=len(cells),all_four_contrast_conditions=len(repeat),
        final_augmentation_auc_ap_tests=len(ac),final_augmentation_nominal_p_under_05=int(ac.p_t.lt(.05).sum()),
        all_four_strict_negative_conditions=neg_all,
        independent_replicates_per_condition=3,
        new_significance_tests_added=False,
        note='Descriptive aggregation only. Original 5040-test correction is unchanged.')
    assert ((aug100.macro_seed_ci95_low_pp <= 0) & (aug100.macro_seed_ci95_high_pp >= 0)).all()
    audit['final_augmentation_macro_ci95_all_include_zero']=True
    (OUT/'compact_summary_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
