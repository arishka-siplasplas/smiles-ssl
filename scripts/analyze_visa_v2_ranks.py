"""Descriptive within-run rank/quality associations for complete VisA v2.

No p-values: repeated checkpoints, shared datasets and duplicated quality for raw/L2
are not independent observations. Each correlation stays within one training run.
"""
from pathlib import Path
import argparse
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ConstantInputWarning

KEYS = ['category', 'architecture', 'augmentation', 'method', 'seed']
RANKS = ['RankMe', 'CER', 'CovER', 'NESum']
METRICS = ['roc_auc', 'average_precision', 'precision_anomaly']
EPOCHS = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100]

def main(source, output):
    output.mkdir(parents=True, exist_ok=True)
    q = pd.read_csv(source / 'quality.csv')
    r = pd.read_csv(source / 'ranks.csv')
    q = q[(q.split == 'test') & (q.method != 'baseline')]
    r = r[(r.split == 'test') & (r.method != 'baseline')]
    assert len(q) == len(r) == 6480
    assert not q.duplicated(KEYS + ['classifier', 'epoch']).any()
    assert not r.duplicated(KEYS + ['normalization', 'epoch']).any()
    merged = q[KEYS + ['epoch','classifier'] + METRICS].merge(
        r[KEYS + ['epoch','normalization'] + RANKS], on=KEYS + ['epoch'], validate='many_to_many')
    run_keys = KEYS + ['classifier', 'normalization']
    corr_records, change_records = [], []
    warnings.simplefilter('ignore', ConstantInputWarning)
    for values, frame in merged.groupby(run_keys):
        meta = dict(zip(run_keys, values))
        frame = frame.sort_values('epoch').set_index('epoch')
        assert frame.index.tolist() == EPOCHS
        for window, epochs in [('all_0_100',EPOCHS), ('trained_1_100',EPOCHS[1:]), ('late_30_100',EPOCHS[-4:])]:
            selected = frame.loc[epochs]
            for rank in RANKS:
                for metric in METRICS:
                    corr_records.append({**meta,'window':window,'n_checkpoints':len(epochs),
                         'rank':rank,'metric':metric,'spearman':float(spearmanr(selected[rank],selected[metric]).statistic)})
        for early in [0, 5, 10, 30]:
            for rank in RANKS:
                for metric in METRICS:
                    rd = float(frame.loc[100,rank]-frame.loc[early,rank])
                    qd = float(100*(frame.loc[100,metric]-frame.loc[early,metric]))
                    change_records.append({**meta,'early_epoch':early,'late_epoch':100,'rank':rank,'metric':metric,
                          'rank_early':float(frame.loc[early,rank]), 'rank_late':float(frame.loc[100,rank]),
                          'quality_early':float(frame.loc[early,metric]), 'quality_late':float(frame.loc[100,metric]),
                          'rank_delta':rd,'quality_delta_pp':qd,'opposite_signs':rd*qd < 0,
                          'rank_up_quality_down':rd > 0 and qd < 0})
    corrs=pd.DataFrame(corr_records)
    changes=pd.DataFrame(change_records)
    corrs.to_csv(output/'rank_within_run_correlations.csv',index=False)
    changes.to_csv(output/'rank_quality_paired_changes.csv',index=False)
    summaries=[]
    for level, grouping in [('overall',[]),('architecture',['architecture']),('architecture_method',['architecture','method'])]:
        keys=grouping+['classifier','normalization','window','rank','metric']
        for values, frame in corrs.groupby(keys):
            vals=frame.spearman.dropna()
            summaries.append({'level':level,**dict(zip(keys,values)), 'n_runs':len(frame),'n_valid':len(vals),
                'median_spearman':float(vals.median()),'q25_spearman':float(vals.quantile(.25)),
                'q75_spearman':float(vals.quantile(.75)), 'positive_runs':int((vals>0).sum()),
                'negative_runs':int((vals<0).sum()),'zero_runs':int((vals==0).sum())})
    summary=pd.DataFrame(summaries)
    summary.to_csv(output/'rank_correlation_summary.csv',index=False)
    cell_keys=[c for c in run_keys if c!='seed']+['early_epoch','late_epoch','rank','metric']
    cell=changes.groupby(cell_keys).agg(n=('seed','nunique'),rank_early=('rank_early','mean'),rank_late=('rank_late','mean'),
         quality_early=('quality_early','mean'),quality_late=('quality_late','mean'),rank_delta=('rank_delta','mean'),
         quality_delta_pp=('quality_delta_pp','mean'),all_three_rank_up_quality_down=('rank_up_quality_down','all'),
         n_opposite=('opposite_signs','sum')).reset_index()
    assert cell.n.eq(3).all()
    cell['mean_opposite_signs']=cell.rank_delta*cell.quality_delta_pp<0
    cell['mean_rank_up_quality_down']=(cell.rank_delta>0)&(cell.quality_delta_pp<0)
    cell.to_csv(output/'rank_quality_cell_changes.csv',index=False)
    counts=cell.groupby(['classifier','normalization','early_epoch','rank','metric']).agg(
         n_cells=('n','size'),mean_opposite=('mean_opposite_signs','sum'),
         mean_rank_up_quality_down=('mean_rank_up_quality_down','sum'),
         all_three_rank_up_quality_down=('all_three_rank_up_quality_down','sum')).reset_index()
    counts.to_csv(output/'rank_quality_discordance_summary.csv',index=False)
    candidates=cell[(cell.all_three_rank_up_quality_down)&(cell.normalization=='raw')&
                    (cell.classifier=='logistic_regression')&(cell['rank']=='RankMe')&
                    (cell.metric=='roc_auc')].sort_values('quality_delta_pp')
    all_rank_example=cell[(cell.category=='fryum')&(cell.architecture=='deit_tiny')&
        (cell.augmentation=='rotate')&(cell.method=='simclr')&(cell.classifier=='logistic_regression')&
        (cell.normalization=='raw')&(cell.early_epoch==30)]
    assert len(all_rank_example)==12 and all_rank_example.all_three_rank_up_quality_down.all()
    all_rank_example.to_csv(output/'rank_counterexample_all_metrics.csv',index=False)
    findings={'source':str(source.resolve()),'n_training_runs':324,
       'note':'Descriptive distributions of within-run correlations. No independence or significance claim. Baseline epoch 0 is a shared initialization; use trained-only window as sensitivity analysis.',
       'overall_auc_lr':summary.astype(object).where(pd.notna(summary),None)[(summary.level=='overall')&(summary.classifier=='logistic_regression')&(summary.metric=='roc_auc')&(summary.window!='late_30_100')].to_dict('records'),
       'all_rank_counterexample_fryum_vit_rotate_simclr_30_100':all_rank_example.to_dict('records'),
       'strong_counterexamples_rankme_raw_auc_lr':candidates.head(12).to_dict('records'),
       'auc_lr_discordance':counts[(counts.classifier=='logistic_regression')&(counts.metric=='roc_auc')].to_dict('records')}
    (output/'rank_findings.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(summary[(summary.level=='overall')&(summary.classifier=='logistic_regression')&(summary.metric=='roc_auc')&(summary.window!='late_30_100')][['normalization','window','rank','median_spearman','q25_spearman','q75_spearman','positive_runs','negative_runs']].to_string(index=False))
    print('\nCOUNTEREXAMPLES\n',candidates.head(8).to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();root=Path(__file__).resolve().parents[1]
    parser.add_argument('--source',type=Path,default=root/'reports/visa-v2-block4')
    parser.add_argument('--output',type=Path,default=root/'reports/visa-v2-final/data')
    args=parser.parse_args();main(args.source,args.output)
