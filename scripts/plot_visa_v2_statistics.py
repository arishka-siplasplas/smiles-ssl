"""Standalone scientific figure: mean changes and nominal seed intervals."""
from pathlib import Path
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'visa-statistics-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parents[1]/'reports/visa-v2-statistics-2026-11-14'
NAMES={'resnet18':'ResNet','deit_tiny':'ViT','baseline':'A0','rotate':'Rotate',
       'cutpaste':'CutPaste','vicreg':'VICReg','simclr':'SimCLR','masked_reconstruction':'MR'}


def main():
    table=pd.read_csv(OUT/'paired_statistics.csv')
    table=table[table.comparison.eq('duration') & table.early_epoch.eq(30) &
                table.classifier.eq('logistic_regression')]
    keys=['category','architecture','augmentation','method']
    chosen=[('cashew','deit_tiny','baseline','vicreg'),
            ('cashew','deit_tiny','rotate','vicreg'),
            ('fryum','deit_tiny','rotate','simclr'),
            ('fryum','resnet18','baseline','vicreg'),
            ('candle','deit_tiny','rotate','vicreg'),
            ('capsules','deit_tiny','baseline','masked_reconstruction'),
            ('capsules','resnet18','rotate','simclr'),
            ('pcb1','deit_tiny','rotate','simclr'),
            ('chewinggum','resnet18','cutpaste','simclr')]
    labels=[' / '.join(NAMES.get(v,v) for v in k) for k in chosen]
    fig,axes=plt.subplots(1,2,figsize=(15.8,7.0),sharey=True)
    fig.subplots_adjust(left=.31,right=.98,top=.80,bottom=.17,wspace=.13)
    for ax,metric,title in zip(axes,['roc_auc','average_precision'],['ROC-AUC','Average precision (AP)']):
        rows=table[table.metric.eq(metric)].set_index(keys).loc[chosen]
        ax.axvline(0,color='#53616f',lw=.8)
        for i,(_,r) in enumerate(rows.iterrows()):
            ax.plot([r.ci95_low_pp,r.ci95_high_pp],[i,i],color='#285d7d',lw=2)
            ax.scatter(r.mean_delta_pp,i,color='#285d7d',s=45,zorder=4)
            ax.scatter([r[f'delta_seed_{s}_pp'] for s in [6101,6102,6103]],
                       [i+.15]*3,color='#53616f',marker='|',s=80,zorder=3)
        ax.set_title(title,fontsize=14,pad=13)
        ax.grid(axis='x',alpha=.18)
        ax.set_xlabel('Изменение 100 − 30, процентные пункты',fontsize=10)
        ax.spines[['top','right','left']].set_visible(False)
    axes[0].set_yticks(np.arange(len(chosen)),labels=labels,fontsize=10)
    axes[0].invert_yaxis()
    axes[1].tick_params(axis='y',left=False)
    axes[0].set_xlim(-40,42)
    axes[1].set_xlim(-70,62)
    fig.suptitle('Разброс изменений между seed: от устойчивых до шумных',fontsize=19,x=.04,ha='left',y=.97)
    fig.text(.04,.90,'VisA v2 · Logistic regression · 3 seed · выбранные примеры, полный набор — в таблице отчёта',fontsize=12,color='#475569')
    fig.text(.04,.06,'Точка — среднее; линия — номинальный 95% t-интервал; штрихи — три парные разности. Фиксированный порог не задан.',fontsize=10)
    fig.text(.04,.025,'Интервалы условны на фиксированных данных и предположении нормальности; не скорректированы за множественные сравнения.',fontsize=10,color='#475569')
    for ext in ('png','svg'):
        fig.savefig(OUT/f'selected_condition_intervals.{ext}',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    main()
