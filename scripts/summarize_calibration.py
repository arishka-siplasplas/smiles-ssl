"""Render the saved train-only diagnostic screens; no models are fitted here."""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from smiles_ssl.io import read_json, write_json


def summarize(root):
    root = Path(root)
    lines = ['# Предварительная проверка аугментаций', '',
             'Это диагностика на внутренних отложенных частях train, **не результаты SSL**.',
             'Метки не участвуют в преобразовании объектов. Validation и test не используются.',
             'Одна маска Bernoulli с общей вероятностью p применяется независимо от класса.', '',
             '| Датасет | Zero: p | Mean: p | Donor: p | Noise: (p, σ) |',
             '|---|---|---|---|---|']
    counts, candidate_runs, identity_runs = {}, 0, 0
    for dataset in ['cardio', 'mammography', 'annthyroid']:
        report = read_json(root / dataset / 'screen.json')
        short = [r for r in report['candidates'] if r['id'] in report['shortlist']]
        candidate_runs += len(short) * 3 * len(report['spec']['pilot_seeds'])
        identity_runs += 2 * len(report['spec']['pilot_seeds'])
        cells = []
        for family in ['zero', 'mean', 'donor', 'noise']:
            augments = [r['augmentation'] for r in short if r['augmentation']['family'] == family]
            cells.append(', '.join(f"({a['fraction']:g}, {a['sigma']:g})" if family == 'noise'
                                   else f"{a['fraction']:g}" for a in augments) or '—')
        lines.append('| ' + ' | '.join([dataset] + cells) + ' |')
        counts[dataset] = dict(Counter(r['risk'] for r in report['candidates']))
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for ax, family in zip(axes.flat, ['zero', 'mean', 'donor', 'noise']):
            rows = [r for r in report['candidates'] if r['augmentation']['family'] == family and 'probes' in r]
            for probe, color in [('logreg', '#276EBA'), ('catboost', '#C95332')]:
                ax.plot(range(len(rows)), [r['probes'][probe]['mean_ap_drop'] for r in rows],
                        'o-', color=color, label=probe)
            labels = [f"p={r['augmentation']['fraction']:g}" +
                      (f"\nσ={r['augmentation']['sigma']:g}" if family == 'noise' else '') for r in rows]
            ax.set_xticks(range(len(rows)), labels, fontsize=8)
            ax.axhline(report['spec']['ap_drop_tolerance'], color='#777777', linestyle='--', label='AP drop tolerance')
            ax.axhline(0, color='#aaaaaa', linewidth=.7)
            ax.set(title=family, ylabel='Clean AP − corrupted AP (origin labels)')
            ax.grid(alpha=.15)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f'{dataset}: train-only proxy screen, NOT an SSL result\n'
                     'Mean of five corruption draws; no independent-sample confidence intervals')
        fig.savefig(root / dataset / 'ap_drop.png', dpi=150)
        svg = root / dataset / 'ap_drop.svg'
        fig.savefig(svg)
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
        plt.close(fig)
    lines.extend(['', 'Таблица содержит короткий список для SSL-пилотов, а не выбранные оптимальные силы.',
                  'Из непомеченных вариантов взяты крайние по силе внутри каждой семьи.',
                  'Порог отбраковки фиксирован до расчёта: AP −0.02, TPR −0.05 или FPR +0.02.',
                  'Ухудшение у обоих учителей → flagged; у одного → review; иначе not_flagged.',
                  'Все три статуса и переходы только среди изменённых объектов сохранены в screen.json.', '',
                  'Увеличение риска у диагностического учителя не устанавливает истинный класс',
                  'аугментированного объекта. Эти показатели не заменяют AP после SSL на чистом validation.', '',
                  f'Подготовлено {candidate_runs + identity_runs} пилотов: {candidate_runs} сочетания кандидат×метод×seed и {identity_runs} контролей p=0.',
                  'Пилоты этапа B и длинная серия этим скриптом не запускаются.'])
    (root / 'SUMMARY_RU.md').write_text('\n'.join(lines) + '\n')
    write_json(root / 'risk_counts.json', counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    summarize(parser.parse_args().root)
