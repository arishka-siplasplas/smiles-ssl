# Воспроизведение анализа VisA v2

Команды выполняются из корня `smiles-ssl` после установки `pip install -e '.[test]'`.
Проверенные версии библиотек перечислены в `requirements-tested.txt`;
`statsmodels` используется для независимой проверки поправки Holm.
Скрипты читают результаты завершённых запусков и не обучают модели.

## Входы

- `reports/visa-v2-block4/quality.csv`, `ranks.csv`, `loss.csv` — финальные
  накопленные результаты всех 324 SSL-запусков и 36 baseline.
- `reports/visa-v2-block4/audit.json` и подкаталоги архитектур — аудит и статусы.
- `manifests/visa-v2/fixed.json` — фиксированный состав разбиений.

Эти входы уже сохранены в Git. Промежуточные блоки не нужно объединять
с block4: это повторило бы одни и те же запуски.

## Основная последовательность

```bash
python scripts/analyze_visa_v2_quality.py
python scripts/analyze_visa_v2_ranks.py
python scripts/build_visa_v2_report.py

python scripts/analyze_visa_v2_statistics.py
python scripts/analyze_visa_v2_early.py
python scripts/analyze_visa_v2_augmentations.py
python scripts/summarize_visa_v2_findings.py

python scripts/build_visa_v2_table_data.py
python scripts/analyze_visa_v2_augmentation_interactions.py
python scripts/build_visa_v2_overview_data.py
python scripts/build_visa_v2_tables.py
```

Результаты описательного анализа качества и рангов находятся в
`reports/visa-v2-final/`. Статистика записывается в
`reports/visa-v2-statistics-2026-11-14/`, актуальный автономный HTML — в
`outputs/visa-v2-stats-20261114/visa_v2_statistics.html`.
Повторное выполнение обновляет соответствующие выходы.

Генератор `analyze_visa_v2_augmentations.py` создаёт таблицы `summary/aug_review_*`.
Они необходимы для последующих сводок и объединённой таблицы.

## Статистические таблицы

Основной актуальный источник —
`reports/visa-v2-statistics-2026-11-14/tables/data_table_payload_with_interactions.json`:

- `columns` и `rows`: 19 800 сравнений с парными значениями трёх seed,
  средним, выборочным SD, номинальным 95% интервалом, исходным p-value,
  поправкой Holm на всю семью и проверкой перестановкой знаков.
- `summary_columns` и `summary_rows`: 1540 сводок на фиксированной сетке условий.
- `meta`: единицы, seed, контрасты, размер семьи проверок и происхождение данных.

Это массивы колонок и строк; их можно записать в CSV без пересчёта статистик.
Числа в JSON сохраняются с точностью до 10 десятичных знаков.
Предыдущий `tables/all_paired_statistics.csv` содержит 19 080 проверок
до добавления 720 взаимодействий. Для актуальных поправленных p-value
используйте `data_table_payload_with_interactions.json`.

Сравнения длительности: 0→1/2/5/10/20/30/100,
1→2, 2→5, 5→10, 10→20, 20→30, 5→100, 10→100, 30→100.
Эффекты Rotate и CutPaste относительно A0 сравниваются на эпохах 5, 10, 30, 100.
Взаимодействие определяется как `(100−10)_augmentation − (100−10)_A0`.
Метрики: ROC-AUC, AP, precision, recall и F1 аномального класса;
классификаторы: Logistic Regression и LightGBM.

Все изменения выражены в процентных пунктах. Число независимых повторов — три.
Групповые сводки сначала усредняют фиксированные условия внутри каждого seed.
Интервалы условны на фиксированных объектах и нормальной модели парных разностей;
при трёх seed проверить эту модель нельзя. Номинальные интервалы не являются
одновременными интервалами всех проверок.

## Предсказания и bootstrap по объектам

Дополнительные команды требуют локальных `runs/visa-ssl-v2-*` с
`identity.json`, `epoch_*/predictions.npz` и `epoch_*/embeddings.npz`.
Эти тяжёлые артефакты не входят в Git. Embeddings используются для проверки
порядка объектов и меток относительно manifest.

После первых трёх команд основной последовательности можно выполнить:

```bash
python scripts/inventory_visa_v2_predictions.py
python scripts/bootstrap_visa_v2_test_objects.py
```

Затем запустите основную последовательность начиная с
`analyze_visa_v2_statistics.py`, чтобы включить готовые bootstrap-интервалы
в сопоставление источников неопределённости. Без локальных предсказаний
основной анализ по seed воспроизводится отдельно.

Параметры bootstrap: 3000 перевыборок, seed 20261114, сравнение 100−30.
В исходном локальном снимке доступны 67 из 108 полных троек условий.
Этот интервал условен на обученных моделях и не объединяет неопределённость
seed и test-объектов в один интервал.

## Дополнительные отчёты и проверки

Генераторы `build_visa_v2_early_report.py`, `build_visa_v2_compact_summary.py`,
`plot_visa_v2_statistics.py` и `build_visa_v2_statistics_report.py` строят
отдельные представления соответствующих промежуточных результатов.
`plot_visa_overlays.py` и `visa_overlays.template.html` относятся к пилоту v1.

```bash
pytest -q
```

Скрипты дополнительно проверяют полноту сетки, парность seed, контрольные суммы,
согласованность с предыдущими расчётами и статистические формулы через SciPy
и statsmodels. Генерируемые HTML, CSV, JSON, рисунки и архивы остаются локально;
для пересылки результатов используется отдельный комплект.
