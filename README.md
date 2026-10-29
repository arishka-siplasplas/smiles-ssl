# VisA: длительность SSL и классификация аномалий

Актуальная ветка: `experiments/visa-ssl-duration`.
Пилот: [протокол VisA](docs/VISA_PROTOCOL_RU.md),
[конфигурация ночного запуска](configs/visa/overnight.json).
Три seed случайного ResNet-18 перед SSL; SimCLR, VICReg и CNN masked reconstruction,
30 эпох, официальные разбиения и фиксированные подвыборки, LogisticRegression и LightGBM.
Новый код: `src/visa_ssl`. Для сборки Kaggle notebook: `python scripts/build_visa_kaggle.py`.

## Прежние эксперименты

Гипотеза: продолжение SSL-обучения может снижать качество классификации
аномалий, даже когда SSL-loss продолжает уменьшаться. Сравниваем SimCLR,
VICReg и masked reconstruction, адаптацию принципа MAE.

Ветка: `experiments/ssl-anomaly-long-pretrain`.

**2026-10-29:** добавлен [отдельный подбор аугментаций для аномалий](docs/AUGMENTATION_CALIBRATION_RU.md).
Для таблиц сначала используется диагностика на внутренних частях train,
затем короткие SSL-пилоты и выбор по чистому validation. Маски Bernoulli
позволяют тестировать слабые воздействия без принудительной замены одного
из шести признаков. Пулы среднего/доноров остаются смешанными и не используют метки.
Предварительная проверка трёх таблиц выполнена; [короткий список, графики и ограничения](reports/augmentation_calibration_v1/SUMMARY_RU.md).
Подготовлен план 180 SSL-пилотов, но этап выбора силы после SSL ещё не запускался.

- [Протокол и критерии проверки гипотезы](docs/PROTOCOL_RU.md).
- [Источники и ограничения данных](docs/DATA_RU.md).
- [Отчёт технической проверки](reports/verification.json).
- [Примеры конфигов](configs/examples) и [фиксированные табличные разбиения](manifests/adbench).

## Установка

Python 3.11+; CPU для проверок, CUDA для длительных запусков с изображениями.
Пакеты из проверенной среды перечислены в `requirements-tested.txt`.
Совместимую пару torch/torchvision для своего CUDA устанавливайте согласно PyTorch.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
```

## Данные и планы

```bash
smiles-ssl download-adbench --output data/adbench
# Разбиения трёх таблиц уже сохранены в manifests/adbench.
# Для отдельной репликации со своим seed используйте НОВУЮ папку:
smiles-ssl prepare --modality table --data-root data/adbench \
  --output manifests/adbench_replication --seed 20261027

# Зафиксированное зеркало архивов MVTec AD, ~5.3 GB; SHA256 проверяется.
smiles-ssl download-mvtec --output data/mvtec
# Реальные разбиения уже сохранены в manifests/mvtec.
# Для независимой репликации:
smiles-ssl prepare --modality image --data-root data/mvtec \
  --output manifests/mvtec_replication --seed 20261027

# Эти команды сохраняют исходные планы v1, обучение не запускают.
# Для новых табличных запусков используйте процедуру калибровки по ссылке выше.
smiles-ssl plan --scope primary --output plans/primary
smiles-ssl plan --scope augmentations --modality table --output plans/table_augmentations
smiles-ssl plan --stage pilot --dataset cardio --output plans/pilot_cardio
smiles-ssl plan --stage pilot --dataset bottle --output plans/pilot_bottle
```

Основная серия: **270 запусков** (3 таблицы + 15 категорий × 3 метода × 5 seed).
Полная сетка аугментаций: **3240 запусков**, включая 270 основных;
это 540 табличных и 2700 изображений. Не запускайте всю сетку до измерения
времени и памяти пилота. Каждый run сохраняет тяжёлые checkpoints:
нужна отдельная оценка дискового бюджета и внешнее хранилище.

## Один запуск

```bash
smiles-ssl run --config configs/examples/cardio_simclr.json \
  --manifest manifests/adbench/cardio.json --data-root data/adbench \
  --output runs/cardio_simclr_s4101

# CUDA, та же конфигурация архитектуры для всех SSL-методов:
export CUBLAS_WORKSPACE_CONFIG=:4096:8
smiles-ssl run --config configs/examples/bottle_simclr.json \
  --manifest manifests/mvtec/bottle.json --data-root data/mvtec \
  --output runs/bottle_simclr_s4101 --device cuda
```

Папка результата должна быть новой. Для продолжения после прерывания
повторите ту же команду с `--resume`; код, конфиг и split должны совпадать.
Возобновление сохраняет состояние optimizer, RNG и shuffle.
В данной локальной рабочей папке также доступен уже проверенный интерпретатор:
`PYTHONPATH=src:.deps ../.venv/bin/python -m smiles_ssl.cli ...`.
На другой машине используется обычная установка пакета, описанная выше.

```bash
# Только после завершения серии и фиксации всех настроек:
smiles-ssl evaluate-test --run runs/cardio_simclr_s4101 --data-root data/adbench
smiles-ssl analyze --runs runs --output reports/analysis_v1
```

При обучении оценивается validation. Отдельная команда test использует уже
обученные probes и все заранее заданные checkpoints, без выбора по test.
Повторная test-оценка в той же папке запрещена. Графики PNG/SVG, CSV траекторий
и парные изменения long−short создаёт `analyze`.

## Что хранится в Git

Код, протокол, конфиги, манифесты split, SHA256 источников и компактные отчёты.
`data/`, `runs/`, веса, эмбеддинги и окружения исключены через `.gitignore`.
Каждый run дополнительно сохраняет снимок исходников, commit, версии среды,
препроцессинг, кандидатов и выбранные параметры probes. Веса и данные нужно
сохранять в отдельном хранилище; Git сам по себе их не резервирует.
