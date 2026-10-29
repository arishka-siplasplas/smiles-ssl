# VisA: длительность SSL и качество классификации

Версия `visa-duration-v1`. Основной вопрос: ухудшаются ли фиксированные представления
при продолжении SSL относительно эпох 0, 1 и 5? Это пилот на 30 эпох, не проверка
асимптотического поведения при сотнях эпох. Гипотеза о деградации не предполагается верной заранее.

## Источники и данные

- [Официальная статья VisA](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900389.pdf).
- [Официальный репозиторий Amazon](https://github.com/amazon-science/spot-diff),
  commit `2a692ab575001cbde74d402d897a7286086c6199`.
- Оригинальные JPEG из [официального S3-архива](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar).
- Неизменённый `split_csv/2cls_highshot.csv` скопирован в `manifests/visa/` вместе с лицензиями.
  Это официальный двухклассовый split 60% train / 40% test; одно-классовый split здесь не подходит.

VisA собран отдельно от MVTec; он примерно вдвое больше по числу изображений.
Выбраны **cashew, chewinggum, fryum**: категории с одиночными примерно выровненными
объектами. Категории зафиксированы до измерения качества. Все аномалии берутся по
официальным меткам; «лёгкие» дефекты по маскам или предсказаниям не отбираются.

Validation выделяется стратифицированно из официального train: 20% каждого класса,
один split-seed `20261030` для всех model-seed, методов, эпох и train-дисбалансов.
Официальный test полностью сохранён. Манифест содержит пути всех объектов и SHA-256.

| Категория | Probe train normal/anomaly | Validation normal/anomaly | Test normal/anomaly |
|---|---:|---:|---:|
| cashew | 240 / 48 | 60 / 12 | 200 / 40 |
| chewinggum | 242 / 48 | 60 / 12 | 201 / 40 |
| fryum | 240 / 48 | 60 / 12 | 200 / 40 |

SSL использует только подмножества probe train: **240 изображений** на категорию,
5% аномалий (228/12) или 20% (192/48). При изменении доли изменяется состав SSL,
но сохраняется размер и число шагов: batch 64, последний batch 48, четыре шага/эпоху.
Пулы каждого класса вложены между долями; точные подвыборки фиксируются единожды.
Метки используются только для построения манифеста и downstream-оценки. SSL Dataset
возвращает только изображение или пару изображений; SSL-loss меток не получает.

Классификаторы во всех условиях обучаются на одном полном **probe train**.
Это позволяет менять именно дисбаланс SSL, не смешивая его с изменением количества
размеченных данных у классификатора. Validation/test не используются в SSL даже без меток.
Во время запуска сохраняются SHA-256 исходных байтов всех 1803 изображений.

## Аугментации

Используются параметры из Appendix A статьи
[Anomaly Detection In The Wild: Can SSL Handle Strong Distribution Imbalances?](https://research.latinxinai.org/papers/neurips/2024/pdf/Daniel_Otero.pdf)
и `parse_arguments.py:604–660` из [авторского ZIP-приложения](https://drive.google.com/file/d/195NxZN3-4tT9sOJCiUD3c6OkauZsQ5UA/view).

- `RandomResizedCrop(224, scale=(0.395, 1.0))`;
- `ColorJitter(brightness=.275, contrast=.275, saturation=.275, hue=.1375)`, вероятность 1;
- преобразование в RGB tensor [0,1].

В авторском ZIP отсутствует импортируемый модуль `src/data/data_module.py`. Поэтому
неизвестные детали полного pipeline не приписываются статье: для aspect ratio
используется стандартный torchvision `(3/4, 4/3)`, interpolation=BILINEAR,
antialias=True. Дополнительные flip/blur/grayscale не добавляются. Значения crop/jitter
и вероятность подтверждены текстом и аргументами авторского кода.
Перед энкодером применяется фиксированная стандартная нормировка ImageNet
(это арифметическая нормировка, не предобученные веса).

SimCLR и VICReg используют один класс Dataset: каждый вызов возвращает два
независимых применения той же аугментации. Masked reconstruction применяет тот же
crop/jitter один раз, затем маскирует патчи.
Оценка: RGB → Resize(224,224) → tensor → та же нормировка; никаких случайных преобразований.

## Модели и обучение

Для каждой категории отдельный стандартный torchvision **ResNet-18(weights=None)**,
без изменения stem, с `fc=Identity`. Эмбеддинг: 512 признаков после global average pool.
Seeds: **6101, 6102, 6103**. В пределах seed начальные параметры и BatchNorm buffers
битово совпадают для всех методов/дисбалансов/категорий; SHA-256 проверяется кодом.
Эпоха 0 использует исходные BatchNorm statistics, без скрытой калибровки.

- SimCLR: NT-Xent, temperature .5, negatives из текущего batch.
- VICReg: invariance/variance/covariance, коэффициенты 25/25/1,
  target std=1, epsilon=1e-4.
- Общий projector у SimCLR/VICReg: 512→512→512→128, BN/ReLU после двух скрытых слоёв.
- Masked reconstruction: CNN-адаптация идеи MAE; **не оригинальная ViT-MAE**.
  Patch **16×16**, mask ratio **.75**: ровно 147 из 196 патчей на изображение.
  Замаскированные нормированные пиксели заменяются нулями (цвет mean RGB).
  Энкодер видит плотное изображение с маской, а не последовательность видимых токенов.
  Spatial map 512×7×7 → пять ConvTranspose2d (512→128→64→32→16→3) → 224×224.
  MSE только по скрытым RGB-пикселям, target [0,1]; видимые пиксели не входят в loss.

AdamW lr=.001, weight_decay=.0001, **постоянный LR**, 30 эпох. Это один
продолжающийся запуск с checkpoint-оценками, а не переобучение с разными cosine schedules.
Число шагов и архитектура фиксированы; общий optimizer является выбранным пилотным
режимом, а не заявлением об оптимальных настройках каждого SSL-метода.
CUDA AMP используется для обучения; NT-Xent и VICReg loss вычисляются в float32.
Извлечение признаков float32. Включены deterministic algorithms, отключены TF32 и cudnn benchmark.
Случайные потоки данных переинициализируются детерминированно на границах эпох;
resume сохраняет модель, optimizer, scaler и историю. Версии библиотек и GPU логируются.

## Оценка

Эпохи **0, 1, 2, 5, 10, 20, 30**. Все девять baseline (3 категории × 3 seed)
завершаются **до любого SSL**. Эпоха 0 вычисляется один раз и копируется в условия
с тем же seed/category; при построении таблицы baseline дубликаты не считаются новыми наблюдениями.
На каждой точке энкодер `.eval()` + `inference_mode`, его состояние до/после оценки
проверяется SHA-256. Классификаторы каждый раз обучаются заново:

- LogisticRegression: StandardScaler, обученный только на probe train;
  C ∈ {.01,.1,1,10}, lbfgs, max_iter=3000. Неконвергенция останавливает эксперимент.
- LightGBM 4.6.0: learning_rate=.05, reg_lambda=1, 4 сочетания num_leaves,
  min_child_samples и n_estimators из JSON. CPU, один поток, deterministic=True.

Гиперпараметры выбираются по **validation Average Precision**; при равенстве берётся
первый вариант сетки. Затем выбирается порог, максимизирующий **validation precision
при validation recall ≥ .5**; tie-break: больший recall, затем меньший threshold.
Основная метрика — **test precision аномального класса при этом фиксированном пороге**.
Ограничение recall действует только на validation; test recall может быть ниже .5.
Просто максимизировать precision без ограничения неинформативно: можно предсказать
единственную аномалию. Recall, F1 и Average Precision необходимо читать рядом с precision.

Сохраняются precision/recall/F1 обоих классов, confusion matrix, ROC-AUC,
Average Precision (именно sklearn AP, не трапецеидальная PR-AUC), prevalence,
полная сетка validation-результатов, выбранный порог и вероятности каждого объекта.
Test используется только для заранее определённых измерений и описательных графиков;
выбор сеток, порогов и «лучшей эпохи» по test не выполняется.

## Четыре ранговые метрики

Определения сохранены из прежних `experiments/ssl/src/common.py::geometry_metrics`
и независимой проверки `experiments/cifar10/src/audit_cifar10.py`.
Для матрицы Z ∈ R^(N×512), Zc=Z−mean(Z), λ=eig(ZcᵀZc/N),
ER(w)=exp(−Σ p log p), p=w/Σw:

- **RankMe** = ER(singular_values(Z)), без центрирования;
- **CER** = ER(singular_values(Zc));
- **CovER** = ER(λ);
- **NESum** = Σλ / max(λ).

Используется прежний float64 eigh/clip(negative,0) расчёт; нулевой спектр возвращает
null и collapse flag. Никакая из четырёх метрик не делится на размерность.
Каждая считается на фиксированных полных probe_train, validation и test, отдельно
для raw Z и построчно L2-нормированного Z. L2 выполняется **до центрирования**.
Сохраняются собственные значения, N, D и признаки для независимого пересчёта.
Классификаторы используют raw-признаки; L2-вариант относится к спектральной диагностике.

## Выходы и интерпретация

`results/analysis/`: baseline таблица и график; CSV качества, loss, рангов;
графики качества обоих классификаторов и всех восьми метрик по эпохам;
SSL-loss на отдельных шкалах; четыре ранга по эпохам для raw/L2;
scatter precision/AP против каждого ранга с отдельными траекториями seed.
На графиках среднее ± sample SD по трём seed, а не доверительный интервал.
Парные Δ сохраняются для 0→1, 0→5, 0→30, 1→30, 5→30 и 10→30.
Категории анализируются отдельно, не используются как дополнительные seed.

В каждой точке сохраняются embeddings, labels, IDs, predictions, fitted probes и metrics.
Начальные веса и конечные модели сохраняются; промежуточные полные веса не архивируются
на каждой оценочной эпохе, чтобы избежать ограничения Kaggle по размеру output.
Для незавершённой задачи сохраняется последний полный checkpoint для resume.
Ночью планируется 54 SSL-run (3 категории × 3 seed × 2 доли × 3 метода),
по одному процессу на GPU, бюджет 9 часов с сохранением на границе эпохи.

Три seed показывают вариацию инициализации при одной фиксированной выборке.
При 12 validation-аномалиях и 40 test-аномалиях precision/threshold шумные;
это ограничение пилота, поэтому рядом публикуются AP, recall и индивидуальные seed.
Отрицательные парные Δ не доказывают универсальную деградацию SSL.

## Запуск

```bash
PYTHONPATH=src python -m pytest tests/test_visa.py -q
python scripts/build_visa_kaggle.py
kaggle kernels push -p kaggle/bundle
kaggle kernels status arinabazukina/visa-ssl-duration-overnight-v1
kaggle kernels logs arinabazukina/visa-ssl-duration-overnight-v1 --follow
kaggle kernels output arinabazukina/visa-ssl-duration-overnight-v1 -p runs/visa-kaggle
```

Notebook содержит неизменяемый снимок исходников и манифестов. Он скачивает изображения
напрямую с официального S3, проверяет наличие каждого пути и сохраняет SHA-256.
Для локального использования: `python scripts/prepare_visa.py --manifest manifests/visa/fixed.json
--output data/visa --archive /path/VisA_20220922.tar`.

## Подтверждённый ночной запуск

[Приватный notebook, version 1](https://www.kaggle.com/code/arinabazukina/visa-ssl-duration-overnight-v1)
запущен из commit `272d869`. Живой журнал подтвердил две CUDA GPU,
torch 2.11.0+cu128, совпадение SHA-256 всех 1803 изображений,
завершение всех девяти baseline без ошибок и переход к SSL.
Подробности проверки: `reports/visa-launch.json`.

В конце сессии скачать outputs командой выше. Основная baseline-таблица лежит
в `visa-experiment/results/analysis/BASELINE.md`, графики и CSV — рядом;
`status.json` показывает, все ли 54 SSL-run завершены или достигнут лимит времени.
Остановка локального клиента `kernels logs --follow` не останавливает Kaggle.
