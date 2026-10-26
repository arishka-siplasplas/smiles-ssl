# Данные и происхождение

## ADBench

Используются NPZ из [официального репозитория ADBench](https://github.com/Minqi824/ADBench)
на commit `3dac8221081e190f157d78e93bfa8867f90d0965`.
Точная ссылка и SHA256 каждого файла сохранены в
[`manifests/adbench/sources.json`](../manifests/adbench/sources.json).
NPZ читаются с `allow_pickle=False`; X должен быть конечным числовым массивом,
y — бинарным. Размерности и число аномалий проверены на скачанных файлах.

| Датасет | N | D | Аномалии | Train normal/anomaly | Val normal/anomaly | Test normal/anomaly |
|---|---:|---:|---:|---:|---:|---:|
| cardio | 1831 | 21 | 176 (9.61%) | 993/106 | 331/35 | 331/35 |
| mammography | 11183 | 6 | 260 (2.32%) | 6551/159 | 2186/51 | 2186/50 |
| annthyroid | 7200 | 6 | 534 (7.42%) | 4000/320 | 1333/107 | 1333/107 |

Обнаружено повторяющихся строк сверх уникальных: cardio **9**, mammography
**3335**, annthyroid **138**. Они остаются в данных, но не пересекают split.
В частности, разбиение mammography без группировки может заметно искажать
оценку обобщения. Сходство, не являющееся точным совпадением всех признаков,
этой проверкой не устраняется. Patient IDs в NPZ нет; разделение по пациентам
не заявляется.

Данные используются в том виде, как выпущены ADBench. Возможное
преобразование признаков авторами источника нельзя отменить. Наши
дополнительные mean/std, доноры и выбор допустимых для noise координат
вычисляются только по новому train. Исходными координатами здесь называются
координаты скачанного NPZ, не физические единицы исходных медицинских приборов.

## MVTec AD

Нужны оригинальные изображения, не предвычисленные ADBench-эмбеддинги.
[Официальная страница](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
описывает 15 категорий и исходный normal-only train.
Лицензия изображений — CC BY-NC-SA 4.0; изображения и маски не размещаются
в этом Git-репозитории.

Ожидаемая структура:

```text
data/mvtec/bottle/train/good/*.png
data/mvtec/bottle/test/good/*.png
data/mvtec/bottle/test/broken_large/*.png
... другие типы дефектов и категории ...
```

Категории: bottle, cable, capsule, carpet, grid, hazelnut, leather,
metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper.
Подготовка создаёт один manifest на категорию с SHA256 и точным
распределением дефектов. Все **5354 изображения** включены ровно в одну
часть; реальные manifests сохранены в `manifests/mvtec/`.
Файл `reports/verification.json` отделяет проверки на синтетических fixtures
от проверок на реальных данных.

2026-10-26 mydrive-ссылки общего архива из
[Anomalib](https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/data/datamodules/image/mvtecad.py)
и отдельной категории bottle вернули HTTP 404. Использовано стороннее
[зеркало архивов категорий](https://huggingface.co/datasets/XavierZXY/MvT-tar/tree/dc1c313de9de1be8e41fe6818dc43caac8601efa)
на commit `dc1c313de9de1be8e41fe6818dc43caac8601efa`, с исходной структурой
train/test/ground_truth. Все 15 архивов проверены по SHA256 и размеру,
заданным в Git LFS-указателях этой версии; значения зафиксированы в
`src/smiles_ssl/download.py` и `manifests/mvtec/sources.json`.
Это контроль целостности конкретного зеркала; независимую побайтовую
сверку с недоступным официальным архивом выполнить нельзя. После распаковки
архивы удаляются, изображения остаются локально вне Git. Для повторной
загрузки есть `smiles-ssl download-mvtec`; по умолчанию два потока.

Везде обозначение **MVTec AD MODIFIED mixed-class**, задача — бинарная
классификация изображения. Pixel-level anomaly localization не оценивается.
