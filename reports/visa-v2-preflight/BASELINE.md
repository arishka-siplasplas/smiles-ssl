# VisA v2 random encoder baseline

Mean ± sample SD across SSL seeds; fixed official test. Thresholds selected on validation.

| Architecture | Category | Classifier | n | Precision | ROC-AUC |
|---|---|---|---:|---:|---:|
| deit_tiny | candle | lightgbm | 3 | 0.7743 ± 0.0595 | 0.9452 ± 0.0082 |
| deit_tiny | candle | logistic_regression | 3 | 0.7008 ± 0.0472 | 0.9682 ± 0.0061 |
| deit_tiny | capsules | lightgbm | 3 | 0.3197 ± 0.0844 | 0.7383 ± 0.0323 |
| deit_tiny | capsules | logistic_regression | 3 | 0.5615 ± 0.0154 | 0.8559 ± 0.0270 |
| deit_tiny | cashew | lightgbm | 3 | 0.7082 ± 0.0529 | 0.9657 ± 0.0064 |
| deit_tiny | cashew | logistic_regression | 3 | 0.7432 ± 0.0331 | 0.9662 ± 0.0039 |
| deit_tiny | chewinggum | lightgbm | 3 | 0.2558 ± 0.0422 | 0.6316 ± 0.0468 |
| deit_tiny | chewinggum | logistic_regression | 3 | 0.3439 ± 0.0769 | 0.6965 ± 0.0047 |
| deit_tiny | fryum | lightgbm | 3 | 0.7099 ± 0.0294 | 0.9420 ± 0.0131 |
| deit_tiny | fryum | logistic_regression | 3 | 0.7334 ± 0.0720 | 0.9422 ± 0.0088 |
| deit_tiny | pcb1 | lightgbm | 3 | 0.6433 ± 0.1501 | 0.9032 ± 0.0194 |
| deit_tiny | pcb1 | logistic_regression | 3 | 0.8175 ± 0.0620 | 0.9011 ± 0.0107 |
| resnet18 | candle | lightgbm | 3 | 0.9479 ± 0.0902 | 0.9770 ± 0.0071 |
| resnet18 | candle | logistic_regression | 3 | 0.8892 ± 0.0826 | 0.9786 ± 0.0066 |
| resnet18 | capsules | lightgbm | 3 | 0.4150 ± 0.0309 | 0.7788 ± 0.0090 |
| resnet18 | capsules | logistic_regression | 3 | 0.4853 ± 0.1696 | 0.8279 ± 0.0445 |
| resnet18 | cashew | lightgbm | 3 | 0.6776 ± 0.0687 | 0.9552 ± 0.0057 |
| resnet18 | cashew | logistic_regression | 3 | 0.7444 ± 0.0473 | 0.9666 ± 0.0073 |
| resnet18 | chewinggum | lightgbm | 3 | 0.2278 ± 0.0440 | 0.6745 ± 0.0148 |
| resnet18 | chewinggum | logistic_regression | 3 | 0.3067 ± 0.0376 | 0.6796 ± 0.0303 |
| resnet18 | fryum | lightgbm | 3 | 0.7031 ± 0.0890 | 0.8616 ± 0.0252 |
| resnet18 | fryum | logistic_regression | 3 | 0.7217 ± 0.0084 | 0.8948 ± 0.0189 |
| resnet18 | pcb1 | lightgbm | 3 | 0.5589 ± 0.0899 | 0.9458 ± 0.0067 |
| resnet18 | pcb1 | logistic_regression | 3 | 0.7820 ± 0.0343 | 0.9536 ± 0.0064 |
