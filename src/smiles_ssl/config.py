from copy import deepcopy

from .data import CATEGORIES

CHECKPOINTS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000]
SEEDS = [4101, 4102, 4103, 4104, 4105]


def default_config(modality, dataset, method, seed=4101):
    return {'protocol_version': 1, 'stage': 'full', 'modality': modality, 'dataset': dataset,
            'method': method, 'seed': seed, 'split_seed': 20261026,
            'epochs': 1000, 'checkpoints': CHECKPOINTS, 'short_epoch': 100,
            'batch_size': 256 if modality == 'table' else 64,
            'learning_rate': .001 if modality == 'table' else .0003,
            'weight_decay': .0001, 'warmup_epochs': 10, 'image_size': 224,
            'initialization': 'random', 'num_workers': 0, 'threads': 1,
            'augmentation': {'family': 'mean', 'fraction': .3 if modality == 'table' else .5,
                             'sigma': 0., 'patch_size': 16},
            'probe': {'logreg_C': [.01, .1, 1., 10., 100.],
                      'class_weights': [None, 'balanced'], 'logreg_max_iter': 10000,
                      'catboost_depth': [4, 6], 'catboost_l2': [1., 3.], 'catboost_iterations': 300}}


def validate_config(c):
    if c['modality'] not in ('table', 'image') or c['method'] not in ('simclr', 'vicreg', 'masked_reconstruction'):
        raise ValueError('Unknown modality/method.')
    if c['initialization'] != 'random' or c['protocol_version'] != 1:
        raise ValueError('v1 uses random initialization only.')
    if c['epochs'] < 1 or c['batch_size'] < 2 or c['learning_rate'] <= 0:
        raise ValueError('Invalid optimization settings.')
    if c.get('schedule_epochs', c['epochs']) < c['epochs']:
        raise ValueError('Learning-rate schedule horizon must cover the run.')
    if sorted(set(c['checkpoints'])) != c['checkpoints'] or c['checkpoints'][0] != 0 or c['checkpoints'][-1] != c['epochs']:
        raise ValueError('Checkpoints must be sorted, unique and include epoch 0 and final epoch.')
    if c['short_epoch'] not in c['checkpoints']:
        raise ValueError('Short comparison epoch must be saved.')
    a = c['augmentation']
    sampling = a.get('mask_sampling', 'fixed_count')
    if sampling not in ('fixed_count', 'bernoulli') or (sampling == 'bernoulli' and c['modality'] != 'table'):
        raise ValueError('Bernoulli masks are defined only for table calibration.')
    identity = sampling == 'bernoulli' and a['fraction'] == 0 and c['method'] != 'masked_reconstruction'
    if a['family'] not in ('zero', 'mean', 'donor', 'noise') or not (identity or 0 < a['fraction'] < 1):
        raise ValueError('Invalid augmentation.')
    if a['family'] == 'noise' and a['sigma'] <= 0:
        raise ValueError('Noise sigma must be positive.')
    if c['image_size'] < 32 or c['image_size'] % a['patch_size'] or c['image_size'] % 32:
        raise ValueError('Image size must be divisible by 32 and patch size.')


def grid(scope='primary', modality=None):
    configs = []
    for mod, datasets in [('table', ['cardio', 'mammography', 'annthyroid']), ('image', CATEGORIES)]:
        if modality and mod != modality:
            continue
        for dataset in datasets:
            for method in ['simclr', 'vicreg', 'masked_reconstruction']:
                for seed in SEEDS:
                    base = default_config(mod, dataset, method, seed)
                    if scope == 'primary':
                        configs.append(base)
                        continue
                    for family in ['zero', 'mean', 'donor', 'noise']:
                        levels = [.1, .3, .5] if mod == 'table' else [.25, .5, .75]
                        if family == 'noise':
                            levels = [.05, .15, .3] if mod == 'table' else [.02, .05, .1]
                        for level in levels:
                            c = deepcopy(base)
                            c['augmentation']['family'] = family
                            c['augmentation']['sigma' if family == 'noise' else 'fraction'] = level
                            configs.append(c)
    return configs


def run_id(c):
    a = c['augmentation']
    suffix = '_bernoulli' if a.get('mask_sampling') == 'bernoulli' else ''
    return (f"{c['modality']}_{c['dataset']}_{c['method']}_{a['family']}"
            f"_p{a['fraction']:g}_s{a['sigma']:g}_seed{c['seed']}{suffix}")
