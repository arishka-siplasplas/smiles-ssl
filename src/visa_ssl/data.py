"""Official boundaries and immutable, seed-independent sample identities."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def make_manifest(split_csv, config):
    rows = list(csv.DictReader(Path(split_csv).open()))
    result = {'official_csv_sha256': hashlib.sha256(Path(split_csv).read_bytes()).hexdigest(),
              'official_commit': config['official_commit'], 'categories': {}}
    for category in config['categories']:
        selected = [dict(path=r['image'], label=int(r['label'] == 'anomaly'),
                         official_split=r['split']) for r in rows if r['object'] == category]
        assert len({r['path'] for r in selected}) == len(selected)
        assert {r['official_split'] for r in selected} == {'train', 'test'}
        assert all(r['label'] in ('normal', 'anomaly') for r in rows if r['object'] == category)
        train, validation = [], []
        # One manifest for ALL model seeds; strata preserve both classes.
        rng = np.random.default_rng(config['split_seed'])
        for label in (0, 1):
            group = sorted([r for r in selected if r['official_split'] == 'train'
                            and r['label'] == label], key=lambda r: r['path'])
            group = [group[i] for i in rng.permutation(len(group))]
            n = round(len(group) * config['validation_fraction'])
            validation.extend(group[:n])
            train.extend(group[n:])
        test = [r for r in selected if r['official_split'] == 'test']
        ssl = {}
        for fraction in config.get('anomaly_fractions', []):
            counts = {1: round(config['ssl_size'] * fraction)}
            counts[0] = config['ssl_size'] - counts[1]
            subset = []
            for label in (0, 1):
                pool = [r for r in train if r['label'] == label]
                if counts[label] > len(pool) or counts[label] < 1:
                    raise ValueError(f'Insufficient {category} class {label}: {counts[label]} / {len(pool)}')
                subset.extend(pool[:counts[label]])
            ssl[str(fraction)] = sorted(subset, key=lambda r: r['path'])
        if config.get('ssl_sampling') == 'full_train':
            ssl['full_train'] = sorted(train, key=lambda r: r['path'])
        sets = {'probe_train': train, 'validation': validation, 'test': test}
        sets = {k: sorted(v, key=lambda r: r['path']) for k, v in sets.items()}
        ids = {k: {r['path'] for r in v} for k, v in sets.items()}
        assert not ids['probe_train'] & ids['validation']
        assert not (ids['probe_train'] | ids['validation']) & ids['test']
        for group in ssl.values():
            assert {r['path'] for r in group} <= ids['probe_train']
        result['categories'][category] = {**sets, 'ssl': ssl}
    result['manifest_sha256'] = digest(result)
    return result


class Images(Dataset):
    def __init__(self, root, records, transform, two_views=False, labeled=False):
        self.root, self.records, self.transform = Path(root), records, transform
        self.two_views, self.labeled = two_views, labeled

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        row = self.records[i]
        with Image.open(self.root / row['path']) as source:
            image = source.convert('RGB')
        if self.two_views:
            return self.transform(image), self.transform(image)
        value = self.transform(image)
        return (value, row['label']) if self.labeled else value
