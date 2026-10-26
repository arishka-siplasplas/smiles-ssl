"""Fixed, disjoint, duplicate-aware splits; train-only preprocessing."""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import urllib.request

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .io import digest, write_json

ADB_COMMIT = '3dac8221081e190f157d78e93bfa8867f90d0965'
ADB_FILES = {'cardio': '6_cardio.npz', 'mammography': '23_mammography.npz',
             'annthyroid': '2_annthyroid.npz'}
EXPECTED = {'cardio': (1831, 21, 176), 'mammography': (11183, 6, 260),
            'annthyroid': (7200, 6, 534)}
ADB_SHA256 = {'cardio': 'bc5009bde6930b4d298b594f31b34485a9a17210cff0e8fb00a8c3068b885a2d',
              'mammography': 'a0308ac9712d78321073b1c2dc98dc37665c0f35c1cc33019416fe8f8d7de7ea',
              'annthyroid': 'e738f7c58a8740d2c4aac4c71ff49ce5e1d1aa16f44178135fc9c09df05fad1b'}
CATEGORIES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
              'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
SPLITS = ('train', 'validation', 'test')


def download_adbench(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, filename in ADB_FILES.items():
        url = f'https://raw.githubusercontent.com/Minqi824/ADBench/{ADB_COMMIT}/adbench/datasets/Classical/{filename}'
        path = root / filename
        if not path.exists():
            tmp = path.with_suffix('.download')
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(path)
        x, y = load_table(path)
        if (len(x), x.shape[1], int(y.sum())) != EXPECTED[name] or digest(path) != ADB_SHA256[name]:
            raise ValueError(f'Unexpected ADBench version: {name}')
        records[name] = {'url': url, 'sha256': digest(path), 'shape': list(x.shape),
                         'anomalies': int(y.sum())}
    write_json(root / 'sources.json', records)
    return records


def load_table(path):
    with np.load(path, allow_pickle=False) as data:
        x, y = np.asarray(data['X'], dtype=np.float64), np.asarray(data['y']).reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError('Expected finite numeric X of shape (N,D) and matching y.')
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Labels must be 0=normal, 1=anomaly, with both classes.')
    return x, y.astype(np.int64)


def grouped_split(labels, strata, groups, seed, fractions=(.6, .2, .2)):
    """Allocate whole duplicate groups within strata, with all strata in all splits.

    A mixed-label duplicate table group is stratified by its majority label.
    Exact row fractions are impossible for indivisible duplicate groups.
    """
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1) or min(fractions) <= 0:
        raise ValueError('Need three positive fractions summing to one.')
    members = defaultdict(list)
    for i, group in enumerate(groups):
        members[group].append(i)
    by_stratum = defaultdict(list)
    for indices in members.values():
        s = Counter(str(strata[i]) for i in indices).most_common(1)[0][0]
        by_stratum[s].append(indices)
    rng = np.random.default_rng(seed)
    result = {s: [] for s in SPLITS}
    for stratum in sorted(by_stratum):
        units = by_stratum[stratum]
        if len(units) < 3:
            raise ValueError(f'Stratum {stratum} has fewer than 3 independent groups.')
        rng.shuffle(units)
        units.sort(key=len, reverse=True)  # stable: random tie breaks
        target = np.asarray(fractions) * sum(map(len, units))
        counts = np.zeros(3)
        for j, indices in enumerate(units):
            dest = j if j < 3 else int(np.argmax(target - counts))
            result[SPLITS[dest]].extend(indices)
            counts[dest] += len(indices)
    for split, indices in result.items():
        indices.sort()
        if set(np.asarray(labels)[indices]) != {0, 1}:
            raise ValueError(f'{split} does not contain both classes.')
    return result


def split_summary(y, strata, splits):
    return {s: {'n': len(ids), 'normal': int((y[ids] == 0).sum()),
                'anomaly': int((y[ids] == 1).sum()),
                'fraction': len(ids) / len(y),
                'strata': dict(Counter(str(strata[i]) for i in ids))}
            for s, ids in splits.items()}


def prepare_table(path, name, seed=20261026, noise_columns=None):
    x, y = load_table(path)
    if name in ADB_SHA256 and digest(path) != ADB_SHA256[name]:
        raise ValueError('ADBench checksum differs from the pinned release.')
    groups = [hashlib.sha256(row.tobytes()).hexdigest() for row in x]
    splits = grouped_split(y, y, groups, seed)
    train = x[splits['train']]
    mean, std = train.mean(0), train.std(0)
    # Conservative automatic eligibility; indices are recorded for semantic review.
    unique = [len(np.unique(train[:, j])) for j in range(x.shape[1])]
    integer = [bool(np.allclose(train[:, j], np.rint(train[:, j]), atol=1e-7, rtol=0))
               for j in range(x.shape[1])]
    candidates = [j for j in range(x.shape[1]) if unique[j] > 20 and not integer[j] and std[j] > 0]
    policy = 'train_noninteger_gt20_unique'
    if noise_columns is None:
        noise_columns = candidates
    else:
        if not noise_columns or len(set(noise_columns)) != len(noise_columns) or not set(noise_columns) <= set(candidates):
            raise ValueError('Noise allowlist must be a nonempty unique subset of train-eligible columns.')
        noise_columns = sorted(noise_columns)
        policy = 'explicit_allowlist_restricted_to_train_eligible'
    return {'protocol': 'adbench-mixed-class-grouped-v1', 'modality': 'table',
            'dataset': name, 'seed': seed, 'fractions': [.6, .2, .2],
            'source_sha256': digest(path), 'source_filename': Path(path).name,
            'splits': splits, 'summary': split_summary(y, y, splits),
            'duplicate_rows': len(x) - len(set(groups)),
            'preprocessing': {'mean': mean.tolist(), 'scale': np.where(std > 0, std, 1).tolist(),
                              'noise_columns': noise_columns, 'unique_train': unique,
                              'integer_train': integer, 'noise_policy': policy},
            'zero_mean_replacement_distance': float(np.linalg.norm(mean / np.where(std > 0, std, 1)))}


def prepare_mvtec(root, category, seed=20261026):
    root = Path(root)
    records = []
    for origin in ('train', 'test'):
        for path in sorted((root / category / origin).glob('*/*.png')):
            defect = path.parent.name
            # Hash decoded pixels so differently encoded duplicates stay together.
            with Image.open(path) as im:
                pixels = np.asarray(im.convert('RGB'))
            pixel_hash = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
            records.append({'path': path.relative_to(root).as_posix(), 'origin': origin,
                            'defect_type': defect, 'label': int(defect != 'good'),
                            'sha256': digest(path), 'pixel_sha256': pixel_hash})
    if not records:
        raise ValueError(f'No images found in {root / category}; expected original MVTec AD layout.')
    y = np.asarray([r['label'] for r in records])
    strata = [r['defect_type'] for r in records]
    splits = grouped_split(y, strata, [r['pixel_sha256'] for r in records], seed)
    return {'protocol': 'mvtec-ad-MODIFIED-mixed-class-v1', 'modality': 'image',
            'dataset': category, 'seed': seed, 'fractions': [.6, .2, .2],
            'records': records, 'splits': splits, 'summary': split_summary(y, strata, splits),
            'moved_original_test_to_train': [records[i]['path'] for i in splits['train']
                                           if records[i]['origin'] == 'test']}


def validate_split_integrity(manifest, y, groups):
    splits = manifest['splits']
    if set(splits) != set(SPLITS):
        raise ValueError('Manifest must contain exactly train/validation/test.')
    flat = sum((splits[s] for s in SPLITS), [])
    if sorted(flat) != list(range(len(y))):
        raise ValueError('Split indices overlap, are missing, or are out of bounds.')
    seen = set()
    for s in SPLITS:
        ids = splits[s]
        group_set = {groups[i] for i in ids}
        if group_set & seen:
            raise ValueError('Duplicate objects cross split boundaries.')
        seen.update(group_set)
        if set(np.asarray(y)[ids]) != {0, 1}:
            raise ValueError(f'{s} does not contain both classes.')


def load_splits(manifest, data_root, image_size=224, include_test=False):
    names = SPLITS if include_test else SPLITS[:2]
    if manifest['modality'] == 'table':
        path = Path(data_root) / manifest['source_filename']
        if digest(path) != manifest['source_sha256']:
            raise ValueError('Source hash differs from prepared split.')
        x, y = load_table(path)
        validate_split_integrity(manifest, y, [hashlib.sha256(row.tobytes()).hexdigest() for row in x])
        prep = manifest['preprocessing']
        normalized = ((x - prep['mean']) / prep['scale']).astype(np.float32)
        return ({s: torch.from_numpy(normalized[manifest['splits'][s]]) for s in names},
                {s: y[manifest['splits'][s]] for s in names})
    validate_split_integrity(manifest, [r['label'] for r in manifest['records']],
                             [r['pixel_sha256'] for r in manifest['records']])
    datasets, labels = {}, {}
    for s in names:
        records = [manifest['records'][i] for i in manifest['splits'][s]]
        for r in records:
            if digest(Path(data_root) / r['path']) != r['sha256']:
                raise ValueError(f"Source hash mismatch: {r['path']}")
        datasets[s] = Images(data_root, records, image_size)
        labels[s] = np.asarray([r['label'] for r in records])
    return datasets, labels


class Images(Dataset):
    def __init__(self, root, records, size):
        self.root, self.records, self.size = Path(root), records, size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(self.root / self.records[index]['path']) as im:
            im = im.convert('RGB').resize((self.size, self.size), Image.Resampling.BILINEAR)
            return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255
