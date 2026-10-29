"""Training, independent checkpoint probes, resumable state and sealed test pass."""
import copy
import importlib.metadata
import math
import os
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .augment import Corruptor, donors
from .config import validate_config
from .data import load_splits
from .io import digest, read_json, write_json
from .metrics import binary_scores, spectral
from .models import SSLModel
from .probes import fit_probes


class Indexed(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return i, self.data[i]  # No labels can reach the SSL training loader.


def seed_everything(seed, threads=1):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False


def runtime_info(device):
    def git(*args):
        p = subprocess.run(['git', *args], text=True, capture_output=True)
        return p.stdout.strip() if p.returncode == 0 else None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'device': str(device),
            'packages': {n: importlib.metadata.version(n) for n in ['torch', 'torchvision', 'numpy',
                         'scikit-learn', 'catboost', 'scipy', 'Pillow', 'matplotlib']},
            'git_commit': git('rev-parse', 'HEAD'), 'git_status': git('status', '--porcelain'),
            'cuda': torch.version.cuda}


def rng_state(generator):
    return {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(),
            'python': random.getstate(), 'loader': generator.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    torch.set_rng_state(state['torch'].cpu())
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    generator.set_state(state['loader'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def atomic_torch_save(value, path):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


@torch.no_grad()
def embed(encoder, data, device, batch_size):
    encoder.eval()
    return np.concatenate([encoder(x.to(device)).cpu().numpy()
                           for x in DataLoader(data, batch_size=batch_size, shuffle=False)])


def evaluate_checkpoint(model, data, labels, c, out, epoch, device, generator):
    # Evaluation must not perturb the next epoch's augmentation/shuffle RNG.
    state = rng_state(generator)
    model.eval()
    folder = out / f'epoch_{epoch:04d}'
    folder.mkdir(exist_ok=True)
    z = {s: embed(model.encoder, data[s], device, c['batch_size']) for s in ['train', 'validation']}
    probes, fitted = fit_probes(z['train'], labels['train'], z['validation'], labels['validation'], c['probe'], c['seed'])
    stats = {s: spectral(z[s]) for s in z}
    write_json(folder / 'evaluation.json', {'epoch': epoch, 'probes': probes, 'spectral': stats})
    np.savez_compressed(folder / 'embeddings.npz', **z)
    joblib.dump(fitted, folder / 'probes.joblib')
    atomic_torch_save(model.encoder.state_dict(), folder / 'encoder.pt')
    restore_rng(state, generator)


def train(config_path, manifest_path, data_root, output, device='cpu', resume=False):
    c, manifest = read_json(config_path), read_json(manifest_path)
    validate_config(c)
    if any(c[k] != manifest[k] for k in ['modality', 'dataset']) or c['split_seed'] != manifest['seed']:
        raise ValueError('Config does not match prepared dataset/split.')
    if device not in ('cpu', 'cuda'):
        raise ValueError('Validated devices: cpu/cuda. MPS is outside the v1 reproducibility protocol.')
    device = torch.device(device)
    out = Path(output)
    if resume:
        if read_json(out / 'config.json') != c or digest(out / 'manifest.json') != digest(manifest_path):
            raise ValueError('Resume config/split must exactly match the original.')
        if (out / 'completion.json').exists():
            raise ValueError('Run already complete.')
        old_hashes = read_json(out / 'source_hashes.json')
        new_hashes = {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}
        if old_hashes != new_hashes:
            raise ValueError('Source changed: resume with the saved source or start a new run.')
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / 'config.json', c)
        shutil.copyfile(manifest_path, out / 'manifest.json')
        shutil.copytree(Path(__file__).parent, out / 'source_snapshot', ignore=shutil.ignore_patterns('__pycache__'))
        write_json(out / 'source_hashes.json', {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')})
        write_json(out / 'environment.json', runtime_info(device))
    seed_everything(c['seed'], c['threads'])
    data, labels = load_splits(manifest, data_root, c['image_size'])
    input_dim = data['train'].shape[1] if c['modality'] == 'table' else None
    model = SSLModel(c['modality'], c['method'], input_dim, c['image_size'], c['seed']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
    # Same shuffle and augmentation RNG streams begin after head initialization.
    torch.manual_seed(c['seed'] + 200000)
    generator = torch.Generator().manual_seed(c['seed'] + 300000)
    loader = DataLoader(Indexed(data['train']), batch_size=min(c['batch_size'], len(data['train'])),
                        shuffle=True, drop_last=False, num_workers=c['num_workers'], generator=generator)
    prep = manifest.get('preprocessing', {})
    image_mean = None
    if c['modality'] == 'image':
        sums, pixels = torch.zeros(3, dtype=torch.float64), 0
        for x in DataLoader(data['train'], batch_size=c['batch_size']):
            sums += x.double().sum((0, 2, 3))
            pixels += x.shape[0] * x.shape[2] * x.shape[3]
        image_mean = (sums / pixels).tolist()
    corrupt = Corruptor(c['modality'], c['augmentation'], prep.get('mean'), prep.get('scale'),
                        prep.get('noise_columns'), image_mean)
    available = ((len(prep['noise_columns']) if c['augmentation']['family'] == 'noise' else input_dim)
                 if c['modality'] == 'table' else (c['image_size'] // c['augmentation']['patch_size']) ** 2)
    if available == 0:
        raise ValueError('No eligible continuous features for noise.')
    total_units = input_dim if c['modality'] == 'table' else available
    bernoulli = c['augmentation'].get('mask_sampling') == 'bernoulli'
    selected_units = (None if bernoulli else max(1, min(available, total_units - 1, round(available * c['augmentation']['fraction']))))
    write_json(out / 'augmentation_stats.json', {'image_mean': image_mean, 'preprocessing': prep,
                                                'eligible_units': available, 'selected_units': selected_units,
                                                'expected_selected_units': available * c['augmentation']['fraction'] if bernoulli else selected_units,
                                                'actual_fraction_of_all_units': None if bernoulli else selected_units / total_units,
                                                'mask_sampling': c['augmentation'].get('mask_sampling', 'fixed_count')})
    history, first_epoch, elapsed_before = [], 1, 0.
    if resume:
        saved = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'], generator)
        history, first_epoch, elapsed_before = saved['history'], saved['epoch'] + 1, saved['elapsed_seconds']
    else:
        if c['modality'] == 'table':
            raw, fitted = fit_probes(data['train'].numpy(), labels['train'], data['validation'].numpy(),
                                    labels['validation'], c['probe'], c['seed'])
            write_json(out / 'raw_baseline.json', raw)
            joblib.dump(fitted, out / 'raw_probes.joblib')
        evaluate_checkpoint(model, data, labels, c, out, 0, device, generator)
    start = time.monotonic()
    def save_latest(epoch):
        atomic_torch_save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                           'rng': rng_state(generator), 'history': history,
                           'elapsed_seconds': elapsed_before + time.monotonic() - start}, out / 'latest.pt')
    if not resume:
        save_latest(0)
    for epoch in range(first_epoch, c['epochs'] + 1):
        model.train()
        schedule_epochs = c.get('schedule_epochs', c['epochs'])
        warmup = min(c['warmup_epochs'], schedule_epochs)
        if warmup and epoch <= warmup:
            factor = epoch / warmup
        else:
            progress = (epoch - warmup - 1) / max(1, schedule_epochs - warmup)
            factor = .5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group['lr'] = c['learning_rate'] * factor
        totals, used, steps, loss_weight_sum = {}, 0, 0, 0
        selected_cells, all_cells, changed_rows, generated_rows, empty_targets = 0, 0, 0, 0, 0
        for indices, clean in loader:
            if len(clean) < 2:
                continue  # Same singleton policy for all three methods.
            clean = clean.to(device)
            def view():
                donor = donors(data['train'], indices, c['modality'], device) if c['augmentation']['family'] == 'donor' else None
                return corrupt(clean, donor)
            a, mask = view()
            b = view()[0] if c['method'] != 'masked_reconstruction' else None
            selected_cells += int(mask.sum())
            all_cells += mask.numel()
            changed_rows += int(mask.flatten(1).any(1).sum())
            generated_rows += len(mask)
            if c['method'] == 'masked_reconstruction' and not mask.any():
                # Never manufacture a target by forcing a mask: that changes the
                # requested weak distribution, especially for six-feature data.
                empty_targets += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            loss, components = model.objective(clean, a, mask, b)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite SSL loss.')
            loss.backward()
            optimizer.step()
            used += len(clean)
            steps += 1
            loss_weight = int(mask.sum()) if c['method'] == 'masked_reconstruction' else len(clean)
            loss_weight_sum += loss_weight
            for k, v in {'ssl_loss': loss, **components}.items():
                totals[k] = totals.get(k, 0.) + float(v.detach()) * loss_weight
        if used == 0:
            raise ValueError('No train minibatches with at least two samples.')
        history.append({'epoch': epoch, 'lr': c['learning_rate'] * factor,
                        'objects_used': used, 'steps': steps,
                        'first_view_selected_cell_fraction': selected_cells / all_cells,
                        'first_view_selected_row_fraction': changed_rows / generated_rows,
                        'empty_target_batches_skipped': empty_targets,
                        'loss_averaging_weight': 'selected_targets' if c['method'] == 'masked_reconstruction' else 'objects',
                        **{k: v / loss_weight_sum for k, v in totals.items()}})
        if epoch in c['checkpoints']:
            evaluate_checkpoint(model, data, labels, c, out, epoch, device, generator)
        write_json(out / 'loss.json', history)
        save_latest(epoch)
        print(f"epoch {epoch}/{c['epochs']} loss={history[-1]['ssl_loss']:.6g}", flush=True)
    write_json(out / 'completion.json', {'stage': c['stage'], 'epochs': c['epochs'],
                                        'elapsed_seconds': elapsed_before + time.monotonic() - start})


def evaluate_test(run, data_root, device='cpu'):
    out = Path(run)
    if not (out / 'completion.json').exists():
        raise ValueError('Complete training before opening test.')
    if (out / 'test.json').exists():
        raise FileExistsError('Test already evaluated; do not overwrite.')
    c, manifest = read_json(out / 'config.json'), read_json(out / 'manifest.json')
    seed_everything(c['seed'], c['threads'])
    data, labels = load_splits(manifest, data_root, c['image_size'], include_test=True)
    device = torch.device(device)
    model = SSLModel(c['modality'], c['method'], data['train'].shape[1] if c['modality'] == 'table' else None,
                     c['image_size'], c['seed']).to(device)
    rows = []
    for epoch in c['checkpoints']:
        folder = out / f'epoch_{epoch:04d}'
        model.encoder.load_state_dict(torch.load(folder / 'encoder.pt', map_location=device, weights_only=True))
        z = embed(model.encoder, data['test'], device, c['batch_size'])
        fitted = joblib.load(folder / 'probes.joblib')
        scores = {name: model.predict_proba(z)[:, 1] for name, model in fitted.items()}
        rows.append({'epoch': epoch, 'probes': {name: binary_scores(labels['test'], v) for name, v in scores.items()},
                     'spectral': spectral(z)})
        np.savez_compressed(folder / 'test_predictions.npz', y=labels['test'], **scores)
    raw = None
    if c['modality'] == 'table':
        fitted = joblib.load(out / 'raw_probes.joblib')
        raw = {name: binary_scores(labels['test'], model.predict_proba(data['test'].numpy())[:, 1])
               for name, model in fitted.items()}
    write_json(out / 'test.json', {'selection': 'validation AP only; all prespecified checkpoints',
                                  'stage': c['stage'], 'rows': rows, 'raw_baseline': raw})
