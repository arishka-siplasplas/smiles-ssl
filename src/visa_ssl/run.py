"""Single category/seed run with resumable epoch-boundary checkpoints."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits

from .data import Images, digest, write_json
from .metrics import spectral
from .models import SSL, encoder, normalize, transform
from .probes import evaluate


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_torch(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


@torch.inference_mode()
def extract(backbone, records, root, config, device, workers):
    backbone.eval()
    loader = DataLoader(Images(root, records, transform(config['image_size']), labeled=True),
                        batch_size=config['batch_size'], shuffle=False, num_workers=workers,
                        generator=torch.Generator().manual_seed(0), pin_memory=device.type == 'cuda')
    features, labels = [], []
    for x, y in loader:
        # Evaluation uses float32, no AMP, no batch-stat updates.
        features.append(backbone(normalize(x.to(device))).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(features), np.concatenate(labels)


def checkpoint(backbone, groups, args, config, directory, epoch):
    target = directory / f'epoch_{epoch:03d}'
    if (target / 'metrics.json').exists():
        return
    target.mkdir(exist_ok=True)
    features, labels, ranks = {}, {}, {}
    before = state_hash(backbone)
    with torch.random.fork_rng(devices=[args.device.index or 0] if args.device.type == 'cuda' else []):
        for split in ('probe_train', 'validation', 'test'):
            z, y = extract(backbone, groups[split], args.data, config, args.device, args.workers)
            features[split], labels[split] = z, y
            norms = np.linalg.norm(z.astype(np.float64), axis=1, keepdims=True)
            unit = z.astype(np.float64) / np.maximum(norms, 1e-12)
            ranks[split] = {kind: spectral(value) for kind, value in [('raw', z), ('l2', unit)]}
        probe_results, predictions, models = evaluate(features, labels, config, args.seed)
    assert before == state_hash(backbone), 'Probe evaluation changed encoder state'
    np.savez_compressed(target / 'embeddings.npz', **features,
                        **{f'{s}_labels': y for s, y in labels.items()},
                        **{f'{s}_paths': np.array([r['path'] for r in groups[s]]) for s in features})
    np.savez_compressed(target / 'predictions.npz', **predictions)
    joblib.dump(models, target / 'probes.joblib')
    write_json(target / 'metrics.json', {'epoch': epoch, 'encoder_sha256': before,
                                        'spectral': ranks, 'classifiers': probe_results})
    print(f'EVALUATED {directory.name} epoch={epoch} ' + json.dumps({
        k: {m: v['splits']['test'][m] for m in ('precision_anomaly', 'average_precision', 'roc_auc')}
        for k, v in probe_results.items()}), flush=True)


def run(args):
    config = json.loads(args.config.read_text())
    manifest = json.loads(args.manifest.read_text())
    if args.seed not in config['seeds'] or args.category not in config['categories']:
        raise ValueError('Run outside frozen protocol')
    if args.method != 'baseline' and args.fraction not in config['anomaly_fractions']:
        raise ValueError('Fraction outside frozen protocol')
    if config['epochs'] > 100 or 0 not in config['checkpoint_epochs']:
        raise ValueError('Protocol requires epoch zero and no more than 100 epochs')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(args.seed)
    groups = manifest['categories'][args.category]
    baseline = args.output / f'{args.category}_baseline_s{args.seed}'
    directory = baseline if args.method == 'baseline' else args.output / f'{args.category}_{args.method}_p{args.fraction:g}_s{args.seed}'
    directory.mkdir(parents=True, exist_ok=True)
    identity = {'category': args.category, 'seed': args.seed, 'method': args.method,
                'fraction': None if args.method == 'baseline' else args.fraction,
                'config_sha256': digest(config), 'manifest_sha256': manifest['manifest_sha256']}
    if (directory / 'identity.json').exists():
        assert json.loads((directory / 'identity.json').read_text()) == identity, 'Resume identity mismatch'
    write_json(directory / 'identity.json', identity)
    if (directory / 'complete.json').exists():
        return 0
    backbone = encoder(args.seed)
    initial_hash = state_hash(backbone)
    backbone.to(args.device)
    if args.method == 'baseline':
        checkpoint(backbone, groups, args, config, directory, 0)
        write_json(directory / 'complete.json', {'initial_encoder_sha256': initial_hash, 'last_epoch': 0})
        return 0
    baseline_done = json.loads((baseline / 'complete.json').read_text())
    assert baseline_done['initial_encoder_sha256'] == initial_hash
    baseline_metrics = json.loads((baseline / 'epoch_000/metrics.json').read_text())
    assert baseline_metrics['encoder_sha256'] == initial_hash
    if not (directory / 'epoch_000').exists():
        shutil.copytree(baseline / 'epoch_000', directory / 'epoch_000')
    # Head creation is isolated from data-order/augmentation streams.
    seed_all(args.seed + 100000)
    model = SSL(backbone, args.method, config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=args.device.type == 'cuda', init_scale=128.)
    start, history = 0, []
    latest = directory / 'latest.pt'
    if latest.exists():
        state = torch.load(latest, map_location=args.device, weights_only=False)
        assert state['identity'] == identity
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start, history = state['epoch'], state['history']
        if start in config['checkpoint_epochs']:
            checkpoint(backbone, groups, args, config, directory, start)
    for epoch in range(start + 1, config['epochs'] + 1):
        if time.time() >= args.deadline:
            write_json(directory / 'progress.json', {'status': 'paused', 'epoch': epoch - 1})
            return 75
        # Re-seeding at epoch boundaries permits exact resume without worker RNG capture.
        epoch_seed = args.seed * 1000 + epoch
        seed_all(epoch_seed)
        loader = DataLoader(Images(args.data, groups['ssl'][str(args.fraction)],
                                   transform(config['image_size'], True),
                                   two_views=args.method != 'masked_reconstruction'),
                            batch_size=config['batch_size'], shuffle=True, drop_last=False,
                            num_workers=args.workers, generator=torch.Generator().manual_seed(epoch_seed),
                            pin_memory=args.device.type == 'cuda')
        model.train()
        totals, seen, started, skipped = {}, 0, time.monotonic(), 0
        for batch in loader:
            batch = [x.to(args.device) for x in batch] if isinstance(batch, list) else batch.to(args.device)
            n = len(batch[0]) if isinstance(batch, list) else len(batch)
            if n < 2:
                raise ValueError('Singleton batch invalid for projector BatchNorm/VICReg')
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast('cuda', dtype=torch.float16) if args.device.type == 'cuda' else nullcontext()
            with context:
                loss, components = model(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
            scaler.scale(loss).backward()
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < previous_scale)
            for key, value in {'loss': loss.detach(), **components}.items():
                totals[key] = totals.get(key, 0.) + float(value) * n
            seen += n
        history.append({'epoch': epoch, 'samples': seen, 'seconds': time.monotonic() - started,
                        'optimizer_steps': len(loader) - skipped, 'amp_skipped_steps': skipped,
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        **{k: v / seen for k, v in totals.items()}})
        save_torch(latest, {'identity': identity, 'epoch': epoch, 'history': history,
                           'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict()})
        write_json(directory / 'loss.json', history)
        write_json(directory / 'progress.json', {'status': 'training', 'epoch': epoch, 'loss': history[-1]['loss']})
        print(f'TRAIN {directory.name} {json.dumps(history[-1])}', flush=True)
        if epoch in config['checkpoint_epochs']:
            checkpoint(backbone, groups, args, config, directory, epoch)
    save_torch(directory / 'final.pt', {'encoder': backbone.state_dict(), 'model': model.state_dict(),
                                      'identity': identity, 'epoch': config['epochs']})
    write_json(directory / 'complete.json', {'initial_encoder_sha256': initial_hash,
                                            'last_epoch': config['epochs'], 'status': 'complete'})
    latest.unlink(missing_ok=True)
    return 0


def main():
    p = argparse.ArgumentParser()
    for flag in ('config', 'manifest', 'data', 'output'):
        p.add_argument('--' + flag, type=Path, required=True)
    p.add_argument('--category', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--method', required=True)
    p.add_argument('--fraction', type=float, default=0.05)
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--deadline', type=float, default=float('inf'))
    args = p.parse_args()
    args.device = torch.device(args.device)
    with threadpool_limits(args.threads):
        raise SystemExit(run(args))


if __name__ == '__main__':
    main()
