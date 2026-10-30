"""Bounded v2 sessions: independent GPU workers, explicit shards, resumable jobs."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from .data import digest, write_json
from .models import encoder
from .run import job_name, save_torch, state_hash


def jobs(config, architecture=None, preflight=False):
    architectures = config['architectures'] if architecture is None else [architecture]
    if not set(architectures) <= set(config['architectures']):
        raise ValueError('Unknown architecture shard')
    baseline = [dict(category=c, architecture=a, seed=s, method='baseline', augmentation='baseline')
                for a in architectures for c in config['categories'] for s in config['seeds']]
    training = [dict(category=c, architecture=a, seed=s, method=m, augmentation=u)
                for u in config['augmentations'] for c in config['categories']
                for m in config['methods'] for a in architectures for s in config['seeds']]
    if preflight:
        training = [j for j in training if j['category'] == config['categories'][0]
                    and j['seed'] == config['seeds'][0] and j['augmentation'] == 'baseline']
    return baseline, training


def restore(source, output, config, manifest, architecture=None):
    """Copy only compatible job states; no overwrite of locally advanced results."""
    source, output = Path(source), Path(output)
    restored = []
    for identity_file in sorted(source.glob('*/identity.json')):
        identity = json.loads(identity_file.read_text())
        if architecture and identity['architecture'] != architecture:
            continue
        if identity['config_sha256'] != digest(config) or identity['manifest_sha256'] != manifest['manifest_sha256']:
            raise ValueError(f'Incompatible resume source: {identity_file}')
        target = output / identity_file.parent.name
        if not target.exists():
            shutil.copytree(identity_file.parent, target)
            restored.append(target.name)
    return restored


def launch(args):
    config = json.loads(args.config.read_text())
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        copied = restore(args.resume, args.output, config, manifest, args.architecture)
        print(f'RESTORED {len(copied)} jobs from {args.resume}', flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('Kaggle CUDA GPUs required')
    write_json(args.output / 'config.json', config)
    write_json(args.output / 'manifest.json', manifest)
    write_json(args.output / 'environment.json', {
        'started_at': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
        'packages': {p: importlib.metadata.version(p) for p in
                     ('torch', 'torchvision', 'timm', 'numpy', 'scikit-learn', 'lightgbm', 'pandas', 'Pillow')},
        'cuda': torch.version.cuda,
        'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    baseline, training = jobs(config, args.architecture, args.preflight)
    initial = args.output / 'initial_encoders'
    initial.mkdir(exist_ok=True)
    for architecture in sorted({j['architecture'] for j in baseline}):
        for seed in config['seeds']:
            model = encoder(seed, architecture, config['image_size'])
            save_torch(initial / f'{architecture}_s{seed}.pt',
                       {'encoder': model.state_dict(), 'sha256': state_hash(model),
                        'seed': seed, 'architecture': architecture})
    del model
    deadline = min(time.time() + args.hours * 3600, args.deadline)
    state = {'protocol': config['protocol'], 'architecture': args.architecture,
             'preflight': args.preflight, 'deadline': deadline,
             'baseline_jobs': baseline, 'train_jobs': training, 'status': 'running'}
    logs = args.output / 'logs'
    logs.mkdir(exist_ok=True)
    stop_file = args.output / 'STOP'
    stop_file.unlink(missing_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               MPLCONFIGDIR=str(args.output / '.matplotlib'))
    failures = []
    for stage, stage_jobs in [('baseline', baseline), ('ssl', training)]:
        queue, active = list(stage_jobs), {}
        state['stage'] = stage
        last_heartbeat = 0
        while queue or active:
            for gpu in range(min(args.gpus, torch.cuda.device_count())):
                if gpu in active or not queue or time.time() >= deadline or failures or stop_file.exists():
                    continue
                job = queue.pop(0)
                name = job_name(job, True)
                command = [sys.executable, '-u', '-m', 'visa_ssl.run',
                           '--config', str(args.config), '--manifest', str(args.manifest),
                           '--data', str(args.data), '--output', str(args.output),
                           '--deadline', str(deadline), '--workers', str(args.workers),
                           '--stop-file', str(stop_file)]
                for key, value in job.items():
                    command += ['--' + key, str(value)]
                if args.preflight:
                    command += ['--max-epoch', '1']
                log = (logs / f'{name}.log').open('a')
                process = subprocess.Popen(command, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                                           stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = (process, log, job, name, time.monotonic())
                job.update(status='running', gpu=gpu)
                print(f'START {name} GPU={gpu}', flush=True)
            for gpu, (process, log, job, name, started) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[gpu]
                job.update(status='complete' if code == 0 else 'paused' if code == 75 else 'failed',
                           exit_code=code, wall_seconds=time.monotonic() - started)
                print(f'END {name} {job["status"]} seconds={job["wall_seconds"]:.1f}', flush=True)
                if code not in (0, 75):
                    failures.append(name)
                    stop_file.write_text('Stop at epoch boundary after another worker failed.\n')
                    print((logs / f'{name}.log').read_text()[-7000:], flush=True)
                loss_path = args.output / name / 'loss.json'
                if loss_path.exists():
                    history = json.loads(loss_path.read_text())
                    job['last_epoch'] = history[-1]['epoch']
                    job['amp_skipped_steps'] = sum(r['amp_skipped_steps'] for r in history)
                    print(f'LATEST {name} ' + json.dumps(history[-1]), flush=True)
                    if args.preflight and job['amp_skipped_steps']:
                        failures.append(name)
                        stop_file.write_text('Preflight had skipped AMP updates.\n')
            write_json(args.output / 'status.json', state)
            if time.time() - last_heartbeat >= 60:
                running = {}
                for _, _, _, name, _ in active.values():
                    progress = args.output / name / 'progress.json'
                    running[name] = json.loads(progress.read_text()) if progress.exists() else 'initializing/evaluating'
                print(f'HEARTBEAT {stage} remaining={len(queue)} failed={len(failures)} ' + json.dumps(running), flush=True)
                last_heartbeat = time.time()
                size = sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
                if size > args.storage_gb * 1024**3:
                    stop_file.write_text('Pause at epoch boundary to preserve Kaggle output storage.\n')
                    state['pause_reason'] = 'storage_limit'
            if not active and (not queue or time.time() >= deadline or failures or stop_file.exists()):
                break
            time.sleep(3)
        from .analyze_expanded import analyze
        analyze(args.output)
        if failures or any(j.get('status') != 'complete' for j in baseline):
            break
    if args.preflight:
        success = all(j.get('status') == 'complete' for j in baseline) and all(j.get('last_epoch') == 1 for j in training)
        state['status'] = 'preflight_complete' if success and not failures else 'failed'
    else:
        state['status'] = ('failed' if failures else 'complete' if all(
            j.get('status') == 'complete' for j in baseline + training) else 'paused')
    write_json(args.output / 'status.json', state)
    print('FINISHED: ' + state['status'], flush=True)
    if failures or state['status'] == 'failed':
        raise RuntimeError(f'Failed jobs: {failures}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for flag in ('config', 'manifest', 'data', 'output'):
        p.add_argument('--' + flag, type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--architecture', choices=['resnet18', 'deit_tiny'])
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--hours', type=float, default=10.)
    p.add_argument('--deadline', type=float, default=float('inf'))
    p.add_argument('--gpus', type=int, default=2)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--storage-gb', type=float, default=16.)
    launch(p.parse_args())
