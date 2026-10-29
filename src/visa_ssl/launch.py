"""All random-encoder baselines first, then independent GPU jobs, bounded to 9h."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from .analyze import analyze
from .data import write_json
from .models import encoder
from .run import save_torch, state_hash


def launch(args):
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    logs = args.output / 'logs'
    logs.mkdir(exist_ok=True)
    write_json(args.output / 'config.json', config)
    write_json(args.output / 'environment.json', {
        'started_at': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
        'packages': {p: importlib.metadata.version(p) for p in
                     ('torch', 'torchvision', 'numpy', 'scikit-learn', 'lightgbm', 'pandas', 'Pillow')},
        'cuda': torch.version.cuda, 'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    if not torch.cuda.is_available():
        raise RuntimeError('Kaggle GPU is required for overnight training')
    initial = args.output / 'initial_encoders'
    initial.mkdir(exist_ok=True)
    for seed in config['seeds']:
        model = encoder(seed)
        save_torch(initial / f'seed_{seed}.pt', {'encoder': model.state_dict(),
                                               'sha256': state_hash(model), 'seed': seed})
    del model
    deadline = time.time() + args.hours * 3600
    gpus = list(range(min(args.gpus, torch.cuda.device_count())))
    baseline_jobs = [dict(category=c, seed=s, method='baseline', fraction=0.05)
                     for s in config['seeds'] for c in config['categories']]
    train_jobs = [dict(category=c, seed=s, method=m, fraction=f)
                  for s in config['seeds'] for f in config['anomaly_fractions']
                  for c in config['categories'] for m in config['methods']]
    state = {'baseline_jobs': baseline_jobs, 'train_jobs': train_jobs, 'deadline': deadline}
    env = dict(os.environ, PYTHONUNBUFFERED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               MPLCONFIGDIR=str(args.output / '.matplotlib'))
    failures = []
    for stage, jobs in [('baseline', baseline_jobs), ('ssl', train_jobs)]:
        queue, active = list(jobs), {}
        state['stage'] = stage
        last_heartbeat = 0
        while queue or active:
            for gpu in gpus:
                if gpu in active or not queue or time.time() >= deadline:
                    continue
                job = queue.pop(0)
                name = f'{job["category"]}_{job["method"]}_p{job["fraction"]}_s{job["seed"]}'
                command = [sys.executable, '-u', '-m', 'visa_ssl.run', '--config', str(args.config),
                           '--manifest', str(args.manifest), '--data', str(args.data), '--output', str(args.output),
                           '--deadline', str(deadline), '--workers', str(args.workers)]
                for key, value in job.items():
                    command += ['--' + key, str(value)]
                log = (logs / f'{name}.log').open('a')
                process = subprocess.Popen(command, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                                           stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = process, log, job, name
                job.update(status='running', gpu=gpu)
                print(f'START {name} GPU={gpu}', flush=True)
            for gpu, (process, log, job, name) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[gpu]
                job.update(status='complete' if code == 0 else 'paused' if code == 75 else 'failed', exit_code=code)
                print(f'END {name} {job["status"]}', flush=True)
                if code not in (0, 75):
                    failures.append(name)
                    print((logs / f'{name}.log').read_text()[-6000:], flush=True)
            write_json(args.output / 'status.json', state)
            if time.time() - last_heartbeat >= 60:
                print(f'HEARTBEAT {stage} remaining={len(queue)} active={len(active)} failed={len(failures)}', flush=True)
                last_heartbeat = time.time()
            if not active and (not queue or time.time() >= deadline):
                break
            time.sleep(3)
        analyze(args.output)
        if stage == 'baseline' and (failures or any(j.get('status') != 'complete' for j in jobs)):
            break
    state['status'] = ('failed' if failures else 'complete' if all(j.get('status') == 'complete'
                       for j in baseline_jobs + train_jobs) else 'paused')
    write_json(args.output / 'status.json', state)
    print(f'FINISHED: {state["status"]}', flush=True)
    if failures:
        raise RuntimeError(f'Failed jobs: {failures}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('config', 'manifest', 'data', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--hours', type=float, default=9.)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--workers', type=int, default=1)
    launch(parser.parse_args())
