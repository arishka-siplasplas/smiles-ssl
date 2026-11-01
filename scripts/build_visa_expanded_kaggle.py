"""Build immutable private v2 notebooks, with optional preceding-output resume."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def build(args):
    files = sorted(list((ROOT / 'src/visa_ssl').glob('*.py')) +
                   list((ROOT / 'manifests/visa-v2').glob('*')) +
                   [ROOT / p for p in ['configs/visa/expanded.json', 'manifests/visa/2cls_highshot.csv',
                    'manifests/visa/LICENSE-UPSTREAM', 'manifests/visa/LICENSE-DATASET',
                    'scripts/prepare_visa.py', 'docs/VISA_PROTOCOL_RU.md',
                    'docs/VISA_EXPANDED_PLAN_RU.md', 'tests/test_visa_expanded.py']])
    if not (ROOT / 'manifests/visa-v2/image_hashes.json').exists():
        raise RuntimeError('Freeze full official image inventory before building')
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as tar:
        for path in files:
            tar.add(path, arcname=str(path.relative_to(ROOT)))
    payload = stream.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    provenance = {'payload_sha256': digest,
                  'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  'mode': args.mode, 'session_hours': args.hours, 'resume_kernel': args.resume_kernel,
                  'resume_dataset': args.resume_dataset,
                  'resume_archive_sha256': args.resume_archive_sha256,
                  'resume_origin_kernel': args.resume_origin_kernel}
    code = '''import os, sys, subprocess, json, base64, tarfile, io, hashlib, shutil, time, importlib.metadata
from pathlib import Path
session_started = time.time()
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  MPLCONFIGDIR='/kaggle/working/.matplotlib', PYTHONUNBUFFERED='1')
root = Path('/kaggle/working/visa-experiment')
root.mkdir(exist_ok=True)
'''
    code += f'payload = base64.b64decode({base64.b64encode(payload).decode()!r})\n'
    code += f'assert hashlib.sha256(payload).hexdigest() == {digest!r}\n'
    code += '''with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
    archive.extractall(root, filter='data')
os.environ['PYTHONPATH'] = str(root / 'src')
sys.path.insert(0, str(root / 'src'))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'lightgbm==4.6.0', 'timm==1.0.22'], check=True)
import torch, timm
print('Torch:', torch.__version__, 'timm:', timm.__version__, 'CUDA devices:', torch.cuda.device_count(), flush=True)
assert torch.cuda.device_count() == 2, 'Expected T4 x2, one independent worker per GPU'
'''
    if args.resume_dataset:
        code += '''archives = list(Path('/kaggle/input').rglob('resume.tar.gz.bin'))
assert len(archives) == 1, f'Expected one private resume archive, got {archives}'
with archives[0].open('rb') as stream:
    archive_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
'''
        code += f'assert archive_sha256 == {args.resume_archive_sha256!r}, "Resume archive SHA-256 differs"\n'
        code += '''resume_root = Path('/kaggle/temp/visa-resume')
resume_root.mkdir(parents=True, exist_ok=True)
with tarfile.open(archives[0], mode='r:gz') as archive:
    archive.extractall(resume_root, filter='data')
resume_source = resume_root / 'results'
transfer = json.loads((resume_root / 'transfer_hashes.json').read_text())
'''
        code += f'assert transfer["source_kernel"] == {args.resume_origin_kernel!r}, "Wrong resume source"\n'
        code += '''actual_files = {str(p.relative_to(resume_source)) for p in resume_source.rglob('*') if p.is_file()}
assert actual_files == set(transfer['files']), 'Resume artifact inventory differs'
for name, expected in transfer['files'].items():
    path = resume_source / name
    assert path.stat().st_size == expected['size'], f'Resume size differs: {name}'
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected['sha256'], f'Resume SHA-256 differs: {name}'
print('RESUME_ARCHIVE_VERIFIED', transfer['source_kernel'], len(actual_files), archive_sha256, flush=True)
'''
    elif args.resume_kernel:
        code += '''candidates = [p.parent for p in Path('/kaggle/input').rglob('status.json')
              if (p.parent / 'config.json').exists() and (p.parent / 'manifest.json').exists()]
assert len(candidates) == 1, f'Expected one source run, got {candidates}'
resume_source = candidates[0]
'''
    if args.resume_kernel or args.resume_dataset:
        code += '''
previous_environment = json.loads((resume_source / 'environment.json').read_text())
current_packages = {p: importlib.metadata.version(p) for p in previous_environment['packages']}
assert current_packages == previous_environment['packages'], (
    'Resume package versions differ', previous_environment['packages'], current_packages)
assert sys.version == previous_environment['python'], 'Resume Python version differs'
assert torch.version.cuda == previous_environment['cuda'], 'Resume CUDA build differs'
assert [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] == previous_environment['gpus'], 'Resume GPU types differ'
print('RESUME_ENVIRONMENT_MATCH', json.dumps(current_packages), flush=True)
'''
    code += '''
data = Path('/kaggle/temp/visa-original')
subprocess.run([sys.executable, '-u', str(root / 'scripts/prepare_visa.py'),
    '--manifest', str(root / 'manifests/visa-v2/fixed.json'), '--output', str(data)], check=True)
output = root / 'results'
output.mkdir(exist_ok=True)
shutil.copy2(data / 'image_hashes.json', output / 'image_hashes.json')
'''
    code += f"(output / 'source_snapshot.json').write_text(json.dumps({provenance!r}, indent=2))\n"
    code += '''command = [sys.executable, '-u', '-m', 'visa_ssl.expanded',
    '--config', str(root / 'configs/visa/expanded.json'),
    '--manifest', str(root / 'manifests/visa-v2/fixed.json'),
    '--data', str(data), '--output', str(output), '--gpus', '2']
'''
    # Leave five minutes for reports and graceful checkpoint/output finalization.
    code += f"command += ['--hours', {str(args.hours)!r}, '--deadline', str(session_started + {args.hours * 3600 - 300})]\n"
    if args.mode == 'preflight':
        code += "command += ['--preflight']\n"
    else:
        code += f"command += ['--architecture', {args.mode!r}]\n"
    if args.resume_kernel or args.resume_dataset:
        code += '''command += ['--resume', str(resume_source)]
'''
    code += '''result = subprocess.run(command)
print((output / 'status.json').read_text(), flush=True)
if (output / 'analysis/BASELINE.md').exists():
    print((output / 'analysis/BASELINE.md').read_text(), flush=True)
result.check_returncode()
'''
    slug = args.slug or 'visa-ssl-v2-' + args.mode.replace('_', '-')
    bundle = ROOT / 'kaggle/bundles' / slug
    bundle.mkdir(parents=True, exist_ok=True)
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'cells': [
            {'cell_type': 'markdown', 'id': 'protocol', 'metadata': {}, 'source': [
                f'# VisA v2 {args.mode}\n',
                'Six fixed official categories, full training pool, three SSL seeds, three augmentations.\n',
                'Random ResNet-18 / DeiT-Tiny; 100 epochs with deterministic frozen-encoder evaluation.\n',
                'No pretrained weights. Resume is checked against config and manifest hashes.\n']},
            {'cell_type': 'code', 'id': 'experiment', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': code.splitlines(True)}]}
    (bundle / 'experiment.ipynb').write_text(json.dumps(notebook, indent=1))
    # Kaggle derives new notebook slugs from the title, even when an id is given.
    metadata = {'id': args.owner + '/' + slug, 'title': slug.replace('-', ' '),
        'code_file': 'experiment.ipynb', 'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': True, 'enable_internet': True,
        'dataset_sources': [args.resume_dataset] if args.resume_dataset else [], 'competition_sources': [],
        'kernel_sources': [args.resume_kernel] if args.resume_kernel else [],
        'machine_shape': 'NvidiaTeslaT4', 'docker_image_pinning_type': 'original',
        'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f2f6b083ad2b2f'}
    (bundle / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (bundle / 'source_snapshot.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'bundle': str(bundle), 'payload_sha256': digest, 'files': len(files), 'id': metadata['id']}))
    return bundle


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['preflight', 'resnet18', 'deit_tiny'], required=True)
    p.add_argument('--hours', type=float, default=10.)
    p.add_argument('--owner', default='arinabazukina')
    resume = p.add_mutually_exclusive_group()
    resume.add_argument('--resume-kernel')
    resume.add_argument('--resume-dataset')
    p.add_argument('--resume-archive-sha256')
    p.add_argument('--resume-origin-kernel')
    p.add_argument('--slug')
    args = p.parse_args()
    if not 0.2 <= args.hours <= 10:
        p.error('Session budget must be between .2 and 10 hours')
    if args.resume_dataset and (not args.resume_origin_kernel or not
            re.fullmatch(r'[0-9a-f]{64}', args.resume_archive_sha256 or '')):
        p.error('Dataset resume requires the original kernel ref and the archive SHA-256')
    if not args.resume_dataset and (args.resume_origin_kernel or args.resume_archive_sha256):
        p.error('Archive provenance flags require --resume-dataset')
    build(args)
