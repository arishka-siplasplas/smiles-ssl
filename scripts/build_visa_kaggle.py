"""Build a private, self-contained Kaggle notebook from audited local sources."""
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
files = sorted(list((ROOT / 'src/visa_ssl').glob('*.py')) +
               list((ROOT / 'configs/visa').glob('*.json')) +
               list((ROOT / 'manifests/visa').glob('*')) +
               [ROOT / 'scripts/prepare_visa.py', ROOT / 'docs/VISA_PROTOCOL_RU.md',
                ROOT / 'tests/test_visa.py'])
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
    for path in files:
        tar.add(path, arcname=str(path.relative_to(ROOT)))
encoded = base64.b64encode(buffer.getvalue()).decode()
snapshot_hash = hashlib.sha256(buffer.getvalue()).hexdigest()
code = '''import os, sys, subprocess, json, base64, tarfile, io, hashlib, shutil
from pathlib import Path
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  MPLCONFIGDIR='/kaggle/working/.matplotlib', PYTHONUNBUFFERED='1')
root = Path('/kaggle/working/visa-experiment')
root.mkdir(exist_ok=True)
'''
code += f"payload = base64.b64decode({encoded!r})\nassert hashlib.sha256(payload).hexdigest() == {snapshot_hash!r}\n"
code += '''with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
    archive.extractall(root, filter='data')
os.environ['PYTHONPATH'] = str(root / 'src')
sys.path.insert(0, str(root / 'src'))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'lightgbm==4.6.0'], check=True)
import torch
print('Torch:', torch.__version__, 'CUDA devices:', torch.cuda.device_count(), flush=True)
assert torch.cuda.is_available(), 'Enable GPU before running this notebook'
data = Path('/kaggle/temp/visa-original')
subprocess.run([sys.executable, '-u', str(root / 'scripts/prepare_visa.py'),
                '--manifest', str(root / 'manifests/visa/fixed.json'), '--output', str(data)], check=True)
output = root / 'results'
output.mkdir(exist_ok=True)
shutil.copy2(data / 'image_hashes.json', output / 'image_hashes.json')
'''
provenance = {'payload_sha256': snapshot_hash,
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
code += f"(output / 'source_snapshot.json').write_text(json.dumps({provenance!r}, indent=2))\n"
code += '''subprocess.run([sys.executable, '-u', '-m', 'visa_ssl.launch',
    '--config', str(root / 'configs/visa/overnight.json'),
    '--manifest', str(root / 'manifests/visa/fixed.json'),
    '--data', str(data), '--output', str(output), '--hours', '9', '--gpus', '2'], check=True)
print((output / 'analysis/BASELINE.md').read_text(), flush=True)
print((output / 'status.json').read_text(), flush=True)
'''
notebook = {'cells': [
    {'cell_type': 'markdown', 'metadata': {}, 'source': [
        '# VisA: random ResNet-18 first, then 30-epoch SSL\n',
        'Official 2-class high-shot split; cashew, chewinggum, fryum. Three seeds; 5%/20% SSL anomalies.\n',
        'All nine epoch-zero evaluations finish before SSL. Outputs include frozen manifests, source hashes,\n',
        'per-checkpoint features, logistic regression / LightGBM predictions, spectral metrics and plots.\n']},
    {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': code.splitlines(True)}],
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
    'nbformat': 4, 'nbformat_minor': 5}
bundle = ROOT / 'kaggle/bundle'
bundle.mkdir(parents=True, exist_ok=True)
(bundle / 'visa_overnight.ipynb').write_text(json.dumps(notebook, indent=1))
metadata = {'id': 'arinabazukina/visa-ssl-duration-overnight-v1',
            'title': 'VisA SSL Duration Overnight V1', 'code_file': 'visa_overnight.ipynb',
            'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
            'enable_gpu': True, 'enable_internet': True, 'dataset_sources': [],
            'competition_sources': [], 'kernel_sources': [], 'machine_shape': 'NvidiaTeslaT4',
            'docker_image_pinning_type': 'original',
            'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'}
(bundle / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps({'bundle': str(bundle), 'payload_sha256': snapshot_hash, 'files': len(files)}))
