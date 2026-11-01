"""Verify notebook transport and reject damaged checkpoint archives before training."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest


@pytest.fixture
def builder(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'scripts/build_visa_expanded_kaggle.py'
    spec = importlib.util.spec_from_file_location('visa_bundle_builder', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / 'repo'
    for name in ['src/visa_ssl/__init__.py', 'manifests/visa-v2/image_hashes.json',
                 'configs/visa/expanded.json', 'manifests/visa/2cls_highshot.csv',
                 'manifests/visa/LICENSE-UPSTREAM', 'manifests/visa/LICENSE-DATASET',
                 'scripts/prepare_visa.py', 'docs/VISA_PROTOCOL_RU.md',
                 'docs/VISA_EXPANDED_PLAN_RU.md', 'tests/test_visa_expanded.py']:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    monkeypatch.setattr(module, 'ROOT', root)
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **kw: 'test-commit')
    return module


def make_args(**changes):
    values = dict(mode='resnet18', hours=10, owner='second-user', slug='test-resume',
                  resume_kernel=None, resume_dataset='second-user/visa-checkpoints',
                  resume_archive_sha256='0' * 64, resume_origin_kernel='first-user/previous')
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize('damage', [None, 'archive', 'file', 'missing', 'source'])
def test_generated_notebook_restores_only_verified_archive(builder, tmp_path, damage):
    staging = tmp_path / 'staging'
    (staging / 'results').mkdir(parents=True)
    original = b'checkpoint-state'
    (staging / 'results/latest.pt').write_bytes(original)
    transfer = {'source_kernel': 'first-user/previous', 'files': {
        'latest.pt': {'size': len(original), 'sha256': hashlib.sha256(original).hexdigest()}}}
    if damage == 'file':
        (staging / 'results/latest.pt').write_bytes(b'corrupted-state!')
    if damage == 'missing':
        (staging / 'results/latest.pt').unlink()
    if damage == 'source':
        transfer['source_kernel'] = 'wrong-user/wrong-run'
    (staging / 'transfer_hashes.json').write_text(json.dumps(transfer))
    inputs = tmp_path / 'input'
    inputs.mkdir()
    archive = inputs / 'resume.tar.gz.bin'
    with tarfile.open(archive, 'w:gz') as target:
        target.add(staging / 'results', arcname='results')
        target.add(staging / 'transfer_hashes.json', arcname='transfer_hashes.json')
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    if damage == 'archive':
        sha = '0' * 64
    folder = builder.build(make_args(resume_archive_sha256=sha))
    meta = json.loads((folder / 'kernel-metadata.json').read_text())
    assert meta['id'] == 'second-user/test-resume' and meta['is_private']
    assert meta['kernel_sources'] == []
    assert meta['dataset_sources'] == ['second-user/visa-checkpoints']
    notebook = json.loads((folder / 'experiment.ipynb').read_text())
    code = ''.join(notebook['cells'][1]['source'])
    ast.parse(code)
    # Exercise the actual archive verification emitted into the remote notebook.
    code = code[code.index('archives = list('):code.index('previous_environment =')]
    output = tmp_path / 'restored'
    def mapped_path(value):
        return {'/kaggle/input': inputs, '/kaggle/temp/visa-resume': output}.get(value, Path(value))
    scope = dict(Path=mapped_path, hashlib=hashlib, json=json, tarfile=tarfile)
    if damage:
        with pytest.raises(AssertionError):
            exec(compile(code, '<resume-cell>', 'exec'), scope)
    else:
        exec(compile(code, '<resume-cell>', 'exec'), scope)
        assert (output / 'results/latest.pt').read_bytes() == original


def test_kernel_resume_keeps_existing_input_mode(builder):
    folder = builder.build(make_args(resume_dataset=None, resume_archive_sha256=None,
                                    resume_origin_kernel=None, resume_kernel='first-user/previous'))
    meta = json.loads((folder / 'kernel-metadata.json').read_text())
    assert meta['kernel_sources'] == ['first-user/previous'] and meta['dataset_sources'] == []
    code = ''.join(json.loads((folder / 'experiment.ipynb').read_text())['cells'][1]['source'])
    ast.parse(code)
    assert 'RESUME_ENVIRONMENT_MATCH' in code and 'RESUME_ARCHIVE_VERIFIED' not in code
