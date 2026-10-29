import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from visa_ssl.data import Images, make_manifest, write_json
from visa_ssl.metrics import spectral, choose_threshold, binary_metrics
from visa_ssl.models import SSL, encoder, patch_mask, transform
from visa_ssl.probes import evaluate
from visa_ssl.run import run, state_hash

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / 'configs/visa/overnight.json').read_text())


def test_manifest_official_boundaries_and_fixed_size(config):
    path = ROOT / 'manifests/visa/2cls_highshot.csv'
    first = make_manifest(path, config)
    alternate = copy.deepcopy(config)
    alternate['seeds'] = [99, 888]
    assert first == make_manifest(path, alternate)
    assert first == json.loads((ROOT / 'manifests/visa/fixed.json').read_text())
    for groups in first['categories'].values():
        evaluation = {r['path'] for s in ('validation', 'test') for r in groups[s]}
        for fraction, subset in groups['ssl'].items():
            assert len(subset) == 240
            assert sum(r['label'] for r in subset) == round(240 * float(fraction))
            assert not {r['path'] for r in subset} & evaluation
            assert all(r['official_split'] == 'train' for r in subset)
        assert all(r['official_split'] == 'test' for r in groups['test'])
        assert sum(r['label'] for r in groups['test']) == 40
        assert sum(r['label'] for r in groups['validation']) == 12


def test_spectral_independent_svd_and_l2():
    x = np.random.default_rng(1).normal(size=(100, 20)) + 3
    def erank(s):
        p = s / s.sum()
        return np.exp(-(p * np.log(p)).sum())
    for z in (x, x / np.linalg.norm(x, axis=1, keepdims=True)):
        raw = np.linalg.svd(z, compute_uv=False)
        centered = np.linalg.svd(z - z.mean(0), compute_uv=False)
        expected = dict(RankMe=erank(raw), CER=erank(centered), CovER=erank(centered**2),
                        NESum=(centered**2).sum() / centered[0]**2)
        actual = spectral(z)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-10)
    assert spectral(np.ones((10, 4)))['centered_collapse']


def test_precision_threshold_ties_and_recall_floor():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    scores = np.array([.9, .7, .6, .1, .8, .3, .2, .1])
    threshold = choose_threshold(y, scores, .5)
    result = binary_metrics(y, scores, threshold)
    assert result['recall_anomaly'] >= .5
    feasible = [binary_metrics(y, scores, t) for t in np.unique(scores)]
    assert result['precision_anomaly'] == max(v['precision_anomaly'] for v in feasible if v['recall_anomaly'] >= .5)
    assert choose_threshold(y, np.ones(8) * .5, .5) == .5


def test_no_test_tuning(config):
    rng = np.random.default_rng(44)
    features = {s: rng.normal(size=(60, 8)) for s in ('probe_train', 'validation', 'test')}
    labels = {s: (v[:, 0] > .5).astype(int) for s, v in features.items()}
    first, _, _ = evaluate(features, labels, config, 1)
    changed = {**labels, 'test': 1 - labels['test']}
    second, _, _ = evaluate(features, changed, config, 1)
    for classifier in first:
        assert first[classifier]['selected_parameters'] == second[classifier]['selected_parameters']
        assert first[classifier]['splits']['validation'] == second[classifier]['splits']['validation']


def test_shared_initialization_mask_and_gradients(config):
    torch.set_num_threads(1)
    states = []
    for method in config['methods']:
        backbone = encoder(6101)
        states.append(state_hash(backbone))
        model = SSL(backbone, method, config)
        x = torch.rand(2, 3, 64, 64)
        inputs = x if method == 'masked_reconstruction' else [x, torch.rand_like(x)]
        loss, _ = model(inputs)
        loss.backward()
        assert torch.isfinite(loss)
        assert backbone.conv1.weight.grad.abs().sum() > 0
    assert len(set(states)) == 1
    assert state_hash(encoder(6102)) != states[0]
    mask = patch_mask(torch.rand(3, 3, 224, 224), 16, .75)
    assert mask.shape == (3, 1, 224, 224)
    assert torch.all(mask.sum((1, 2, 3)) == 147 * 16 * 16)


def test_end_to_end_and_epoch_boundary_resume(tmp_path, config, monkeypatch):
    torch.set_num_threads(1)
    c = copy.deepcopy(config)
    c.update(categories=['toy'], seeds=[1], epochs=2, checkpoint_epochs=[0, 1, 2], image_size=64,
             batch_size=4, logreg_C=[.1], lightgbm_grid=[dict(num_leaves=3, min_child_samples=2, n_estimators=5)])
    groups = {}
    rng = np.random.default_rng(4)
    for split in ('probe_train', 'validation', 'test'):
        groups[split] = []
        for i in range(8):
            path = f'{split}_{i}.png'
            Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(tmp_path / path)
            groups[split].append(dict(path=path, label=i % 2))
    groups['ssl'] = {'0.05': groups['probe_train']}
    manifest = {'manifest_sha256': 'toy', 'categories': {'toy': groups}}
    write_json(tmp_path / 'config.json', c)
    write_json(tmp_path / 'manifest.json', manifest)
    args = SimpleNamespace(config=tmp_path / 'config.json', manifest=tmp_path / 'manifest.json',
                           category='toy', seed=1, data=tmp_path, output=tmp_path / 'results',
                           method='baseline', fraction=.05, device=torch.device('cpu'), threads=1,
                           workers=0, deadline=float('inf'))
    assert run(args) == 0
    baseline = json.loads((args.output / 'toy_baseline_s1/epoch_000/metrics.json').read_text())
    for method in c['methods']:
        args.method = method
        assert run(args) == 0
        target = args.output / f'toy_{method}_p0.05_s1'
        zero = json.loads((target / 'epoch_000/metrics.json').read_text())
        assert zero == baseline
        assert (target / 'epoch_002/embeddings.npz').exists()
        assert (target / 'complete.json').exists()
        assert run(args) == 0
    # Interrupt after epoch one, then verify exact weights and metrics after resume.
    args.method = 'vicreg'
    reference_root = args.output
    args.output = tmp_path / 'resumed'
    args.method = 'baseline'
    assert run(args) == 0
    args.method = 'vicreg'
    module = importlib.import_module('visa_ssl.run')
    real_time = module.time
    times = iter([0., 2.])
    monkeypatch.setattr(module, 'time', SimpleNamespace(time=lambda: next(times), monotonic=real_time.monotonic))
    args.deadline = 1.
    assert run(args) == 75
    monkeypatch.setattr(module, 'time', real_time)
    args.deadline = float('inf')
    assert run(args) == 0
    name = 'toy_vicreg_p0.05_s1'
    original = torch.load(reference_root / name / 'final.pt', weights_only=False)['model']
    resumed = torch.load(args.output / name / 'final.pt', weights_only=False)['model']
    assert all(torch.equal(value, resumed[key]) for key, value in original.items())
    assert json.loads((reference_root / name / 'epoch_002/metrics.json').read_text()) == json.loads(
        (args.output / name / 'epoch_002/metrics.json').read_text())
