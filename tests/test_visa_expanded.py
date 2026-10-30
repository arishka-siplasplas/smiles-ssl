import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pandas as pd
import pytest
import torch

from visa_ssl.data import make_manifest, write_json
from visa_ssl.expanded import jobs, restore
from visa_ssl.models import encoder, SSL, transform, UniformRotation, CutPaste, normalize
from visa_ssl.run import run, seed_all, state_hash, job_name
from visa_ssl.analyze_expanded import analyze

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def expanded_config():
    return json.loads((ROOT / 'configs/visa/expanded.json').read_text())


def test_full_train_fixed_official_splits(expanded_config):
    manifest = make_manifest(ROOT / 'manifests/visa/2cls_highshot.csv', expanded_config)
    assert manifest == json.loads((ROOT / 'manifests/visa-v2/fixed.json').read_text())
    changed = {**expanded_config, 'seeds': [55], 'augmentations': ['cutpaste'], 'architectures': ['deit_tiny']}
    assert make_manifest(ROOT / 'manifests/visa/2cls_highshot.csv', changed) == manifest
    previous = json.loads((ROOT / 'manifests/visa/fixed.json').read_text())
    for cat, group in manifest['categories'].items():
        assert group['ssl']['full_train'] == group['probe_train']
        assert sum(r['label'] for r in group['probe_train']) == 48
        assert sum(r['label'] for r in group['validation']) == 12
        assert sum(r['label'] for r in group['test']) == 40
        ids = [{r['path'] for r in group[s]} for s in ['probe_train', 'validation', 'test']]
        assert all(not ids[i] & ids[j] for i, j in [(0, 1), (0, 2), (1, 2)])
        if cat in previous['categories']:
            for split in ['probe_train', 'validation', 'test']:
                assert group[split] == previous['categories'][cat][split]
    assert sum(len(g['ssl']['full_train']) for g in manifest['categories'].values()) == 2261


def test_shards_cover_matrix_once_and_preflight_is_resumable(expanded_config):
    baseline, training = jobs(expanded_config)
    assert len(baseline) == 36 and len(training) == 324
    names = {job_name(j, True) for j in baseline + training}
    assert len(names) == 360
    shards = []
    for arch in expanded_config['architectures']:
        b, t = jobs(expanded_config, arch)
        assert len(b) == 18 and len(t) == 162
        shards.append({job_name(j, True) for j in b + t})
    assert not shards[0] & shards[1]
    assert shards[0] | shards[1] == names
    b, t = jobs(expanded_config, preflight=True)
    assert len(b) == 36 and len(t) == 6
    assert {job_name(j, True) for j in t} <= names
    for i in range(0, len(training), 3):
        assert [j['seed'] for j in training[i:i+3]] == expanded_config['seeds']


def test_augmentations_reproducible_independent_and_evaluation_fixed():
    pixels = np.random.default_rng(1).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    evaluation = transform(224)(image)
    for aug in ['baseline', 'rotate', 'cutpaste']:
        assert torch.equal(evaluation, transform(224, augmentation=aug)(image))
        seed_all(12)
        a = transform(224, True, aug)(image)
        b = transform(224, True, aug)(image)
        seed_all(12)
        assert torch.equal(a, transform(224, True, aug)(image))
        assert not torch.equal(a, b)
        assert a.shape == (3, 224, 224) and a.min() >= 0 and a.max() <= 1
    seed_all(4)
    rotated = np.array(UniformRotation()(image))
    assert any(np.array_equal(rotated, np.rot90(pixels, k)) for k in range(4))
    paste = CutPaste()
    paste.jitter = lambda x: x
    x = evaluation.clone()
    seed_all(42)
    result = paste(x)
    assert torch.equal(x, evaluation), 'CutPaste mutated the reconstruction source in-place'
    changed = (result != x).any(0)
    assert .015 <= changed.float().mean() <= .15
    rows, cols = changed.nonzero(as_tuple=True)
    rectangle = changed[rows.min():rows.max()+1, cols.min():cols.max()+1]
    assert rectangle.all()


def test_vit_pooling_initialization_and_all_ssl_gradients(expanded_config):
    torch.set_num_threads(1)
    initial = state_hash(encoder(6101, 'deit_tiny'))
    assert initial != state_hash(encoder(6102, 'deit_tiny'))
    x = torch.rand(2, 3, 224, 224)
    for method in expanded_config['methods']:
        model = encoder(6101, 'deit_tiny')
        assert state_hash(model) == initial
        assert len(model.blocks) == 12 and model.num_features == 192
        assert all(not block.attn.fused_attn for block in model.blocks)
        model.eval()
        with torch.no_grad():
            expected = model.forward_features(normalize(x))[:, 1:].mean(1)
            torch.testing.assert_close(model(normalize(x)), expected, rtol=0, atol=0)
        model.train()
        ssl = SSL(model, method, expanded_config)
        loss, _ = ssl(x if method == 'masked_reconstruction' else [x, x.flip(-1)])
        loss.backward()
        assert torch.isfinite(loss)
        assert model.patch_embed.proj.weight.grad.abs().sum() > 0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize('method,augmentation', [('simclr', 'baseline'), ('vicreg', 'rotate'),
                                                ('masked_reconstruction', 'cutpaste')])
def test_vit_full_protocol_and_exact_epoch_resume(tmp_path, expanded_config, method, augmentation):
    torch.set_num_threads(1)
    c = copy.deepcopy(expanded_config)
    c.update(categories=['toy'], seeds=[1], epochs=2, checkpoint_epochs=[0, 1, 2],
             image_size=64, batch_size=4, logreg_C=[.1],
             lightgbm_grid=[dict(num_leaves=3, min_child_samples=2, n_estimators=5)])
    groups = {}
    rng = np.random.default_rng(8)
    for split in ['probe_train', 'validation', 'test']:
        groups[split] = []
        for i in range(8):
            path = f'{split}_{i}.png'
            Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(tmp_path / path)
            groups[split].append(dict(path=path, label=i % 2))
    groups['ssl'] = {'full_train': groups['probe_train']}
    manifest = {'manifest_sha256': 'toy', 'categories': {'toy': groups}}
    write_json(tmp_path / 'config.json', c)
    write_json(tmp_path / 'manifest.json', manifest)
    args = SimpleNamespace(config=tmp_path / 'config.json', manifest=tmp_path / 'manifest.json',
        category='toy', seed=1, data=tmp_path, output=tmp_path / 'reference',
        architecture='deit_tiny', augmentation=augmentation, method='baseline',
        device=torch.device('cpu'), threads=1, workers=0, deadline=float('inf'), max_epoch=2)
    assert run(args) == 0
    args.method = method
    assert run(args) == 0
    name = job_name(vars(args), True)
    reference = args.output / name
    args.output = tmp_path / 'partial'
    args.method = 'baseline'
    assert run(args) == 0
    args.method, args.max_epoch = method, 1
    assert run(args) == 75
    assert not (args.output / name / 'complete.json').exists()
    state = torch.load(args.output / name / 'latest.pt', weights_only=False)
    assert state['epoch'] == 1 and state['optimizer'] and state['rng_state']
    resumed_root = tmp_path / 'resumed_session'
    resumed_root.mkdir()
    restored = restore(args.output, resumed_root, c, manifest, 'deit_tiny')
    assert len(restored) == 2
    with pytest.raises(ValueError, match='Incompatible'):
        restore(args.output, tmp_path / 'bad', {**c, 'learning_rate': .02}, manifest)
    args.output, args.max_epoch = resumed_root, 2
    assert run(args) == 0
    original = torch.load(reference / 'final.pt', weights_only=False)['encoder']
    resumed = torch.load(args.output / name / 'final.pt', weights_only=False)['encoder']
    assert all(torch.equal(v, resumed[k]) for k, v in original.items())
    assert json.loads((reference / 'epoch_002/metrics.json').read_text()) == json.loads(
        (args.output / name / 'epoch_002/metrics.json').read_text())
    baseline = args.output / 'toy_deit_tiny_baseline_s1/epoch_000/metrics.json'
    assert json.loads((reference / 'epoch_000/metrics.json').read_text()) == json.loads(baseline.read_text())
    analyze(args.output)
    q = pd.read_csv(args.output / 'analysis/quality.csv')
    assert set(q.architecture) == {'deit_tiny'}
    assert set(q.augmentation) == {'none', augmentation}
    assert set(q[q.method != 'baseline'].epoch) == {0, 1, 2}
