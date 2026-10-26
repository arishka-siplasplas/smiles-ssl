import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from smiles_ssl.augment import Corruptor, donors
from smiles_ssl.config import default_config, grid, validate_config
from smiles_ssl.data import grouped_split, prepare_table, prepare_mvtec, load_splits
from smiles_ssl.io import write_json, read_json
from smiles_ssl.metrics import spectral
from smiles_ssl.models import SSLModel, masked_mse, nt_xent, vicreg
from smiles_ssl.run import train, evaluate_test

METHODS = ['simclr', 'vicreg', 'masked_reconstruction']


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def table_fixture(root):
    rng = np.random.default_rng(8)
    x = rng.normal(size=(90, 6))
    y = np.tile([0, 0, 1], 30)
    x[:, 0] += y
    path = root / 'fixture.npz'
    np.savez(path, X=x, y=y)
    return path


def image_fixture(root):
    rng = np.random.default_rng(9)
    for origin, defect, n in [('train', 'good', 12), ('test', 'good', 6), ('test', 'scratch', 9), ('test', 'dent', 9)]:
        folder = root / 'bottle' / origin / defect
        folder.mkdir(parents=True)
        for i in range(n):
            x = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(x).save(folder / f'{i:03d}.png')


def smoke_config(modality, dataset, method):
    c = default_config(modality, dataset, method, seed=19)
    c.update(stage='smoke', epochs=1, checkpoints=[0, 1], short_epoch=0,
             batch_size=12, image_size=32, warmup_epochs=0)
    c['augmentation']['patch_size'] = 8
    c['probe'].update(logreg_C=[1.], class_weights=[None], catboost_depth=[2],
                      catboost_l2=[3.], catboost_iterations=3)
    return c


def test_grouped_split_has_no_duplicate_leakage():
    y = np.tile([0, 1], 30)
    groups = np.repeat(np.arange(30), 2)  # deliberately mixed-label duplicate groups
    a = grouped_split(y, y, groups, 71)
    assert a == grouped_split(y, y, groups, 71)
    assert sorted(sum(a.values(), [])) == list(range(len(y)))
    sets = [set(groups[ids]) for ids in a.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    for ids in a.values():
        assert set(y[ids]) == {0, 1}


def test_preprocessing_uses_train_only_and_checks_source(tmp_path):
    path = table_fixture(tmp_path)
    m = prepare_table(path, 'fixture')
    with np.load(path) as f:
        x, y = f['X'].copy(), f['y'].copy()
    assert np.allclose(m['preprocessing']['mean'], x[m['splits']['train']].mean(0))
    original = copy.deepcopy(m['preprocessing'])
    x[m['splits']['validation']] += 500
    # Fixed split train stats cannot depend on held-out rows.
    assert np.allclose(original['mean'], x[m['splits']['train']].mean(0))
    np.savez(path, X=x, y=y)
    with pytest.raises(ValueError, match='hash'):
        load_splits(m, tmp_path)


def test_mvtec_modified_split_preserves_defect_types(tmp_path):
    image_fixture(tmp_path)
    m = prepare_mvtec(tmp_path, 'bottle')
    assert 'MODIFIED' in m['protocol']
    moved = set(m['moved_original_test_to_train'])
    test = {m['records'][i]['path'] for i in m['splits']['test']}
    assert moved and not moved & test
    for s in m['summary'].values():
        assert set(s['strata']) == {'good', 'scratch', 'dent'}
        assert s['anomaly'] > 0 and s['normal'] > 0


def test_manifest_overlap_is_rejected(tmp_path):
    path = table_fixture(tmp_path)
    m = prepare_table(path, 'fixture')
    m['splits']['test'][0] = m['splits']['train'][0]
    with pytest.raises(ValueError, match='overlap'):
        load_splits(m, tmp_path)


def test_archive_path_traversal_is_rejected(tmp_path):
    import io
    import tarfile
    from smiles_ssl.download import extract_category
    archive = tmp_path / 'bad.tar.xz'
    with tarfile.open(archive, 'w:xz') as f:
        entry = tarfile.TarInfo('bottle/../../escaped.txt')
        entry.size = 1
        f.addfile(entry, io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='Unexpected archive path'):
        extract_category(archive, tmp_path / 'output', 'bottle')


def test_table_corruptions_zero_mean_donor_and_noise():
    x = torch.ones(12, 6)
    cfg = {'family': 'zero', 'fraction': .5, 'sigma': .2}
    zero = Corruptor('table', cfg, mean=[4] * 6, scale=[2] * 6)
    a, mask = zero(x)
    assert torch.all(a[mask] == -2) and torch.equal(a[~mask], x[~mask])
    mean = Corruptor('table', {**cfg, 'family': 'mean'})
    a, mask = mean(x)
    assert torch.all(a[mask] == 0)
    noise = Corruptor('table', {**cfg, 'family': 'noise'}, noise_columns=[1, 3])
    a, mask = noise(x)
    assert torch.equal(a[:, [0, 2, 4, 5]], x[:, [0, 2, 4, 5]])
    assert not mask[:, [0, 2, 4, 5]].any()
    bank = torch.arange(60).reshape(10, 6).float()
    idx = torch.arange(10)
    d = donors(bank, idx, 'table', 'cpu')
    assert torch.all(d != bank)
    for j in range(6):
        assert set(d[:, j].tolist()) <= set(bank[:, j].tolist())


@pytest.mark.parametrize('family', ['zero', 'mean', 'donor', 'noise'])
def test_patch_corruptions(family):
    x = torch.rand(4, 3, 32, 32)
    cfg = {'family': family, 'fraction': .5, 'sigma': .1, 'patch_size': 8}
    a, mask = Corruptor('image', cfg, image_mean=[.1, .2, .3])(x, torch.zeros_like(x))
    assert mask.shape == (4, 1, 32, 32)
    assert mask.sum() == 4 * 32 * 32 / 2
    visible = ~mask.expand_as(x)
    assert torch.equal(a[visible], x[visible])
    assert a.min() >= 0 and a.max() <= 1


def test_loss_formulas_and_reconstruction_gradient():
    a = torch.tensor([[1., 0.], [0., 1.]])
    loss, _ = nt_xent(a, a, 1.)
    assert float(loss) == pytest.approx(np.log(np.e + 2) - 1)
    collapsed, _ = vicreg(torch.zeros(8, 6), torch.zeros(8, 6))
    assert float(collapsed) == pytest.approx(24.75)
    p = torch.randn(2, 6, requires_grad=True)
    mask = torch.tensor([[1, 0, 1, 0, 0, 0]] * 2, dtype=torch.bool)
    masked_mse(p, torch.zeros_like(p), mask).backward()
    assert torch.all(p.grad[~mask] == 0) and torch.any(p.grad[mask] != 0)


@pytest.mark.parametrize('modality', ['table', 'image'])
def test_same_encoder_initialization_and_gradients(modality):
    models = [SSLModel(modality, m, 6, 32, 11) for m in METHODS]
    for key, value in models[0].encoder.state_dict().items():
        assert all(torch.equal(value, m.encoder.state_dict()[key]) for m in models[1:])
    x = torch.rand(4, 6) if modality == 'table' else torch.rand(4, 3, 32, 32)
    mask = torch.rand_like(x) > .5
    for m in models:
        loss, _ = m.objective(x, x.masked_fill(mask, 0), mask, x * .9)
        loss.backward()
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in m.encoder.parameters())


def test_spectrum_invariance_and_collapse():
    rng = np.random.default_rng(42)
    z = rng.normal(size=(100, 6))
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    a, b = spectral(z), spectral(7 * z @ q + 5)
    for name in ['effective_rank', 'singular_effective_rank', 'participation_ratio', 'stable_rank']:
        assert a[name] == pytest.approx(b[name])
    assert b['total_variance'] == pytest.approx(49 * a['total_variance'])
    assert spectral(np.zeros((10, 6)))['effective_rank'] is None
    assert spectral(np.arange(10)[:, None] * np.ones((1, 6)))['effective_rank'] == pytest.approx(1.)


def test_probes_report_collapse_without_catboost_crash():
    from smiles_ssl.probes import fit_probes
    c = smoke_config('table', 'fixture', 'simclr')
    train = np.zeros((30, 6))
    y = np.tile([0, 0, 1], 10)
    result, _ = fit_probes(train, y, train, y, c['probe'], 1)
    assert result['catboost']['selected']['constant_feature_fallback']
    assert result['catboost']['validation']['ap'] == pytest.approx(1 / 3)
    assert result['logreg']['validation']['roc_auc'] == pytest.approx(.5)


def test_noise_allowlist_can_only_restrict_train_eligible_columns(tmp_path):
    path = table_fixture(tmp_path)
    m = prepare_table(path, 'fixture', noise_columns=[1, 3])
    assert m['preprocessing']['noise_columns'] == [1, 3]
    with pytest.raises(ValueError, match='allowlist'):
        prepare_table(path, 'fixture', noise_columns=[100])


def test_grid_is_complete():
    assert len(grid()) == 270
    assert len(grid('augmentations')) == 3240
    for c in grid():
        validate_config(c)


@pytest.mark.parametrize('modality', ['table', 'image'])
@pytest.mark.parametrize('method', METHODS)
def test_end_to_end(tmp_path, modality, method):
    if modality == 'table':
        m = prepare_table(table_fixture(tmp_path), 'fixture')
    else:
        image_fixture(tmp_path)
        m = prepare_mvtec(tmp_path, 'bottle')
    c = smoke_config(modality, m['dataset'], method)
    cp, mp, out = tmp_path / 'config.json', tmp_path / 'manifest.json', tmp_path / 'run'
    write_json(cp, c)
    write_json(mp, m)
    train(cp, mp, tmp_path, out)
    assert not (out / 'test.json').exists()
    evaluate_test(out, tmp_path)
    result = read_json(out / 'test.json')
    assert len(result['rows']) == 2
    for row in result['rows']:
        assert set(row['probes']) == {'logreg', 'catboost'}
        assert all(0 <= v['ap'] <= 1 and 0 <= v['roc_auc'] <= 1 for v in row['probes'].values())
    with pytest.raises(FileExistsError):
        evaluate_test(out, tmp_path)
    if method == 'simclr' and modality == 'table':
        from smiles_ssl.analyze import analyze
        analyze(tmp_path, tmp_path / 'plots')
        assert len(list((tmp_path / 'plots').glob('*.png'))) == 2


def test_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    import smiles_ssl.run as runner
    m = prepare_table(table_fixture(tmp_path), 'fixture')
    c = smoke_config('table', 'fixture', 'vicreg')
    c.update(epochs=2, checkpoints=[0, 1, 2])
    cp, mp = tmp_path / 'c.json', tmp_path / 'm.json'
    write_json(cp, c)
    write_json(mp, m)
    train(cp, mp, tmp_path, tmp_path / 'gold')
    original = runner.evaluate_checkpoint
    def interrupted(*args, **kwargs):
        if args[5] == 2:
            raise RuntimeError('simulated interruption')
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, 'evaluate_checkpoint', interrupted)
    with pytest.raises(RuntimeError, match='interruption'):
        train(cp, mp, tmp_path, tmp_path / 'resumed')
    monkeypatch.setattr(runner, 'evaluate_checkpoint', original)
    train(cp, mp, tmp_path, tmp_path / 'resumed', resume=True)
    assert read_json(tmp_path / 'gold/loss.json') == read_json(tmp_path / 'resumed/loss.json')
    a = torch.load(tmp_path / 'gold/epoch_0002/encoder.pt', weights_only=True)
    b = torch.load(tmp_path / 'resumed/epoch_0002/encoder.pt', weights_only=True)
    assert all(torch.equal(a[k], b[k]) for k in a)
