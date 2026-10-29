from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from smiles_ssl.augment import select_mask, Corruptor
from smiles_ssl.calibration import (screen, screen_spec, candidates, candidate_id, corrupt_queries,
                                    make_pilot_plan, select_and_promote, shortlist)
from smiles_ssl.config import default_config, validate_config, run_id
from smiles_ssl.data import prepare_table
from smiles_ssl.io import read_json, write_json
from smiles_ssl.run import train


def test_weak_masks_do_not_round_up_to_one_feature():
    torch.manual_seed(18)
    a = select_mask((50000, 6), .01, 'cpu', sampling='bernoulli')
    assert a.float().mean().item() == pytest.approx(.01, abs=.001)
    assert (~a.any(1)).float().mean().item() == pytest.approx(.99 ** 6, abs=.004)
    assert not select_mask((10, 6), 0., 'cpu', sampling='bernoulli').any()
    assert select_mask((100, 6), .01, 'cpu').sum().item() == 100  # historical default retained


def test_mask_modes_have_distinct_run_ids_and_no_zero_target_mae():
    c = default_config('table', 'cardio', 'simclr')
    old = run_id(c)
    c['augmentation']['mask_sampling'] = 'bernoulli'
    assert run_id(c) != old
    c['augmentation']['fraction'] = 0.
    validate_config(c)
    c['method'] = 'masked_reconstruction'
    with pytest.raises(ValueError):
        validate_config(c)


def test_held_out_donors_only_use_fit_and_corruptor_has_no_label_input():
    query = torch.full((500, 6), -999.)
    bank = torch.arange(120).reshape(20, 6).float()
    a = {'family': 'donor', 'fraction': .2, 'sigma': 0., 'mask_sampling': 'bernoulli'}
    prep = {'mean': [0.] * 6, 'scale': [1.] * 6, 'noise_columns': list(range(6))}
    z, mask = corrupt_queries(query, bank, prep, a, 61)
    assert torch.equal(z[~mask], query[~mask])
    for j in range(6):
        assert set(z[mask[:, j], j].tolist()) <= set(bank[:, j].tolist())
    with pytest.raises(TypeError):
        Corruptor('table', a)(query, labels=torch.ones(500))


def test_shortlist_does_not_rescue_flagged_family():
    rows = [{'id': candidate_id(a), 'augmentation': a,
             'risk': 'flagged' if a['family'] == 'donor' else 'not_flagged'} for a in candidates()]
    chosen = shortlist(rows)
    assert len(chosen) == 6
    assert not any(k.startswith('donor') for k in chosen)


def test_screen_internal_folds_and_complete_selection(tmp_path):
    rng = np.random.default_rng(91)
    x = rng.normal(size=(240, 6))
    y = np.tile([0, 0, 1], 80)
    x[:, 0] += y
    x[1] = x[0]
    np.savez(tmp_path / 'fixture.npz', X=x, y=y)
    m = prepare_table(tmp_path / 'fixture.npz', 'fixture')
    mp = tmp_path / 'manifest.json'
    write_json(mp, m)
    spec = screen_spec()
    spec.update(draw_seeds=[7101, 7102], catboost_iterations=3)
    report = screen(mp, tmp_path, tmp_path / 'screen', spec)
    assert len(report['candidates']) == 21
    folds = read_json(tmp_path / 'screen/inner_folds.json')
    train_ids = set(m['splits']['train'])
    seen_audit = []
    for f in folds:
        fit, audit = f['fit_source_indices'], f['audit_source_indices']
        assert set(fit) | set(audit) == train_ids
        assert not set(fit) & set(audit)
        assert np.allclose(f['preprocessing']['mean'], x[fit].mean(0))
        seen_audit.extend(audit)
    assert set(seen_audit) == train_ids and len(seen_audit) == len(train_ids)
    # A fixture plan with two candidates from one family, all methods/seeds.
    report['shortlist'] = ['mean_p0.01_s0', 'mean_p0.2_s0']
    write_json(tmp_path / 'screen/screen.json', report)
    jobs = make_pilot_plan([tmp_path / 'screen/screen.json'], tmp_path / 'plan')
    assert len(jobs) == 24  # 18 candidates + 6 identity controls
    for job in jobs:
        validate_config(read_json(tmp_path / 'plan' / job['config']))
    with pytest.raises(ValueError, match='incomplete'):
        select_and_promote(tmp_path / 'plan/plan.json', tmp_path / 'missing', tmp_path / 'must_not_exist')
    assert not (tmp_path / 'must_not_exist').exists()
    # Deliberately use clean validation scores only; no test artifacts exist.
    for job in jobs:
        folder = tmp_path / 'runs' / job['id']
        c = read_json(tmp_path / 'plan' / job['config'])
        write_json(folder / 'config.json', c)
        write_json(folder / 'manifest.json', m)
        write_json(folder / 'completion.json', {'epochs': 100})
        ap = .802 if job['candidate_id'] == 'mean_p0.2_s0' else .8
        write_json(folder / 'epoch_0100/evaluation.json', {'probes': {
            name: {'validation': {'ap': ap}} for name in ['logreg', 'catboost']}})
    select_and_promote(tmp_path / 'plan/plan.json', tmp_path / 'runs', tmp_path / 'selected')
    result = read_json(tmp_path / 'selected/selection.json')
    assert result['selections'][0]['chosen']['candidate'] == 'mean_p0.01_s0'
    assert not result['selections'][0]['review_before_long_runs']
    assert result['selections'][0]['chosen_minus_identity_ap']['simclr']['logreg'] == pytest.approx(0.)
    assert read_json(tmp_path / 'selected/plan.json')['n_runs'] == 15


def test_sparse_reconstruction_training_reports_real_mask_rate(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(12)
    np.savez(tmp_path / 'fixture.npz', X=rng.normal(size=(180, 6)), y=np.tile([0, 0, 1], 60))
    m = prepare_table(tmp_path / 'fixture.npz', 'fixture')
    c = default_config('table', 'fixture', 'masked_reconstruction', seed=19)
    c.update(stage='smoke', epochs=2, schedule_epochs=1000, checkpoints=[0, 2], short_epoch=0, batch_size=16)
    c['augmentation'].update(mask_sampling='bernoulli', fraction=.01)
    c['probe'].update(logreg_C=[1.], class_weights=[None], catboost_depth=[2], catboost_l2=[3.], catboost_iterations=3)
    write_json(tmp_path / 'c.json', c)
    write_json(tmp_path / 'm.json', m)
    train(tmp_path / 'c.json', tmp_path / 'm.json', tmp_path, tmp_path / 'run')
    rows = read_json(tmp_path / 'run/loss.json')
    assert sum(r['empty_target_batches_skipped'] for r in rows) > 0
    assert all(0 < r['first_view_selected_cell_fraction'] < .05 for r in rows)
    assert all(r['steps'] > 0 for r in rows)
