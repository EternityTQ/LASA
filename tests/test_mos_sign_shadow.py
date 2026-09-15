import copy
import csv
import os
import random
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from algorithms.defense.lasa import _lasa_legacy, audit_candidate_under_lasa, lasa
from algorithms.engine import mos_sign_shadow as shadow


def _args(**overrides):
    values = dict(
        mos_sign_shadow_diagnostics=1, mos_diag_rounds='0,20,40,59',
        sparsity=0.0, lambda_n=10.0, lambda_s=10.0,
        radius_quantile=0.95, weight_radial=1.0, weight_sign=0.5,
        sign_layer_quantile=0.9, device='cpu', test_batch_size=4,
        dataset='toy', model='linear', num_selected_users=5,
        mos_sign_shared_candidates=10, mos_lasa_boundary_evals=10)
    values.update(overrides)
    return SimpleNamespace(**values)


def _update(values):
    values = torch.tensor(values, dtype=torch.float32)
    return {'weight': values[:4].reshape(2, 2), 'bias': values[4:]}


def _snapshot():
    updates = [
        _update([.2, -.1, .1, -.2, .05, -.05]),
        _update([.1, .3, -.2, .2, .10, .02]),
        _update([.3, .2, -.1, .1, .02, .08]),
        _update([.2, .1, .3, -.1, -.03, .04]),
        _update([.1, .2, .2, .3, .06, -.02]),
    ]
    benign = torch.stack([torch.cat([v.flatten() for v in u.values()])
                          for u in updates[1:]])
    model = torch.nn.Linear(2, 2)
    return {
        'round': 0, 'global_model': copy.deepcopy(model.state_dict()),
        'round_start_global_model': copy.deepcopy(model.state_dict()),
        'model_template': model, 'local_updates': updates,
        'benign_updates': updates[1:], 'malicious_count': 1,
        'selected_client_ids': list(range(5)), 'malicious_client_ids': [0],
        'benign_grads': benign, 'benign_mean': benign.mean(0),
        'g_attack': torch.nn.functional.normalize(torch.ones(6), dim=0),
        'g_ce': torch.ones(6), 'g_cw': torch.ones(6),
        'layer_dims': [('weight', 0, 4), ('bias', 4, 6)],
        'max_dev_threshold': torch.tensor(.2), 'attack_budget': .2,
        'radial_threshold': .2, 'sign_threshold': 1.,
        'sign_config': {}, 'lasa_config': {},
        'diagnostic_config': vars(_args()),
        'diagnostic_images': torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 1.]]),
        'diagnostic_labels': torch.tensor([0, 1, 0, 1]),
        'random_state': shadow._rng_state_cpu(), 'update_template': updates[0],
    }


def _same_rng(left, right):
    assert left['python'] == right['python']
    assert left['numpy'][0] == right['numpy'][0]
    assert np.array_equal(left['numpy'][1], right['numpy'][1])
    assert left['numpy'][2:] == right['numpy'][2:]
    assert torch.equal(left['torch'], right['torch'])
    if left['cuda'] is not None:
        assert all(torch.equal(a, b) for a, b in zip(left['cuda'], right['cuda']))


def test_default_gate_and_rounds():
    assert shadow.enabled_rounds(_args(mos_sign_shadow_diagnostics=0)) == set()
    assert shadow.enabled_rounds(_args()) == {0, 20, 40, 59}


def test_lasa_audit_is_read_only_and_has_per_layer_data():
    snapshot = _snapshot()
    before = copy.deepcopy(snapshot['local_updates'])
    audit = audit_candidate_under_lasa(
        snapshot['local_updates'][0], snapshot['local_updates'],
        snapshot['global_model'], _args(), 0)
    assert len(audit['lasa_layer_audit']) == 2
    assert 0 <= audit['lasa_joint_pass_fraction'] <= 1
    for current, original in zip(snapshot['local_updates'], before):
        for key in current:
            assert torch.equal(current[key], original[key])


def test_shared_lasa_path_matches_legacy_on_finite_inputs():
    snapshot = _snapshot()
    expected = _lasa_legacy(copy.deepcopy(snapshot['local_updates']),
                            copy.deepcopy(snapshot['global_model']), _args())
    actual = lasa(copy.deepcopy(snapshot['local_updates']),
                  copy.deepcopy(snapshot['global_model']), _args())
    for key in expected:
        assert torch.equal(actual[key], expected[key])


def test_all_proxies_snapshot_reload_and_csv():
    snapshot = _snapshot()
    updates_before = copy.deepcopy(snapshot['local_updates'])
    state_before = copy.deepcopy(snapshot['global_model'])
    model_before = copy.deepcopy(snapshot['model_template'].state_dict())
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'diagnostic_snapshot_round_0.pt')
        torch.save(snapshot, path)
        loaded = shadow.load_snapshot(path)
        rows = shadow.analyze_snapshot(loaded, _args(), directory)
        modes = {row['proxy_mode'] for row in rows if not row.get('failure')}
        assert {'current', 'global_sign', 'lasa_faithful'} <= modes
        assert os.path.exists(os.path.join(directory, 'sign_proxy_summary.csv'))
        assert os.path.exists(os.path.join(directory, 'sign_proxy_common_correlations.csv'))
        with open(os.path.join(directory, 'sign_proxy_comparison.csv'),
                  newline='', encoding='utf-8') as handle:
            csv_rows = list(csv.DictReader(handle))
        assert len(csv_rows) >= 6
        with open(os.path.join(directory, 'sign_proxy_common_candidates.csv'),
                  newline='', encoding='utf-8') as handle:
            common = list(csv.DictReader(handle))
        candidate_ids = {row['candidate_id'] for row in common}
        assert len(candidate_ids) >= 10
        for mode in shadow.PROXY_MODES:
            assert {row['candidate_id'] for row in common
                    if row['proxy_mode'] == mode} == candidate_ids
        with open(os.path.join(directory, 'sign_proxy_recommendation.txt'),
                  encoding='utf-8') as handle:
            assert 'provisional/incomplete' in handle.read()
    for current, original in zip(snapshot['local_updates'], updates_before):
        for key in current:
            assert torch.equal(current[key], original[key])
    for key in snapshot['global_model']:
        assert torch.equal(snapshot['global_model'][key], state_before[key])
    for key in snapshot['model_template'].state_dict():
        assert torch.equal(snapshot['model_template'].state_dict()[key],
                           model_before[key])


def test_lasa_faithful_skips_batch_counters():
    benign = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [-1., 0.]])
    proxy = shadow._LasaFaithful(
        benign, benign.mean(0),
        {'weight': torch.ones(1, 1), 'num_batches_tracked': torch.zeros(1)},
        [('weight', 0, 1), ('num_batches_tracked', 1, 2)], _args(), 1)
    value = proxy.loss(benign.mean(0).unsqueeze(0), benign.mean(0), {})
    assert torch.isfinite(value).all()


def test_full_diagnostic_preserves_training_state_and_rng():
    snapshot = _snapshot()
    updates_before = copy.deepcopy(snapshot['local_updates'])
    state_before = copy.deepcopy(snapshot['global_model'])
    model_before = copy.deepcopy(snapshot['model_template'].state_dict())
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    rng_before = shadow._capture_rng()
    with tempfile.TemporaryDirectory() as directory:
        args = _args(mos_diagnostics_dir=directory, mos_diag_samples=4)
        mos_snapshot = {key: snapshot[key] for key in (
            'benign_grads', 'benign_mean', 'g_attack', 'layer_dims',
            'max_dev_threshold', 'radial_threshold', 'sign_threshold')}
        original_analyzer = shadow.analyze_snapshot
        shadow.analyze_snapshot = lambda *unused, **unused_kw: (_ for _ in ()).throw(
            AssertionError('online snapshot path must not analyze'))
        try:
            shadow.run_sign_shadow_diagnostics(
                0, mos_snapshot, snapshot['local_updates'], 1, list(range(5)), [0],
                snapshot['global_model'], snapshot['model_template'],
                list(zip(snapshot['diagnostic_images'], snapshot['diagnostic_labels'])),
                [], snapshot['g_ce'], snapshot['g_cw'], args)
        finally:
            shadow.analyze_snapshot = original_analyzer
        assert os.path.exists(os.path.join(
            directory, 'diagnostic_snapshot_round_0.pt'))
        compact = shadow.load_snapshot(os.path.join(
            directory, 'diagnostic_snapshot_round_0.pt'))
        assert compact['snapshot_version'] == 3
        assert 'client_updates' in compact
        assert 'local_updates' not in compact
        assert 'benign_updates' not in compact
        assert 'benign_grads' not in compact
        assert compact['attacker_positions'] == [0]
        assert compact['benign_positions'] == [1, 2, 3, 4]
        assert compact['canonical_client_ids'] == [0, 1, 2, 3, 4]
    _same_rng(rng_before, shadow._capture_rng())
    for current, original in zip(snapshot['local_updates'], updates_before):
        for key in current:
            assert torch.equal(current[key], original[key])
    for key in snapshot['global_model']:
        assert torch.equal(snapshot['global_model'][key], state_before[key])
    for key in snapshot['model_template'].state_dict():
        assert torch.equal(snapshot['model_template'].state_dict()[key],
                           model_before[key])


def test_shared_candidates_execute_one_lasa_audit_each():
    snapshot = _snapshot()
    shadow._prepare_snapshot(snapshot, torch.device('cpu'))
    args = _args()
    proxy_data = {}
    for mode, alpha in (('current', .2), ('global_sign', .4)):
        proxy_data[mode] = {
            'constraints': shadow._make_constraints(snapshot, args, mode),
            'boundary': (alpha, alpha * .2, None, None),
            'status': 'valid', 'status_reason': '',
        }
    calls = []
    original = shadow._evaluate_candidate_under_lasa
    shadow._evaluate_candidate_under_lasa = lambda candidate, unused, unused_args: (
        calls.append(candidate.clone()) or ({}, []))
    try:
        cache = shadow._LasaEvaluationCache(snapshot, args)
        rows, _ = shadow._common_candidate_rows(snapshot, args, proxy_data, cache)
    finally:
        shadow._evaluate_candidate_under_lasa = original
    candidate_ids = {row['candidate_id'] for row in rows}
    assert len(calls) == len(candidate_ids)
    assert len(rows) == len(candidate_ids) * len(proxy_data)


if __name__ == '__main__':
    test_default_gate_and_rounds()
    test_lasa_audit_is_read_only_and_has_per_layer_data()
    test_shared_lasa_path_matches_legacy_on_finite_inputs()
    test_all_proxies_snapshot_reload_and_csv()
    test_lasa_faithful_skips_batch_counters()
    test_full_diagnostic_preserves_training_state_and_rng()
    test_shared_candidates_execute_one_lasa_audit_each()
    print('7/7 MOS Sign shadow smoke tests passed')
