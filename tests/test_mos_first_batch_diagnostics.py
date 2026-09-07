import copy
import csv
import os
import random
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from algorithms.engine import mos_diagnostics as diag


def _args(**overrides):
    values = dict(mos_diagnostics=0, mos_diag_rounds='0,2', mos_diag_candidates=2,
                  radius_quantile=0.95, weight_radial=1.0, weight_sign=0.5,
                  sign_layer_reduce='quantile', sign_layer_quantile=0.9)
    values.update(overrides)
    return SimpleNamespace(**values)


def _snapshot():
    benign = torch.tensor([[0.0, 1.0], [0.2, 0.8], [-0.2, 1.2]])
    result = {
        'population': torch.tensor([[0.0, 1.0], [0.1, 1.0], [0.2, 1.0]]),
        'initial_population': torch.tensor([[0.0, 1.0], [0.05, 1.0]]),
        'benign_grads': benign, 'benign_mean': benign.mean(0),
        'benign_std': benign.std(0, correction=0) + 1e-9,
        'g_attack': torch.tensor([1.0, 0.0]),
        'max_dev_threshold': torch.tensor(0.2),
        'layer_dims': [('weight', 0, 2)], 'best_idx': 1,
    }
    result['constraints'] = diag._constraints(result, _args(), ['radial', 'sign'])
    return result


def test_diagnostics_default_off():
    assert diag.diagnostic_rounds(_args()) == set()
    assert diag.diagnostic_rounds(_args(mos_diagnostics=1)) == {0, 2}


def test_three_constraint_sets_run_and_are_finite():
    from algorithms.attack import mos
    rows = diag._sign_rows(0, _snapshot(), _args(), mos)
    assert [row['constraint_set'] for row in rows] == ['radial_only', 'sign_only', 'radial+sign']
    for row in rows:
        for key in ('alpha_feasible', 'max_feasible_A', 'boundary_left'):
            assert np.isfinite(row[key])
        assert row['boundary_right'] == '' or np.isfinite(row['boundary_right'])


def test_csv_writes_one_finite_row():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'sample.csv')
        row = {'round': 0, 'A': 0.2, 'CV': 0.0}
        diag._append_csv(path, list(row), [row])
        with open(path, newline='', encoding='utf-8') as handle:
            values = list(csv.DictReader(handle))
        assert len(values) == 1 and all(np.isfinite(float(values[0][key])) for key in ('A', 'CV'))


def test_effect_uses_copies():
    updates = [{'weight': torch.tensor([[1.0]])}, {'weight': torch.tensor([[0.0]])}]
    global_model = {'weight': torch.tensor([[3.0]])}
    original_updates, original_global = copy.deepcopy(updates), copy.deepcopy(global_model)
    model = torch.nn.Linear(1, 1, bias=False)
    with patch.object(diag, 'multi_krum', lambda values, multi_k: (values[0], np.array([0]))), \
         patch.object(diag, 'test_img', lambda model, dataset, args: (model.weight.item(), 0.0)):
        diag._effect(torch.tensor([2.0]), updates, 1, global_model, model, [0], _args())
    assert torch.equal(updates[0]['weight'], original_updates[0]['weight'])
    assert torch.equal(global_model['weight'], original_global['weight'])


def test_shadow_restores_state_and_reuses_initial_population():
    snapshot = _snapshot()
    updates = [{'weight': torch.tensor([0.0, 1.0])}, {'weight': torch.tensor([0.1, 1.0])}]
    updates_before = copy.deepcopy(updates)
    global_model = {'weight': torch.zeros(2)}
    global_before = copy.deepcopy(global_model)
    population_before = snapshot['population'].clone()

    class FakeMos:
        _LAST_VALID_GUIDANCE = torch.tensor([9.0])

        @staticmethod
        def _estimate_feasible_alpha(*unused_args, **unused_kwargs):
            return 0.5

        @staticmethod
        def mos_attack(values, args, malicious_count, **unused_kwargs):
            initial = torch.rand(2, 2)
            args._mos_diagnostic_snapshot = {
                'initial_population': initial, 'selection_mode': args.mos_objective_mode}
            values[0]['weight'] = torch.tensor([0.1, 1.0])
            FakeMos._LAST_VALID_GUIDANCE = torch.tensor([-1.0])
            return values, None

    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    cache_before = FakeMos._LAST_VALID_GUIDANCE.clone()
    live_args = _args(mos_objective_mode='dual')
    with patch.object(diag, '_effect', lambda *unused: (1.0, 2.0, 50.0, 40.0)):
        rows = diag._shadow_rows(0, snapshot, updates, 1, None, torch.ones(2), torch.ones(2),
                                 global_model, torch.nn.Linear(2, 1), [0],
                                 live_args, FakeMos)
    assert len(rows) == 2 and all(row['initial_population_equal'] for row in rows)
    assert torch.equal(snapshot['population'], population_before)
    assert all(torch.equal(updates[i]['weight'], updates_before[i]['weight']) for i in range(2))
    assert torch.equal(global_model['weight'], global_before['weight'])
    assert live_args.mos_objective_mode == 'dual'
    assert torch.equal(FakeMos._LAST_VALID_GUIDANCE, cache_before)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)


if __name__ == '__main__':
    tests = [value for name, value in sorted(globals().items()) if name.startswith('test_')]
    for test in tests:
        test()
    print(f'{len(tests)}/{len(tests)} MOS diagnostic smoke tests passed')
