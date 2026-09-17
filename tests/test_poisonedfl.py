import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import numpy as np
import copy


MODULE_PATH = Path(__file__).parents[1] / "algorithms" / "attack" / "poisonedfl.py"
SPEC = importlib.util.spec_from_file_location("poisonedfl_under_test", MODULE_PATH)
POISONEDFL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POISONEDFL)


def _args(scale=8.0, interval=50):
    return SimpleNamespace(
        poisonedfl_scale_factor=scale,
        poisonedfl_feedback_interval=interval,
    )


def _global(weight_value=1.0):
    return {
        "weight": torch.full((2, 2), weight_value, dtype=torch.float32),
        "bias": torch.full((2,), weight_value, dtype=torch.float64),
        "running_mean": torch.full((2,), 3.0, dtype=torch.float32),
        "num_batches_tracked": torch.tensor(7, dtype=torch.int64),
    }


def _update(offset):
    return {
        "weight": torch.full((2, 2), offset, dtype=torch.float32),
        "bias": torch.full((2,), offset, dtype=torch.float64),
        "running_mean": torch.full((2,), offset + 10, dtype=torch.float32),
        "num_batches_tracked": torch.tensor(offset + 20, dtype=torch.int64),
    }


class PoisonedFLTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.keys = ("weight", "bias")

    def test_warmup_replaces_only_malicious_trainable_floats(self):
        updates = [_update(1), _update(2), _update(3)]
        original_buffers = [
            (item["running_mean"].clone(), item["num_batches_tracked"].clone())
            for item in updates
        ]
        benign_before = {key: value.clone() for key, value in updates[2].items()}

        result, state = POISONEDFL.poisonedfl_attack(
            updates, _args(), 2, global_model=_global(),
            trainable_keys=self.keys, round_idx=0,
        )

        for index in (0, 1):
            self.assertEqual(result[index]["weight"].shape, (2, 2))
            self.assertEqual(result[index]["weight"].dtype, torch.float32)
            self.assertEqual(result[index]["bias"].dtype, torch.float64)
            self.assertTrue(torch.count_nonzero(result[index]["weight"]) == 0)
            self.assertTrue(torch.count_nonzero(result[index]["bias"]) == 0)
            self.assertTrue(torch.equal(result[index]["running_mean"], original_buffers[index][0]))
            self.assertTrue(torch.equal(result[index]["num_batches_tracked"], original_buffers[index][1]))
        for key in benign_before:
            self.assertTrue(torch.equal(result[2][key], benign_before[key]))
        self.assertEqual(state["last_round"], 0)
        self.assertEqual(state["dimension"], 6)

    def test_second_round_is_nonzero_and_zero_attacker_round_keeps_last_attack(self):
        _, state = POISONEDFL.poisonedfl_attack(
            [_update(1), _update(2)], _args(), 1,
            global_model=_global(1), trainable_keys=self.keys, round_idx=0,
        )
        result, state = POISONEDFL.poisonedfl_attack(
            [_update(1), _update(2)], _args(), 1,
            global_model=_global(2), trainable_keys=self.keys, round_idx=1,
            state=state,
        )
        self.assertGreater(torch.linalg.vector_norm(result[0]["weight"]).item(), 0)
        last_malicious = state["last_malicious"].clone()
        benign_before = _update(5)
        no_attack, state = POISONEDFL.poisonedfl_attack(
            [{key: value.clone() for key, value in benign_before.items()}],
            _args(), 0, global_model=_global(3), trainable_keys=self.keys,
            round_idx=2, state=state,
        )
        for key in benign_before:
            self.assertTrue(torch.equal(no_attack[0][key], benign_before[key]))
        self.assertTrue(torch.equal(state["last_malicious"], last_malicious))
        self.assertEqual(state["last_round"], 2)

    def test_feedback_decay_and_official_threshold_approximation(self):
        _, state = POISONEDFL.poisonedfl_attack(
            [_update(1)], _args(interval=2), 1,
            global_model=_global(0), trainable_keys=self.keys, round_idx=0,
        )
        _, state = POISONEDFL.poisonedfl_attack(
            [_update(1)], _args(interval=2), 1,
            global_model=_global(1), trainable_keys=self.keys, round_idx=1,
            state=state,
        )
        state["fixed_sign"].fill_(1)
        _, state = POISONEDFL.poisonedfl_attack(
            [_update(1)], _args(interval=2), 1,
            global_model=_global(0), trainable_keys=self.keys, round_idx=2,
            state=state,
        )
        self.assertAlmostEqual(state["scale_factor"], 5.6)

        official_k99 = {
            1204682: 603618,
            139960: 70415,
            717924: 359948,
            145212: 73049,
        }
        for dimension, expected in official_k99.items():
            actual = POISONEDFL._alignment_threshold_99(dimension)
            self.assertLessEqual(abs(actual - expected), 1)

    def test_rejects_integer_trainable_parameter(self):
        with self.assertRaises(TypeError):
            POISONEDFL.poisonedfl_attack(
                [_update(1)], _args(), 1, global_model=_global(),
                trainable_keys=("num_batches_tracked",), round_idx=0,
            )

    def test_single_synthetic_update_matches_official_numpy_steps(self):
        # Independent NumPy transcription of official byzantine.py lines 50-69.
        _, state = POISONEDFL.poisonedfl_attack(
            [_update(1), _update(2)], _args(), 1, global_model=_global(0),
            trainable_keys=self.keys, round_idx=0)
        old = np.array([0.3, -0.2, 0.6, 0.1, -0.5, 0.4])
        state['last_malicious'] = torch.tensor(old)
        history = np.array([1., -2., 3., -4., 5., -6.])
        current = _global(0)
        current['weight'] = torch.tensor(history[:4], dtype=torch.float32).reshape(2, 2)
        current['bias'] = torch.tensor(history[4:])
        residual = np.abs(history - old * np.linalg.norm(history) / (np.linalg.norm(old) + 1e-9))
        expected = 8 * np.linalg.norm(history) * residual * state['fixed_sign'].numpy() / (np.linalg.norm(residual) + 1e-9)
        result, state = POISONEDFL.poisonedfl_attack(
            [_update(1), _update(2)], _args(), 1, global_model=current,
            trainable_keys=self.keys, round_idx=1, state=state)
        actual = POISONEDFL._flatten_named(result[0], self.keys).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        # One synthetic server application; no data, optimizer or training.
        updated = copy.deepcopy(current)
        for key in self.keys:
            updated[key] += (result[0][key] + result[1][key]) / 2
            self.assertTrue(torch.isfinite(updated[key]).all())
        self.assertEqual(updated['num_batches_tracked'].dtype, torch.int64)

    def test_zero_feedback_stays_zero_and_experiments_are_isolated(self):
        _, first = POISONEDFL.poisonedfl_attack(
            [_update(1)], _args(), 1, global_model=_global(),
            trainable_keys=self.keys, round_idx=0)
        result, first = POISONEDFL.poisonedfl_attack(
            [_update(1)], _args(), 1, global_model=_global(),
            trainable_keys=self.keys, round_idx=1, state=first)
        self.assertEqual(torch.count_nonzero(result[0]['weight']).item(), 0)
        _, second = POISONEDFL.poisonedfl_attack(
            [_update(1)], _args(scale=3), 1, global_model=_global(),
            trainable_keys=self.keys, round_idx=0)
        self.assertEqual(second['scale_factor'], 3)
        self.assertEqual(first['scale_factor'], 8)
        self.assertNotEqual(first['fixed_sign'].data_ptr(), second['fixed_sign'].data_ptr())

    def test_feedback_checkpoint_is_post_aggregation_and_floor_is_preserved(self):
        state = None
        for t in range(5):
            _, state = POISONEDFL.poisonedfl_attack(
                [_update(1)], _args(scale=0.6, interval=2), 1,
                global_model=_global(t), trainable_keys=self.keys, round_idx=t, state=state)
            if t == 1:
                torch.testing.assert_close(state['feedback_checkpoint'], torch.ones(6, dtype=torch.float64))
            if t == 3:
                torch.testing.assert_close(state['feedback_checkpoint'], torch.full((6,), 3., dtype=torch.float64))
        self.assertEqual(state['scale_factor'], 0.6)

    def test_shallow_rollback_snapshot_preserves_all_state_tensors(self):
        _, state = POISONEDFL.poisonedfl_attack(
            [_update(1)], _args(), 1, global_model=_global(),
            trainable_keys=self.keys, round_idx=0)
        snapshot, expected = copy.copy(state), copy.deepcopy(state)
        POISONEDFL.poisonedfl_attack(
            [_update(1)], _args(), 1, global_model=_global(2),
            trainable_keys=self.keys, round_idx=1, state=state)
        for key in expected:
            if isinstance(expected[key], torch.Tensor):
                torch.testing.assert_close(snapshot[key], expected[key])
            else:
                self.assertEqual(snapshot[key], expected[key])

    def test_actual_batchnorm_model_buffers_and_frozen_parameters(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
        model[0].bias.requires_grad_(False)
        model_state = model.state_dict()
        keys = tuple(k for k, p in model.named_parameters() if p.requires_grad)
        updates = [{k: torch.ones_like(v) for k, v in model_state.items()} for _ in range(3)]
        result, _ = POISONEDFL.poisonedfl_attack(
            updates, _args(), 2, global_model=model_state, trainable_keys=keys, round_idx=0)
        for key, tensor in result[0].items():
            self.assertEqual(tensor.shape, model_state[key].shape)
            self.assertEqual(tensor.dtype, model_state[key].dtype)
            self.assertEqual(tensor.device, model_state[key].device)
            torch.testing.assert_close(tensor, torch.zeros_like(tensor) if key in keys else torch.ones_like(tensor))


if __name__ == "__main__":
    unittest.main()
