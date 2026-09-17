"""PyTorch adaptation of the official PoisonedFL attack.

Official source:
https://github.com/xyq7/PoisonedFL/blob/266488e2cbe5953aab61712f315518546f457e55/byzantine.py

The paper implementation injects additional fake clients into an MXNet
training loop.  This repository instead uses one common compromised-client
protocol for every attack.  Consequently this adaptation replaces exactly the
leading ``malicious_attackers_this_round`` update slots; it never adds clients.

Other deliberate adaptation details:

* Cross-round state is supplied and returned explicitly.  No module global is
  used, so repeated experiments cannot inherit an attack direction or scale.
* Only floating-point entries named by ``trainable_keys`` are poisoned.
  BatchNorm counters and all other buffers retain their locally trained values.
* The official code hard-codes binomial 99% thresholds for four model sizes.
  Here the same Binomial(d, 0.5) threshold is approximated for arbitrary large
  d with its normal quantile, which is necessary for this repository's models.
* The first FL round is the official warm-up: controlled trainable updates are
  zero.  Global-model feedback drives the attack starting in the next round.
* Model/local training settings follow this framework (ResNet18/CNNFmnist,
  100 users, 25 selected, tau=3), not the official 1,200+240 fake-client setup.
* Zero-participant rounds still observe global feedback; before any malicious
  participation last_malicious is zero (official last_grad=None is undefined).
* Existing engine finite-audit rollbacks restore this state with the model;
  no official NaN/Inf replacement or norm-defense mutation is imported.
"""

import copy
import math
from statistics import NormalDist

import torch


_OFFICIAL_SCALE_DECAY = 0.7
_OFFICIAL_MIN_SCALE = 0.5
_OFFICIAL_ALIGNMENT_QUANTILE = 0.99


def _flatten_named(state, keys):
    return torch.cat([state[key].detach().reshape(-1) for key in keys])


def _replace_trainable(update, keys, vector):
    """Clone an update and replace only selected floating trainable entries."""
    result = copy.deepcopy(update)
    offset = 0
    for key in keys:
        target = result[key]
        count = target.numel()
        result[key] = vector[offset:offset + count].reshape_as(target).to(
            device=target.device, dtype=target.dtype)
        offset += count
    if offset != vector.numel():
        raise ValueError("PoisonedFL vector length does not match trainable parameters")
    return result


def _alignment_threshold_99(dimension):
    """Large-d normal approximation to official Binomial(d, 0.5) k_99."""
    mean = dimension * 0.5
    std = math.sqrt(dimension * 0.25)
    z_value = NormalDist().inv_cdf(_OFFICIAL_ALIGNMENT_QUANTILE)
    return math.ceil(mean + z_value * std)


def _initial_state(global_vector, keys, args):
    random_sign = torch.sign(torch.randn_like(global_vector))
    # torch.sign can theoretically produce zero.  The official sign vector is
    # Rademacher-valued, so make that vanishingly rare case positive.
    random_sign[random_sign == 0] = 1
    scale_factor = float(getattr(args, "poisonedfl_scale_factor", 8.0))
    interval = int(getattr(args, "poisonedfl_feedback_interval", 50))
    if not math.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("poisonedfl_scale_factor must be positive")
    if interval <= 0:
        raise ValueError("poisonedfl_feedback_interval must be positive")
    state = {
        "trainable_keys": tuple(keys),
        "dimension": global_vector.numel(),
        "fixed_sign": random_sign,
        "scale_factor": scale_factor,
        "feedback_interval": interval,
        "previous_global": global_vector.clone(),
        "feedback_checkpoint": None,
        "checkpoint_after_round": False,
        "last_malicious": torch.zeros_like(global_vector),
        "last_round": -1,
    }
    print("[PoisonedFL] implementation=official_pytorch_adaptation "
          "protocol=compromised_clients added_fake_clients=0 "
          "trainable_float_parameters_only=1 warmup=zero_first_round "
          "binomial_k99=large_d_normal_approximation "
          "zero_attacker_feedback=observe last_grad_initialization=zero "
          "rollback=engine_checkpoint local_training=framework "
          "source_commit=266488e2cbe5953aab61712f315518546f457e55")
    print(f"[PoisonedFL] dimension={state['dimension']} "
          f"initial_scale_factor={scale_factor:.6g} "
          f"feedback_interval={interval} scale_decay={_OFFICIAL_SCALE_DECAY} "
          f"min_scale={_OFFICIAL_MIN_SCALE}")
    return state


def poisonedfl_attack(all_updates, args, malicious_attackers_this_round,
                      *, global_model, trainable_keys, round_idx, state=None):
    """Replace the current malicious slots using PoisonedFL.

    Args follow the repository attack hook, with explicit global-model context
    and explicit state added for PoisonedFL's multi-round feedback.  The return
    value is ``(updates, state)``; callers must retain state only within one
    invocation of ``fedavg_all``.
    """
    keys = tuple(trainable_keys)
    if not keys:
        raise ValueError("PoisonedFL requires at least one trainable parameter")
    for key in keys:
        if key not in global_model:
            raise KeyError(f"PoisonedFL trainable key missing from global model: {key}")
        if not global_model[key].is_floating_point():
            raise TypeError(f"PoisonedFL trainable parameter must be floating point: {key}")

    global_vector = _flatten_named(global_model, keys)
    if state is None:
        state = _initial_state(global_vector, keys, args)
    elif (state["trainable_keys"] != keys
          or state["dimension"] != global_vector.numel()):
        raise ValueError("PoisonedFL state/model mismatch; state must reset per experiment")
    if round_idx <= state["last_round"]:
        raise ValueError("PoisonedFL round indices must be strictly increasing")

    k = int(malicious_attackers_this_round)
    if k < 0 or k > len(all_updates):
        raise ValueError("invalid malicious attacker count for PoisonedFL")

    # The state initialized at round zero cannot yet observe an aggregated
    # update.  Official fake-client placeholders are zero in this warm-up.
    warmup = state["last_round"] < 0
    if warmup:
        malicious_vector = torch.zeros_like(global_vector)
        history_norm = 0.0
    else:
        if state["checkpoint_after_round"]:
            state["feedback_checkpoint"] = global_vector.clone()
            state["checkpoint_after_round"] = False
        if state["feedback_checkpoint"] is None:
            # Mirrors official e=0 bookkeeping: the first post-aggregation
            # model becomes the baseline for the first 50-round feedback test.
            state["feedback_checkpoint"] = global_vector.clone()

        history = global_vector - state["previous_global"]
        history_norm_tensor = torch.linalg.vector_norm(history)
        history_norm = float(history_norm_tensor.item())
        last_malicious = state["last_malicious"]
        last_norm = torch.linalg.vector_norm(last_malicious)
        residual_scale = torch.abs(
            history - last_malicious * history_norm_tensor / (last_norm + 1e-9))
        scale_norm = torch.linalg.vector_norm(residual_scale)
        # Match official byzantine.py exactly: a zero residual stays zero.
        # Do not introduce an extra random/uniform direction on cancellation.
        deviation = residual_scale * state["fixed_sign"] / (scale_norm + 1e-9)

        interval = state["feedback_interval"]
        if round_idx > 0 and round_idx % interval == 0:
            total_update = global_vector - state["feedback_checkpoint"]
            total_update = torch.where(total_update == 0, global_vector, total_update)
            aligned = int((torch.sign(total_update) == state["fixed_sign"]).sum().item())
            threshold = _alignment_threshold_99(state["dimension"])
            old_scale = state["scale_factor"]
            if aligned < threshold and old_scale * _OFFICIAL_SCALE_DECAY >= _OFFICIAL_MIN_SCALE:
                state["scale_factor"] = old_scale * _OFFICIAL_SCALE_DECAY
            state["checkpoint_after_round"] = True
            print(f"[PoisonedFL] round={round_idx} feedback=1 "
                  f"aligned_dimensions={aligned} k99={threshold} "
                  f"scale_before={old_scale:.6g} "
                  f"scale_after={state['scale_factor']:.6g}")

        malicious_vector = state["scale_factor"] * history_norm_tensor * deviation

    for index in range(k):
        all_updates[index] = _replace_trainable(
            all_updates[index], keys, malicious_vector)

    state["previous_global"] = global_vector.clone()
    # Official code updates ``last_grad`` only when at least one fake client
    # participates.  Keep the previous malicious vector in zero-attacker
    # rounds so random client sampling does not silently change the algorithm.
    if k:
        state["last_malicious"] = malicious_vector.clone()
    state["last_round"] = int(round_idx)
    print(f"[PoisonedFL] round={round_idx} warmup={int(warmup)} "
          f"active_malicious={k} history_norm={history_norm:.6g} "
          f"scale_factor={state['scale_factor']:.6g}")
    return all_updates, state
