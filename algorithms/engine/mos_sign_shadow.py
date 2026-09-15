"""LASA-aware, read-only Sign proxy diagnostics for MOS.

Nothing in this module is used unless ``--mos_sign_shadow_diagnostics=1``.
All candidates, LASA calls, models, updates, and RNG state are private copies.
"""

import copy
import csv
import datetime
import json
import math
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

from ..attack.lie import vector_to_net_dict
from ..attack.mos_constraints import RadialConstraint, SignConstraint
from ..defense.lasa import audit_candidate_under_lasa, lasa


PROXY_MODES = (
    'current', 'current_q975', 'current_q100', 'layer_q75',
    'layer_mean', 'global_sign', 'lasa_faithful',
)

COMPARISON_FIELDS = [
    'round', 'proxy_mode', 'candidate_type', 'alpha_feasible',
    'proxy_status', 'status_reason',
    'max_feasible_A', 'boundary_left', 'boundary_right', 'last_feasible_A',
    'first_infeasible_A', 'A', 'norm', 'alignment', 'radial_ratio',
    'sign_ratio', 'CV', 'lasa_sign_pass_fraction',
    'lasa_norm_pass_fraction', 'lasa_joint_pass_fraction',
    'lasa_mean_sign_mz', 'lasa_p90_sign_mz', 'lasa_max_sign_mz',
    'effect_loss_before', 'effect_loss_after', 'delta_loss_lasa',
    'effect_acc_before', 'effect_acc_after', 'delta_acc_lasa', 'failure',
]

COMMON_FIELDS = [
    'round', 'candidate_id', 'alpha', 'A', 'norm', 'alignment',
    'proxy_mode', 'proxy_status', 'status_reason', 'sign_score', 'sign_ratio',
    'radial_ratio', 'CV', 'lasa_sign_pass_fraction',
    'lasa_norm_pass_fraction', 'lasa_joint_pass_fraction',
    'lasa_mean_sign_mz', 'lasa_p90_sign_mz', 'lasa_max_sign_mz',
    'effect_loss_before', 'effect_loss_after', 'delta_loss_lasa',
    'effect_acc_before', 'effect_acc_after', 'delta_acc_lasa', 'failure',
]


def enabled_rounds(args):
    if not bool(getattr(args, 'mos_sign_shadow_diagnostics', 0)):
        return set()
    result = set()
    for token in str(getattr(args, 'mos_diag_rounds', '0,20,40,59')).split(','):
        token = token.strip()
        if token:
            value = int(token)
            if value >= 0:
                result.add(value)
    return result


def clone_updates(updates, device=None):
    return [{key: value.detach().clone().to(device=device) for key, value in update.items()}
            for update in updates]


def _benign_matrix_from_updates(updates, positions):
    template = updates[positions[0]]
    width = sum(value.numel() for value in template.values())
    first = next(iter(template.values()))
    matrix = torch.empty((len(positions), width), dtype=first.dtype,
                         device=first.device)
    for row, position in enumerate(positions):
        offset = 0
        for value in updates[position].values():
            end = offset + value.numel()
            matrix[row, offset:end].copy_(value.reshape(-1))
            offset = end
    return matrix


def _prepare_snapshot(snapshot, device):
    """Normalize both legacy v2 and compact v3 snapshots without data copies."""
    if 'client_updates' not in snapshot:
        snapshot['client_updates'] = snapshot['local_updates']
    snapshot['local_updates'] = snapshot['client_updates']
    malicious_count = int(snapshot['malicious_count'])
    benign_positions = snapshot.get(
        'benign_positions', list(range(malicious_count, len(snapshot['client_updates']))))
    snapshot['benign_positions'] = list(benign_positions)
    snapshot.setdefault('attacker_positions', list(range(malicious_count)))
    snapshot.setdefault('update_template', snapshot['client_updates'][0])
    if 'benign_grads' not in snapshot:
        snapshot['benign_grads'] = _benign_matrix_from_updates(
            snapshot['client_updates'], snapshot['benign_positions'])
    for key in ('benign_grads', 'benign_mean', 'g_attack', 'max_dev_threshold'):
        snapshot[key] = snapshot[key].to(device)
    if device.type != 'cpu':
        snapshot['client_updates'] = [
            {key: value.to(device) for key, value in update.items()}
            for update in snapshot['client_updates']
        ]
        snapshot['local_updates'] = snapshot['client_updates']
        snapshot['update_template'] = snapshot['client_updates'][0]
    return snapshot


def _state_cpu(state):
    return {key: value.detach().clone().cpu() for key, value in state.items()}


def _rng_state_cpu():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.random.get_rng_state().cpu(),
        'cuda': [value.cpu() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else None,
    }


def _capture_rng():
    return _rng_state_cpu()


def _restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def _append_csv(path, rows, fields=COMPARISON_FIELDS):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _vector_to_update(candidate, template):
    return vector_to_net_dict(candidate, copy.deepcopy(template))


def _sparsified_vector(vector, template, sparsity):
    """Mirror LASA's global top-k mask (only 2-D/4-D tensors participate)."""
    result = vector.clone()
    eligible = []
    offsets = []
    offset = 0
    for value in template.values():
        end = offset + value.numel()
        if value.dim() in (2, 4):
            eligible.append(result[offset:end].abs())
            offsets.append((offset, end))
        offset = end
    if not eligible or sparsity == 0.0:
        return result
    scores = torch.cat(eligible)
    keep = int(scores.numel() * (1 - sparsity))
    if keep <= 0:
        for start, end in offsets:
            result[start:end] = 0
        return result
    threshold = torch.topk(scores, keep, sorted=True).values[-1]
    for start, end in offsets:
        part = result[start:end]
        result[start:end] = part * (part.abs() > threshold)
    return result


class _GlobalSign:
    name = 'sign'
    weight = 0.5

    def __init__(self, quantile):
        self.quantile = quantile
        self.threshold = None

    @staticmethod
    def loss(population, benign_mean, context):
        reference = torch.sign(benign_mean)
        active = reference != 0
        if not active.any():
            return population.new_zeros(population.shape[0])
        signs = torch.sign(population[:, active])
        return (signs != reference[active].unsqueeze(0)).float().mean(dim=1)

    def fit(self, benign, benign_mean, context):
        self.threshold = torch.clamp(
            torch.quantile(self.loss(benign, benign_mean, context), self.quantile),
            min=1e-6)


class _LasaFaithful:
    name = 'sign'
    weight = 0.5
    strict_threshold = True

    def __init__(self, benign, benign_mean, template, layer_dims, args,
                 malicious_count=1):
        self.benign = benign
        self.benign_mean = benign_mean
        self.template = template
        self.layer_dims = layer_dims
        self.args = args
        self.malicious_count = max(1, int(malicious_count))
        self.threshold = benign_mean.new_tensor(float(args.lambda_s)).clamp_min(1e-6)

    def loss(self, population, benign_mean, context):
        values = []
        for candidate in population:
            combined = torch.cat([
                candidate.unsqueeze(0).repeat(self.malicious_count, 1),
                self.benign,
            ], dim=0)
            norms = torch.norm(combined, dim=1, keepdim=True)
            clip = torch.median(norms)
            combined = (combined / norms) * torch.clamp(norms, max=clip)
            sparse = torch.stack([
                _sparsified_vector(row, self.template, self.args.sparsity)
                for row in combined
            ])
            layer_mz = []
            for layer_name, start, end in self.layer_dims:
                # The production LASA loop explicitly skips these state
                # counters. Including them creates 0/0 sign statistics and was
                # the implementation bug behind the original alpha=0 result.
                if 'num_batches_tracked' in layer_name:
                    continue
                signs = torch.sign(sparse[:, start:end])
                denominator = signs.abs().sum(dim=1)
                statistic = 0.5 * (1 + signs.sum(dim=1) / denominator
                                   * (1 - self.args.sparsity))
                # np.median in the real LASA averages the middle pair for an
                # even client count; torch.quantile(q=.5) has the same rule.
                median = torch.quantile(statistic, 0.5)
                std = torch.std(statistic, correction=0)
                mz = torch.abs((statistic[0] - median) / std)
                layer_mz.append(mz)
            layer_mz = torch.stack(layer_mz)
            # Hard feasibility means every real LASA sign-layer check passes.
            # A non-finite MZ is rejected by LASA's ``mz < lambda_s`` test.
            layer_mz = torch.where(torch.isfinite(layer_mz), layer_mz,
                                   torch.full_like(layer_mz, float('inf')))
            values.append(layer_mz.max())
        return torch.stack(values)


def _make_constraints(snapshot, args, mode):
    radial = RadialConstraint(float(getattr(args, 'weight_radial', 1.0)),
                              float(getattr(args, 'radius_quantile', 0.95)))
    q = float(getattr(args, 'radius_quantile', 0.95))
    reduce = getattr(args, 'sign_layer_reduce', 'quantile')
    layer_q = float(getattr(args, 'sign_layer_quantile', 0.9))
    if mode == 'current_q975':
        q = 0.975
    elif mode == 'current_q100':
        q = 1.0
    elif mode == 'layer_q75':
        layer_q = 0.75
    elif mode == 'layer_mean':
        reduce = 'mean'
    if mode == 'global_sign':
        sign = _GlobalSign(q)
    elif mode == 'lasa_faithful':
        sign = _LasaFaithful(snapshot['benign_grads'], snapshot['benign_mean'],
                             snapshot['update_template'], snapshot['layer_dims'], args,
                             snapshot['malicious_count'])
    else:
        sign = SignConstraint(float(getattr(args, 'weight_sign', 0.5)), q,
                              reduce, layer_q)
    sign.weight = float(getattr(args, 'weight_sign', 0.5))
    context = {'layer_dims': snapshot['layer_dims']}
    radial.fit(snapshot['benign_grads'], snapshot['benign_mean'], context)
    if not isinstance(sign, _LasaFaithful):
        sign.fit(snapshot['benign_grads'], snapshot['benign_mean'], context)
    return radial, sign


def _ratios(candidate, snapshot, constraints):
    population = candidate.unsqueeze(0)
    radial, sign = constraints
    radial_ratio = (radial.loss(population, snapshot['benign_mean'], {}) /
                    (radial.threshold + 1e-12))[0]
    sign_ratio = (sign.loss(population, snapshot['benign_mean'],
                            {'layer_dims': snapshot['layer_dims']}) /
                  (sign.threshold + 1e-12))[0]
    cv = radial.weight * torch.relu(radial_ratio - 1) + \
        sign.weight * torch.relu(sign_ratio - 1)
    return float(radial_ratio), float(sign_ratio), float(cv)


def _boundary(snapshot, constraints, max_evaluations=None):
    mean, guidance = snapshot['benign_mean'], snapshot['g_attack']
    budget = float(snapshot['max_dev_threshold'])
    evaluations = 0

    def sample(alpha):
        nonlocal evaluations
        evaluations += 1
        candidate = mean + alpha * budget * guidance
        rr, sr, cv = _ratios(candidate, snapshot, constraints)
        feasible = cv <= 1e-6 and math.isfinite(cv)
        if getattr(constraints[1], 'strict_threshold', False):
            feasible = feasible and sr < 1.0
        return candidate, rr, sr, cv, feasible

    left = (0.0, *sample(0.0))
    if not left[-1]:
        return 0.0, 0.0, left, left
    if max_evaluations is not None:
        # LASA-faithful scoring runs full clipping/sparsification.  Use a
        # bounded endpoint+bisection search and never change its threshold.
        limit = max(2, int(max_evaluations))
        right = (1.0, *sample(1.0))
        if right[-1]:
            return 1.0, budget, right, None
        while evaluations < limit and right[0] - left[0] >= 1e-5:
            alpha = (left[0] + right[0]) / 2
            value = (alpha, *sample(alpha))
            if value[-1]:
                left = value
            else:
                right = value
        return left[0], left[0] * budget, left, right
    right = None
    for alpha in np.linspace(1 / 16, 1, 16):
        value = (float(alpha), *sample(float(alpha)))
        if value[-1]:
            left = value
        else:
            right = value
            break
    if right is None:
        return 1.0, budget, left, None
    for _ in range(20):
        if right[0] - left[0] < 1e-5:
            break
        alpha = (left[0] + right[0]) / 2
        value = (alpha, *sample(alpha))
        if value[-1]:
            left = value
        else:
            right = value
    return left[0], left[0] * budget, left, right


def _evaluate_model(model, state, images, labels, device):
    model = copy.deepcopy(model).to(device)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        logits = model(images.to(device))
        loss = torch.nn.functional.cross_entropy(logits, labels.to(device)).item()
        accuracy = 100.0 * (logits.argmax(dim=-1) == labels.to(device)).float().mean().item()
    return loss, accuracy


def _lasa_effect(candidate, snapshot, args):
    device = candidate.device
    updates = clone_updates(snapshot['local_updates'], device)
    candidate_update = _vector_to_update(candidate, updates[0])
    for index in range(snapshot['malicious_count']):
        updates[index] = copy.deepcopy(candidate_update)
    before = {key: value.detach().clone().to(device) for key, value in snapshot['global_model'].items()}
    shadow_args = copy.deepcopy(args)
    shadow_args.device = device
    shadow_args.num_selected_users = len(updates)
    after = lasa(updates, copy.deepcopy(before), shadow_args)
    images, labels = snapshot['diagnostic_images'], snapshot['diagnostic_labels']
    before_loss, before_acc = _evaluate_model(snapshot['model_template'], before,
                                               images, labels, device)
    after_loss, after_acc = _evaluate_model(snapshot['model_template'], after,
                                             images, labels, device)
    return {
        'effect_loss_before': before_loss, 'effect_loss_after': after_loss,
        'delta_loss_lasa': after_loss - before_loss,
        'effect_acc_before': before_acc, 'effect_acc_after': after_acc,
        'delta_acc_lasa': after_acc - before_acc,
    }


def _evaluate_candidate_under_lasa(candidate, snapshot, args):
    candidate_update = _vector_to_update(candidate, snapshot['local_updates'][0])
    audit_updates = clone_updates(snapshot['local_updates'], candidate.device)
    for index in range(snapshot['malicious_count']):
        audit_updates[index] = copy.deepcopy(candidate_update)
    audit_args = copy.deepcopy(args)
    audit_args.num_selected_users = len(audit_updates)
    audit = audit_candidate_under_lasa(candidate_update, audit_updates,
                                       snapshot['global_model'], audit_args, 0)
    layer_audit = audit.pop('lasa_layer_audit')
    audit.update(_lasa_effect(candidate, snapshot, args))
    return audit, layer_audit


class _LasaEvaluationCache:
    """Evaluate each ray alpha once and share the real LASA result."""

    def __init__(self, snapshot, args):
        self.snapshot = snapshot
        self.args = args
        self.values = {}
        self.calls = 0

    def evaluate(self, alpha, candidate):
        key = float(alpha).hex()
        if key not in self.values:
            self.calls += 1
            try:
                value = _evaluate_candidate_under_lasa(
                    candidate, self.snapshot, self.args)
                self.values[key] = (True, value)
            except Exception as exc:
                self.values[key] = (False, exc)
        succeeded, value = self.values[key]
        if not succeeded:
            raise value
        return value


def _candidate_row(round_index, mode, candidate_type, alpha, boundary_data,
                   candidate, snapshot, constraints, evaluation_cache):
    feasible_alpha, max_a, left, right = boundary_data
    rr, sr, cv = _ratios(candidate, snapshot, constraints)
    centered = candidate - snapshot['benign_mean']
    norm = torch.norm(centered).item()
    proxy_status = ('invalid_not_applicable_hard_feasibility'
                    if mode == 'lasa_faithful' and not left[-1] else 'valid')
    row = {
        'round': round_index, 'proxy_mode': mode, 'candidate_type': candidate_type,
        'proxy_status': proxy_status,
        'status_reason': ('benign_mean_alpha_0_fails_one_or_more_LASA_sign_layers'
                          if proxy_status != 'valid' else ''),
        'alpha_feasible': feasible_alpha, 'max_feasible_A': max_a,
        'boundary_left': left[0], 'boundary_right': '' if right is None else right[0],
        'last_feasible_A': left[0] * float(snapshot['max_dev_threshold']),
        'first_infeasible_A': '' if right is None else right[0] * float(snapshot['max_dev_threshold']),
        'A': torch.dot(centered, snapshot['g_attack']).item(), 'norm': norm,
        'alignment': torch.dot(centered, snapshot['g_attack']).item() / max(norm, 1e-12),
        'radial_ratio': rr, 'sign_ratio': sr, 'CV': cv, 'failure': '',
    }
    audit, layer_audit = evaluation_cache.evaluate(alpha, candidate)
    if proxy_status != 'valid':
        failed = [layer['layer'] for layer in layer_audit
                  if not layer['sign_pass']]
        row['status_reason'] = (
            f'benign_mean_fails_{len(failed)}_of_{len(layer_audit)}_real_sign_layers')
    row.update(audit)
    return row, layer_audit


def _common_alphas(boundaries, target_count=10):
    positive = [float(boundary[0]) for boundary in boundaries.values()
                if boundary[0] > 0 and math.isfinite(boundary[0])]
    # Alpha is the normalized formal attack budget, so retain the full [0, 1]
    # ray even when every proxy boundary is conservative.
    upper = max([1.0] + positive)
    # Boundaries are mandatory anchors. Fill the remaining target slots with
    # log spacing; if anchors exceed the target they are never discarded.
    anchors = {0.0, upper, *positive}
    target_count = max(1, int(target_count))
    fill_count = max(target_count - len(anchors), 0)
    if fill_count:
        anchors.update(np.geomspace(max(upper * 1e-4, 1e-8), upper, fill_count + 1)[:-1])
    return sorted(float(value) for value in anchors)


def _common_candidate_rows(snapshot, args, proxy_data, evaluation_cache):
    rows, audits = [], {}
    for candidate_id, alpha in enumerate(_common_alphas(
            {mode: data['boundary'] for mode, data in proxy_data.items()},
            getattr(args, 'mos_sign_shared_candidates', 10))):
        candidate = (snapshot['benign_mean'] + alpha * float(
            snapshot['max_dev_threshold']) * snapshot['g_attack'])
        centered = candidate - snapshot['benign_mean']
        norm = torch.norm(centered).item()
        try:
            evaluation, layer_audit = evaluation_cache.evaluate(alpha, candidate)
            audits[f'common:{candidate_id}'] = layer_audit
            failure = ''
        except Exception as exc:
            evaluation, failure = {}, repr(exc)
        for mode, data in proxy_data.items():
            if data['status'] != 'valid':
                rr = sr = cv = float('nan')
                proxy_failure = failure
            else:
                try:
                    rr, sr, cv = _ratios(candidate, snapshot, data['constraints'])
                    proxy_failure = failure
                except Exception as exc:
                    rr = sr = cv = float('nan')
                    proxy_failure = repr(exc)
            row = {
                'round': snapshot['round'], 'candidate_id': candidate_id,
                'alpha': alpha, 'A': torch.dot(centered, snapshot['g_attack']).item(),
                'norm': norm,
                'alignment': torch.dot(centered, snapshot['g_attack']).item() /
                             max(norm, 1e-12),
                'proxy_mode': mode, 'proxy_status': data['status'],
                'status_reason': data.get('status_reason', ''),
                'sign_score': 1.0 / (1.0 + sr) if not math.isnan(sr) else float('nan'),
                'sign_ratio': sr, 'radial_ratio': rr, 'CV': cv,
                'failure': proxy_failure,
            }
            row.update(evaluation)
            rows.append(row)
    return rows, audits


def analyze_snapshot(snapshot, args=None, output_dir=None):
    """Recompute all proxy boundaries/audits/effects from one portable snapshot."""
    started = time.perf_counter()
    config = dict(snapshot['diagnostic_config'])
    if args is None:
        args = SimpleNamespace(**config)
    device = torch.device(getattr(args, 'device', 'cpu'))
    _prepare_snapshot(snapshot, device)
    rows, audits, proxy_data = [], {}, {}
    evaluation_cache = _LasaEvaluationCache(snapshot, args)
    shared_audits_before = 0
    caller_rng = _capture_rng()
    shared_rng = copy.deepcopy(snapshot.get('random_state', caller_rng))
    if device.type == 'cpu':
        shared_rng['cuda'] = None
    try:
        for mode in PROXY_MODES:
            _restore_rng(shared_rng)
            try:
                constraints = _make_constraints(snapshot, args, mode)
                boundary = _boundary(
                    snapshot, constraints,
                    getattr(args, 'mos_lasa_boundary_evals', 10)
                    if mode == 'lasa_faithful' else None)
                status = ('invalid_not_applicable_hard_feasibility'
                          if mode == 'lasa_faithful' and not boundary[2][-1]
                          else 'valid')
                proxy_data[mode] = {
                    'constraints': constraints, 'boundary': boundary,
                    'status': status,
                    'status_reason': (
                        'benign_mean_fails_real_lasa_sign_hard_feasibility'
                        if status != 'valid' else ''),
                }
                alpha = boundary[0]
                for candidate_type, candidate_alpha in (
                        ('boundary_candidate', alpha),
                        ('representative_candidate', 0.8 * alpha)):
                    candidate = snapshot['benign_mean'] + candidate_alpha * float(
                        snapshot['max_dev_threshold']) * snapshot['g_attack']
                    row, layer_audit = _candidate_row(
                        snapshot['round'], mode, candidate_type, candidate_alpha,
                        boundary, candidate, snapshot, constraints, evaluation_cache)
                    rows.append(row)
                    audits[f'{mode}:{candidate_type}'] = layer_audit
            except Exception as exc:
                rows.append({'round': snapshot['round'], 'proxy_mode': mode,
                             'candidate_type': 'proxy_failure',
                             'proxy_status': 'failed', 'failure': repr(exc)})
                proxy_data[mode] = {
                    'constraints': None, 'boundary': (0.0,), 'status': 'failed',
                    'status_reason': repr(exc),
                }
                print(f'[MOS-SignShadow] round={snapshot["round"]} proxy={mode} failure={exc!r}')
        _restore_rng(shared_rng)
        shared_audits_before = evaluation_cache.calls
        common_rows, common_audits = _common_candidate_rows(
            snapshot, args, proxy_data, evaluation_cache)
        audits.update(common_audits)
    finally:
        _restore_rng(caller_rng)
    if output_dir:
        _append_csv(os.path.join(output_dir, 'sign_proxy_comparison.csv'), rows)
        _append_csv(os.path.join(output_dir, 'sign_proxy_common_candidates.csv'),
                    common_rows, COMMON_FIELDS)
        torch.save(audits, os.path.join(output_dir,
                                       f'lasa_layer_audit_round_{snapshot["round"]}.pt'))
        write_derived_csvs(
            output_dir, snapshot.get('requested_diagnostic_rounds'),
            getattr(args, 'mos_sign_shared_candidates', 10))
        metrics = {
            'round': int(snapshot['round']),
            'analysis_seconds': time.perf_counter() - started,
            'shared_candidate_target': int(getattr(
                args, 'mos_sign_shared_candidates', 10)),
            'shared_candidate_count': len({row['candidate_id'] for row in common_rows}),
            'shared_candidate_lasa_audits_executed': (
                evaluation_cache.calls - shared_audits_before),
            'shared_candidate_lasa_results_reused_from_boundary': (
                len({row['candidate_id'] for row in common_rows}) -
                (evaluation_cache.calls - shared_audits_before)),
            'unique_candidate_lasa_audits': evaluation_cache.calls,
            'audit_candidate_under_lasa_calls': evaluation_cache.calls,
            'lasa_effect_aggregation_calls': evaluation_cache.calls,
        }
        with open(os.path.join(
                output_dir, f'analysis_metrics_round_{snapshot["round"]}.json'),
                'w', encoding='utf-8') as handle:
            json.dump(metrics, handle, indent=2)
        analyze_snapshot.last_metrics = metrics
    return rows


def write_derived_csvs(output_dir, required_rounds=None, shared_candidate_target=10):
    import pandas as pd

    path = os.path.join(output_dir, 'sign_proxy_comparison.csv')
    frame = pd.read_csv(path)
    valid = frame[(frame['failure'].fillna('') == '') &
                  (frame['proxy_status'].fillna('valid') == 'valid')].copy()
    summaries = []
    for (round_index, mode), group in valid.groupby(['round', 'proxy_mode']):
        summaries.append({
            'round': round_index, 'proxy_mode': mode,
            'alpha': group['alpha_feasible'].max(),
            'max_A': group['max_feasible_A'].max(),
            'best_lasa_survival': group['lasa_joint_pass_fraction'].max(),
            'best_lasa_delta_loss': group['delta_loss_lasa'].max(),
            'best_lasa_accuracy_harm': (-group['delta_acc_lasa']).max(),
        })
    pd.DataFrame(summaries).to_csv(
        os.path.join(output_dir, 'sign_proxy_summary.csv'), index=False)

    common_path = os.path.join(output_dir, 'sign_proxy_common_candidates.csv')
    common = pd.read_csv(common_path)
    correlation_rows = []
    targets = ['lasa_joint_pass_fraction', 'delta_loss_lasa', 'delta_acc_lasa']
    scopes = [('all', common)] + [(int(round_index), group)
                                  for round_index, group in common.groupby('round')]
    for scope, scoped in scopes:
        scoped_valid = scoped[(scoped['failure'].fillna('') == '') &
                              (scoped['proxy_status'] == 'valid')]
        bad_modes = set(scoped.loc[
            (scoped['failure'].fillna('') != '') |
            (scoped['proxy_status'] != 'valid'), 'proxy_mode'])
        for mode in PROXY_MODES:
            group = scoped_valid[scoped_valid['proxy_mode'] == mode]
            for target in targets:
                pair = group[['sign_score', target]].replace(
                    [np.inf, -np.inf], np.nan).dropna()
                n = len(pair)
                target_unique = pair[target].nunique()
                score_unique = pair['sign_score'].nunique()
                eligible = (mode not in bad_modes and n >= 8 and
                            target_unique > 1 and score_unique > 1)
                reason = ('ok' if eligible else 'proxy_failed_or_invalid'
                          if mode in bad_modes else 'n_lt_8' if n < 8 else
                          'constant_target' if target_unique <= 1 else 'constant_score')
                correlation_rows.append({
                    'round': scope, 'proxy_mode': mode,
                    'proxy_metric': 'sign_score', 'target': target,
                    'spearman': pair['sign_score'].corr(pair[target], method='spearman')
                    if eligible else np.nan,
                    'n': n, 'eligible': eligible, 'exclusion_reason': reason,
                })
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(os.path.join(
        output_dir, 'sign_proxy_common_correlations.csv'), index=False)
    # Preserve the legacy filename while fixing its statistical source.
    aggregate_correlations = correlations[correlations['round'] == 'all'].drop(
        columns='round')
    aggregate_correlations.to_csv(os.path.join(
        output_dir, 'sign_proxy_correlations.csv'), index=False)

    if summaries:
        summary = pd.DataFrame(summaries)
        space = summary.groupby('proxy_mode')['alpha'].mean().sort_values(ascending=False)
        survival = summary.groupby('proxy_mode')['best_lasa_survival'].mean().sort_values(ascending=False)
        loss = summary.groupby('proxy_mode')['best_lasa_delta_loss'].mean().sort_values(ascending=False)
        harm = summary.groupby('proxy_mode')['best_lasa_accuracy_harm'].mean().sort_values(ascending=False)
        corr = aggregate_correlations[aggregate_correlations['eligible']].copy()
        pred = corr[corr['target'] == 'lasa_joint_pass_fraction'].copy()
        pred['strength'] = pred['spearman'].abs()
        pred = pred.sort_values(['strength', 'n'], ascending=False)
        current_alpha = space.get('current', np.nan)
        current_survival = survival.get('current', np.nan)
        global_alpha = space.get('global_sign', np.nan)
        global_survival = survival.get('global_sign', np.nan)
        faithful_better = (not pred.empty and pred.iloc[0]['proxy_mode'] == 'lasa_faithful')
        candidates = summary.groupby('proxy_mode').agg(
            alpha=('alpha', 'mean'), survival=('best_lasa_survival', 'mean'),
            loss=('best_lasa_delta_loss', 'mean'), harm=('best_lasa_accuracy_harm', 'mean'))
        valid_modes = set(corr.groupby('proxy_mode').filter(
            lambda group: set(group['target']) == set(targets))['proxy_mode'])
        candidates = candidates.loc[candidates.index.intersection(valid_modes)].dropna()
        candidates['balanced_score'] = (
            candidates['alpha'].rank(pct=True) + candidates['survival'].rank(pct=True) +
            candidates['loss'].rank(pct=True) + candidates['harm'].rank(pct=True))
        if required_rounds is None:
            required_rounds = enabled_rounds(SimpleNamespace(
                mos_sign_shadow_diagnostics=1, mos_diag_rounds='0,20,40,59'))
        else:
            required_rounds = set(int(value) for value in required_rounds)
        completed_rounds = set()
        for round_index, group in common.groupby('round'):
            good = group.replace([np.inf, -np.inf], np.nan).dropna(
                subset=['candidate_id', 'lasa_joint_pass_fraction'])
            if good['candidate_id'].nunique() >= max(8, int(shared_candidate_target)):
                completed_rounds.add(int(round_index))
        complete = required_rounds <= completed_rounds
        status = 'complete' if complete else 'provisional/incomplete'
        recommendation = (candidates['balanced_score'].idxmax()
                          if complete and not candidates.empty else None)
        lines = [
            f'1. largest_feasible_space={space.index[0]} mean_alpha={space.iloc[0]:.6g}',
            ('2. best_layer_survival_predictor=insufficient_samples' if pred.empty else
             f'2. best_layer_survival_predictor={pred.iloc[0]["proxy_mode"]} '
             f'spearman={pred.iloc[0]["spearman"]:.6g} n={int(pred.iloc[0]["n"])}'),
            f'3. largest_real_lasa_effect_loss={loss.index[0]} mean_delta_loss={loss.iloc[0]:.6g}; '
            f'largest_accuracy_harm={harm.index[0]} mean_harm={harm.iloc[0]:.6g}',
            f'4. current_over_conservative={bool(np.isfinite(current_alpha) and current_alpha < space.median())} '
            f'alpha={current_alpha:.6g} survival={current_survival:.6g}',
            f'5. global_sign_too_loose={bool(np.isfinite(global_alpha) and np.isfinite(global_survival) and global_alpha > current_alpha and global_survival < current_survival)} '
            f'alpha={global_alpha:.6g} survival={global_survival:.6g}',
            f'6. lasa_faithful_closer_than_current={faithful_better}',
            f'7. recommendation_status={status} required_rounds={sorted(required_rounds)} '
            f'completed_rounds={sorted(completed_rounds)}',
            ('8. recommended_next_60_round_proxy=insufficient_valid_correlations'
             if complete and recommendation is None else
             '8. recommended_next_60_round_proxy=withheld_until_all_rounds_complete'
             if not complete else
             f'8. recommended_next_60_round_proxy={recommendation} '
             '(balanced rank; proxies with failed/NaN correlations excluded)'),
        ]
        with open(os.path.join(output_dir, 'sign_proxy_recommendation.txt'),
                  'w', encoding='utf-8') as handle:
            handle.write('\n'.join(lines) + '\n')


def _diagnostic_batch(dataset_val, dataset_test, count):
    source = dataset_val if len(dataset_val) else dataset_test
    items = [source[index] for index in range(min(len(source), count))]
    return (torch.stack([item[0] for item in items]),
            torch.stack([torch.as_tensor(item[1]) for item in items]).long())


def run_sign_shadow_diagnostics(round_index, mos_snapshot, original_updates,
                                malicious_count, selected_client_ids,
                                malicious_client_ids, global_model, net_glob,
                                dataset_val, dataset_test, g_ce, g_cw, args):
    """Save one portable snapshot; all LASA shadow computation is offline."""
    saved_rng = _capture_rng()
    try:
        images, labels = _diagnostic_batch(
            dataset_val, dataset_test,
            max(1, int(getattr(args, 'mos_diag_samples', 256))))
        output_dir = os.path.abspath(getattr(args, 'mos_diagnostics_dir', '.'))
        os.makedirs(output_dir, exist_ok=True)
        defaults = {
            'sparsity': 0.3, 'lambda_n': 1.0, 'lambda_s': 1.0,
            'radius_quantile': 0.95, 'weight_radial': 1.0,
            'weight_sign': 0.5, 'sign_layer_reduce': 'quantile',
            'sign_layer_quantile': 0.9,
            'test_batch_size': 128, 'dataset': 'cifar', 'model': 'resnet18',
            'mos_diag_rounds': '0,20,40,59',
            'mos_constraint_mode': 'strict', 'mos_objective_mode': 'dual',
            'mos_adaptive_guided_init': 1,
            'mos_sign_shared_candidates': 10, 'mos_lasa_boundary_evals': 10,
        }
        config = {name: getattr(args, name, default)
                  for name, default in defaults.items()}
        config['device'] = 'cpu'
        client_updates = original_updates
        if any(value.device.type != 'cpu' for update in client_updates
               for value in update.values()):
            client_updates = clone_updates(client_updates, 'cpu')
        malicious_client_set = set(malicious_client_ids)
        canonical_client_ids = (
            [value for value in selected_client_ids if value in malicious_client_set] +
            [value for value in selected_client_ids if value not in malicious_client_set])
        snapshot = {
            'snapshot_version': 3, 'round': round_index,
            'requested_diagnostic_rounds': sorted(enabled_rounds(args)),
            'global_model': _state_cpu(global_model),
            'model_template': copy.deepcopy(net_glob).cpu(),
            'client_updates': client_updates,
            'malicious_count': malicious_count,
            'attacker_positions': list(range(malicious_count)),
            'benign_positions': list(range(malicious_count, len(client_updates))),
            'selected_client_ids': list(selected_client_ids),
            'malicious_client_ids': list(malicious_client_ids),
            'canonical_client_ids': canonical_client_ids,
            'benign_mean': mos_snapshot['benign_mean'].detach().cpu(),
            'g_attack': mos_snapshot['g_attack'].detach().cpu(),
            'g_ce': g_ce.detach().cpu(), 'g_cw': g_cw.detach().cpu(),
            'layer_dims': list(mos_snapshot['layer_dims']),
            'max_dev_threshold': mos_snapshot['max_dev_threshold'].detach().cpu(),
            'attack_budget': float(mos_snapshot['max_dev_threshold']),
            'radial_threshold': float(mos_snapshot.get('radial_threshold', float('nan'))),
            'sign_threshold': float(mos_snapshot.get('sign_threshold', float('nan'))),
            'constraint_thresholds': {
                'radial': float(mos_snapshot.get('radial_threshold', float('nan'))),
                'sign': float(mos_snapshot.get('sign_threshold', float('nan'))),
            },
            'sign_config': {
                'benign_quantile': float(getattr(args, 'radius_quantile', 0.95)),
                'layer_reduce': getattr(args, 'sign_layer_reduce', 'quantile'),
                'layer_quantile': float(getattr(args, 'sign_layer_quantile', 0.9)),
            },
            'lasa_config': {name: getattr(args, name) for name in
                            ('sparsity', 'lambda_n', 'lambda_s')},
            'diagnostic_config': config,
            'diagnostic_images': images.cpu(), 'diagnostic_labels': labels.cpu(),
            'random_state': saved_rng,
        }
        snapshot_path = os.path.join(
            output_dir, f'diagnostic_snapshot_round_{round_index}.pt')
        temporary_path = snapshot_path + '.tmp'
        torch.save(snapshot, temporary_path)
        os.replace(temporary_path, snapshot_path)
        timestamp = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
        print(f'[SIGN-SHADOW] round={round_index} snapshot saved '
              f'timestamp={timestamp} path={snapshot_path}', flush=True)
    finally:
        _restore_rng(saved_rng)


def load_snapshot(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')
