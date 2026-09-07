"""Opt-in, read-only diagnostics for MOS search rounds."""

import copy
import csv
import os
import random

import numpy as np
import torch
from torch.utils.data import Subset

from ..attack.lie import vector_to_net_dict
from ..attack.mos_constraints import RadialConstraint, SignConstraint, compute_dual_objectives
from ..defense.byzantine_robust_aggregation import multi_krum
from ..solver.global_aggregator import average
from test import test_img


def diagnostic_rounds(args):
    if not bool(getattr(args, 'mos_diagnostics', 0)):
        return set()
    result = set()
    for token in str(getattr(args, 'mos_diag_rounds', '0,20,40,59')).split(','):
        token = token.strip()
        if token:
            value = int(token)
            if value >= 0:
                result.add(value)
    return result


def _append_csv(path, fieldnames, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _rng_state():
    return {
        'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def clone_updates(updates):
    return [{key: value.detach().clone() for key, value in update.items()} for update in updates]


def _candidate_metrics(candidate, snapshot, constraints=None):
    constraints = constraints or snapshot['constraints']
    population = candidate.unsqueeze(0)
    objectives, scores, _, cv, ratios = compute_dual_objectives(
        population, snapshot['benign_mean'], constraints, snapshot['g_attack'],
        {'layer_dims': snapshot['layer_dims']})
    centered = candidate - snapshot['benign_mean']
    norm = torch.norm(centered).item()
    return {
        'A': -objectives[1, 0].item(), 'norm': norm,
        'alignment': torch.dot(centered, snapshot['g_attack']).item() / max(norm, 1e-12),
        'R': scores[0].item(), 'CV': cv[0].item(),
        'radial_ratio': ratios['radial'][0].item() if 'radial' in ratios else '',
        'sign_ratio': ratios['sign'][0].item() if 'sign' in ratios else '',
        'feasible': cv[0].item() <= 1e-6,
    }


def _effect(candidate, original_updates, malicious_count, global_model, net_glob,
            diagnostic_dataset, args):
    updates = clone_updates(original_updates)
    for index in range(malicious_count):
        updates[index] = vector_to_net_dict(candidate, copy.deepcopy(updates[index]))
    aggregate, _ = multi_krum(updates, multi_k=True)
    before_state = {key: value.detach().clone() for key, value in global_model.items()}
    after_state = average(copy.deepcopy(before_state), [aggregate])
    before_model, after_model = copy.deepcopy(net_glob), copy.deepcopy(net_glob)
    before_model.load_state_dict(before_state)
    after_model.load_state_dict(after_state)
    with torch.no_grad():
        before_acc, before_loss = test_img(before_model, diagnostic_dataset, args)
        after_acc, after_loss = test_img(after_model, diagnostic_dataset, args)
    return before_loss, after_loss, before_acc, after_acc


def _constraints(snapshot, args, names):
    available = {
        'radial': RadialConstraint(getattr(args, 'weight_radial', 1.0),
                                   getattr(args, 'radius_quantile', 0.95)),
        'sign': SignConstraint(getattr(args, 'weight_sign', 0.5),
                               getattr(args, 'radius_quantile', 0.95),
                               getattr(args, 'sign_layer_reduce', 'quantile'),
                               getattr(args, 'sign_layer_quantile', 0.9)),
    }
    result = [available[name] for name in names]
    context = {'layer_dims': snapshot['layer_dims']}
    for constraint in result:
        constraint.fit(snapshot['benign_grads'], snapshot['benign_mean'], context)
    return result


def _proxy_rows(round_index, snapshot, original_updates, malicious_count,
                global_model, net_glob, dataset, args):
    population = snapshot['population']
    metrics = [_candidate_metrics(candidate, snapshot) for candidate in population]
    feasible = sorted((i for i, value in enumerate(metrics) if value['feasible']),
                      key=lambda i: metrics[i]['A'])
    limit = max(1, int(getattr(args, 'mos_diag_candidates', 16)))
    if len(feasible) <= limit:
        chosen = feasible
    else:
        positions = torch.linspace(0, len(feasible) - 1, limit).round().long().tolist()
        chosen = [feasible[position] for position in positions]
    best_idx = snapshot['best_idx']
    if best_idx not in chosen:
        if len(chosen) >= limit:
            chosen[-1] = best_idx
        else:
            chosen.append(best_idx)
    rows = []
    for candidate_id in dict.fromkeys(chosen):
        value = metrics[candidate_id]
        before_loss, after_loss, before_acc, after_acc = _effect(
            population[candidate_id], original_updates, malicious_count,
            global_model, net_glob, dataset, args)
        rows.append(dict(round=round_index, candidate_id=candidate_id,
                         selected=candidate_id == best_idx,
                         effect_loss_before=before_loss, effect_loss_after=after_loss,
                         delta_loss=after_loss - before_loss,
                         effect_acc_before=before_acc, effect_acc_after=after_acc,
                         delta_acc=after_acc - before_acc, **value))
    return rows


def _sign_rows(round_index, snapshot, args, mos_module):
    rows = []
    context = {'layer_dims': snapshot['layer_dims']}
    for label, names in [('radial_only', ['radial']), ('sign_only', ['sign']),
                         ('radial+sign', ['radial', 'sign'])]:
        constraints = _constraints(snapshot, args, names)
        alpha = mos_module._estimate_feasible_alpha(
            snapshot['benign_mean'], snapshot['max_dev_threshold'],
            snapshot['g_attack'], constraints, context)
        right = min(1.0, alpha + 2e-5)
        left_candidate = snapshot['benign_mean'] + alpha * snapshot['max_dev_threshold'] * snapshot['g_attack']
        right_candidate = snapshot['benign_mean'] + right * snapshot['max_dev_threshold'] * snapshot['g_attack']
        left = _candidate_metrics(left_candidate, snapshot, constraints)
        right_metrics = _candidate_metrics(right_candidate, snapshot, constraints)
        ratio_map = {'radial': right_metrics['radial_ratio'], 'sign': right_metrics['sign_ratio']}
        has_infeasible_right = alpha < 1.0 and not right_metrics['feasible']
        limiting = max(names, key=lambda name: ratio_map[name]) if has_infeasible_right else 'none'
        rows.append(dict(round=round_index, constraint_set=label, alpha_feasible=alpha,
                         max_feasible_A=left['A'], radial_ratio=left['radial_ratio'],
                         sign_ratio=left['sign_ratio'], limiting_constraint=limiting,
                         boundary_left=alpha, boundary_right=right if has_infeasible_right else '',
                         last_feasible_A=left['A'],
                         first_infeasible_A=right_metrics['A'] if has_infeasible_right else ''))
    return rows


def _shadow_rows(round_index, snapshot, original_updates, malicious_count, historical_pop,
                 g_ce, g_cw, global_model, net_glob, dataset, args, mos_module):
    saved_rng = _rng_state()
    saved_cache = None if mos_module._LAST_VALID_GUIDANCE is None else mos_module._LAST_VALID_GUIDANCE.detach().clone()
    rows, initial_populations = [], []
    alpha_feasible = mos_module._estimate_feasible_alpha(
        snapshot['benign_mean'], snapshot['max_dev_threshold'], snapshot['g_attack'],
        snapshot['constraints'], {'layer_dims': snapshot['layer_dims']})
    try:
        for objective_mode in ('dual', 'a_only'):
            _restore_rng(saved_rng)
            mos_module._LAST_VALID_GUIDANCE = None if saved_cache is None else saved_cache.detach().clone()
            # MMEngine Config's shallow copy may share its internal ConfigDict.
            # A deep copy is required so objective/diagnostic flags cannot leak
            # from either shadow branch into the live training configuration.
            shadow_args = copy.deepcopy(args)
            shadow_args.mos_diagnostics = 0
            shadow_args._mos_diagnostics_active = True
            shadow_args.mos_objective_mode = objective_mode
            shadow_updates, _ = mos_module.mos_attack(
                clone_updates(original_updates), shadow_args, malicious_count,
                g_ce=g_ce.detach().clone(), g_cw=g_cw.detach().clone(),
                historical_pop=None if historical_pop is None else historical_pop.detach().clone())
            shadow = shadow_args._mos_diagnostic_snapshot
            initial_populations.append(shadow['initial_population'])
            candidate = torch.cat([value.flatten() for value in shadow_updates[0].values()])
            value = _candidate_metrics(candidate, snapshot)
            before_loss, after_loss, before_acc, after_acc = _effect(
                candidate, original_updates, malicious_count, global_model,
                net_glob, dataset, args)
            rows.append(dict(round=round_index, objective_mode=objective_mode,
                             selection_mode=shadow['selection_mode'],
                             alpha_feasible=alpha_feasible, delta_loss=after_loss - before_loss,
                             delta_acc=after_acc - before_acc, **value))
        equal = torch.equal(initial_populations[0], initial_populations[1])
        for row in rows:
            row['initial_population_equal'] = equal
        if not equal:
            raise AssertionError('MOS shadow searches did not receive identical initial populations')
    finally:
        _restore_rng(saved_rng)
        mos_module._LAST_VALID_GUIDANCE = saved_cache
    return rows


def run_first_batch_diagnostics(round_index, snapshot, original_updates, malicious_count,
                                historical_pop, g_ce, g_cw, global_model, net_glob,
                                dataset_val, dataset_test, args, mos_module):
    """Run all diagnostics against copies and append the three requested CSVs."""
    saved_rng = _rng_state()
    saved_cache = None if mos_module._LAST_VALID_GUIDANCE is None else mos_module._LAST_VALID_GUIDANCE.detach().clone()
    try:
        snapshot = copy.deepcopy(snapshot)
        snapshot['constraints'] = _constraints(snapshot, args, ['radial', 'sign'])
        source = dataset_val if len(dataset_val) else dataset_test
        count = min(len(source), max(1, int(getattr(args, 'mos_diag_samples', 256))))
        diagnostic_dataset = Subset(source, list(range(count)))
        output_dir = getattr(args, 'mos_diagnostics_dir', '.')
        proxy = _proxy_rows(round_index, snapshot, original_updates, malicious_count,
                            global_model, net_glob, diagnostic_dataset, args)
        signs = _sign_rows(round_index, snapshot, args, mos_module)
        shadows = _shadow_rows(round_index, snapshot, original_updates, malicious_count,
                               historical_pop, g_ce, g_cw, global_model, net_glob,
                               diagnostic_dataset, args, mos_module)
        _append_csv(os.path.join(output_dir, 'proxy_candidates.csv'), list(proxy[0]), proxy)
        _append_csv(os.path.join(output_dir, 'sign_bottleneck.csv'), list(signs[0]), signs)
        _append_csv(os.path.join(output_dir, 'same_checkpoint_objectives.csv'), list(shadows[0]), shadows)
        print(f'[MOS-Diagnostics] round={round_index} proxy={len(proxy)} sign={len(signs)} shadow={len(shadows)}')
    finally:
        _restore_rng(saved_rng)
        mos_module._LAST_VALID_GUIDANCE = saved_cache
