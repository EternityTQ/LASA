"""
MOS-Attack (Multi-Objective Stealth Attack) - Simplified Paper Version
Dual-objective NSGA-II with Radial + Sign constraints. See mos_experimental.py for experimental features.
"""

import torch
import copy
from .lie import vector_to_net_dict
from typing import Dict, List, Tuple, Optional

# Import split modules and re-export for backward compatibility
from .mos_constraints import (
    ConstraintPlugin, RadialConstraint, SignConstraint,
    project_to_attack_budget, compute_constraint_pass_score,
    compute_dual_objectives, LayerDims
)
from .mos_nsga2 import (
    nondominated_sort, crowding_distance, compute_rank_and_crowding,
    binary_tournament_selection, nsga2_select, sbx_crossover,
    mutation, select_final_solution
)

# ============================================================================
# Type Annotations
# ============================================================================
TensorDict = Dict[str, torch.Tensor]

# Lightweight stability state shared across consecutive federated rounds.
# Tensors are always detached; incompatible model dimensions reset the state.
_LAST_VALID_GUIDANCE: Optional[torch.Tensor] = None
_STABILITY_STATE_NUMEL: Optional[int] = None

def safe_normalize(
    vector: Optional[torch.Tensor],
    eps: float = 1e-12,
    min_norm: float = 1e-8,
    expected_numel: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], float, bool]:
    """Return a unit vector only when shape, values, and norm are valid."""
    if vector is None:
        return None, 0.0, False
    flat = vector.detach().reshape(-1)
    if expected_numel is not None and flat.numel() != expected_numel:
        return None, 0.0, False
    if not torch.isfinite(flat).all():
        return None, float("nan"), False
    norm = torch.norm(flat).item()
    if not torch.isfinite(torch.tensor(norm)) or norm < min_norm:
        return None, norm, False
    # Division is performed only after the explicit lower-bound check.
    return flat / max(norm, eps), norm, True


def _estimate_feasible_alpha(benign_mean, max_dev_threshold, g_attack, constraints, context, batch_size=2):
    """Return the last CV-feasible point in the ray's feasible prefix."""
    def evaluate(alphas):
        results = []
        for start in range(0, len(alphas), batch_size):
            chunk = alphas[start:start + batch_size]
            candidates = torch.empty((len(chunk), benign_mean.numel()),
                                     device=benign_mean.device, dtype=benign_mean.dtype)
            for row, alpha in enumerate(chunk):
                candidates[row] = benign_mean + alpha * max_dev_threshold * g_attack
            _, _, _, total_cv, _ = compute_dual_objectives(candidates, benign_mean, constraints,
                                                           g_attack, context)
            for alpha, cv in zip(chunk, total_cv):
                value = cv.item()
                results.append((alpha, value, torch.isfinite(cv).item() and value <= 1e-6))
        return results

    samples = evaluate([i / 16.0 for i in range(17)])
    if not samples[0][2]:
        return 0.0
    left, right = samples[0][0], None
    for alpha, _, feasible in samples[1:]:
        if not feasible:
            right = alpha
            break
        left = alpha
    if right is None:
        return 1.0
    for _ in range(2):
        samples = evaluate([left + (right - left) * i / 8.0 for i in range(9)])
        for alpha, _, feasible in samples[1:]:
            if not feasible:
                right = alpha
                break
            left = alpha
    return left

# ============================================================================
# Attack Guidance Construction
# ============================================================================
def _construct_attack_guidance(
    g_ce: Optional[torch.Tensor],
    g_cw: Optional[torch.Tensor],
    historical_pop: Optional[torch.Tensor],
    benign_grads: torch.Tensor,
    benign_mean: torch.Tensor,
    total_params: int,
    lam: float,
    min_norm: float,
) -> Tuple[Optional[torch.Tensor], Dict]:
    """Choose current, cached, historical, or variance guidance in that order."""
    global _LAST_VALID_GUIDANCE

    device = benign_mean.device
    ce_unit, ce_norm, ce_valid = safe_normalize(g_ce, min_norm=min_norm, expected_numel=total_params)
    cw_unit, cw_norm, cw_valid = safe_normalize(g_cw, min_norm=min_norm, expected_numel=total_params)
    if ce_valid:
        ce_unit = ce_unit.to(device)
    if cw_valid:
        cw_unit = cw_unit.to(device)

    hist_unit, hist_norm, hist_valid = safe_normalize(
        historical_pop, min_norm=min_norm, expected_numel=total_params
    )
    if hist_valid:
        hist_unit = hist_unit.to(device)

    cached_unit, _, cached_valid = safe_normalize(
        _LAST_VALID_GUIDANCE, min_norm=min_norm, expected_numel=total_params
    )
    if cached_valid:
        cached_unit = cached_unit.to(device)
    elif _LAST_VALID_GUIDANCE is not None:
        _LAST_VALID_GUIDANCE = None

    ce_cw_cosine = float("nan")
    combined_pre_norm = 0.0
    source = "none"
    fallback_used = False
    guidance = None

    if ce_valid and cw_valid:
        ce_cw_cosine = torch.dot(ce_unit, cw_unit).item()
        combined = lam * ce_unit + (1.0 - lam) * cw_unit
        guidance, combined_pre_norm, combined_valid = safe_normalize(
            combined, min_norm=min_norm, expected_numel=total_params
        )
        if combined_valid:
            source = "ce_cw_combined"
        else:
            # Cancellation: prefer agreement with history, then the larger raw gradient.
            if hist_valid:
                ce_hist = torch.dot(ce_unit, hist_unit).item()
                cw_hist = torch.dot(cw_unit, hist_unit).item()
                guidance, source = ((ce_unit, "ce_conflict_history")
                                    if ce_hist >= cw_hist else (cw_unit, "cw_conflict_history"))
            else:
                guidance, source = ((ce_unit, "ce_conflict_raw_norm")
                                    if ce_norm >= cw_norm else (cw_unit, "cw_conflict_raw_norm"))
            fallback_used = True
    elif ce_valid:
        guidance, source = ce_unit, "ce"
    elif cw_valid:
        guidance, source = cw_unit, "cw"
    elif cached_valid:
        guidance, source, fallback_used = cached_unit, "cached_last_valid", True
    elif hist_valid:
        guidance, source, fallback_used = hist_unit, "historical_perturbation", True
    else:
        deviations = benign_grads - benign_mean
        dev_norms = torch.norm(deviations, dim=1)
        if dev_norms.numel():
            idx = torch.argmax(dev_norms)
            guidance, _, variance_valid = safe_normalize(
                deviations[idx], min_norm=min_norm, expected_numel=total_params
            )
            if variance_valid:
                source, fallback_used = "benign_variance_fallback", True

    guidance_history_cosine = float("nan")
    if guidance is not None and hist_valid:
        guidance_history_cosine = torch.dot(guidance.to(device), hist_unit).item()
    if guidance is not None:
        guidance = guidance.to(device).detach()
        # Only genuine CE/CW attack guidance is promoted to the cross-round
        # cache.  In particular, do not turn the benign-variance search-space
        # fallback into a persistent (and misleading) attack direction.
        if source in {
            "ce_cw_combined",
            "ce_conflict_history",
            "cw_conflict_history",
            "ce_conflict_raw_norm",
            "cw_conflict_raw_norm",
            "ce",
            "cw",
        }:
            _LAST_VALID_GUIDANCE = guidance.clone().detach()

    diagnostics = {
        "ce_raw_norm": ce_norm, "cw_raw_norm": cw_norm,
        "ce_valid": ce_valid, "cw_valid": cw_valid,
        "ce_cw_cosine": ce_cw_cosine, "combined_pre_norm": combined_pre_norm,
        "historical_direction_valid": hist_valid,
        "historical_direction_norm": hist_norm,
        "guidance_history_cosine": guidance_history_cosine,
        "guidance_source": source, "guidance_fallback_used": fallback_used,
    }
    return guidance, diagnostics

# ============================================================================
# Surrogate Guidance Construction
# ============================================================================
def compute_surrogate_guidance(
    global_model: torch.nn.Module,
    poison_images: torch.Tensor,
    target_labels: torch.Tensor,
    criterion_ce,
    args=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute CE and CW surrogate gradients

    Returns:
        g_ce: Cross-entropy guidance gradient
        g_cw: CW margin loss guidance gradient
    """
    global_model.eval()
    device = poison_images.device

    def extract_gradient_vector(model):
        """Extract flattened gradient vector from model state_dict"""
        g_list = []
        named_params = dict(model.named_parameters())

        for name, tensor in model.state_dict().items():
            if name in named_params:
                param = named_params[name]
                if param.grad is not None:
                    g_list.append(param.grad.clone().flatten())
                else:
                    g_list.append(torch.zeros(tensor.numel(), device=device))
            else:
                # Non-trainable buffer
                g_list.append(torch.zeros(tensor.numel(), device=device))

        return torch.cat(g_list) if g_list else torch.zeros(0, device=device)

    # ===== CE Loss =====
    global_model.zero_grad(set_to_none=True)
    outputs_ce = global_model(poison_images)

    # Check for numerical issues
    if not torch.isfinite(outputs_ce).all():
        print("[MOS WARNING] CE logits contain NaN/Inf, using safe fallback")
        total_numel = sum(t.numel() for t in global_model.state_dict().values())
        safe_grad = torch.zeros(total_numel, device=device)
        return safe_grad.clone(), safe_grad.clone()

    loss_ce = criterion_ce(outputs_ce, target_labels)
    loss_ce.backward()
    g_ce = extract_gradient_vector(global_model)

    del outputs_ce, loss_ce

    # ===== CW Margin Loss =====
    global_model.zero_grad(set_to_none=True)
    outputs_cw = global_model(poison_images)

    # Extract correct class logits
    correct_logits = torch.gather(outputs_cw, 1, target_labels.unsqueeze(1)).squeeze(1)

    # Find max other class logit
    outputs_clone = outputs_cw.clone()
    outputs_clone.scatter_(1, target_labels.unsqueeze(1), -1e4)
    max_other_logits, _ = torch.max(outputs_clone, dim=1)

    # CW loss: maximize (max_other - correct)
    loss_cw = torch.mean(torch.relu(max_other_logits - correct_logits + 20.0))
    loss_cw.backward()
    g_cw = extract_gradient_vector(global_model)

    del outputs_cw, loss_cw, outputs_clone

    global_model.zero_grad(set_to_none=True)

    return g_ce, g_cw

# ============================================================================
# Main Attack Entry Point
# ============================================================================
@torch.no_grad()
def mos_attack(
    all_updates: List[TensorDict],
    args,
    malicious_attackers_this_round: int,
    g_ce: Optional[torch.Tensor] = None,
    g_cw: Optional[torch.Tensor] = None,
    historical_pop: Optional[torch.Tensor] = None,
    lam: float = 0.5
) -> Tuple[List[TensorDict], Optional[torch.Tensor]]:
    """
    MOS-Attack: Multi-Objective Stealth Attack (Simplified Paper Version)

    Args:
        all_updates: List of client updates (first K are malicious slots)
        args: Configuration object
        malicious_attackers_this_round: Number of malicious clients (K)
        g_ce: CE surrogate gradient (optional)
        g_cw: CW surrogate gradient (optional)
        historical_pop: Historical population for warm start (optional)
        lam: CE/CW mixing ratio

    Returns:
        all_updates: Modified updates with malicious updates replaced
        historical_perturbation: Best template perturbation for next round
    """
    if malicious_attackers_this_round == 0:
        return all_updates, historical_pop

    K = malicious_attackers_this_round
    device = args.device if hasattr(args, 'device') else 'cpu'

    print(f"\n{'='*60}")
    print(f"[MOS-Core] Starting MOS-Attack")
    print(f"{'='*60}")

    # ========================================================================
    # Step 1: Flatten updates and record layer dimensions
    # ========================================================================
    layer_dims = []
    idx_current = 0

    for k, v in all_updates[0].items():
        num_params = v.numel()
        layer_dims.append((k, idx_current, idx_current + num_params))
        idx_current += num_params

    total_params = idx_current
    global _LAST_VALID_GUIDANCE, _STABILITY_STATE_NUMEL
    if _STABILITY_STATE_NUMEL is not None and _STABILITY_STATE_NUMEL != total_params:
        _LAST_VALID_GUIDANCE = None
        print("[MOS-Core] Stability state reset: model dimension changed")
    _STABILITY_STATE_NUMEL = total_params
    print(f"[MOS-Core] Total parameters: {total_params:,}")

    # Extract benign updates
    benign_updates = all_updates[malicious_attackers_this_round:]
    benign_count = len(benign_updates)

    if benign_count == 0:
        print(f"[MOS-Core] WARNING: No benign clients, returning noisy original")
        for i in range(K):
            original_vec = torch.cat([p.flatten() for p in all_updates[i].values()])
            noise = torch.randn_like(original_vec) * 1e-6
            all_updates[i] = vector_to_net_dict(original_vec + noise, copy.deepcopy(all_updates[i]))
        return all_updates, historical_pop

    print(f"[MOS-Core] Benign clients: {benign_count}")
    print(f"[MOS-Core] Malicious slots: {K}")

    # Flatten benign updates
    benign_grads = torch.empty(
        (benign_count, total_params),
        device=device,
        dtype=next(iter(benign_updates[0].values())).dtype
    )

    for row, update in enumerate(benign_updates):
        offset = 0
        for k in update.keys():
            tensor = update[k]
            flat = tensor.reshape(-1)
            end = offset + flat.numel()
            benign_grads[row, offset:end].copy_(flat)
            offset = end

    # ========================================================================
    # Step 2: Compute benign statistics
    # ========================================================================
    benign_mean = torch.mean(benign_grads, dim=0)
    benign_std = torch.std(benign_grads, dim=0, correction=0) + 1e-9

    b_mean_norm = torch.norm(benign_mean).item()
    b_std_mean = torch.mean(benign_std).item()

    print(f"[MOS-Core] Benign mean norm: {b_mean_norm:.4f}")
    print(f"[MOS-Core] Benign std (mean): {b_std_mean:.4f}")

    # Check for numerical issues
    if not torch.isfinite(benign_mean).all():
        print(f"[MOS-Core] ERROR: Benign mean contains NaN/Inf")
        return all_updates, historical_pop

    # ========================================================================
    # Step 3: Construct attack guidance
    # ========================================================================
    min_guidance_norm = getattr(args, 'min_guidance_norm', 1e-8)
    g_attack, guidance_diag = _construct_attack_guidance(
        g_ce, g_cw, historical_pop, benign_grads, benign_mean,
        total_params, lam, min_guidance_norm
    )
    for key in ("ce_raw_norm", "cw_raw_norm", "ce_valid", "cw_valid",
                "ce_cw_cosine", "combined_pre_norm",
                "historical_direction_valid", "historical_direction_norm",
                "guidance_history_cosine", "guidance_source",
                "guidance_fallback_used"):
        print(f"[MOS-Core] {key}={guidance_diag[key]}")
    final_guidance_norm = torch.norm(g_attack).item() if g_attack is not None else 0.0
    print(f"[MOS-Core] final_guidance_norm={final_guidance_norm:.6f}")

    # ========================================================================
    # Step 4: Compute attack budget
    # ========================================================================
    dists_benign = torch.norm(benign_grads - benign_mean, dim=1)
    radius_quantile = getattr(args, 'radius_quantile', 0.95)
    base_threshold = torch.quantile(dists_benign, q=radius_quantile)

    attack_budget_ratio = getattr(args, 'attack_budget_ratio', 1.0)
    raw_budget = (attack_budget_ratio * base_threshold).item()
    raw_budget_valid = raw_budget >= 0 and torch.isfinite(torch.tensor(raw_budget)).item()

    if not raw_budget_valid or raw_budget < 1e-9:
        # Invalid current budget: cannot reliably attack this round
        print("[MOS-Core] WARNING: Current round budget is invalid or near-zero")
        print("[MOS-Core] current_budget=invalid")
        print("[MOS-Core] Skipping attack: using benign mean template")

        best_template = benign_mean.clone()
        for i in range(K):
            all_updates[i] = vector_to_net_dict(best_template, copy.deepcopy(all_updates[i]))

        # Preserve incoming historical_pop
        return all_updates, historical_pop

    current_budget = raw_budget
    max_dev_threshold = benign_mean.new_tensor(current_budget)

    print(f"[MOS-Core] Base radial threshold (q={radius_quantile}): {base_threshold:.4f}")
    print(f"[MOS-Core] Attack budget ratio: {attack_budget_ratio}")
    print(f"[MOS-Core] current_budget={current_budget:.6f}")
    print(f"[MOS-Core] current_budget_valid={raw_budget_valid}")

    # Never spend generations optimizing a zero/invalid attack objective.
    if g_attack is None or final_guidance_norm < min_guidance_norm:
        best_template = benign_mean.clone()
        hist_unit, _, hist_valid = safe_normalize(
            historical_pop, min_norm=min_guidance_norm, expected_numel=total_params
        )
        if hist_valid and current_budget > 0:
            historical_seed_scale = max(
                0.0, min(float(getattr(args, 'historical_seed_scale', 0.5)), 1.0)
            )
            scale = historical_seed_scale * current_budget
            best_template = benign_mean + scale * hist_unit.to(device)
            # Keep the skipped-round template within the same hard budget
            # guarantee as normal evolved/output candidates.
            best_template, _, _ = project_to_attack_budget(
                best_template.unsqueeze(0), benign_mean, max_dev_threshold
            )
            best_template = best_template.squeeze(0)
        print("[MOS-Core] ATTACK SKIPPED: no valid attack guidance")
        print("[MOS-Core] attack_skipped=True")
        print("[MOS-Core] skip_reason=no_valid_guidance")
        for i in range(K):
            all_updates[i] = vector_to_net_dict(best_template, copy.deepcopy(all_updates[i]))
        historical_perturbation = (best_template - benign_mean).unsqueeze(0).detach()
        return all_updates, historical_perturbation

    # ========================================================================
    # Step 5: Initialize constraints
    # ========================================================================
    print(f"\n[MOS-Core] Initializing constraints...")

    radial_weight = getattr(args, 'weight_radial', 1.0)
    sign_weight = getattr(args, 'weight_sign', 0.5)
    sign_layer_reduce = getattr(args, 'sign_layer_reduce', 'quantile')
    sign_layer_quantile = getattr(args, 'sign_layer_quantile', 0.9)

    constraints = [
        RadialConstraint(weight=radial_weight, quantile=radius_quantile),
        SignConstraint(
            weight=sign_weight,
            quantile=radius_quantile,
            layer_reduce=sign_layer_reduce,
            layer_quantile=sign_layer_quantile
        )
    ]

    print(f"[MOS-Core] Enabled constraints: Radial, Sign")
    print(f"[MOS-Core] Radial weight: {radial_weight}")
    print(f"[MOS-Core] Sign weight: {sign_weight}, layer_reduce: {sign_layer_reduce}")

    # Fit constraints on benign updates
    context = {'layer_dims': layer_dims}

    for constraint in constraints:
        constraint.fit(benign_grads, benign_mean, context)
        print(f"[MOS-Core]   {constraint.name.capitalize()} threshold: {constraint.threshold:.4f}")

    # ========================================================================
    # Step 6: Initialize population with multi-scale attack directions
    # ========================================================================
    pop_size = getattr(args, 'evo_pop_size', 10)
    print(f"\n[MOS-Core] Initializing population (size={pop_size})...")

    # Multi-scale initialization
    adaptive_guided_init = bool(getattr(args, 'mos_adaptive_guided_init', False))
    if adaptive_guided_init:
        alpha_feasible = _estimate_feasible_alpha(
            benign_mean, max_dev_threshold, g_attack, constraints, context
        )
        default_scales = [fraction * alpha_feasible
                          for fraction in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
        print(f"[MOS-Core] adaptive_guided_init=True "
              f"alpha_feasible={alpha_feasible:.6f} scales={default_scales}")
    else:
        default_scales = [0.0, 0.10, 0.25, 0.50, 0.75, 0.95]

    # Select scales based on pop_size
    if pop_size <= len(default_scales):
        # Use evenly spaced scales, always keeping 0.0 and 0.95
        if pop_size == 1:
            selected_scales = [default_scales[-1]]
        elif pop_size == 2:
            selected_scales = [0.0, default_scales[-1]]
        elif pop_size == 3:
            selected_scales = ([0.0, 0.5 * alpha_feasible, alpha_feasible]
                               if adaptive_guided_init else [0.0, 0.5, 0.95])
        else:
            # Uniformly sample from default scales
            indices = torch.linspace(0, len(default_scales) - 1, pop_size).long()
            selected_scales = [default_scales[i] for i in indices]
    else:
        selected_scales = default_scales.copy()

    num_scale_seeds = len(selected_scales)
    population = torch.empty(pop_size, total_params, device=device)

    # Generate scale-based seeds
    for i, scale in enumerate(selected_scales):
        population[i] = benign_mean + scale * max_dev_threshold * g_attack

    print(f"[MOS-Core] Generated {num_scale_seeds} scale-based seeds: {selected_scales}")

    # Historical seed injection
    historical_used = False
    if historical_pop is not None and pop_size > 0:
        try:
            # historical_pop is expected to be a perturbation relative to benign mean
            hist_pert = historical_pop.squeeze() if historical_pop.dim() > 1 else historical_pop

            if hist_pert.shape[0] == total_params and torch.isfinite(hist_pert).all():
                hist_unit, hist_raw_norm, hist_valid = safe_normalize(
                    hist_pert, min_norm=min_guidance_norm, expected_numel=total_params
                )
                if not hist_valid:
                    raise ValueError("historical perturbation has near-zero norm")
                historical_seed_scale = max(
                    0.0, min(float(getattr(args, 'historical_seed_scale', 0.5)), 1.0)
                )
                hist_target_scale = historical_seed_scale * max_dev_threshold
                # Preserve direction but assign the current bounded scale.
                hist_candidate = benign_mean + hist_target_scale * hist_unit.to(device)
                hist_candidate_proj, _, _ = project_to_attack_budget(
                    hist_candidate.unsqueeze(0), benign_mean, max_dev_threshold
                )
                hist_candidate_proj = hist_candidate_proj.squeeze(0)
                hist_norm_post = torch.norm(hist_candidate_proj - benign_mean).item()

                # Replace a mid-scale seed (e.g., 0.25 or 0.5 position)
                if adaptive_guided_init:
                    replace_idx = num_scale_seeds // 2
                elif 0.25 in selected_scales:
                    replace_idx = selected_scales.index(0.25)
                elif 0.5 in selected_scales:
                    replace_idx = selected_scales.index(0.5)
                else:
                    replace_idx = min(1, num_scale_seeds - 1)  # Avoid replacing 0.0 or 0.95

                population[replace_idx] = hist_candidate_proj
                historical_used = True

                print(f"[MOS-Core] Historical seed injected at position {replace_idx}")
                print(f"[MOS-Core] historical_raw_norm={hist_raw_norm:.6f}")
                print(f"[MOS-Core] historical_direction_norm={torch.norm(hist_unit).item():.6f}")
                print(f"[MOS-Core] historical_seed_target_scale={hist_target_scale.item():.6f}")
                print(f"[MOS-Core] historical_seed_final_norm={hist_norm_post:.6f}")
            else:
                print(f"[MOS-Core] WARNING: Historical seed dimension mismatch or contains NaN/Inf, ignoring")
        except Exception as e:
            print(f"[MOS-Core] WARNING: Failed to use historical seed: {e}")

    # Fill remaining slots with random candidates
    random_noise_scale = 0.1
    for i in range(num_scale_seeds, pop_size):
        alpha = torch.rand(1, device=device).item() * 0.95  # Random scale in [0, 0.95]
        candidate = (
            benign_mean +
            alpha * max_dev_threshold * g_attack +
            random_noise_scale * benign_std * torch.randn_like(benign_mean)
        )
        population[i] = candidate

    if pop_size > num_scale_seeds:
        print(f"[MOS-Core] Generated {pop_size - num_scale_seeds} random candidates "
              f"(noise_scale={random_noise_scale})")

    ray_diagnostics = bool(getattr(args, 'mos_inject_attack_ray_diagnostics', False))
    ray_alphas = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    provenance = [None] * pop_size
    if ray_diagnostics:
        ray_population = torch.stack([
            benign_mean + alpha * max_dev_threshold * g_attack for alpha in ray_alphas
        ])
        population = torch.cat([population, ray_population], dim=0)
        provenance.extend(ray_alphas)

    # Project entire population to budget
    population, init_clipped, init_norms = project_to_attack_budget(
        population, benign_mean, max_dev_threshold
    )

    budget_ratios = init_norms / max_dev_threshold
    print(f"[MOS-Core] Initial population norms: "
          f"min={init_norms.min():.4f}, mean={init_norms.mean():.4f}, max={init_norms.max():.4f}")
    print(f"[MOS-Core] Budget ratios: "
          f"min={budget_ratios.min():.4f}, mean={budget_ratios.mean():.4f}, max={budget_ratios.max():.4f}")
    print(f"[MOS-Core] Clipped ratio: {init_clipped.float().mean():.2%}")
    print(f"[MOS-Core] Historical seed used: {historical_used}")

    if ray_diagnostics:
        init_obj, _, _, init_cv, _ = compute_dual_objectives(
            population, benign_mean, constraints, g_attack, context
        )
        print("[MOS-RayDiag] stage=post_injection")
        for idx, alpha in enumerate(provenance):
            if alpha is not None:
                print(f"[MOS-RayDiag] alpha={alpha:.6f} stealth={-init_obj[0, idx].item():.6f} "
                      f"destructiveness={-init_obj[1, idx].item():.6f} "
                      f"cv={init_cv[idx].item():.6f} feasible={init_cv[idx].item() <= 1e-6}")

        selected_indices = nsga2_select(init_obj, pop_size, total_cv=init_cv)
        population = population[selected_indices]
        provenance = [provenance[i] for i in selected_indices]
        survivors = [alpha for alpha in provenance if alpha is not None]
        statuses = {alpha: alpha in survivors for alpha in ray_alphas}
        print(f"[MOS-RayDiag] stage=post_first_environmental_selection survival={statuses}")

    # ========================================================================
    # Step 7: NSGA-II Evolution
    # ========================================================================
    generations = getattr(args, 'nsga_generations', 100)
    eta = float(getattr(args, 'sbx_eta', 15.0))
    crossover_prob = getattr(args, 'sbx_crossover_prob', 0.9)
    mutation_scale = getattr(args, 'mos_mutation_scale', 0.05)

    print(f"\n[MOS-Core] Starting NSGA-II evolution...")
    print(f"[MOS-Core] Generations: {generations}")
    print(f"[MOS-Core] Population size: {pop_size}")
    print(f"[MOS-Core] SBX eta: {eta}, crossover_prob: {crossover_prob}")
    print(f"[MOS-Core] Mutation scale: {mutation_scale}")

    for gen in range(generations):
        # Evaluate current population
        objectives, constraint_scores, scores_dict, total_cv, cv_ratios = compute_dual_objectives(
            population, benign_mean, constraints, g_attack, context
        )

        # Binary tournament parent selection
        parent_indices = binary_tournament_selection(objectives, pop_size, total_cv=total_cv)

        # Generate offspring
        offspring = torch.empty_like(population)
        perm = torch.randperm(pop_size, device=device)

        for i in range(0, pop_size, 2):
            p1_idx = parent_indices[perm[i % pop_size]]
            p2_idx = parent_indices[perm[(i + 1) % pop_size]]

            parent1 = population[p1_idx]
            parent2 = population[p2_idx]

            # Crossover
            child1, child2 = sbx_crossover(parent1, parent2, eta, crossover_prob)

            # Mutation
            child1 = mutation(child1, benign_std, mutation_scale)
            child2 = mutation(child2, benign_std, mutation_scale)

            offspring[i] = child1
            if i + 1 < pop_size:
                offspring[i + 1] = child2

        # Check offspring for NaN/Inf
        if not torch.isfinite(offspring).all():
            print(f"[MOS-Core] WARNING: Offspring contains NaN/Inf at gen {gen+1}, skipping generation")
            continue

        # Project offspring to budget
        offspring, _, _ = project_to_attack_budget(offspring, benign_mean, max_dev_threshold)

        # Evaluate offspring
        objectives_offspring, _, _, offspring_cv, _ = compute_dual_objectives(
            offspring, benign_mean, constraints, g_attack, context
        )

        # Combine parent and offspring
        combined_pop = torch.cat([population, offspring], dim=0)
        combined_obj = torch.cat([objectives, objectives_offspring], dim=1)
        combined_cv = torch.cat([total_cv, offspring_cv], dim=0)

        # Environmental selection
        selected_indices = nsga2_select(combined_obj, pop_size, total_cv=combined_cv)

        # Update population
        population = combined_pop[selected_indices]
        if ray_diagnostics:
            combined_provenance = provenance + [None] * pop_size
            provenance = [combined_provenance[i] for i in selected_indices]

        # Re-evaluate updated population for accurate logging
        objectives, constraint_scores, scores_dict, total_cv, cv_ratios = compute_dual_objectives(
            population, benign_mean, constraints, g_attack, context
        )

        # Logging
        if (gen + 1) % 10 == 0 or gen == 0:
            # Stealth and destructiveness
            stealth = -objectives[0]
            destructiveness = -objectives[1]
            best_stealth = stealth.max().item()
            mean_stealth = stealth.mean().item()
            best_destruct = destructiveness.max().item()
            mean_destruct = destructiveness.mean().item()

            # CV metrics
            min_cv = total_cv.min().item()
            mean_cv = total_cv.mean().item()
            feasible_count = (total_cv <= 1e-6).sum().item()
            feasible_ratio = feasible_count / pop_size

            # Feasible destructiveness
            feasible_mask = total_cv <= 1e-6
            if feasible_mask.any():
                max_feasible_destruct = destructiveness[feasible_mask].max().item()
            else:
                max_feasible_destruct = None

            # Pareto front size
            fronts = nondominated_sort(objectives)
            pareto_front_size = len(fronts[0]) if fronts else 0

            # Budget usage
            norms = torch.norm(population - benign_mean, dim=1)
            budget_ratios = norms / max_dev_threshold
            mean_budget = budget_ratios.mean().item()
            max_budget = budget_ratios.max().item()

            # Per-constraint ratios
            radial_ratios = cv_ratios['radial']
            radial_scores = scores_dict['radial']
            sign_ratios = cv_ratios['sign']
            sign_scores = scores_dict['sign']

            print(f"[MOS-Core] Gen {gen+1}/{generations}:")
            print(f"  Stealth: mean={mean_stealth:.3f}, best={best_stealth:.3f}")
            print(f"  Destructiveness: mean={mean_destruct:.3f}, best={best_destruct:.3f}")
            print(f"  CV: mean={mean_cv:.4f}, min={min_cv:.4f}")
            print(f"  Feasibility: count={feasible_count}, ratio={feasible_ratio:.2%}")
            if max_feasible_destruct is not None:
                print(f"  Max feasible destruct: {max_feasible_destruct:.3f}")
            else:
                print(f"  Max feasible destruct: N/A")
            print(f"  Pareto front size: {pareto_front_size}")
            print(f"  Budget ratio: mean={mean_budget:.3f}, max={max_budget:.3f}")
            print(f"  Radial: mean_ratio={radial_ratios.mean():.3f}, max_ratio={radial_ratios.max():.3f}, "
                  f"mean_score={radial_scores.mean():.3f}")
            print(f"  Sign: mean_ratio={sign_ratios.mean():.3f}, max_ratio={sign_ratios.max():.3f}, "
                  f"mean_score={sign_scores.mean():.3f}")

    # ========================================================================
    # Step 8: Final solution selection
    # ========================================================================
    print(f"\n[MOS-Core] Selecting final solution...")

    # Final evaluation
    final_objectives, final_constraint_scores, final_scores_dict, final_cv, final_cv_ratios = compute_dual_objectives(
        population, benign_mean, constraints, g_attack, context
    )

    # Compute final norms and budget ratios
    final_norms = torch.norm(population - benign_mean, dim=1)
    final_budget_ratios = final_norms / max_dev_threshold

    # Select best solution with Deb constraint-domination
    best_idx, selection_diagnostics = select_final_solution(
        population,
        final_objectives,
        benign_mean=benign_mean,
        total_cv=final_cv,
        constraint_scores=final_constraint_scores,
        scores_dict=final_scores_dict,
        args=args
    )

    if ray_diagnostics:
        survivors = [alpha for alpha in provenance if alpha is not None]
        selected_alpha = provenance[best_idx]
        statuses = {alpha: alpha in survivors for alpha in ray_alphas}
        print(f"[MOS-RayDiag] stage=final survival={statuses} "
              f"selected_is_injected={selected_alpha is not None} selected_alpha={selected_alpha}")

    # Print selection parameters
    print(f"[MOS-Core] Selection parameters:")
    print(f"  Lambda_s (stealth weight): {selection_diagnostics.get('lambda_s', 0.5):.2f}")
    print(f"  Lambda_a (attack weight): {selection_diagnostics.get('lambda_a', 0.5):.2f}")
    print(f"  Attack floor ratio: {selection_diagnostics.get('attack_floor', 0.0):.4f}")

    # Print feasibility information
    if 'feasible_count' in selection_diagnostics:
        print(f"\n[MOS-Core] Feasibility summary:")
        print(f"  Feasible solutions: {selection_diagnostics['feasible_count']}")
        print(f"  Feasible ratio: {selection_diagnostics['feasible_ratio']:.2%}")
        print(f"  Selection mode: {selection_diagnostics.get('selection_feasibility_mode', 'N/A')}")
        print(f"  Min CV in front: {selection_diagnostics.get('min_cv', 0):.4f}")

    # Print Pareto front summary
    print(f"\n[MOS-Core] Pareto front summary:")
    print(f"  Front size: {selection_diagnostics.get('pareto_front_size', 0)}")
    print(f"  Stealth range: [{selection_diagnostics.get('pareto_stealth_min', 0):.3f}, "
          f"{selection_diagnostics.get('pareto_stealth_max', 0):.3f}]")
    print(f"  Destructiveness range: [{selection_diagnostics.get('pareto_destructiveness_min', 0):.3f}, "
          f"{selection_diagnostics.get('pareto_destructiveness_max', 0):.3f}]")
    if 'pareto_cv_min' in selection_diagnostics:
        print(f"  CV range: [{selection_diagnostics.get('pareto_cv_min', 0):.4f}, "
              f"{selection_diagnostics.get('pareto_cv_max', 0):.4f}]")
    print(f"  Candidates after attack floor: {selection_diagnostics.get('candidates_after_floor', 0)}")

    # Extract best solution details
    best_template = population[best_idx].clone().detach()
    best_stealth = -final_objectives[0, best_idx].item()
    best_destruct = -final_objectives[1, best_idx].item()
    best_cv = final_cv[best_idx].item()
    best_norm = final_norms[best_idx].item()
    best_budget_ratio = final_budget_ratios[best_idx].item()
    selected_guidance_alignment = torch.dot(
        best_template - benign_mean, g_attack
    ).item() / max(best_norm, 1e-12)

    print(f"\n[MOS-Core] Selected solution (index={best_idx}):")
    print(f"[MOS-Core]   Constraint pass score (R): {best_stealth:.3f}")
    print(f"[MOS-Core]   Destructiveness (A): {best_destruct:.3f}")
    print(f"[MOS-Core]   Total CV: {best_cv:.4f}")
    print(f"[MOS-Core]   Deviation norm: {best_norm:.4f}")
    print(f"[MOS-Core]   Budget usage: {best_budget_ratio:.2%}")
    print(f"[MOS-Core] selection_mode={selection_diagnostics.get('selection_mode', 'balanced_knee')}")
    print(f"[MOS-Core] selected_budget_ratio={best_budget_ratio:.6f}")
    print(f"[MOS-Core] selected_stealth={best_stealth:.6f}")
    print(f"[MOS-Core] selected_destructiveness={best_destruct:.6f}")
    print(f"[MOS-Core] selected_cv={best_cv:.6f}")
    print(f"[MOS-Core] selected_feasible={best_cv <= 1e-6}")
    print(f"[MOS-Core] selected_guidance_alignment={selected_guidance_alignment:.6f}")

    # Per-constraint diagnostics
    print(f"[MOS-Core] Per-constraint details:")
    for constraint in constraints:
        score = final_scores_dict[constraint.name][best_idx].item()
        ratio = final_cv_ratios[constraint.name][best_idx].item()
        threshold = constraint.threshold.item()
        loss = ratio * (threshold + 1e-12)
        print(f"[MOS-Core]   {constraint.name.capitalize()}: score={score:.3f}, "
              f"ratio={ratio:.3f}, loss={loss:.4f}, threshold={threshold:.4f}")

    # ========================================================================
    # Step 9: Generate K malicious updates
    # ========================================================================
    print(f"\n[MOS-Core] Generating {K} malicious updates...")

    noise_scale = getattr(args, 'template_noise_scale', 1e-4)
    output_norms = []

    for i in range(K):
        # Add small noise to avoid identical updates
        noise_i = torch.randn_like(best_template) * noise_scale * benign_std
        optimized_grad_i = best_template + noise_i

        # Project to budget
        optimized_grad_i, _, _ = project_to_attack_budget(
            optimized_grad_i.unsqueeze(0), benign_mean, max_dev_threshold
        )
        optimized_grad_i = optimized_grad_i.squeeze(0)

        output_norms.append(torch.norm(optimized_grad_i - benign_mean).item())

        # Convert back to dictionary format
        all_updates[i] = vector_to_net_dict(
            optimized_grad_i,
            copy.deepcopy(all_updates[i])
        )

    print(f"[MOS-Core] Template noise scale: {noise_scale}")
    print(f"[MOS-Core] Output norms: min={min(output_norms):.4f}, "
          f"mean={sum(output_norms)/len(output_norms):.4f}, max={max(output_norms):.4f}")

    # ========================================================================
    # Step 10: Return
    # ========================================================================
    historical_perturbation = (best_template - benign_mean).unsqueeze(0).detach()

    print(f"[MOS-Core] Attack completed successfully")
    print(f"{'='*60}\n")

    return all_updates, historical_perturbation
