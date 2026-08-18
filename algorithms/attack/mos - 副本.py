"""
MOS-Attack (Multi-Objective Stealth Attack) - Simplified Paper Version
Dual-objective NSGA-II with Radial + Sign constraints. See mos_experimental.py for experimental features.
"""

import torch
import copy
from .lie import vector_to_net_dict
from typing import Dict, List, Tuple, Optional

# ============================================================================
# Type Annotations
# ============================================================================
TensorDict = Dict[str, torch.Tensor]
LayerDims = List[Tuple[str, int, int]]

# Lightweight stability state shared across consecutive federated rounds.
# Tensors are always detached; incompatible model dimensions reset the state.
_LAST_VALID_GUIDANCE: Optional[torch.Tensor] = None
_LAST_VALID_BUDGET: Optional[float] = None
_BUDGET_EMA: Optional[float] = None
_LAST_BENIGN_MEAN_NORM: Optional[float] = None
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

def _bounded_budget(
    raw_budget: float,
    previous_budget: Optional[float],
    previous_ema: Optional[float],
    beta: float,
    growth_cap: float,
    shrink_cap: float,
) -> Tuple[float, float]:
    """EMA diagnostic plus hard inter-round growth/shrink bounds."""
    ema = raw_budget if previous_ema is None else beta * previous_ema + (1.0 - beta) * raw_budget
    if previous_budget is None or previous_budget <= 0 or not torch.isfinite(torch.tensor(previous_budget)):
        return raw_budget, ema
    lower = shrink_cap * previous_budget
    upper = growth_cap * previous_budget
    return min(max(raw_budget, lower), upper), ema

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
# Constraint Plugin Base
# ============================================================================
class ConstraintPlugin:
    """Base class for constraint plugins"""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.threshold: Optional[torch.Tensor] = None

    def fit(self, benign_updates: torch.Tensor, benign_mean: torch.Tensor,
            context: Dict) -> None:
        raise NotImplementedError

    def loss(self, population: torch.Tensor, benign_mean: torch.Tensor,
             context: Dict) -> torch.Tensor:
        raise NotImplementedError

    def score(self, population: torch.Tensor, benign_mean: torch.Tensor,
              context: Dict, eps: float = 1e-12) -> torch.Tensor:
        loss_val = self.loss(population, benign_mean, context)
        ratio = loss_val / (self.threshold + eps)
        return self.score_from_ratio(ratio)

    def score_from_ratio(self, ratio: torch.Tensor) -> torch.Tensor:
        """Compute score from pre-computed ratio. Override for custom scoring."""
        return 1.0 / (1.0 + ratio)

# ============================================================================
# Radial Constraint (Distance from benign mean)
# ============================================================================
class RadialConstraint(ConstraintPlugin):
    """Radial distance constraint (L2 norm from benign mean)"""

    def __init__(self, weight: float = 1.0, quantile: float = 0.95):
        super().__init__(name='radial', weight=weight)
        self.quantile = quantile

    def fit(self, benign_updates: torch.Tensor, benign_mean: torch.Tensor,
            context: Dict) -> None:
        dists = torch.norm(benign_updates - benign_mean, dim=1)
        self.threshold = torch.quantile(dists, q=self.quantile)
        self.threshold = torch.clamp(self.threshold, min=1e-6)

    def loss(self, population: torch.Tensor, benign_mean: torch.Tensor,
             context: Dict) -> torch.Tensor:
        """Compute L2 distance from benign mean"""
        return torch.norm(population - benign_mean, dim=1)

# ============================================================================
# Sign Constraint (Layer-normalized sign violation)
# ============================================================================
class SignConstraint(ConstraintPlugin):
    """Layer-normalized sign constraint"""

    def __init__(self, weight: float = 0.5, quantile: float = 0.95,
                 layer_reduce: str = 'quantile', layer_quantile: float = 0.9):
        super().__init__(name='sign', weight=weight)
        self.quantile = quantile
        self.layer_reduce = layer_reduce
        self.layer_quantile = layer_quantile

    def fit(self, benign_updates: torch.Tensor, benign_mean: torch.Tensor,
            context: Dict) -> None:
        layer_dims = context['layer_dims']
        benign_losses = self._compute_layer_losses(
            benign_updates, benign_mean, layer_dims
        )
        self.threshold = torch.quantile(benign_losses, q=self.quantile)
        self.threshold = torch.clamp(self.threshold, min=1e-6)

    def _compute_layer_losses(self, population: torch.Tensor,
                             benign_mean: torch.Tensor,
                             layer_dims: LayerDims) -> torch.Tensor:
        layer_losses = []

        for layer_name, start_idx, end_idx in layer_dims:
            layer_pop = population[:, start_idx:end_idx]
            layer_mean = benign_mean[start_idx:end_idx]

            # Sign violation: negative dot product with mean's sign
            sign_violation = -layer_pop * torch.sign(layer_mean).unsqueeze(0)

            # ReLU to keep only violations, then normalize by layer norm
            violation_norm = torch.norm(torch.relu(sign_violation), dim=1)
            reference_norm = torch.norm(layer_mean) + 1e-12
            layer_loss = violation_norm / reference_norm

            layer_losses.append(layer_loss)

        layer_losses = torch.stack(layer_losses, dim=0)  # (L, P)

        # Aggregate across layers
        if self.layer_reduce == 'max':
            return layer_losses.max(dim=0)[0]
        elif self.layer_reduce == 'mean':
            return layer_losses.mean(dim=0)
        elif self.layer_reduce == 'quantile':
            return torch.quantile(layer_losses, q=self.layer_quantile, dim=0)
        else:
            return layer_losses.max(dim=0)[0]

    def loss(self, population: torch.Tensor, benign_mean: torch.Tensor,
             context: Dict) -> torch.Tensor:
        """Compute sign constraint loss"""
        layer_dims = context['layer_dims']
        return self._compute_layer_losses(population, benign_mean, layer_dims)

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
# Attack Budget Projection
# ============================================================================
def project_to_attack_budget(
    population: torch.Tensor,
    benign_mean: torch.Tensor,
    max_dev_threshold: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project population to attack budget

    Returns:
        projected: Projected population
        clipped_mask: Boolean mask of clipped individuals
        pre_norms: Pre-projection norms
    """
    centered = population - benign_mean
    norms = torch.norm(centered, dim=1, keepdim=True)

    clipped_mask = norms.squeeze(1) > max_dev_threshold
    scales = torch.clamp(max_dev_threshold / (norms + 1e-12), max=1.0)
    projected = benign_mean + centered * scales

    return projected, clipped_mask, norms.squeeze(1)

# ============================================================================
# Constraint Scoring and CV Computation
# ============================================================================
def compute_constraint_pass_score(
    population: torch.Tensor,
    benign_mean: torch.Tensor,
    constraints: List[ConstraintPlugin],
    context: Dict,
    eps: float = 1e-12,
    pre_computed_losses: Optional[Dict[str, torch.Tensor]] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute weighted constraint pass score

    Returns:
        total_score: Weighted average score (higher = more stealthy)
        scores_dict: Individual constraint scores
    """
    P = population.shape[0]
    device = population.device

    scores_dict = {}
    total_score = torch.zeros(P, device=device)
    total_weight = 0.0

    for constraint in constraints:
        if pre_computed_losses is not None and constraint.name in pre_computed_losses:
            loss_val = pre_computed_losses[constraint.name]
        else:
            loss_val = constraint.loss(population, benign_mean, context)

        ratio = loss_val / (constraint.threshold + eps)
        score_val = constraint.score_from_ratio(ratio)

        scores_dict[constraint.name] = score_val
        total_score += constraint.weight * score_val
        total_weight += constraint.weight

    total_score = total_score / (total_weight + eps)
    return total_score, scores_dict

# ============================================================================
# NSGA-II: Non-dominated Sorting
# ============================================================================
def nondominated_sort(objectives: torch.Tensor) -> List[List[int]]:
    """
    NSGA-II non-dominated sorting (minimization form)
    Returns list of Pareto fronts (each front is a list of indices)
    """
    M, N = objectives.shape
    device = objectives.device

    # Domination tracking
    S = [set() for _ in range(N)]  # S[p]: individuals dominated by p
    n = torch.zeros(N, dtype=torch.int32, device=device)  # n[p]: count dominating p
    rank = torch.full((N,), -1, dtype=torch.int32, device=device)
    fronts = []

    # Build domination relationships
    for p in range(N):
        for q in range(N):
            if p == q:
                continue

            # p dominates q if: all objectives <= and at least one <
            less_eq = torch.all(objectives[:, p] <= objectives[:, q])
            strictly_less = torch.any(objectives[:, p] < objectives[:, q])

            if less_eq and strictly_less:
                S[p].add(q)
            elif torch.all(objectives[:, q] <= objectives[:, p]) and \
                 torch.any(objectives[:, q] < objectives[:, p]):
                n[p] += 1

        if n[p] == 0:
            rank[p] = 0

    # Front 0
    current_front = [i for i in range(N) if rank[i] == 0]
    front_idx = 0

    # Build subsequent fronts
    while current_front:
        fronts.append(current_front)
        next_front = []

        for p in current_front:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = front_idx + 1
                    next_front.append(q)

        front_idx += 1
        current_front = next_front

    return fronts

# ============================================================================
# NSGA-II: Crowding Distance
# ============================================================================
def crowding_distance(front_indices: List[int], objectives: torch.Tensor) -> torch.Tensor:
    """
    Compute crowding distance for a Pareto front
    Returns distances for each individual in the front
    """
    M, _ = objectives.shape
    F = len(front_indices)
    device = objectives.device

    distances = torch.zeros(F, device=device)

    # Edge case: 1-2 individuals get infinite distance
    if F <= 2:
        return torch.full((F,), float('inf'), device=device)

    # For each objective dimension
    for m in range(M):
        obj_values = objectives[m, front_indices]
        sorted_idx = torch.argsort(obj_values)

        # Boundary points get infinite distance
        distances[sorted_idx[0]] = float('inf')
        distances[sorted_idx[-1]] = float('inf')

        obj_range = obj_values[sorted_idx[-1]] - obj_values[sorted_idx[0]]
        if obj_range < 1e-9:
            continue  # No diversity in this objective

        # Interior points: distance = (right - left) / range
        for i in range(1, F - 1):
            distances[sorted_idx[i]] += (
                obj_values[sorted_idx[i + 1]] - obj_values[sorted_idx[i - 1]]
            ) / obj_range

    return distances

# ============================================================================
# NSGA-II: Compute Rank and Crowding Distance
# ============================================================================
def compute_rank_and_crowding(objectives: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Pareto rank and crowding distance for each individual
    Returns ranks (0 = best front) and crowding distances
    """
    M, N = objectives.shape
    device = objectives.device

    # Compute Pareto fronts
    fronts = nondominated_sort(objectives)

    # Initialize rank and crowding arrays
    ranks = torch.full((N,), -1, dtype=torch.int32, device=device)
    crowding = torch.zeros(N, device=device)

    # Assign ranks and crowding distances
    for rank_val, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = rank_val

        # Compute crowding distance for this front
        front_crowding = crowding_distance(front, objectives)
        for i, idx in enumerate(front):
            crowding[idx] = front_crowding[i]

    return ranks, crowding

# ============================================================================
# Binary Tournament Selection
# ============================================================================
def binary_tournament_selection(
    objectives: torch.Tensor,
    num_parents: int
) -> List[int]:
    """
    Binary tournament selection based on Pareto rank and crowding distance
    Returns list of selected parent indices
    """
    M, N = objectives.shape
    device = objectives.device

    # Compute rank and crowding distance
    ranks, crowding = compute_rank_and_crowding(objectives)

    parent_indices = []

    for _ in range(num_parents):
        # Randomly select two candidates
        candidates = torch.randint(0, N, (2,), device=device)
        idx1, idx2 = candidates[0].item(), candidates[1].item()

        # Compare: better rank wins, tie-break by crowding distance
        if ranks[idx1] < ranks[idx2]:
            winner = idx1
        elif ranks[idx1] > ranks[idx2]:
            winner = idx2
        else:
            # Same rank: larger crowding distance wins
            if crowding[idx1] >= crowding[idx2]:
                winner = idx1
            else:
                winner = idx2

        parent_indices.append(winner)

    return parent_indices

# ============================================================================
# NSGA-II: Environmental Selection
# ============================================================================
def nsga2_select(
    objectives: torch.Tensor,
    pop_size: int
) -> List[int]:
    """
    NSGA-II environmental selection
    Returns indices of selected individuals
    """
    fronts = nondominated_sort(objectives)

    chosen = []
    for front in fronts:
        if len(chosen) + len(front) <= pop_size:
            # Whole front fits
            chosen.extend(front)
        else:
            # Last front: select by crowding distance
            remaining_slots = pop_size - len(chosen)
            distances = crowding_distance(front, objectives)

            # Sort by distance (descending), pick most diverse
            sorted_indices = torch.argsort(distances, descending=True)
            selected = [front[i] for i in sorted_indices[:remaining_slots].tolist()]
            chosen.extend(selected)
            break

    return chosen

# ============================================================================
# Genetic Operators: SBX Crossover
# ============================================================================
def sbx_crossover(
    parent1: torch.Tensor,
    parent2: torch.Tensor,
    eta: float = 15.0,
    crossover_prob: float = 0.9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simulated Binary Crossover (SBX)"""
    D = parent1.shape[0]
    device = parent1.device

    u = torch.rand(D, device=device)
    mask = torch.rand(D, device=device) <= crossover_prob

    # Compute beta distribution
    beta = torch.empty(D, device=device)
    le = u <= 0.5
    beta[le] = (2.0 * u[le]) ** (1.0 / (eta + 1.0))
    beta[~le] = (1.0 / (2.0 * (1.0 - u[~le]))) ** (1.0 / (eta + 1.0))

    # Generate children
    child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)

    # Apply crossover mask
    child1[~mask] = parent1[~mask]
    child2[~mask] = parent2[~mask]

    return child1, child2

# ============================================================================
# Genetic Operators: Mutation
# ============================================================================
def mutation(
    individual: torch.Tensor,
    benign_std: torch.Tensor,
    mutation_scale: float = 0.05
) -> torch.Tensor:
    """Gaussian mutation scaled by benign standard deviation"""
    noise = torch.randn_like(individual) * benign_std * mutation_scale
    return individual + noise

# ============================================================================
# Dual-Objective Computation
# ============================================================================
def compute_dual_objectives(
    population: torch.Tensor,
    benign_mean: torch.Tensor,
    constraints: List[ConstraintPlugin],
    attack_guidance: torch.Tensor,
    context: Dict
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute dual objectives for NSGA-II

    Objective 1: max R(x) - constraint pass score
    Objective 2: max A(x) - destructiveness (alignment with guidance)

    Returns:
        objectives: (2, P) matrix in minimization form
        constraint_pass_scores: (P,) total constraint scores
        scores_dict: Per-constraint scores
        total_cv: (P,) constraint violations
        ratios_dict: Per-constraint loss/threshold ratios
    """
    eps = 1e-12

    # Compute all constraint losses once
    losses_dict = {}
    for constraint in constraints:
        losses_dict[constraint.name] = constraint.loss(population, benign_mean, context)

    # Compute constraint pass scores using pre-computed losses
    constraint_pass_scores, scores_dict = compute_constraint_pass_score(
        population, benign_mean, constraints, context, eps, pre_computed_losses=losses_dict
    )

    # Compute destructiveness (alignment with attack guidance)
    centered = population - benign_mean
    destructiveness = centered @ attack_guidance

    # Compute CV and ratios inline
    total_cv = torch.zeros(population.shape[0], device=population.device)
    ratios_dict = {}
    for constraint in constraints:
        loss_val = losses_dict[constraint.name]
        ratio = loss_val / (constraint.threshold + eps)
        ratios_dict[constraint.name] = ratio
        violation = torch.relu(ratio - 1.0)
        total_cv += constraint.weight * violation

    # Convert to minimization form
    obj1 = -constraint_pass_scores
    obj2 = -destructiveness
    objectives = torch.stack([obj1, obj2], dim=0)

    return objectives, constraint_pass_scores, scores_dict, total_cv, ratios_dict

# ============================================================================
# Final Solution Selection
# ============================================================================
def select_final_solution(
    population: torch.Tensor,
    objectives: torch.Tensor,
    total_cv: Optional[torch.Tensor] = None,
    constraint_scores: Optional[torch.Tensor] = None,
    scores_dict: Optional[Dict[str, torch.Tensor]] = None,
    lambda_s: float = 0.5,
    lambda_a: float = 0.5,
    attack_floor_ratio: float = 0.10,
    args=None,
    benign_mean: Optional[torch.Tensor] = None,
) -> Tuple[int, Dict]:
    """
    Select final solution from first Pareto front.
    Default 'balanced_knee' mode: minimize distance to ideal point (1, 1).
    Alternative 'weighted_sum' mode: maximize weighted combination.
    """
    # Override with args if provided
    if args is not None:
        lambda_s = getattr(args, 'final_stealth_weight', lambda_s)
        lambda_a = getattr(args, 'final_attack_weight', lambda_a)
        attack_floor_ratio = getattr(args, 'final_attack_floor_ratio', attack_floor_ratio)

    # Get first Pareto front
    fronts = nondominated_sort(objectives)

    if not fronts or not fronts[0]:
        print("[MOS-Core] WARNING: Empty first Pareto front, selecting index 0")
        return 0, {}

    front_indices = fronts[0]

    # Convert objectives back to maximization form
    front_stealth = -objectives[0, front_indices]  # R(x)
    front_destructiveness = -objectives[1, front_indices]  # A(x)

    # Normalize stealth within front
    stealth_min = front_stealth.min()
    stealth_max = front_stealth.max()
    if stealth_max - stealth_min > 1e-9:
        norm_stealth = (front_stealth - stealth_min) / (stealth_max - stealth_min)
    else:
        # No variance: set to 0.5 (neutral)
        norm_stealth = torch.full_like(front_stealth, 0.5)

    # Normalize destructiveness within front
    dest_min = front_destructiveness.min()
    dest_max = front_destructiveness.max()
    if dest_max - dest_min > 1e-9:
        norm_destructiveness = (front_destructiveness - dest_min) / (dest_max - dest_min)
    else:
        # No variance: set to 0.5 (neutral)
        norm_destructiveness = torch.full_like(front_destructiveness, 0.5)

    # Attack floor protection (optional)
    attack_floor = attack_floor_ratio * max(dest_max.item(), 0.0)
    passes_floor = front_destructiveness >= attack_floor

    if attack_floor_ratio > 0 and passes_floor.any():
        # Filter to candidates passing attack floor
        candidate_mask = passes_floor
        candidate_indices = [front_indices[i] for i in range(len(front_indices)) if candidate_mask[i]]
        candidate_norm_stealth = norm_stealth[candidate_mask]
        candidate_norm_destruct = norm_destructiveness[candidate_mask]
    else:
        # Use full front
        candidate_indices = front_indices
        candidate_norm_stealth = norm_stealth
        candidate_norm_destruct = norm_destructiveness

    selection_mode = getattr(args, 'final_selection_mode', 'balanced_knee') if args is not None else 'balanced_knee'
    tie_tol = getattr(args, 'selection_tie_tol', 1e-6) if args is not None else 1e-6
    if selection_mode == 'weighted_sum':
        selection_values = lambda_s * candidate_norm_stealth + lambda_a * candidate_norm_destruct
        best_value = selection_values.max()
        tied = torch.nonzero(selection_values >= best_value - tie_tol).flatten()
    else:
        selection_mode = 'balanced_knee'
        selection_values = torch.sqrt(
            lambda_s * (1.0 - candidate_norm_stealth).square()
            + lambda_a * (1.0 - candidate_norm_destruct).square()
        )
        best_value = selection_values.min()
        tied = torch.nonzero(selection_values <= best_value + tie_tol).flatten()

    # Deterministic tie break: higher pass score, lower budget usage, higher attack.
    candidate_population = population[candidate_indices]
    budget_center = benign_mean if benign_mean is not None else population.mean(dim=0)
    candidate_budget = torch.norm(candidate_population - budget_center, dim=1)
    if constraint_scores is not None:
        candidate_stealth = constraint_scores[candidate_indices]
    else:
        candidate_stealth = candidate_norm_stealth
    best_idx_in_candidates = tied[0].item()
    for pos_tensor in tied[1:]:
        pos = pos_tensor.item()
        cur = best_idx_in_candidates
        cur_key = (candidate_stealth[cur].item(), -candidate_budget[cur].item(),
                   candidate_norm_destruct[cur].item())
        new_key = (candidate_stealth[pos].item(), -candidate_budget[pos].item(),
                   candidate_norm_destruct[pos].item())
        if new_key > cur_key:
            best_idx_in_candidates = pos
    best_idx = candidate_indices[best_idx_in_candidates]

    # Prepare diagnostics
    diagnostics = {
        'pareto_front_size': len(front_indices),
        'pareto_stealth_min': stealth_min.item(),
        'pareto_stealth_max': stealth_max.item(),
        'pareto_destructiveness_min': dest_min.item(),
        'pareto_destructiveness_max': dest_max.item(),
        'attack_floor': attack_floor,
        'candidates_after_floor': len(candidate_indices),
        'lambda_s': lambda_s,
        'lambda_a': lambda_a,
        'selected_idx': best_idx,
        'selection_mode': selection_mode,
    }

    if total_cv is not None:
        front_cv = total_cv[front_indices]
        diagnostics['pareto_cv_min'] = front_cv.min().item()
        diagnostics['pareto_cv_max'] = front_cv.max().item()

    return best_idx, diagnostics

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
) -> Tuple[List[TensorDict], torch.Tensor]:
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
        return all_updates

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
    global _LAST_VALID_GUIDANCE, _LAST_VALID_BUDGET, _BUDGET_EMA
    global _LAST_BENIGN_MEAN_NORM, _STABILITY_STATE_NUMEL
    if _STABILITY_STATE_NUMEL is not None and _STABILITY_STATE_NUMEL != total_params:
        _LAST_VALID_GUIDANCE = None
        _LAST_VALID_BUDGET = None
        _BUDGET_EMA = None
        _LAST_BENIGN_MEAN_NORM = None
        print("[MOS-Core] Stability state reset: model dimension changed")
    _STABILITY_STATE_NUMEL = total_params
    print(f"[MOS-Core] Total parameters: {total_params:,}")

    # Extract benign updates
    benign_updates = all_updates[malicious_attackers_this_round:]
    benign_count = len(benign_updates)

    if benign_count == 0:
        print(f"[MOS-Core] WARNING: No benign clients, returning noisy original")
        for i in range(K):
            original_vec = torch.cat([torch.flatten(all_updates[i][k]) for k in all_updates[i].keys()])
            noise = torch.randn_like(original_vec) * 1e-6
            all_updates[i] = vector_to_net_dict(original_vec + noise, copy.deepcopy(all_updates[i]))
        return all_updates

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
        return all_updates

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
    previous_budget = _LAST_VALID_BUDGET
    beta = getattr(args, 'budget_ema_beta', 0.9)
    growth_cap = getattr(args, 'budget_growth_cap', 2.0)
    shrink_cap = getattr(args, 'budget_shrink_cap', 0.25)
    raw_budget_valid = raw_budget >= 0 and torch.isfinite(torch.tensor(raw_budget)).item()
    if not raw_budget_valid:
        print("[MOS-Core] WARNING: raw attack budget is invalid; using last valid budget")
        raw_budget = previous_budget if previous_budget is not None else 0.0
    bounded_budget, budget_ema = _bounded_budget(
        raw_budget, previous_budget, _BUDGET_EMA, beta, growth_cap, shrink_cap
    )
    anomaly_threshold = getattr(args, 'update_scale_anomaly_threshold', 10.0)
    budget_growth_ratio = (raw_budget / (previous_budget + 1e-12)
                           if previous_budget is not None else 1.0)
    benign_mean_growth_ratio = (b_mean_norm / (_LAST_BENIGN_MEAN_NORM + 1e-12)
                                if _LAST_BENIGN_MEAN_NORM is not None else 1.0)
    update_scale_anomaly = (
        not raw_budget_valid
        or budget_growth_ratio > anomaly_threshold
        or benign_mean_growth_ratio > anomaly_threshold
    )
    max_dev_threshold = benign_mean.new_tensor(bounded_budget)
    if raw_budget_valid and raw_budget > 0:
        _LAST_VALID_BUDGET = bounded_budget
        _BUDGET_EMA = budget_ema
    _LAST_BENIGN_MEAN_NORM = b_mean_norm

    print(f"[MOS-Core] Base radial threshold (q={radius_quantile}): {base_threshold:.4f}")
    print(f"[MOS-Core] Attack budget ratio: {attack_budget_ratio}")
    print(f"[MOS-Core] raw_radial_budget={raw_budget:.6f}")
    print(f"[MOS-Core] previous_budget={previous_budget}")
    print(f"[MOS-Core] budget_ema={budget_ema:.6f}")
    print(f"[MOS-Core] bounded_attack_budget={bounded_budget:.6f}")
    print(f"[MOS-Core] budget_growth_ratio={budget_growth_ratio:.6f}")
    print(f"[MOS-Core] benign_mean_growth_ratio={benign_mean_growth_ratio:.6f}")
    print(f"[MOS-Core] update_scale_anomaly={update_scale_anomaly}")
    if update_scale_anomaly:
        print("[MOS-Core] WARNING: update scale anomaly detected; bounded budget enforced")

    # Never spend generations optimizing a zero/invalid attack objective.
    if g_attack is None or final_guidance_norm < min_guidance_norm:
        best_template = benign_mean.clone()
        hist_unit, _, hist_valid = safe_normalize(
            historical_pop, min_norm=min_guidance_norm, expected_numel=total_params
        )
        if hist_valid and bounded_budget > 0:
            historical_seed_scale = max(
                0.0, min(float(getattr(args, 'historical_seed_scale', 0.5)), 1.0)
            )
            scale = historical_seed_scale * bounded_budget
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
    default_scales = [0.0, 0.10, 0.25, 0.50, 0.75, 0.95]

    # Select scales based on pop_size
    if pop_size <= len(default_scales):
        # Use evenly spaced scales, always keeping 0.0 and 0.95
        if pop_size == 1:
            selected_scales = [0.95]
        elif pop_size == 2:
            selected_scales = [0.0, 0.95]
        elif pop_size == 3:
            selected_scales = [0.0, 0.5, 0.95]
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
                if 0.25 in selected_scales:
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
        parent_indices = binary_tournament_selection(objectives, pop_size)

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
        objectives_offspring, _, _, _, _ = compute_dual_objectives(
            offspring, benign_mean, constraints, g_attack, context
        )

        # Combine parent and offspring
        combined_pop = torch.cat([population, offspring], dim=0)
        combined_obj = torch.cat([objectives, objectives_offspring], dim=1)

        # Environmental selection
        selected_indices = nsga2_select(combined_obj, pop_size)

        # Update population
        population = combined_pop[selected_indices]

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
            feasible_ratio = (total_cv <= 1e-6).float().mean().item()

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
            print(f"  CV: mean={mean_cv:.4f}, min={min_cv:.4f}, feasible={feasible_ratio:.2%}")
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

    # Select best solution (NO CV filtering)
    best_idx, selection_diagnostics = select_final_solution(
        population,
        final_objectives,
        benign_mean=benign_mean,
        total_cv=final_cv,
        constraint_scores=final_constraint_scores,
        scores_dict=final_scores_dict,
        args=args
    )

    # Print selection parameters
    print(f"[MOS-Core] Selection parameters:")
    print(f"  Lambda_s (stealth weight): {selection_diagnostics.get('lambda_s', 0.5):.2f}")
    print(f"  Lambda_a (attack weight): {selection_diagnostics.get('lambda_a', 0.5):.2f}")
    print(f"  Attack floor ratio: {selection_diagnostics.get('attack_floor', 0.0):.4f}")

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

    # Detailed first Pareto front logging
    fronts = nondominated_sort(final_objectives)
    if fronts and fronts[0]:
        print(f"\n[MOS-Core] First Pareto front detailed breakdown:")
        print(f"[MOS-Core] {'Pos':<4} {'Idx':<4} {'Stealth':<8} {'Destruct':<9} {'CV':<8} "
              f"{'R_ratio':<8} {'R_score':<8} {'S_ratio':<8} {'S_score':<8} "
              f"{'Norm':<8} {'Budget%':<8} {'NormS':<7} {'NormA':<7} {'Score':<7} {'Floor':<6} {'Sel':<4}")

        front_indices = fronts[0]
        front_stealth = -final_objectives[0, front_indices]
        front_destruct = -final_objectives[1, front_indices]

        # Normalize for display
        s_min, s_max = front_stealth.min(), front_stealth.max()
        d_min, d_max = front_destruct.min(), front_destruct.max()

        if s_max - s_min > 1e-9:
            norm_s = (front_stealth - s_min) / (s_max - s_min)
        else:
            norm_s = torch.full_like(front_stealth, 0.5)

        if d_max - d_min > 1e-9:
            norm_d = (front_destruct - d_min) / (d_max - d_min)
        else:
            norm_d = torch.full_like(front_destruct, 0.5)

        lambda_s = selection_diagnostics.get('lambda_s', 0.5)
        lambda_a = selection_diagnostics.get('lambda_a', 0.5)
        selection_mode = selection_diagnostics.get('selection_mode', 'balanced_knee')

        if selection_mode == 'weighted_sum':
            score_metric = lambda_s * norm_s + lambda_a * norm_d
        else:  # balanced_knee
            score_metric = torch.sqrt(
                lambda_s * (1.0 - norm_s).square() + lambda_a * (1.0 - norm_d).square()
            )

        attack_floor = selection_diagnostics.get('attack_floor', 0.0)

        for pos, idx in enumerate(front_indices):
            idx_val = idx.item() if isinstance(idx, torch.Tensor) else idx
            stealth_val = front_stealth[pos].item()
            destruct_val = front_destruct[pos].item()
            cv_val = final_cv[idx].item()
            radial_ratio = final_cv_ratios['radial'][idx].item()
            radial_score = final_scores_dict['radial'][idx].item()
            sign_ratio = final_cv_ratios['sign'][idx].item()
            sign_score = final_scores_dict['sign'][idx].item()
            norm_val = final_norms[idx].item()
            budget_pct = final_budget_ratios[idx].item() * 100
            norm_s_val = norm_s[pos].item()
            norm_d_val = norm_d[pos].item()
            score_val = score_metric[pos].item()
            passes_floor = destruct_val >= attack_floor
            is_selected = (idx_val == best_idx)

            print(f"[MOS-Core] {pos:<4} {idx_val:<4} {stealth_val:<8.3f} {destruct_val:<9.2f} {cv_val:<8.4f} "
                  f"{radial_ratio:<8.3f} {radial_score:<8.3f} {sign_ratio:<8.3f} {sign_score:<8.3f} "
                  f"{norm_val:<8.3f} {budget_pct:<8.1f} {norm_s_val:<7.3f} {norm_d_val:<7.3f} "
                  f"{score_val:<7.3f} {'Y' if passes_floor else 'N':<6} {'***' if is_selected else '':<4}")

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
