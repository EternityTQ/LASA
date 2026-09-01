"""
MOS-Attack NSGA-II Genetic Algorithm
Multi-objective optimization with Pareto sorting and crowding distance.
"""

import torch
from typing import List, Tuple, Dict, Optional

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
    num_parents: int,
    total_cv: Optional[torch.Tensor] = None,
    feasibility_tol: float = 1e-6
) -> List[int]:
    """
    Binary tournament selection with Deb constraint-domination
    Returns list of selected parent indices
    """
    M, N = objectives.shape
    device = objectives.device

    # When CV is provided, compute rank/crowding for feasible subset independently
    if total_cv is not None:
        feasible_mask = total_cv <= feasibility_tol
        feasible_indices = [i for i in range(N) if feasible_mask[i].item()]

        # Compute ranks and crowding only for feasible subset
        if feasible_indices:
            feasible_obj = objectives[:, feasible_indices]
            feasible_ranks, feasible_crowding = compute_rank_and_crowding(feasible_obj)
            # Map back to global indices
            feasible_rank_map = {feasible_indices[i]: feasible_ranks[i].item() for i in range(len(feasible_indices))}
            feasible_crowding_map = {feasible_indices[i]: feasible_crowding[i].item() for i in range(len(feasible_indices))}
        else:
            feasible_rank_map = {}
            feasible_crowding_map = {}
    else:
        # No CV: compute rank/crowding for entire population
        ranks, crowding = compute_rank_and_crowding(objectives)

    parent_indices = []

    for _ in range(num_parents):
        # Randomly select two candidates
        candidates = torch.randint(0, N, (2,), device=device)
        idx1, idx2 = candidates[0].item(), candidates[1].item()

        # Constraint-domination comparison
        if total_cv is not None:
            feasible1 = idx1 in feasible_rank_map
            feasible2 = idx2 in feasible_rank_map

            if feasible1 and not feasible2:
                winner = idx1
            elif feasible2 and not feasible1:
                winner = idx2
            elif feasible1 and feasible2:
                # Both feasible: use feasible-subset rank/crowding
                if feasible_rank_map[idx1] < feasible_rank_map[idx2]:
                    winner = idx1
                elif feasible_rank_map[idx1] > feasible_rank_map[idx2]:
                    winner = idx2
                else:
                    winner = idx1 if feasible_crowding_map[idx1] >= feasible_crowding_map[idx2] else idx2
            else:
                # Both infeasible: CV primary, rank/crowding for CV ties
                cv1 = total_cv[idx1].item()
                cv2 = total_cv[idx2].item()
                if abs(cv1 - cv2) > feasibility_tol:
                    winner = idx1 if cv1 < cv2 else idx2
                else:
                    # CV tie: compute local rank/crowding for these two
                    tie_obj = objectives[:, [idx1, idx2]]
                    tie_ranks, tie_crowding = compute_rank_and_crowding(tie_obj)
                    if tie_ranks[0] < tie_ranks[1]:
                        winner = idx1
                    elif tie_ranks[0] > tie_ranks[1]:
                        winner = idx2
                    else:
                        winner = idx1 if tie_crowding[0] >= tie_crowding[1] else idx2
        else:
            # No CV: original NSGA-II logic
            if ranks[idx1] < ranks[idx2]:
                winner = idx1
            elif ranks[idx1] > ranks[idx2]:
                winner = idx2
            else:
                winner = idx1 if crowding[idx1] >= crowding[idx2] else idx2

        parent_indices.append(winner)

    return parent_indices

# ============================================================================
# NSGA-II: Environmental Selection
# ============================================================================
def nsga2_select(
    objectives: torch.Tensor,
    pop_size: int,
    total_cv: Optional[torch.Tensor] = None,
    feasibility_tol: float = 1e-6
) -> List[int]:
    """NSGA-II environmental selection with Deb constraint handling"""

    if total_cv is None:
        # No CV: original NSGA-II
        fronts = nondominated_sort(objectives)
        chosen = []
        for front in fronts:
            if len(chosen) + len(front) <= pop_size:
                chosen.extend(front)
            else:
                remaining = pop_size - len(chosen)
                distances = crowding_distance(front, objectives)
                sorted_indices = torch.argsort(distances, descending=True)
                selected = [front[i] for i in sorted_indices[:remaining].tolist()]
                chosen.extend(selected)
                break
        return chosen

    # Constraint-aware selection
    N = objectives.shape[1]
    feasible_mask = total_cv <= feasibility_tol
    feasible_indices = [i for i in range(N) if feasible_mask[i].item()]
    infeasible_indices = [i for i in range(N) if not feasible_mask[i].item()]

    chosen = []

    # Step 1: Fill with feasible solutions using standard NSGA-II
    if feasible_indices:
        feasible_obj = objectives[:, feasible_indices]
        feasible_fronts = nondominated_sort(feasible_obj)

        for front in feasible_fronts:
            # Map back to global indices
            global_front = [feasible_indices[i] for i in front]
            if len(chosen) + len(global_front) <= pop_size:
                chosen.extend(global_front)
            else:
                remaining = pop_size - len(chosen)
                # Crowding distance on feasible subset
                front_obj = objectives[:, global_front]
                distances = crowding_distance(list(range(len(global_front))), front_obj)
                sorted_idx = torch.argsort(distances, descending=True)
                chosen.extend([global_front[i] for i in sorted_idx[:remaining].tolist()])
                break

    # Step 2: Fill remaining slots with infeasible solutions (if needed)
    if len(chosen) < pop_size and infeasible_indices:
        # Sort by CV (ascending)
        infeasible_cv_pairs = [(i, total_cv[i].item()) for i in infeasible_indices]
        infeasible_cv_pairs.sort(key=lambda x: x[1])

        # Process CV groups (equal within tolerance) in ascending order
        idx_pos = 0
        while idx_pos < len(infeasible_cv_pairs) and len(chosen) < pop_size:
            current_cv = infeasible_cv_pairs[idx_pos][1]
            # Find all indices with CV equal to current_cv (within tolerance)
            cv_group = []
            for i, cv in infeasible_cv_pairs[idx_pos:]:
                if abs(cv - current_cv) <= feasibility_tol:
                    cv_group.append(i)
                else:
                    break
            idx_pos += len(cv_group)

            remaining = pop_size - len(chosen)

            # If group fits entirely, add all
            if len(cv_group) <= remaining:
                chosen.extend(cv_group)
            else:
                # Need to select subset: use Pareto rank + crowding on objectives
                group_obj = objectives[:, cv_group]
                group_fronts = nondominated_sort(group_obj)

                for front in group_fronts:
                    global_front = [cv_group[i] for i in front]
                    remaining = pop_size - len(chosen)

                    if len(global_front) <= remaining:
                        chosen.extend(global_front)
                    else:
                        front_obj = objectives[:, global_front]
                        distances = crowding_distance(list(range(len(global_front))), front_obj)
                        sorted_idx = torch.argsort(distances, descending=True)
                        chosen.extend([global_front[i] for i in sorted_idx[:remaining].tolist()])
                        break
                    if len(chosen) >= pop_size:
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
    Select final solution with Deb constraint-domination.
    Feasible solutions prioritized; infeasible fall back to minimum-CV subset.
    """
    # Override with args if provided
    if args is not None:
        lambda_s = getattr(args, 'final_stealth_weight', lambda_s)
        lambda_a = getattr(args, 'final_attack_weight', lambda_a)
        attack_floor_ratio = getattr(args, 'final_attack_floor_ratio', attack_floor_ratio)

    objective_mode = getattr(args, 'mos_objective_mode', 'dual') if args is not None else 'dual'
    if objective_mode == 'a_only':
        feasible_indices = [i for i in range(len(total_cv)) if total_cv[i].item() <= 1e-6]
        if feasible_indices:
            candidates, feasibility_mode = feasible_indices, 'feasible'
        else:
            min_cv = total_cv.min().item()
            candidates = [i for i in range(len(total_cv))
                          if abs(total_cv[i].item() - min_cv) <= 1e-6]
            feasibility_mode = 'minimum_cv_fallback'
        best_idx = min(candidates, key=lambda i: objectives[1][i].item())
        candidate_r, candidate_a = -objectives[0, candidates], -objectives[1, candidates]
        candidate_cv = total_cv[candidates]
        return best_idx, {
            'pareto_front_size': len(candidates), 'candidates_after_floor': len(candidates),
            'pareto_stealth_min': candidate_r.min().item(), 'pareto_stealth_max': candidate_r.max().item(),
            'pareto_destructiveness_min': candidate_a.min().item(), 'pareto_destructiveness_max': candidate_a.max().item(),
            'attack_floor': 0.0, 'lambda_s': lambda_s, 'lambda_a': lambda_a,
            'selected_idx': best_idx, 'selection_mode': 'a_only',
            'feasible_count': len(feasible_indices), 'feasible_ratio': len(feasible_indices) / len(total_cv),
            'selection_feasibility_mode': feasibility_mode,
            'min_cv': candidate_cv.min().item(), 'pareto_cv_min': candidate_cv.min().item(),
            'pareto_cv_max': candidate_cv.max().item(), 'selected_cv': total_cv[best_idx].item(),
        }

    # IMPORTANT: Do NOT compute global Pareto front first
    # Step 1: Check feasibility across entire population
    if total_cv is not None:
        feasible_mask = total_cv <= 1e-6
        feasible_indices = [i for i in range(len(total_cv)) if feasible_mask[i].item()]

        if feasible_indices:
            # Case 1: Has feasible solutions
            # Restrict to feasible subset, compute Pareto fronts inside that subset
            feasible_obj = objectives[:, feasible_indices]
            feasible_fronts = nondominated_sort(feasible_obj)

            # Get first Pareto front within feasible subset
            if not feasible_fronts or not feasible_fronts[0]:
                print("[MOS-Core] WARNING: Empty feasible Pareto front, selecting index 0")
                return 0, {}

            # Map back to global indices
            front_indices = [feasible_indices[i] for i in feasible_fronts[0]]
            selection_feasibility_mode = 'feasible'

            # Store feasibility info for later diagnostics
            feasible_count = len(feasible_indices)
            feasible_ratio = len(feasible_indices) / len(total_cv)
            min_cv_val = total_cv[feasible_indices].min().item()
        else:
            # Case 2: No feasible solutions
            # Find global minimum-CV subset first
            min_cv = total_cv.min().item()
            min_cv_mask = torch.abs(total_cv - min_cv) <= 1e-6
            min_cv_indices = [i for i in range(len(total_cv)) if min_cv_mask[i].item()]

            # Compute Pareto fronts inside minimum-CV subset
            min_cv_obj = objectives[:, min_cv_indices]
            min_cv_fronts = nondominated_sort(min_cv_obj)

            if not min_cv_fronts or not min_cv_fronts[0]:
                print("[MOS-Core] WARNING: Empty minimum-CV Pareto front, selecting index 0")
                return 0, {}

            # Map back to global indices
            front_indices = [min_cv_indices[i] for i in min_cv_fronts[0]]
            selection_feasibility_mode = 'minimum_cv_fallback'

            # Store feasibility info for later diagnostics
            feasible_count = 0
            feasible_ratio = 0.0
            min_cv_val = min_cv
    else:
        # No CV: use global Pareto front
        fronts = nondominated_sort(objectives)
        if not fronts or not fronts[0]:
            print("[MOS-Core] WARNING: Empty first Pareto front, selecting index 0")
            return 0, {}
        front_indices = fronts[0]
        selection_feasibility_mode = 'no_cv'
        feasible_count = None
        feasible_ratio = None
        min_cv_val = None

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
        candidate_indices = [front_indices[i] for i in range(len(front_indices)) if candidate_mask[i].item()]
        candidate_norm_stealth = torch.stack([norm_stealth[i] for i in range(len(norm_stealth)) if candidate_mask[i].item()])
        candidate_norm_destruct = torch.stack([norm_destructiveness[i] for i in range(len(norm_destructiveness)) if candidate_mask[i].item()])
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

    # Add feasibility information to diagnostics
    if feasible_count is not None:
        diagnostics['feasible_count'] = feasible_count
        diagnostics['feasible_ratio'] = feasible_ratio
        diagnostics['selection_feasibility_mode'] = selection_feasibility_mode
        diagnostics['min_cv'] = min_cv_val

    if total_cv is not None:
        front_cv = total_cv[front_indices]
        diagnostics['pareto_cv_min'] = front_cv.min().item()
        diagnostics['pareto_cv_max'] = front_cv.max().item()
        diagnostics['selected_cv'] = total_cv[best_idx].item()

    return best_idx, diagnostics
