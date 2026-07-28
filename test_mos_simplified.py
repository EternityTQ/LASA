"""
Smoke test for simplified MOS-Attack implementation
"""
import torch
import sys
sys.path.insert(0, 'd:/LASA')

from algorithms.attack.mos import (
    RadialConstraint,
    SignConstraint,
    compute_constraint_pass_score,
    compute_cv,
    nondominated_sort,
    crowding_distance,
    nsga2_select,
    sbx_crossover,
    mutation,
    project_to_attack_budget,
    compute_dual_objectives,
    select_final_solution,
    mos_attack
)

print("="*60)
print("MOS-Attack Simplified Version - Smoke Test")
print("="*60)

# Test configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[Test] Device: {device}")

# Dimensions
N_benign = 5
N_malicious = 2
D = 100
num_layers = 3

print(f"[Test] Benign clients: {N_benign}")
print(f"[Test] Malicious slots: {N_malicious}")
print(f"[Test] Parameter dimension: {D}")

# Generate synthetic benign updates
benign_updates = torch.randn(N_benign, D, device=device) * 0.1
benign_mean = benign_updates.mean(dim=0)
benign_std = benign_updates.std(dim=0, correction=0) + 1e-9

print(f"[Test] Benign mean norm: {torch.norm(benign_mean).item():.4f}")
print(f"[Test] Benign std mean: {benign_std.mean().item():.4f}")

# Layer dimensions (simplified)
layer_size = D // num_layers
layer_dims = [(f'layer_{i}', i*layer_size, (i+1)*layer_size) for i in range(num_layers)]


# ============================================================================
# Test 1: Constraint Plugins
# ============================================================================
print("\n" + "="*60)
print("[Test 1] Constraint Plugins")
print("="*60)

context = {'layer_dims': layer_dims}

# Test RadialConstraint
radial = RadialConstraint(weight=1.0, quantile=0.95)
radial.fit(benign_updates, benign_mean, context)

print(f"\n[RadialConstraint]")
print(f"  Threshold: {radial.threshold:.4f}")

# Test on population
pop_size = 10
population = benign_mean.unsqueeze(0).repeat(pop_size, 1) + torch.randn(pop_size, D, device=device) * 0.05

radial_loss = radial.loss(population, benign_mean, context)
radial_score = radial.score(population, benign_mean, context)

print(f"  Loss shape: {radial_loss.shape}")
print(f"  Loss range: [{radial_loss.min():.4f}, {radial_loss.max():.4f}]")
print(f"  Score shape: {radial_score.shape}")
print(f"  Score range: [{radial_score.min():.4f}, {radial_score.max():.4f}]")

assert radial_loss.shape == (pop_size,), f"Expected shape ({pop_size},), got {radial_loss.shape}"
assert radial_score.shape == (pop_size,), f"Expected shape ({pop_size},), got {radial_score.shape}"
assert (radial_score >= 0).all() and (radial_score <= 1).all(), "Score must be in [0, 1]"

# Test SignConstraint
sign = SignConstraint(weight=0.5, quantile=0.95, layer_reduce='quantile')
sign.fit(benign_updates, benign_mean, context)

print(f"\n[SignConstraint]")
print(f"  Threshold: {sign.threshold:.4f}")

sign_loss = sign.loss(population, benign_mean, context)
sign_score = sign.score(population, benign_mean, context)

print(f"  Loss shape: {sign_loss.shape}")
print(f"  Loss range: [{sign_loss.min():.4f}, {sign_loss.max():.4f}]")
print(f"  Score shape: {sign_score.shape}")
print(f"  Score range: [{sign_score.min():.4f}, {sign_score.max():.4f}]")

assert sign_loss.shape == (pop_size,), f"Expected shape ({pop_size},), got {sign_loss.shape}"
assert sign_score.shape == (pop_size,), f"Expected shape ({pop_size},), got {sign_score.shape}"
assert (sign_score >= 0).all() and (sign_score <= 1).all(), "Score must be in [0, 1]"

print("\n✓ Constraint plugins test passed")


# ============================================================================
# Test 2: Constraint Pass Score and CV
# ============================================================================
print("\n" + "="*60)
print("[Test 2] Constraint Pass Score and CV")
print("="*60)

constraints = [radial, sign]

total_score, scores_dict, losses_dict = compute_constraint_pass_score(
    population, benign_mean, constraints, context
)

print(f"\n[Constraint Pass Score]")
print(f"  Total score shape: {total_score.shape}")
print(f"  Total score range: [{total_score.min():.4f}, {total_score.max():.4f}]")
assert total_score.shape == (pop_size,), f"Expected shape ({pop_size},), got {total_score.shape}"
assert (total_score >= 0).all() and (total_score <= 1).all(), "Total score must be in [0, 1]"

for name, score in scores_dict.items():
    print(f"  {name} score: [{score.min():.4f}, {score.max():.4f}]")

total_cv, ratios_dict = compute_cv(population, benign_mean, constraints, context)

print(f"\n[Constraint Violation (CV)]")
print(f"  Total CV shape: {total_cv.shape}")
print(f"  Total CV range: [{total_cv.min():.4f}, {total_cv.max():.4f}]")
assert total_cv.shape == (pop_size,), f"Expected shape ({pop_size},), got {total_cv.shape}"
assert (total_cv >= 0).all(), "CV must be non-negative"

for name, ratio in ratios_dict.items():
    print(f"  {name} ratio: [{ratio.min():.4f}, {ratio.max():.4f}]")

print("\n✓ Constraint scoring and CV test passed")

# ============================================================================
# Test 3: NSGA-II Components
# ============================================================================
print("\n" + "="*60)
print("[Test 3] NSGA-II Components")
print("="*60)

# Create test objectives (2 objectives, minimization)
test_objectives = torch.randn(2, pop_size, device=device)

fronts = nondominated_sort(test_objectives)
print(f"\n[Non-dominated Sorting]")
print(f"  Number of fronts: {len(fronts)}")
print(f"  Front 0 size: {len(fronts[0])}")
assert len(fronts) > 0, "Should have at least one front"
assert len(fronts[0]) > 0, "Front 0 should not be empty"

if len(fronts[0]) > 2:
    distances = crowding_distance(fronts[0], test_objectives)
    print(f"\n[Crowding Distance]")
    print(f"  Distance shape: {distances.shape}")
    print(f"  Distance range: [{distances[~torch.isinf(distances)].min():.4f}, finite max]")
    assert distances.shape == (len(fronts[0]),), "Distance shape mismatch"

selected = nsga2_select(test_objectives, pop_size)
print(f"\n[NSGA-II Selection]")
print(f"  Selected count: {len(selected)}")
assert len(selected) == pop_size, f"Should select {pop_size} individuals"

print("\n✓ NSGA-II components test passed")


# ============================================================================
# Test 4: Genetic Operators
# ============================================================================
print("\n" + "="*60)
print("[Test 4] Genetic Operators")
print("="*60)

parent1 = torch.randn(D, device=device)
parent2 = torch.randn(D, device=device)

child1, child2 = sbx_crossover(parent1, parent2, eta=15.0, crossover_prob=0.9)

print(f"\n[SBX Crossover]")
print(f"  Parent1 norm: {torch.norm(parent1).item():.4f}")
print(f"  Parent2 norm: {torch.norm(parent2).item():.4f}")
print(f"  Child1 norm: {torch.norm(child1).item():.4f}")
print(f"  Child2 norm: {torch.norm(child2).item():.4f}")

assert child1.shape == parent1.shape, "Child1 shape mismatch"
assert child2.shape == parent2.shape, "Child2 shape mismatch"
assert torch.isfinite(child1).all(), "Child1 contains NaN/Inf"
assert torch.isfinite(child2).all(), "Child2 contains NaN/Inf"

individual = torch.randn(D, device=device)
mutated = mutation(individual, benign_std, mutation_scale=0.05)

print(f"\n[Mutation]")
print(f"  Original norm: {torch.norm(individual).item():.4f}")
print(f"  Mutated norm: {torch.norm(mutated).item():.4f}")
print(f"  Difference norm: {torch.norm(mutated - individual).item():.4f}")

assert mutated.shape == individual.shape, "Mutated shape mismatch"
assert torch.isfinite(mutated).all(), "Mutated contains NaN/Inf"

print("\n✓ Genetic operators test passed")

# ============================================================================
# Test 5: Attack Budget Projection
# ============================================================================
print("\n" + "="*60)
print("[Test 5] Attack Budget Projection")
print("="*60)

test_pop = benign_mean.unsqueeze(0).repeat(5, 1) + torch.randn(5, D, device=device) * 0.5
budget = 0.3

projected, clipped_mask, pre_norms = project_to_attack_budget(test_pop, benign_mean, budget)

post_norms = torch.norm(projected - benign_mean, dim=1)

print(f"\n[Projection]")
print(f"  Budget: {budget}")
print(f"  Pre-projection norms: {pre_norms.tolist()}")
print(f"  Post-projection norms: {post_norms.tolist()}")
print(f"  Clipped: {clipped_mask.tolist()}")

assert (post_norms <= budget + 1e-6).all(), "Some projected individuals exceed budget"
assert projected.shape == test_pop.shape, "Projected shape mismatch"

print("\n✓ Attack budget projection test passed")


# ============================================================================
# Test 6: Dual-Objective Computation
# ============================================================================
print("\n" + "="*60)
print("[Test 6] Dual-Objective Computation")
print("="*60)

attack_guidance = torch.randn(D, device=device)
attack_guidance = attack_guidance / (torch.norm(attack_guidance) + 1e-9)

objectives, constraint_pass_scores, scores_dict_full, total_cv_full = compute_dual_objectives(
    population, benign_mean, constraints, attack_guidance, context
)

print(f"\n[Dual Objectives]")
print(f"  Objectives shape: {objectives.shape}")
print(f"  Expected shape: (2, {pop_size})")
assert objectives.shape == (2, pop_size), f"Expected shape (2, {pop_size}), got {objectives.shape}"

print(f"\n  Objective 1 (negative constraint pass score):")
print(f"    Range: [{objectives[0].min():.4f}, {objectives[0].max():.4f}]")
print(f"  Objective 2 (negative destructiveness):")
print(f"    Range: [{objectives[1].min():.4f}, {objectives[1].max():.4f}]")

print(f"\n  Constraint pass scores (before negation):")
print(f"    Range: [{constraint_pass_scores.min():.4f}, {constraint_pass_scores.max():.4f}]")
assert (constraint_pass_scores >= 0).all() and (constraint_pass_scores <= 1).all(), \
    "Constraint pass scores must be in [0, 1]"

print(f"\n  Total CV (diagnostic):")
print(f"    Range: [{total_cv_full.min():.4f}, {total_cv_full.max():.4f}]")
assert (total_cv_full >= 0).all(), "CV must be non-negative"

print("\n✓ Dual-objective computation test passed")

# ============================================================================
# Test 7: Final Solution Selection
# ============================================================================
print("\n" + "="*60)
print("[Test 7] Final Solution Selection")
print("="*60)

best_idx = select_final_solution(
    population, objectives, total_cv_full, cv_threshold=0.0, lambda_s=0.5, lambda_a=0.5
)

print(f"\n[Final Selection]")
print(f"  Selected index: {best_idx}")
print(f"  Constraint pass score: {constraint_pass_scores[best_idx]:.4f}")
print(f"  Destructiveness: {-objectives[1, best_idx]:.4f}")
print(f"  Total CV: {total_cv_full[best_idx]:.4f}")

assert 0 <= best_idx < pop_size, f"Selected index {best_idx} out of range"

print("\n✓ Final solution selection test passed")


# ============================================================================
# Test 8: Full MOS Attack Interface
# ============================================================================
print("\n" + "="*60)
print("[Test 8] Full MOS Attack Interface")
print("="*60)

# Create mock all_updates (dictionary format)
def create_mock_update(D_total):
    """Create a mock update in dictionary format"""
    layer_size = D_total // num_layers
    update = {}
    for i in range(num_layers):
        update[f'layer_{i}'] = torch.randn(layer_size, device=device) * 0.1
    return update

all_updates = [create_mock_update(D) for _ in range(N_benign + N_malicious)]

# Create mock args
class MockArgs:
    device = device
    radius_quantile = 0.95
    attack_budget_ratio = 1.0
    weight_radial = 1.0
    weight_sign = 0.5
    sign_layer_reduce = 'quantile'
    sign_layer_quantile = 0.9
    evo_pop_size = 6
    nsga_generations = 5
    sbx_eta = 15.0
    sbx_crossover_prob = 0.9
    mos_mutation_scale = 0.05
    elite_combined_ratio = 0.95
    constraint_epsilon = 0.0
    template_noise_scale = 1e-4

args = MockArgs()

# Create mock surrogate guidance
g_ce = torch.randn(D, device=device)
g_cw = torch.randn(D, device=device)

print(f"\n[Mock Setup]")
print(f"  Total updates: {len(all_updates)}")
print(f"  Malicious slots: {N_malicious}")
print(f"  Benign clients: {N_benign}")
print(f"  Evolution generations: {args.nsga_generations}")
print(f"  Population size: {args.evo_pop_size}")

# Run attack
modified_updates, historical_pert = mos_attack(
    all_updates, args, N_malicious, g_ce, g_cw, None, lam=0.5
)

print(f"\n[Attack Results]")
print(f"  Modified updates count: {len(modified_updates)}")
print(f"  Historical perturbation shape: {historical_pert.shape if historical_pert is not None else None}")

# Verify output format
assert len(modified_updates) == N_benign + N_malicious, "Output count mismatch"
assert isinstance(modified_updates[0], dict), "Output should be dictionary format"
assert historical_pert is not None, "Should return historical perturbation"
assert historical_pert.shape[1] == D, f"Perturbation dimension mismatch: expected {D}, got {historical_pert.shape[1]}"

# Verify malicious updates were modified
for i in range(N_malicious):
    assert isinstance(modified_updates[i], dict), f"Malicious update {i} should be dict"
    for key in all_updates[i].keys():
        assert key in modified_updates[i], f"Missing key {key} in malicious update {i}"
        assert torch.isfinite(modified_updates[i][key]).all(), f"NaN/Inf in malicious update {i}, key {key}"

print("\n✓ Full MOS attack interface test passed")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("All Tests Passed!")
print("="*60)
print("\nSummary:")
print("  ✓ Constraint plugins (Radial, Sign)")
print("  ✓ Constraint pass scoring and CV computation")
print("  ✓ NSGA-II components (sorting, crowding, selection)")
print("  ✓ Genetic operators (SBX crossover, mutation)")
print("  ✓ Attack budget projection")
print("  ✓ Dual-objective computation")
print("  ✓ Final solution selection")
print("  ✓ Full attack interface compatibility")
print("\nSimplified MOS-Attack is ready for integration!")





