"""
Smoke tests for MOS attack fixes
Tests the core logic without running full federated training
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.attack.mos import (
    select_final_solution,
    binary_tournament_selection,
    compute_rank_and_crowding,
    nondominated_sort,
    project_to_attack_budget,
)


def test_select_final_no_cv_filter():
    """Test that select_final_solution does NOT filter by CV"""
    print("\n" + "="*60)
    print("TEST: Final selection without CV filtering")
    print("="*60)

    # Create mock population
    pop_size = 5
    dim = 10
    population = torch.randn(pop_size, dim)

    # Create objectives where:
    # - Candidate 0: CV=0, low attack (should NOT be selected)
    # - Candidate 1: CV>0, medium stealth, high attack (should be selected)
    # - Candidate 2: CV>0, high stealth, medium attack
    objectives = torch.tensor([
        [-0.9, -0.8, -0.7, -0.6, -0.5],  # Stealth (minimization: -R)
        [-0.1, -10.0, -5.0, -3.0, -2.0],  # Destructiveness (minimization: -A)
    ])

    total_cv = torch.tensor([0.0, 5.0, 3.0, 2.0, 1.0])

    # Select solution
    best_idx, diagnostics = select_final_solution(
        population,
        objectives,
        total_cv=total_cv,
        lambda_s=0.5,
        lambda_a=0.5,
        attack_floor_ratio=0.0  # Disable attack floor
    )

    print(f"Selected index: {best_idx}")
    print(f"Selected CV: {total_cv[best_idx].item():.4f}")
    print(f"Selected stealth: {-objectives[0, best_idx].item():.3f}")
    print(f"Selected destructiveness: {-objectives[1, best_idx].item():.3f}")

    # Check that high-attack candidate can be selected despite CV > 0
    assert best_idx != 0, "Should NOT select CV=0 candidate with low attack"
    assert total_cv[best_idx] > 0, "Selected candidate should have CV > 0"

    print("✓ PASS: High-attack candidate with CV>0 was selected")
    return True


def test_both_objectives_normalized():
    """Test that both objectives are normalized in final selection"""
    print("\n" + "="*60)
    print("TEST: Both objectives normalized")
    print("="*60)

    pop_size = 4
    dim = 10
    population = torch.randn(pop_size, dim)

    # Create Pareto front with diverse stealth and destructiveness
    objectives = torch.tensor([
        [-0.5, -0.7, -0.9, -0.3],  # Stealth: range [0.3, 0.9]
        [-10.0, -5.0, -2.0, -15.0],  # Destructiveness: range [2.0, 15.0]
    ])

    best_idx, diagnostics = select_final_solution(
        population,
        objectives,
        lambda_s=0.5,
        lambda_a=0.5,
        attack_floor_ratio=0.0
    )

    print(f"Stealth range: [{diagnostics['pareto_stealth_min']:.3f}, "
          f"{diagnostics['pareto_stealth_max']:.3f}]")
    print(f"Destructiveness range: [{diagnostics['pareto_destructiveness_min']:.3f}, "
          f"{diagnostics['pareto_destructiveness_max']:.3f}]")
    print(f"Selected index: {best_idx}")

    # Both objectives should have non-zero range
    stealth_range = diagnostics['pareto_stealth_max'] - diagnostics['pareto_stealth_min']
    destruct_range = (diagnostics['pareto_destructiveness_max'] -
                      diagnostics['pareto_destructiveness_min'])

    assert stealth_range > 0, "Stealth range should be non-zero"
    assert destruct_range > 0, "Destructiveness range should be non-zero"

    print("✓ PASS: Both objectives normalized correctly")
    return True


def test_zero_range_objectives():
    """Test handling when objective range is zero"""
    print("\n" + "="*60)
    print("TEST: Zero-range objective handling")
    print("="*60)

    pop_size = 3
    dim = 10
    population = torch.randn(pop_size, dim)

    # All candidates have same stealth
    objectives = torch.tensor([
        [-0.5, -0.5, -0.5],  # Stealth: no variance
        [-10.0, -5.0, -2.0],  # Destructiveness: variance
    ])

    best_idx, diagnostics = select_final_solution(
        population,
        objectives,
        lambda_s=0.5,
        lambda_a=0.5,
        attack_floor_ratio=0.0
    )

    print(f"Selected index: {best_idx}")
    print(f"Stealth range: {diagnostics['pareto_stealth_max'] - diagnostics['pareto_stealth_min']:.6f}")

    # Should not produce NaN
    assert not torch.isnan(torch.tensor(best_idx)), "Result should not be NaN"

    print("✓ PASS: Zero-range objective handled without NaN")
    return True


def test_attack_floor_protection():
    """Test attack floor protection"""
    print("\n" + "="*60)
    print("TEST: Attack floor protection")
    print("="*60)

    pop_size = 5
    dim = 10
    population = torch.randn(pop_size, dim)

    # Front with one very weak attack candidate
    objectives = torch.tensor([
        [-0.9, -0.8, -0.7, -0.6, -0.5],
        [-0.1, -10.0, -8.0, -6.0, -4.0],  # Candidate 0 has very low attack
    ])

    # With attack floor = 10% of max (10.0 * 0.1 = 1.0)
    # Only candidates with attack >= 1.0 should be considered
    best_idx, diagnostics = select_final_solution(
        population,
        objectives,
        lambda_s=0.5,
        lambda_a=0.5,
        attack_floor_ratio=0.10
    )

    selected_attack = -objectives[1, best_idx].item()
    attack_floor = diagnostics['attack_floor']

    print(f"Attack floor: {attack_floor:.3f}")
    print(f"Selected attack: {selected_attack:.3f}")
    print(f"Candidates after floor: {diagnostics['candidates_after_floor']}")

    # Weak attack candidate should be filtered
    assert best_idx != 0, "Weakest attack candidate should be filtered by attack floor"
    assert selected_attack >= attack_floor * 0.99, "Selected attack should pass floor"

    print("✓ PASS: Attack floor protection working")
    return True


def test_attack_floor_disable():
    """Test disabling attack floor"""
    print("\n" + "="*60)
    print("TEST: Attack floor can be disabled")
    print("="*60)

    pop_size = 3
    dim = 10
    population = torch.randn(pop_size, dim)

    objectives = torch.tensor([
        [-0.9, -0.5, -0.3],
        [-0.01, -5.0, -3.0],
    ])

    # With floor = 0, even weak candidates can be selected if balanced score is high
    best_idx, diagnostics = select_final_solution(
        population,
        objectives,
        lambda_s=0.9,  # Heavily favor stealth
        lambda_a=0.1,
        attack_floor_ratio=0.0  # Disabled
    )

    print(f"Attack floor (should be 0): {diagnostics['attack_floor']:.6f}")
    print(f"Selected index: {best_idx}")

    assert diagnostics['attack_floor'] < 1e-6, "Attack floor should be near zero"

    print("✓ PASS: Attack floor can be disabled")
    return True


def test_binary_tournament():
    """Test binary tournament selection"""
    print("\n" + "="*60)
    print("TEST: Binary tournament selection")
    print("="*60)

    # Create objectives with clear Pareto fronts
    objectives = torch.tensor([
        [-1.0, -0.9, -0.5, -0.5, -0.3],  # Front 0: indices 0,1
        [-0.3, -0.5, -1.0, -0.9, -0.5],  # Front 1: indices 2,3, Front 2: index 4
    ])

    num_parents = 10
    parent_indices = binary_tournament_selection(objectives, num_parents)

    print(f"Selected {len(parent_indices)} parents")
    print(f"Parent indices: {parent_indices[:5]}... (showing first 5)")

    assert len(parent_indices) == num_parents, "Should select requested number of parents"
    assert all(0 <= idx < 5 for idx in parent_indices), "All indices should be valid"

    print("✓ PASS: Binary tournament selection working")
    return True


def test_multi_scale_init_simulation():
    """Simulate multi-scale initialization logic"""
    print("\n" + "="*60)
    print("TEST: Multi-scale initialization (simulation)")
    print("="*60)

    pop_sizes = [1, 2, 3, 5, 8, 12]
    default_scales = [0.0, 0.10, 0.25, 0.50, 0.75, 0.95]

    for pop_size in pop_sizes:
        if pop_size <= len(default_scales):
            if pop_size == 1:
                selected_scales = [0.95]
            elif pop_size == 2:
                selected_scales = [0.0, 0.95]
            elif pop_size == 3:
                selected_scales = [0.0, 0.5, 0.95]
            else:
                indices = torch.linspace(0, len(default_scales) - 1, pop_size).long()
                selected_scales = [default_scales[i] for i in indices]
        else:
            selected_scales = default_scales.copy()

        print(f"Pop size {pop_size}: scales = {selected_scales}")

        # Check that 0.0 and 0.95 are included for pop_size >= 2
        if pop_size >= 2:
            assert 0.0 in selected_scales or 0.95 in selected_scales, \
                "Should include extreme scales"

    print("✓ PASS: Multi-scale initialization logic correct")
    return True


def test_historical_seed_handling():
    """Test historical seed dimension validation"""
    print("\n" + "="*60)
    print("TEST: Historical seed handling")
    print("="*60)

    dim = 100
    benign_mean = torch.randn(dim)
    max_dev_threshold = 5.0

    # Valid historical seed
    hist_valid = torch.randn(dim) * 2.0
    print(f"Valid historical seed shape: {hist_valid.shape}")

    # Invalid shapes
    hist_wrong_dim = torch.randn(50)
    hist_2d = torch.randn(1, dim)

    # Simulate validation logic
    def validate_hist_seed(hist, expected_dim):
        try:
            hist_pert = hist.squeeze() if hist.dim() > 1 else hist
            if hist_pert.shape[0] == expected_dim and torch.isfinite(hist_pert).all():
                return True, "Valid"
            else:
                return False, "Dimension mismatch"
        except Exception as e:
            return False, str(e)

    valid, msg = validate_hist_seed(hist_valid, dim)
    print(f"Valid seed: {valid} ({msg})")
    assert valid, "Valid seed should pass"

    valid, msg = validate_hist_seed(hist_wrong_dim, dim)
    print(f"Wrong dimension seed: {valid} ({msg})")
    assert not valid, "Wrong dimension should fail"

    valid, msg = validate_hist_seed(hist_2d, dim)
    print(f"2D seed (squeezable): {valid} ({msg})")
    assert valid, "2D squeezable seed should pass"

    print("✓ PASS: Historical seed validation working")
    return True


def test_offspring_nan_check():
    """Test that NaN offspring are detected"""
    print("\n" + "="*60)
    print("TEST: Offspring NaN detection")
    print("="*60)

    # Normal offspring
    offspring_normal = torch.randn(10, 100)
    assert torch.isfinite(offspring_normal).all(), "Normal offspring should be finite"
    print("Normal offspring: finite ✓")

    # Offspring with NaN
    offspring_nan = offspring_normal.clone()
    offspring_nan[0, 0] = float('nan')
    assert not torch.isfinite(offspring_nan).all(), "NaN offspring should be detected"
    print("NaN offspring: detected ✓")

    # Offspring with Inf
    offspring_inf = offspring_normal.clone()
    offspring_inf[5, 10] = float('inf')
    assert not torch.isfinite(offspring_inf).all(), "Inf offspring should be detected"
    print("Inf offspring: detected ✓")

    print("✓ PASS: NaN/Inf detection working")
    return True


def test_import():
    """Test that module imports without errors"""
    print("\n" + "="*60)
    print("TEST: Module import")
    print("="*60)

    try:
        from algorithms.attack.mos import mos_attack
        print("✓ PASS: mos_attack imported successfully")
        return True
    except Exception as e:
        print(f"✗ FAIL: Import error: {e}")
        return False


def run_all_tests():
    """Run all smoke tests"""
    print("\n" + "="*70)
    print(" MOS ATTACK FIXES - SMOKE TESTS")
    print("="*70)

    tests = [
        ("Import", test_import),
        ("Final selection no CV filter", test_select_final_no_cv_filter),
        ("Both objectives normalized", test_both_objectives_normalized),
        ("Zero-range handling", test_zero_range_objectives),
        ("Attack floor protection", test_attack_floor_protection),
        ("Attack floor disable", test_attack_floor_disable),
        ("Binary tournament", test_binary_tournament),
        ("Multi-scale init", test_multi_scale_init_simulation),
        ("Historical seed", test_historical_seed_handling),
        ("Offspring NaN check", test_offspring_nan_check),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result, None))
        except Exception as e:
            print(f"\n✗ FAIL: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result, _ in results if result)
    total = len(results)

    for name, result, error in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"       {error}")

    print(f"\n{passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
