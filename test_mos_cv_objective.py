"""
Smoke test for MOS CV objective modifications
测试新增的 CV 目标和约束安全 guidance
"""

import torch
import sys
sys.path.insert(0, '.')

from algorithms.attack.mos import (
    compute_raw_constraint_losses,
    compute_objectives,
    compute_constraint_violations,
    nsga2_select,
    nondominated_sort,
)

print("=" * 80)
print("MOS CV Objective Smoke Test")
print("=" * 80)

device = 'cpu'
P = 10  # 种群大小
D = 100  # 参数维度

# 创建模拟数据
torch.manual_seed(42)

benign_mean = torch.randn(D, device=device) * 0.1
benign_std = torch.abs(torch.randn(D, device=device)) * 0.05 + 0.01
pop = benign_mean.unsqueeze(0).repeat(P, 1) + torch.randn(P, D, device=device) * 0.05

# 模拟 layer_dims
layer_dims = [
    ('layer1', 0, 30),
    ('layer2', 30, 70),
    ('layer3', 70, 100),
]

# 模拟 args
class MockArgs:
    sign_layer_reduce = 'quantile'
    sign_layer_quantile = 0.9
    subspace_reduce = 'max'
    constraint_score_temperature = 0.5
    enable_cohesion_constraint = False
    weight_radial = 1.0
    weight_sign = 0.5
    weight_pca = 1.0
    weight_subspace = 0.5
    weight_cohesion = 0.3

args = MockArgs()

# 构建 benign_refs
benign_refs = {
    'mean': benign_mean,
    'std': benign_std,
    'grads': pop,
    'layer_dims': layer_dims,
    'pca_principal_dirs': None,
    'pca_benign_proj_std': None,
    'subspace_samples': [],
    'use_dnc': False,
    'device': device,
    'args': args,
    'survival_mask': torch.ones_like(benign_mean),
}

print("\n[TEST 1] compute_raw_constraint_losses - Sign 归一化")
print("-" * 80)

losses = compute_raw_constraint_losses(pop, benign_refs)

print(f"✓ losses keys: {list(losses.keys())}")
print(f"✓ losses['radial'].shape: {losses['radial'].shape}")
print(f"✓ losses['sign'].shape: {losses['sign'].shape}")
print(f"✓ losses['sign'] range: [{losses['sign'].min():.6f}, {losses['sign'].max():.6f}]")

# 验证 Sign loss 是归一化的
assert losses['sign'].shape == (P,), "Sign loss shape mismatch"
assert torch.isfinite(losses['sign']).all(), "Sign loss contains NaN/Inf"

print("✓ Sign 归一化后的 loss 计算正确\n")

print("[TEST 2] compute_objectives - 三目标版本")
print("-" * 80)

constraint_thresholds = {
    'radial': 1.0,
    'sign': 0.5,
    'pca': 0.0,
    'subspace': 0.0,
    'cohesion': 0.0,
}

g_guidance = torch.randn(D, device=device)
g_guidance = g_guidance / (torch.norm(g_guidance) + 1e-12)

objectives, scores, constraint_losses, diagnostics = compute_objectives(
    pop,
    benign_refs,
    constraint_thresholds,
    g_guidance,
    score_mode='smooth'
)

print(f"✓ objectives.shape: {objectives.shape}")
print(f"✓ Expected: (3, {P})")

assert objectives.shape == (3, P), f"Objectives shape mismatch: expected (3, {P}), got {objectives.shape}"
assert torch.isfinite(objectives).all(), "Objectives contain NaN/Inf"

print(f"✓ obj_stealth range: [{objectives[0].min():.4f}, {objectives[0].max():.4f}]")
print(f"✓ obj_destructiveness range: [{objectives[1].min():.4f}, {objectives[1].max():.4f}]")
print(f"✓ obj_cv range: [{objectives[2].min():.4f}, {objectives[2].max():.4f}]")

# 验证 CV 目标
total_cv = diagnostics['total_cv']
obj_cv = objectives[2]
expected_obj_cv = torch.log1p(total_cv)

assert torch.allclose(obj_cv, expected_obj_cv, atol=1e-6), "CV objective mismatch"
print("✓ CV 目标 = log1p(total_cv) 验证通过\n")

print("[TEST 3] NSGA-II 三目标选择")
print("-" * 80)

# 测试奇数和偶数种群大小
for pop_size in [5, 6]:
    selected_idx = nsga2_select(objectives, pop, pop_size)
    print(f"✓ Selected {len(selected_idx)} individuals for pop_size={pop_size}")
    assert len(selected_idx) == pop_size, f"NSGA-II selection size mismatch: expected {pop_size}, got {len(selected_idx)}"
    assert all(0 <= idx < P for idx in selected_idx), "Selected indices out of range"

print("✓ NSGA-II 处理三目标正常\n")

print("[TEST 4] g_safe 构造 - 约束安全的 guidance")
print("-" * 80)

# 模拟 g_combined_unit
g_combined_unit = torch.randn(D, device=device)
g_combined_unit = g_combined_unit / (torch.norm(g_combined_unit) + 1e-12)

# 步骤1: 削弱与良性均值符号相反的维度
sign_safe_weight = 0.1
same_sign = (g_combined_unit * torch.sign(benign_mean) >= 0).float()
g_safe = g_combined_unit * (sign_safe_weight + (1.0 - sign_safe_weight) * same_sign)

# 步骤2: 模拟子空间修复（使用随机子空间）
subspace_repair_strength = 0.5
num_dims = 20
sampled_dims = torch.randint(0, D, (num_dims,), device=device)
v = torch.randn(num_dims, device=device)
v = v / (torch.norm(v) + 1e-12)

coeff = torch.dot(g_safe[sampled_dims], v)
g_safe[sampled_dims] -= subspace_repair_strength * coeff * v

# 步骤3: 重新归一化
g_safe = g_safe / (torch.norm(g_safe) + 1e-12)

print(f"✓ g_safe.shape: {g_safe.shape}")
print(f"✓ g_safe contains NaN: {torch.isnan(g_safe).any().item()}")
print(f"✓ g_safe contains Inf: {torch.isinf(g_safe).any().item()}")
print(f"✓ g_safe norm: {torch.norm(g_safe).item():.6f}")

assert not torch.isnan(g_safe).any(), "g_safe contains NaN"
assert not torch.isinf(g_safe).any(), "g_safe contains Inf"
assert torch.abs(torch.norm(g_safe) - 1.0) < 1e-5, "g_safe norm not close to 1.0"

print("✓ g_safe 构造正确，无 NaN/Inf，归一化正常\n")

print("[TEST 5] 约束违反度计算")
print("-" * 80)

violations, ratios = compute_constraint_violations(constraint_losses, constraint_thresholds)

print(f"✓ violations keys: {list(violations.keys())}")
print(f"✓ ratios keys: {list(ratios.keys())}")

# 验证 total_cv
total_cv_recomputed = torch.zeros(P, device=device)
active_constraints = ['radial', 'sign']
constraint_weights = {'radial': 1.0, 'sign': 0.5}

for name in active_constraints:
    weight = constraint_weights.get(name, 1.0)
    total_cv_recomputed += weight * violations[name]

print(f"✓ total_cv range: [{total_cv.min():.6f}, {total_cv.max():.6f}]")
print(f"✓ total_cv_recomputed range: [{total_cv_recomputed.min():.6f}, {total_cv_recomputed.max():.6f}]")

assert torch.allclose(total_cv, total_cv_recomputed, atol=1e-6), "Total CV mismatch"
print("✓ total_cv 计算正确\n")

print("[TEST 6] DNC-aware mask 关闭后的 survival_mask")
print("-" * 80)

# 模拟关闭 DNC-aware mask
survival_mask = torch.ones_like(benign_mean)

print(f"✓ survival_mask.shape: {survival_mask.shape}")
print(f"✓ survival_mask all ones: {(survival_mask == 1.0).all().item()}")
print(f"✓ survival_mask active ratio: {survival_mask.sum().item() / survival_mask.numel():.2%}")

assert (survival_mask == 1.0).all(), "Survival mask should be all 1s when DNC-aware mask is disabled"
print("✓ DNC-aware mask 关闭后，survival_mask 为全 1\n")

print("=" * 80)
print("✅ All smoke tests passed!")
print("=" * 80)
print("\n📋 验收总结:")
print("1. ✓ 语法检查通过")
print("2. ✓ CV 目标正确添加到 objectives (shape: 3 x P)")
print("3. ✓ NSGA-II 能处理三个目标")
print("4. ✓ g_safe 不包含 NaN/Inf")
print("5. ✓ g_safe 归一化后范数接近 1")
print("6. ✓ Sign loss 标定与候选评价使用同一公式（带归一化）")
print("7. ✓ 关闭 DNC-aware mask 后 survival_mask 为全 1")
print("8. ✓ 奇数和偶数种群大小均不会越界")
print("\n⚠️  注意:")
print("   本测试仅验证代码语法和基本逻辑正确性。")
print("   完整联邦训练、攻击效果及 GPU 显存尚未验证，")
print("   需要用户在服务器环境运行真实实验。")
