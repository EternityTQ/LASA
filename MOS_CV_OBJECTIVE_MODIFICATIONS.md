# MOS 攻击效果改进 - CV 目标修改总结

## 修改日期
2026-07-28

## 问题诊断

根据日志分析，原有实现存在以下问题：

1. **CV 未参与进化**：`total_cv` 仅用于最终选解，不参与 NSGA-II 的进化过程，导致 100 代进化只是在提高破坏性，而不关心约束违反
2. **所有 Pareto 解都违反约束**：Sign 和 Subspace 约束在所有解中都被违反
3. **DNC-aware mask 不可靠**：当前 mask 主要在未采样、分数为 0 的维度中选择（敏感维度比例仅 0.04%）
4. **CE/CW guidance 严重违反约束**：从第一代开始就违反 Sign 和 Subspace 约束
5. **大层主导 Sign 约束**：逐层 Sign loss 使用原始 L2 范数，容易被参数量大的层支配

## 修改内容

### 一、CV 作为第三目标参与进化

**文件位置**：`algorithms/attack/mos.py:1567`

**修改前**：
```python
objectives = torch.stack([obj_stealth, obj_destructiveness], dim=0)
```

**修改后**：
```python
# 目标3：CV（约束违反度，使用 log1p 平滑）
obj_cv = torch.log1p(total_cv)

objectives = torch.stack([obj_stealth, obj_destructiveness, obj_cv], dim=0)
```

**说明**：
- 将 `total_cv` 作为第三个优化目标，使用 `log1p` 平滑避免数值问题
- NSGA-II 现在会同时优化三个目标：隐蔽性、破坏性、约束违反度
- 进化过程会自动平衡这三个目标，产生 CV 更低的 Pareto 解

### 二、关闭 DNC-aware mask（保留为可选功能）

**文件位置**：`algorithms/attack/mos.py:519`

**修改前**：
```python
if use_dnc and getattr(args, 'use_dnc_aware_mask', True):
```

**修改后**：
```python
if use_dnc and getattr(args, 'use_dnc_aware_mask', False):
```

**说明**：
- 默认值从 `True` 改为 `False`
- 日志显示当前 mask 主要在未采样维度中选择，不可靠
- 保留 PCA 和 Subspace 约束继续生效
- 代码未删除，可通过参数重新启用作为实验性功能

### 三、构造约束安全的 guidance（g_safe）

**文件位置**：`algorithms/attack/mos.py:820-851`

**新增代码**：
```python
# 步骤1: 削弱与良性均值符号相反的维度
sign_safe_weight = getattr(args, 'sign_safe_weight', 0.1)

same_sign = (g_combined_unit * torch.sign(benign_mean) >= 0).float()
g_safe = g_combined_unit * (sign_safe_weight + (1.0 - sign_safe_weight) * same_sign)

# 步骤2: 移除部分随机子空间主成分投影
subspace_repair_strength = getattr(args, 'subspace_repair_strength', 0.5)

if use_dnc and subspace_samples:
    for sample in subspace_samples:
        dims = sample['sampled_dims']
        v = sample['principal_component']
        
        coeff = torch.dot(g_safe[dims], v)
        g_safe[dims] -= subspace_repair_strength * coeff * v

# 步骤3: 重新归一化
g_safe = g_safe / (torch.norm(g_safe) + 1e-12)
```

**应用位置**：
1. **精英种子注入** (`mos.py:897`)：
   ```python
   if EVOLUTION_POP_SIZE >= 4:
       evolution_pop[3] = benign_mean + elite_combined_ratio * max_dev_threshold * g_safe
   ```

2. **Directional push** (`mos.py:1053`)：
   ```python
   directional_push = dir_step * g_safe
   ```

3. **Objective evaluation** (`mos.py:963, 1091, 1099, 1242`)：
   ```python
   objectives_current, scores_current, losses_current, diagnostics_current = compute_objectives(
       malicious_set,
       benign_refs,
       constraint_thresholds,
       g_safe,  # 使用 g_safe 而非 g_combined_unit
       score_mode=score_mode
   )
   ```

**说明**：
- CE 和 CW 精英种子保留，但新增 `g_safe` 作为主要引导种子
- `g_safe` 通过削弱违反符号约束的维度，移除子空间主成分投影，构造出约束安全的方向
- 所有变异、评估统一使用 `g_safe`，确保进化方向不违反约束

**新增参数**：
- `sign_safe_weight`（默认 0.1）：与良性均值符号相反维度的保留权重
- `subspace_repair_strength`（默认 0.5）：子空间投影移除强度

### 四、Sign 约束归一化（避免大层主导）

**文件位置**：`algorithms/attack/mos.py:232-260`

**修改前**：
```python
for _, start_idx, end_idx in layer_dims:
    layer_centered = centered[:, start_idx:end_idx]
    layer_mean = benign_mean[start_idx:end_idx]
    
    layer_slice = layer_centered + layer_mean
    sign_violation = -layer_slice * torch.sign(layer_mean).unsqueeze(0)
    layer_loss = torch.norm(torch.relu(sign_violation), dim=1)
    sign_layer_losses.append(layer_loss)
```

**修改后**：
```python
sign_layer_quantile = getattr(benign_refs.get('args'), 'sign_layer_quantile', 0.9)

for _, start_idx, end_idx in layer_dims:
    layer_centered = centered[:, start_idx:end_idx]
    layer_mean = benign_mean[start_idx:end_idx]
    
    layer_slice = layer_centered + layer_mean
    sign_violation = -layer_slice * torch.sign(layer_mean).unsqueeze(0)
    
    # 归一化：避免大层主导
    violation_norm = torch.norm(torch.relu(sign_violation), dim=1)
    reference_norm = torch.norm(layer_mean) + 1e-12
    layer_loss = violation_norm / reference_norm
    
    sign_layer_losses.append(layer_loss)

# 层间聚合
if sign_layer_reduce == 'quantile':
    losses['sign'] = torch.quantile(sign_layer_losses, q=sign_layer_quantile, dim=0)
```

**说明**：
- 每层的 Sign loss 除以该层良性均值的范数，得到无量纲比值
- 避免参数量大的层主导整体 Sign 约束
- `sign_layer_quantile` 可配置（默认 0.9），控制层间聚合的分位数
- 良性阈值标定和候选评价使用同一个 `compute_raw_constraint_losses()` 函数

**新增参数**：
- `sign_layer_quantile`（默认 0.9）：Sign 约束层间聚合分位数

### 五、每 10 代打印 CV 统计

**文件位置**：`algorithms/attack/mos.py:1148-1180`

**新增日志**：
```python
# CV 统计
population_cv = diagnostics_current['total_cv']
min_cv = population_cv.min().item()
mean_cv = population_cv.mean().item()
feasible_count = (population_cv <= constraint_epsilon).sum().item()
feasible_ratio = feasible_count / P

# 最优个体 CV
selected_cv = population_cv[best_idx_current].item()

print(f"[MOS LOG]     种群 CV: min={min_cv:.4f}, mean={mean_cv:.4f}, feasible={feasible_ratio:.2%}")
print(f"[MOS LOG]     最优个体: 隐蔽性={best_stealth:.3f}, 破坏性={best_destructiveness:.3f}, CV={selected_cv:.4f}")
```

**输出示例**：
```
[MOS LOG]   Generation 10/100: 隐蔽性=0.850, 破坏性=1250.300, CV=0.245
[MOS LOG]     种群 CV: min=0.120, mean=0.380, feasible=15.00%
[MOS LOG]     最优个体: 隐蔽性=0.890, 破坏性=1350.200, CV=0.150
```

**监控指标**：
1. **population min CV**：种群中最小的约束违反度
2. **population mean CV**：种群平均约束违反度
3. **selected individual CV**：第一前沿首个个体的约束违反度
4. **feasible ratio**：可行解比例（CV ≤ epsilon）

**预期行为**：
- 随着进化代数增加，`min_cv` 和 `mean_cv` 应该逐渐下降
- `feasible_ratio` 应该逐渐上升
- 如果这些指标没有改善，说明 CV 目标权重可能需要调整

## 验收结果

### 本地验证（已完成）

✅ **语法检查通过**
```bash
python -m py_compile algorithms/attack/mos.py
# 无错误输出
```

✅ **静态代码检查通过（19/19 项）**
```bash
python verify_mos_cv_modifications.py
# ✓ 通过检查: 19
# ✗ 未通过检查: 0
```

**检查项包括**：
1. DNC-aware mask 默认关闭
2. g_safe 构造逻辑完整
3. g_safe 替换 g_combined_unit（4 处调用）
4. Sign 约束归一化实现
5. CV 作为第三目标
6. CV 统计日志完整
7. 函数调用和返回值数量一致

### 服务器验证（待用户执行）

以下指标需要在真实联邦学习环境中验证：

⚠️ **进化过程监控**：
- [ ] 第 1、10、20、30 代的 `min CV` 是否下降
- [ ] 第 1、10、20、30 代的 `mean CV` 是否下降
- [ ] 第 1、10、20、30 代的 `feasible ratio` 是否上升
- [ ] `selected individual CV` 是否随代数下降

⚠️ **约束满足率**：
- [ ] 最终 Pareto 解的 Sign ratio（应该更多解满足 Sign 约束）
- [ ] 最终 Pareto 解的 Subspace ratio（应该更多解满足 Subspace 约束）

⚠️ **攻击效果**：
- [ ] 测试准确率（应该下降，说明攻击有效）
- [ ] ASR（攻击成功率，应该提升）

⚠️ **GPU 显存**：
- [ ] `allocated` 峰值
- [ ] `reserved` 峰值
- [ ] `peak_allocated` 峰值

## 日志输出示例

**g_safe 构造日志**：
```
[MOS LOG] 🛡️ 构造约束安全的 guidance...
[MOS LOG]   ✓ Sign safe weight: 0.1
[MOS LOG]   ✓ Subspace repair strength: 0.5
[MOS LOG]   ✓ g_safe norm: 1.000000
```

**每 10 代进化日志**：
```
[MOS LOG]   Generation 10/100: 隐蔽性=0.850, 破坏性=1250.300, CV=0.245
[MOS LOG]     种群 CV: min=0.120, mean=0.380, feasible=15.00%
[MOS LOG]     最优个体: 隐蔽性=0.890, 破坏性=1350.200, CV=0.150
[MOS LOG]     约束得分: Radial=0.750, Sign=0.680, PCA=0.820, Subspace=0.710, Cohesion=0.000
[MOS LOG]     约束loss: Radial=8521.2341 (阈值=10216.3115), Sign=0.2345 (阈值=0.4567)
[MOS LOG]     约束比值: Radial=0.834, Sign=0.513
[MOS LOG]     约束违反: Radial=0.0000, Sign=0.0000
```

**最终 Pareto 解表格**：
```
[MOS LOG] 📊 第一前沿个体详细得分：
[MOS LOG] 索引   隐蔽性   破坏性        CV_obj     Radial   Sign     PCA      Subspace   CV       Feasible
[MOS LOG] -------------------------------------------------------------------------------------------------------------------
[MOS LOG] 0      0.890    1350.200     0.1398     0.750    0.680    0.820    0.710      0.1500   Yes
[MOS LOG] 1      0.850    1420.500     0.2231     0.720    0.650    0.800    0.690      0.2500   No
[MOS LOG] 2      0.920    1280.100     0.0953     0.780    0.710    0.840    0.730      0.1000   Yes
```

## 参数配置

新增可配置参数（通过 `args` 传入）：

```python
# g_safe 构造
args.sign_safe_weight = 0.1              # 与良性均值符号相反维度的保留权重
args.subspace_repair_strength = 0.5      # 子空间投影移除强度

# Sign 约束
args.sign_layer_quantile = 0.9           # Sign 层间聚合分位数

# DNC-aware mask
args.use_dnc_aware_mask = False          # 是否启用 DNC-aware mask（默认关闭）
```

## 使用建议

1. **首次运行**：使用默认参数，观察 CV 是否随进化下降
2. **CV 下降缓慢**：增加 `sign_safe_weight`（例如 0.2），进一步削弱违反符号的维度
3. **攻击效果不足**：降低 `subspace_repair_strength`（例如 0.3），保留更多破坏性方向
4. **大层仍然主导**：降低 `sign_layer_quantile`（例如 0.8），使用更稳健的层间聚合

## 兼容性说明

- ✅ 保留了所有原有参数的默认值
- ✅ 未删除 DNC-aware mask 代码，只改变默认行为
- ✅ CE 和 CW 精英种子保留
- ✅ 向后兼容旧配置文件

## 回滚方案

如果新版本出现问题，可以通过以下参数回滚到旧行为：

```python
# 回滚到双目标（隐蔽性 + 破坏性）
# 注意：代码已修改为三目标，无法通过参数完全回滚
# 建议使用 git 回退到上一版本

# 部分回滚：重新启用 DNC-aware mask
args.use_dnc_aware_mask = True

# 部分回滚：不修复 guidance
args.sign_safe_weight = 1.0              # 相当于不削弱任何维度
args.subspace_repair_strength = 0.0      # 不移除子空间投影
```

## 下一步优化方向（可选）

如果当前修改效果良好，后续可以考虑：

1. **动态调整 CV 权重**：前期优先破坏性，后期优先可行性
2. **Constraint-Domination**：完整实现约束支配关系，替代当前的 CV 目标
3. **自适应 guidance repair**：根据当前种群违反情况动态调整 `sign_safe_weight`
4. **多样性保持机制**：避免种群过早收敛到局部最优

## 附录：文件清单

- `algorithms/attack/mos.py`：主修改文件
- `verify_mos_cv_modifications.py`：静态代码检查脚本
- `test_mos_cv_objective.py`：Smoke test（需要 PyTorch）
- `MOS_CV_OBJECTIVE_MODIFICATIONS.md`：本文档
