# MOS 攻击 Priority 1 改进 - 修改总结

## 📋 修改概览

本次修改针对 MOS 攻击对 DNC 防御效果有限的问题，实施了 3 个 Priority 1 级别的改进：

### ✅ 修改 1：扩展 g_ce 的覆盖范围 + 分层稀疏化

**文件**: `algorithms/attack/mos.py` - `compute_surrogate_guidance()` 函数

**问题**：
- 原先 g_ce 只覆盖分类器层（~5% 参数），导致 DNC 在其他 95% 维度上轻易检测到攻击
- DNC 随机采样 1000 维时，有 95% 概率采样到未被攻击的维度

**解决方案**：
- 移除了对卷积层梯度的冻结
- 实现分层稀疏化策略：
  - 分类器层：100% 保留（最重要）
  - 卷积层：30% 保留（战略覆盖，避免 DNC 采样盲区）
  - BN 层：10% 保留
  - 其他层：20% 保留
- 使用 TopK 保留每层梯度绝对值最大的参数

**新增超参数**：
```bash
--mos_conv_sparsity 0.3          # 卷积层稀疏度（0.0-1.0）
--mos_classifier_sparsity 1.0    # 分类器层稀疏度（0.0-1.0）
```

**预期效果**：
- 有效维度从 ~25k 增加到 ~150k
- DNC 无论采样哪 1000 维，都能看到一致的攻击模式
- 维度灾难可控（通过稀疏化）

---

### ✅ 修改 2：移除 survival_mask + 替换为 DNC-aware 掩码

**文件**: `algorithms/attack/mos.py` - `mos_attack()` 函数

**问题**：
- 原先的 `survival_mask` 假设防御方做 TopK 压缩（保留 50% 参数）
- 但 DNC 和 TR_MEAN 都不做稀疏化，这个假设是错误的

**解决方案**：
- 实现 `compute_dnc_sensitive_mask()` 函数：
  - 模拟 DNC 的随机采样 + SVD 检测过程
  - 识别 DNC 敏感的维度（在主成分上权重大的维度）
  - 创建掩码：只在 DNC **不敏感**的维度上攻击
- **智能开关**：仅在 DNC 防御启用时使用此掩码
  - 检查 `args.defend_methods` 中是否包含 `'dnc'`
  - 如果没有 DNC，回退到全 1 掩码（不限制）

**新增超参数**：
```bash
--use_dnc_aware_mask 1    # 启用 DNC-aware 掩码（0: 禁用, 1: 启用）
```

**预期效果**：
- 针对 DNC：在其零空间（低敏感度子空间）上攻击，理论上不可检测
- 针对其他防御：保持全参数攻击，不影响效果
- 自适应：根据当前轮次的良性梯度动态计算

---

### ✅ 修改 3：添加 PCA-aware 约束（DNC 专用核心武器）

**文件**: `algorithms/attack/mos.py` - `compute_losses()` 函数

**问题**：
- DNC 的检测逻辑：`score = ((grad - mu) @ v)^2`，其中 `v` 是第一主成分
- 原先只有 Krum 半径约束（球形），但 DNC 关心的是**方向性异常**（主成分投影）

**解决方案**：
- 实现 `compute_pca_constraint()` 函数：
  - 对良性梯度做 SVD 分解（和 DNC 一致）
  - 计算恶意梯度在前 5 个主成分上的投影
  - 加权求和（第一主成分权重最大）
  - 归一化（相对于良性梯度的投影范围）

- 实现 `compute_subspace_constraint()` 函数：
  - 模拟 DNC 的随机采样（多次采样 1000 维子空间）
  - 计算每个子空间上的异常分数
  - 取最大值（最坏情况），确保在任意子空间上都不被检测

**目标函数扩展**：
- 原先：5 个目标 `[l_ce, l_cw, l_magnitude, l_group, l_sign]`
- 现在：7 个目标 `[l_ce, l_cw, l_magnitude, l_group, l_sign, l_pca, l_subspace]`

**loss_mask 更新**（7 位）：
```
位 0: l_ce           - CE 攻击性
位 1: l_cw           - CW 攻击性
位 2: l_magnitude    - 幅度约束
位 3: l_group        - 群体约束
位 4: l_sign         - 符号约束
位 5: l_pca          - PCA 主成分约束（新增）
位 6: l_subspace     - 子空间鲁棒性约束（新增）
```

**新增超参数**：
```bash
--enable_subspace_constraint 1    # 启用子空间约束（0: 禁用, 1: 启用）
```

**预期效果**：
- 显式约束在 DNC 检测维度上的投影
- 子空间鲁棒性确保在任意随机采样上都隐蔽
- 只在 DNC 启用时计算，不影响其他防御的性能

---

## 🚀 使用方法

### 基础测试（针对 DNC）
```bash
python main.py \
  --dataset cifar \
  --attack mos_attack \
  --defend1 rlr \
  --defend2 tr_mean \
  --defend3 dnc \
  --num_attackers 25 \
  --gpu 4 \
  --loss_mask 1100110 \
  --mos_conv_sparsity 0.3 \
  --use_dnc_aware_mask 1 \
  --enable_subspace_constraint 1
```

### 参数说明

#### 核心参数
- `--loss_mask 1100110`: 启用 CE + CW + 幅度 + PCA + 子空间（推荐配置）
  - `1100100`: 更快收敛（只有 PCA，没有子空间）
  - `11000`: 最快（只有攻击性 + 幅度，用于快速测试）

#### 稀疏化参数
- `--mos_conv_sparsity 0.3`: 卷积层保留 30% 梯度
  - 攻击性不够时提高到 `0.5`
  - 被检测率高时降低到 `0.2`
- `--mos_classifier_sparsity 1.0`: 分类器层保留 100%（建议不变）

#### DNC 专用参数
- `--use_dnc_aware_mask 1`: 启用 DNC-aware 掩码（仅在 DNC 启用时生效）
- `--enable_subspace_constraint 1`: 启用子空间约束（计算成本较高）

---

## 📊 预期效果对比

| 指标 | 修改前 | 修改后 |
|------|--------|--------|
| **梯度覆盖率** | 5% (仅分类器) | 100% (所有层) |
| **有效维度** | ~25k | ~150k |
| **DNC 检测盲区** | 95% | 0% |
| **对 DNC 隐蔽性** | 低 | 高 |
| **优化速度** | 快 | 中等 |

---

## 🔧 调试建议

### 监控日志
运行时会输出以下关键日志：

```
[MOS LOG] Surrogate Outputs - Max Logit: X.XX, Min Logit: X.XX
[MOS LOG] DNC 模式启动 - 敏感维度占比: XX.XX%
[MOS LOG] 良性梯度范数 (Benign Mean Norm): X.XXXX
[MOS LOG] 良性梯度方差 (Benign STD Mean): X.XXXX
[MOS LOG] 最终输出的恶意扰动平均范数 (Perturbation Norm): X.XXXX
```

### 调优步骤

1. **第一步**：测试覆盖率提升
   - 使用 `--loss_mask 11000`（只保留攻击性 + 幅度）
   - 观察攻击效果是否提升

2. **第二步**：添加 DNC-aware 掩码
   - 使用 `--use_dnc_aware_mask 1`
   - 观察 `[MOS LOG] DNC 敏感维度占比`，应该在 40-60% 之间

3. **第三步**：添加 PCA 约束
   - 使用 `--loss_mask 1100100`（添加 PCA）
   - 观察 DNC 检测率是否下降

4. **第四步**：添加子空间约束（可选）
   - 使用 `--loss_mask 1100110`（添加子空间）
   - 注意：计算成本增加约 30%

### 常见问题

**Q: 如果攻击性下降怎么办？**
A: 提高 `--mos_conv_sparsity` 到 0.5 或更高，或减少 loss_mask 中的约束项

**Q: 如果 DNC 检测率仍然很高怎么办？**
A: 
1. 确认 `use_dnc_aware_mask=1` 已启用
2. 降低 `mos_conv_sparsity` 到 0.2
3. 启用 `enable_subspace_constraint=1`

**Q: 如果优化速度太慢怎么办？**
A:
1. 降低 `mos_conv_sparsity` 到 0.2（减少有效维度）
2. 使用 `--loss_mask 11000` 或 `1100100`（减少目标数）
3. 禁用子空间约束 `--enable_subspace_constraint 0`

**Q: 对其他防御（非 DNC）效果如何？**
A: 
- 修改 1（分层覆盖）对所有防御都有提升
- 修改 2 和 3 只在 DNC 启用时生效，不影响其他防御

---

## 📁 修改的文件

1. `algorithms/attack/mos.py` - 核心攻击逻辑
   - `compute_surrogate_guidance()`: 分层稀疏化
   - `mos_attack()`: DNC-aware 掩码
   - `compute_losses()`: PCA 和子空间约束

2. `algorithms/engine/fedavg_all.py` - 调用接口
   - 导入 `mos.py` 中的函数
   - 传递 `args` 参数

3. `main.py` - 参数解析
   - 添加 4 个新超参数

---

## 🎯 后续优化方向

如果效果仍不理想，可以考虑：

1. **模拟防御链级联**（Priority 2）
   - 在 `compute_losses` 中模拟 RLR + TR_MEAN 的预处理
   - 优化对 DNC 实际输入的攻击

2. **两阶段优化**（Priority 3）
   - Stage 1: 最大化攻击性
   - Stage 2: 投影到 DNC 零空间

3. **动态 Krum 半径估计**（Priority 3）
   - 用 99 分位数代替 max
   - 或用 median + 2*MAD

---

## 📝 版本信息

- 修改日期：2026-07-02
- 修改人：Claude (Opus 4.8)
- 优先级：Priority 1（必须修改）
- 状态：已完成，待测试
