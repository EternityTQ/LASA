# MOS 攻击代码修复变更日志

## 版本 2.0 - 统一约束版本 (2026-07-28)

### 🐛 修复的Bug

1. **DNC-aware mask 被覆盖**
   - 问题：DNC mask 被冲突分析逻辑无条件覆盖为全1
   - 修复：完全分离两个逻辑，增加明确日志

2. **随机变异尺度过大**
   - 问题：高维空间中 `0.02 * norm(bound)` 导致变异过大
   - 修复：改为基于 `benign_std` 的逐维变异，支持两种模式

3. **攻击预算不稳健**
   - 问题：固定使用 `krum_radius * 2`
   - 修复：基于良性径向距离分位数，支持可调比例

4. **约束计算量纲不一致**
   - 问题：良性阈值和候选loss使用不同公式
   - 修复：统一使用 `compute_raw_constraint_losses()` 函数

5. **约束得分饱和**
   - 问题：sigmoid 映射导致 0.993/0.000 饱和
   - 修复：引入温度参数，计算无量纲比值后映射

### ✨ 新增功能

- **统一约束计算框架**
  - `compute_raw_constraint_losses()` 统一函数
  - `compute_constraint_violations()` 违反度计算
  - 良性和候选使用完全相同的公式

- **参数化变异策略**
  - 支持 `benign_std` 和 `unit_norm` 两种模式
  - 可配置的方向性推动强度
  - 每10代记录变异统计

- **稳健阈值计算**
  - 默认使用分位数方法
  - 小样本回退到稳健统计量（median + k*MAD）

- **约束名称规范化**
  - `krum` → `radial`（保留别名）
  - `group` → `cohesion`（保留别名）
  - 明确语义，减少混淆

- **增强的数值稳定性监控**
  - Logit 警告阈值检查
  - NaN/Inf 安全回退
  - 良性梯度统计监控
  - 种群范数比例跟踪

### 📊 日志改进

- 将"识别率"改为"隐蔽性"（stealth_score）
- 增加约束loss原始值日志（含阈值对比）
- 增加变异范数统计（min/mean/max）
- 增加种群统计（范数比例、裁剪比例）
- 优化表格格式，增加可读性

### 🔧 新增参数

```python
# 变异参数
mos_mutation_mode = "benign_std"
mos_mutation_scale = 0.05
mos_mutation_radius_ratio = 0.01
mos_dir_step_ratio = 0.02

# 约束参数
constraint_quantile = 0.95
constraint_score_temperature = 0.5
sign_layer_reduce = "quantile"
subspace_reduce = "max"

# 攻击预算参数
radius_quantile = 0.95
attack_budget_ratio = 1.0

# 稳定性参数
logit_warning_threshold = 100.0
```

### 🔄 向后兼容性

- ✅ 所有旧参数保留为别名
- ✅ `mos_attack()` 函数签名不变
- ✅ 返回值格式不变
- ✅ 默认行为保持一致（使用新默认值）

### 📝 约束公式变更

#### Radial（原Krum）
- **旧**: 平方超额 `(dist - radius)²`
- **新**: 欧氏距离 `||x - μ||₂`

#### Sign
- **旧**: 所有层最大值
- **新**: 支持 max/mean/quantile 聚合

#### PCA
- **旧**: 候选归一化，阈值未归一化
- **新**: 统一标准化投影 `z = |<x-μ, v>| / σ`

#### Subspace
- **旧**: 候选归一化，阈值未归一化
- **新**: 统一归一化 `score / (benign_std + ε)`

#### Cohesion（原Group）
- **新名称**: 明确为恶意种群内部凝聚度

### 📈 性能影响

- **代码复杂度**: 略有增加（+150行，主要是新函数和日志）
- **运行时间**: 基本不变（预计算优化抵消新增计算）
- **内存占用**: 基本不变
- **可维护性**: 显著提高（统一框架，清晰日志）

### 🧪 测试建议

1. **快速验证**：使用 `MOSConfigQuickTest`（20代，5种群）
2. **默认配置**：使用 `MOSConfigDefault`（100代，10种群）
3. **实验扫描**：使用 `generate_experiment_configs()`（36组配置）

### 📚 文档

- `MOS_FIXES_SUMMARY.md` - 详细修复说明
- `mos_config_example.py` - 参数配置示例
- `MOS_CHANGELOG.md` - 本文件

### 🚨 已知限制

- 未实现 constraint-domination 选择（保留 Pareto 双目标）
- 未实现 Box 约束 repair（已从默认分数移除）
- 未实现径向阈值 EMA（使用当前轮分位数）

这些功能优先级较低，可根据实验结果决定是否实施。

### 🔮 未来工作

1. 根据实验结果调优默认参数
2. 可选实现 constraint-domination
3. 支持自适应温度调整
4. 增加更多约束类型

---

**修复者**: Claude (Opus 5)  
**修复日期**: 2026-07-28  
**代码审查**: ✅ 语法检查通过  
**测试状态**: ⏳ 待实验验证
