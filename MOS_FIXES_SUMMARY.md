# MOS 攻击代码修复总结

修复日期：2026-07-28
修复文件：`algorithms/attack/mos.py`

---

## 一、修复的Bug列表

### 1. ✅ DNC-aware mask 被覆盖 (修复 1/12)
**问题**：DNC-aware survival_mask 在计算后被冲突分析的 else 分支无条件覆盖为全1。

**修复**：
- 将 DNC mask 计算和冲突分析完全分离为两个独立的 if 块
- DNC mask 逻辑不再被后续代码修改
- 增加明确日志：`DNC-aware mask enabled: True/False` 和 `survival mask active ratio`

**位置**：第 295-338 行


### 2. ✅ 随机变异尺度过大 (修复 2/12 & 4/12)
**问题**：使用 `0.02 * torch.norm(upper_bound - lower_bound)` 导致高维空间中变异范数过大。

**修复**：
- 引入 `mos_mutation_mode` 参数，支持两种模式：
  - `'benign_std'`（默认）：逐维变异 `noise = randn * benign_std * mutation_scale`
  - `'unit_norm'`：固定范数随机方向 `noise = unit_vector * radius`
- 方向性推动改为参数化：`dir_step = dir_step_ratio * smoothed_radial_threshold`
- 每10代记录变异范数统计（min/mean/max）

**新增参数**：
```python
mos_mutation_mode = "benign_std"  # 或 "unit_norm"
mos_mutation_scale = 0.05
mos_mutation_radius_ratio = 0.01
mos_dir_step_ratio = 0.02
```

**位置**：第 609-631 行（配置），第 717-762 行（变异实现）


### 3. ✅ 攻击预算改为稳健、可调 (修复 3/12)
**问题**：使用固定的 `krum_radius * 2` 作为攻击半径，不稳健且不可调。

**修复**：
- 使用良性径向距离的分位数：`smoothed_radial_threshold = quantile(benign_dists, q=0.95)`
- 引入攻击预算比例：`max_dev_threshold = attack_budget_ratio * smoothed_radial_threshold`
- 增加日志：base radial threshold、attack budget ratio、max deviation threshold

**新增参数**：
```python
radius_quantile = 0.95
attack_budget_ratio = 1.0
```

**位置**：第 833-850 行


### 4. ✅ 统一约束计算函数 (修复 5/12 & 7/12)
**问题**：良性阈值和候选loss使用不同公式，导致量纲不一致。

**修复**：
- 新增 `compute_raw_constraint_losses(pop, benign_refs)` 统一函数
- 良性更新和候选更新都调用同一函数
- 所有约束使用相同的归一化方式

**新函数**：
- `compute_raw_constraint_losses()` - 第 133-257 行
- 支持的约束：radial, sign, pca, subspace, cohesion

**位置**：第 133-257 行（新函数），第 652-710 行（阈值计算）


### 5. ✅ 约束名称规范化 (修复 5/12)
**修改**：
- `krum` → `radial`（保留 `krum` 作为别名）
- `group` → `cohesion`（保留 `group` 作为别名）
- 注释明确说明 radial 是径向距离代理，而非完整 Krum score

**位置**：遍布全文（164-167行、703-704行等）


### 6. ✅ Sign 约束层间聚合 (修复 5/12)
**问题**：使用所有层的最大loss，容易被超大层主导。

**修复**：
- 新增 `sign_layer_reduce` 参数
- 支持三种模式：`'max'`、`'mean'`、`'quantile'`（默认0.9分位数）

**新增参数**：
```python
sign_layer_reduce = "quantile"  # 或 "max", "mean"
```

**位置**：第 172-193 行


### 7. ✅ PCA 约束标准化 (修复 5/12)
**问题**：候选loss除以良性投影标准差，但阈值使用未归一化投影。

**修复**：
- 统一为标准化形式：`z_j = |<x-μ, v_j>| / (σ_j + ε)`
- 良性阈值和候选计算完全一致

**位置**：第 199-217 行


### 8. ✅ Subspace 约束标准化 (修复 5/12)
**问题**：候选使用归一化，良性阈值未归一化。

**修复**：
- 统一为：`normalized_score = raw_score / (benign_std + ε)`
- 新增 `subspace_reduce` 参数：`'max'` 或 `'mean'`
- 支持固定随机种子（通过预计算实现）

**新增参数**：
```python
subspace_reduce = "max"  # 或 "mean"
```

**位置**：第 219-246 行


### 9. ✅ 重构约束得分映射 (修复 6/12)
**问题**：sigmoid 曲线过陡，导致 0.993/0.000 饱和。

**修复**：
- 先计算无量纲比值：`r = loss / threshold`
- 使用温度参数：`score = sigmoid((1 - r) / temperature)`
- 默认 `temperature = 0.5`，避免饱和

**新增参数**：
```python
constraint_score_temperature = 0.5
```

**新增函数**：
- `compute_constraint_score(loss, threshold, mode, temperature)` - 第 264-308 行
- `compute_constraint_violations(losses, thresholds)` - 第 310-331 行

**位置**：第 260-331 行


### 10. ✅ 阈值计算改为分位数 (修复 7/12)
**问题**：使用 `mean + k*std` 对小样本不稳健。

**修复**：
- 默认使用分位数：`threshold = quantile(benign_losses, q=0.95)`
- 样本不足时回退到稳健统计量：`median + k * MAD`

**新增参数**：
```python
constraint_quantile = 0.95
```

**位置**：第 652-710 行


### 11. ✅ compute_objectives 重构 (修复 8/12)
**修复**：
- 完全使用新的统一约束计算函数
- 计算约束违反度和比值
- 将"识别率"改名为"隐蔽性" (stealth_score)
- 增加温度参数控制

**位置**：第 1226-1330 行


### 12. ✅ 数值稳定性监控 (修复 9/12 & 10/12)
**修复**：
- `compute_surrogate_guidance` 中增加 logit 警告阈值检查
- 检测 NaN/Inf 并返回安全回退
- 主函数中记录良性梯度统计（范数、标准差）
- 每10代记录种群范数比例和投影裁剪比例

**新增参数**：
```python
logit_warning_threshold = 100.0
```

**位置**：
- compute_surrogate_guidance: 第 14-44 行
- 主函数监控: 第 448-465 行
- 进化循环监控: 第 983-1009 行

---

## 二、新增参数及默认值

```python
# 变异参数
mos_mutation_mode = "benign_std"        # 变异模式
mos_mutation_scale = 0.05               # benign_std 模式的尺度
mos_mutation_radius_ratio = 0.01        # unit_norm 模式的半径比例
mos_dir_step_ratio = 0.02               # 方向性推动比例

# 约束阈值参数
constraint_quantile = 0.95              # 阈值分位数
constraint_score_temperature = 0.5      # 得分映射温度

# 攻击预算参数
radius_quantile = 0.95                  # 径向阈值分位数
attack_budget_ratio = 1.0               # 攻击预算比例

# 约束聚合参数
sign_layer_reduce = "quantile"          # Sign 层间聚合：max/mean/quantile
subspace_reduce = "max"                 # Subspace 聚合：max/mean

# DNC 参数
use_dnc_aware_mask = True               # 是否启用 DNC-aware mask

# 数值稳定性参数
logit_warning_threshold = 100.0         # Logit 警告阈值
```

---

## 三、约束公式修改

### 3.1 Radial 约束（原Krum）
```
L_radial(x) = ||x - μ_benign||_2
```
- 良性和候选统一使用欧氏距离
- 不再使用平方超额

### 3.2 Sign 约束
```
L_sign(x) = aggregate({||ReLU(-x_l ⊙ sign(μ_l))||_2}_l)
```
- 层间聚合支持 max/mean/quantile
- 良性和候选完全一致

### 3.3 PCA 约束
```
z_j(x) = |<x - μ, v_j>| / (σ_j + ε)
L_PCA(x) = Σ_j w_j * z_j(x)
```
- 统一使用标准化投影
- 权重递减：[1.0, 0.5, 0.3, 0.2, 0.1]

### 3.4 Subspace 约束
```
L_subspace(x) = aggregate({|<x_S - μ_S, v_S>| / (σ_S + ε)}_S)
```
- 所有子空间统一归一化
- 聚合支持 max/mean

### 3.5 Cohesion 约束（原Group）
```
L_cohesion(x_i) = ||x_i - μ_malicious||_2
```
- 表示恶意种群内部凝聚度
- 不等价于防御生存率

---

## 四、约束得分映射

### 旧版（饱和）
```python
k = 5.0 / threshold
score = sigmoid(-k * (loss - threshold))
```
结果：loss=0 → 0.993, loss=T → 0.5, loss=2T → 0.0067

### 新版（平滑）
```python
ratio = loss / (threshold + 1e-12)
score = sigmoid((1.0 - ratio) / temperature)
```
结果：更平滑，避免饱和

---

## 五、日志改进

### 5.1 新增日志项

**初始化阶段**：
```
DNC-aware mask enabled: True/False
survival mask active ratio: xx.xx%
Mutation configuration: mode, scale, directional step ratio
Attack budget configuration: base threshold, ratio, max deviation
约束阈值（分位数 q=0.95）: Radial, Sign, PCA, Subspace, Cohesion
良性梯度统计: 范数, 标准差 min/mean/max
```

**进化过程中（每10代）**：
```
隐蔽性 (stealth_score)、破坏性
约束得分: Radial, Sign, PCA, Subspace, Cohesion
约束loss: Radial, Sign（含阈值对比）
变异范数: min/mean/max
方向推动范数
种群范数比例: min/mean/max
投影裁剪比例
```

**最终输出**：
```
第一前沿个体详细得分（表格）
隐蔽性、破坏性、各约束得分
最优解选择：隐蔽性≥0.5，最接近理想点
```

### 5.2 术语更新
- "识别率" → "隐蔽性" (stealth_score)
- "Krum" → "Radial"（日志中保留Krum标签以兼容）
- "Group" → "Cohesion"

---

## 六、实验兼容性保证

### 6.1 保持的接口
- `mos_attack()` 主函数签名不变
- `vector_to_net_dict()` 使用方式不变
- 所有旧参数保留为兼容别名
- 返回值格式不变

### 6.2 向后兼容
- `'krum'` 约束名自动映射到 `'radial'`
- `'group'` 约束名自动映射到 `'cohesion'`
- 旧的 `compute_constraint_score_legacy` 保留但标注为废弃
- 所有新参数通过 `getattr(args, ..., default)` 提供默认值

### 6.3 不修改的部分
- 项目中其他攻击、防御代码
- 训练循环逻辑
- 数据加载流程
- 评估指标计算

---

## 七、建议实验配置

### 7.1 最小实验矩阵

**变量1：attack_budget_ratio**
```
0.75, 1.0, 1.25, 1.5
```

**变量2：constraint_score_temperature**
```
0.25, 0.5, 1.0
```

**变量3：mos_mutation_scale**
```
0.01, 0.05, 0.1
```

总计：4 × 3 × 3 = 36 组实验

### 7.2 控制变量实验

**基线配置**：
```python
attack_budget_ratio = 1.0
constraint_score_temperature = 0.5
mos_mutation_scale = 0.05
mos_mutation_mode = "benign_std"
sign_layer_reduce = "quantile"
subspace_reduce = "max"
```

**单变量扫描**：
- 固定其他变量，仅改变一个参数
- 观察对隐蔽性和破坏性的影响

---

## 八、未实现的可选功能

以下功能在需求中提到，但优先级较低，未在此次修复中实现：

### 8.1 Constraint-domination 选择
**现状**：继续使用 Pareto 双目标优化

**可选升级**：
- 实现约束违反度 CV(x) = Σ w_j * v_j(x)
- 动态 epsilon: ε_t = ε_start * (1 - t/T) + ε_end * t/T
- 可行性优先比较

**建议**：若当前双目标效果良好，可暂不实施

### 8.2 Box 约束的 repair
**现状**：Box 约束已从默认加权分数中移除

**可选**：
- `use_box_repair = False`（默认）
- 可通过参数启用稳健边界 repair

### 8.3 径向阈值的 EMA
**现状**：使用当前轮分位数

**可选**：
```python
smoothed_radial_threshold = ema_decay * prev + (1 - ema_decay) * current
```
需要跨轮状态保存

---

## 九、验证清单

### 9.1 语法验证
✅ Python 语法检查通过：`python -m py_compile mos.py`

### 9.2 逻辑验证
✅ DNC mask 不再被覆盖
✅ 所有约束使用统一计算函数
✅ 阈值和候选使用相同公式
✅ 变异尺度可控且参数化
✅ 攻击预算基于稳健统计量
✅ 日志完整且信息丰富
✅ 数值稳定性检查到位

### 9.3 兼容性验证
✅ 旧参数名保留为别名
✅ 函数接口未改变
✅ 默认值通过 getattr 提供
✅ 设备一致性保持
✅ 除法均加入 epsilon

---

## 十、其他发现的问题（未修改）

在审查代码时发现以下潜在问题，但未在此次修复中处理（避免过度改动）：

1. **历史种群衰减系数硬编码**：`decay_factor = 0.95`（第584行）
2. **精英种子注入的尺度**：`scale_factor = krum_radius * 0.95`（第599行）
3. **Archive 大小固定**：`archive_size = EVOLUTION_POP_SIZE`（第631行）
4. **SBX 交叉参数**：`eta = 15.0`（第689行）

建议：若需进一步调优，可将这些值参数化。

---

## 十一、运行建议

### 11.1 首次测试
```python
# 使用默认参数运行小规模测试
args.nsga_generations = 20  # 减少代数
args.evo_pop_size = 10      # 减少种群
```

### 11.2 日志分析重点
- 检查 `DNC-aware mask enabled` 状态
- 观察约束 loss 是否在阈值附近
- 监控变异范数是否合理
- 确认种群范数比例不会全部被裁剪
- 验证隐蔽性得分不会饱和在 0.99 或 0.00

### 11.3 异常排查
若出现以下情况：
- **隐蔽性始终接近0**：降低 `constraint_score_temperature`
- **攻击效果不明显**：提高 `attack_budget_ratio`
- **种群过度裁剪**：检查 `max_dev_threshold` 是否过小
- **Logit 爆炸**：检查全局模型训练过程

---

## 十二、文件修改统计

- **新增函数**：3个
  - `compute_raw_constraint_losses()`
  - `compute_constraint_score()` (重写)
  - `compute_constraint_violations()`

- **重构函数**：3个
  - `compute_objectives()`
  - `compute_surrogate_guidance()`
  - `mos_attack()` (部分逻辑)

- **新增参数**：12个

- **修改日志输出**：15+ 处

- **代码行数变化**：
  - 原文件：~1303 行
  - 修复后：~1450 行
  - 净增加：~150 行（主要是新函数和日志）

---

## 联系与反馈

若在使用过程中发现问题，请检查：
1. 所有新参数是否正确传入 args
2. 日志中是否有 WARNING 或 ERROR
3. 约束阈值是否异常（过大或过小）

修复完成时间：2026-07-28
修复版本：v2.0 (Unified Constraints)
