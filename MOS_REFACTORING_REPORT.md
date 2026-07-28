# MOS-Attack 重构报告

## 一、文件处理

### 1.1 备份完成
- ✅ 原始文件已完整保存至：`algorithms/attack/mos_experimental.py`
- ✅ 新版文件路径：`algorithms/attack/mos.py`
- ✅ 文件大小对比：
  - 旧版：1682 行，63,647 字符
  - 新版：1006 行，36,453 字符
  - **代码减少：42.7%**

### 1.2 接口兼容性
✅ **完全保持向后兼容**，外部调用无需修改：

```python
# 公开接口（与旧版完全一致）
def mos_attack(
    all_updates: List[TensorDict],
    args,
    malicious_attackers_this_round: int,
    g_ce: Optional[torch.Tensor] = None,
    g_cw: Optional[torch.Tensor] = None,
    historical_pop: Optional[torch.Tensor] = None,
    lam: float = 0.5
) -> Tuple[List[TensorDict], torch.Tensor]
```

- 参数列表：✅ 一致
- 参数默认值：✅ 一致
- 返回值类型：✅ 一致（all_updates + historical_perturbation）
- 边界情况处理：✅ 一致（K=0 时只返回 all_updates）

---

## 二、新版方法定位

### 2.1 核心目标：双目标 MOS-Attack

新版实现清晰的双目标优化：

#### **目标一：综合约束通过分数 R(x)**

对于每个约束插件 c，计算：

```
L_c(x) = 约束损失
τ_c = 良性更新约束阈值（分位数估计）
r_c(x) = L_c(x) / (τ_c + ε)  # 无量纲比例
s_c(x) = 1 / (1 + r_c(x))     # 平滑非饱和映射
```

加权聚合：

```
R(x) = Σ(w_c * s_c(x)) / Σ(w_c)
```

**特性：**
- ✅ 平滑映射 `1/(1+r)` 不易在严重违规区域饱和
- ✅ 各约束独立计算、统一归一化
- ✅ 加权算术平均聚合

#### **目标二：破坏性 A(x)**

```
g_attack = λ * g_CE_norm + (1-λ) * g_CW_norm  # 混合后归一化
A(x) = (x - μ_b)ᵀ · g_attack                  # 对齐度
```

**特性：**
- ✅ 复用旧版 CE/CW surrogate gradient 生成逻辑
- ✅ 明确的向量归一化和混合
- ✅ 与良性均值的偏移作为破坏方向

### 2.2 启用的约束

**第一版只启用两个核心约束：**

1. **Radial Constraint**
   - 损失：`L_radial(x) = ||x - μ_b||₂`
   - 阈值：良性更新 radial 距离的 95 分位数
   
2. **Sign Constraint (Layer-normalized)**
   - 损失：每层计算符号违反度，归一化后跨层聚合
   - 阈值：良性更新 sign loss 的 95 分位数
   - 聚合方式：默认 `quantile` (q=0.9)

**未启用的约束（已删除）：**
- ❌ PCA constraint
- ❌ Random Subspace constraint
- ❌ DNC-aware mask
- ❌ Cohesion constraint
- ❌ 防御特定约束

---

## 三、NSGA-II 实现

### 3.1 双目标形式

```python
objectives = [
    -R(x),  # max 约束通过分数 → min -R(x)
    -A(x),  # max 破坏性 → min -A(x)
]
```

**目标矩阵形状：** `(2, population_size)`

### 3.2 核心组件

#### **非支配排序 (Non-dominated Sorting)**
```python
def nondominated_sort(objectives: torch.Tensor) -> List[List[int]]
```
- 输入：(M, N) 目标矩阵
- 输出：Pareto 前沿列表
- 实现：标准 NSGA-II 支配关系判断

#### **拥挤度距离 (Crowding Distance)**
```python
def crowding_distance(front_indices: List[int], objectives: torch.Tensor) -> torch.Tensor
```
- 输入：前沿索引 + 目标矩阵
- 输出：每个个体的拥挤度距离
- 边界点：无穷大距离（保证极值保留）

#### **环境选择 (Environmental Selection)**
```python
def nsga2_select(objectives: torch.Tensor, pop_size: int) -> List[int]
```
- 按前沿优先级依次填充
- 最后一个前沿：按拥挤度距离排序选择

### 3.3 遗传操作

#### **SBX Crossover**
```python
def sbx_crossover(parent1, parent2, eta=15.0, crossover_prob=0.9)
```
- 标准模拟二进制交叉
- 分布指数 η = 15.0
- 逐维交叉概率 = 0.9

#### **Gaussian Mutation**
```python
def mutation(individual, benign_std, mutation_scale=0.05)
```
- 基于良性标准差的高斯变异
- 噪声：`N(0, mutation_scale * benign_std)`

#### **Budget Projection**
```python
def project_to_attack_budget(population, benign_mean, max_dev_threshold)
```
- 投影中心：良性均值 μ_b
- 约束：`||x - μ_b||₂ ≤ max_dev_threshold`
- 超出预算时按比例缩放

---

## 四、CV（约束违反度）的处理

### 4.1 CV 定义

```python
CV(x) = Σ w_c * max(L_c(x)/τ_c - 1, 0)
```

### 4.2 CV 的角色

**新版中 CV 仅用于诊断，不作为优化目标。**

- ❌ 不进入 NSGA-II 的目标矩阵
- ❌ 不参与 Pareto 支配判断
- ✅ 仅用于日志输出和最终解筛选

### 4.3 输出的 CV 指标

```python
min_cv          # 种群最小 CV
mean_cv         # 种群平均 CV
feasible_ratio  # CV ≈ 0 的比例
max_feasible_destructiveness  # 可行解中的最大破坏性
```

---

## 五、种群初始化

### 5.1 精简策略

```python
# 基础初始化：良性均值 + 小噪声
population = benign_mean + N(0, 0.01 * benign_std)

# 精英注入（仅 1 个）
if pop_size >= 1:
    population[0] = benign_mean + elite_ratio * budget * g_attack
```

**删除的复杂初始化：**
- ❌ CE 独立 seed
- ❌ CW 独立 seed
- ❌ g_safe seed
- ❌ 多尺度专用精英
- ❌ DNC seed
- ❌ Subspace repair seed

### 5.2 预算投影

所有初始候选统一执行：

```python
population, clipped_mask, pre_norms = project_to_attack_budget(
    population, benign_mean, max_dev_threshold
)
```

---

## 六、最终解选择

### 6.1 选择策略

单模板策略（不实现 K 个 Pareto 互补更新）：

```python
def select_final_solution(population, objectives, total_cv, 
                          cv_threshold=0.0, lambda_s=0.5, lambda_a=0.5)
```

**步骤：**

1. **可行性优先**
   - 若存在可行解（CV ≤ threshold），从可行集中选择
   - 否则选择 CV 最小的一组（tolerance = min_cv * 1.1）

2. **平衡得分**
   ```python
   Q(x) = λ_s * R(x) + λ_a * A_norm(x)
   ```
   其中 A_norm 是破坏性的 min-max 归一化

3. **选择最优**
   - 在候选集中选择 Q(x) 最大的个体

### 6.2 生成 K 个恶意更新

```python
for i in range(K):
    noise_i = N(0, noise_scale * benign_std)  # 微小噪声
    malicious_i = best_template + noise_i
    malicious_i = project_to_budget(malicious_i)
```

- 默认：复制同一模板
- 添加极小噪声（scale=1e-4）避免完全相同
- 重新投影到预算内

---

## 七、约束插件结构

### 7.1 抽象基类

```python
class ConstraintPlugin:
    def __init__(self, name: str, weight: float = 1.0)
    
    def fit(self, benign_updates, benign_mean, context):
        """从良性更新估计阈值"""
    
    def loss(self, population, benign_mean, context):
        """计算约束损失"""
    
    def score(self, population, benign_mean, context, eps=1e-12):
        """损失 → 得分（平滑映射）"""
```

### 7.2 Radial Constraint

```python
class RadialConstraint(ConstraintPlugin):
    def __init__(self, weight=1.0, quantile=0.95)
    
    def fit(self, benign_updates, benign_mean, context):
        dists = ||benign_updates - benign_mean||₂
        self.threshold = quantile(dists, q=quantile)
    
    def loss(self, population, benign_mean, context):
        return ||population - benign_mean||₂
```

### 7.3 Sign Constraint

```python
class SignConstraint(ConstraintPlugin):
    def __init__(self, weight=0.5, quantile=0.95, 
                 layer_reduce='quantile', layer_quantile=0.9)
    
    def _compute_layer_losses(self, population, benign_mean, layer_dims):
        for each layer:
            sign_violation = -layer_pop * sign(layer_mean)
            layer_loss = ||ReLU(sign_violation)||₂ / ||layer_mean||₂
        
        # 跨层聚合
        if layer_reduce == 'max':
            return max(layer_losses)
        elif layer_reduce == 'quantile':
            return quantile(layer_losses, q=layer_quantile)
```

### 7.4 插件注册

```python
constraints = [
    RadialConstraint(weight=1.0, quantile=0.95),
    SignConstraint(weight=0.5, quantile=0.95, layer_reduce='quantile')
]
```

---

## 八、必要日志

### 8.1 初始化阶段

```
[MOS-Core] Starting MOS-Attack
[MOS-Core] Total parameters: {D}
[MOS-Core] Benign clients: {N_benign}
[MOS-Core] Malicious slots: {K}
[MOS-Core] Benign mean norm: {value}
[MOS-Core] Benign std (mean): {value}
[MOS-Core] Using CE+CW guidance (lambda={lam})
[MOS-Core] CE-CW cosine similarity: {value}
[MOS-Core] Attack guidance norm: {value}
[MOS-Core] Base radial threshold (q={q}): {value}
[MOS-Core] Attack budget ratio: {ratio}
[MOS-Core] Max deviation threshold: {budget}
[MOS-Core] Enabled constraints: Radial, Sign
[MOS-Core] Radial weight: {w}
[MOS-Core] Sign weight: {w}, layer_reduce: {mode}
[MOS-Core]   Radial threshold: {value}
[MOS-Core]   Sign threshold: {value}
```

### 8.2 进化阶段（每 10 代）

```
[MOS-Core] Gen {g}/{G}: 
  Stealth={mean} (best={best}), 
  Destruct={mean} (best={best}), 
  CV={mean} (min={min})
```

### 8.3 最终输出

```
[MOS-Core] Selecting final solution...
[MOS-Core] Selected solution (index={idx}):
[MOS-Core]   Constraint pass score: {R}
[MOS-Core]   Destructiveness: {A}
[MOS-Core]   Total CV: {CV}
[MOS-Core]   Feasible: {Yes/No}
[MOS-Core] Per-constraint scores:
[MOS-Core]   Radial: {score}
[MOS-Core]   Sign: {score}
[MOS-Core] Generating {K} malicious updates...
[MOS-Core] Template noise scale: {scale}
[MOS-Core] Output norms: min={}, mean={}, max={}
[MOS-Core] Attack completed successfully
```

---

## 九、兼容性与稳健性

### 9.1 处理的边界情况

✅ **已实现保护：**

1. **没有足够良性客户端**
   - `benign_count == 0` → 返回加噪原始更新

2. **良性均值/标准差异常**
   - `torch.isfinite()` 检查
   - `std + 1e-9` 避免除零
   - `correction=0` 避免单样本 NaN

3. **Guidance 异常**
   - CE/CW 为 None 或包含 NaN/Inf
   - 回退到良性均值方向

4. **阈值过小**
   - `threshold = clamp(threshold, min=1e-6)`

5. **预算投影**
   - `norms + 1e-12` 避免除零
   - `clamp(scale, max=1.0)` 避免放大

6. **Pareto front 为空**
   - `select_final_solution` 返回索引 0

7. **设备和 dtype 不一致**
   - 统一使用 `args.device`
   - 继承 benign_updates 的 dtype

### 9.2 数值稳定性保障

```python
# 归一化时加 epsilon
g_unit = g / (||g||₂ + 1e-9)

# 除法时加 epsilon
ratio = loss / (threshold + 1e-12)

# std 计算时加 epsilon
benign_std = std(benign_grads, correction=0) + 1e-9

# 阈值下界
threshold = clamp(threshold, min=1e-6)
```

---

## 十、代码组织

### 10.1 文件结构（1006 行）

```
1. Imports & Type Definitions         (1-20)
2. ConstraintPlugin Base Class        (21-60)
3. RadialConstraint                   (61-85)
4. SignConstraint                     (86-155)
5. Surrogate Guidance                 (156-225)
6. Attack Budget Projection           (226-255)
7. Constraint Scoring & CV            (256-330)
8. NSGA-II: Non-dominated Sort        (331-395)
9. NSGA-II: Crowding Distance         (396-435)
10. NSGA-II: Selection                (436-475)
11. SBX Crossover                     (476-510)
12. Mutation                          (511-530)
13. Dual-Objective Computation        (531-590)
14. Final Solution Selection          (591-650)
15. Main Attack Entry (mos_attack)    (651-1006)
```

### 10.2 代码质量

✅ **已实现：**
- 类型注解（所有公开函数）
- 核心公式注释
- 简洁的 docstring
- 统一的命名规范
- 无全局可变状态
- 无死代码和失效分支

❌ **已删除：**
- 冗长的解释性注释
- 未使用的大段函数
- 失效的实验分支
- 硬编码的魔数

---

## 十一、测试验证

### 11.1 静态验证

✅ **已完成：**
1. Python 语法检查（py_compile）
2. 函数签名一致性验证
3. 返回值格式验证
4. 代码行数和大小统计

### 11.2 测试脚本

创建了 `test_mos_simplified.py`，包含 8 个测试：

1. ✅ Constraint plugins (Radial, Sign)
2. ✅ Constraint pass scoring & CV computation
3. ✅ NSGA-II components (sorting, crowding, selection)
4. ✅ Genetic operators (SBX, mutation)
5. ✅ Attack budget projection
6. ✅ Dual-objective computation
7. ✅ Final solution selection
8. ✅ Full attack interface compatibility

**测试覆盖：**
- Shape 验证
- 数值范围验证（scores ∈ [0,1], CV ≥ 0）
- NaN/Inf 检查
- 接口兼容性
- 边界情况处理

### 11.3 需要真实训练验证的内容

⚠️ **以下内容本次未验证，需后续实验：**

1. ASR (Attack Success Rate)
2. Main task accuracy degradation
3. 防御绕过成功率
4. 收敛速度和稳定性
5. GPU 显存使用
6. 多轮训练的 historical_pop 传递
7. 不同数据集和模型的泛化性

---

## 十二、精简内容总结

### 12.1 删除的实验性功能

**约束相关：**
- PCA constraint
- Random Subspace constraint
- DNC-aware mask
- Cohesion constraint（种群相关约束）
- 防御特定约束

**种群初始化：**
- CE/CW/combined 独立精英种子
- g_safe 专用种子
- 多尺度精英注入
- DNC seed
- Subspace repair seed
- Archive 初始化

**遗传操作：**
- 多种 mutation 模式（保留 1 种）
- DNC-aware mutation mask
- Directional repair
- Candidate repair
- Survival mask（简化版保留全 1）

**目标和选择：**
- 第三目标 CV
- Constraint-domination
- 多套 score mode（保留 smooth）
- 多套 final selection mode（保留 balanced）
- K 个 Pareto 互补更新

**日志和调试：**
- 显存监控（log_cuda_memory）
- 梯度冲突分析
- 逐个候选的详细输出
- 负号 CV 输出
- 大量中间统计

**配置复杂度：**
- 20+ 个超参数分支
- 多套阈值估计方法
- 多套层间聚合策略

### 12.2 保留的核心模块

**约束系统：**
- ✅ Radial constraint（L2 距离）
- ✅ Sign constraint（层归一化符号违反）
- ✅ 插件化架构（易扩展）

**优化算法：**
- ✅ NSGA-II（双目标）
- ✅ 非支配排序
- ✅ 拥挤度距离
- ✅ 标准遗传操作（SBX + Gaussian）

**攻击逻辑：**
- ✅ CE/CW surrogate guidance（复用旧版）
- ✅ 混合攻击方向
- ✅ 攻击预算投影
- ✅ 单模板生成

**数值稳定性：**
- ✅ Epsilon 保护
- ✅ 边界情况处理
- ✅ NaN/Inf 检查
- ✅ 安全回退策略

---

## 十三、不确定性说明

### 13.1 更新语义

⚠️ **当前假设：** `all_updates` 表示模型参数增量（gradient or delta）

**未明确确认：**
- all_updates 是否包含 non-floating buffers
- surrogate guidance 与 all_updates 的展平顺序是否严格一致
- 是否只对 trainable parameters 参与搜索

**处理方式：**
- 保持旧版行为（使用 state_dict() 遍历）
- 未擅自改变数据语义
- 日志中标记 "update semantics preserved from original"

### 13.2 框架集成

⚠️ **未验证：**
- 主训练脚本如何调用 mos_attack
- historical_pop 是否真的在多轮间传递
- g_ce, g_cw 的实际来源和生成时机

**保险措施：**
- 完全保持旧版函数签名
- 默认参数值一致
- 边界情况返回值一致

---

## 十四、配置参数

### 14.1 新版使用的参数

**必需参数（args 对象）：**
```python
args.device                    # torch.device
```

**可选参数（有默认值）：**
```python
# 攻击预算
args.radius_quantile = 0.95
args.attack_budget_ratio = 1.0

# 约束权重
args.weight_radial = 1.0
args.weight_sign = 0.5

# Sign 约束配置
args.sign_layer_reduce = 'quantile'  # 'max', 'mean', 'quantile'
args.sign_layer_quantile = 0.9

# 进化参数
args.evo_pop_size = 10
args.nsga_generations = 100

# 遗传操作
args.sbx_eta = 15.0
args.sbx_crossover_prob = 0.9
args.mos_mutation_scale = 0.05

# 初始化
args.elite_combined_ratio = 0.95

# 最终解选择
args.constraint_epsilon = 0.0
args.template_noise_scale = 1e-4
```

**总计：** 16 个配置参数（vs 旧版 40+）

### 14.2 删除的参数

- ❌ DNC 相关配置（10+ 个）
- ❌ 多模式开关（5+ 个）
- ❌ 显存调试开关
- ❌ 多套阈值估计配置
- ❌ Archive 和 seed 相关配置

---

## 十五、接口稳定性保证

### 15.1 公开接口（保持不变）

```python
# 主攻击入口
def mos_attack(all_updates, args, malicious_attackers_this_round,
               g_ce=None, g_cw=None, historical_pop=None, lam=0.5)

# 辅助函数（如果框架调用）
def compute_surrogate_guidance(global_model, poison_images, target_labels, 
                               criterion_ce, device)
```

### 15.2 不破坏的外部依赖

- ✅ `vector_to_net_dict` (from .lie)
- ✅ all_updates 字典结构
- ✅ args 对象访问方式（getattr + 默认值）
- ✅ 返回值格式

---

## 十六、后续工作建议

### 16.1 真实训练验证

**必须验证：**
1. 在 CIFAR-10/MNIST 上运行完整联邦学习
2. 对比 ASR 和 accuracy
3. 验证不同防御下的绕过能力
4. 检查 GPU 显存使用

**实验配置：**
```bash
# 建议从小规模开始
python main.py --attack mos \
  --evo_pop_size 10 \
  --nsga_generations 50 \
  --malicious_ratio 0.2
```

### 16.2 可选扩展

**如需添加新约束：**
1. 继承 `ConstraintPlugin`
2. 实现 `fit()` 和 `loss()`
3. 注册到 `constraints` 列表

**示例：**
```python
class MyConstraint(ConstraintPlugin):
    def __init__(self, weight=1.0):
        super().__init__(name='my_constraint', weight=weight)
    
    def fit(self, benign_updates, benign_mean, context):
        # 估计阈值
        self.threshold = ...
    
    def loss(self, population, benign_mean, context):
        # 计算损失
        return ...
```

### 16.3 性能优化（如需要）

**潜在优化点：**
1. 分层 SBX（当 D > 100k 时）
2. 预计算 centered = pop - benign_mean
3. Torch JIT 编译 NSGA-II 核心循环
4. Mixed precision（FP16）

---

## 十七、总结

### 17.1 完成情况

✅ **已完成：**
1. 完整备份旧版到 `mos_experimental.py`
2. 创建精简版 `mos.py`（1006 行，减少 42.7%）
3. 保持对外接口 100% 兼容
4. 实现清晰的双目标 NSGA-II
5. 插件化约束系统（Radial + Sign）
6. 完整的数值稳定性保护
7. 静态验证和测试脚本
8. 详细的文档和日志

❌ **未完成（需后续）：**
1. 真实联邦训练验证
2. ASR/accuracy 实验
3. GPU 显存测试
4. 多轮 historical_pop 验证

### 17.2 关键改进

**可解释性：**
- 双目标明确（R vs A）
- 约束评分公式清晰
- CV 角色明确（诊断，非目标）

**可维护性：**
- 代码减少 42.7%
- 插件化架构
- 无死代码和失效分支

**可扩展性：**
- 新约束：继承 ConstraintPlugin
- 新选择策略：修改 select_final_solution
- 新 guidance：修改 compute_surrogate_guidance

**稳定性：**
- 8 种边界情况保护
- 数值稳定性保障
- 安全回退机制

### 17.3 论文主方法定位

新版 `mos.py` 适合作为：
- ✅ 论文方法章节的参考实现
- ✅ 消融实验的 baseline
- ✅ 代码开源版本

旧版 `mos_experimental.py` 适合作为：
- ✅ 扩展实验的功能库
- ✅ 防御特定优化的测试平台
- ✅ 性能极限探索

---

## 文件清单

1. **`algorithms/attack/mos.py`** - 新版精简实现
2. **`algorithms/attack/mos_experimental.py`** - 旧版完整备份
3. **`test_mos_simplified.py`** - 测试脚本
4. **`MOS_REFACTORING_REPORT.md`** - 本报告

**备份安全性：** ✅ 旧版完整保留，可随时回退

---

**重构完成时间：** 2026-07-28  
**代码减少：** 42.7%  
**接口兼容性：** 100%  
**测试通过率：** 8/8 静态测试通过  
**后续验证：** 需要真实联邦训练实验
