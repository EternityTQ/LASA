# MOS-Attack 重构完成汇报

## ✅ 一、文件处理

### 1. 原文件备份
- **备份路径：** `d:\LASA\algorithms\attack\mos_experimental.py`
- **备份完整性：** ✅ 完整保留（1682 行，63,647 字符）
- **新文件路径：** `d:\LASA\algorithms\attack\mos.py`
- **代码减少：** 42.7%（1006 行，36,453 字符）

### 2. 接口兼容性
```python
# 公开接口（完全不变）
def mos_attack(all_updates, args, malicious_attackers_this_round,
               g_ce=None, g_cw=None, historical_pop=None, lam=0.5)
```

**验证结果：**
- ✅ 函数签名一致
- ✅ 参数默认值一致
- ✅ 返回值格式一致
- ✅ 边界情况处理一致

## ✅ 二、公开接口

新版 `mos.py` 对外暴露的函数与旧版完全一致：

```python
# 主攻击入口
mos_attack(all_updates, args, malicious_attackers_this_round, 
           g_ce, g_cw, historical_pop, lam)
→ Returns: (all_updates, historical_perturbation) 或 all_updates

# 辅助函数（如果框架调用）
compute_surrogate_guidance(global_model, poison_images, target_labels, 
                          criterion_ce, device)
→ Returns: (g_ce, g_cw)

# 工具函数（如果需要）
project_to_attack_budget(population, benign_mean, max_dev_threshold)
→ Returns: (projected, clipped_mask, pre_norms)
```

**重要：** 框架中的其他文件无需任何修改！

## ✅ 三、精简版删除的模块

### 1. 约束相关（删除 5 个）
- ❌ PCA constraint
- ❌ Random Subspace constraint  
- ❌ DNC-aware mask
- ❌ Cohesion constraint
- ❌ 防御特定约束

### 2. 初始化策略（简化为 1 个）
- ❌ CE/CW/combined 独立精英
- ❌ g_safe 专用种子
- ❌ 多尺度精英注入
- ❌ DNC seed
- ❌ Subspace repair seed
- ✅ 保留：基础随机 + 1 个攻击方向精英

### 3. 遗传操作（简化）
- ❌ 多种 mutation 模式（保留 1 种）
- ❌ DNC-aware mutation mask
- ❌ Directional/candidate repair
- ✅ 保留：标准 SBX + Gaussian mutation

### 4. 目标和选择（简化为 2 目标）
- ❌ 第三目标 CV
- ❌ Constraint-domination
- ❌ 多套 score/selection mode
- ❌ K 个 Pareto 互补更新
- ✅ 保留：双目标 NSGA-II + 单模板

### 5. 调试和日志（精简）
- ❌ 显存监控函数
- ❌ 梯度冲突分析
- ❌ 逐个候选详细输出
- ✅ 保留：核心指标日志

## ✅ 四、保留的核心模块

### 1. 约束系统
```python
class ConstraintPlugin:  # 插件基类
    def fit(...)    # 估计阈值
    def loss(...)   # 计算损失
    def score(...)  # 损失→得分映射

class RadialConstraint(ConstraintPlugin):
    # L2 距离约束

class SignConstraint(ConstraintPlugin):
    # 层归一化符号约束
```

**特性：**
- ✅ 插件化架构（易扩展）
- ✅ 统一的 fit-loss-score 接口
- ✅ 平滑非饱和映射：`s = 1/(1+r)`

### 2. 优化算法（NSGA-II）
```python
nondominated_sort(objectives)     # 非支配排序
crowding_distance(front, objs)    # 拥挤度距离
nsga2_select(objectives, pop_size) # 环境选择
sbx_crossover(p1, p2, eta, prob)  # SBX 交叉
mutation(ind, benign_std, scale)  # 高斯变异
```

### 3. 攻击逻辑
```python
compute_surrogate_guidance(...)   # CE + CW 梯度
compute_dual_objectives(...)      # 双目标评估
select_final_solution(...)        # 最终解选择
project_to_attack_budget(...)     # 预算投影
```

### 4. 数值稳定性
- ✅ 所有除法 + epsilon
- ✅ 阈值下界保护
- ✅ NaN/Inf 检查
- ✅ 安全回退机制

## ✅ 五、两个目标的具体公式

### 目标 1：综合约束通过分数 R(x)

**单约束得分：**
```
L_c(x) = 约束损失（如 ||x - μ_b||₂）
τ_c = 良性更新的约束阈值（分位数估计）
r_c(x) = L_c(x) / (τ_c + ε)        # 无量纲比例
s_c(x) = 1 / (1 + r_c(x))          # 平滑映射
```

**加权聚合：**
```
R(x) = [Σ w_c · s_c(x)] / [Σ w_c]  # 加权平均
```

**优化形式：**
```
max R(x)  →  min -R(x)  （NSGA-II 最小化）
```

### 目标 2：破坏性 A(x)

**攻击方向构造：**
```
g_CE_norm = g_CE / ||g_CE||₂
g_CW_norm = g_CW / ||g_CW||₂
g_attack = [λ·g_CE_norm + (1-λ)·g_CW_norm] / ||...||₂
```

**破坏性度量：**
```
A(x) = (x - μ_b)ᵀ · g_attack       # 与攻击方向的对齐度
```

**优化形式：**
```
max A(x)  →  min -A(x)  （NSGA-II 最小化）
```

**NSGA-II 目标矩阵：**
```python
objectives = [
    -R(x),  # shape: (P,)
    -A(x),  # shape: (P,)
]  # shape: (2, P)
```

## ✅ 六、当前启用的约束插件

### 1. Radial Constraint
```python
RadialConstraint(weight=1.0, quantile=0.95)
```

**损失定义：**
```
L_radial(x) = ||x - μ_b||₂
```

**阈值估计：**
```
τ_radial = quantile({||x_i - μ_b||₂ : x_i ∈ benign_updates}, q=0.95)
```

**物理意义：** 距离良性均值的欧氏距离

### 2. Sign Constraint (Layer-normalized)
```python
SignConstraint(weight=0.5, quantile=0.95, 
               layer_reduce='quantile', layer_quantile=0.9)
```

**损失定义（每层）：**
```
sign_violation^(l) = ReLU(-x^(l) ⊙ sign(μ_b^(l)))
L_sign^(l) = ||sign_violation^(l)||₂ / (||μ_b^(l)||₂ + ε)
```

**跨层聚合：**
```
L_sign(x) = quantile({L_sign^(l) : l ∈ layers}, q=0.9)
或
L_sign(x) = max({L_sign^(l) : l ∈ layers})
或
L_sign(x) = mean({L_sign^(l) : l ∈ layers})
```

**物理意义：** 与良性均值符号相反的程度（归一化）

### 3. 约束权重

**默认配置：**
```python
w_radial = 1.0   # Radial 约束权重
w_sign = 0.5     # Sign 约束权重
```

**综合得分：**
```
R(x) = (1.0 · s_radial(x) + 0.5 · s_sign(x)) / (1.0 + 0.5)
     = (s_radial(x) + 0.5 · s_sign(x)) / 1.5
```

## ✅ 七、CV 当前仅用于

### CV 定义
```
CV(x) = Σ w_c · max(L_c(x)/τ_c - 1, 0)
```

### CV 的角色（重要！）

**❌ CV 不是优化目标：**
- 不进入 NSGA-II 的 objectives 矩阵
- 不参与 Pareto 支配判断
- 不影响遗传操作

**✅ CV 仅用于：**

1. **诊断日志输出**
   ```
   min_cv          # 种群最小 CV
   mean_cv         # 种群平均 CV
   feasible_ratio  # CV ≈ 0 的个体比例
   ```

2. **最终解筛选**
   ```python
   # 优先选择可行解（CV <= threshold）
   if exists(CV(x) <= ε):
       candidates = {x : CV(x) <= ε}
   else:
       candidates = {x : CV(x) ≈ min_cv}
   
   # 在候选集中选择平衡得分最高的
   best = argmax(λ_s·R(x) + λ_a·A(x))
   ```

3. **可行性分析**
   ```
   feasible_count = |{x : CV(x) <= ε}|
   max_feasible_destructiveness = max({A(x) : CV(x) <= ε})
   ```

**关键区别：**
- 旧版某些实验分支：CV 作为第三目标
- 新版（论文主方法）：CV 仅作诊断指标

## ✅ 八、完成的静态测试

### 1. Python 语法检查
```bash
✅ python -m py_compile mos.py
   无语法错误
```

### 2. 导入检查
```bash
✅ 依赖项检查通过
   - torch (✓)
   - torch.nn.functional (✓)
   - copy (✓)
   - .lie.vector_to_net_dict (✓)
```

### 3. 接口一致性
```bash
✅ 函数签名验证
   - mos_attack 参数列表一致
   - mos_attack 返回值类型一致
   - 边界情况返回值一致
```

### 4. Shape 验证（测试脚本）
```bash
✅ Radial loss shape: (P,)
✅ Radial score shape: (P,), range: [0, 1]
✅ Sign loss shape: (P,)
✅ Sign score shape: (P,), range: [0, 1]
✅ Constraint pass score shape: (P,), range: [0, 1]
✅ Total CV shape: (P,), CV >= 0
✅ Objectives shape: (2, P)
```

### 5. NSGA-II 组件测试
```bash
✅ nondominated_sort 返回非空前沿
✅ crowding_distance 边界点距离 = ∞
✅ nsga2_select 选择正确数量
✅ sbx_crossover 无 NaN/Inf
✅ mutation 无 NaN/Inf
```

### 6. 数值稳定性测试
```bash
✅ project_to_attack_budget: 所有投影后范数 <= budget
✅ 除零保护: 所有除法 + epsilon
✅ 阈值下界: threshold >= 1e-6
✅ NaN/Inf 检查: 所有关键路径有检查
```

### 7. 接口兼容性测试
```bash
✅ all_updates 字典格式输入/输出
✅ historical_perturbation 返回格式正确
✅ K=0 时返回 all_updates（不带元组）
✅ benign_count=0 时安全回退
```

### 8. 代码质量检查
```bash
✅ 无死代码
✅ 无失效分支
✅ 类型注解完整
✅ 核心公式有注释
```

## ⚠️ 九、需要真实训练验证的问题

### 1. ASR 和 Accuracy
```
未验证：
- Attack Success Rate 是否保持
- Main task accuracy 是否下降足够
- 是否与旧版性能相当
```

**验证方法：**
```bash
python main.py --attack mos --dataset cifar10 \
  --malicious_ratio 0.2 --defend_methods none
```

### 2. 防御绕过能力
```
未验证：
- Krum/Trimmed-Mean 绕过
- Norm clipping 绕过
- DNC 绕过（虽然删除了 DNC 专用约束）
```

### 3. 收敛性和稳定性
```
未验证：
- 是否在合理代数内收敛
- 是否出现数值不稳定
- 不同随机种子下的方差
```

### 4. GPU 显存使用
```
未验证：
- 峰值显存是否降低（预期降低，因为代码精简）
- 是否仍有 OOM 风险
- 是否需要进一步优化
```

### 5. Historical Population 传递
```
未验证：
- 跨轮传递是否正常工作
- 是否真的加速收敛
- 衰减策略是否合适
```

### 6. 不同数据集和模型
```
未验证：
- MNIST / CIFAR-10 / CIFAR-100
- ResNet / VGG / MobileNet
- 不同数据分布下的泛化性
```

## ⚠️ 十、发现的不确定性

### 1. 更新语义
```
当前假设：all_updates 表示模型参数增量（gradient or delta）

未明确确认：
- all_updates 是否包含 non-floating buffers？
- surrogate guidance 与 all_updates 的展平顺序是否严格一致？
- 是否只对 trainable floating-point parameters 参与搜索？

处理方式：
✅ 保持旧版行为（使用 state_dict() 遍历）
✅ 未擅自改变数据语义
```

### 2. 框架调用方式
```
未确认：
- 主训练脚本如何调用 mos_attack？
- historical_pop 是否真的在多轮间传递？
- g_ce, g_cw 的实际来源和生成时机？

保险措施：
✅ 完全保持旧版函数签名
✅ 默认参数值一致
✅ 边界情况返回值一致
```

### 3. 维度顺序
```
假设：
- all_updates 字典的 keys() 顺序稳定
- state_dict() 遍历顺序与 all_updates 一致

风险：
- 如果顺序不一致，guidance 对齐会出错

缓解：
✅ 使用 state_dict() 统一遍历（与旧版一致）
```

## 📋 十一、文件清单

### 生成的文件
1. **`algorithms/attack/mos.py`** (1006 行)
   - 精简版主实现
   
2. **`algorithms/attack/mos_experimental.py`** (1682 行)
   - 旧版完整备份
   
3. **`test_mos_simplified.py`**
   - 静态测试脚本（8 个测试）
   
4. **`MOS_REFACTORING_REPORT.md`**
   - 详细重构报告（17 节）
   
5. **`MOS_REFACTORING_SUMMARY.md`** (本文件)
   - 简明总结汇报

### 修改的文件
- ✅ 无！（所有修改集中在 `mos.py`，其他文件不变）

## 📊 十二、统计数据

### 代码规模
```
旧版：1682 行，63,647 字符
新版：1006 行，36,453 字符
减少：676 行（40.2%），27,194 字符（42.7%）
```

### 函数数量
```
旧版：约 30+ 个函数
新版：24 个函数
精简：约 20%
```

### 配置参数
```
旧版：40+ 个超参数
新版：16 个超参数
精简：60%
```

### 约束类型
```
旧版：5+ 种约束
新版：2 种约束（Radial + Sign）
精简：60%
```

### 测试覆盖
```
静态测试：8/8 通过
语法检查：✅ 通过
接口验证：✅ 通过
真实训练：⚠️ 待验证
```

## 🎯 十三、最终验证清单

### ✅ 已完成
- [x] 完整备份旧版
- [x] 创建精简版主文件
- [x] 保持接口 100% 兼容
- [x] 实现双目标 NSGA-II
- [x] 实现 2 个核心约束
- [x] 插件化架构
- [x] 数值稳定性保护
- [x] 静态测试通过
- [x] Python 语法检查通过
- [x] 代码质量检查通过
- [x] 详细文档编写

### ⚠️ 待完成（需后续实验）
- [ ] 真实联邦训练验证
- [ ] ASR/Accuracy 对比实验
- [ ] 防御绕过能力测试
- [ ] GPU 显存测试
- [ ] 收敛性分析
- [ ] 不同数据集泛化测试

## ✅ 十四、重构成功指标

### 代码质量
- ✅ 代码减少 42.7%
- ✅ 无死代码和失效分支
- ✅ 完整类型注解
- ✅ 清晰的代码结构

### 可解释性
- ✅ 双目标明确定义
- ✅ 约束公式清晰
- ✅ CV 角色明确
- ✅ 日志输出清晰

### 可维护性
- ✅ 插件化架构
- ✅ 统一的约束接口
- ✅ 模块化函数设计
- ✅ 详细的文档

### 可扩展性
- ✅ 新约束：继承 ConstraintPlugin
- ✅ 新选择策略：修改 select_final_solution
- ✅ 新 guidance：修改 compute_surrogate_guidance

### 兼容性
- ✅ 接口完全兼容
- ✅ 框架无需修改
- ✅ 可随时回退到旧版

## 🚀 十五、下一步行动建议

### 1. 立即验证（推荐）
```bash
# 1. 简单 smoke test
cd d:\LASA
python main.py --attack mos --dataset mnist \
  --num_clients 10 --malicious_ratio 0.2 \
  --evo_pop_size 6 --nsga_generations 20 \
  --epochs 1

# 2. 完整实验（如果 smoke test 通过）
python main.py --attack mos --dataset cifar10 \
  --num_clients 100 --malicious_ratio 0.2 \
  --evo_pop_size 10 --nsga_generations 100 \
  --epochs 50
```

### 2. 性能对比
```bash
# 对比新旧版本
# 1. 修改 main.py 临时导入 mos_experimental
# 2. 运行相同配置
# 3. 对比 ASR, accuracy, 运行时间, 显存
```

### 3. 文档补充
```bash
# 如果实验成功，补充：
# - 实验结果对比表
# - 性能 profiling 数据
# - GPU 显存使用对比
# - 收敛曲线对比
```

### 4. 可选优化
```bash
# 如果发现性能瓶颈：
# - Torch JIT 编译 NSGA-II
# - Mixed precision (FP16)
# - 分层 SBX（大模型）
```

## 📝 十六、重要提醒

### 对于论文
- ✅ 新版适合作为方法章节的参考实现
- ✅ 公式清晰，易于描述
- ✅ 双目标明确，便于理论分析
- ⚠️ 必须用真实实验验证后才能宣称性能

### 对于代码
- ✅ 旧版完整备份，可随时回退
- ✅ 新版接口兼容，框架无需改动
- ⚠️ 如遇问题，优先检查 all_updates 语义和维度顺序

### 对于实验
- ⚠️ 先小规模测试（MNIST, 10 clients, 20 gens）
- ⚠️ 再扩展到 CIFAR-10（100 clients, 100 gens）
- ⚠️ 最后测试不同防御（Krum, Trimmed-Mean, DNC）

---

## ✅ 重构完成声明

**新版 MOS-Attack (`mos.py`) 已完成重构，满足所有设计要求：**

1. ✅ 完整备份旧版到 `mos_experimental.py`
2. ✅ 实现清晰的双目标 NSGA-II（R vs A）
3. ✅ 启用 2 个核心约束（Radial + Sign）
4. ✅ 平滑非饱和约束映射 `1/(1+r)`
5. ✅ CV 仅用于诊断，不作为目标
6. ✅ 插件化约束架构
7. ✅ 保持接口 100% 兼容
8. ✅ 代码减少 42.7%
9. ✅ 静态测试全部通过
10. ✅ 文档完整详细

**可以安全集成到现有框架，建议先进行小规模真实训练验证。**

---

**重构完成日期：** 2026-07-28  
**版本标识：** MOS-Attack Simplified v1.0  
**代码状态：** 静态验证通过，待真实训练验证  
**兼容性：** 100% 向后兼容
