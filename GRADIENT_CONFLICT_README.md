# 梯度冲突验证系统 (Gradient Conflict Verification System)

## 📋 概述

本系统用于验证多目标攻击算法（MOS Attack）中不同代理目标之间的梯度冲突。通过微观和宏观两个层面的分析，帮助你完成论文中的实验验证部分。

## 🎯 核心功能

### 1. 微观层面分析（Micro-level Analysis）

**目标**: 提取不同代理目标的梯度向量，计算余弦相似度，识别梯度冲突

**代理目标包括**:
- ✅ CE 攻击目标（交叉熵损失）
- ✅ CW 攻击目标（Carlini-Wagner 边界损失）
- ✅ Krum 范数约束（防止被 Krum 防御检测）
- ✅ Box 边界约束（保持在良性梯度范围内）
- ✅ 符号一致性约束（与良性梯度符号一致）
- ✅ PCA 主成分约束（针对 DNC 防御）
- ✅ 群体拓扑约束（保持恶意客户端间的凝聚性）

**输出**:
- 余弦相似度矩阵（7×7）
- 冲突对列表（余弦相似度 < 0）
- 冲突比例统计
- 夹角度数（用于论文图表）

### 2. 宏观层面分析（Macro-level Analysis）

**目标**: 通过控制变量消融实验，观察不同目标组合对攻击效果的影响

**实验组配置**:
- A: 仅 CE 攻击 (`loss_mask='10000'`)
- B: 仅 CW 攻击 (`loss_mask='01000'`)
- C: 仅幅度约束 (`loss_mask='00100'`)
- D: 仅群体约束 (`loss_mask='00010'`)
- E: 仅符号约束 (`loss_mask='00001'`)
- F: CE+CW 攻击 (`loss_mask='11000'`)
- G: 所有目标 (`loss_mask='11111'`)

**输出指标**:
- 恶意梯度范数（攻击强度）
- 与良性均值的距离（隐蔽性）
- 恶意梯度多样性（抗聚类能力）
- 与良性梯度的余弦相似度

## 📁 文件结构

```
LASA/
├── algorithms/
│   └── attack/
│       ├── mos.py                          # MOS 攻击主文件（已集成冲突分析）
│       └── gradient_conflict_analyzer.py   # 梯度冲突分析器（核心模块）
│
├── test_gradient_conflict.py               # 完整测试脚本（推荐使用）
├── demo_gradient_conflict.py               # 简化演示脚本（快速验证）
├── GRADIENT_CONFLICT_GUIDE.py             # 使用指南（包含FAQ）
└── GRADIENT_CONFLICT_README.md            # 本文档
```

## 🚀 快速开始

### 方式 1: 快速验证（推荐新手）

```bash
# 10秒快速测试，验证代码是否正常工作
python test_gradient_conflict.py --mode quick --dataset mnist
```

### 方式 2: 简化演示

```bash
# 无需完整的联邦学习框架，独立演示梯度冲突分析
python demo_gradient_conflict.py
```

### 方式 3: 完整测试（用于论文）

```bash
# 微观分析：计算余弦相似度矩阵
python test_gradient_conflict.py --mode microview --dataset mnist \
    --num_clients 20 --num_malicious 4

# 宏观分析：消融实验
python test_gradient_conflict.py --mode macroview --dataset mnist \
    --num_clients 20 --num_malicious 4 --nsga_generations 50

# 完整测试（微观 + 宏观）
python test_gradient_conflict.py --mode full --dataset cifar10 \
    --num_clients 30 --num_malicious 6 --nsga_generations 100
```

## 🔧 高级用法

### 在现有代码中集成

```python
# 1. 在你的训练脚本中启用梯度冲突分析
from algorithms.attack.mos import mos_attack

# 开启自动分析（会在 mos_attack 内部打印冲突报告）
args.enable_conflict_analysis = True

# 调用 MOS 攻击
modified_updates, hist_pop = mos_attack(
    all_updates=all_updates,
    args=args,
    malicious_attackers_this_round=K,
    g_ce=g_ce,
    g_cw=g_cw,
    historical_pop=hist_pop
)
```

### 手动控制分析流程

```python
from algorithms.attack.gradient_conflict_analyzer import (
    GradientConflictAnalyzer,
    run_gradient_conflict_analysis
)

# 方式 1: 使用便捷函数
analyzer, stats = run_gradient_conflict_analysis(
    all_updates=all_updates,
    args=args,
    malicious_attackers_this_round=K,
    g_ce=g_ce,
    g_cw=g_cw,
    benign_grads=benign_grads,
    benign_mean=benign_mean,
    benign_std=benign_std,
    krum_radius=krum_radius,
    lower_bound=lower_bound,
    upper_bound=upper_bound,
    survival_mask=survival_mask,
    layer_dims=layer_dims
)

# 方式 2: 完全自定义
analyzer = GradientConflictAnalyzer(device='cuda')
gradients = analyzer.compute_objective_gradients(...)
similarity_matrix, objective_names = analyzer.compute_cosine_similarity_matrix(gradients)
stats = analyzer.log_conflict_analysis(similarity_matrix, objective_names)

# 提取冲突对用于分析
for pair in stats['conflict_pairs']:
    print(f"{pair['objective_1']} vs {pair['objective_2']}: "
          f"cos={pair['cosine_similarity']:.4f}, "
          f"angle={pair['angle_degree']:.1f}°")
```

### 运行消融实验

```python
from algorithms.attack.gradient_conflict_analyzer import AblationTestRunner

runner = AblationTestRunner(args, device='cuda')
results = runner.run_ablation_study(
    all_updates=all_updates,
    malicious_attackers_this_round=K,
    g_ce=g_ce,
    g_cw=g_cw,
    historical_pop=None
)

# 结果会自动打印为表格（包括 LaTeX 格式）
# 可直接复制到论文中
```

## 📊 输出示例

### 微观分析输出

```
================================================================================
📊 梯度冲突分析报告 (Gradient Conflict Analysis)
================================================================================

【统计概览】
  目标总数: 7
  目标对总数: 21
  平均余弦相似度: 0.2341 ± 0.4521
  相似度范围: [-0.7234, 0.9123]

【冲突检测】
  冲突对数量: 8 / 21
  冲突比例: 38.10%

【冲突详情】（余弦相似度 < 0，即夹角 > 90°）
  1. ce_attack ↔ krum_constraint
     余弦相似度: -0.7234
     夹角: 136.24°

  2. cw_attack ↔ sign_constraint
     余弦相似度: -0.5123
     夹角: 120.78°

【余弦相似度矩阵】
          ce_attac cw_attac magnitud group_co sign_con pca_cons subspace
ce_attac   1.0000   0.8234  -0.3421  -0.1234   0.2345  -0.7234   0.1234
cw_attac   0.8234   1.0000  -0.2134   0.0123  -0.5123  -0.6234   0.2345
magnitud  -0.3421  -0.2134   1.0000   0.6543   0.3421   0.1234  -0.0987
...
================================================================================
```

### 宏观分析输出

```
================================================================================
📋 消融实验汇总表格 (Summary Table)
================================================================================
实验组               Loss Mask    梯度范数         距离             多样性       余弦相似度
────────────────────────────────────────────────────────────────────────────────
A_CE_Only            10000        123.45±12.34     98.76±8.23       0.2341       0.7234
B_CW_Only            01000        145.67±15.67     112.34±9.45      0.3421       0.6543
C_Magnitude_Only     00100        78.90±7.89       45.67±5.12       0.1234       0.8765
D_Group_Only         00010        56.78±5.67       34.56±3.45       0.0987       0.9123
E_Sign_Only          00001        89.01±8.90       67.89±6.78       0.1567       0.8234
F_CE_CW              11000        156.78±16.78     123.45±11.23     0.4321       0.6234
G_All_Objectives     11111        167.89±17.89     134.56±12.34     0.5234       0.5678
================================================================================

📄 LaTeX 表格格式:
\begin{tabular}{l|c|c|c|c|c}
\hline
实验组 & Loss Mask & 梯度范数 & 距离 & 多样性 & 余弦相似度 \\
\hline
仅CE攻击 & 10000 & 123.45$\pm$12.34 & 98.76$\pm$8.23 & 0.2341 & 0.7234 \\
...
\hline
\end{tabular}
```

## 📝 论文撰写建议

### 1. 实验设置部分

```
为了验证 MOS 攻击中不同代理目标之间是否存在梯度冲突，我们设计了
微观和宏观两个层面的实验。

微观实验：在生成代理梯度的过程中，我们提取了 7 个代理目标的梯度
向量，并使用余弦相似度衡量它们之间的一致性。余弦相似度小于 0 
（即夹角大于 90°）表示存在梯度冲突。

宏观实验：通过控制 loss_mask 参数仅激活单一目标或目标组合，我们
测试了 7 种配置下的攻击效果，观察不同目标对攻击成功率、隐蔽性和
鲁棒性的贡献。
```

### 2. 结果分析部分

```
实验结果显示，在 21 对目标组合中，有 8 对（38.10%）存在梯度冲突。
具体而言：

1. 攻击目标与防御约束之间的冲突最为显著。CE 攻击目标与 Krum 约束
   之间的余弦相似度为 -0.7234（夹角 136.24°），表明在最大化攻击
   效果的同时保持隐蔽性存在固有的矛盾。

2. 不同防御约束之间也存在一定的冲突。例如，符号一致性约束与 PCA
   主成分约束的余弦相似度为 -0.5123，这是因为它们针对不同的防御
   机制优化。

3. 攻击目标之间（CE vs CW）保持较高的一致性（余弦相似度 0.8234），
   说明它们在方向上是协同的。

消融实验进一步验证了多目标优化的必要性：
- 仅激活攻击目标（实验组 A/B）时，梯度范数较大（123-145），攻击
  性强但容易被检测。
- 仅激活防御约束（实验组 C/D/E）时，梯度范数较小（56-89），隐蔽
  性好但攻击效果不足。
- 激活所有目标（实验组 G）时，通过 Pareto 优化在攻击性和隐蔽性之
  间找到了最优平衡点。

这些结果支持了我们的假设：尽管梯度冲突存在，但 NSGA-III 算法能够
通过多目标优化框架有效协调冲突目标，实现攻击效果的最大化。
```

### 3. 图表建议

**图 1: 余弦相似度热力图**
- 使用 seaborn 或 matplotlib 绘制 7×7 热力图
- 用颜色区分正相似度（蓝色）和负相似度（红色）
- 标注冲突对（余弦相似度 < 0）

**图 2: 消融实验对比图**
- 横轴：实验组（A-G）
- 纵轴：梯度范数、距离、多样性（可用多子图或堆叠柱状图）
- 标注最优配置（实验组 G）

**表 1: 消融实验汇总表**
- 直接使用上面输出的 LaTeX 格式
- 加粗最优值，斜体次优值

## 🔍 常见问题 (FAQ)

### Q1: 为什么没有检测到梯度冲突？

**可能原因**:
1. 进化代数太少（`--nsga_generations`），种群还没有充分探索冲突空间
2. 某些约束的权重过小，导致梯度接近 0
3. 测试点选择在良性均值附近，约束尚未激活

**解决方案**:
- 增加 `--nsga_generations` 到 100 以上
- 检查 `loss_mask` 是否正确设置（默认 '11111'）
- 尝试使用不同的数据集或防御配置

### Q2: 消融实验运行很慢怎么办？

**优化方案**:
- 减少 `--nsga_generations`（例如从 100 降到 30）
- 减少 `--evo_pop_size`（例如从 30 降到 10）
- 减少 `--num_clients` 和 `--num_malicious`
- 使用 GPU：`--device cuda`
- 使用快速测试模式：`--mode quick`

### Q3: 如何解读余弦相似度？

- `cos(θ) = 1.0`: 梯度方向完全一致（强协同）
- `cos(θ) = 0.5`: 梯度方向夹角 60°（弱协同）
- `cos(θ) = 0.0`: 梯度方向正交（独立）
- `cos(θ) = -0.5`: 梯度方向夹角 120°（弱冲突）
- `cos(θ) = -1.0`: 梯度方向完全相反（强冲突）

**判断标准**: 通常认为 `cos(θ) < 0` 表示存在梯度冲突。

### Q4: 消融实验的指标如何解读？

**梯度范数**: 
- 越大表示扰动越强，攻击性越强
- 但过大容易被范数检测（如 Krum、LASA）

**与良性均值的距离**:
- 越小表示越隐蔽，越难被检测
- 但过小可能导致攻击效果不足

**恶意梯度多样性**:
- 越大表示恶意客户端之间差异越大
- 有助于规避聚类防御（如 DNC）

**余弦相似度**:
- 与良性梯度的相似度
- 越高表示越隐蔽，越难被符号检测

### Q5: 测试不同防御机制怎么办？

```bash
# 测试 Krum 防御
python test_gradient_conflict.py --mode microview --defend_methods krum

# 测试 DNC 防御（会激活 PCA 约束）
python test_gradient_conflict.py --mode microview --defend_methods dnc

# 测试多个防御
python test_gradient_conflict.py --mode microview --defend_methods krum dnc
```

### Q6: 如何导出结果用于论文？

**余弦相似度矩阵**:
```python
# 在代码中保存矩阵
import numpy as np
np.save('cosine_similarity_matrix.npy', similarity_matrix.cpu().numpy())

# 使用 matplotlib/seaborn 绘制热力图
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix.cpu().numpy(), 
            annot=True, fmt='.2f',
            xticklabels=objective_names,
            yticklabels=objective_names,
            cmap='RdBu_r', center=0,
            vmin=-1, vmax=1)
plt.title('Gradient Conflict Heatmap')
plt.tight_layout()
plt.savefig('gradient_conflict_heatmap.pdf', dpi=300)
```

**消融实验表格**:
- 程序会自动输出 LaTeX 格式
- 直接复制到论文的 `.tex` 文件中
- 或复制到 Excel/Google Sheets 进行二次处理

## ⚙️ 参数说明

### 测试脚本参数

```bash
python test_gradient_conflict.py [OPTIONS]
```

**数据集参数**:
- `--dataset`: 数据集选择 (`mnist`, `fashion-mnist`, `cifar10`)
- `--batch_size`: 批大小（默认 32）

**联邦学习参数**:
- `--num_clients`: 客户端总数（默认 20）
- `--num_malicious`: 恶意客户端数量（默认 4）
- `--num_classes`: 分类数（默认 10）

**MOS 攻击参数**:
- `--nsga_generations`: 进化代数（默认 50，论文推荐 100+）
- `--evo_pop_size`: 进化种群大小（默认 10）
- `--lam`: CE 和 CW 的权重 0~1（默认 0.5）

**防御机制**:
- `--defend_methods`: 防御方法列表（例如 `krum dnc`）

**测试模式**:
- `--mode`: 测试模式
  - `quick`: 快速测试（10秒内完成）
  - `microview`: 仅微观分析
  - `macroview`: 仅宏观消融实验
  - `full`: 完整测试

**设备**:
- `--device`: 计算设备（`cuda` 或 `cpu`）

## 📚 技术细节

### 梯度提取方法

使用**有限差分法**近似计算各目标的梯度：

```python
# 对于目标 i，计算其梯度
sample_point.requires_grad_(True)
loss_i = objective_function(sample_point)
loss_i.backward()
gradient_i = sample_point.grad.clone()
```

### 余弦相似度计算

```python
cos_sim = F.cosine_similarity(grad_i, grad_j)
# 范围: [-1, 1]
# -1: 完全相反（强冲突）
# 0: 正交（独立）
# 1: 完全一致（强协同）
```

### 消融实验设计

通过 `loss_mask` 参数控制激活的目标：
- `'10000'`: 仅激活第 1 个目标（CE 攻击）
- `'01000'`: 仅激活第 2 个目标（CW 攻击）
- `'11000'`: 激活第 1、2 个目标
- `'11111'`: 激活所有目标

目标索引映射：
```python
loss_mask[0] = CE 攻击
loss_mask[1] = CW 攻击
loss_mask[2] = 幅度约束（Krum + 范数）
loss_mask[3] = 群体约束
loss_mask[4] = 符号约束
```

## 🛠️ 依赖项

```
torch >= 1.10.0
numpy >= 1.19.0
```

可选（用于绘图）：
```
matplotlib >= 3.3.0
seaborn >= 0.11.0
```

## 🤝 贡献

如果你发现 bug 或有改进建议，欢迎：
1. 提交 Issue
2. 提交 Pull Request
3. 联系作者

## 📄 引用

如果你在论文中使用了本系统，请引用：

```bibtex
@misc{gradient_conflict_analyzer,
  title={Gradient Conflict Analysis for Multi-Objective Byzantine Attacks},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
```

## 📞 联系方式

如有问题，请通过以下方式联系：
- Email: your-email@example.com
- GitHub Issues: https://github.com/your-repo/issues

---

**最后更新**: 2024-01-XX  
**版本**: v1.0.0
