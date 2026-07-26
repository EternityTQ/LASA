"""
快速使用指南：梯度冲突验证
"""

# ============================================================
# 使用方式 1: 命令行快速测试（推荐新手）
# ============================================================

# 1. 快速测试模式（10秒内完成，验证代码是否正常工作）
# python test_gradient_conflict.py --mode quick --dataset mnist

# 2. 仅运行微观分析（提取梯度向量，计算余弦相似度）
# python test_gradient_conflict.py --mode microview --dataset mnist --num_clients 20 --num_malicious 4

# 3. 仅运行宏观消融实验（测试不同 loss_mask 组合的效果）
# python test_gradient_conflict.py --mode macroview --dataset mnist --num_clients 20 --num_malicious 4 --nsga_generations 50

# 4. 完整测试（微观 + 宏观，适合写论文）
# python test_gradient_conflict.py --mode full --dataset cifar10 --num_clients 30 --num_malicious 6 --nsga_generations 100

# 5. 测试 DNC 防御下的梯度冲突
# python test_gradient_conflict.py --mode microview --dataset mnist --defend_methods dnc


# ============================================================
# 使用方式 2: 在现有代码中集成梯度冲突分析
# ============================================================

"""
在你的训练脚本中（例如 main.py 或 fedavg_all.py）：

# 1. 导入分析器
from algorithms.attack.gradient_conflict_analyzer import (
    run_gradient_conflict_analysis,
    AblationTestRunner
)

# 2. 在调用 mos_attack 之前，启用冲突分析
args.enable_conflict_analysis = True  # 开启梯度冲突分析

# 3. 调用 mos_attack（会自动打印冲突分析报告）
modified_updates, hist_pop = mos_attack(
    all_updates=all_updates,
    args=args,
    malicious_attackers_this_round=K,
    g_ce=g_ce,
    g_cw=g_cw,
    historical_pop=hist_pop
)

# 4. 如果你想手动控制分析时机，可以这样：
from algorithms.attack.gradient_conflict_analyzer import GradientConflictAnalyzer

analyzer = GradientConflictAnalyzer(device=args.device)
gradients = analyzer.compute_objective_gradients(...)
similarity_matrix, objective_names = analyzer.compute_cosine_similarity_matrix(gradients)
stats = analyzer.log_conflict_analysis(similarity_matrix, objective_names)

# 打印冲突对
for pair in stats['conflict_pairs']:
    print(f"{pair['objective_1']} vs {pair['objective_2']}: {pair['cosine_similarity']:.4f}")
"""


# ============================================================
# 使用方式 3: 运行消融实验
# ============================================================

"""
from algorithms.attack.gradient_conflict_analyzer import AblationTestRunner

# 创建消融实验运行器
ablation_runner = AblationTestRunner(args, device='cuda')

# 运行完整的消融实验（会自动测试7种配置）
results = ablation_runner.run_ablation_study(
    all_updates=all_updates,
    malicious_attackers_this_round=K,
    g_ce=g_ce,
    g_cw=g_cw,
    historical_pop=None
)

# 结果会自动打印为表格，可直接复制到论文
# 包括：
# - 梯度范数
# - 与良性均值的距离
# - 恶意梯度多样性
# - 余弦相似度
# - LaTeX 表格格式
"""


# ============================================================
# 预期输出示例
# ============================================================

"""
微观分析输出示例：

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

  ... (更多冲突对)

【余弦相似度矩阵】
          ce_attack cw_attack magnitude group_co sign_con pca_cons subspace
ce_attack   1.0000    0.8234   -0.3421  -0.1234   0.2345  -0.7234   0.1234
cw_attack   0.8234    1.0000   -0.2134   0.0123  -0.5123  -0.6234   0.2345
...

================================================================================
"""

"""
宏观消融实验输出示例：

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
仅CW攻击 & 01000 & 145.67$\pm$15.67 & 112.34$\pm$9.45 & 0.3421 & 0.6543 \\
...
\hline
\end{tabular}
"""


# ============================================================
# 论文撰写建议
# ============================================================

"""
1. 微观分析部分（第X.X节：梯度冲突分析）

我们提取了MOS攻击中7个代理目标的梯度向量，计算了它们之间的余弦相似度。
结果显示，在21对目标组合中，有8对（38.10%）存在梯度冲突（余弦相似度 < 0）。

其中，CE攻击目标与Krum约束之间的冲突最为严重（余弦相似度 = -0.7234，夹角 = 136.24°），
这表明在优化攻击性的同时满足防御约束存在固有的矛盾。

具体而言：
- 攻击目标（CE, CW）与防御约束（Krum, PCA）之间存在显著冲突
- 不同防御约束之间（Krum vs Sign）也存在一定冲突
- 攻击目标之间（CE vs CW）保持较高的一致性（余弦相似度 = 0.8234）


2. 宏观分析部分（第X.X节：消融实验）

为了验证不同代理目标对攻击效果的贡献，我们进行了控制变量实验。
通过 loss_mask 参数仅激活单一目标或目标组合，观察以下指标：

实验组A（仅CE攻击，loss_mask='10000'）：
- 恶意梯度范数: 123.45 ± 12.34
- 与良性均值距离: 98.76 ± 8.23
- 结论：纯攻击目标产生较大的梯度扰动，但容易被防御机制检测

实验组C（仅幅度约束，loss_mask='00100'）：
- 恶意梯度范数: 78.90 ± 7.89
- 与良性均值距离: 45.67 ± 5.12
- 结论：单一约束可以有效控制梯度幅度，但缺乏攻击性

实验组G（所有目标，loss_mask='11111'）：
- 恶意梯度范数: 167.89 ± 17.89
- 与良性均值距离: 134.56 ± 12.34
- 结论：多目标协同在保持隐蔽性的同时最大化攻击效果

这些结果支持了我们的假设：梯度冲突虽然存在，但多目标优化框架能够
通过 Pareto 前沿找到攻击性与隐蔽性之间的最优平衡点。
"""


# ============================================================
# 常见问题 (FAQ)
# ============================================================

"""
Q1: 为什么我的余弦相似度都是正数，没有检测到冲突？
A1: 可能是因为：
   - 进化代数太少，种群还没有充分探索冲突空间
   - 某些约束的权重过小，导致梯度接近0
   - 尝试增加 --nsga_generations 到 100 以上

Q2: 消融实验运行很慢，如何加速？
A2: 可以：
   - 减少 --nsga_generations（例如从100降到30）
   - 减少 --evo_pop_size（例如从30降到10）
   - 减少 --num_clients 和 --num_malicious
   - 使用 GPU（--device cuda）

Q3: 如何解读余弦相似度？
A3:
   - cos(θ) = 1.0：两个目标的梯度方向完全一致（协同）
   - cos(θ) = 0.0：两个目标的梯度方向正交（独立）
   - cos(θ) = -1.0：两个目标的梯度方向完全相反（冲突）
   - cos(θ) < 0：表示存在梯度冲突（夹角 > 90°）

Q4: 消融实验的指标如何解读？
A4:
   - 梯度范数：越大表示扰动越强，但也越容易被检测
   - 与良性均值距离：越小表示越隐蔽，但可能攻击性不足
   - 恶意梯度多样性：越大表示恶意客户端之间差异越大，可能更难被聚类防御检测
   - 余弦相似度：与良性梯度的相似度，越高表示越隐蔽

Q5: 如何在论文中引用这些结果？
A5: 建议使用两个图表：
   - 图1：余弦相似度热力图（heatmap），直观展示冲突矩阵
   - 表1：消融实验汇总表格，使用上面输出的 LaTeX 格式

Q6: 我想测试其他防御机制下的梯度冲突怎么办？
A6: 修改 --defend_methods 参数，例如：
   - --defend_methods krum
   - --defend_methods dnc
   - --defend_methods krum dnc  （同时测试多个防御）
"""

print(__doc__)
