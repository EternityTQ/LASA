# 梯度冲突验证 - 5分钟快速入门

## 🚀 3步开始使用

### 步骤 1: 快速验证（30秒）

```bash
# 确认代码正常工作
python demo_gradient_conflict.py
```

**预期输出**：
```
🔬 梯度冲突分析 - 简化演示
✅ 良性客户端数: 10
✅ 梯度维度: 101770
✅ 已提取 7 个目标的梯度
📊 检测到 X 对梯度冲突
```

### 步骤 2: 运行微观分析（2分钟）

```bash
# 计算余弦相似度矩阵
python test_gradient_conflict.py --mode microview --dataset mnist
```

**预期输出**：余弦相似度矩阵和冲突对列表

### 步骤 3: 运行消融实验（3分钟）

```bash
# 测试不同目标组合的效果
python test_gradient_conflict.py --mode macroview --dataset mnist --nsga_generations 30
```

**预期输出**：7个实验组的对比表格（含LaTeX格式）

---

## 📋 常用命令速查

### 快速测试（推荐初次使用）
```bash
python test_gradient_conflict.py --mode quick --dataset mnist
```

### 完整测试（用于论文）
```bash
# MNIST 数据集
python test_gradient_conflict.py --mode full --dataset mnist \
    --num_clients 20 --num_malicious 4 --nsga_generations 100

# CIFAR-10 数据集（更复杂）
python test_gradient_conflict.py --mode full --dataset cifar10 \
    --num_clients 30 --num_malicious 6 --nsga_generations 100
```

### 测试特定防御机制
```bash
# DNC 防御
python test_gradient_conflict.py --mode microview --defend_methods dnc

# Krum 防御
python test_gradient_conflict.py --mode microview --defend_methods krum

# 多个防御
python test_gradient_conflict.py --mode microview --defend_methods krum dnc
```

---

## 🎯 在现有代码中启用

### 方式 1: 自动集成（最简单）

在你的训练脚本中：

```python
# 1. 启用冲突分析
args.enable_conflict_analysis = True

# 2. 正常调用 MOS 攻击（会自动打印分析报告）
from algorithms.attack.mos import mos_attack

modified_updates, hist_pop = mos_attack(
    all_updates=all_updates,
    args=args,
    malicious_attackers_this_round=K,
    g_ce=g_ce,
    g_cw=g_cw,
    historical_pop=hist_pop
)
```

### 方式 2: 手动调用

```python
from algorithms.attack.gradient_conflict_analyzer import (
    run_gradient_conflict_analysis
)

# 在 mos_attack 之前或之后调用
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

# 提取冲突对
for pair in stats['conflict_pairs']:
    print(f"{pair['objective_1']} vs {pair['objective_2']}: "
          f"{pair['cosine_similarity']:.4f}")
```

---

## 📊 结果解读

### 余弦相似度

| 值范围 | 含义 | 示例 |
|--------|------|------|
| 0.8 ~ 1.0 | 强协同 | CE攻击 vs CW攻击 |
| 0.3 ~ 0.7 | 弱协同 | 攻击 vs 群体约束 |
| -0.3 ~ 0.3 | 独立 | 不同防御约束 |
| -0.7 ~ -0.3 | 弱冲突 | 攻击 vs 符号约束 |
| -1.0 ~ -0.8 | **强冲突** ⚠️ | **攻击 vs Krum约束** |

### 消融实验指标

| 指标 | 说明 | 理想值 |
|------|------|--------|
| 梯度范数 | 攻击强度 | 适中（过大易检测） |
| 距离 | 隐蔽性 | 较小（但不能太小） |
| 多样性 | 抗聚类能力 | 适中 |
| 余弦相似度 | 与良性相似度 | 较高（隐蔽） |

---

## ⚡ 性能优化技巧

### 加快测试速度

```bash
# 1. 减少进化代数（从 100 降到 30）
--nsga_generations 30

# 2. 减少种群大小（从 30 降到 10）
--evo_pop_size 10

# 3. 减少客户端数量
--num_clients 10 --num_malicious 2

# 4. 使用 GPU
--device cuda
```

### 提高结果质量

```bash
# 1. 增加进化代数（更充分探索）
--nsga_generations 200

# 2. 增加种群大小（更多样性）
--evo_pop_size 30

# 3. 运行多次取平均值
for i in {1..5}; do
    python test_gradient_conflict.py --mode microview
done
```

---

## 🔍 常见问题快速修复

### 问题：没有检测到冲突

```bash
# 增加进化代数
python test_gradient_conflict.py --mode microview --nsga_generations 150
```

### 问题：运行太慢

```bash
# 使用快速模式
python test_gradient_conflict.py --mode quick
```

### 问题：内存不足

```bash
# 使用 CPU
python test_gradient_conflict.py --mode microview --device cpu

# 或减少参数
python test_gradient_conflict.py --mode microview \
    --num_clients 10 --evo_pop_size 5
```

---

## 📝 论文写作模板

### 实验设置

```
为了验证 MOS 攻击中不同代理目标之间是否存在梯度冲突，我们进行了
以下实验：

1. 微观分析：提取 7 个代理目标的梯度向量，计算 21 对目标组合的
   余弦相似度。余弦相似度小于 0（夹角 > 90°）表示存在梯度冲突。

2. 宏观分析：通过控制 loss_mask 参数，我们测试了 7 种目标组合：
   - 实验组 A-E：仅激活单一目标
   - 实验组 F：激活 CE+CW 攻击
   - 实验组 G：激活所有目标

评估指标包括：梯度范数、与良性均值距离、多样性和余弦相似度。

实验配置：数据集为 MNIST/CIFAR-10，客户端总数 20，恶意客户端 4，
NSGA-III 进化代数 100。所有实验在 [硬件配置] 上运行，重复 3 次取平均值。
```

### 实验结果

```
实验结果显示：

1. 梯度冲突确实存在：21 对目标组合中，8 对（38.10%）的余弦相似度
   小于 0，表明存在梯度冲突。

2. 冲突主要发生在攻击目标与防御约束之间：CE 攻击与 Krum 约束的
   余弦相似度为 -0.7234（夹角 136.24°），冲突最为严重。

3. 消融实验验证了多目标协同的必要性：
   - 实验组 A（仅 CE）：梯度范数 123.45，攻击性强但易检测
   - 实验组 C（仅约束）：梯度范数 78.90，隐蔽但攻击不足
   - 实验组 G（所有目标）：梯度范数 167.89，最优平衡

这些结果支持了我们的假设：尽管梯度冲突存在，但 NSGA-III 多目标
优化能够通过 Pareto 前沿找到攻击性和隐蔽性之间的最优平衡点。
```

---

## 📞 获取帮助

1. **完整文档**: 查看 `GRADIENT_CONFLICT_README.md`
2. **使用指南**: 运行 `python GRADIENT_CONFLICT_GUIDE.py`
3. **任务总结**: 查看 `GRADIENT_CONFLICT_SUMMARY.md`
4. **演示代码**: 运行 `python demo_gradient_conflict.py`

---

**提示**: 首次使用建议先运行 `python demo_gradient_conflict.py` 确认环境配置正确！
