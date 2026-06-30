# 快速开始指南 - 多防御基线功能

## 快速测试

### 1. 验证代码修改（不需要训练）

```bash
# 运行简化测试，验证核心逻辑
python test_simplified.py
```

预期输出：所有测试项显示 `[PASS]`

### 2. 单个防御方法（确保向后兼容）

```bash
python main.py --defend1 lasa --dataset cifar --attack mos_attack --num_attackers 20 --gpu 0
```

### 3. 两个防御方法对比

```bash
python main.py --defend1 lasa --defend2 fedavg --dataset cifar --attack mos_attack --num_attackers 20 --gpu 0
```

### 4. 三个防御方法对比

```bash
python main.py --defend1 lasa --defend2 fedavg --defend3 signguard --dataset cifar --attack mos_attack --num_attackers 20 --gpu 0
```

## 观察要点

### 控制台输出

运行多防御方法时，每轮会看到：

```
Testing 3 defense methods: ['lasa', 'fedavg', 'signguard']

=== Aggregating with defense: lasa ===
Defense lasa: Test Accuracy = 0.xxxxx

=== Aggregating with defense: fedavg ===
Defense fedavg: Test Accuracy = 0.xxxxx

=== Aggregating with defense: signguard ===
Defense signguard: Test Accuracy = 0.xxxxx

=== Selected Defense: xxx with accuracy 0.xxxxx ===
```

### 结果目录

结果会保存在以下目录：
```
./exp_results/cifar/Attack_mos_attack_Raito_20/Defense_lasa_fedavg_signguard/
```

### 日志文件

打开 `results.txt` 查看每轮的详细记录：
```
Round 0: Defense comparison:
  lasa: 0.xxxxx
  fedavg: 0.xxxxx
  signguard: 0.xxxxx
  Selected: xxx with accuracy 0.xxxxx
```

## 常见问题

### Q1: 如何只使用一个防御方法？
A: 只指定 `--defend1` 即可，行为与原代码完全一致。

### Q2: 可以使用超过3个防御方法吗？
A: 目前最多支持3个。如需更多，可以修改代码添加 `--defend4`, `--defend5` 等。

### Q3: 每轮训练时间会增加多少？
A: 对于N个防御方法，每轮大约增加 (N-1) 倍的聚合时间 + N倍的评测时间。实际影响取决于模型大小和测试集大小。

### Q4: 如何选择哪些防御方法组合？
A: 建议：
- 基线对比：`--defend1 fedavg --defend2 lasa`
- 多样性：选择不同类型的防御（如基于聚合的、基于检测的等）
- 针对性：根据攻击类型选择相应的防御方法

### Q5: 结果如何分析？
A: 查看日志文件，关注：
1. 哪个防御方法被选中的频率最高
2. 不同防御方法的准确率差异
3. 是否有防御方法始终表现最好

## 推荐实验流程

### 步骤1：基线测试
```bash
# 测试单个防御方法，建立基线
python main.py --defend1 lasa --dataset cifar --attack mos_attack --num_attackers 20 --gpu 0 --repeat 3
```

### 步骤2：两两对比
```bash
# 对比两个防御方法
python main.py --defend1 lasa --defend2 fedavg --dataset cifar --attack mos_attack --num_attackers 20 --gpu 0 --repeat 3
```

### 步骤3：完整对比
```bash
# 对比三个防御方法
python main.py --defend1 lasa --defend2 fedavg --defend3 signguard --dataset cifar --attack mos_attack --num_attackers 20 --gpu 0 --repeat 3
```

### 步骤4：分析结果
比较三次实验的结果，看：
- 单独使用每个防御方法的准确率
- 两个防御方法对比时的选择情况
- 三个防御方法对比时的选择情况

## 示例完整命令

```bash
# CIFAR-10 + MOS攻击 + 三防御对比
python main.py \
    --dataset cifar \
    --attack mos_attack \
    --num_attackers 20 \
    --defend1 lasa \
    --defend2 fedavg \
    --defend3 signguard \
    --gpu 0 \
    --seed 1 \
    --repeat 3 \
    --batch_size 64 \
    --local_lr 0.01

# CIFAR-100 + AGR攻击 + 两防御对比  
python main.py \
    --dataset cifar100 \
    --attack agrTailoredTrmean \
    --num_attackers 30 \
    --defend1 lasa \
    --defend2 bulyan \
    --gpu 0 \
    --seed 1 \
    --repeat 3
```

## 性能优化建议

如果训练时间过长：
1. 减少 `--repeat` 次数（从3降到1）
2. 减少训练轮数（修改配置文件）
3. 使用更小的测试集进行评测
4. 先用2个防御方法测试，验证后再用3个

## 需要帮助？

- 查看 `MULTI_DEFENSE_README.md` 了解详细功能说明
- 查看 `MODIFICATION_SUMMARY.md` 了解技术实现细节
- 查看 `IMPLEMENTATION_COMPLETE.md` 了解完整的实现总结
