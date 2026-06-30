# 多防御基线功能使用说明

## 功能概述

本次更新实现了以下两个主要功能：

1. **支持多个防御基线**：在运行实验时可以同时指定最多3个防御方法
2. **自动选择最优模型**：服务器端会对每个防御基线生成的模型进行评测，自动选择准确率最高的模型作为下一轮的全局模型

## 使用方法

### 1. 使用单个防御方法（向后兼容）

```bash
python main.py --defend1 lasa --gpu 0
```

### 2. 使用两个防御方法

```bash
python main.py --defend1 lasa --defend2 fedavg --gpu 0
```

### 3. 使用三个防御方法

```bash
python main.py --defend1 lasa --defend2 fedavg --defend3 signguard --gpu 0
```

## 参数说明

- `--defend1`: 第一个防御方法（必填）
- `--defend2`: 第二个防御方法（可选）
- `--defend3`: 第三个防御方法（可选）

支持的防御方法包括：
- `fedavg`: FedAvg 基线
- `signguard`: SignGuard
- `dnc`: DnC
- `lasa`: LASA
- `bulyan`: Bulyan
- `tr_mean`: Trimmed Mean
- `multi_krum`: Multi-Krum
- `sparsefed`: SparseFed
- `geomed`: GeoMed
- `rlr`: RLR
- `lfd`: LFD

## 工作流程

当使用多个防御方法时，每轮训练的流程如下：

1. **客户端训练**：所有选中的客户端使用本地数据训练模型
2. **攻击执行**（如果有）：恶意客户端对模型进行攻击
3. **多防御聚合**：
   - 对每个防御方法，使用相同的客户端更新生成一个候选全局模型
   - 在测试集上评测每个候选模型的准确率
4. **模型选择**：选择准确率最高的候选模型作为本轮的全局模型
5. **日志记录**：记录每个防御方法的准确率和最终选择的防御方法

## 输出示例

### 控制台输出

```
=== Testing 3 defense methods: ['lasa', 'fedavg', 'signguard'] ===

=== Aggregating with defense: lasa ===
Defense lasa: Test Accuracy = 0.85234

=== Aggregating with defense: fedavg ===
Defense fedavg: Test Accuracy = 0.82145

=== Aggregating with defense: signguard ===
Defense signguard: Test Accuracy = 0.87321

=== Selected Defense: signguard with accuracy 0.87321 ===
All defense accuracies: {'lasa': 0.85234, 'fedavg': 0.82145, 'signguard': 0.87321}
```

### 日志文件输出

日志文件会记录每轮的防御方法比较结果：

```
Round 0: Defense comparison:
  lasa: 0.85234
  fedavg: 0.82145
  signguard: 0.87321
  Selected: signguard with accuracy 0.87321
```

## 注意事项

1. **性能开销**：使用多个防御方法会增加每轮训练的时间，因为需要对每个防御方法进行模型聚合和评测
2. **内存使用**：同时维护多个候选模型会增加内存使用
3. **结果目录**：使用多个防御方法时，结果会保存在包含所有防御方法名称的目录下，例如：
   ```
   ./exp_results/cifar/Attack_agrTailoredTrmean_Raito_20/Defense_lasa_fedavg_signguard/
   ```

## 示例完整命令

```bash
# 在 CIFAR-10 上测试三个防御方法对抗 AGR 攻击
python main.py \
    --dataset cifar \
    --attack agrTailoredTrmean \
    --num_attackers 20 \
    --defend1 lasa \
    --defend2 fedavg \
    --defend3 signguard \
    --gpu 0 \
    --seed 1 \
    --repeat 3
```

## 技术实现细节

1. **参数处理**：在 `main.py` 中将多个防御参数收集到 `defend_methods` 列表中
2. **模型聚合**：在 `fedavg_all.py` 中对每个防御方法独立进行模型聚合
3. **模型评测**：使用测试集对每个候选模型进行准确率评测
4. **模型选择**：使用 `np.argmax` 选择准确率最高的模型
5. **向后兼容**：当只使用一个防御方法时，行为与原代码完全一致
