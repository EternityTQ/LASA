# 多防御基线功能修改总结

## 修改文件

### 1. main.py

#### 修改内容：
1. **参数解析（第31-33行）**：
   - 将原来的单个 `--defend` 参数改为 `--defend1`, `--defend2`, `--defend3` 三个参数
   - 第一个防御方法必填，后两个可选

2. **参数处理（第37-50行之后）**：
   - 收集所有非空的防御方法到 `defend_methods` 列表
   - 保持 `defend` 属性用于向后兼容（设为 `defend1`）
   
3. **结果目录命名（第51行）**：
   - 使用所有防御方法的组合名称作为目录名
   - 例如：`Defense_lasa_fedavg_signguard/`

4. **日志输出（第99行）**：
   - 输出所有使用的防御方法而不是单个

### 2. algorithms/engine/fedavg_all.py

#### 修改内容：

1. **防御方法初始化（第170-197行）**：
   - 处理 `defend_methods` 列表
   - 支持向后兼容单个 `defend` 参数
   
2. **多防御方法聚合与评测（第340-450行）**：
   ```python
   if len(defend_methods) > 1:
       # 对每个防御方法：
       # 1. 创建候选模型副本
       # 2. 执行防御聚合
       # 3. 在测试集上评测
       # 4. 记录准确率
       
       # 选择准确率最高的模型
       # 输出对比结果
       # 记录到日志文件
   else:
       # 单个防御方法的原有逻辑
   ```

3. **数值稳定性检查**：
   - 在多防御方法模式下，对选中的最优模型进行 NaN/Inf 检查
   - 如果最优模型有问题，尝试选择其他健康的候选模型
   - 如果所有候选模型都有问题，回退到上一轮模型

4. **日志记录增强**：
   - 记录每个防御方法的评测准确率
   - 记录最终选择的防御方法及其准确率
   - 在多防御方法模式下跳过重复的准确率记录

## 使用示例

### 基本用法

```bash
# 单个防御方法（向后兼容）
python main.py --defend1 lasa --gpu 0

# 两个防御方法
python main.py --defend1 lasa --defend2 fedavg --gpu 0

# 三个防御方法
python main.py --defend1 lasa --defend2 fedavg --defend3 signguard --gpu 0
```

### 完整示例

```bash
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

## 输出说明

### 控制台输出示例

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

t   0: train_loss = 2.135, test_acc = 0.873
```

### 日志文件输出示例

```
Round 0: Defense comparison:
  lasa: 0.85234
  fedavg: 0.82145
  signguard: 0.87321
  Selected: signguard with accuracy 0.87321
```

## 技术细节

### 工作流程

1. **客户端训练**：所有选中的客户端使用本地数据训练模型
2. **模型更新收集**：收集所有客户端的模型更新
3. **攻击执行**（如果有）：恶意客户端对模型进行攻击
4. **多防御聚合与评测**：
   - 对每个防御方法使用相同的客户端更新独立进行聚合
   - 在测试集上评测每个候选模型
   - 选择准确率最高的模型
5. **数值稳定性检查**：
   - 检查选中的模型是否包含 NaN/Inf
   - 如果有问题，选择备用健康模型或回退到上一轮
6. **模型下发**：将选中的最优模型作为下一轮的全局模型

### 关键设计考虑

1. **独立性**：每个防御方法使用相同的输入（local_updates）但独立聚合
2. **公平性**：所有防御方法在相同的数据和条件下评测
3. **健壮性**：包含数值稳定性检查和回退机制
4. **效率**：虽然需要多次聚合和评测，但避免了重复的客户端训练
5. **向后兼容**：单个防御方法时行为与原代码完全一致

### 性能考虑

- **时间开销**：每轮增加 (N-1) 次模型聚合和 N 次模型评测（N为防御方法数量）
- **内存开销**：需要同时维护 N 个候选模型的副本
- **优化空间**：
  - 可以并行化不同防御方法的聚合（当前是顺序执行）
  - 可以在评测中使用子集而不是完整测试集（快速筛选）

## 注意事项

1. 每个防御方法都会对相同的 `local_updates` 进行操作，确保公平比较
2. `sparsefed` 防御方法需要维护 `momentum` 和 `error` 状态，在多防御模式下可能需要额外处理
3. 结果目录会包含所有防御方法的名称，确保不同组合的实验结果分开存储
4. 日志文件会详细记录每轮每个防御方法的准确率，便于后续分析

## 测试验证

已创建测试脚本：
- `test_multi_defense.py`：测试参数解析逻辑
- `test_defense_selection.py`：测试防御方法选择逻辑

两个测试脚本都通过验证。

## 后续改进建议

1. **并行化评测**：使用多进程或多线程并行评测不同防御方法
2. **早停机制**：如果某个防御方法的准确率显著高于其他，可以提前停止评测
3. **自适应选择**：可以根据历史表现动态调整防御方法的权重
4. **更丰富的评测指标**：除了准确率，还可以考虑鲁棒性、收敛速度等指标
