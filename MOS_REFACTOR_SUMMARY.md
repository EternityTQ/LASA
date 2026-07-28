# MOS.py 重构完成总结

## 重构时间
2026-07-26

## 重构目标
将 MOS 攻击从 NSGA-III 多目标优化（5-7个目标）重构为基于打分系统的 NSGA-II 双目标优化

## 核心改动

### 1. 打分系统 (Scoring System)
**新增函数**: `compute_constraint_score()` (行133-166)

将所有约束性损失转换为得分（0-1范围）：
- **Sigmoid映射** (默认): 平滑过渡，梯度稳定
- **ReLU映射**: 线性映射，计算快速
- **Linear映射**: 分段线性，兼顾两者

公式：
```python
# Sigmoid (默认)
score = sigmoid(-k * (loss - threshold))

# ReLU
score = max(0, (threshold - loss) / threshold)
```

### 2. 良性参考预计算系统
**新增代码段**: 行382-557

在进化循环外部预先计算：
- **PCA主成分**: 使用SVD提取前5个主成分
- **子空间采样**: 预生成3组随机采样的子空间基向量
- **约束阈值**: 每个约束loss的统计阈值 (mean + k*std)
  - k=2.0 (默认，覆盖95%良性梯度)
  - 可通过 `--constraint_k_sigma` 参数调整

**性能提升**: 避免每代重复计算SVD，预期15-25%加速

### 3. NSGA-II 选择机制
**新增函数**:
- `crowding_distance()` (行618-667): 计算拥挤度距离
- `nsga2_select()` (行669-713): NSGA-II选择（非支配排序+拥挤度）

**移除函数**:
- `generate_reference_directions()` - NSGA-III参考方向生成
- `niche_selection()` - NSGA-III生态位选择
- `nsga3_select()` - NSGA-III完整选择机制

### 4. 双目标系统
**新增函数**: `compute_objectives()` (行715-871)

**目标1 - 识别率得分** (Recognition Score):
- 聚合所有约束得分的加权和
- 约束类型：Krum半径、Box边界、符号一致性、PCA投影、子空间鲁棒性、群体聚类
- 默认权重：所有为1.0（群体约束0.5）

**目标2 - 破坏性** (Destructiveness):
- 与指导梯度的对齐度（余弦相似度）
- 负号转换：最大化对齐 = 最小化负对齐

**返回值**:
```python
objectives: (2, P)  # 两个目标，P个个体
scores: dict        # 每个约束的得分字典
losses: dict        # 每个约束的原始loss字典
```

### 5. 进化循环重构
**修改位置**: 行945-1047

**第一处评估** (当前种群):
```python
objectives_current, scores_current, losses_current = compute_objectives(
    malicious_set, benign_refs, constraint_thresholds,
    g_combined_unit, score_mode=score_mode
)
parent_idx = nsga2_select(objectives_current, malicious_set, EVOLUTION_POP_SIZE)
```

**第二处评估** (合并种群):
```python
combined = torch.cat([malicious_set, offspring, archive], dim=0)
objectives_combined, _, _ = compute_objectives(
    combined, benign_refs, constraint_thresholds,
    g_combined_unit, score_mode=score_mode
)
selected_idx = nsga2_select(objectives_combined, combined, EVOLUTION_POP_SIZE)
```

**第三处评估** (最终选择):
```python
final_objectives, final_scores, final_losses = compute_objectives(
    malicious_set, benign_refs, constraint_thresholds,
    g_combined_unit, score_mode=score_mode
)
final_fronts = nondominated_sort(final_objectives)
best_idx = final_fronts[0][0]
```

### 6. 移除的组件
- **LossNormalizer类**: 不再需要动态归一化（打分系统已归一化到[0,1]）
- **loss_mask参数处理**: 所有约束统一纳入打分系统
- **旧的compute_losses函数**: 被compute_objectives替代
- **旧的compute_pca_constraint函数**: 使用预计算的PCA主成分
- **旧的compute_subspace_constraint函数**: 使用预计算的子空间

### 7. 新增日志输出
**循环开始日志** (行942-943):
```
[MOS LOG] 🧬 开始进化循环，代数=100，种群大小=30
[MOS LOG] 📊 打分系统：sigmoid映射，阈值系数k=2.0
```

**进度日志** (每10代，行1041-1046):
```
[MOS LOG]   Generation 10/100: 识别率=0.845, 破坏性=0.723
```

**最终日志** (行1063-1066):
```
[MOS LOG] 🏆 进化完成！最优模板索引: 5
[MOS LOG]   最优个体识别率得分: 0.891
[MOS LOG]   最优个体破坏性: 0.756
[MOS LOG] 📋 将最优模板复制 20 份并添加微小噪声以规避聚类检测...
```

## 新增命令行参数

在 `main.py` 中添加 (行46-50):

```python
parser.add_argument('--score_mode', type=str, default='sigmoid', 
                    choices=['sigmoid', 'relu', 'linear'],
                    help='Scoring function mapping mode')
parser.add_argument('--constraint_k_sigma', type=float, default=2.0,
                    help='Threshold coefficient (threshold = mean + k * std)')
```

## 使用示例

### 基础使用（默认sigmoid映射）
```bash
python main.py \
    --dataset cifar \
    --attack mos_attack \
    --defend1 rlr --defend2 tr_mean --defend3 dnc \
    --num_attackers 25 \
    --gpu 0
```

### 使用ReLU映射
```bash
python main.py \
    --dataset cifar \
    --attack mos_attack \
    --defend1 rlr --defend2 tr_mean --defend3 dnc \
    --num_attackers 25 \
    --score_mode relu \
    --gpu 0
```

### 调整阈值系数（更严格）
```bash
python main.py \
    --dataset cifar \
    --attack mos_attack \
    --defend1 rlr --defend2 tr_mean --defend3 dnc \
    --num_attackers 25 \
    --constraint_k_sigma 1.5 \
    --gpu 0
```

## 兼容性保证

1. **函数签名不变**: `mos_attack()` 的输入输出完全兼容旧版本
2. **返回值不变**: 仍返回 `(all_updates, historical_perturbation)`
3. **参数向后兼容**: 所有新参数都有默认值
4. **防御兼容**: DNC/Krum/RLR等防御机制的适配逻辑保持不变

## 技术优势

### 性能优化
- **预计算策略**: SVD/PCA只计算一次，避免每代重复
- **计算复杂度**: NSGA-II选择 O(MN²) < NSGA-III O(MN² + HN)
- **预期加速**: 15-25% (主要来自预计算和简化的选择机制)

### 算法优势
- **目标清晰**: 识别率和破坏性两个维度更直观
- **收敛稳定**: 双目标Pareto前沿更容易收敛
- **可解释性**: 打分系统提供详细的约束满足情况

### 可扩展性
- **打分函数插件化**: 支持3种映射，易于添加新函数
- **约束模块化**: 每个约束独立计算，方便增删
- **权重可配置**: 预留权重接口，可后续调参

## 验证建议

由于本地没有合适的Python环境，建议在服务器上进行以下验证：

### 1. 语法检查
```bash
python -c "import ast; ast.parse(open('algorithms/attack/mos.py', encoding='utf-8').read())"
```

### 2. 导入测试
```bash
python -c "from algorithms.attack.mos import mos_attack; print('Import successful')"
```

### 3. 功能测试
```bash
# 运行测试脚本
python test_mos_improvements.py

# 或直接运行一个简单配置
python main.py --dataset cifar --attack mos_attack --defend1 fedavg --num_attackers 10 --gpu 0
```

### 4. 对比测试
建议保留旧版本 `mos.py` 的备份，对比：
- 攻击成功率
- 运行时间
- 最终梯度的统计特性

## 注意事项

1. **阈值k的选择**:
   - k=2.0 (默认): 覆盖95%良性梯度，平衡严格性
   - k=1.5: 更严格，识别率得分更难获得
   - k=3.0: 更宽松，约束较弱

2. **打分函数的选择**:
   - **Sigmoid** (推荐): 平滑梯度，适合优化
   - **ReLU**: 计算快，但阈值处有突变
   - **Linear**: 折中方案

3. **约束权重调整**:
   - 默认权重在 `compute_objectives()` 中定义
   - 可通过修改 `benign_refs['constraint_weights']` 调整

## 文件清单

- `algorithms/attack/mos.py` - 主重构文件 (1100行)
- `main.py` - 添加新参数 (2处修改)
- `MOS_REFACTOR_SUMMARY.md` - 本文档
- `fix_mos_refactor.py` - 自动化修复脚本（未执行，可删除）
- `test_mos_improvements.py` - 测试脚本（需要服务器环境）

## 后续优化建议

1. **自适应阈值**: 根据防御类型动态调整k值
2. **权重学习**: 通过历史攻击成功率学习最优约束权重
3. **混合映射**: 对不同约束使用不同的映射函数
4. **多模态引导**: 在指导梯度生成阶段融入打分系统

---

**重构完成时间**: 2026-07-26  
**重构状态**: ✅ 完成（待服务器环境验证）
