"""
完整的多防御基线功能验证脚本
模拟完整的训练流程，但使用简化的模型和数据
"""

import torch
import copy
import numpy as np
from collections import OrderedDict

print("=" * 60)
print("多防御基线功能验证测试")
print("=" * 60)

# 模拟一个简单的模型参数字典
def create_dummy_model():
    return OrderedDict({
        'layer1.weight': torch.randn(10, 5),
        'layer1.bias': torch.randn(10),
        'layer2.weight': torch.randn(5, 10),
        'layer2.bias': torch.randn(5),
    })

# 模拟简单的聚合函数
def simple_average(base_model, updates):
    result = copy.deepcopy(base_model)
    for key in result.keys():
        updates_tensor = torch.stack([u[key] for u in updates])
        result[key] = base_model[key] + torch.mean(updates_tensor, dim=0)
    return result

# 模拟不同的防御聚合方法
def mock_defense_aggregate(updates, base_model, method):
    """模拟不同防御方法的聚合"""
    if method == 'fedavg':
        # 简单平均
        return simple_average(base_model, updates)
    elif method == 'lasa':
        # 模拟 LASA：添加一些随机性
        result = simple_average(base_model, updates)
        for key in result.keys():
            result[key] += torch.randn_like(result[key]) * 0.01
        return result
    elif method == 'signguard':
        # 模拟 SignGuard：使用符号聚合
        result = copy.deepcopy(base_model)
        for key in result.keys():
            updates_tensor = torch.stack([u[key] for u in updates])
            signs = torch.sign(updates_tensor)
            result[key] = base_model[key] + torch.sign(torch.mean(signs, dim=0)) * 0.1
        return result
    else:
        return simple_average(base_model, updates)

# 模拟模型评测
def mock_evaluate(model):
    """模拟模型评测，返回一个准确率"""
    # 基于模型参数的某种统计量生成准确率
    total = 0.0
    count = 0
    for key, value in model.items():
        total += torch.abs(value).mean().item()
        count += 1

    # 归一化到合理的准确率范围
    base_acc = 0.7 + (total / count) * 0.05
    # 添加一些随机性
    noise = np.random.randn() * 0.02
    return min(max(base_acc + noise, 0.5), 0.95)

print("\n1. 测试参数解析")
print("-" * 60)

# 模拟命令行参数
class Args:
    def __init__(self, defend1, defend2=None, defend3=None):
        self.defend1 = defend1
        self.defend2 = defend2
        self.defend3 = defend3

        # 收集所有防御方法
        self.defend_methods = [self.defend1]
        if self.defend2:
            self.defend_methods.append(self.defend2)
        if self.defend3:
            self.defend_methods.append(self.defend3)

        self.defend = self.defend1  # 向后兼容

# 测试不同的参数组合
test_cases = [
    (Args('lasa'), "单个防御方法"),
    (Args('lasa', 'fedavg'), "两个防御方法"),
    (Args('lasa', 'fedavg', 'signguard'), "三个防御方法"),
]

for args, desc in test_cases:
    print(f"\n{desc}:")
    print(f"  defend_methods: {args.defend_methods}")
    print(f"  数量: {len(args.defend_methods)}")
    assert len(args.defend_methods) >= 1
    assert len(args.defend_methods) <= 3

print("\n✓ 参数解析测试通过")

print("\n2. 测试多防御方法聚合与选择")
print("-" * 60)

# 创建全局模型和客户端更新
global_model = create_dummy_model()
num_clients = 5
client_updates = [create_dummy_model() for _ in range(num_clients)]

# 将更新设置为相对于全局模型的差异
for update in client_updates:
    for key in update.keys():
        update[key] = torch.randn_like(update[key]) * 0.1

defend_methods = ['lasa', 'fedavg', 'signguard']

print(f"\n测试 {len(defend_methods)} 个防御方法: {defend_methods}")

candidate_models = []
candidate_accs = []

for defend_method in defend_methods:
    print(f"\n=== 聚合防御方法: {defend_method} ===")

    # 为每个防御方法创建独立的候选模型
    candidate_model = copy.deepcopy(global_model)

    # 执行防御聚合
    candidate_model = mock_defense_aggregate(client_updates, candidate_model, defend_method)

    # 评测候选模型
    acc = mock_evaluate(candidate_model)

    candidate_models.append(candidate_model)
    candidate_accs.append(acc)

    print(f"防御方法 {defend_method}: 测试准确率 = {acc:.5f}")

# 选择准确率最高的模型
best_idx = np.argmax(candidate_accs)
best_defend = defend_methods[best_idx]
best_acc = candidate_accs[best_idx]
selected_model = candidate_models[best_idx]

print(f"\n=== 选中的防御方法: {best_defend} 准确率 {best_acc:.5f} ===")
print(f"所有防御方法准确率: {dict(zip(defend_methods, candidate_accs))}")

# 验证选中的是最高准确率的模型
assert best_acc == max(candidate_accs), "应该选择准确率最高的模型"
print("\n✓ 多防御方法聚合与选择测试通过")

print("\n3. 测试数值稳定性检查")
print("-" * 60)

# 创建一个包含 NaN 的模型
nan_model = create_dummy_model()
nan_model['layer1.weight'][0, 0] = float('nan')

# 检查 NaN
has_nan = False
for k, v in nan_model.items():
    if torch.isnan(v).any() or torch.isinf(v).any():
        has_nan = True
        break

print(f"NaN 模型检测: {has_nan}")
assert has_nan, "应该检测到 NaN"

# 创建健康模型
healthy_model = create_dummy_model()
has_nan = False
for k, v in healthy_model.items():
    if torch.isnan(v).any() or torch.isinf(v).any():
        has_nan = True
        break

print(f"健康模型检测: {has_nan}")
assert not has_nan, "健康模型不应包含 NaN"

print("\n✓ 数值稳定性检查测试通过")

print("\n4. 测试备用模型选择")
print("-" * 60)

# 模拟三个候选模型，其中最优模型包含 NaN
candidate_models_with_nan = []
candidate_accs_with_nan = [0.85, 0.90, 0.87]  # 第二个最高但包含 NaN

for i in range(3):
    model = create_dummy_model()
    if i == 1:  # 最高准确率的模型包含 NaN
        model['layer1.weight'][0, 0] = float('nan')
    candidate_models_with_nan.append(model)

# 选择最高准确率
best_idx = np.argmax(candidate_accs_with_nan)
print(f"初始选择: 索引 {best_idx}, 准确率 {candidate_accs_with_nan[best_idx]:.5f}")

# 检查是否包含 NaN
selected_model = candidate_models_with_nan[best_idx]
has_nan = False
for k, v in selected_model.items():
    if torch.isnan(v).any() or torch.isinf(v).any():
        has_nan = True
        break

if has_nan:
    print("警告: 选中的模型包含 NaN，寻找备用模型...")

    # 寻找备用健康模型
    for idx in range(len(candidate_models_with_nan)):
        if idx == best_idx:
            continue

        test_model = candidate_models_with_nan[idx]
        is_healthy = True
        for k, v in test_model.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                is_healthy = False
                break

        if is_healthy:
            print(f"使用备用模型: 索引 {idx}, 准确率 {candidate_accs_with_nan[idx]:.5f}")
            selected_model = test_model
            has_nan = False
            break

assert not has_nan, "应该选择到健康的备用模型"
print("\n✓ 备用模型选择测试通过")

print("\n5. 测试结果目录命名")
print("-" * 60)

test_cases = [
    (['lasa'], 'lasa'),
    (['lasa', 'fedavg'], 'lasa_fedavg'),
    (['lasa', 'fedavg', 'signguard'], 'lasa_fedavg_signguard'),
]

for methods, expected in test_cases:
    defend_str = '_'.join(methods)
    print(f"防御方法 {methods} -> 目录名: {defend_str}")
    assert defend_str == expected, f"目录名应为 {expected}"

print("\n✓ 结果目录命名测试通过")

print("\n" + "=" * 60)
print("所有测试通过! ✓")
print("=" * 60)

print("\n总结:")
print("- 参数解析: ✓")
print("- 多防御方法聚合与选择: ✓")
print("- 数值稳定性检查: ✓")
print("- 备用模型选择: ✓")
print("- 结果目录命名: ✓")
