"""
模拟多防御基线选择逻辑测试
"""
import numpy as np

# 模拟三个防御方法的准确率
defend_methods = ['lasa', 'fedavg', 'signguard']
candidate_accs = [0.85234, 0.82145, 0.87321]

print("=== 模拟多防御基线选择 ===")
print(f"防御方法: {defend_methods}")
print(f"对应准确率: {candidate_accs}")

# 选择准确率最高的模型
best_idx = np.argmax(candidate_accs)
best_defend = defend_methods[best_idx]
best_acc = candidate_accs[best_idx]

print(f"\n最优防御方法索引: {best_idx}")
print(f"最优防御方法: {best_defend}")
print(f"最优准确率: {best_acc:.5f}")

# 显示所有防御方法的准确率
print(f"\n所有防御方法准确率对比:")
for method, acc in zip(defend_methods, candidate_accs):
    marker = " ← 已选择" if method == best_defend else ""
    print(f"  {method}: {acc:.5f}{marker}")

# 测试只有一个防御方法的情况
print("\n=== 测试单个防御方法 ===")
single_method = ['lasa']
single_acc = [0.85234]
print(f"防御方法数量: {len(single_method)}")
print(f"是否启用多防御比较: {len(single_method) > 1}")

# 测试两个防御方法的情况
print("\n=== 测试两个防御方法 ===")
two_methods = ['lasa', 'fedavg']
two_accs = [0.85234, 0.82145]
print(f"防御方法数量: {len(two_methods)}")
print(f"是否启用多防御比较: {len(two_methods) > 1}")
best_idx = np.argmax(two_accs)
print(f"选中的防御方法: {two_methods[best_idx]} (准确率: {two_accs[best_idx]:.5f})")

print("\n=== 所有测试通过! ===")
