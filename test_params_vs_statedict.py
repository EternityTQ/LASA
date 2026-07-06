#!/usr/bin/env python3
"""
验证 named_parameters() vs state_dict() 的区别
"""

import torch
import torch.nn as nn

# 创建一个简单的模型（包含 BatchNorm）
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.bn = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        return self.fc(self.bn(self.conv(x)).mean(dim=[2, 3]))

model = SimpleModel()

# 比较两种方法
print("="*70)
print("named_parameters() vs state_dict()")
print("="*70)

named_params = {n for n, _ in model.named_parameters()}
state_dict_keys = set(model.state_dict().keys())

print(f"\nnamed_parameters() 数量: {len(named_params)}")
print(f"state_dict() 数量: {len(state_dict_keys)}")

print(f"\n在 state_dict() 中但不在 named_parameters() 中的:")
diff = state_dict_keys - named_params
for name in sorted(diff):
    print(f"  - {name}")

print(f"\n这些就是 buffers（如 num_batches_tracked）")

# 验证维度
print("\n" + "="*70)
print("维度验证")
print("="*70)

# 方法 1：使用 named_parameters()
dim1 = sum(p.numel() for _, p in model.named_parameters())
print(f"named_parameters() 总维度: {dim1}")

# 方法 2：使用 state_dict()
dim2 = sum(v.numel() for v in model.state_dict().values())
print(f"state_dict() 总维度: {dim2}")

print(f"\n差值: {dim2 - dim1} (这就是 buffers 的维度)")

# 解决方案
print("\n" + "="*70)
print("解决方案")
print("="*70)
print("如果 local_updates 来自 state_dict()，那么 g_ce/g_cw 也必须遍历 state_dict()")
print("或者，在提取 g_ce/g_cw 时，为 buffers 补零")
