#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证维度修复是否正确
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_dimension_consistency():
    """测试 g_ce/g_cw 与 benign_mean 的维度是否一致"""
    print("="*60)
    print("测试：维度一致性检查")
    print("="*60)

    try:
        import torch
        from algorithms.attack.mos import compute_surrogate_guidance

        # 模拟一个简单的模型（包含 BatchNorm）
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
                self.bn = torch.nn.BatchNorm2d(16)
                self.fc = torch.nn.Linear(16 * 32 * 32, 10)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x

        model = SimpleModel()
        device = 'cpu'

        # 模拟输入
        images = torch.randn(4, 3, 32, 32)
        target_labels = torch.tensor([0, 1, 2, 3])
        criterion_ce = torch.nn.CrossEntropyLoss()

        # 模拟 args
        class Args:
            mos_conv_sparsity = 0.3
            mos_classifier_sparsity = 1.0

        args = Args()

        # 计算指导梯度
        g_ce, g_cw = compute_surrogate_guidance(model, images, target_labels, criterion_ce, args)

        print(f"g_ce 维度: {g_ce.shape[0]}")
        print(f"g_cw 维度: {g_cw.shape[0]}")

        # 计算 benign_mean 的维度（模拟 mos_attack 中的 flatten 方式）
        model_params = []
        for name, param in model.named_parameters():
            if 'num_batches_tracked' in name:
                continue
            model_params.append(param.flatten())
        benign_mean = torch.cat(model_params)

        print(f"benign_mean 维度: {benign_mean.shape[0]}")

        # 检查维度是否一致
        if g_ce.shape[0] == benign_mean.shape[0] and g_cw.shape[0] == benign_mean.shape[0]:
            print("\n[PASS] ✓ 维度一致性检查通过！")
            print(f"       所有向量维度均为: {benign_mean.shape[0]}")
            return True
        else:
            print(f"\n[FAIL] ✗ 维度不一致！")
            print(f"       g_ce: {g_ce.shape[0]}")
            print(f"       g_cw: {g_cw.shape[0]}")
            print(f"       benign_mean: {benign_mean.shape[0]}")
            return False

    except Exception as e:
        print(f"\n[FAIL] ✗ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_num_batches_tracked_skipped():
    """测试 num_batches_tracked 是否被正确跳过"""
    print("\n" + "="*60)
    print("测试：num_batches_tracked 跳过检查")
    print("="*60)

    try:
        # 检查代码中是否包含跳过逻辑
        with open('algorithms/attack/mos.py', 'r', encoding='utf-8') as f:
            code = f.read()

        # 检查是否有跳过 num_batches_tracked 的逻辑
        skip_patterns = [
            "if 'num_batches_tracked' in n:",
            "if 'num_batches_tracked' in k:"
        ]

        found_count = sum(1 for pattern in skip_patterns if pattern in code)

        # 应该至少有 3 处：mos_attack 中 1 处，compute_surrogate_guidance 中 2 处
        if found_count >= 3:
            print(f"[PASS] ✓ 找到 {found_count} 处 num_batches_tracked 跳过逻辑")
            return True
        else:
            print(f"[FAIL] ✗ 只找到 {found_count} 处跳过逻辑，应该至少 3 处")
            return False

    except Exception as e:
        print(f"[FAIL] ✗ 测试执行失败: {e}")
        return False

if __name__ == "__main__":
    print("MOS 攻击维度修复验证\n")

    test1 = test_num_batches_tracked_skipped()
    test2 = test_dimension_consistency()

    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"num_batches_tracked 跳过检查: {'PASS' if test1 else 'FAIL'}")
    print(f"维度一致性检查: {'PASS' if test2 else 'FAIL'}")

    if test1 and test2:
        print("\n✓ 所有测试通过！维度修复成功。")
        sys.exit(0)
    else:
        print("\n✗ 部分测试失败，请检查修复。")
        sys.exit(1)
