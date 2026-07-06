#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终验证：所有 4 处维度修复是否正确
"""

import sys
import os

def check_code_consistency():
    """检查代码中所有 flatten 操作是否一致处理 num_batches_tracked"""
    print("="*70)
    print("检查代码一致性：num_batches_tracked 处理")
    print("="*70)

    try:
        with open('algorithms/attack/mos.py', 'r', encoding='utf-8') as f:
            code = f.read()

        # 检查 4 处关键位置
        checks = {
            "第1处 - 输出层索引计算": (
                "for k, v in all_updates[0].items():" in code and
                code.count("if 'num_batches_tracked' in k:") >= 2
            ),
            "第2处 - layer_dims 计算": (
                "layer_dims = []" in code and
                "if 'num_batches_tracked' in k:" in code
            ),
            "第3处 - all_updates_flatten": (
                "if 'num_batches_tracked' not in k" in code
            ),
            "第4处 - g_ce/g_cw 提取": (
                "if 'num_batches_tracked' in n:" in code and
                code.count("if 'num_batches_tracked' in n:") >= 2
            ),
        }

        all_pass = True
        for check_name, result in checks.items():
            status = "[PASS]" if result else "[FAIL]"
            print(f"{status} {check_name}")
            if not result:
                all_pass = False

        print("\n" + "="*70)
        if all_pass:
            print("[PASS] 所有 4 处 num_batches_tracked 处理一致！")
        else:
            print("[FAIL] 部分位置处理不一致")
        print("="*70)

        return all_pass

    except Exception as e:
        print(f"[FAIL] 检查失败: {e}")
        return False

def main():
    print("\nMOS 攻击维度修复 - 最终验证")
    print("="*70)
    print("本次修复共涉及 4 处代码：")
    print("  1. 输出层索引计算（idx_w_start 等）")
    print("  2. layer_dims 计算")
    print("  3. all_updates_flatten 计算")
    print("  4. g_ce/g_cw 梯度提取")
    print("="*70)
    print()

    success = check_code_consistency()

    print("\n" + "="*70)
    print("验证结果")
    print("="*70)

    if success:
        print("[SUCCESS] 所有修复已完成且一致！")
        print("\n现在可以运行测试：")
        print("python main.py --dataset cifar --attack mos_attack \\")
        print("  --defend1 rlr --defend2 tr_mean --defend3 dnc \\")
        print("  --num_attackers 25 --gpu 4 --loss_mask 11000 \\")
        print("  --mos_conv_sparsity 0.3")
        print("\n预期结果：")
        print("  - 维度：benign_mean (11173962) == g_ce (11173962) == g_cw (11173962)")
        print("  - 没有 RuntimeError")
        print("  - 看到 [MOS LOG] DNC 模式启动")
        return 0
    else:
        print("[FAILED] 部分修复不完整，请检查")
        return 1

if __name__ == "__main__":
    sys.exit(main())
