#!/usr/bin/env python3
"""
MOS 攻击改进测试脚本
用于验证 Priority 1 修改的效果
"""

import subprocess
import sys

def run_test(test_name, description, args):
    """运行单个测试"""
    print("\n" + "="*60)
    print(f"[{test_name}] {description}")
    print("="*60)
    print(f"配置: {args}")
    print()

    cmd = ["python", "main.py"] + args.split()

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ {test_name} 完成")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {test_name} 失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {test_name} 被用户中断")
        return False

    return True

def main():
    print("="*60)
    print("MOS Attack Priority 1 Improvements Test")
    print("="*60)

    tests = [
        {
            "name": "Test 1",
            "description": "基础配置 - 测试梯度覆盖率提升",
            "args": "--dataset cifar --attack mos_attack --defend1 rlr --defend2 tr_mean --defend3 dnc --num_attackers 25 --gpu 4 --loss_mask 11000 --mos_conv_sparsity 0.3"
        },
        {
            "name": "Test 2",
            "description": "DNC-aware 掩码测试",
            "args": "--dataset cifar --attack mos_attack --defend1 rlr --defend2 tr_mean --defend3 dnc --num_attackers 25 --gpu 4 --loss_mask 11000 --mos_conv_sparsity 0.3 --use_dnc_aware_mask 1"
        },
        {
            "name": "Test 3",
            "description": "PCA 约束测试",
            "args": "--dataset cifar --attack mos_attack --defend1 rlr --defend2 tr_mean --defend3 dnc --num_attackers 25 --gpu 4 --loss_mask 1100100 --mos_conv_sparsity 0.3 --use_dnc_aware_mask 1"
        },
        {
            "name": "Test 4",
            "description": "完整配置 - 所有 Priority 1 改进",
            "args": "--dataset cifar --attack mos_attack --defend1 rlr --defend2 tr_mean --defend3 dnc --num_attackers 25 --gpu 4 --loss_mask 1100110 --mos_conv_sparsity 0.3 --use_dnc_aware_mask 1 --enable_subspace_constraint 1"
        }
    ]

    results = []
    for test in tests:
        success = run_test(test["name"], test["description"], test["args"])
        results.append((test["name"], success))

    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{name}: {status}")

    print("="*60)
    print("所有测试完成！")
    print("="*60)

if __name__ == "__main__":
    main()
