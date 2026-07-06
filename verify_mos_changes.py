#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOS 攻击改进验证脚本
快速检查代码修改是否正确，无需运行完整训练
"""

import sys
import os
import io

# 设置 stdout 编码为 utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试导入是否正常"""
    print("[Test 1] 检查导入...")
    try:
        from algorithms.attack.mos import compute_surrogate_guidance, mos_attack
        from algorithms.engine.fedavg_all import compute_surrogate_guidance as csg_wrapper
        print("   [PASS] 所有导入成功")
        return True
    except Exception as e:
        print(f"   [FAIL] 导入失败: {e}")
        return False

def test_function_signatures():
    """测试函数签名是否正确"""
    print("\n[Test 2] 检查函数签名...")
    try:
        from algorithms.attack.mos import compute_surrogate_guidance
        import inspect

        sig = inspect.signature(compute_surrogate_guidance)
        params = list(sig.parameters.keys())

        expected_params = ['global_model', 'poison_images', 'target_labels', 'criterion_ce', 'args']

        if params == expected_params:
            print(f"   [PASS] compute_surrogate_guidance 签名正确: {params}")
            return True
        else:
            print(f"   [FAIL] 签名不匹配")
            print(f"      期望: {expected_params}")
            print(f"      实际: {params}")
            return False
    except Exception as e:
        print(f"   [FAIL] 检查失败: {e}")
        return False

def test_args_parsing():
    """测试新增参数是否存在"""
    print("\n[Test 3] 检查新增参数...")
    try:
        new_args = [
            'mos_conv_sparsity',
            'mos_classifier_sparsity',
            'use_dnc_aware_mask',
            'enable_subspace_constraint'
        ]

        all_present = True
        main_content = open('main.py', encoding='utf-8').read()
        for arg in new_args:
            # 检查参数是否在 main.py 中定义
            if f"--{arg}" in main_content:
                print(f"   [PASS] 参数 --{arg} 存在")
            else:
                print(f"   [FAIL] 参数 --{arg} 缺失")
                all_present = False

        return all_present
    except Exception as e:
        print(f"   [FAIL] 检查失败: {e}")
        return False

def test_key_functions():
    """测试关键函数是否存在"""
    print("\n[Test 4] 检查关键函数...")
    try:
        # 检查 mos.py 中是否包含关键代码片段
        mos_code = open('algorithms/attack/mos.py', encoding='utf-8').read()

        checks = {
            "分层稀疏化": "layer_groups" in mos_code and "sparsity_ratio" in mos_code,
            "DNC-aware 掩码": "compute_dnc_sensitive_mask" in mos_code,
            "PCA 约束": "compute_pca_constraint" in mos_code,
            "子空间约束": "compute_subspace_constraint" in mos_code,
            "DNC 检测": "use_dnc" in mos_code or "'dnc' in defend_methods" in mos_code,
        }

        all_present = True
        for name, present in checks.items():
            if present:
                print(f"   [PASS] {name}: 已实现")
            else:
                print(f"   [FAIL] {name}: 未找到")
                all_present = False

        return all_present
    except Exception as e:
        print(f"   [FAIL] 检查失败: {e}")
        return False

def test_loss_mask_expansion():
    """测试 loss_mask 是否扩展到 7 位"""
    print("\n[Test 5] 检查 loss_mask 扩展...")
    try:
        mos_code = open('algorithms/attack/mos.py', encoding='utf-8').read()
        main_code = open('main.py', encoding='utf-8').read()

        # 检查是否有 7 个损失项
        has_7_losses = "l_pca" in mos_code and "l_subspace" in mos_code

        # 检查注释中是否说明了 7 位
        has_comment = "Index 5" in mos_code or "Index 6" in mos_code or "7 个目标" in mos_code or "7 bits" in main_code

        if has_7_losses and has_comment:
            print(f"   [PASS] loss_mask 已扩展到 7 位")
            print(f"   [PASS] 新增 l_pca (Index 5) 和 l_subspace (Index 6)")
            return True
        else:
            print(f"   [FAIL] loss_mask 扩展不完整")
            print(f"      7 个损失项: {has_7_losses}")
            print(f"      注释更新: {has_comment}")
            return False
    except Exception as e:
        print(f"   [FAIL] 检查失败: {e}")
        return False

def main():
    print("="*60)
    print("MOS Attack Priority 1 Improvements - Code Verification")
    print("="*60)

    tests = [
        ("Import Check", test_imports),
        ("Function Signature Check", test_function_signatures),
        ("Parameters Check", test_args_parsing),
        ("Key Functions Check", test_key_functions),
        ("loss_mask Expansion Check", test_loss_mask_expansion),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n[FAIL] {name} execution error: {e}")
            results.append((name, False))

    # Print summary
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{name}: {status}")

    print("="*60)
    print(f"Total: {passed}/{total} passed")

    if passed == total:
        print("All verifications passed! Ready to test.")
        print("\nRecommended test command:")
        print("python main.py --dataset cifar --attack mos_attack --defend1 rlr --defend2 tr_mean --defend3 dnc --num_attackers 25 --gpu 4 --loss_mask 1100110")
    else:
        print("Some verifications failed, please check code modifications.")

    print("="*60)

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
