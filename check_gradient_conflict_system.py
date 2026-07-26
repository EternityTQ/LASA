#!/usr/bin/env python
"""
梯度冲突验证系统 - 功能检查脚本

运行此脚本以验证所有组件是否正确安装和配置。
"""

import sys
import os

def check_environment():
    """检查环境配置"""
    print("="*80)
    print("🔍 环境检查")
    print("="*80)

    checks = []

    # 检查 Python 版本
    print("\n[1/7] Python 版本检查...")
    python_version = sys.version_info
    if python_version >= (3, 7):
        print(f"  ✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        checks.append(True)
    else:
        print(f"  ❌ Python 版本过低: {python_version.major}.{python_version.minor}")
        print(f"     需要 Python >= 3.7")
        checks.append(False)

    # 检查 PyTorch
    print("\n[2/7] PyTorch 检查...")
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
        print(f"     CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"     CUDA 版本: {torch.version.cuda}")
            print(f"     GPU 数量: {torch.cuda.device_count()}")
        checks.append(True)
    except ImportError:
        print("  ❌ PyTorch 未安装")
        print("     安装: pip install torch")
        checks.append(False)

    # 检查 NumPy
    print("\n[3/7] NumPy 检查...")
    try:
        import numpy as np
        print(f"  ✅ NumPy {np.__version__}")
        checks.append(True)
    except ImportError:
        print("  ❌ NumPy 未安装")
        print("     安装: pip install numpy")
        checks.append(False)

    # 检查文件结构
    print("\n[4/7] 文件结构检查...")
    required_files = [
        'algorithms/attack/gradient_conflict_analyzer.py',
        'algorithms/attack/mos.py',
        'test_gradient_conflict.py',
        'demo_gradient_conflict.py',
    ]

    all_files_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} 不存在")
            all_files_exist = False

    checks.append(all_files_exist)

    # 检查模块导入
    print("\n[5/7] 模块导入检查...")
    try:
        from algorithms.attack.gradient_conflict_analyzer import (
            GradientConflictAnalyzer,
            AblationTestRunner,
            run_gradient_conflict_analysis
        )
        print("  ✅ GradientConflictAnalyzer")
        print("  ✅ AblationTestRunner")
        print("  ✅ run_gradient_conflict_analysis")
        checks.append(True)
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
        checks.append(False)

    # 检查 MOS 集成
    print("\n[6/7] MOS 集成检查...")
    try:
        from algorithms.attack.mos import mos_attack, compute_surrogate_guidance
        print("  ✅ mos_attack")
        print("  ✅ compute_surrogate_guidance")
        checks.append(True)
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
        checks.append(False)

    # 检查文档
    print("\n[7/7] 文档检查...")
    doc_files = [
        'GRADIENT_CONFLICT_README.md',
        'GRADIENT_CONFLICT_SUMMARY.md',
        'quick_start.md',
    ]

    doc_count = sum(1 for f in doc_files if os.path.exists(f))
    print(f"  ✅ 找到 {doc_count}/{len(doc_files)} 个文档文件")
    checks.append(doc_count >= 2)  # 至少有2个文档

    # 汇总
    print("\n" + "="*80)
    print("📊 检查结果")
    print("="*80)

    passed = sum(checks)
    total = len(checks)

    if passed == total:
        print(f"\n✅ 所有检查通过 ({passed}/{total})")
        print("\n🎉 系统已准备就绪！")
        print("\n下一步:")
        print("  1. 运行快速测试: python demo_gradient_conflict.py")
        print("  2. 查看使用指南: cat quick_start.md")
        return True
    else:
        print(f"\n⚠️  部分检查失败 ({passed}/{total})")
        print("\n请修复上述问题后重新运行此脚本。")
        return False


def run_quick_test():
    """运行快速功能测试"""
    print("\n" + "="*80)
    print("🧪 快速功能测试")
    print("="*80)

    try:
        import torch
        from algorithms.attack.gradient_conflict_analyzer import GradientConflictAnalyzer

        print("\n[测试 1/3] 创建分析器...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        analyzer = GradientConflictAnalyzer(device=device)
        print(f"  ✅ 分析器创建成功 (device: {device})")

        print("\n[测试 2/3] 计算余弦相似度...")
        # 创建测试梯度
        grad1 = torch.randn(1000, device=device)
        grad2 = torch.randn(1000, device=device)
        grad3 = -grad1 + torch.randn(1000, device=device) * 0.1  # 与 grad1 冲突

        gradients = {
            'grad1': grad1,
            'grad2': grad2,
            'grad3': grad3,
        }

        similarity_matrix, objective_names = analyzer.compute_cosine_similarity_matrix(gradients)
        print(f"  ✅ 相似度矩阵计算成功 (shape: {similarity_matrix.shape})")

        print("\n[测试 3/3] 冲突分析...")
        stats = analyzer.analyze_conflicts(similarity_matrix, objective_names)
        print(f"  ✅ 冲突分析完成")
        print(f"     检测到 {len(stats['conflict_pairs'])} 对冲突")
        print(f"     平均相似度: {stats['mean_similarity']:.4f}")

        print("\n" + "="*80)
        print("✅ 所有功能测试通过！")
        print("="*80)

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_next_steps():
    """显示下一步操作"""
    print("\n" + "="*80)
    print("📚 推荐的学习路径")
    print("="*80)

    steps = [
        ("1️⃣  快速演示", "python demo_gradient_conflict.py", "30秒，了解核心功能"),
        ("2️⃣  快速测试", "python test_gradient_conflict.py --mode quick", "1分钟，验证完整流程"),
        ("3️⃣  微观分析", "python test_gradient_conflict.py --mode microview", "2分钟，计算余弦相似度"),
        ("4️⃣  消融实验", "python test_gradient_conflict.py --mode macroview", "5分钟，测试不同配置"),
        ("5️⃣  完整测试", "python test_gradient_conflict.py --mode full --nsga_generations 100", "15分钟，用于论文"),
    ]

    for title, command, desc in steps:
        print(f"\n{title}")
        print(f"  命令: {command}")
        print(f"  说明: {desc}")

    print("\n" + "="*80)
    print("📖 文档资源")
    print("="*80)
    print("\n  • quick_start.md              - 5分钟快速入门")
    print("  • GRADIENT_CONFLICT_README.md - 完整文档")
    print("  • GRADIENT_CONFLICT_SUMMARY.md - 任务总结")
    print("  • GRADIENT_CONFLICT_GUIDE.py   - 使用指南和FAQ")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("🚀 梯度冲突验证系统 - 功能检查")
    print("="*80)
    print("\n此脚本将检查系统配置并运行基本功能测试。\n")

    # 环境检查
    env_ok = check_environment()

    if not env_ok:
        print("\n❌ 环境检查失败，请修复后重试。")
        return 1

    # 询问是否运行功能测试
    print("\n" + "="*80)
    response = input("\n是否运行功能测试？(y/n) [y]: ").strip().lower()

    if response in ['', 'y', 'yes']:
        test_ok = run_quick_test()
        if not test_ok:
            print("\n❌ 功能测试失败。")
            return 1

    # 显示下一步
    show_next_steps()

    print("\n" + "="*80)
    print("✅ 检查完成！系统已准备就绪。")
    print("="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
