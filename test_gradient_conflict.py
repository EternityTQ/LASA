"""
梯度冲突验证脚本
用于测试 MOS 攻击中不同代理目标之间的梯度冲突
"""

import torch
import numpy as np
import argparse
from algorithms.attack.mos import mos_attack, compute_surrogate_guidance
from algorithms.attack.gradient_conflict_analyzer import (
    GradientConflictAnalyzer,
    AblationTestRunner,
    run_gradient_conflict_analysis
)
from utils.data_pre_process import load_partition
from utils.model_utils import model_setup
import copy


def setup_mock_data(args):
    """
    创建模拟数据用于测试
    """
    # 使用随机数据（避免依赖完整的数据加载）
    if args.dataset in ['mnist', 'fashion-mnist', 'fmnist']:
        img_shape = (1, 28, 28)
    elif args.dataset in ['cifar', 'cifar10', 'noniidcifar']:
        img_shape = (3, 32, 32)
    else:
        img_shape = (3, 32, 32)  # 默认

    poison_images = torch.randn(args.batch_size, *img_shape).to(args.device)
    target_labels = torch.randint(0, args.num_classes, (args.batch_size,)).to(args.device)

    return poison_images, target_labels


def setup_mock_updates(args, model):
    """
    创建模拟的客户端更新（梯度）
    """
    all_updates = []

    # 获取模型参数维度
    state_dict = model.state_dict()

    # 生成良性客户端的梯度（模拟正常训练）
    num_benign = args.num_clients - args.num_malicious

    for i in range(args.num_clients):
        update = {}
        for key, param in state_dict.items():
            if i < args.num_malicious:
                # 恶意客户端：初始化为小随机梯度（稍后会被 MOS 替换）
                update[key] = torch.randn_like(param) * 0.01
            else:
                # 良性客户端：模拟正常的梯度
                update[key] = torch.randn_like(param) * 0.1

        all_updates.append(update)

    return all_updates


def test_microview_conflict_analysis(args):
    """
    任务 1: 微观层面的梯度冲突分析
    提取不同目标的梯度向量，计算余弦相似度
    """
    print("\n" + "="*80)
    print("🔬 任务 1: 微观层面梯度冲突分析")
    print("="*80)

    # 1. 创建模型
    args, model, global_model, model_dim_val = model_setup(args)
    model.eval()

    # 2. 准备数据
    poison_images, target_labels = setup_mock_data(args)

    # 3. 计算代理梯度
    print("\n[步骤 1] 计算代理梯度 (g_ce, g_cw)...")
    criterion_ce = torch.nn.CrossEntropyLoss()
    g_ce, g_cw = compute_surrogate_guidance(
        global_model=model,
        poison_images=poison_images,
        target_labels=target_labels,
        criterion_ce=criterion_ce,
        args=args
    )

    print(f"✅ g_ce 范数: {torch.norm(g_ce).item():.4f}")
    print(f"✅ g_cw 范数: {torch.norm(g_cw).item():.4f}")
    print(f"✅ g_ce 与 g_cw 余弦相似度: {torch.nn.functional.cosine_similarity(g_ce.unsqueeze(0), g_cw.unsqueeze(0)).item():.4f}")

    # 4. 创建模拟的客户端更新
    print("\n[步骤 2] 创建模拟的客户端梯度...")
    all_updates = setup_mock_updates(args, model)

    # 5. 准备 MOS 攻击所需的中间变量
    # 提取并展平所有更新
    all_updates_flatten = []
    layer_dims = []
    idx_current = 0

    for k, v in all_updates[0].items():
        num_params = v.numel()
        layer_dims.append((k, idx_current, idx_current + num_params))
        idx_current += num_params

    for update in all_updates:
        vec = torch.cat([torch.flatten(update[k]) for k in update.keys()])
        all_updates_flatten.append(vec)

    all_stack = torch.stack(all_updates_flatten).to(args.device)

    # 提取良性梯度
    benign_grads = all_stack[args.num_malicious:].detach()
    benign_mean = torch.mean(benign_grads, dim=0)
    benign_std = torch.std(benign_grads, dim=0) + 1e-9

    # 计算约束边界
    dists_benign = torch.norm(benign_grads - benign_mean, dim=1)
    krum_radius = torch.max(dists_benign) * 1.1
    benign_min, _ = torch.min(benign_grads, dim=0)
    benign_max, _ = torch.max(benign_grads, dim=0)
    lower_bound = benign_min
    upper_bound = benign_max

    # 创建 survival_mask（简化版，全1）
    survival_mask = torch.ones_like(benign_mean)

    print(f"✅ 良性客户端数: {benign_grads.shape[0]}")
    print(f"✅ 梯度维度: {benign_mean.numel()}")
    print(f"✅ Krum 半径: {krum_radius.item():.4f}")

    # 6. 运行梯度冲突分析
    print("\n[步骤 3] 运行梯度冲突分析...")
    analyzer, stats = run_gradient_conflict_analysis(
        all_updates=all_updates,
        args=args,
        malicious_attackers_this_round=args.num_malicious,
        g_ce=g_ce,
        g_cw=g_cw,
        benign_grads=benign_grads,
        benign_mean=benign_mean,
        benign_std=benign_std,
        krum_radius=krum_radius,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        survival_mask=survival_mask,
        layer_dims=layer_dims
    )

    return analyzer, stats


def test_macroview_ablation_study(args):
    """
    任务 2: 宏观层面的控制变量消融实验
    通过控制 loss_mask 观察不同目标的影响
    """
    print("\n" + "="*80)
    print("🧪 任务 2: 宏观层面控制变量消融实验")
    print("="*80)

    # 1. 创建模型
    args, model, global_model, model_dim_val = model_setup(args)
    model.eval()

    # 2. 准备数据
    poison_images, target_labels = setup_mock_data(args)

    # 3. 计算代理梯度
    print("\n[步骤 1] 计算代理梯度...")
    criterion_ce = torch.nn.CrossEntropyLoss()
    g_ce, g_cw = compute_surrogate_guidance(
        global_model=model,
        poison_images=poison_images,
        target_labels=target_labels,
        criterion_ce=criterion_ce,
        args=args
    )

    print(f"✅ 代理梯度已计算完成")

    # 4. 创建模拟的客户端更新
    print("\n[步骤 2] 创建模拟的客户端梯度...")
    all_updates = setup_mock_updates(args, model)
    print(f"✅ 已创建 {len(all_updates)} 个客户端更新")

    # 5. 运行消融实验
    print("\n[步骤 3] 运行消融实验...")
    print(f"  - 恶意客户端数: {args.num_malicious}")
    print(f"  - 进化代数: {args.nsga_generations}")
    print(f"  - 种群大小: {getattr(args, 'evo_pop_size', 10)}")

    ablation_runner = AblationTestRunner(args, device=args.device)
    results = ablation_runner.run_ablation_study(
        all_updates=all_updates,
        malicious_attackers_this_round=args.num_malicious,
        g_ce=g_ce,
        g_cw=g_cw,
        historical_pop=None
    )

    return ablation_runner, results


def quick_test(args):
    """
    快速测试：运行简化版本的梯度冲突分析
    用于验证代码正确性，减少计算时间
    """
    print("\n" + "="*80)
    print("⚡ 快速测试模式")
    print("="*80)

    # 减少计算量
    args.batch_size = 8
    args.num_clients = 10
    args.num_malicious = 2
    args.nsga_generations = 10
    args.evo_pop_size = 5

    print(f"配置: batch_size={args.batch_size}, num_clients={args.num_clients}, "
          f"num_malicious={args.num_malicious}, generations={args.nsga_generations}")

    # 运行微观分析
    print("\n" + "-"*80)
    print("运行微观分析...")
    print("-"*80)
    analyzer, stats = test_microview_conflict_analysis(args)

    print("\n✅ 快速测试完成！")
    print(f"   检测到 {len(stats['conflict_pairs'])} 对梯度冲突")
    print(f"   冲突比例: {stats['conflict_ratio']*100:.2f}%")

    return analyzer, stats


def main():
    parser = argparse.ArgumentParser(description='梯度冲突验证测试')

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fashion-mnist', 'cifar10'],
                       help='数据集')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批大小')

    # 联邦学习参数
    parser.add_argument('--num_clients', type=int, default=20,
                       help='客户端总数')
    parser.add_argument('--num_malicious', type=int, default=4,
                       help='恶意客户端数量')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='分类数')

    # MOS 攻击参数
    parser.add_argument('--nsga_generations', type=int, default=50,
                       help='NSGA-III 进化代数')
    parser.add_argument('--evo_pop_size', type=int, default=10,
                       help='进化种群大小')
    parser.add_argument('--lam', type=float, default=0.5,
                       help='CE 和 CW 的权重 (0~1)')

    # 防御机制
    parser.add_argument('--defend_methods', type=str, nargs='+', default=[],
                       help='防御方法列表，例如 ["krum", "dnc"]')

    # 测试模式
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'microview', 'macroview', 'full'],
                       help='测试模式: quick(快速), microview(微观), macroview(宏观), full(完整)')

    # 设备
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='计算设备')

    args = parser.parse_args()

    # 根据数据集设置模型参数
    if args.dataset == 'mnist':
        args.model = 'cnn'
        args.num_classes = 10
    elif args.dataset == 'fashion-mnist':
        args.model = 'cnn'
        args.num_classes = 10
    elif args.dataset == 'cifar10':
        args.model = 'resnet18'
        args.num_classes = 10

    print("="*80)
    print("🚀 梯度冲突验证测试启动")
    print("="*80)
    print(f"配置:")
    print(f"  数据集: {args.dataset}")
    print(f"  模型: {args.model}")
    print(f"  客户端总数: {args.num_clients}")
    print(f"  恶意客户端数: {args.num_malicious}")
    print(f"  进化代数: {args.nsga_generations}")
    print(f"  种群大小: {args.evo_pop_size}")
    print(f"  防御机制: {args.defend_methods if args.defend_methods else '无'}")
    print(f"  设备: {args.device}")
    print(f"  测试模式: {args.mode}")
    print("="*80)

    if args.mode == 'quick':
        # 快速测试
        analyzer, stats = quick_test(args)

    elif args.mode == 'microview':
        # 仅运行微观分析
        analyzer, stats = test_microview_conflict_analysis(args)

    elif args.mode == 'macroview':
        # 仅运行宏观消融实验
        ablation_runner, results = test_macroview_ablation_study(args)

    elif args.mode == 'full':
        # 运行完整测试
        print("\n📌 运行完整测试（微观 + 宏观）")

        # 微观分析
        analyzer, stats = test_microview_conflict_analysis(args)

        # 宏观消融实验
        ablation_runner, results = test_macroview_ablation_study(args)

    print("\n" + "="*80)
    print("✅ 所有测试完成！")
    print("="*80)


if __name__ == '__main__':
    main()
