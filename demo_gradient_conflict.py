"""
简化示例：快速验证梯度冲突分析功能

这个脚本展示如何在最小化的环境中运行梯度冲突分析，
无需完整的联邦学习框架。
"""

import torch
import torch.nn as nn
import sys
sys.path.append('.')

from algorithms.attack.gradient_conflict_analyzer import GradientConflictAnalyzer


def create_simple_model(input_dim=784, hidden_dim=128, num_classes=10):
    """创建一个简单的MLP模型用于演示"""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes)
    )


def demonstrate_gradient_conflict():
    """演示梯度冲突分析的核心功能"""

    print("="*80)
    print("🔬 梯度冲突分析 - 简化演示")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}\n")

    # 1. 创建模型和数据
    print("[步骤 1] 创建模型和数据...")
    model = create_simple_model().to(device)
    batch_size = 32
    X = torch.randn(batch_size, 784).to(device)
    y = torch.randint(0, 10, (batch_size,)).to(device)

    # 2. 模拟良性梯度（来自多个客户端）
    print("[步骤 2] 模拟良性客户端梯度...")
    num_benign = 10
    benign_grads_list = []

    for i in range(num_benign):
        model.zero_grad()
        outputs = model(X)
        loss = nn.CrossEntropyLoss()(outputs, y)
        loss.backward()

        # 提取梯度向量
        grad_vec = torch.cat([p.grad.flatten() for p in model.parameters()])
        benign_grads_list.append(grad_vec)

    benign_grads = torch.stack(benign_grads_list)
    benign_mean = benign_grads.mean(dim=0)
    benign_std = benign_grads.std(dim=0) + 1e-9

    print(f"✅ 良性客户端数: {num_benign}")
    print(f"✅ 梯度维度: {benign_mean.numel()}")
    print(f"✅ 良性梯度范数: {torch.norm(benign_mean).item():.4f}\n")

    # 3. 计算约束边界
    print("[步骤 3] 计算约束边界...")
    dists = torch.norm(benign_grads - benign_mean, dim=1)
    krum_radius = dists.max() * 1.1
    lower_bound = benign_grads.min(dim=0)[0]
    upper_bound = benign_grads.max(dim=0)[0]

    print(f"✅ Krum 半径: {krum_radius.item():.4f}\n")

    # 4. 创建代理梯度（攻击目标）
    print("[步骤 4] 计算代理梯度...")
    model.zero_grad()
    outputs = model(X)

    # CE 攻击
    loss_ce = nn.CrossEntropyLoss()(outputs, y)
    loss_ce.backward(retain_graph=True)
    g_ce = torch.cat([p.grad.clone().flatten() for p in model.parameters()])

    # CW 攻击
    model.zero_grad()
    correct_logits = torch.gather(outputs, 1, y.unsqueeze(1)).squeeze(1)
    outputs_clone = outputs.clone()
    outputs_clone.scatter_(1, y.unsqueeze(1), -1e4)
    max_other_logits = outputs_clone.max(dim=1)[0]
    loss_cw = torch.relu(max_other_logits - correct_logits + 20.0).mean()
    loss_cw.backward()
    g_cw = torch.cat([p.grad.clone().flatten() for p in model.parameters()])

    # 归一化
    g_ce_unit = g_ce / (torch.norm(g_ce) + 1e-9)
    g_cw_unit = g_cw / (torch.norm(g_cw) + 1e-9)
    g_combined_unit = (0.5 * g_ce_unit + 0.5 * g_cw_unit)
    g_combined_unit = g_combined_unit / (torch.norm(g_combined_unit) + 1e-9)

    print(f"✅ g_ce 范数: {torch.norm(g_ce).item():.4f}")
    print(f"✅ g_cw 范数: {torch.norm(g_cw).item():.4f}")
    print(f"✅ CE-CW 余弦相似度: {torch.nn.functional.cosine_similarity(g_ce.unsqueeze(0), g_cw.unsqueeze(0)).item():.4f}\n")

    # 5. 创建分析器并提取各目标的梯度
    print("[步骤 5] 提取各目标的梯度...")
    analyzer = GradientConflictAnalyzer(device=device)

    # 创建一个测试点
    test_point = benign_mean.clone().detach().requires_grad_(True)
    pop = test_point.unsqueeze(0)

    # 手动构建 layer_dims（简化版）
    layer_dims = []
    idx = 0
    for name, param in model.named_parameters():
        layer_dims.append((name, idx, idx + param.numel()))
        idx += param.numel()

    survival_mask = torch.ones_like(benign_mean)

    # 提取各目标的梯度
    gradients = analyzer.compute_objective_gradients(
        pop=pop,
        benign_mean=benign_mean,
        benign_grads=benign_grads,
        benign_std=benign_std,
        krum_radius=krum_radius,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        g_ce_unit=g_ce_unit,
        g_cw_unit=g_cw_unit,
        g_combined_unit=g_combined_unit,
        survival_mask=survival_mask,
        layer_dims=layer_dims,
        use_dnc=False
    )

    print(f"✅ 已提取 {len(gradients)} 个目标的梯度\n")

    # 6. 计算余弦相似度矩阵
    print("[步骤 6] 计算余弦相似度矩阵...")
    similarity_matrix, objective_names = analyzer.compute_cosine_similarity_matrix(gradients)

    # 7. 分析冲突
    print("[步骤 7] 分析梯度冲突...")
    stats = analyzer.log_conflict_analysis(similarity_matrix, objective_names)

    # 8. 输出关键发现
    print("\n" + "="*80)
    print("📊 关键发现总结")
    print("="*80)

    if stats['conflict_pairs']:
        print(f"\n✅ 检测到 {len(stats['conflict_pairs'])} 对梯度冲突！")
        print(f"   冲突比例: {stats['conflict_ratio']*100:.2f}%")
        print(f"\n最严重的3对冲突:")
        sorted_conflicts = sorted(stats['conflict_pairs'], key=lambda x: x['cosine_similarity'])
        for i, pair in enumerate(sorted_conflicts[:3], 1):
            print(f"   {i}. {pair['objective_1']} ↔ {pair['objective_2']}")
            print(f"      余弦相似度: {pair['cosine_similarity']:.4f}, 夹角: {pair['angle_degree']:.1f}°")
    else:
        print("\n⚠️  未检测到梯度冲突（所有目标对的余弦相似度 ≥ 0）")
        print("   这可能是因为:")
        print("   - 测试点选择在良性均值附近")
        print("   - 约束条件较弱")
        print("   - 需要更多的进化代数来探索冲突空间")

    print("\n" + "="*80)
    print("✅ 演示完成！")
    print("="*80)

    return analyzer, stats


def demonstrate_simple_ablation():
    """演示简化版的消融实验"""

    print("\n\n" + "="*80)
    print("🧪 消融实验 - 简化演示")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模拟不同 loss_mask 配置的效果
    configs = [
        ('CE_Only', '10000'),
        ('CW_Only', '01000'),
        ('Constraints_Only', '00111'),
        ('All_Objectives', '11111'),
    ]

    model = create_simple_model().to(device)
    X = torch.randn(32, 784).to(device)
    y = torch.randint(0, 10, (32,)).to(device)

    # 生成良性梯度
    num_benign = 10
    benign_grads_list = []
    for _ in range(num_benign):
        model.zero_grad()
        loss = nn.CrossEntropyLoss()(model(X), y)
        loss.backward()
        grad_vec = torch.cat([p.grad.flatten() for p in model.parameters()])
        benign_grads_list.append(grad_vec)

    benign_grads = torch.stack(benign_grads_list)
    benign_mean = benign_grads.mean(dim=0)

    print("\n不同配置下的模拟指标:")
    print("-"*80)
    print(f"{'配置':<20} {'Loss Mask':<12} {'梯度范数':<15} {'与良性距离':<15}")
    print("-"*80)

    for name, mask in configs:
        # 模拟：根据 mask 生成不同强度的恶意梯度
        num_active = mask.count('1')
        strength = num_active * 0.3  # 激活的目标越多，梯度越强

        malicious_grad = benign_mean + torch.randn_like(benign_mean) * strength
        mal_norm = torch.norm(malicious_grad).item()
        distance = torch.norm(malicious_grad - benign_mean).item()

        print(f"{name:<20} {mask:<12} {mal_norm:>14.2f} {distance:>14.2f}")

    print("-"*80)
    print("\n💡 提示: 这是简化演示，实际的消融实验需要运行完整的 MOS 攻击")
    print("   使用 test_gradient_conflict.py --mode macroview 运行完整实验")
    print("="*80)


if __name__ == '__main__':
    # 运行梯度冲突分析演示
    analyzer, stats = demonstrate_gradient_conflict()

    # 运行简化的消融实验演示
    demonstrate_simple_ablation()

    print("\n" + "="*80)
    print("📝 下一步:")
    print("="*80)
    print("1. 运行完整测试: python test_gradient_conflict.py --mode quick")
    print("2. 运行微观分析: python test_gradient_conflict.py --mode microview")
    print("3. 运行消融实验: python test_gradient_conflict.py --mode macroview")
    print("4. 查看使用指南: python GRADIENT_CONFLICT_GUIDE.py")
    print("="*80)
