"""
梯度冲突分析模块
用于验证多目标攻击算法中不同代理目标之间的梯度冲突
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import copy


class GradientConflictAnalyzer:
    """
    梯度冲突分析器
    微观层面：提取并计算不同目标之间的余弦相似度
    """

    def __init__(self, device='cpu'):
        self.device = device
        self.conflict_history = []

    def compute_objective_gradients(self, pop, benign_mean, benign_grads, benign_std,
                                    krum_radius, lower_bound, upper_bound,
                                    g_ce_unit, g_cw_unit, g_combined_unit,
                                    survival_mask, layer_dims, use_dnc=False):
        """
        为每个目标计算梯度（通过有限差分法近似）

        返回: Dict[str, torch.Tensor]
            键: 目标名称 ('ce_attack', 'cw_attack', 'magnitude_constraint', ...)
            值: 梯度向量 shape (D,) - 对单个样本点的损失梯度
        """
        # 选择种群中的一个代表性个体（例如中心点）
        sample_point = pop[0].clone().detach().requires_grad_(True)

        gradients = {}
        P = 1  # 单个样本点

        # ============ 目标 1: CE 攻击性 ============
        target_ce = benign_mean + g_ce_unit
        l_ce = torch.norm(sample_point - target_ce)

        if sample_point.grad is not None:
            sample_point.grad.zero_()
        l_ce.backward(retain_graph=True)
        gradients['ce_attack'] = sample_point.grad.clone().flatten()

        # ============ 目标 2: CW 攻击性 ============
        sample_point.grad.zero_()
        target_cw = benign_mean + g_cw_unit
        l_cw = torch.norm(sample_point - target_cw)
        l_cw.backward(retain_graph=True)
        gradients['cw_attack'] = sample_point.grad.clone().flatten()

        # ============ 目标 3: Krum 范数约束 ============
        sample_point.grad.zero_()
        current_dist = torch.norm(sample_point - benign_mean)
        excess_dist = torch.relu(current_dist - krum_radius)
        l_krum = excess_dist ** 2
        l_krum.backward(retain_graph=True)
        gradients['krum_constraint'] = sample_point.grad.clone().flatten()

        # ============ 目标 4: Box 边界约束 ============
        sample_point.grad.zero_()
        excess_lower = torch.relu(lower_bound - sample_point)
        excess_upper = torch.relu(sample_point - upper_bound)
        l_box = torch.norm(excess_lower + excess_upper)
        l_box.backward(retain_graph=True)
        gradients['box_constraint'] = sample_point.grad.clone().flatten()

        # ============ 目标 5: 符号一致性约束 ============
        sample_point.grad.zero_()
        l_sign_total = torch.tensor(0.0, device=self.device, requires_grad=True)
        for _, start_idx, end_idx in layer_dims:
            mal_layer = sample_point[start_idx:end_idx]
            ben_layer_mean = benign_mean[start_idx:end_idx]
            sign_violation = -mal_layer * torch.sign(ben_layer_mean)
            layer_sign_loss = torch.norm(torch.relu(sign_violation))
            l_sign_total = l_sign_total + layer_sign_loss
        l_sign_total.backward(retain_graph=True)
        gradients['sign_constraint'] = sample_point.grad.clone().flatten()

        # ============ 目标 6: PCA 主成分约束 (DNC专用) ============
        if use_dnc:
            sample_point.grad.zero_()
            l_pca = self._compute_pca_loss_single(sample_point, benign_grads, benign_mean)
            l_pca.backward(retain_graph=True)
            gradients['pca_constraint'] = sample_point.grad.clone().flatten()

        # ============ 目标 7: 群体拓扑约束 ============
        sample_point.grad.zero_()
        # 对于单点，使用到良性均值的距离作为代理
        l_group = torch.norm(sample_point - benign_mean)
        l_group.backward(retain_graph=True)
        gradients['group_constraint'] = sample_point.grad.clone().flatten()

        return gradients

    def _compute_pca_loss_single(self, sample_point, benign_grads, benign_mean):
        """计算单个样本点的PCA损失"""
        benign_centered = benign_grads - benign_mean

        try:
            U, S, V = torch.linalg.svd(benign_centered, full_matrices=False)
            num_principal_components = min(5, V.shape[0])
            principal_dirs = V[:num_principal_components, :]

            mal_deviation = sample_point - benign_mean
            projections = torch.matmul(mal_deviation, principal_dirs.T)

            weights = torch.tensor([1.0, 0.5, 0.3, 0.2, 0.1], device=self.device)[:num_principal_components]
            l_pca = torch.sum(projections.abs() * weights)

            return l_pca
        except:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def compute_cosine_similarity_matrix(self, gradients: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[str]]:
        """
        计算所有目标梯度之间的余弦相似度矩阵

        返回:
            similarity_matrix: (N, N) 余弦相似度矩阵
            objective_names: 目标名称列表
        """
        objective_names = list(gradients.keys())
        N = len(objective_names)

        similarity_matrix = torch.zeros(N, N, device=self.device)

        for i, name_i in enumerate(objective_names):
            for j, name_j in enumerate(objective_names):
                grad_i = gradients[name_i]
                grad_j = gradients[name_j]

                # 计算余弦相似度
                cos_sim = F.cosine_similarity(grad_i.unsqueeze(0), grad_j.unsqueeze(0), dim=1)
                similarity_matrix[i, j] = cos_sim.item()

        return similarity_matrix, objective_names

    def analyze_conflicts(self, similarity_matrix: torch.Tensor, objective_names: List[str]) -> Dict:
        """
        分析梯度冲突

        返回统计信息：
            - mean_similarity: 平均余弦相似度
            - conflict_pairs: 冲突对（余弦相似度 < 0）
            - conflict_ratio: 冲突比例
        """
        N = len(objective_names)

        # 提取上三角矩阵（不包括对角线）
        mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
        similarities = similarity_matrix[mask]

        # 统计
        mean_sim = similarities.mean().item()
        std_sim = similarities.std().item()
        min_sim = similarities.min().item()
        max_sim = similarities.max().item()

        # 找出冲突对（余弦相似度 < 0，即夹角 > 90°）
        conflict_pairs = []
        for i in range(N):
            for j in range(i+1, N):
                sim = similarity_matrix[i, j].item()
                if sim < 0:
                    conflict_pairs.append({
                        'objective_1': objective_names[i],
                        'objective_2': objective_names[j],
                        'cosine_similarity': sim,
                        'angle_degree': np.arccos(np.clip(sim, -1, 1)) * 180 / np.pi
                    })

        conflict_ratio = len(conflict_pairs) / (N * (N - 1) / 2) if N > 1 else 0.0

        return {
            'mean_similarity': mean_sim,
            'std_similarity': std_sim,
            'min_similarity': min_sim,
            'max_similarity': max_sim,
            'conflict_pairs': conflict_pairs,
            'conflict_ratio': conflict_ratio,
            'total_pairs': int(N * (N - 1) / 2)
        }

    def log_conflict_analysis(self, similarity_matrix: torch.Tensor, objective_names: List[str]):
        """打印梯度冲突分析报告"""
        stats = self.analyze_conflicts(similarity_matrix, objective_names)

        print("\n" + "="*80)
        print("📊 梯度冲突分析报告 (Gradient Conflict Analysis)")
        print("="*80)

        print(f"\n【统计概览】")
        print(f"  目标总数: {len(objective_names)}")
        print(f"  目标对总数: {stats['total_pairs']}")
        print(f"  平均余弦相似度: {stats['mean_similarity']:.4f} ± {stats['std_similarity']:.4f}")
        print(f"  相似度范围: [{stats['min_similarity']:.4f}, {stats['max_similarity']:.4f}]")

        print(f"\n【冲突检测】")
        print(f"  冲突对数量: {len(stats['conflict_pairs'])} / {stats['total_pairs']}")
        print(f"  冲突比例: {stats['conflict_ratio']*100:.2f}%")

        if stats['conflict_pairs']:
            print(f"\n【冲突详情】（余弦相似度 < 0，即夹角 > 90°）")
            for idx, pair in enumerate(stats['conflict_pairs'], 1):
                print(f"  {idx}. {pair['objective_1']} ↔ {pair['objective_2']}")
                print(f"     余弦相似度: {pair['cosine_similarity']:.4f}")
                print(f"     夹角: {pair['angle_degree']:.2f}°")
        else:
            print(f"  ✅ 未检测到梯度冲突（所有目标对的余弦相似度 ≥ 0）")

        print(f"\n【余弦相似度矩阵】")
        print("  " + " ".join([f"{name[:8]:>8}" for name in objective_names]))
        for i, name in enumerate(objective_names):
            row_str = f"{name[:8]:>8} "
            for j in range(len(objective_names)):
                sim = similarity_matrix[i, j].item()
                if i == j:
                    row_str += f"{'1.0000':>8} "
                elif sim < 0:
                    row_str += f"\033[91m{sim:>8.4f}\033[0m "  # 红色标记负值
                else:
                    row_str += f"{sim:>8.4f} "
            print("  " + row_str)

        print("="*80 + "\n")

        # 保存到历史记录
        self.conflict_history.append({
            'similarity_matrix': similarity_matrix.cpu().numpy(),
            'objective_names': objective_names,
            'stats': stats
        })

        return stats


class AblationTestRunner:
    """
    宏观层面：控制变量消融实验
    通过控制 loss_mask 仅激活单一目标，观察不同目标对攻击效果的影响
    """

    def __init__(self, args, device='cpu'):
        self.args = args
        self.device = device
        self.results = []

    def run_ablation_study(self, all_updates, malicious_attackers_this_round,
                          g_ce, g_cw, historical_pop=None):
        """
        运行完整的消融实验

        实验组：
        - A: 仅CE攻击 (loss_mask='11000')
        - B: 仅CW攻击 (loss_mask='01000')
        - C: 仅幅度约束 (loss_mask='00100')
        - D: 仅群体约束 (loss_mask='00010')
        - E: 仅符号约束 (loss_mask='00001')
        - F: CE+CW攻击 (loss_mask='11000')
        - G: 所有目标 (loss_mask='11111')
        """
        from .mos import mos_attack

        # 定义实验组配置
        experiment_configs = [
            {'name': 'A_CE_Only', 'loss_mask': '10000', 'desc': '仅CE攻击'},
            {'name': 'B_CW_Only', 'loss_mask': '01000', 'desc': '仅CW攻击'},
            {'name': 'C_Magnitude_Only', 'loss_mask': '00100', 'desc': '仅幅度约束'},
            {'name': 'D_Group_Only', 'loss_mask': '00010', 'desc': '仅群体约束'},
            {'name': 'E_Sign_Only', 'loss_mask': '00001', 'desc': '仅符号约束'},
            {'name': 'F_CE_CW', 'loss_mask': '11000', 'desc': 'CE+CW攻击'},
            {'name': 'G_All_Objectives', 'loss_mask': '11111', 'desc': '所有目标'},
        ]

        print("\n" + "="*80)
        print("🧪 梯度冲突消融实验 (Gradient Conflict Ablation Study)")
        print("="*80)

        results_table = []

        for config in experiment_configs:
            print(f"\n{'─'*80}")
            print(f"实验组: {config['name']} - {config['desc']}")
            print(f"Loss Mask: {config['loss_mask']}")
            print(f"{'─'*80}")

            # 深拷贝数据以避免影响原始数据
            all_updates_copy = copy.deepcopy(all_updates)

            # 临时修改 args 的 loss_mask
            original_mask = getattr(self.args, 'loss_mask', '11111')
            self.args.loss_mask = config['loss_mask']

            # 运行攻击
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            with torch.no_grad():
                # 注意：这里不使用 no_grad，因为 mos_attack 内部需要计算梯度
                pass

            try:
                modified_updates, hist_pop = mos_attack(
                    all_updates=all_updates_copy,
                    args=self.args,
                    malicious_attackers_this_round=malicious_attackers_this_round,
                    g_ce=g_ce,
                    g_cw=g_cw,
                    historical_pop=historical_pop
                )

                # 计算指标
                metrics = self._compute_metrics(
                    modified_updates,
                    all_updates,
                    malicious_attackers_this_round
                )

                # 记录结果
                result = {
                    'experiment': config['name'],
                    'description': config['desc'],
                    'loss_mask': config['loss_mask'],
                    **metrics
                }
                results_table.append(result)

                # 打印当前实验结果
                self._print_experiment_result(result)

            except Exception as e:
                print(f"❌ 实验失败: {str(e)}")
                result = {
                    'experiment': config['name'],
                    'description': config['desc'],
                    'loss_mask': config['loss_mask'],
                    'error': str(e)
                }
                results_table.append(result)

            # 恢复原始 mask
            self.args.loss_mask = original_mask

            # 清理内存
            del all_updates_copy
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 打印汇总表格
        self._print_summary_table(results_table)

        # 保存结果
        self.results = results_table

        return results_table

    def _compute_metrics(self, modified_updates, original_updates, K):
        """计算攻击效果指标"""
        # 提取恶意客户端的梯度
        malicious_grads = []
        benign_grads = []

        for i, update in enumerate(modified_updates):
            vec = torch.cat([torch.flatten(update[k]) for k in update.keys()])
            if i < K:
                malicious_grads.append(vec)
            else:
                benign_grads.append(vec)

        malicious_grads = torch.stack(malicious_grads).to(self.device)
        benign_grads = torch.stack(benign_grads).to(self.device)

        benign_mean = torch.mean(benign_grads, dim=0)

        # 指标1: 恶意梯度范数
        mal_norms = torch.norm(malicious_grads, dim=1)
        mean_mal_norm = mal_norms.mean().item()
        std_mal_norm = mal_norms.std().item()

        # 指标2: 与良性均值的距离
        distances = torch.norm(malicious_grads - benign_mean, dim=1)
        mean_distance = distances.mean().item()
        std_distance = distances.std().item()

        # 指标3: 恶意梯度间的方差（多样性）
        mal_mean = malicious_grads.mean(dim=0)
        mal_variance = torch.norm(malicious_grads - mal_mean, dim=1).mean().item()

        # 指标4: 恶意梯度与良性梯度的余弦相似度
        cos_sims = []
        for mal_grad in malicious_grads:
            cos_sim = F.cosine_similarity(mal_grad.unsqueeze(0), benign_mean.unsqueeze(0), dim=1)
            cos_sims.append(cos_sim.item())
        mean_cos_sim = np.mean(cos_sims)

        return {
            'mean_malicious_norm': mean_mal_norm,
            'std_malicious_norm': std_mal_norm,
            'mean_distance_to_benign': mean_distance,
            'std_distance_to_benign': std_distance,
            'malicious_variance': mal_variance,
            'mean_cosine_similarity': mean_cos_sim
        }

    def _print_experiment_result(self, result):
        """打印单个实验结果"""
        if 'error' in result:
            print(f"  ❌ 错误: {result['error']}")
            return

        print(f"\n  📈 指标:")
        print(f"    恶意梯度范数: {result['mean_malicious_norm']:.4f} ± {result['std_malicious_norm']:.4f}")
        print(f"    与良性均值距离: {result['mean_distance_to_benign']:.4f} ± {result['std_distance_to_benign']:.4f}")
        print(f"    恶意梯度多样性: {result['malicious_variance']:.4f}")
        print(f"    与良性余弦相似度: {result['mean_cosine_similarity']:.4f}")

    def _print_summary_table(self, results_table):
        """打印汇总表格（适合复制到论文）"""
        print("\n" + "="*80)
        print("📋 消融实验汇总表格 (Summary Table)")
        print("="*80)

        # 表头
        header = f"{'实验组':<20} {'Loss Mask':<12} {'梯度范数':<15} {'距离':<15} {'多样性':<12} {'余弦相似度':<12}"
        print(header)
        print("─" * 80)

        # 数据行
        for result in results_table:
            if 'error' in result:
                row = f"{result['experiment']:<20} {result['loss_mask']:<12} {'ERROR':<15} {'ERROR':<15} {'ERROR':<12} {'ERROR':<12}"
            else:
                row = f"{result['experiment']:<20} {result['loss_mask']:<12} " \
                      f"{result['mean_malicious_norm']:>7.2f}±{result['std_malicious_norm']:<5.2f} " \
                      f"{result['mean_distance_to_benign']:>7.2f}±{result['std_distance_to_benign']:<5.2f} " \
                      f"{result['malicious_variance']:>11.4f} " \
                      f"{result['mean_cosine_similarity']:>11.4f}"
            print(row)

        print("="*80 + "\n")

        # LaTeX 表格格式（方便直接复制到论文）
        print("📄 LaTeX 表格格式:")
        print("\\begin{tabular}{l|c|c|c|c|c}")
        print("\\hline")
        print("实验组 & Loss Mask & 梯度范数 & 距离 & 多样性 & 余弦相似度 \\\\")
        print("\\hline")
        for result in results_table:
            if 'error' not in result:
                print(f"{result['description']} & {result['loss_mask']} & "
                      f"{result['mean_malicious_norm']:.2f}$\\pm${result['std_malicious_norm']:.2f} & "
                      f"{result['mean_distance_to_benign']:.2f}$\\pm${result['std_distance_to_benign']:.2f} & "
                      f"{result['malicious_variance']:.4f} & "
                      f"{result['mean_cosine_similarity']:.4f} \\\\")
        print("\\hline")
        print("\\end{tabular}")
        print()


def run_gradient_conflict_analysis(all_updates, args, malicious_attackers_this_round,
                                   g_ce, g_cw, benign_grads, benign_mean, benign_std,
                                   krum_radius, lower_bound, upper_bound, survival_mask, layer_dims):
    """
    便捷函数：运行梯度冲突分析

    可以在 mos_attack 内部或外部调用
    """
    device = args.device if hasattr(args, 'device') else 'cpu'
    analyzer = GradientConflictAnalyzer(device=device)

    # 准备单位梯度向量
    g_ce_unit = g_ce / (torch.norm(g_ce) + 1e-9)
    g_cw_unit = g_cw / (torch.norm(g_cw) + 1e-9)
    lam = getattr(args, 'lam', 0.5)
    g_combined = lam * g_ce_unit + (1.0 - lam) * g_cw_unit
    g_combined_unit = g_combined / (torch.norm(g_combined) + 1e-9)

    # 创建一个临时种群用于梯度提取
    temp_pop = benign_mean.clone().detach().unsqueeze(0)

    # 提取各目标的梯度
    use_dnc = 'dnc' in getattr(args, 'defend_methods', [])
    gradients = analyzer.compute_objective_gradients(
        pop=temp_pop,
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
        use_dnc=use_dnc
    )

    # 计算余弦相似度矩阵
    similarity_matrix, objective_names = analyzer.compute_cosine_similarity_matrix(gradients)

    # 打印分析报告
    stats = analyzer.log_conflict_analysis(similarity_matrix, objective_names)

    return analyzer, stats
