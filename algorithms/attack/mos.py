
import torch
import torch.nn.functional as F
import copy
from .lie import vector_to_net_dict

def compute_surrogate_guidance(global_model, poison_images, target_labels, criterion_ce, args=None):
    """
    计算多代理损失的指导梯度（改进版：分层覆盖 + 战略稀疏化）
    """
    global_model.eval() # 确保在 eval 模式下计算梯度
    global_model.zero_grad()

    # 1. 前向传播
    outputs = global_model(poison_images)
    # ================= 探针 1：监控 Logit 量级 =================
    max_logit = outputs.max().item()
    min_logit = outputs.min().item()
    print(f"\n[MOS LOG] Surrogate Outputs - Max Logit: {max_logit:.2f}, Min Logit: {min_logit:.2f}")
    if abs(max_logit) > 100 or abs(min_logit) > 100:
        print(f"[MOS WARNING] ⚠️ Logits 正在爆炸，即将导致 NaN！")
    # =========================================================

    # -----------------------------------------
    # 代理目标 1：Cross Entropy (CE) - 分层战略覆盖
    # -----------------------------------------
    loss_ce = criterion_ce(outputs, target_labels)

    # 保留计算图以便后续还能算第二个梯度的反向传播
    loss_ce.backward(retain_graph=True)

    # ============ 新方案：分层梯度稀疏化（覆盖所有层，避免 DNC 采样盲区）============
    named = list(global_model.named_parameters())

    # 策略 1：对每一层都保留梯度，但按重要性稀疏化
    layer_groups = {}
    for n, p in named:
        if p.grad is None:
            continue
        # 按层类型分组
        if 'conv' in n.lower():
            group = 'conv'
        elif 'bn' in n.lower() or 'norm' in n.lower():
            group = 'bn'
        elif any(k in n.lower() for k in ('fc', 'classifier', 'head', 'linear', 'dense', 'out')):
            group = 'classifier'
        else:
            group = 'other'

        if group not in layer_groups:
            layer_groups[group] = []
        layer_groups[group].append((n, p))

    # 策略 2：分层稀疏化比例（从 args 读取，提供默认值）
    conv_sparsity = getattr(args, 'mos_conv_sparsity', 0.3) if args else 0.3
    classifier_sparsity = getattr(args, 'mos_classifier_sparsity', 1.0) if args else 1.0

    sparsity_ratio = {
        'classifier': classifier_sparsity,  # 分类器层：100% 保留（最重要）
        'conv': conv_sparsity,              # 卷积层：30% 保留（战略覆盖，避免 DNC 采样盲区）
        'bn': 0.1,                          # BN 层：10% 保留（低优先级）
        'other': 0.2                        # 其他层：20% 保留
    }

    # 策略 3：TopK 稀疏化（保留每层梯度绝对值最大的 k%）
    for group, params_list in layer_groups.items():
        ratio = sparsity_ratio.get(group, 0.2)
        for n, p in params_list:
            grad_flat = p.grad.flatten()
            k = max(int(grad_flat.numel() * ratio), 1)  # 至少保留 1 个
            _, topk_indices = torch.topk(grad_flat.abs(), k)

            # 创建稀疏掩码
            mask = torch.zeros_like(grad_flat)
            mask[topk_indices] = 1.0

            # 应用掩码（关键：保留原始梯度的幅度信息）
            p.grad.data = (grad_flat * mask).view_as(p.grad)

    # 提取稀疏化后的 g_ce（关键：必须与 all_updates 的 flatten 方式一致）
    def extract_gradient_vector(model, named_dict, device):
        """从模型中提取梯度向量，包含 parameters 和 buffers"""
        g_list = []
        for name, tensor in model.state_dict().items():
            if name in named_dict:
                param = named_dict[name]
                if param.grad is not None:
                    g_list.append(param.grad.clone().flatten())
                else:
                    g_list.append(torch.zeros(tensor.numel(), device=device))
            else:
                g_list.append(torch.zeros(tensor.numel(), device=device))
        return torch.cat(g_list) if g_list else torch.zeros(0, device=device)

    named_dict = dict(named)
    g_ce = extract_gradient_vector(global_model, named_dict, poison_images.device)
    global_model.zero_grad()
    
    # -----------------------------------------
    # 代理目标 2：Margin Loss (CW) - 深度破坏
    # -----------------------------------------
    # 参考论文中的 Marginal Loss 公式 [cite: 184]
    # 让目标类的 logit 远大于其他所有类的最大 logit
    correct_logits = torch.gather(outputs, 1, target_labels.unsqueeze(1)).squeeze(1)
    
    # 把正确类别的 logit 设为极小值，方便找出第二大的 logit
    outputs_clone = outputs.clone()
    outputs_clone.scatter_(1, target_labels.unsqueeze(1), -1e4)
    max_other_logits, _ = torch.max(outputs_clone, dim=1)
    
    # CW Loss: 希望 max_other - correct 越大越好 (即模型错得越离谱越好)
    loss_cw = torch.mean(torch.relu(max_other_logits - correct_logits + 20.0)) # 50.0 是 margin 余量
    
    loss_cw.backward()

    # 提取并展平 CW 梯度
    g_cw = extract_gradient_vector(global_model, named_dict, poison_images.device)
    global_model.zero_grad()

    return g_ce, g_cw

class LossNormalizer:
    """
    动态追踪每个 Loss 的最小值和最大值，用于实时归一化。
    参考 MOS-Attack 中的 normalization 思想。
    """
    def __init__(self, num_objectives, momentum=0.9):
        self.num_objectives = num_objectives
        self.momentum = momentum
        # 初始化为反向极值
        self.min_vals = None 
        self.max_vals = None
        
    def update_and_normalize(self, losses):
        """
        输入: losses shape (num_objectives, num_attackers) 或 (num_objectives,)
        输出: normalized_losses (0~1之间)
        """
        # 如果输入包含多个攻击者，先取平均或最值代表当前水平
        current_vals = torch.mean(losses, dim=1) if losses.dim() > 1 else losses

        if self.min_vals is None:
            self.min_vals = current_vals.clone().detach()
            self.max_vals = current_vals.clone().detach()
        else:
            # 使用动量更新历史极值，避免震荡
            self.min_vals = self.momentum * self.min_vals + (1 - self.momentum) * torch.min(self.min_vals, current_vals.detach())
            self.max_vals = self.momentum * self.max_vals + (1 - self.momentum) * torch.max(self.max_vals, current_vals.detach())

        # 防止除以 0
        range_vals = torch.clamp(self.max_vals - self.min_vals, min=1e-6)

        # 归一化到 [0, 1]
        mins = self.min_vals.unsqueeze(-1) if losses.dim() > 1 else self.min_vals
        ranges = range_vals.unsqueeze(-1) if losses.dim() > 1 else range_vals
        normalized = (losses - mins) / ranges
        return normalized

# 1. 签名里增加默认的 kwargs：g_ce 和 g_cw
def mos_attack(all_updates, args, malicious_attackers_this_round, g_ce=None, g_cw=None, historical_pop=None, lam=0.5):
    if malicious_attackers_this_round == 0: return all_updates

    device = args.device if hasattr(args, 'device') else 'cpu'
    K = malicious_attackers_this_round

    # --- 1. 数据准备与层维度记录 ---
    all_updates_flatten = []
    layer_dims = []  # 记录每一层在 flatten 向量中的位置
    idx_current = 0
    idx_w_start, idx_w_end = 0, 0
    idx_b_start, idx_b_end = 0, 0
    num_classes = 0

    # 获取输出层的 key
    keys = list(all_updates[0].keys())
    out_weight_key = keys[-2]
    out_bias_key = keys[-1]

    for k, v in all_updates[0].items():
        num_params = v.numel()
        layer_dims.append((k, idx_current, idx_current + num_params))

        # 记录输出层位置
        if k == out_weight_key:
            idx_w_start = idx_current
            idx_w_end = idx_current + num_params
            num_classes = v.shape[0]
        elif k == out_bias_key:
            idx_b_start = idx_current
            idx_b_end = idx_current + num_params

        idx_current += num_params

    for update in all_updates:
        vec = torch.cat([torch.flatten(update[k]) for k in update.keys()])
        all_updates_flatten.append(vec)
    all_stack = torch.stack(all_updates_flatten).to(device)
    
    
    
    # 提取良性梯度
    benign_grads = all_stack[malicious_attackers_this_round:].detach()
    # 防护：确保存在良性梯度（否则会在 torch.mean/torch.norm 等处产生 NaN）
    if benign_grads.size(0) == 0:
        # 没有良性客户端可参考，退化为不修改（或用小高斯噪声保持数值稳定）
        # 返回原始 updates 和一个保守的 malicious_set（避免后续空操作）
        noisy = torch.randn_like(all_stack[:malicious_attackers_this_round]) * 1e-6
        for i in range(malicious_attackers_this_round):
            all_updates[i] = vector_to_net_dict(all_stack[i] + noisy[i], copy.deepcopy(all_updates[i]))
        return all_updates, (malicious_set.detach() if 'malicious_set' in locals() else None)
    benign_mean = torch.mean(benign_grads, dim=0)
    
    # 获取约束边界 (和以前一样)
    benign_std = torch.std(benign_grads, dim=0) + 1e-9
    # ================= 探针 2A：监控良性梯度的反抗激烈程度 =================
    b_mean_norm = torch.norm(benign_mean).item()
    b_std_mean = torch.mean(benign_std).item()
    print(f"[MOS LOG] 良性梯度范数 (Benign Mean Norm): {b_mean_norm:.4f}")
    print(f"[MOS LOG] 良性梯度方差 (Benign STD Mean): {b_std_mean:.4f}")
    # ====================================================================
    dists_benign = torch.norm(benign_grads - benign_mean, dim=1)
    krum_radius = torch.max(dists_benign) * 1.1
    benign_min, _ = torch.min(benign_grads, dim=0)
    benign_max, _ = torch.max(benign_grads, dim=0)
    lower_bound = benign_min
    upper_bound = benign_max
    
    # ================= 修改 2：DNC-aware 掩码（仅在 DNC 开启时使用）=================
    def compute_dnc_sensitive_mask(benign_grads, benign_mean, device, num_samples=5):
        """计算 DNC 敏感的维度掩码，找出在多次随机采样中都被第一主成分覆盖的维度"""
        num_params = benign_mean.numel()
        sensitivity_scores = torch.zeros(num_params, device=device)

        for _ in range(num_samples):
            sampled_indices = torch.randperm(num_params, device=device)[:1000]
            benign_sub = benign_grads[:, sampled_indices]
            benign_sub_centered = benign_sub - benign_sub.mean(dim=0)

            try:
                U, S, V = torch.linalg.svd(benign_sub_centered, full_matrices=False)
                principal_component = V[0, :]
                weights = principal_component.abs()
                sensitivity_scores[sampled_indices] += weights
            except:
                continue

        sensitivity_scores = sensitivity_scores / (num_samples + 1e-9)

        # 创建掩码：保留敏感度最低的 50% 维度
        k_dim = int(0.5 * num_params)
        _, low_sensitivity_indices = torch.topk(sensitivity_scores, k_dim, largest=False)

        dnc_stealth_mask = torch.zeros_like(benign_mean)
        dnc_stealth_mask[low_sensitivity_indices] = 1.0

        return dnc_stealth_mask, sensitivity_scores

    # 检查是否启用 DNC 防御
    defend_methods = getattr(args, 'defend_methods', [])
    use_dnc = 'dnc' in defend_methods

    if use_dnc and getattr(args, 'use_dnc_aware_mask', True):
        # 使用 DNC-aware 掩码（针对性强）
        survival_mask, sensitivity_scores = compute_dnc_sensitive_mask(
            benign_grads, benign_mean, device, num_samples=5
        )
        print(f"[MOS LOG] DNC 模式启动 - 敏感维度占比: {(sensitivity_scores > sensitivity_scores.median()).float().mean():.2%}")
    else:
        # 回退到全 1 掩码（不做限制，适用于其他防御）
        survival_mask = torch.ones_like(benign_mean)
        if use_dnc:
            print(f"[MOS LOG] DNC 检测到但 DNC-aware 掩码已禁用")
    # ============================================================================
    
    # ================= 关键新增区：处理传进来的指导梯度 =================
    # 拦截 NaN 梯度，并限制最大推力
    has_valid_guidance = (g_ce is not None and g_cw is not None and
                          not torch.isnan(g_ce).any() and not torch.isinf(g_ce).any() and
                          not torch.isnan(g_cw).any() and not torch.isinf(g_cw).any())

    if has_valid_guidance:
        
        b_norm = torch.norm(benign_mean)
        #scale_factor = min(float(b_norm * 1), 5000.0) 
        # 将原本可能的 5000.0 限制死
        #scale_factor = min(float(b_norm * 1.0), float(b_norm * 2.0))
        min_attack_scale = 1.0  # 可以根据实际情况微调
        scale_factor = max(float(krum_radius * 0.3), min_attack_scale)
        
        g_ce_unit = g_ce.to(device) / (torch.norm(g_ce.to(device)) + 1e-9)
        g_cw_unit = g_cw.to(device) / (torch.norm(g_cw.to(device)) + 1e-9)
        
        
        # 合并目标向量（由参数 lam 控制 CE 与 CW 的配比），并归一化为 unit 向量
        lam = float(lam)
        g_combined = lam * g_ce_unit + (1.0 - lam) * g_cw_unit
        g_combined_unit = g_combined / (torch.norm(g_combined) + 1e-9)

        target_ce = benign_mean + scale_factor * g_ce_unit
        target_cw = benign_mean + scale_factor * g_cw_unit
    else:
        # 安全回退策略：检测异常并输出诊断信息
        warnings = []
        if g_ce is None: warnings.append("g_ce为空")
        if g_cw is None: warnings.append("g_cw为空")
        if g_ce is not None and torch.isnan(g_ce).any(): warnings.append("g_ce为nan")
        if g_cw is not None and torch.isnan(g_cw).any(): warnings.append("g_cw为nan")
        if g_ce is not None and torch.isinf(g_ce).any(): warnings.append("g_ce为inf")
        if g_cw is not None and torch.isinf(g_cw).any(): warnings.append("g_cw为inf")

        print(f"[MOS WARNING] ⚠️ 检测到代理梯度异常，启动安全回退策略！({', '.join(warnings)})")

        target_ce = benign_mean
        target_cw = benign_mean

        # 【填补黑洞】：给一个长度为安全半径 10%、方向与良性均值相同的推力
        safe_spark = krum_radius * 0.1
        benign_dir = benign_mean / (torch.norm(benign_mean) + 1e-9)

        g_ce_unit = benign_dir * safe_spark
        g_cw_unit = benign_dir * safe_spark
        g_combined_unit = benign_dir * safe_spark
        
    target_ce = target_ce.detach()
    target_cw = target_cw.detach()

    # ================= 修复1：动态适配与”相对扰动”平移 =================
    if historical_pop is not None:
        historical_pop = historical_pop.to(device)
        hist_K = historical_pop.shape[0]
        
        if hist_K == K:
            base_perturbation = historical_pop
        elif hist_K > K:
            base_perturbation = historical_pop[:K]
        else:
            indices = torch.randint(0, hist_K, (K - hist_K,), device=device)
            extra_pop = historical_pop[indices]
            base_perturbation = torch.cat([historical_pop, extra_pop], dim=0)
            
        # 紧箍咒：计算出历史记忆当前的长度
        hist_norms = torch.norm(base_perturbation, dim=1, keepdim=True)
        # 允许历史记忆存在的最大长度（例如：当前轮次 krum_radius 的 1.5 倍）
        max_allowed_hist = krum_radius * 3 
        # 如果超标了，就等比例压缩它；如果没超标，保持原样
        shrink_factor = torch.clamp(max_allowed_hist / (hist_norms + 1e-9), max=1.0)
        base_perturbation = base_perturbation * shrink_factor

        # 衰减系数依然保留，进一步维稳
        decay_factor = 0.95 
        malicious_set = benign_mean + (base_perturbation * decay_factor) + \
                        torch.randn(K, benign_mean.shape[0]).to(device) * (0.01 * benign_std)
    else:
        # 第一轮攻击时，没有历史经验，从头开始探索
        # 先生成纯随机的基底
        malicious_set = benign_mean.clone().detach().repeat(K, 1) + \
                        torch.randn(K, benign_mean.shape[0]).to(device) * (0.01 * benign_std)
        
    # 【关键修改】：将前几个恶意个体直接替换为沿着指导方向走到极致的”精英种子”
    if g_ce is not None and g_cw is not None:
        if K >= 1:
            malicious_set[0] = target_ce  # 极致的 CE 破坏者
        if K >= 2:
            malicious_set[1] = target_cw  # 极致的 CW 破坏者
        if K >= 3:
            # 沿着混合方向的破坏者
            scale_factor = krum_radius * 0.95
            malicious_set[2] = benign_mean + scale_factor * g_combined_unit

    # 1. 解析传入的 loss_mask 参数 (默认全开，即4位掩码 '1111')
    # 支持传入短于4位的字符串（右侧补'0'），或长于4位的字符串（截取前4位）
    loss_mask = getattr(args, 'loss_mask', '11111')
    loss_mask = loss_mask.ljust(4, '0')[:5]

    # 2. 统计开启的 loss 数量，作为 num_objectives 传给 normalizer
    num_active_losses = loss_mask.count('1')
    if num_active_losses == 0:  # 兜底，至少保留第一个目标
        loss_mask = '11000'
        num_active_losses = 2
        
    normalizer = LossNormalizer(num_objectives=num_active_losses, momentum=0.8)

    # ------ NSGA-III 帮助器（轻量实现，近似 NSGA-III 的参考方向配额选择） ------
    def nondominated_sort(objs):
        # objs: (M, N) torch tensor, M objectives, N individuals
        M, N = objs.shape
        S = [set() for _ in range(N)]
        n = torch.zeros(N, dtype=torch.int32)
        rank = torch.full((N,), -1, dtype=torch.int32)
        fronts = []
        for p in range(N):
            for q in range(N):
                if p == q: continue
                # p dominates q?
                less_eq = torch.all(objs[:, p] <= objs[:, q])
                strictly_less = torch.any(objs[:, p] < objs[:, q])
                if less_eq and strictly_less:
                    S[p].add(q)
                elif torch.all(objs[:, q] <= objs[:, p]) and torch.any(objs[:, q] < objs[:, p]):
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
        current_front = [i for i in range(N) if rank[i] == 0]
        i = 0
        while current_front:
            fronts.append(current_front)
            next_front = []
            for p in current_front:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            i += 1
            current_front = next_front
        return fronts

    def generate_reference_directions(M, H):
        # 生成 H 个参考方向（近似均匀），使用简单的随机正向单位向量并归一化
        refs = torch.rand(M, H, device=device)
        refs = refs / (torch.norm(refs, dim=0, keepdim=True) + 1e-12)
        return refs

    def niche_selection(last_front, objs_norm, refs, slots):
        # objs_norm: (M, F) normalized objectives for last front
        # refs: (M, R)
        F = objs_norm.shape[1]
        R = refs.shape[1]
        # 计算点到参考方向的垂直距离
        # assign each individual to nearest ref by angle (use cosine similarity)
        proj = torch.matmul(refs.t(), objs_norm)  # (R, F)
        ref_norms = torch.norm(refs, dim=0).unsqueeze(1) + 1e-12
        ind_norms = torch.norm(objs_norm, dim=0).unsqueeze(0) + 1e-12
        cosine = proj / (ref_norms * ind_norms)
        # 用 1 - cosine 作为距离近似
        dists = 1.0 - cosine
        assign = torch.argmin(dists, dim=0)  # size F
        selected = []
        niche_count = {r: 0 for r in range(R)}
        # for each niche, pick individuals with smallest perpendicular distance
        remaining = list(range(F))
        while len(selected) < slots and remaining:
            # for each niche, try select one
            for r in range(R):
                if len(selected) >= slots:
                    break
                # candidates in niche r
                cand = [i for i in remaining if int(assign[i].item()) == r]
                if not cand: continue
                # choose candidate with minimal distance to reference
                best = min(cand, key=lambda i: float(dists[r, i].item()))
                selected.append(best)
                remaining.remove(best)
                niche_count[r] += 1
                if len(selected) >= slots:
                    break
            # if some niches empty but still slots, pick smallest global distance
            if len(selected) < slots and remaining:
                best = min(remaining, key=lambda i: float(torch.min(dists[:, i]).item()))
                selected.append(best)
                remaining.remove(best)
        return selected

    def nsga3_select(objs, population, pop_size):
        # objs: (M, N)
        M, N = objs.shape
        fronts = nondominated_sort(objs)
        chosen = []
        for front in fronts:
            if len(chosen) + len(front) <= pop_size:
                chosen.extend(front)
            else:
                # need to select partial from this front via niching with reference directions
                remaining_slots = pop_size - len(chosen)
                # normalize objectives to positive [0,1]
                mins = torch.min(objs, dim=1)[0].unsqueeze(1)
                maxs = torch.max(objs, dim=1)[0].unsqueeze(1)
                denom = (maxs - mins)
                denom[denom < 1e-9] = 1.0
                objs_norm = (objs[:, front] - mins) / denom
                refs = generate_reference_directions(M, max(pop_size, 20))
                sel_local = niche_selection(front, objs_norm, refs, remaining_slots)
                # map local indices back to original
                chosen.extend([front[i] for i in sel_local])
                break
        return chosen

    # 进化参数
    generations = getattr(args, 'nsga_generations', 100)
    mutation_sigma = min(float(0.02 * torch.norm(upper_bound - lower_bound)), 1.0)
    if mutation_sigma <= 1e-12:
        mutation_sigma = 1e-6

    # 引入 archive 作为跨代记忆
    archive = malicious_set.clone().detach()
    archive_size = K

    # 最大允许偏移阈值
    max_dev_threshold = getattr(args, 'max_dev_threshold', 10000.0)
    large_norm_penalty_coeff = getattr(args, 'large_norm_penalty_coeff', 1e3)

    # ================= 修改 3：添加 PCA-aware 约束（DNC 专用核心武器）=================
    def compute_pca_constraint(pop, benign_grads, benign_mean, device):
        """计算恶意梯度在良性梯度主成分上的投影约束"""
        P = pop.shape[0]
        benign_centered = benign_grads - benign_mean

        try:
            U, S, V = torch.linalg.svd(benign_centered, full_matrices=False)
            num_principal_components = min(5, V.shape[0])
            principal_dirs = V[:num_principal_components, :]

            mal_deviation = pop - benign_mean
            projections = torch.matmul(mal_deviation, principal_dirs.T)

            weights = torch.tensor([1.0, 0.5, 0.3, 0.2, 0.1], device=device)[:num_principal_components]
            l_pca_projection = torch.sum(projections.abs() * weights, dim=1)

            benign_projections = torch.matmul(benign_centered, principal_dirs.T)
            benign_proj_std = benign_projections.std(dim=0) + 1e-9
            l_pca_normalized = l_pca_projection / (benign_proj_std * weights).sum()
        except Exception as e:
            print(f"[MOS WARNING] PCA constraint SVD failed: {e}, using fallback")
            l_pca_normalized = torch.zeros(P, device=device)

        return l_pca_normalized

    def compute_subspace_constraint(pop, benign_grads, benign_mean, device, num_samples=3):
        """计算子空间鲁棒性约束，模拟 DNC 的随机采样"""
        P = pop.shape[0]
        l_subspace_max = torch.zeros(P, device=device)

        for _ in range(num_samples):
            sampled_dims = torch.randperm(benign_mean.numel(), device=device)[:1000]
            benign_sub = benign_grads[:, sampled_dims]
            benign_sub_mean = benign_sub.mean(dim=0)
            benign_sub_centered = benign_sub - benign_sub_mean

            try:
                _, _, V_sub = torch.linalg.svd(benign_sub_centered, full_matrices=False)
                v_sub = V_sub[0, :]

                mal_sub = pop[:, sampled_dims]
                mal_sub_deviation = mal_sub - benign_sub_mean
                sub_scores = (mal_sub_deviation @ v_sub).abs()
                l_subspace_max = torch.max(l_subspace_max, sub_scores)
            except:
                continue

        try:
            benign_sub_scores = torch.abs((benign_sub_centered @ v_sub))
            benign_sub_std = benign_sub_scores.std() + 1e-9
            l_subspace_normalized = l_subspace_max / benign_sub_std
        except:
            l_subspace_normalized = l_subspace_max

        return l_subspace_normalized

    def compute_losses(pop):
        # pop: (P, D)
        P = pop.shape[0]
        # direction losses
        current_deviation = pop - benign_mean
        masked_deviation = current_deviation * survival_mask.unsqueeze(0)
        # 使用合并后的单一 guidance loss（由 lam 控制 CE/CW 比例）
        l_combined_p = -torch.sum(masked_deviation * g_combined_unit.unsqueeze(0), dim=1)
        target_ce = benign_mean + g_ce_unit

        target_cw = benign_mean + g_cw_unit

        l_ce = torch.norm(pop - target_ce.unsqueeze(0), dim=1)
        l_cw = torch.norm(pop - target_cw.unsqueeze(0), dim=1)

        current_dist_p = torch.norm(pop - benign_mean, dim=1)
        excess_dist = torch.relu(current_dist_p - krum_radius)
        l_krum_p = excess_dist ** 2
        # 原本的越界计算
        excess_lower_p = torch.relu(lower_bound - pop)
        excess_upper_p = torch.relu(pop - upper_bound)

        l_box_p = torch.norm(excess_lower_p + excess_upper_p, dim=1)

        malicious_mean_p = torch.mean(pop, dim=0)
        l_group_p = torch.norm(pop - malicious_mean_p, dim=1)

        # 逐层符号和范数损失
        l_sign_layer_max_p = torch.zeros(P, device=device)
        l_norm_layer_max_p = torch.zeros(P, device=device)

        for _, start_idx, end_idx in layer_dims:
            mal_layer = pop[:, start_idx:end_idx]
            ben_layer_all = benign_grads[:, start_idx:end_idx]
            ben_layer_norms = torch.norm(ben_layer_all, dim=1)
            target_norm = torch.median(ben_layer_norms).detach()
            mal_layer_norm = torch.norm(mal_layer, dim=1)
            layer_norm_loss = torch.abs(mal_layer_norm - target_norm)
            ben_layer_mean = benign_mean[start_idx:end_idx]
            sign_violation = -mal_layer * torch.sign(ben_layer_mean).unsqueeze(0)
            layer_sign_loss = torch.norm(torch.relu(sign_violation), dim=1)
            l_sign_layer_max_p = torch.max(l_sign_layer_max_p, layer_sign_loss)
            l_norm_layer_max_p = torch.max(l_norm_layer_max_p, layer_norm_loss)

        deviation_p = torch.abs(pop - benign_mean)
        excess_lasa_p = torch.relu(deviation_p - 0.1 * benign_std)
        l_lasa_norm_p = torch.norm(excess_lasa_p, dim=1)

        # 合并幅度相关的同质化损失为单一的 l_magnitude_p
        l_magnitude_p = l_krum_p + l_norm_layer_max_p + 0.5 * l_lasa_norm_p

        # ================= 新增：PCA 和子空间约束 =================
        if use_dnc:
            l_pca_p = compute_pca_constraint(pop, benign_grads, benign_mean, device)
            l_subspace_p = compute_subspace_constraint(pop, benign_grads, benign_mean, device, num_samples=3) \
                if getattr(args, 'enable_subspace_constraint', True) else torch.zeros(P, device=device)
        else:
            l_pca_p = torch.zeros(P, device=device)
            l_subspace_p = torch.zeros(P, device=device)

        # 扩展后的目标列表
        all_possible_losses_p = [
            l_ce,               # Index 0: CE 攻击性
            l_cw,               # Index 1: CW 攻击性
            l_magnitude_p,      # Index 2: 合并后的幅度约束
            l_group_p,          # Index 3: 拓扑/群体约束
            l_sign_layer_max_p, # Index 4: 符号一致性约束
            l_pca_p,            # Index 5: PCA 主成分约束
            l_subspace_p        # Index 6: 子空间鲁棒性约束
        ]

        active_losses_p = [all_possible_losses_p[i] for i, bit in enumerate(loss_mask) if bit == '1']

        # 越界惩罚和大范数惩罚
        is_out_of_bounds = (torch.sum(excess_lower_p + excess_upper_p, dim=1) > 0.01).float()
        penalty = is_out_of_bounds * 1e5

        current_norms = torch.norm(pop - benign_mean, dim=1)
        large_norm_excess = torch.relu(current_norms - max_dev_threshold)
        large_norm_penalty = large_norm_excess * large_norm_penalty_coeff

        combined_penalty = penalty + large_norm_penalty
        raw_losses_p = torch.stack(active_losses_p) + combined_penalty.unsqueeze(0)

        return raw_losses_p

    # 方向性变异强度已移除：不再优先沿 g_ce/g_cw 方向进化
    # 保留占位符以兼容 args，但默认不使用方向性偏移
    dir_scale = getattr(args, 'dir_scale', 0.0)

    for it in range(generations):
        # 评估当前种群
        raw_current = compute_losses(malicious_set)
        norm_current = normalizer.update_and_normalize(raw_current)
        objs_current = norm_current.detach()  # (M, N)

        # 选择父代（通过 NSGA-III 从当前种群中选择）
        parent_idx = nsga3_select(objs_current, malicious_set, K)
        parents = malicious_set[parent_idx]

        # 生成子代：使用 SBX（Simulated Binary Crossover）+ 生存掩码位置上的小幅高斯突变
        num_children = K
        perm = torch.randperm(K)
        children = []
        crossover_prob = getattr(args, 'sbx_crossover_prob', 0.9)
        eta = float(getattr(args, 'sbx_eta', 15.0))
        for i in range(0, num_children, 2):
            p1 = parents[perm[i % K]]
            p2 = parents[perm[(i+1) % K]]
            D = p1.numel()
            # SBX per-dimension
            u = torch.rand(D, device=device)
            mask = torch.rand(D, device=device) <= crossover_prob
            beta = torch.empty(D, device=device)
            le = u <= 0.5
            beta[le] = (2.0 * u[le]) ** (1.0 / (eta + 1.0))
            beta[~le] = (1.0 / (2.0 * (1.0 - u[~le]))) ** (1.0 / (eta + 1.0))
            child1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
            child2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
            # 如果该维度不做交叉，则直接拷贝父本对应值
            child1[~mask] = p1[~mask]
            child2[~mask] = p2[~mask]

            # 只在 survival_mask 指示的位置上加入小幅高斯突变
            noise1 = torch.randn_like(child1) * mutation_sigma
            noise2 = torch.randn_like(child2) * mutation_sigma
            dir_step = 0.05 * krum_radius 
            
            # 让子代不仅有随机突变，还顺着指导梯度往前走一步
            if g_combined_unit is not None:
                directional_push = dir_step * g_combined_unit
                # 为了维持多样性，可以对推力也加一点随机性（例如 0.5 到 1.5 倍的力度）
                push_scale1 = torch.empty(1).uniform_(0.5, 1.5).item()
                push_scale2 = torch.empty(1).uniform_(0.5, 1.5).item()
                
                noise1 = noise1 + directional_push * push_scale1
                noise2 = noise2 + directional_push * push_scale2
            noise1 = noise1 * survival_mask
            noise2 = noise2 * survival_mask
            child1 = child1 + noise1
            child2 = child2 + noise2

            children.append(child1)
            children.append(child2)
        offspring = torch.stack(children)[:K]

        # 合并：当前、子代、以及 archive（跨代记忆）
        combined = torch.cat([malicious_set, offspring, archive], dim=0)
        raw_combined = compute_losses(combined)
        norm_combined = normalizer.update_and_normalize(raw_combined)
        objs_combined = norm_combined.detach()

        # 从 combined 中选择下一代
        selected_idx = nsga3_select(objs_combined, combined, K)
        malicious_set = combined[selected_idx].clone().detach()
        
        # ================= 新策略：基于固定阈值的检测与强制收缩 =================
        # 使用 max_dev_threshold（上面定义）作为硬阈值：如果某个恶意向量超过阈值则等比例缩小
        deviation = malicious_set - benign_mean
        dev_norms = torch.norm(deviation, dim=1, keepdim=True)

        # 对超过阈值的个体进行等比例收缩到阈值以内，阈值由 args.max_dev_threshold 控制
        scales = torch.clamp(max_dev_threshold / (dev_norms + 1e-9), max=1.0)
        deviation = deviation * scales
        malicious_set = benign_mean + deviation

        # 备注：对于超过阈值的个体，我们在 compute_losses 中也施加了软惩罚项
        # 以避免种群长期朝着过大的方向发展，从而替代先前的巨常数惩罚副作用。
        # =========================================================================

        # 更新 archive：保留 combined 的非支配前沿，最多 archive_size
        fronts = nondominated_sort(objs_combined)
        new_archive_idx = []
        for f in fronts:
            for idx in f:
                new_archive_idx.append(idx)
                if len(new_archive_idx) >= archive_size:
                    break
            if len(new_archive_idx) >= archive_size:
                break
        archive = combined[new_archive_idx].clone().detach() if new_archive_idx else archive

    # 还原代码并输出监控信息
    optimized_grads = malicious_set.detach()
    for i in range(K):
        all_updates[i] = vector_to_net_dict(optimized_grads[i], copy.deepcopy(all_updates[i]))

    # ================= 探针 3：监控历史扰动是否无限滚雪球 =================
    historical_perturbation = (malicious_set - benign_mean).detach()
    pert_norm = torch.norm(historical_perturbation, dim=1).mean().item()
    print(f"[MOS LOG] 最终输出的恶意扰动平均范数 (Perturbation Norm): {pert_norm:.4f}")
    print("-" * 50)

    return all_updates, historical_perturbation


"""
print(f"DEBUG: Final Attack Norm: {atk_norm:.2f} (Target < {ben_norm + krum_radius:.2f})")
        print(f"DEBUG: Norm Losses -> Attack:{score_per_obj[0]:.3f} | Krum:{score_per_obj[1]:.3f} | Box:{score_per_obj[2]:.3f}")
        print(f"DEBUG: Attack Mean Norm: {torch.norm(torch.mean(optimized_grads, 0)):.4f} | Benign Mean Norm: {torch.norm(benign_mean):.4f}")
    """