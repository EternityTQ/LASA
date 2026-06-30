
import torch
import torch.nn.functional as F
import copy
from .lie import vector_to_net_dict

def compute_surrogate_guidance(global_model, poison_images, target_labels, criterion_ce):
    """
    计算多代理损失的指导梯度
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
    # 代理目标 1：Cross Entropy (CE) - 基础破坏
    # -----------------------------------------
    loss_ce = criterion_ce(outputs, target_labels)
    
    # 保留计算图以便后续还能算第二个梯度的反向传播
    loss_ce.backward(retain_graph=True)

    # 强制冻结前置卷积层的梯度：只保留分类器/最后两层的梯度用于 g_ce 提取
    named = list(global_model.named_parameters())
    names = [n for n, _ in named]
    allowed = set()
    # 模式匹配：包含常见分类器关键字的参数名
    for n, p in named:
        if any(k in n.lower() for k in ('classifier', 'fc', 'head', 'linear', 'dense', 'out')):
            allowed.add(n)
    # 额外确保保留最后两项参数（通常是输出层的 weight/bias）
    if len(names) >= 2:
        allowed.add(names[-1])
        allowed.add(names[-2])

    # 将不在 allowed 中的参数梯度置为 0
    for n, p in named:
        if p.grad is not None and n not in allowed:
            p.grad.zero_()

    # 提取并展平 CE 梯度（此时只包含分类器/最后两层的梯度）
    g_ce_list = [p.grad.clone().flatten() for n, p in named if p.grad is not None]
    g_ce = torch.cat(g_ce_list) if g_ce_list else torch.zeros(0, device=poison_images.device)

    # 清空梯度，准备算下一个
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
    g_cw_list = [param.grad.clone().flatten() for param in global_model.parameters() if param.grad is not None]
    g_cw = torch.cat(g_cw_list)
    
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
        if losses.dim() > 1:
            current_vals = torch.mean(losses, dim=1) # (num_obj,)
        else:
            current_vals = losses
            
        if self.min_vals is None:
            self.min_vals = current_vals.clone().detach()
            self.max_vals = current_vals.clone().detach()
        else:
            # 使用动量更新历史极值，避免震荡
            self.min_vals = self.momentum * self.min_vals + (1 - self.momentum) * torch.min(self.min_vals, current_vals.detach())
            self.max_vals = self.momentum * self.max_vals + (1 - self.momentum) * torch.max(self.max_vals, current_vals.detach())
            
        # 防止除以 0
        range_vals = self.max_vals - self.min_vals
        range_vals[range_vals < 1e-6] = 1.0
        
        # 归一化到 [0, 1]
        # 注意：这里需要 detach 极值，不让梯度传给 normalizer
        normalized = (losses - self.min_vals.unsqueeze(-1) if losses.dim() > 1 else self.min_vals) / (range_vals.unsqueeze(-1) if losses.dim() > 1 else range_vals)
        return normalized

# 1. 签名里增加默认的 kwargs：g_ce 和 g_cw
def mos_attack(all_updates, args, malicious_attackers_this_round, g_ce=None, g_cw=None, historical_pop=None, lam=0.5):
    if malicious_attackers_this_round == 0: return all_updates

    device = args.device if hasattr(args, 'device') else 'cpu'
    K = malicious_attackers_this_round
    
    # --- 1. 数据准备 ---
    all_updates_flatten = []
    
    # ================= 新增 1：追踪输出层参数在 Flatten 向量中的位置 =================
    idx_current = 0
    idx_w_start, idx_w_end = 0, 0
    idx_b_start, idx_b_end = 0, 0
    num_classes = 0
    
    # 获取字典里的 key 列表，通常 PyTorch 字典最后两个就是输出层的 weight 和 bias
    keys = list(all_updates[0].keys())
    out_weight_key = keys[-2]  # 例如 'fc.weight' 或 'classifier.weight'
    out_bias_key = keys[-1]    # 例如 'fc.bias' 或 'classifier.bias'
    
    for k, v in all_updates[0].items():
        num_params = v.numel()
        if k == out_weight_key:
            idx_w_start = idx_current
            idx_w_end = idx_current + num_params
            num_classes = v.shape[0]  # 输出层的类别数（比如 CIFAR10 就是 10）
        elif k == out_bias_key:
            idx_b_start = idx_current
            idx_b_end = idx_current + num_params
        idx_current += num_params
    # ==========================================================================
    
    
    # --- 1. 数据准备 (修改部分) ---
    all_updates_flatten = []
    layer_dims = [] # 新增：用来记录每一层在 flatten 向量中的位置
    idx_current = 0
    
    for k, v in all_updates[0].items():
        if 'num_batches_tracked' in k: 
            continue # 忽略 BN 层的统计量
        num_params = v.numel()
        layer_dims.append((k, idx_current, idx_current + num_params))
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
    
    # ================= 新增：模拟 LASA 的稀疏化掩码 (Sparsification) =================
# 假设防御方保留 50% 的参数 (你可以根据你的 args.sparsity 调整 0.5 这个值)
    keep_ratio = 0.5 
    k_dim = int(keep_ratio * benign_mean.numel())
    
    # 找出良性均值中绝对值最大的前 k_dim 个参数的索引
    _, topk_indices = torch.topk(benign_mean**2, k_dim)
    
    # 生成一个 mask，只在防御方大概率保留的地方设为 1
    survival_mask = torch.zeros_like(benign_mean)
    survival_mask[topk_indices] = 1.0
    survival_mask = survival_mask.to(device)
    # ============================================================================
    
    # ================= 关键新增区：处理传进来的指导梯度 =================
    # 增加严格的检查：确保 g_ce 和 g_cw 里面没有任何 nan 或 inf
# ================= 修复 1：拦截 NaN 梯度，并限制最大推力 =================
    if (g_ce is not None and g_cw is not None and 
        not torch.isnan(g_ce).any() and not torch.isnan(g_cw).any() and
        not torch.isinf(g_ce).any() and not torch.isinf(g_cw).any()):
        
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
        print("[MOS WARNING] ⚠️ 检测到代理梯度异常 (NaN/Inf)，启动安全回退策略！")
        if g_ce is None:
            print("[MOS WARNING] g_ce为空！")
        if g_cw is None:
            print("[MOS WARNING] g_cw为空！")
        if torch.isnan(g_ce).any():
            print("[MOS WARNING] g_ce为nan！")
        if torch.isnan(g_cw).any():
            print("[MOS WARNING] g_cw为nan！")
        if torch.isinf(g_ce).any():
            print("[MOS WARNING] g_ce为inf！")
        if torch.isinf(g_cw).any():
            print("[MOS WARNING] g_cw为inf！")
        target_ce = benign_mean
        target_cw = benign_mean
        
        # 【填补黑洞】：绝不能给 0 向量！给一个长度为安全半径 10%、方向与良性均值相同的推力
        # 保证 Perturbation Norm 必定 > 0，让下一轮的 historical_pop 拥有活下去的遗传火种
        safe_spark = krum_radius * 0.1
        benign_dir = benign_mean / (torch.norm(benign_mean) + 1e-9)
        
        g_ce_unit = benign_dir * safe_spark
        g_cw_unit = benign_dir * safe_spark
        g_combined_unit = benign_dir * safe_spark
        # ============================================================
        
    target_ce = target_ce.detach()
    target_cw = target_cw.detach()
    # ====================================================================

# ================= 修复1：动态适配与“相对扰动”平移 =================
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
        
        # 【关键修改】：将前几个恶意个体直接替换为沿着指导方向走到极致的“精英种子”
        # 确保 target_ce 和 target_cw 的距离在 Krum 允许的范围内 (前面建议修改的 krum_radius * 0.8)
    if g_ce is not None and g_cw is not None:
        if K >= 1:
            malicious_set[0] = target_ce  # 极致的 CE 破坏者
        if K >= 2:
            malicious_set[1] = target_cw  # 极致的 CW 破坏者
        if K >= 3:
                # 沿着混合方向的破坏者
            scale_factor = krum_radius * 0.95
            malicious_set[2] = benign_mean + scale_factor * g_combined_unit
    # =========================================================================

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
    offspring_size = K
    mutation_sigma = 0.02 * torch.norm(upper_bound - lower_bound)
    mutation_sigma = float(mutation_sigma)
    if mutation_sigma <= 1e-12:
        mutation_sigma = 1e-6
    mutation_sigma = min(mutation_sigma, 1.0)  # 防止过大

    # 引入 archive 作为跨代记忆，初始时为空
    archive = malicious_set.clone().detach()
    archive_size = K

    # 最大允许偏移阈值（固定值，后续可改为历史窗口动态估计）
    max_dev_threshold = getattr(args, 'max_dev_threshold', 10000.0)
    # 惩罚系数（用于对超过阈值的个体施加软惩罚，避免使用巨大的常数）
    large_norm_penalty_coeff = getattr(args, 'large_norm_penalty_coeff', 1e3)

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

            # 符号损失
# 定义温度系数 k (可以作为一个超参数调优，先从 10.0 开始)
        temperature_k = 10.0 

        l_sign_layer_max_p = torch.zeros(P, device=device)
        l_norm_layer_max_p = torch.zeros(P, device=device)
        
        for name, start_idx, end_idx in layer_dims:
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

        # 输出层 neuron magnitude（保留计算但不纳入最终目标列表）
        mal_out_w_p = pop[:, idx_w_start:idx_w_end].view(P, num_classes, -1)
        mal_out_b_p = pop[:, idx_b_start:idx_b_end].view(P, num_classes, 1)
        mal_out_params_p = torch.cat([mal_out_w_p, mal_out_b_p], dim=2)
        neuron_norms_p = torch.norm(mal_out_params_p, dim=2)
        with torch.no_grad():
            ben_out_w = benign_mean[idx_w_start:idx_w_end].view(num_classes, -1)
            ben_out_b = benign_mean[idx_b_start:idx_b_end].view(num_classes, 1)
            ben_out_params = torch.cat([ben_out_w, ben_out_b], dim=1)
            ben_neuron_norms = torch.norm(ben_out_params, dim=1)
        excess_magnitude_p = torch.relu(neuron_norms_p - ben_neuron_norms.unsqueeze(0))
        l_output_magnitude_p = torch.norm(excess_magnitude_p, dim=1)

        # ================================================================
        # 合并幅度相关的同质化损失为单一的 l_magnitude_p（用于缓解维数灾难）
        # 可调权重（便于后续调参）：
        krum_weight = 1.0
        layer_weight = 1.0
        lasa_weight = 0.5
        # l_krum_p, l_lasa_norm_p, l_norm_layer_max_p 都是形状为 (P,) 的张量
        l_magnitude_p = (krum_weight * l_krum_p) + (layer_weight * l_norm_layer_max_p) + (lasa_weight * l_lasa_norm_p)
        # 注意：不再将 l_output_magnitude_p 纳入最终目标列表（其权重被移除）
        # ================================================================

        # 精简后的目标列表：4 个核心冲突目标
        all_possible_losses_p = [
            #l_combined_p,   # Index 0: 攻击破坏性
            l_ce,
            l_cw,
            l_magnitude_p,  # Index 1: 合并后的幅度约束
            l_group_p,      # Index 2: 拓扑/群体约束
            l_sign_layer_max_p  # Index 3: 符号一致性约束
        ]

        active_losses_p = []
        for i, bit in enumerate(loss_mask):
            if bit == '1':
                active_losses_p.append(all_possible_losses_p[i])
        
        # 如果越界严重，保留原始的 box 惩罚（较小概率触发）
        is_out_of_bounds = (torch.sum(excess_lower_p + excess_upper_p, dim=1) > 0.01).float()
        penalty = is_out_of_bounds * 1e5

        # 对超出范数阈值的个体给予软惩罚，避免使用巨大常数造成副作用
        current_norms = torch.norm(pop - benign_mean, dim=1)
        large_norm_excess = torch.relu(current_norms - max_dev_threshold)
        large_norm_penalty = large_norm_excess * large_norm_penalty_coeff

        combined_penalty = penalty + large_norm_penalty

        # 将 penalty 加到所有 active_losses 上（每个个体都会累加惩罚）
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

    # 还原代码 (保持不变)
    optimized_grads = malicious_set.detach()
    for i in range(K):
        all_updates[i] = vector_to_net_dict(optimized_grads[i], copy.deepcopy(all_updates[i]))

    historical_perturbation = (malicious_set - benign_mean).detach()
    # ================= 探针 3：监控历史扰动是否无限滚雪球 =================
    historical_perturbation = (malicious_set - benign_mean).detach()
    pert_norm = torch.norm(historical_perturbation, dim=1).mean().item()
    print(f"[MOS LOG] 最终输出的恶意扰动平均范数 (Perturbation Norm): {pert_norm:.4f}")
    print("-" * 50)
    # ==================================================================
    
    return all_updates, historical_perturbation


"""
print(f"DEBUG: Final Attack Norm: {atk_norm:.2f} (Target < {ben_norm + krum_radius:.2f})")
        print(f"DEBUG: Norm Losses -> Attack:{score_per_obj[0]:.3f} | Krum:{score_per_obj[1]:.3f} | Box:{score_per_obj[2]:.3f}")
        print(f"DEBUG: Attack Mean Norm: {torch.norm(torch.mean(optimized_grads, 0)):.4f} | Benign Mean Norm: {torch.norm(benign_mean):.4f}")
    """