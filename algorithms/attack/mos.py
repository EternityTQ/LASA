
import torch
import torch.nn.functional as F
import copy
import random
from .lie import vector_to_net_dict

# 梯度冲突分析模块（可选导入）
try:
    from .gradient_conflict_analyzer import run_gradient_conflict_analysis
    CONFLICT_ANALYZER_AVAILABLE = True
except ImportError:
    CONFLICT_ANALYZER_AVAILABLE = False


# ============================================================================
# 显存监控函数
# ============================================================================
def log_cuda_memory(tag, enabled=False):
    """GPU 显存监控（仅在 enabled=True 时打印）"""
    if not enabled or not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

    print(
        f"[CUDA] {tag}: "
        f"allocated={allocated:.2f} GB, "
        f"reserved={reserved:.2f} GB, "
        f"peak_allocated={peak_allocated:.2f} GB, "
        f"peak_reserved={peak_reserved:.2f} GB"
    )


# ============================================================================
# 攻击预算投影函数
# ============================================================================
def project_to_attack_budget(pop, benign_mean, max_dev_threshold):
    """
    将种群投影到攻击预算范围内
    返回：(投影后的种群, 裁剪掩码, 投影前的范数)
    """
    centered = pop - benign_mean
    norms = torch.norm(centered, dim=1, keepdim=True)

    clipped_mask = norms.squeeze(1) > max_dev_threshold

    scales = torch.clamp(
        max_dev_threshold / (norms + 1e-12),
        max=1.0,
    )

    projected = benign_mean + centered * scales

    return projected, clipped_mask, norms.squeeze(1)

def compute_surrogate_guidance(global_model, poison_images, target_labels, criterion_ce, args=None):
    """
    计算多代理损失的指导梯度（改进版：分层覆盖 + 战略稀疏化）
    修复：分离 CE 和 CW 的 forward，减少显存峰值
    """
    global_model.eval()
    device = poison_images.device

    # ================= 第一次前向传播：CE Loss =================
    global_model.zero_grad(set_to_none=True)

    outputs_ce = global_model(poison_images)

    # 数值稳定性监控
    max_logit = outputs_ce.max().item()
    min_logit = outputs_ce.min().item()
    abs_max_logit = max(abs(max_logit), abs(min_logit))

    logit_warning_threshold = getattr(args, 'logit_warning_threshold', 100.0)

    print(f"\n[MOS LOG] Surrogate CE Forward - Max Logit: {max_logit:.2f}, Min Logit: {min_logit:.2f}")

    if abs_max_logit > logit_warning_threshold:
        print(f"[MOS WARNING] ⚠️ Logits 超过警告阈值 {logit_warning_threshold}，当前最大绝对值: {abs_max_logit:.2f}")
        print(f"[MOS WARNING] ⚠️ 模型可能正在数值爆炸，建议检查学习率和梯度裁剪设置")

    # 检查 NaN 或 Inf
    if not torch.isfinite(outputs_ce).all():
        print(f"[MOS ERROR] ⚠️⚠️⚠️ Logits 包含 NaN 或 Inf，返回安全回退梯度")
        # 修复 9/10：使用 state_dict 计算正确长度
        total_numel = sum(
            tensor.numel()
            for tensor in global_model.state_dict().values()
        )
        safe_fallback = torch.zeros(total_numel, device=device)
        return safe_fallback.clone(), safe_fallback.clone()

    # -----------------------------------------
    # 代理目标 1：Cross Entropy (CE)
    # -----------------------------------------
    loss_ce = criterion_ce(outputs_ce, target_labels)
    loss_ce.backward()

    # ============ 分层梯度稀疏化（覆盖所有层，避免 DNC 采样盲区）============
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
    g_ce = extract_gradient_vector(global_model, named_dict, device)

    del outputs_ce
    del loss_ce

    # -----------------------------------------
    # 代理目标 2：Margin Loss (CW) - 深度破坏
    # -----------------------------------------
    global_model.zero_grad(set_to_none=True)

    outputs_cw = global_model(poison_images)

    # 参考论文中的 Marginal Loss 公式 [cite: 184]
    # 让目标类的 logit 远大于其他所有类的最大 logit
    correct_logits = torch.gather(outputs_cw, 1, target_labels.unsqueeze(1)).squeeze(1)

    # 把正确类别的 logit 设为极小值，方便找出第二大的 logit
    outputs_clone = outputs_cw.clone()
    outputs_clone.scatter_(1, target_labels.unsqueeze(1), -1e4)
    max_other_logits, _ = torch.max(outputs_clone, dim=1)

    # CW Loss: 希望 max_other - correct 越大越好 (即模型错得越离谱越好)
    loss_cw = torch.mean(torch.relu(max_other_logits - correct_logits + 20.0))

    loss_cw.backward()

    # 提取并展平 CW 梯度
    g_cw = extract_gradient_vector(global_model, named_dict, device)

    del outputs_cw
    del loss_cw
    del outputs_clone

    global_model.zero_grad(set_to_none=True)

    return g_ce, g_cw

def compute_raw_constraint_losses(pop, benign_refs, centered=None):
    """
    计算原始约束损失

    Args:
        pop: 种群张量 (P, D)
        benign_refs: 良性参考字典
        centered: 预计算的 (pop - benign_mean)，避免重复计算
    """
    P = pop.shape[0]
    device = benign_refs['device']

    benign_mean = benign_refs['mean']
    benign_std = benign_refs['std']
    layer_dims = benign_refs['layer_dims']
    use_dnc = benign_refs['use_dnc']

    losses = {}

    # 修复 15/10：使用传入的 centered 或计算一次
    if centered is None:
        centered = pop - benign_mean

    losses['radial'] = torch.norm(centered, dim=1)

    # 保留 'krum' 别名以兼容旧代码
    losses['krum'] = losses['radial']

    # Sign 约束：使用 centered 避免重复计算
    sign_layer_reduce = getattr(benign_refs.get('args'), 'sign_layer_reduce', 'quantile')
    sign_layer_quantile = getattr(benign_refs.get('args'), 'sign_layer_quantile', 0.9)
    sign_layer_losses = []

    for _, start_idx, end_idx in layer_dims:
        layer_centered = centered[:, start_idx:end_idx]
        layer_mean = benign_mean[start_idx:end_idx]

        # 符号违反：如果与均值符号相反，计算违反幅度
        # 注意：这里用 layer_centered + layer_mean 得到原始值
        layer_slice = layer_centered + layer_mean
        sign_violation = -layer_slice * torch.sign(layer_mean).unsqueeze(0)

        # 归一化：避免大层主导
        violation_norm = torch.norm(torch.relu(sign_violation), dim=1)
        reference_norm = torch.norm(layer_mean) + 1e-12
        layer_loss = violation_norm / reference_norm

        sign_layer_losses.append(layer_loss)

    sign_layer_losses = torch.stack(sign_layer_losses, dim=0)  # (L, P)

    # 层间聚合
    if sign_layer_reduce == 'max':
        losses['sign'] = sign_layer_losses.max(dim=0)[0]
    elif sign_layer_reduce == 'mean':
        losses['sign'] = sign_layer_losses.mean(dim=0)
    elif sign_layer_reduce == 'quantile':
        losses['sign'] = torch.quantile(sign_layer_losses, q=sign_layer_quantile, dim=0)
    else:
        losses['sign'] = sign_layer_losses.max(dim=0)[0]  # 回退

    if use_dnc and benign_refs.get('pca_principal_dirs') is not None:
        pca_principal_dirs = benign_refs['pca_principal_dirs']
        pca_benign_proj_std = benign_refs['pca_benign_proj_std']

        # 修复 15/10：使用传入的 centered
        projections = torch.matmul(centered, pca_principal_dirs.T)  # (P, K)

        # 标准化投影
        normalized_proj = projections.abs() / (pca_benign_proj_std.unsqueeze(0) + 1e-12)

        # 加权求和（高阶成分权重递减）
        num_components = pca_principal_dirs.shape[0]
        weights = torch.tensor([1.0, 0.5, 0.3, 0.2, 0.1], device=device)[:num_components]

        losses['pca'] = torch.sum(normalized_proj * weights.unsqueeze(0), dim=1)
    else:
        losses['pca'] = torch.zeros(P, device=device)

    # ========================================================================
    # 4. Subspace 约束：标准化的随机子空间得分
    # ========================================================================
    subspace_reduce = getattr(benign_refs.get('args'), 'subspace_reduce', 'max')

    if use_dnc and benign_refs.get('subspace_samples'):
        subspace_scores = []

        for sample in benign_refs['subspace_samples']:
            # 使用 centered 的子集
            sub_centered = centered[:, sample['sampled_dims']]

            # 投影到子空间主成分
            raw_scores = torch.abs(sub_centered @ sample['principal_component'])

            # 标准化
            normalized_scores = raw_scores / (sample['benign_std'] + 1e-12)
            subspace_scores.append(normalized_scores)

        subspace_scores = torch.stack(subspace_scores, dim=0)  # (S, P)

        # 子空间间聚合
        if subspace_reduce == 'max':
            losses['subspace'] = subspace_scores.max(dim=0)[0]
        elif subspace_reduce == 'mean':
            losses['subspace'] = subspace_scores.mean(dim=0)
        else:
            losses['subspace'] = subspace_scores.max(dim=0)[0]
    else:
        losses['subspace'] = torch.zeros(P, device=device)

    # ========================================================================
    # 5. Cohesion 约束（原Group）：恶意种群内部凝聚度
    # 修复 3/10：默认禁用 Cohesion 约束（仅在明确启用时计算）
    # 注意：Cohesion 是种群相关的，不适合作为普通约束
    # ========================================================================
    enable_cohesion = getattr(benign_refs.get('args'), 'enable_cohesion_constraint', False)

    if enable_cohesion:
        # 警告：Cohesion is population-dependent and is an experimental metric.
        malicious_mean = torch.mean(pop, dim=0)
        losses['cohesion'] = torch.norm(pop - malicious_mean, dim=1)
    else:
        # 不要返回全零（会给虚假高分），直接不计算
        losses['cohesion'] = torch.zeros(P, device=device)

    # 保留 'group' 别名以兼容旧代码
    losses['group'] = losses['cohesion']

    return losses


# ============================================================================
# 修复 6/12：重构约束得分映射函数，避免饱和
# ============================================================================

def compute_constraint_score(loss_value, threshold, mode='smooth', temperature=0.5):
    # 计算无量纲比值
    ratio = loss_value / (threshold + 1e-12)

    if mode == 'smooth':
        # 平滑 sigmoid 映射
        # ratio < 1: 高分; ratio = 1: 0.5; ratio > 1: 低分
        return torch.sigmoid((1.0 - ratio) / temperature)

    elif mode == 'relu':
        # ReLU 反向映射
        return torch.clamp(1.0 - ratio, min=0.0, max=1.0)

    elif mode == 'linear':
        # 分段线性映射
        return torch.where(
            ratio < 1.0,
            torch.ones_like(ratio),
            torch.where(
                ratio < 2.0,
                2.0 - ratio,
                torch.zeros_like(ratio)
            )
        )
    else:
        # 回退到 smooth
        return torch.sigmoid((1.0 - ratio) / temperature)


def compute_constraint_violations(losses, thresholds):
    violations = {}
    ratios = {}

    for name, loss in losses.items():
        threshold = thresholds.get(name, 1.0)
        ratio = loss / (threshold + 1e-12)
        ratios[name] = ratio
        violations[name] = torch.relu(ratio - 1.0)

    return violations, ratios


# ============================================================================
# 原有的旧版 compute_constraint_score（保留以兼容，但已废弃）
# ============================================================================


@torch.no_grad()
def mos_attack(all_updates, args, malicious_attackers_this_round, g_ce=None, g_cw=None, historical_pop=None, lam=0.5):
    """
    MOS 攻击主函数（修复版：内存优化 + 约束修复）
    """
    if malicious_attackers_this_round == 0:
        return all_updates

    device = args.device if hasattr(args, 'device') else 'cpu'
    K = malicious_attackers_this_round

    # 显存调试开关
    mos_memory_debug = getattr(args, 'mos_memory_debug', False)

    if mos_memory_debug and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        log_cuda_memory("attack entry", enabled=True)

    # ================= 核心修改：固定进化种群大小 =================
    EVOLUTION_POP_SIZE = getattr(args, 'evo_pop_size', 10)
    print(f"[MOS LOG] 🧬 进化模式：固定种群大小 = {EVOLUTION_POP_SIZE}，目标恶意客户端数 K = {K}")
    # ============================================================

    # --- 1. 数据准备与层维度记录 ---
    layer_dims = []
    idx_current = 0

    # 获取输出层的 key
    keys = list(all_updates[0].keys())
    out_weight_key = keys[-2]
    out_bias_key = keys[-1]

    for k, v in all_updates[0].items():
        num_params = v.numel()
        layer_dims.append((k, idx_current, idx_current + num_params))
        idx_current += num_params

    # 修复 12/10：只展平良性客户端更新
    benign_updates = all_updates[malicious_attackers_this_round:]
    total_params = idx_current

    if len(benign_updates) == 0:
        # 没有良性客户端，使用安全回退
        print(f"[MOS WARNING] 没有良性客户端，返回加噪原始更新")
        noisy = torch.randn(K, total_params, device=device) * 1e-6
        for i in range(K):
            original_vec = torch.cat([torch.flatten(all_updates[i][k]) for k in all_updates[i].keys()])
            all_updates[i] = vector_to_net_dict(
                original_vec.to(device) + noisy[i],
                copy.deepcopy(all_updates[i])
            )
        return all_updates

    # 预分配良性梯度矩阵
    benign_grads = torch.empty(
        (len(benign_updates), total_params),
        device=device,
        dtype=next(iter(benign_updates[0].values())).dtype,
    )

    # 逐个复制
    for row, update in enumerate(benign_updates):
        offset = 0
        for k in update.keys():
            tensor = update[k]
            flat = tensor.reshape(-1)
            end = offset + flat.numel()
            benign_grads[row, offset:end].copy_(flat)
            offset = end

    log_cuda_memory("after benign stack", mos_memory_debug)

    # 计算良性统计
    benign_mean = torch.mean(benign_grads, dim=0)

    # 修复 10/10：使用 correction=0 避免单个良性客户端时的 NaN
    benign_std = torch.std(benign_grads, dim=0, correction=0) + 1e-9

    b_mean_norm = torch.norm(benign_mean).item()
    b_std_mean = torch.mean(benign_std).item()
    b_std_max = torch.max(benign_std).item()

    print(f"[MOS LOG] 📊 良性梯度统计:")
    print(f"[MOS LOG]   良性均值范数 (Benign Mean Norm): {b_mean_norm:.4f}")
    print(f"[MOS LOG]   良性标准差 (Benign STD): mean={b_std_mean:.4f}, max={b_std_max:.4f}")

    # 检查数值异常
    if not torch.isfinite(benign_mean).all():
        print(f"[MOS ERROR] ⚠️⚠️⚠️ 良性均值包含 NaN 或 Inf，数据异常！")
    if not torch.isfinite(benign_std).all():
        print(f"[MOS ERROR] ⚠️⚠️⚠️ 良性标准差包含 NaN 或 Inf，数据异常！")

    dists_benign = torch.norm(benign_grads - benign_mean, dim=1)
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
            # 修复 14/10：使用 random.sample 避免大 randperm
            sample_size = min(1000, num_params)
            sampled_indices = torch.tensor(
                random.sample(range(num_params), sample_size),
                device=device,
                dtype=torch.long,
            )

            benign_sub = benign_grads[:, sampled_indices]
            benign_sub_centered = benign_sub - benign_sub.mean(dim=0)

            try:
                U, S, V = torch.linalg.svd(benign_sub_centered, full_matrices=False)
                principal_component = V[0, :]
                weights = principal_component.abs()
                sensitivity_scores[sampled_indices] += weights
            except:
                continue

        sensitivity_scores = sensitivity_scores / (num_samples + 1e-12)

        # 创建掩码：保留敏感度最低的 50% 维度
        k_dim = int(0.5 * num_params)
        _, low_sensitivity_indices = torch.topk(sensitivity_scores, k_dim, largest=False)

        dnc_stealth_mask = torch.zeros_like(benign_mean)
        dnc_stealth_mask[low_sensitivity_indices] = 1.0

        return dnc_stealth_mask, sensitivity_scores

    # 检查是否启用 DNC 防御
    defend_methods = getattr(args, 'defend_methods', [])
    use_dnc = 'dnc' in defend_methods

    if use_dnc and getattr(args, 'use_dnc_aware_mask', False):
        # 使用 DNC-aware 掩码（针对性强）
        survival_mask, sensitivity_scores = compute_dnc_sensitive_mask(
            benign_grads, benign_mean, device, num_samples=5
        )
        active_ratio = survival_mask.sum().item() / survival_mask.numel()
        print(f"[MOS LOG] ✓ DNC-aware mask enabled: True")
        print(f"[MOS LOG]   Survival mask active ratio: {active_ratio:.2%}")
        print(f"[MOS LOG]   Sensitive dimension ratio: {(sensitivity_scores > sensitivity_scores.median()).float().mean():.2%}")
    else:
        # 回退到全 1 掩码（不做限制，适用于其他防御）
        survival_mask = torch.ones_like(benign_mean)
        sensitivity_scores = None
        print(f"[MOS LOG] ✓ DNC-aware mask enabled: False")
        if use_dnc:
            print(f"[MOS LOG]   (DNC detected but mask disabled)")

    # 修复 3/10：计算攻击预算（使用稳健的分位数方法）
    radius_quantile = getattr(args, 'radius_quantile', 0.95)
    smoothed_radial_threshold = torch.quantile(dists_benign, q=radius_quantile)

    attack_budget_ratio = getattr(args, 'attack_budget_ratio', 1.0)
    max_dev_threshold = attack_budget_ratio * smoothed_radial_threshold

    print(f"[MOS LOG] 📐 Attack budget configuration:")
    print(f"[MOS LOG]   Base radial threshold (q={radius_quantile}): {smoothed_radial_threshold:.4f}")
    print(f"[MOS LOG]   Attack budget ratio: {attack_budget_ratio}")
    print(f"[MOS LOG]   Max deviation threshold: {max_dev_threshold:.4f}")

    # 修复 8/10：统一所有攻击尺度到新预算
    elite_ce_ratio = getattr(args, 'elite_ce_ratio', 0.8)
    elite_cw_ratio = getattr(args, 'elite_cw_ratio', 0.8)
    elite_combined_ratio = getattr(args, 'elite_combined_ratio', 0.95)
    history_max_ratio = getattr(args, 'history_max_ratio', 1.0)
    safe_fallback_ratio = getattr(args, 'safe_fallback_ratio', 0.1)

    has_valid_guidance = (g_ce is not None and g_cw is not None and
                          not torch.isnan(g_ce).any() and not torch.isinf(g_ce).any() and
                          not torch.isnan(g_cw).any() and not torch.isinf(g_cw).any())

    if has_valid_guidance:
        # 修复 8/10：使用新预算计算精英目标
        g_ce_unit = g_ce.to(device) / (torch.norm(g_ce.to(device)) + 1e-9)
        g_cw_unit = g_cw.to(device) / (torch.norm(g_cw.to(device)) + 1e-9)

        lam = float(lam)
        g_combined = lam * g_ce_unit + (1.0 - lam) * g_cw_unit
        g_combined_unit = g_combined / (torch.norm(g_combined) + 1e-9)

        target_ce = benign_mean + elite_ce_ratio * max_dev_threshold * g_ce_unit
        target_cw = benign_mean + elite_cw_ratio * max_dev_threshold * g_cw_unit
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

        # 修复 9/10：使用 state_dict 计算正确长度
        safe_spark = safe_fallback_ratio * max_dev_threshold
        benign_dir = benign_mean / (torch.norm(benign_mean) + 1e-9)

        g_ce_unit = benign_dir
        g_cw_unit = benign_dir
        g_combined_unit = benign_dir

        target_ce = benign_mean + safe_spark * benign_dir
        target_cw = benign_mean + safe_spark * benign_dir

    target_ce = target_ce.detach()
    target_cw = target_cw.detach()

    # ============================================================================
    # 预计算良性参考系统（循环外抽离，避免每代重复计算）
    # ============================================================================

    print(f"\n[MOS LOG] 🔧 开始预计算良性参考系统...")

    # 1. PCA主成分预计算（用于后续约束评分）
    pca_principal_dirs = None
    pca_benign_proj_std = None

    if use_dnc:
        try:
            benign_centered = benign_grads - benign_mean
            U, S, V = torch.linalg.svd(benign_centered, full_matrices=False)
            num_principal_components = min(5, V.shape[0])

            # 修复 13/10：clone 后释放完整 V
            pca_principal_dirs = V[:num_principal_components].clone().detach()

            # 计算良性梯度在主成分上的投影标准差（用于归一化）
            benign_projections = torch.matmul(benign_centered, pca_principal_dirs.T)
            pca_benign_proj_std = benign_projections.std(dim=0, correction=0) + 1e-9

            print(f"[MOS LOG]   ✓ PCA主成分提取完成，保留前{num_principal_components}个主成分")

            # 释放临时变量
            del U
            del S
            del V
            del benign_centered
            del benign_projections

        except Exception as e:
            print(f"[MOS WARNING]   ✗ PCA预计算失败: {e}，将跳过PCA约束")
            pca_principal_dirs = None

    log_cuda_memory("after PCA", mos_memory_debug)

    # 2. 子空间采样预计算（用于DNC子空间鲁棒性约束）
    subspace_samples = []
    enable_subspace = use_dnc and getattr(args, 'enable_subspace_constraint', True)

    if enable_subspace:
        num_subspace_samples = 3

        # 修复 9/10: 支持固定种子的子空间采样
        use_fixed_seed = getattr(args, 'subspace_fixed_seed', False)
        rng = random.Random(42) if use_fixed_seed else random.Random()

        if use_fixed_seed:
            print(f"[MOS LOG]   ✓ 使用固定种子(42)进行子空间采样（可复现）")

        for sample_idx in range(num_subspace_samples):
            try:
                # 修复 14/10：使用 random.sample 避免大 randperm
                D = benign_mean.numel()
                sample_size = min(1000, D)
                sampled_dims = torch.tensor(
                    rng.sample(range(D), sample_size),
                    device=device,
                    dtype=torch.long,
                )

                benign_sub = benign_grads[:, sampled_dims]
                benign_sub_mean = benign_sub.mean(dim=0)
                benign_sub_centered = benign_sub - benign_sub_mean

                _, _, V_sub = torch.linalg.svd(benign_sub_centered, full_matrices=False)
                v_sub = V_sub[0, :].clone().detach()

                # 计算良性梯度在该子空间主成分上的投影标准差
                benign_sub_scores = torch.abs((benign_sub_centered @ v_sub))
                benign_sub_std = benign_sub_scores.std(correction=0) + 1e-9

                subspace_samples.append({
                    'sampled_dims': sampled_dims,
                    'principal_component': v_sub,
                    'sub_mean': benign_sub_mean,
                    'benign_std': benign_sub_std
                })

                # 释放临时变量
                del benign_sub
                del benign_sub_centered
                del V_sub

            except Exception as e:
                print(f"[MOS WARNING]   ✗ 子空间样本{sample_idx}预计算失败: {e}")
                continue

        if subspace_samples:
            print(f"[MOS LOG]   ✓ 子空间采样完成，成功预计算{len(subspace_samples)}个子空间")
        else:
            enable_subspace = False

    log_cuda_memory("after subspace", mos_memory_debug)

    # 3. 计算良性梯度的约束loss阈值（打分系统的"及格线"）
    # 修复 7/12：使用统一的 compute_raw_constraint_losses 函数
    print(f"[MOS LOG]   🎯 计算良性梯度约束loss阈值...")

    # 获取超参数
    constraint_quantile = getattr(args, 'constraint_quantile', 0.95)

    # 构建临时 benign_refs（用于计算良性更新的约束 loss）
    temp_benign_refs = {
        'mean': benign_mean,
        'std': benign_std,
        'grads': benign_grads,
        'layer_dims': layer_dims,
        'pca_principal_dirs': pca_principal_dirs,
        'pca_benign_proj_std': pca_benign_proj_std,
        'subspace_samples': subspace_samples,
        'use_dnc': use_dnc,
        'device': device,
        'args': args,
    }

    # 使用统一函数计算良性更新的约束 loss
    with torch.no_grad():
        benign_losses = compute_raw_constraint_losses(benign_grads, temp_benign_refs)

        # 计算每个约束的阈值（使用分位数或稳健统计量）
        constraint_thresholds = {}

        # 修复 4/10：跳过未启用的约束
        active_constraints = ['radial', 'sign']
        if use_dnc:
            active_constraints.append('pca')
            if enable_subspace:
                active_constraints.append('subspace')

        enable_cohesion = getattr(args, 'enable_cohesion_constraint', False)
        if enable_cohesion:
            active_constraints.append('cohesion')

        for name, loss_tensor in benign_losses.items():
            if name in ['krum', 'group']:  # 别名，跳过
                continue

            # 跳过未启用的约束
            if name not in active_constraints:
                constraint_thresholds[name] = 0.0  # 设置为0，表示未启用
                continue

            # 检查样本数量
            if loss_tensor.numel() < 2:
                # 样本太少，使用保守阈值
                constraint_thresholds[name] = loss_tensor.max() * 1.5 + 0.1
                print(f"[MOS WARNING]   约束 {name}: 样本不足，使用保守阈值")
                continue

            # 使用分位数方法
            try:
                threshold = torch.quantile(loss_tensor, q=constraint_quantile)
            except:
                # 回退到稳健统计量: median + k * MAD
                median = torch.median(loss_tensor)
                mad = torch.median(torch.abs(loss_tensor - median))
                threshold = median + 2.0 * 1.4826 * mad

            constraint_thresholds[name] = threshold.item() + 1e-6  # 加小偏置避免数值问题

        # 添加别名以兼容旧代码
        constraint_thresholds['krum'] = constraint_thresholds.get('radial', 1.0)
        constraint_thresholds['group'] = constraint_thresholds.get('cohesion', 0.0)

    # 打印阈值信息和配置
    print(f"[MOS LOG]   📊 约束阈值（分位数 q={constraint_quantile}）:")
    print(f"[MOS LOG]      - Radial: {constraint_thresholds['radial']:.4f}")
    print(f"[MOS LOG]      - Sign: {constraint_thresholds['sign']:.4f}")
    if use_dnc and pca_principal_dirs is not None:
        print(f"[MOS LOG]      - PCA: {constraint_thresholds['pca']:.4f}")
    if enable_subspace:
        print(f"[MOS LOG]      - Subspace: {constraint_thresholds['subspace']:.4f}")
    if enable_cohesion:
        print(f"[MOS LOG]      - Cohesion: {constraint_thresholds['cohesion']:.4f} (experimental)")

    # 修复 1/10：打印配置信息
    sign_layer_reduce = getattr(args, 'sign_layer_reduce', 'quantile')
    subspace_reduce = getattr(args, 'subspace_reduce', 'max')
    constraint_score_temperature = getattr(args, 'constraint_score_temperature', 0.5)

    print(f"\n[MOS LOG] 🔧 Constraint configuration:")
    print(f"[MOS LOG]   Sign layer reduce: {sign_layer_reduce}")
    print(f"[MOS LOG]   Subspace reduce: {subspace_reduce}")
    print(f"[MOS LOG]   Constraint score temperature: {constraint_score_temperature}")

    print(f"[MOS LOG]   Active constraints: {', '.join(active_constraints)}")
    disabled = [name for name in ['radial', 'sign', 'pca', 'subspace', 'cohesion'] if name not in active_constraints]
    if disabled:
        print(f"[MOS LOG]   Disabled constraints: {', '.join(disabled)}")

    # 4. 打包良性参考数据（传递给compute_objectives）
    benign_refs = {
        'mean': benign_mean,
        'std': benign_std,
        'grads': benign_grads,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'survival_mask': survival_mask,
        'layer_dims': layer_dims,
        'pca_principal_dirs': pca_principal_dirs,
        'pca_benign_proj_std': pca_benign_proj_std,
        'subspace_samples': subspace_samples,
        'use_dnc': use_dnc,
        'device': device,
        'args': args,  # 修复 1/10：添加 args 字段
    }

    print(f"[MOS LOG] ✅ 良性参考系统预计算完成！\n")
    log_cuda_memory("after benign references", mos_memory_debug)
    target_ce = target_ce.detach()
    target_cw = target_cw.detach()

    # ========================================================================
    # 构造约束安全的 guidance (在 benign_refs 定义之后)
    # ========================================================================
    if has_valid_guidance:
        print(f"[MOS LOG] 🛡️ 构造约束安全的 guidance...")

        # 步骤1: 削弱与良性均值符号相反的维度
        sign_safe_weight = getattr(args, 'sign_safe_weight', 0.1)

        same_sign = (g_combined_unit * torch.sign(benign_mean) >= 0).float()
        g_safe = g_combined_unit * (sign_safe_weight + (1.0 - sign_safe_weight) * same_sign)

        # 步骤2: 移除部分随机子空间主成分投影
        subspace_repair_strength = getattr(args, 'subspace_repair_strength', 0.5)

        if use_dnc and subspace_samples:
            for sample in subspace_samples:
                dims = sample['sampled_dims']
                v = sample['principal_component']

                coeff = torch.dot(g_safe[dims], v)
                g_safe[dims] -= subspace_repair_strength * coeff * v

        # 步骤3: 重新归一化
        g_safe = g_safe / (torch.norm(g_safe) + 1e-12)

        print(f"[MOS LOG]   ✓ Sign safe weight: {sign_safe_weight}")
        print(f"[MOS LOG]   ✓ Subspace repair strength: {subspace_repair_strength}")
        print(f"[MOS LOG]   ✓ g_safe norm: {torch.norm(g_safe).item():.6f}")

        g_safe = g_safe.detach()
    else:
        # 安全回退：使用 benign_dir
        g_safe = benign_mean / (torch.norm(benign_mean) + 1e-9)
        g_safe = g_safe.detach()

    # 初始化或恢复历史种群
    if historical_pop is not None:
        historical_pop = historical_pop.to(device)
        hist_P = historical_pop.shape[0]

        # 适配到固定种群大小
        if hist_P == EVOLUTION_POP_SIZE:
            evolution_pop = historical_pop
        elif hist_P > EVOLUTION_POP_SIZE:
            evolution_pop = historical_pop[:EVOLUTION_POP_SIZE]
        else:
            indices = torch.randint(0, hist_P, (EVOLUTION_POP_SIZE - hist_P,), device=device)
            extra_pop = historical_pop[indices]
            evolution_pop = torch.cat([historical_pop, extra_pop], dim=0)

        # 修复 8/10：使用新预算和投影函数
        max_allowed_hist = history_max_ratio * max_dev_threshold
        evolution_pop, _, _ = project_to_attack_budget(
            evolution_pop,
            benign_mean,
            max_allowed_hist
        )

        # 衰减系数
        decay_factor = 0.95
        evolution_pop = benign_mean + (evolution_pop - benign_mean) * decay_factor + \
                        torch.randn(EVOLUTION_POP_SIZE, benign_mean.shape[0], device=device) * (0.01 * benign_std)
    else:
        # 第一轮：初始化固定大小的进化种群
        evolution_pop = benign_mean.clone().detach().repeat(EVOLUTION_POP_SIZE, 1) + \
                        torch.randn(EVOLUTION_POP_SIZE, benign_mean.shape[0], device=device) * (0.01 * benign_std)

    # 【精英种子注入】：将前几个个体替换为沿指导方向的精英
    if has_valid_guidance:
        if EVOLUTION_POP_SIZE >= 1:
            evolution_pop[0] = target_ce  # 极致的 CE 破坏者
        if EVOLUTION_POP_SIZE >= 2:
            evolution_pop[1] = target_cw  # 极致的 CW 破坏者
        if EVOLUTION_POP_SIZE >= 3:
            # 沿着混合方向的破坏者
            evolution_pop[2] = benign_mean + elite_combined_ratio * max_dev_threshold * g_combined_unit
        if EVOLUTION_POP_SIZE >= 4:
            # g_safe 精英作为主要引导种子
            evolution_pop[3] = benign_mean + elite_combined_ratio * max_dev_threshold * g_safe

    # 修复 6/10：初始种群投影到预算内
    evolution_pop, init_clipped, init_norms = project_to_attack_budget(
        evolution_pop,
        benign_mean,
        max_dev_threshold
    )

    init_clipped_ratio = init_clipped.float().mean().item()
    print(f"[MOS LOG] 📊 Initial population:")
    print(f"[MOS LOG]   Pre-projection norm: min={init_norms.min():.4f}, "
          f"mean={init_norms.mean():.4f}, max={init_norms.max():.4f}")
    print(f"[MOS LOG]   Clipped ratio: {init_clipped_ratio:.2%}")

    # 用于进化的种群（固定大小）
    malicious_set = evolution_pop

    log_cuda_memory("after population initialization", mos_memory_debug)

    log_cuda_memory("after population initialization", mos_memory_debug)

    generations = getattr(args, 'nsga_generations', 100)

    # 变异模式和尺度参数
    mutation_mode = getattr(args, 'mos_mutation_mode', 'benign_std')
    mutation_scale = getattr(args, 'mos_mutation_scale', 0.05)
    mutation_radius_ratio = getattr(args, 'mos_mutation_radius_ratio', 0.01)
    dir_step_ratio = getattr(args, 'mos_dir_step_ratio', 0.02)

    print(f"[MOS LOG] 🔧 Mutation configuration:")
    print(f"[MOS LOG]   Mode: {mutation_mode}")
    if mutation_mode == 'benign_std':
        print(f"[MOS LOG]   Scale: {mutation_scale} * benign_std")
    elif mutation_mode == 'unit_norm':
        print(f"[MOS LOG]   Radius ratio: {mutation_radius_ratio}")
    print(f"[MOS LOG]   Directional step ratio: {dir_step_ratio}")

    # 预先记录日志探针变量
    mutation_norms_log = []
    directional_push_norms_log = []

    # 引入 archive 作为跨代记忆（使用固定种群大小）
    archive = malicious_set.clone().detach()
    archive_size = EVOLUTION_POP_SIZE

    # 获取打分系统配置
    # 修复 2/10：统一命名为 'smooth'，兼容旧配置
    score_mode = getattr(args, "score_mode", "smooth")
    if score_mode == "sigmoid":
        score_mode = "smooth"  # 兼容旧配置

    constraint_epsilon = getattr(args, 'constraint_epsilon', 0.0)

    print(f"\n[MOS LOG] 🧬 开始进化循环，代数={generations}，种群大小={EVOLUTION_POP_SIZE}")
    print(f"[MOS LOG] 📊 打分系统：{score_mode} 映射")
    print(f"[MOS LOG] 📊 约束违反阈值 epsilon: {constraint_epsilon}")

    # 进化循环（使用 torch.no_grad() 优化显存）
    for it in range(generations):
        # 评估当前种群（双目标）
        # 修复 5/10：更新返回值处理
        objectives_current, scores_current, losses_current, diagnostics_current = compute_objectives(
            malicious_set,
            benign_refs,
            constraint_thresholds,
            g_safe,
            score_mode=score_mode
        )

        # 选择父代（NSGA-II选择）- 修复 18/10：不创建完整 parents 副本
        parent_idx = nsga2_select(objectives_current, malicious_set, EVOLUTION_POP_SIZE)

        # 生成子代：使用 SBX（Simulated Binary Crossover）+ 变异
        # 修复 19/10：预分配 offspring，不创建 children 列表
        num_children = EVOLUTION_POP_SIZE
        perm = torch.randperm(EVOLUTION_POP_SIZE, device=device)
        offspring = torch.empty_like(malicious_set)

        crossover_prob = getattr(args, 'sbx_crossover_prob', 0.9)
        eta = float(getattr(args, 'sbx_eta', 15.0))
        D = benign_mean.numel()

        # 修复 20/10：SBX 按维度分块（可选）
        mos_dimension_chunk_size = getattr(args, 'mos_dimension_chunk_size', 262144)
        use_chunked_sbx = D > mos_dimension_chunk_size

        for i in range(0, num_children, 2):
            p1 = malicious_set[parent_idx[perm[i % EVOLUTION_POP_SIZE]]]
            p2 = malicious_set[parent_idx[perm[(i+1) % EVOLUTION_POP_SIZE]]]

            if use_chunked_sbx:
                # 分块 SBX
                child1 = torch.empty_like(p1)
                child2 = torch.empty_like(p2)

                for start in range(0, D, mos_dimension_chunk_size):
                    end = min(start + mos_dimension_chunk_size, D)
                    chunk_size = end - start

                    p1_chunk = p1[start:end]
                    p2_chunk = p2[start:end]

                    u = torch.rand(chunk_size, device=device)
                    mask = torch.rand(chunk_size, device=device) <= crossover_prob
                    beta = torch.empty(chunk_size, device=device)
                    le = u <= 0.5
                    beta[le] = (2.0 * u[le]) ** (1.0 / (eta + 1.0))
                    beta[~le] = (1.0 / (2.0 * (1.0 - u[~le]))) ** (1.0 / (eta + 1.0))

                    c1_chunk = 0.5 * ((1 + beta) * p1_chunk + (1 - beta) * p2_chunk)
                    c2_chunk = 0.5 * ((1 - beta) * p1_chunk + (1 + beta) * p2_chunk)
                    c1_chunk[~mask] = p1_chunk[~mask]
                    c2_chunk[~mask] = p2_chunk[~mask]

                    child1[start:end] = c1_chunk
                    child2[start:end] = c2_chunk
            else:
                # 原始完整 SBX
                u = torch.rand(D, device=device)
                mask = torch.rand(D, device=device) <= crossover_prob
                beta = torch.empty(D, device=device)
                le = u <= 0.5
                beta[le] = (2.0 * u[le]) ** (1.0 / (eta + 1.0))
                beta[~le] = (1.0 / (2.0 * (1.0 - u[~le]))) ** (1.0 / (eta + 1.0))

                child1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                child2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
                child1[~mask] = p1[~mask]
                child2[~mask] = p2[~mask]

            # 修复 4/12：使用新的变异策略
            if mutation_mode == 'benign_std':
                # 基于良性标准差的逐维变异（默认推荐）
                noise1 = torch.randn_like(child1) * benign_std * mutation_scale
                noise2 = torch.randn_like(child2) * benign_std * mutation_scale
            elif mutation_mode == 'unit_norm':
                # 固定范数随机方向变异
                noise1 = torch.randn_like(child1)
                noise1 = noise1 / (torch.norm(noise1) + 1e-12)
                mutation_radius = mutation_radius_ratio * max_dev_threshold
                noise1 = noise1 * mutation_radius

                noise2 = torch.randn_like(child2)
                noise2 = noise2 / (torch.norm(noise2) + 1e-12)
                noise2 = noise2 * mutation_radius
            else:
                # 回退到默认（兼容旧代码）
                noise1 = torch.randn_like(child1) * benign_std * 0.05
                noise2 = torch.randn_like(child2) * benign_std * 0.05

            # 方向性推动（参数化，不再硬编码）
            dir_step = dir_step_ratio * max_dev_threshold

            # 让子代不仅有随机突变，还顺着指导梯度往前走一步
            if has_valid_guidance:
                directional_push = dir_step * g_safe
                push_scale1 = torch.empty(1, device=device).uniform_(0.5, 1.5).item()
                push_scale2 = torch.empty(1, device=device).uniform_(0.5, 1.5).item()
                noise1 = noise1 + directional_push * push_scale1
                noise2 = noise2 + directional_push * push_scale2

                # 记录方向性推动范数（用于日志）
                if (it + 1) % 10 == 0 or it == 0:
                    directional_push_norms_log.append(torch.norm(directional_push).item())

            # 记录变异噪声范数（用于日志）
            if (it + 1) % 10 == 0 or it == 0:
                mutation_norms_log.extend([torch.norm(noise1).item(), torch.norm(noise2).item()])

            # 应用生存掩码
            noise1 = noise1 * survival_mask
            noise2 = noise2 * survival_mask
            child1 = child1 + noise1
            child2 = child2 + noise2

            # 直接写入 offspring
            offspring[i] = child1
            if i + 1 < num_children:
                offspring[i + 1] = child2

        # 修复 6/10: 投影裁剪（变异后、评估前）
        offspring, offspring_clipped, offspring_pre_norms = project_to_attack_budget(
            offspring,
            benign_mean,
            max_dev_threshold
        )

        log_cuda_memory("after offspring", mos_memory_debug)

        # 修复 21/10：分别评估三个种群，避免创建完整 combined 张量
        log_cuda_memory("before objective evaluation", mos_memory_debug)

        objectives_offspring, _, _, _ = compute_objectives(
            offspring,
            benign_refs,
            constraint_thresholds,
            g_safe,
            score_mode=score_mode
        )

        objectives_archive, _, _, _ = compute_objectives(
            archive,
            benign_refs,
            constraint_thresholds,
            g_safe,
            score_mode=score_mode
        )

        # 只拼接小的目标矩阵
        objectives_combined = torch.cat(
            [objectives_current, objectives_offspring, objectives_archive],
            dim=1,
        )

        log_cuda_memory("after objective evaluation", mos_memory_debug)

        # NSGA-II 选择（基于拼接的目标）
        selected_global_idx = nsga2_select(objectives_combined, None, EVOLUTION_POP_SIZE)

        # 修复 21/10：根据索引从三个来源收集
        P = EVOLUTION_POP_SIZE
        next_population = torch.empty_like(malicious_set)

        for i, global_idx in enumerate(selected_global_idx):
            if global_idx < P:
                # 来自当前种群
                next_population[i] = malicious_set[global_idx]
            elif global_idx < 2 * P:
                # 来自 offspring
                next_population[i] = offspring[global_idx - P]
            else:
                # 来自 archive
                next_population[i] = archive[global_idx - 2 * P]

        malicious_set = next_population

        # 更新 archive（同样从三个来源收集）
        fronts = nondominated_sort(objectives_combined)
        new_archive_idx = []
        for f in fronts:
            for idx in f:
                new_archive_idx.append(idx)
                if len(new_archive_idx) >= archive_size:
                    break
            if len(new_archive_idx) >= archive_size:
                break

        if new_archive_idx:
            new_archive = torch.empty(len(new_archive_idx), D, device=device)
            for i, global_idx in enumerate(new_archive_idx):
                if global_idx < P:
                    new_archive[i] = malicious_set[global_idx]
                elif global_idx < 2 * P:
                    new_archive[i] = offspring[global_idx - P]
                else:
                    new_archive[i] = archive[global_idx - 2 * P]
            archive = new_archive[:archive_size]

        log_cuda_memory("generation end", mos_memory_debug)

        # 每10代打印一次进度（包含详细得分、变异统计和约束详情）
        if (it + 1) % 10 == 0 or it == 0:
            avg_stealth = -objectives_current[0].mean().item()
            avg_destructiveness = -objectives_current[1].mean().item()
            avg_cv = -objectives_current[2].mean().item()

            # 计算种群统计
            population_norms = torch.norm(malicious_set - benign_mean, dim=1)
            norm_ratios = population_norms / (max_dev_threshold + 1e-12)
            # 修复 7/10：使用投影前的范数统计真实裁剪比例（在 offspring 生成时已记录）

            # CV 统计
            population_cv = diagnostics_current['total_cv']
            min_cv = population_cv.min().item()
            mean_cv = population_cv.mean().item()
            feasible_count = (population_cv <= constraint_epsilon).sum().item()
            feasible_ratio = feasible_count / P

            # 提取当前代最优个体（第一前沿的首个）
            current_fronts = nondominated_sort(objectives_current)
            if current_fronts and current_fronts[0]:
                best_idx_current = current_fronts[0][0]
                best_stealth = -objectives_current[0, best_idx_current].item()
                best_destructiveness = -objectives_current[1, best_idx_current].item()
                best_cv_obj = -objectives_current[2, best_idx_current].item()
                selected_cv = population_cv[best_idx_current].item()

                # 提取各约束得分
                radial_s = scores_current.get('radial', scores_current.get('krum'))[best_idx_current].item()
                sign_s = scores_current['sign'][best_idx_current].item()
                pca_s = scores_current['pca'][best_idx_current].item()
                subspace_s = scores_current['subspace'][best_idx_current].item()
                cohesion_s = scores_current.get('cohesion', scores_current.get('group'))[best_idx_current].item()

                # 提取约束 loss（原始值，用于判断是否真正违反约束）
                radial_l = losses_current.get('radial', losses_current.get('krum'))[best_idx_current].item()
                sign_l = losses_current['sign'][best_idx_current].item()

                # 修复 5/10: 从 diagnostics 中提取约束违反度
                ratios_current = diagnostics_current['ratios']
                violations_current = diagnostics_current['violations']

                radial_ratio = ratios_current.get('radial', ratios_current.get('krum'))[best_idx_current].item()
                sign_ratio = ratios_current['sign'][best_idx_current].item()
                radial_violation = violations_current.get('radial', violations_current.get('krum'))[best_idx_current].item()
                sign_violation = violations_current['sign'][best_idx_current].item()

                print(f"[MOS LOG]   Generation {it+1}/{generations}: "
                      f"隐蔽性={avg_stealth:.3f}, 破坏性={avg_destructiveness:.3f}, CV={avg_cv:.3f}")
                print(f"[MOS LOG]     种群 CV: min={min_cv:.4f}, mean={mean_cv:.4f}, feasible={feasible_ratio:.2%}")
                print(f"[MOS LOG]     最优个体: 隐蔽性={best_stealth:.3f}, 破坏性={best_destructiveness:.3f}, CV={selected_cv:.4f}")
                print(f"[MOS LOG]     约束得分: Radial={radial_s:.3f}, Sign={sign_s:.3f}, "
                      f"PCA={pca_s:.3f}, Subspace={subspace_s:.3f}, Cohesion={cohesion_s:.3f}")
                print(f"[MOS LOG]     约束loss: Radial={radial_l:.4f} (阈值={constraint_thresholds.get('radial', constraint_thresholds.get('krum')):.4f}), "
                      f"Sign={sign_l:.4f} (阈值={constraint_thresholds['sign']:.4f})")
                print(f"[MOS LOG]     约束比值: Radial={radial_ratio:.3f}, Sign={sign_ratio:.3f}")
                print(f"[MOS LOG]     约束违反: Radial={radial_violation:.4f}, Sign={sign_violation:.4f}")

                # 变异统计
                if mutation_norms_log:
                    mut_min = min(mutation_norms_log)
                    mut_max = max(mutation_norms_log)
                    mut_mean = sum(mutation_norms_log) / len(mutation_norms_log)
                    print(f"[MOS LOG]     变异范数: min={mut_min:.4f}, mean={mut_mean:.4f}, max={mut_max:.4f}")
                    mutation_norms_log.clear()

                if directional_push_norms_log:
                    dir_norm = sum(directional_push_norms_log) / len(directional_push_norms_log)
                    print(f"[MOS LOG]     方向推动范数: {dir_norm:.4f}")
                    directional_push_norms_log.clear()

                # 种群统计
                print(f"[MOS LOG]     种群范数比例: min={norm_ratios.min():.3f}, "
                      f"mean={norm_ratios.mean():.3f}, max={norm_ratios.max():.3f}")
            else:
                print(f"[MOS LOG]   Generation {it+1}/{generations}: "
                      f"隐蔽性={avg_stealth:.3f}, 破坏性={avg_destructiveness:.3f}")

    # 进化完成后，选出最优模板并复制 K 份
    # 修复 5/10：更新返回值处理
    final_objectives, final_scores, final_losses, final_diagnostics = compute_objectives(
        malicious_set,
        benign_refs,
        constraint_thresholds,
        g_safe,
        score_mode=score_mode
    )

    final_fronts = nondominated_sort(final_objectives)

    print(f"\n[MOS LOG] 🏆 进化完成！Pareto前沿包含 {len(final_fronts[0])} 个个体")

    # 修复 5/10: 从 diagnostics 中提取约束违反统计
    final_violations = final_diagnostics['violations']
    final_ratios = final_diagnostics['ratios']
    final_total_cv = final_diagnostics['total_cv']

    print(f"[MOS LOG] 📊 约束违反统计（第一前沿）:")
    for name in diagnostics_current['active_constraints']:
        if name in final_violations:
            violations = final_violations[name][final_fronts[0]]
            num_violated = (violations > 0).sum().item()
            max_violation = violations.max().item()
            avg_violation = violations.mean().item()
            print(f"[MOS LOG]   {name.capitalize()}: {num_violated}/{len(final_fronts[0])} 违反, "
                  f"最大={max_violation:.4f}, 平均={avg_violation:.4f}")

    print(f"[MOS LOG] 📊 第一前沿个体详细得分：")
    print(f"[MOS LOG] {'索引':<6} {'隐蔽性':<8} {'破坏性':<12} {'CV_obj':<10} {'Radial':<8} {'Sign':<8} {'PCA':<8} {'Subspace':<10} {'CV':<8} {'Feasible':<10}")
    print(f"[MOS LOG] {'-'*115}")

    for idx in final_fronts[0]:
        stealth = -final_objectives[0, idx].item()
        destructiveness = -final_objectives[1, idx].item()
        cv_obj = final_objectives[2, idx].item()
        radial_score = final_scores.get('radial', final_scores.get('krum'))[idx].item() if idx < len(final_scores.get('radial', final_scores.get('krum'))) else 0.0
        sign_score = final_scores['sign'][idx].item() if idx < len(final_scores['sign']) else 0.0
        pca_score = final_scores['pca'][idx].item() if idx < len(final_scores['pca']) else 0.0
        subspace_score = final_scores['subspace'][idx].item() if idx < len(final_scores['subspace']) else 0.0
        cv = final_total_cv[idx].item()
        feasible = "Yes" if cv <= constraint_epsilon else "No"

        print(f"[MOS LOG] {idx:<6} {stealth:<8.3f} {destructiveness:<12.3f} {cv_obj:<10.4f} {radial_score:<8.3f} {sign_score:<8.3f} {pca_score:<8.3f} {subspace_score:<10.3f} {cv:<8.4f} {feasible:<10}")

    # 修复 5/10：选择最优解时 CV 优先级高于隐蔽性
    # 1. 优先选择可行解（CV <= epsilon）
    # 2. 若无可行解，选择 CV 最小的一组
    # 3. 在满足 CV 条件的解中，选择最接近理想点的

    front_indices = final_fronts[0]
    front_cv = final_total_cv[front_indices]
    stealth_scores = -final_objectives[0, front_indices]
    destructiveness_scores = -final_objectives[1, front_indices]

    # 步骤 1：筛选可行解
    feasible_mask = front_cv <= constraint_epsilon
    feasible_indices = [front_indices[i] for i in range(len(front_indices)) if feasible_mask[i]]

    if feasible_indices:
        print(f"\n[MOS LOG] 🎯 找到 {len(feasible_indices)} 个可行解（CV <= {constraint_epsilon}）")
        candidate_indices = feasible_indices
        candidate_cv = front_cv[feasible_mask]
    else:
        # 步骤 2：无可行解，选择 CV 最小的一组
        min_cv = front_cv.min().item()
        cv_threshold = min_cv * 1.1  # 允许 10% 的误差
        low_cv_mask = front_cv <= cv_threshold
        candidate_indices = [front_indices[i] for i in range(len(front_indices)) if low_cv_mask[i]]
        candidate_cv = front_cv[low_cv_mask]
        print(f"\n[MOS LOG] ⚠️ 无可行解，选择 CV 最小的 {len(candidate_indices)} 个个体（CV <= {cv_threshold:.4f}）")

    # 步骤 3：在候选集中选择最接近理想点的
    candidate_stealth = torch.tensor(
        [stealth_scores[front_indices.index(idx)] for idx in candidate_indices],
        device=device
    )
    candidate_destructiveness = torch.tensor(
        [destructiveness_scores[front_indices.index(idx)] for idx in candidate_indices],
        device=device
    )

    # 归一化破坏性到[0, 1]
    dest_min = candidate_destructiveness.min()
    dest_max = candidate_destructiveness.max()
    if dest_max - dest_min > 1e-9:
        candidate_destructiveness_norm = (candidate_destructiveness - dest_min) / (dest_max - dest_min)
    else:
        candidate_destructiveness_norm = torch.ones_like(candidate_destructiveness)

    # 计算欧氏距离到理想点(1.0, 1.0)
    distances = torch.sqrt((1.0 - candidate_stealth)**2 + (1.0 - candidate_destructiveness_norm)**2)

    # 选择距离最小的个体
    best_idx_in_candidate = torch.argmin(distances).item()
    best_idx = candidate_indices[best_idx_in_candidate]
    best_template = malicious_set[best_idx].clone().detach()

    best_cv = final_total_cv[best_idx].item()
    best_feasible = "Yes" if best_cv <= constraint_epsilon else "No"

    print(f"\n[MOS LOG] 🎯 最优解选择策略：CV 优先 + 最接近理想点")
    print(f"[MOS LOG] 🏆 选中个体索引: {best_idx}")
    print(f"[MOS LOG]   ✓ 隐蔽性得分: {-final_objectives[0, best_idx].item():.3f}")
    print(f"[MOS LOG]   ✓ 破坏性: {-final_objectives[1, best_idx].item():.3f}")
    print(f"[MOS LOG]   ✓ Total CV: {best_cv:.4f}")
    print(f"[MOS LOG]   ✓ Feasible: {best_feasible}")
    print(f"[MOS LOG]   ✓ 到理想点距离: {distances[best_idx_in_candidate].item():.4f}")
    print(f"[MOS LOG] 📋 将最优模板复制 {K} 份并添加微小噪声以规避聚类检测...")

    noise_scale = getattr(args, 'template_noise_scale', 1e-4)

    # 修复 23/10：逐个生成，避免同时创建 K×D 的三份张量
    log_cuda_memory("before final output", mos_memory_debug)

    output_norms = []
    noise_norms = []

    for i in range(K):
        # 生成噪声
        noise_i = torch.randn_like(best_template) * noise_scale * benign_std

        # 添加噪声
        optimized_grad_i = best_template + noise_i

        # 投影到预算内
        optimized_grad_i, _, norm_i = project_to_attack_budget(
            optimized_grad_i.unsqueeze(0),
            benign_mean,
            max_dev_threshold
        )
        optimized_grad_i = optimized_grad_i.squeeze(0)

        # 记录统计
        noise_norms.append(torch.norm(noise_i).item())
        output_norms.append(torch.norm(optimized_grad_i - benign_mean).item())

        # 转换回网络字典格式
        all_updates[i] = vector_to_net_dict(
            optimized_grad_i,
            copy.deepcopy(all_updates[i])
        )

    print(f"[MOS LOG] 🔊 噪声强度: {noise_scale} * benign_std")
    print(f"[MOS LOG] 🔊 实际噪声范数范围: [{min(noise_norms):.6f}, {max(noise_norms):.6f}]")
    print(f"[MOS LOG] 🔊 输出范数范围: [{min(output_norms):.4f}, {max(output_norms):.4f}], mean={sum(output_norms)/len(output_norms):.4f}")

    historical_perturbation = (best_template - benign_mean).unsqueeze(0).detach()
    pert_norm = torch.norm(historical_perturbation).item()
    print(f"[MOS LOG] 最优模板扰动范数: {pert_norm:.4f}")

    log_cuda_memory("attack exit", mos_memory_debug)
    print("-" * 50)

    return all_updates, historical_perturbation


def crowding_distance(front_indices, objs):
    M, _ = objs.shape
    F = len(front_indices)
    distances = torch.zeros(F, device=objs.device)

    # 边界情况：只有1-2个个体，全部保留
    if F <= 2:
        return torch.full((F,), float('inf'), device=objs.device)

    # 对每个目标维度计算拥挤度
    for m in range(M):
        # 提取该前沿在第m个目标上的值
        obj_values = objs[m, front_indices]
        sorted_idx = torch.argsort(obj_values)

        # 边界点设为无穷大（保证Pareto前沿的极值点被保留）
        distances[sorted_idx[0]] = float('inf')
        distances[sorted_idx[-1]] = float('inf')

        # 计算目标范围（归一化用）
        obj_range = obj_values[sorted_idx[-1]] - obj_values[sorted_idx[0]]
        if obj_range < 1e-9:
            continue  # 该目标维度无差异，跳过

        # 中间点：距离 = (右邻居 - 左邻居) / 范围
        for i in range(1, F - 1):
            distances[sorted_idx[i]] += (
                obj_values[sorted_idx[i+1]] - obj_values[sorted_idx[i-1]]
            ) / obj_range

    return distances


def nsga2_select(objs, population, pop_size):
    M, N = objs.shape
    fronts = nondominated_sort(objs)  # 复用现有的非支配排序

    chosen = []
    for front in fronts:
        if len(chosen) + len(front) <= pop_size:
            # 整个前沿都能放下，全选
            chosen.extend(front)
        else:
            # 最后一个前沿：需要筛选部分个体
            remaining_slots = pop_size - len(chosen)

            # 计算该前沿的拥挤度距离
            distances = crowding_distance(front, objs)

            # 按拥挤度降序排序，选择距离最大的个体（更分散）
            sorted_indices = torch.argsort(distances, descending=True)
            selected = [front[i] for i in sorted_indices[:remaining_slots].tolist()]
            chosen.extend(selected)
            break

    return chosen


def compute_objectives(pop, benign_refs, constraint_thresholds, g_combined_unit, score_mode='smooth'):
    """
    计算种群的双目标（隐蔽性和破坏性）

    Returns:
        objectives: (2, P) 目标矩阵
        constraint_scores: 各约束得分字典
        constraint_losses: 各约束原始损失字典
        diagnostics: 诊断信息字典（包含 violations, ratios, total_cv）
    """
    P = pop.shape[0]
    device = benign_refs['device']

    benign_mean = benign_refs['mean']
    survival_mask = benign_refs['survival_mask']

    # 获取温度参数
    temperature = getattr(benign_refs.get('args'), 'constraint_score_temperature', 0.5)

    # ========================================================================
    # Step 1: 计算 centered 一次，传递给约束计算
    # ========================================================================
    centered = pop - benign_mean

    # ========================================================================
    # Step 2: 使用统一函数计算所有约束 loss
    # ========================================================================
    constraint_losses = compute_raw_constraint_losses(pop, benign_refs, centered=centered)

    # ========================================================================
    # Step 3: 将每个约束 loss 转换为得分
    # ========================================================================
    # 修复 4/10：只处理已启用的约束
    args_obj = benign_refs.get('args')
    enable_cohesion = getattr(args_obj, 'enable_cohesion_constraint', False)
    use_dnc = benign_refs['use_dnc']
    enable_subspace = use_dnc and len(benign_refs.get('subspace_samples', [])) > 0

    # 确定活跃约束
    active_constraints = ['radial', 'sign']
    if use_dnc and benign_refs.get('pca_principal_dirs') is not None:
        active_constraints.append('pca')
    if enable_subspace:
        active_constraints.append('subspace')
    if enable_cohesion:
        active_constraints.append('cohesion')

    constraint_scores = {}
    for name, loss in constraint_losses.items():
        if name in ['krum', 'group']:  # 跳过别名
            continue

        threshold = constraint_thresholds.get(name, 1.0)

        # 只为活跃约束计算得分
        if name in active_constraints and threshold > 0:
            constraint_scores[name] = compute_constraint_score(
                loss, threshold, mode=score_mode, temperature=temperature
            )
        else:
            # 未启用的约束不计算得分
            constraint_scores[name] = torch.zeros(P, device=device)

    # 添加别名
    constraint_scores['krum'] = constraint_scores.get('radial', torch.zeros(P, device=device))
    constraint_scores['group'] = constraint_scores.get('cohesion', torch.zeros(P, device=device))

    # ========================================================================
    # Step 4: 计算约束违反度 (Constraint Violation)
    # ========================================================================
    constraint_violations, constraint_ratios = compute_constraint_violations(
        constraint_losses, constraint_thresholds
    )

    # 计算总违反度（只对活跃约束）
    constraint_weights = {
        'radial': getattr(args_obj, 'weight_radial', 1.0),
        'sign': getattr(args_obj, 'weight_sign', 0.5),
        'pca': getattr(args_obj, 'weight_pca', 1.0),
        'subspace': getattr(args_obj, 'weight_subspace', 0.5),
        'cohesion': getattr(args_obj, 'weight_cohesion', 0.3),
    }

    total_cv = torch.zeros(P, device=device)
    for name in active_constraints:
        weight = constraint_weights.get(name, 1.0)
        total_cv += weight * constraint_violations[name]

    # 打包诊断信息
    diagnostics = {
        'ratios': constraint_ratios,
        'violations': constraint_violations,
        'total_cv': total_cv,
        'active_constraints': active_constraints,
    }

    # ========================================================================
    # Step 5: 加权求和得到总隐蔽性得分 (stealth_score)
    # ========================================================================
    total_score = torch.zeros(P, device=device)
    total_weight = 0.0

    for name in active_constraints:
        weight = constraint_weights.get(name, 1.0)
        threshold = constraint_thresholds.get(name, 0.0)

        # 只计入已启用的约束（阈值 > 0）
        if threshold > 0:
            total_score += constraint_scores[name] * weight
            total_weight += weight

    # 归一化到 [0, 1]
    stealth_score = total_score / (total_weight + 1e-12)

    # ========================================================================
    # Step 6: 计算破坏性（与指导梯度的对齐度）
    # ========================================================================
    # 修复 16/10：使用 centered 避免重复计算 masked_deviation
    masked_guidance = survival_mask * g_combined_unit

    # 对齐度：与指导梯度的点积（越大越好）
    alignment = centered @ masked_guidance

    # 目标1：隐蔽性（负号转换：最大化得分 → 最小化负得分）
    obj_stealth = -stealth_score

    # 目标2：破坏性（负号转换：最大化对齐 → 最小化负对齐）
    obj_destructiveness = -alignment

    # 目标3：CV（约束违反度，使用 log1p 平滑）
    obj_cv = torch.log1p(total_cv)

    objectives = torch.stack([obj_stealth, obj_destructiveness, obj_cv], dim=0)

    return objectives, constraint_scores, constraint_losses, diagnostics


# ------ 非支配排序（NSGA-II 和 NSGA-III 共用）------
def nondominated_sort(objs):
    M, N = objs.shape
    S = [set() for _ in range(N)]  # 每个个体支配的个体集合
    n = torch.zeros(N, dtype=torch.int32, device=objs.device)  # 支配该个体的个体数
    rank = torch.full((N,), -1, dtype=torch.int32, device=objs.device)
    fronts = []

    # 计算支配关系
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            # p支配q？（所有目标不差，至少一个目标更优）
            less_eq = torch.all(objs[:, p] <= objs[:, q])
            strictly_less = torch.any(objs[:, p] < objs[:, q])
            if less_eq and strictly_less:
                S[p].add(q)  # p支配q
            elif torch.all(objs[:, q] <= objs[:, p]) and torch.any(objs[:, q] < objs[:, p]):
                n[p] += 1  # q支配p

        # 没有被支配的个体属于Front 0
        if n[p] == 0:
            rank[p] = 0

    # Front 0
    current_front = [i for i in range(N) if rank[i] == 0]
    i = 0

    # 逐层构建后续前沿
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

def crowding_distance(front_indices, objs):
    """
    计算拥挤度距离（NSGA-II核心组件）
    """
    M, _ = objs.shape
    F = len(front_indices)
    distances = torch.zeros(F, device=objs.device)

    if F <= 2:
        return torch.full((F,), float('inf'), device=objs.device)

    for m in range(M):
        obj_values = objs[m, front_indices]
        sorted_idx = torch.argsort(obj_values)
        distances[sorted_idx[0]] = float('inf')
        distances[sorted_idx[-1]] = float('inf')
        obj_range = obj_values[sorted_idx[-1]] - obj_values[sorted_idx[0]]
        if obj_range < 1e-9:
            continue
        for i in range(1, F - 1):
            distances[sorted_idx[i]] += (
                obj_values[sorted_idx[i+1]] - obj_values[sorted_idx[i-1]]
            ) / obj_range

    return distances


def nsga2_select(objs, population, pop_size):
    M, N = objs.shape
    fronts = nondominated_sort(objs)  # 复用现有的非支配排序

    chosen = []
    for front in fronts:
        if len(chosen) + len(front) <= pop_size:
            # 整个前沿都能放下，全选
            chosen.extend(front)
        else:
            # 最后一个前沿：需要筛选部分个体
            remaining_slots = pop_size - len(chosen)

            # 计算该前沿的拥挤度距离
            distances = crowding_distance(front, objs)

            # 按拥挤度降序排序，选择距离最大的个体（更分散）
            sorted_indices = torch.argsort(distances, descending=True)
            selected = [front[i] for i in sorted_indices[:remaining_slots].tolist()]
            chosen.extend(selected)
            break

    return chosen
