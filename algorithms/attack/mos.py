
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
    
    # -----------------------------------------
    # 代理目标 1：Cross Entropy (CE) - 基础破坏
    # -----------------------------------------
    loss_ce = criterion_ce(outputs, target_labels)
    
    # 保留计算图以便后续还能算第二个梯度的反向传播
    loss_ce.backward(retain_graph=True) 
    
    # 提取并展平 CE 梯度
    g_ce_list = [param.grad.clone().flatten() for param in global_model.parameters() if param.grad is not None]
    g_ce = torch.cat(g_ce_list)
    
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
    loss_cw = torch.mean(torch.relu(max_other_logits - correct_logits + 50.0)) # 50.0 是 margin 余量
    
    loss_cw.backward()
    
    # 提取并展平 CW 梯度
    g_cw_list = [param.grad.clone().flatten() for param in global_model.parameters() if param.grad is not None]
    g_cw = torch.cat(g_cw_list)
    
    global_model.zero_grad()
    
    return g_ce, g_cw

def tr_mean(all_updates, n_attackers):
    # 简单的 TrMean 辅助函数，用于二分搜索中的验证
    sorted_updates, _ = torch.sort(all_updates, dim=0)
    out = torch.mean(sorted_updates[n_attackers:-n_attackers], dim=0)
    return out


def binary_search_box_fit(benign_mean, malicious_direction, benign_min, benign_max, n_attackers, all_updates_benign):
    """
    二分搜索：沿着 malicious_direction 寻找最大的缩放系数 lambda。
    约束 1 (硬): 必须严格在 [benign_min, benign_max] 盒子里。
    约束 2 (软): (可选) 放入 TrMean 模拟器中，确保不被切除（针对 Oracle）。
    """
    # 1. 预处理方向
    # 归一化方向，方便控制 lambda 的物理含义
    dir_norm = torch.norm(malicious_direction)
    if dir_norm < 1e-9:
        return benign_mean
    unit_direction = malicious_direction / dir_norm
    
    # 2. 计算几何极限 (Ray Casting)
    # 我们先算出如果不考虑 TrMean 切除，仅仅是为了在 Box 里，lambda 最大能是多少？
    # 公式：lambda_max = min( (Bound - Mean) / Direction )
    # 对于每个维度 i:
    #   如果 direction[i] > 0: limit = (max[i] - mean[i]) / direction[i]
    #   如果 direction[i] < 0: limit = (min[i] - mean[i]) / direction[i]
    
    # 避免除以 0
    safe_dir = unit_direction.clone()
    safe_dir[abs(safe_dir) < 1e-9] = 1e-9
    
    t1 = (benign_max - benign_mean) / safe_dir
    t2 = (benign_min - benign_mean) / safe_dir
    
    # 取正值的 t (代表沿方向延伸的距离)
    # 当 dir > 0 时，t1 是正的边界；当 dir < 0 时，t2 是正的边界
    # 我们需要取所有维度中限制最死的那个 (min)
    potential_t = torch.max(t1, t2) # 只有同号的除法才是正的距离
    # 过滤掉异号除法产生的负值 (反方向的边界)
    potential_t[potential_t < 0] = float('inf') 
    
    # 理论上的最大几何边界
    max_geom_lambda = torch.min(potential_t).item()
    
    # 3. 开始二分搜索 (在几何边界内进一步试探 Oracle TrMean)
    # 如果你觉得 Oracle TrMean 太难搞，可以直接返回 max_geom_lambda * unit_direction
    # 但为了稳妥，我们还是搜一下
    
    low = 0.0
    high = max_geom_lambda
    # 稍微退缩一点点，防止浮点数误差导致刚好触线被判出局
    high = high * 0.99 
    
    best_mal_update = benign_mean
    
    # 二分搜索 20 次足以达到极高精度
    for _ in range(20):
        mid = (low + high) / 2
        
        # 生成候选攻击
        candidate = benign_mean + mid * unit_direction
        
        # --- Oracle TrMean 检测 ---
        # 构造场景
        mal_stack = torch.stack([candidate] * n_attackers)
        combined = torch.cat((all_updates_benign, mal_stack), 0) # 注意这里需要外部传入 benign updates
        
        # 模拟 TrMean (假设切 n_attackers 个)
        # 这里的逻辑是：如果我这个候选向量被切了，那我的攻击就是无效的(或者变成了良性均值)
        # 我们如何判断是否被切？
        # 简单方法：看 loss 是否比仅仅填入 benign_mean 要大
        
        agg_result = tr_mean(combined, n_attackers)
        loss = torch.norm(agg_result - benign_mean)
        
        # 基准 loss (如果攻击者全是良性均值)
        # 其实只要 loss > 0 且随着 mid 增大而增大，说明没被切
        # 但在 Oracle TrMean 下，一旦被切，loss 就会瞬间掉下去或变成常数
        
        # 更直接的检测：检查 candidate 是否处于 sorted_updates 的保留区间内
        # 这需要把 tr_mean 拆开写，或者简单通过 Loss 判断
        # 这里我们假设：如果 mid 越大，loss 越大，说明有效。如果 mid 变大 loss 反而变小，说明被切。
        
        # 为了简单，我们只用几何边界 (In-lier 策略)，因为 min/max 已经是完美伪装了
        # 只要在 min/max 内，Oracle TrMean 就没法把你当做"明显离群值"
        # 除非你的数量太多占据了优势。
        
        # 所以，在这个特定场景下，直接返回几何边界可能是最优解！
        pass 
        
    return benign_mean + high * unit_direction


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
def mos_attack(all_updates, args, malicious_attackers_this_round, g_ce=None, g_cw=None):
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
    
    for update in all_updates:
        vec = torch.cat([torch.flatten(update[k]) for k in update.keys()])
        all_updates_flatten.append(vec)
    all_stack = torch.stack(all_updates_flatten).to(device)
    
    # 提取良性梯度
    benign_grads = all_stack[malicious_attackers_this_round:].detach()
    benign_mean = torch.mean(benign_grads, dim=0)
    
    # 获取约束边界 (和以前一样)
    benign_std = torch.std(benign_grads, dim=0) + 1e-9
    dists_benign = torch.norm(benign_grads - benign_mean, dim=1)
    krum_radius = torch.max(dists_benign) * 1.1
    benign_min, _ = torch.min(benign_grads, dim=0)
    benign_max, _ = torch.max(benign_grads, dim=0)
    lower_bound = benign_min
    upper_bound = benign_max
    
# ================= 新增：提取 Skew 攻击理论方向 =================
    # 这个似乎有点菜，先废弃了
    # 1. 计算中位数 (Median)
    benign_median, _ = torch.median(benign_grads, dim=0)
    
    # 2. 计算偏斜方向 (Direction of Skew) [论文中的 u_search]
    skew_direction = benign_median - benign_mean
    
    # 3. 归一化方向向量，防止因为原始梯度数值太小导致 Loss 梯度爆炸/消失
    skew_dir_norm = torch.norm(skew_direction)
    skew_dir_unit = skew_direction / (skew_dir_norm + 1e-9)
    skew_dir_unit = skew_dir_unit.to(device)
    # ================================================================
    
    # ================= 关键新增区：处理传进来的指导梯度 =================
    if g_ce is not None and g_cw is not None:
        # 指导梯度只给出了最佳破坏方向，我们赋予它合适的"长度"，使其与正常梯度量级匹配
        scale_factor = torch.norm(benign_mean) * 2
        
        g_ce_unit = g_ce.to(device) / (torch.norm(g_ce.to(device)) + 1e-9)
        g_cw_unit = g_cw.to(device) / (torch.norm(g_cw.to(device)) + 1e-9)
        
        # 定义真实的有毒目标位置 (良性中心点 + 恶意的语义方向)
        target_ce = benign_mean + scale_factor * g_ce_unit
        target_cw = benign_mean + scale_factor * g_cw_unit
    else:
        # 兜底：如果外部忘了传，退化回以前瞎猜的坐标
        target_ce = benign_mean - 5.0 * benign_mean
        target_cw = benign_mean - 8.0 * benign_mean
        
    target_ce = target_ce.detach()
    target_cw = target_cw.detach()
    # ====================================================================

    # 初始化 MOS 优化器
    malicious_set = benign_mean.clone().detach().repeat(K, 1) + \
                    torch.randn(K, benign_mean.shape[0]).to(device) * (0.5 * benign_std)
    malicious_set.requires_grad = True
    
    optimizer = torch.optim.Adam([malicious_set], lr=0.1) 
    normalizer = LossNormalizer(num_objectives=7, momentum=0.8)  # 你的代码里原有的 normalizer
    mu = 0.5 

    for it in range(100):
        optimizer.zero_grad()
        
        # ================= Loss 计算更新 =================
        # Objective 1: 向真实的 CE 漏洞方向冲锋
        l_ce = torch.norm(malicious_set - target_ce, dim=1)
        
        # Objective 2: 向真实的 CW 漏洞方向冲锋
        l_cw = torch.norm(malicious_set - target_cw, dim=1)
        
        # 新代码：方向最大化 (负内积)
        # 我们希望 malicious_set 沿着指导梯度方向走得越远越好
        # 也就是说，(malicious_set - benign_mean) 与 指导梯度的夹角越小越好，投影越长越好

        # 1. 计算当前偏离中心的方向向量
        current_deviation = malicious_set - benign_mean

        # 2. 计算与指导梯度的内积 (点乘)
        # 内积越大，说明沿着破坏方向走得越远。为了配合框架最小化 Loss 的逻辑，我们取负数。
        #l_ce = -torch.sum(current_deviation * g_ce_unit, dim=1)
        #l_cw = -torch.sum(current_deviation * g_cw_unit, dim=1)
        
        l_skew = -torch.sum(current_deviation * skew_dir_unit, dim=1)
        
        # Objective 3: 隐蔽性 - Krum 保护圈
        current_dist = torch.norm(malicious_set - benign_mean, dim=1)
        l_krum = torch.relu(current_dist - krum_radius)
        
        # Objective 4: 隐蔽性 - TrMean 盒式伪装
        excess_lower = torch.relu(lower_bound - malicious_set)
        excess_upper = torch.relu(malicious_set - upper_bound)
        l_box = torch.norm(excess_lower + excess_upper, dim=1)
        
        # Objective 5 (新增): 恶意团体极度抱团
        malicious_mean = torch.mean(malicious_set, dim=0)
        l_group = torch.norm(malicious_set - malicious_mean, dim=1)
        
        # Objective 6: 针对LASA的符号锁死器
        sign_violation = -malicious_set * torch.sign(benign_mean)
            # 只有当违反符号（相乘为负，再取反变正）时，才产生惩罚
        l_sign = torch.norm(torch.relu(sign_violation), dim=1)
        
        # Objective 7: 针对LASA的中位数截断
        deviation = torch.abs(malicious_set - benign_mean)
        # 极其保守：活动空间只有 0.1 个标准差，相当于紧紧贴着良性均值
        excess_lasa = torch.relu(deviation - 0.1 * benign_std)
        l_lasa_norm = torch.norm(excess_lasa, dim=1)
        
        # ================= 新增 2：计算输出层精准限幅 Loss =================
        # 1. 切片提取我们恶意样本的输出层参数
        # view 操作将其恢复成 (K, num_classes, in_features) 的三维形状
        mal_out_w = malicious_set[:, idx_w_start:idx_w_end].view(K, num_classes, -1)
        mal_out_b = malicious_set[:, idx_b_start:idx_b_end].view(K, num_classes, 1)
        
        # 将 weight 和 bias 拼在一起，方便一次性计算每个神经元的总梯度大小
        mal_out_params = torch.cat([mal_out_w, mal_out_b], dim=2) 
        
        # 计算恶意更新中，每个神经元的 L2 范数 -> shape: (K, num_classes)
        neuron_norms = torch.norm(mal_out_params, dim=2)
        
        # 2. 提取良性均值 (benign_mean) 的输出层参数作为“安全上限”
        with torch.no_grad():
            ben_out_w = benign_mean[idx_w_start:idx_w_end].view(num_classes, -1)
            ben_out_b = benign_mean[idx_b_start:idx_b_end].view(num_classes, 1)
            ben_out_params = torch.cat([ben_out_w, ben_out_b], dim=1)
            ben_neuron_norms = torch.norm(ben_out_params, dim=1) # shape: (num_classes,)
            
        # 3. 核心避险逻辑：如果恶意神经元的幅度超过了良性均值，就产生 Loss
        # 使用 ReLU 截断：只有 neuron_norms > ben_neuron_norms 时，才有值
        excess_magnitude = torch.relu(neuron_norms - ben_neuron_norms.unsqueeze(0))
        
        # 把超出的部分转化为标量 Loss
        l_output_magnitude = torch.norm(excess_magnitude, dim=1)
        # =================================================================
        
        
        
        # 堆叠 Loss (和原逻辑一致)
        raw_losses = torch.stack([l_ce,l_cw, l_krum, l_lasa_norm ,l_sign,l_box,l_output_magnitude])
        # ===============================================
        
        # 执行你写好的平滑极值 MOS 优化
        norm_losses = normalizer.update_and_normalize(raw_losses)
        score_per_obj = -mu * torch.logsumexp(-norm_losses / mu, dim=1)
        total_loss = mu * torch.logsumexp(score_per_obj / mu, dim=0)
        
        total_loss.backward()
        optimizer.step()
        
        # 投影兜底
        with torch.no_grad():
            malicious_set.data = torch.max(torch.min(malicious_set.data, upper_bound), lower_bound)

    # 还原代码 (保持不变)
    optimized_grads = malicious_set.detach()
    for i in range(K):
        all_updates[i] = vector_to_net_dict(optimized_grads[i], copy.deepcopy(all_updates[i]))

    return all_updates


"""
print(f"DEBUG: Final Attack Norm: {atk_norm:.2f} (Target < {ben_norm + krum_radius:.2f})")
        print(f"DEBUG: Norm Losses -> Attack:{score_per_obj[0]:.3f} | Krum:{score_per_obj[1]:.3f} | Box:{score_per_obj[2]:.3f}")
        print(f"DEBUG: Attack Mean Norm: {torch.norm(torch.mean(optimized_grads, 0)):.4f} | Benign Mean Norm: {torch.norm(benign_mean):.4f}")
    """