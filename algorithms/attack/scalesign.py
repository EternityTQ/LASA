import torch
import numpy as np

def scalesign_attack(all_updates, args, malicious_attackers_this_round, threshold=10):
    """
    ScaleSign 攻击：结合“缩放”和“符号修改”来绕过多种防御机制。
    
    参数:
    - all_updates: 本轮所有客户端上传的梯度（前 malicious_attackers_this_round 个属于攻击者）
    - args: 框架配置参数
    - malicious_attackers_this_round: 本轮参与攻击的恶意客户端数量
    - threshold: 初始探索步长
    """
    
    # 1. 提取诚实客户端的梯度，模拟攻击者“窃听”到的全局或诚实信息
    all_updates_flatten = []
    for update in all_updates:
        # 将每个客户端的模型参数字典展平成一维向量，方便后续做数学运算
        update_flat = torch.cat([torch.flatten(update[k]) for k in update.keys()])
        all_updates_flatten.append(update_flat)
    all_updates_flatten = torch.stack(all_updates_flatten)
    
    # 计算诚实客户端的平均梯度，作为我们要伪装的“参考目标”
    model_re = torch.mean(all_updates_flatten[malicious_attackers_this_round:], dim=0)
    
    # 统计正常梯度的“符号特征”：正数(P)、负数(N)、零(Z)的数量
    P_ref = torch.sum(model_re > 0).item()
    N_ref = torch.sum(model_re < 0).item()
    Z_ref = torch.sum(model_re == 0).item()

    # 2. 缩放攻击 (Scaling Attack)
    # 目的：放大正梯度的毒性，同时保证和正常梯度的“方向（余弦相似度）”高度一致
    pos_mask = model_re > 0
    
    lamda = torch.Tensor([threshold]).float().to(args.device)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    
    # 计算诚实梯度之间最大的距离，作为隐蔽性的约束条件
    benign_dists = []
    for update in all_updates_flatten[malicious_attackers_this_round:]:
        benign_dists.append(torch.norm(model_re - update) ** 2)
    max_distance = torch.max(torch.tensor(benign_dists)) if len(benign_dists) > 0 else 10.0

    # 使用二分查找法，寻找在不被发现（距离小于 max_distance）的前提下，能放大的最大倍数 gamma (即这里的 lamda)
    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = model_re.clone()
        mal_update[pos_mask] = mal_update[pos_mask] * lamda  # 只缩放正数部分
        
        distance = torch.norm(model_re - mal_update) ** 2
        
        if distance <= max_distance:
            lamda_succ = lamda  # 成功，尝试进一步放大
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2 # 失败（被发现了），缩小倍数
        lamda_fail = lamda_fail / 2

    # 应用找到的最佳放大倍数
    mal_update = model_re.clone()
    mal_update[pos_mask] = mal_update[pos_mask] * lamda_succ
    
    # 3. 符号修改 (Sign Modification)
    # 目的：把生成的恶意梯度的正负比例，强制修剪得跟正常梯度一模一样，专门欺骗 SignGuard
    P_mal = torch.sum(mal_update > 0).item()
    
    if P_ref < P_mal:
        # 如果恶意梯度的正数偏多，就把最小的那几个正数强制变成 0
        pos_indices = torch.where(mal_update > 0)[0]
        pos_values = mal_update[pos_indices]
        _, sort_idx = torch.sort(pos_values) 
        to_zero_indices = pos_indices[sort_idx[:(P_mal - P_ref)]]
        mal_update[to_zero_indices] = 0
    elif P_ref > P_mal:
        # 如果正数偏少，就把最大的几个负数或零变成一个极其微小的正数
        non_pos_indices = torch.where(mal_update <= 0)[0]
        non_pos_values = mal_update[non_pos_indices]
        _, sort_idx = torch.sort(non_pos_values, descending=True) 
        min_pos = torch.min(mal_update[mal_update > 0]) if torch.sum(mal_update > 0) > 0 else 1e-6
        to_pos_indices = non_pos_indices[sort_idx[:(P_ref - P_mal)]]
        mal_update[to_pos_indices] = min_pos
        
    # 对负数做类似的修剪
    N_mal_new = torch.sum(mal_update < 0).item()
    if N_ref < N_mal_new:
        neg_indices = torch.where(mal_update < 0)[0]
        neg_values = mal_update[neg_indices]
        _, sort_idx = torch.sort(neg_values, descending=True) # 取绝对值最小的负数
        to_zero_indices = neg_indices[sort_idx[:(N_mal_new - N_ref)]]
        mal_update[to_zero_indices] = 0

    # 4. 组装并替换攻击者的本地梯度
    # 将一维向量重新变回网络模型的字典结构
    mal_update = mal_update.unsqueeze(0).repeat(malicious_attackers_this_round, 1)
    
    flattened = [torch.flatten(all_updates[0][k]) for k in all_updates[0].keys()]
    idx = []
    s = 0
    for p in flattened:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
        
    for i in range(malicious_attackers_this_round):
        all_updates[i] = {k: mal_update[i,:][s:d].reshape(all_updates[-1][k].shape) for k, (s, d) in zip(all_updates[-1].keys(), idx)}

    return all_updates