import torch
import copy

def bisection(a, b, tol, f):
    """
    用于寻找攻击强度 s 的二分查找函数
    """
    low, high = a, b
    while high - low > tol:
        mid = (low + high) / 2.0
        if f(mid) > 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0

def skew_attack(all_updates, args, malicious_attackers_this_round):
    """
    LASA 框架兼容的 Skew 攻击算法
    """
    if malicious_attackers_this_round == 0:
        return all_updates

    # 1. 提取并展平所有客户端的更新 (Flatten updates)
    all_updates_flatten = []
    for update in all_updates:
        vec = torch.cat([torch.flatten(update[k]) for k in update.keys()])
        all_updates_flatten.append(vec)
    all_updates_tensor = torch.stack(all_updates_flatten)

    n_byz = malicious_attackers_this_round
    n_ben = len(all_updates) - n_byz

    # 2. 确定参考更新 (Reference updates)
    # 通常 Skew 攻击需要利用诚实客户端的梯度作为参考，如果诚实客户端被全部控制，则使用恶意客户端自身的真实梯度
    if n_ben > 0:
        flat_ref_updates = all_updates_tensor[n_byz:]
    else:
        flat_ref_updates = all_updates_tensor[:n_byz]

    # 3. 计算偏度方向 (Skew direction)
    flat_avg = flat_ref_updates.mean(dim=0)
    flat_med = flat_ref_updates.median(dim=0)[0]
    flat_skew_dir = flat_med - flat_avg

    # 4. 核心攻击逻辑
    if n_ben <= n_byz:
        if n_ben == 1:
            flat_byz_update = -10 * flat_avg
        else:
            flat_byz_update = flat_avg + flat_skew_dir * 10
    else:
        # 寻找偏移比例最高的 n_skew 个梯度
        n_skew = int((n_ben - n_byz) / n_ben * len(flat_ref_updates))
        inner_product = flat_ref_updates @ flat_skew_dir
        _, skew_idxs = inner_product.topk(k=n_skew, sorted=False)
        flat_skew_updates = flat_ref_updates[skew_idxs]
        
        flat_skew_avg = flat_skew_updates.mean(dim=0)
        flat_dev = (flat_skew_avg - flat_avg).sign() * flat_skew_updates.std(dim=0, unbiased=False)
        skew_diameter = torch.cdist(flat_skew_updates, flat_skew_updates).max().item()
        
        # 定义二分查找的优化目标 f(s)
        def f(s: float):
            flat_byz_update_tmp = flat_skew_avg + s * flat_dev
            dists = (flat_byz_update_tmp - flat_skew_updates).norm(dim=-1)
            max_dist = dists.max().item()
            return max_dist - skew_diameter
        
        max_s = 10.0
        s = bisection(0.0, max_s, 1e-5, f)
        
        # 如果 args 中没有设定 skew_lambda，默认赋予 1.0 的强度
        skew_lambda = getattr(args, 'skew_lambda', 1.0)
        strength = skew_lambda * s
        flat_byz_update = flat_skew_avg + strength * flat_dev

    # 5. 将生成的恶意 1D 向量还原回模型的字典结构 (Unflatten updates)
    pointer = 0
    mal_update_dict = copy.deepcopy(all_updates[0])
    for key in mal_update_dict.keys():
        num_param = mal_update_dict[key].numel()
        # 切片截取对应长度，并 reshape 成原本的参数形状
        mal_update_dict[key].data = flat_byz_update[pointer:pointer + num_param].view_as(mal_update_dict[key]).data
        pointer += num_param

    # 6. 将所有恶意客户端的更新替换为中毒梯度
    for i in range(malicious_attackers_this_round):
        all_updates[i] = copy.deepcopy(mal_update_dict)

    return all_updates