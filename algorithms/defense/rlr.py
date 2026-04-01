import torch
import numpy as np
import copy

# ==========================================
# 辅助函数：字典与一维向量的互相转换
# (借用自 LASA 框架自带的工具函数)
# ==========================================
def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    """将模型参数字典拉平成一个一维向量"""
    vec = []
    for key, param in net_dict.items():
        vec.append(param.view(-1))
    return torch.cat(vec)

def vector_to_net_dict(vec: torch.Tensor, net_dict) -> dict:
    """将一维向量还原回模型参数字典的形状"""
    pointer = 0
    for param in net_dict.values():
        num_param = param.numel()
        param.data = vec[pointer:pointer + num_param].view_as(param).data
        pointer += num_param
    return net_dict

# ==========================================
# 核心防御算法：Robust LR + Comed/Sign/Avg
# ==========================================
def robust_aggregation(local_updates, global_model, args):
    """
    local_updates: List[Dict]，各个客户端传上来的更新/模型参数
    global_model: Dict，当前全局模型的参数
    args: 配置参数集
    """
    # 1. 数据格式转换：把 LASA 的 List[Dict] 转成你代码习惯的字典形式 {id: flat_tensor}
    flat_local_updates = []
    for update_dict in local_updates:
        flat_param = parameters_dict_to_vector_flt(update_dict)
        flat_local_updates.append(flat_param)
        
    agent_updates_dict = {i: param for i, param in enumerate(flat_local_updates)}
    
    # 获取全局模型的扁平化一维向量
    flat_global_params = parameters_dict_to_vector_flt(global_model)
    n_params = flat_global_params.numel()

    # 2. 提取 args 中的超参数 (如果没有设置，提供默认值)
    server_lr = getattr(args, 'server_lr', 1.0)
    robustLR_threshold = getattr(args, 'robustLR_threshold', 0)
    aggr_method = getattr(args, 'aggr', 'comed') # 默认用 comed
    device = flat_global_params.device

    # 3. 计算 Robust LR (对应原代码的 compute_robustLR)
    lr_vector = torch.ones(n_params, device=device) * server_lr
    if robustLR_threshold > 0:
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        
        # 符号一致性低的地方，反向更新或设为极小值；一致性高的地方，正常更新
        lr_vector[sm_of_signs < robustLR_threshold] = -server_lr
        lr_vector[sm_of_signs >= robustLR_threshold] = server_lr

    # 4. 执行不同的聚合策略 (对应原代码的 agg_comed, agg_sign 等)
    if aggr_method == 'comed':
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        aggregated_updates = torch.median(concat_col_vectors, dim=1).values
        
    elif aggr_method == 'sign':
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_signs = torch.sign(sum(agent_updates_sign))
        aggregated_updates = torch.sign(sm_signs)
        
    else: # 默认 avg
        aggregated_updates = sum(agent_updates_dict.values()) / len(agent_updates_dict)

    # 5. 加噪机制 (对应原代码的 noise > 0)
    noise_level = getattr(args, 'noise', 0)
    if noise_level > 0:
        clip_val = getattr(args, 'clip', 1.0)
        noise = torch.normal(mean=0, std=noise_level * clip_val, size=(n_params,)).to(device)
        aggregated_updates.add_(noise)

    # 6. 将聚合后的更新应用到全局模型上
    new_global_params = (flat_global_params + lr_vector * aggregated_updates).float()

    # 7. 转换回 LASA 框架需要的 Dict 格式返回
    new_global_model = copy.deepcopy(global_model)
    new_global_model = vector_to_net_dict(new_global_params, new_global_model)

    return new_global_model