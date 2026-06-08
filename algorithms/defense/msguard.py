import torch
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import MeanShift

def msguard(all_updates, args=None, n_attackers=10):
    """
    MSGuard 防御机制：
    1. 接收所有客户端本轮的更新 (all_updates)
    2. 计算符号比例、余弦相似度、奇异值异常得分
    3. 利用 Mean Shift 聚类找出大多数“诚实者”
    4. 返回干净的聚合梯度
    """
    num_users = len(all_updates)
    
    # 1. 展平所有客户端的模型梯度
    all_updates_flatten = []
    for update in all_updates:
        # 这里就是报错的地方！如果 update 是字符串，就没有 update.keys()
        update_flat = torch.cat([torch.flatten(update[k]) for k in update.keys()])
        all_updates_flatten.append(update_flat)
    grads_matrix = torch.stack(all_updates_flatten)
    
    # 计算每个梯度的 L2 范数（大小）
    norms = torch.norm(grads_matrix, dim=1)
    med_norm = torch.median(norms)
    
    # 2. 提取特征 A：符号统计特征 (正、零、负的比例)
    total_dim = grads_matrix.shape[1]
    P_ratios = torch.sum(grads_matrix > 0, dim=1).float() / total_dim
    Z_ratios = torch.sum(grads_matrix == 0, dim=1).float() / total_dim
    N_ratios = torch.sum(grads_matrix < 0, dim=1).float() / total_dim
    
    # 3. 提取特征 B：余弦相似度得分
    cosine_scores = []
    for i in range(num_users):
        cos_sims = []
        for j in range(num_users):
            if i != j:
                sim = F.cosine_similarity(grads_matrix[i].unsqueeze(0), grads_matrix[j].unsqueeze(0))
                cos_sims.append(sim.item())
        cosine_scores.append(np.median(cos_sims))
    
    cosine_scores = torch.tensor(cosine_scores).to(grads_matrix.device)
    theta_cos = torch.median(cosine_scores)
    min_cos = torch.min(cosine_scores)
    
    # 标准化余弦得分到 [0, 0.2] 区间
    cos_scaled = torch.where(cosine_scores >= theta_cos, 
                             torch.tensor(0.2).to(grads_matrix.device), 
                             0.2 * (cosine_scores - min_cos) / (theta_cos - min_cos + 1e-9))
    
    # 4. 提取特征 C：基于 SVD 奇异值分解的异常得分
    mean_grad = torch.mean(grads_matrix, dim=0)
    centered_grads = grads_matrix - mean_grad
    
    # 奇异值分解，提取最重要的右奇异向量
    _, _, V = torch.svd(centered_grads)
    top_right_singular = V[:, 0]
    
    # 计算异常得分并标准化到 [0, 0.2]
    outlier_scores = torch.pow(torch.matmul(centered_grads, top_right_singular), 2)
    med_outlier = torch.median(outlier_scores)
    max_outlier = torch.max(outlier_scores)
    
    out_scaled = torch.where(outlier_scores <= med_outlier,
                             torch.tensor(0.0).to(grads_matrix.device),
                             0.2 * (outlier_scores - med_outlier) / (max_outlier - med_outlier + 1e-9))
    
    # 5. 特征融合与 Mean Shift 聚类
    features = torch.stack([P_ratios, Z_ratios, N_ratios, cos_scaled, out_scaled], dim=1).cpu().numpy()
    
    # 让算法自动抱团找组织
    clustering = MeanShift().fit(features)
    labels = clustering.labels_
    
    # 找出人数最多的那个“簇”，认定为诚实客户端
    counts = np.bincount(labels)
    largest_cluster_label = np.argmax(counts)
    trusted_indices = np.where(labels == largest_cluster_label)[0]
    
    # 6. 加权聚合
    trusted_grads = grads_matrix[trusted_indices]
    trusted_norms = norms[trusted_indices]
    
    # 计算权重：限制过大的梯度
    weights = torch.clamp(med_norm / (trusted_norms + 1e-9), max=1.0).unsqueeze(1)
    
    # 最终的安全聚合结果
    aggregate = torch.sum(trusted_grads * weights, dim=0) / len(trusted_indices)
    
    # 7. 转换回模型的字典格式，交还给主程序
    flattened = [torch.flatten(all_updates[0][k]) for k in all_updates[0].keys()]
    idx = []
    s = 0
    for p in flattened:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
    aggregate_model = {k: aggregate[s:d].reshape(all_updates[0][k].shape) for k, (s, d) in zip(all_updates[0].keys(), idx)}

    # 返回聚合后的安全更新，以及被判定为诚实的客户端编号
    return aggregate_model, trusted_indices