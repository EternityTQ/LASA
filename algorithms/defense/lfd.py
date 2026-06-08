import torch
import numpy as np
import sklearn.metrics.pairwise as smp
from sklearn.cluster import KMeans

def clusters_dissimilarity(clusters):
    """
    计算簇内差异度。越小的代表更新越紧凑，通常被认为是良性的。
    """
    n0 = len(clusters[0])
    n1 = len(clusters[1])
    m = n0 + n1 
    
    if n0 == 0: return 1.0, 0.0
    if n1 == 0: return 0.0, 1.0

    cs0 = smp.cosine_similarity(clusters[0]) - np.eye(n0)
    cs1 = smp.cosine_similarity(clusters[1]) - np.eye(n1)
    
    mincs0 = np.min(cs0, axis=1) if n0 > 1 else np.array([0.0])
    mincs1 = np.min(cs1, axis=1) if n1 > 1 else np.array([0.0])
    
    ds0 = n0 / m * (1 - np.mean(mincs0))
    ds1 = n1 / m * (1 - np.mean(mincs1))
    return ds0, ds1

def lfd(local_updates, global_model, args):
    """
    LASA 框架兼容的 LFighter (LFD) 防御算法 (增加 NaN 过滤机制)
    """
    # ---------------------------------------------------------
    # [新增防御机制] 数据清洗：过滤掉发生梯度爆炸 (NaN/Inf) 的异常更新
    # ---------------------------------------------------------
    valid_updates = []
    for update in local_updates:
        is_valid = True
        for key in update.keys():
            # 如果这层参数包含任何非数字或无穷大，直接判定为非法
            if torch.isnan(update[key]).any() or torch.isinf(update[key]).any():
                is_valid = False
                break
        if is_valid:
            valid_updates.append(update)
            
    # 如果极度巧合，所有客户端发来的更新都炸了，那就放弃这轮聚合，维持原模型不变
    if len(valid_updates) == 0:
        print("[LFD Warning] 本轮所有客户端更新均包含 NaN/Inf，跳过聚合。")
        return global_model
        
    local_updates = valid_updates
    m = len(local_updates)
    # ---------------------------------------------------------

    valid_keys = [k for k in local_updates[0].keys() if 'num_batches_tracked' not in k and 'running' not in k]
    weight_key = valid_keys[-2]
    bias_key = valid_keys[-1]

    dw = []
    db = []
    for i in range(m):
        dw.append(-local_updates[i][weight_key].cpu().numpy())
        db.append(-local_updates[i][bias_key].cpu().numpy())
        
    dw = np.asarray(dw)
    db = np.asarray(db)

    if len(db[0]) <= 2:
        data = [dw[i].reshape(-1) for i in range(m)]
    else:
        norms = np.linalg.norm(dw, axis=-1) 
        memory = np.sum(norms, axis=0) + np.sum(np.abs(db), axis=0)
        max_two_freq_classes = memory.argsort()[-2:]
        data = [dw[i][max_two_freq_classes].reshape(-1) for i in range(m)]

    if m >= 2:
        kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(data)
        labels = kmeans.labels_
    else:
        labels = [0] * m

    clusters = {0: [], 1: []}
    for i, l in enumerate(labels):
        clusters[l].append(data[i])

    good_cl = 0
    cs0, cs1 = clusters_dissimilarity(clusters)
    if cs0 < cs1:  
        good_cl = 1

    scores = np.ones([m])
    for i, l in enumerate(labels):
        if l != good_cl:
            scores[i] = 0
            
    good_indices = [i for i, s in enumerate(scores) if s == 1]
    
    if len(good_indices) == 0:
        good_indices = list(range(m))

    key_mean_weight = {}
    for key in local_updates[0].keys():
        if 'num_batches_tracked' in key:
            continue
        stacked_updates = torch.stack([local_updates[i][key] for i in good_indices], dim=0)
        key_mean_weight[key] = torch.mean(stacked_updates, dim=0)

    for key in key_mean_weight.keys():
        global_model[key].data += key_mean_weight[key].data

    return global_model