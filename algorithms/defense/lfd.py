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
    
    # 防止因某个簇为空而导致除零报错
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
    LASA 框架兼容的 LFighter (LFD) 防御算法
    """
    m = len(local_updates)
    if m == 0:
        return global_model
    
    # 1. 自动定位最后的全连接层（分类器层）的 weight 和 bias
    # 我们过滤掉 'num_batches_tracked' 等 BatchNorm 追踪参数，只看核心权重
    valid_keys = [k for k in local_updates[0].keys() if 'num_batches_tracked' not in k and 'running' not in k]
    weight_key = valid_keys[-2]
    bias_key = valid_keys[-1]

    dw = []
    db = []
    for i in range(m):
        # 注意：LASA 框架传入的 local_updates 是“更新量 (delta)”
        # LFighter 原始论文的 dw 是 w_old - w_new = -delta，这里我们取负以对齐原版逻辑
        dw.append(-local_updates[i][weight_key].cpu().numpy())
        db.append(-local_updates[i][bias_key].cpu().numpy())
        
    dw = np.asarray(dw)
    db = np.asarray(db)

    # 2. 判断分类类别数量，分离特征
    if len(db[0]) <= 2:
        # 二分类或单分类模型直接全部展平
        data = [dw[i].reshape(-1) for i in range(m)]
    else:
        # 多分类模型：找出被更新影响最大的两个类（很可能是攻击者设定的源类和目标类）
        norms = np.linalg.norm(dw, axis=-1) 
        memory = np.sum(norms, axis=0) + np.sum(np.abs(db), axis=0)
        max_two_freq_classes = memory.argsort()[-2:]
        data = [dw[i][max_two_freq_classes].reshape(-1) for i in range(m)]

    # 3. 使用 K-Means 将客户端强行分为两拨
    if m >= 2:
        kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(data)
        labels = kmeans.labels_
    else:
        labels = [0] * m

    clusters = {0: [], 1: []}
    for i, l in enumerate(labels):
        clusters[l].append(data[i])

    # 4. 判断哪一拨是诚实节点
    good_cl = 0
    cs0, cs1 = clusters_dissimilarity(clusters)
    if cs0 < cs1:  # 差异度小的（更抱团的）被认为是诚实节点
        good_cl = 1

    # 5. 过滤掉被判定为恶意的客户端梯度
    scores = np.ones([m])
    for i, l in enumerate(labels):
        if l != good_cl:
            scores[i] = 0
            
    good_indices = [i for i, s in enumerate(scores) if s == 1]
    
    # 安全兜底：如果所有人都不幸被判定为一类或出错，退化为全部聚合（防止报错崩溃）
    if len(good_indices) == 0:
        good_indices = list(range(m))

    # 6. 基于良性节点的索引，进行最终的均值聚合
    key_mean_weight = {}
    for key in local_updates[0].keys():
        if 'num_batches_tracked' in key:
            continue
        stacked_updates = torch.stack([local_updates[i][key] for i in good_indices], dim=0)
        key_mean_weight[key] = torch.mean(stacked_updates, dim=0)

    # 7. 将筛选后的干净更新安全合并回全局模型
    for key in key_mean_weight.keys():
        global_model[key].data += key_mean_weight[key].data

    return global_model