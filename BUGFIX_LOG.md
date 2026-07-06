# Bug 修复记录

## 🐛 Bug #1: 维度不匹配错误（最终正确修复）

### 错误信息
```
RuntimeError: The size of tensor a (11183582) must match the size of tensor b (11173962) at non-singleton dimension 0
```

### 问题根源（真正的原因）

**之前的错误分析**：我们以为应该跳过 `num_batches_tracked`，因为它不参与梯度计算。

**真正的问题**：
1. 在 `fedavg_all.py` 中，`model_update` 是从 `global_model.keys()` 创建的：
   ```python
   model_update = {k: local_model[k] - global_model[k] for k in global_model.keys()}
   ```
   这意味着 `model_update` **包含 `num_batches_tracked`**（如果模型有 BatchNorm）

2. `local_updates` 是 `model_update` 的列表，所以它们**也包含 `num_batches_tracked`**

3. 但在 `compute_surrogate_guidance` 中，我们使用 `named_parameters()` 提取梯度：
   ```python
   for n, p in named:
       if p.grad is not None:
           g_ce_list.append(p.grad.clone().flatten())
   ```
   
   **关键**：`num_batches_tracked` **不在 `named_parameters()` 中**（因为它的 `requires_grad=False`），但它**在 `state_dict()` 中**！

4. 这导致：
   - `benign_mean`（从 `local_updates` flatten）：包含 `num_batches_tracked`
   - `g_ce/g_cw`（从 `named_parameters()` 提取）：**不包含 `num_batches_tracked`**

### 关键区别

| 方法 | 包含 num_batches_tracked? |
|------|---------------------------|
| `model.state_dict().keys()` | ✅ 是 |
| `model.named_parameters()` | ❌ 否（因为 requires_grad=False）|

### 正确的解决方案

**不要跳过 `num_batches_tracked`**，而是在提取 `g_ce/g_cw` 时，遍历 `state_dict()` 而不是 `named_parameters()`。

但更简单的方法是：**在提取时，对 `named_parameters()` 中没有的参数（如 `num_batches_tracked`），用零填充**。

### 修复代码

**文件**: `algorithms/attack/mos.py`

#### 修改：g_ce 和 g_cw 提取（第 80-125 行）

**原理**：遍历 `state_dict()` 而不是 `named_parameters()`，确保包含所有参数。

```python
# 提取稀疏化后的 g_ce（关键：必须包含所有 state_dict 中的参数）
g_ce_list = []
for n, p in named:
    if p.grad is not None:
        g_ce_list.append(p.grad.clone().flatten())
    else:
        # 对于没有梯度的参数（如 num_batches_tracked），用零填充
        g_ce_list.append(torch.zeros(p.numel(), device=poison_images.device))

# ⚠️ 关键：还需要添加不在 named_parameters 中但在 state_dict 中的参数
param_names = {n for n, _ in named}
for n, tensor in global_model.state_dict().items():
    if n not in param_names:
        # 这些是 buffers（如 num_batches_tracked），不是 parameters
        g_ce_list.append(torch.zeros(tensor.numel(), device=poison_images.device))

g_ce = torch.cat(g_ce_list) if g_ce_list else torch.zeros(0, device=poison_images.device)
```

**等等，这样太复杂了...**

### 更简单的解决方案

**直接遍历 `state_dict()` 而不是 `named_parameters()`**：

```python
# 提取稀疏化后的 g_ce
g_ce_list = []
for n, tensor in global_model.state_dict().items():
    param = dict(named).get(n)  # 尝试获取对应的 parameter
    if param is not None and param.grad is not None:
        g_ce_list.append(param.grad.clone().flatten())
    else:
        # 对于 buffers 或没有梯度的参数，用零填充
        g_ce_list.append(torch.zeros(tensor.numel(), device=poison_images.device))
g_ce = torch.cat(g_ce_list)
```

### 实际修复（已应用）

保持原有逻辑，但确保维度一致：
1. ✅ 移除所有 "跳过 num_batches_tracked" 的逻辑
2. ✅ 在 `g_ce/g_cw` 提取时，为所有参数（包括 buffers）分配空间

---

**修复时间**: 2026-07-02  
**修复者**: Claude (Opus 4.8)  
**状态**: ✅ 已修复（第 5 次，最终正确版本）  
**关键教训**: `named_parameters()` 和 `state_dict()` 返回的内容不同！
