# Bug 修复记录（最终正确版本）

## 🐛 Bug #1: 维度不匹配错误

### 错误信息
```
RuntimeError: The size of tensor a (11183582) must match the size of tensor b (11173962)
```

### 问题根源（最终正确的分析）

#### 核心问题：`named_parameters()` vs `state_dict()`

在 PyTorch 中：
- `model.named_parameters()` 只返回 **parameters**（`requires_grad=True/False` 的张量）
- `model.state_dict()` 返回 **parameters + buffers**（包括 `num_batches_tracked` 等）

#### 具体问题

1. **在 `fedavg_all.py` 中**（第 224 行）：
   ```python
   model_update = {k: local_model[k] - global_model[k] for k in global_model.keys()}
   ```
   - `global_model.keys()` 来自 `state_dict()`
   - 所以 `model_update` **包含 buffers**（如 `num_batches_tracked`）

2. **在 `compute_surrogate_guidance` 中**（第 34-80 行）：
   ```python
   named = list(global_model.named_parameters())
   # ...
   for n, p in named:
       g_ce_list.append(p.grad.clone().flatten())
   ```
   - `named_parameters()` **不包含 buffers**
   - 所以 `g_ce` 和 `g_cw` 缺少 buffers 的维度

3. **结果**：
   - `benign_mean`（从 `local_updates` flatten）：维度 11183582（包含 buffers）
   - `g_ce/g_cw`（从 `named_parameters()` 提取）：维度 11173962（不包含 buffers）
   - **差值 9620** = buffers 的总维度（主要是 `num_batches_tracked`）

### 解决方案（最终正确版本）

**遍历 `state_dict()` 而不是 `named_parameters()`**，确保包含所有内容。

#### 修改：compute_surrogate_guidance（第 80-105 和 127-148 行）

```python
# 提取 g_ce：遍历 state_dict() 确保包含所有内容
g_ce_list = []
named_dict = dict(named)  # 将 named_parameters 转为字典

for name, tensor in global_model.state_dict().items():
    if name in named_dict:
        # 这是一个 parameter，提取其梯度
        param = named_dict[name]
        if param.grad is not None:
            g_ce_list.append(param.grad.clone().flatten())
        else:
            g_ce_list.append(torch.zeros(tensor.numel(), device=poison_images.device))
    else:
        # 这是一个 buffer（如 num_batches_tracked），用零填充
        g_ce_list.append(torch.zeros(tensor.numel(), device=poison_images.device))

g_ce = torch.cat(g_ce_list)
```

同样的逻辑应用于 `g_cw` 的提取。

### 验证

修复后：
- ✅ `benign_mean`: 11183582（state_dict 完整维度）
- ✅ `g_ce`: 11183582（state_dict 完整维度）
- ✅ `g_cw`: 11183582（state_dict 完整维度）
- ✅ 所有向量维度一致！

### 关键教训

#### 1. `named_parameters()` ≠ `state_dict()`

| 方法 | 包含内容 | 用途 |
|------|----------|------|
| `named_parameters()` | 只有 parameters | 计算梯度、优化 |
| `state_dict()` | parameters + buffers | 保存/加载模型、联邦聚合 |

#### 2. Buffers 的例子
- `BatchNorm.num_batches_tracked` (long 型，不参与梯度)
- `BatchNorm.running_mean` (requires_grad=False)
- `BatchNorm.running_var` (requires_grad=False)

#### 3. 在联邦学习中
`model_update` 通常基于 `state_dict()` 计算：
```python
model_update = {k: local[k] - global[k] for k in global.keys()}
```
所以在处理 `model_update` 时，必须考虑 buffers。

### 影响范围
- 影响所有使用 BatchNorm 的模型（ResNet, VGG, MobileNet 等）
- 影响所有使用 `state_dict()` 进行参数聚合的联邦学习框架

---

**修复时间**: 2026-07-02  
**修复者**: Claude (Opus 4.8)  
**状态**: ✅ 已修复（第 5 次，最终正确版本）  
**关键发现**: `named_parameters()` 和 `state_dict()` 的内容不同！  
**修改文件**: `algorithms/attack/mos.py`（2 处修改：g_ce 和 g_cw 提取）
