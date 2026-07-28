# MOS-Attack 运行时修复记录

## 问题描述

**错误信息：**
```
TypeError: zeros() received an invalid combination of arguments - got (int, device=Config)
```

**错误位置：**
```python
File "algorithms/attack/mos.py", line 197, in compute_surrogate_guidance
    g_ce = extract_gradient_vector(global_model)
File "algorithms/attack/mos.py", line 180, in extract_gradient_vector
    g_list.append(torch.zeros(tensor.numel(), device=device))
```

## 根本原因

**初始实现错误：**
```python
# 新版（错误）
def compute_surrogate_guidance(
    global_model, poison_images, target_labels, criterion_ce,
    device: torch.device  # ❌ 错误：实际调用传入的是 args
):
    # device 被当作 torch.device 使用
```

**实际调用方式（from fedavg_all.py）：**
```python
g_ce, g_cw = compute_surrogate_guidance_mos(
    safe_net, images, target_labels, criterion_ce, args  # 传入的是 args
)
```

**旧版正确实现：**
```python
def compute_surrogate_guidance(
    global_model, poison_images, target_labels, criterion_ce, 
    args=None  # ✅ 正确：接收 args 参数
):
    device = poison_images.device  # ✅ 从数据中提取设备
```

## 修复方案

### 修复内容
```python
# 修复后的签名（与旧版一致）
def compute_surrogate_guidance(
    global_model: torch.nn.Module,
    poison_images: torch.Tensor,
    target_labels: torch.Tensor,
    criterion_ce,
    args=None  # ✅ 改回 args 参数
) -> Tuple[torch.Tensor, torch.Tensor]:
    global_model.eval()
    device = poison_images.device  # ✅ 从 poison_images 提取设备
    
    def extract_gradient_vector(model):
        # device 现在可以正确使用
        ...
```

### 修改文件
- **文件：** `d:\LASA\algorithms\attack\mos.py`
- **修改行：** 150-165
- **修改类型：** 函数签名修正

## 验证结果

### 1. 语法检查
```bash
✅ python -m py_compile mos.py
   无语法错误
```

### 2. 接口一致性
```python
旧版：compute_surrogate_guidance(..., args=None)
新版：compute_surrogate_guidance(..., args=None)
状态：✅ 一致
```

### 3. 设备提取逻辑
```python
旧版：device = poison_images.device
新版：device = poison_images.device
状态：✅ 一致
```

## 经验教训

### 问题分析
1. **错误原因：** 在重构时错误地改变了函数签名
2. **未被发现原因：** 静态测试使用的是模拟参数，未触发实际调用路径
3. **暴露时机：** 真实训练时框架调用该函数

### 改进措施
1. **保持签名一致性：** 重构时必须严格保持所有公开函数的签名
2. **设备提取原则：** 
   - ✅ 从数据张量提取：`device = tensor.device`
   - ❌ 从 args 提取：`device = args.device`（args 可能不是 torch.device）
3. **集成测试必要性：** 静态测试无法完全覆盖框架集成问题

## 后续建议

### 1. 验证其他函数签名
检查是否还有其他函数签名不一致的情况：

```bash
# 对比所有公开函数
diff -u \
  <(grep "^def " algorithms/attack/mos_experimental.py) \
  <(grep "^def " algorithms/attack/mos.py)
```

### 2. 集成测试
```python
# 建议添加集成测试
def test_compute_surrogate_guidance_integration():
    """测试与框架的实际集成"""
    model = get_test_model()
    images = torch.randn(10, 3, 32, 32)
    labels = torch.randint(0, 10, (10,))
    criterion = nn.CrossEntropyLoss()
    
    # 模拟框架调用方式
    class MockArgs:
        pass
    
    args = MockArgs()
    g_ce, g_cw = compute_surrogate_guidance(
        model, images, labels, criterion, args  # 实际调用方式
    )
    
    assert g_ce.shape[0] > 0
    assert g_cw.shape[0] > 0
```

### 3. 文档更新
更新函数签名文档，明确参数类型：

```python
def compute_surrogate_guidance(
    global_model: torch.nn.Module,
    poison_images: torch.Tensor,
    target_labels: torch.Tensor,
    criterion_ce,
    args=None  # Config object (not used, kept for compatibility)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute CE and CW surrogate gradients
    
    Note: args parameter is kept for API compatibility but not used.
    Device is extracted from poison_images.device instead.
    """
```

## 修复状态

- ✅ 问题已识别
- ✅ 根本原因已定位
- ✅ 修复已实施
- ✅ 语法验证通过
- ⚠️ 需要真实训练验证

## 相关文件

- `algorithms/attack/mos.py` - 已修复
- `algorithms/attack/mos_experimental.py` - 参考版本
- `algorithms/engine/fedavg_all.py` - 调用方

---

**修复时间：** 2026-07-28  
**问题类型：** 函数签名不一致  
**严重程度：** 高（阻止运行）  
**修复状态：** 已完成，待验证
