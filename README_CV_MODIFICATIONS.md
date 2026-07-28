## MOS 攻击效果改进 - 任务完成总结

### ✅ 已完成的修改

#### 1. CV 作为第三目标参与进化
- `obj_cv = torch.log1p(total_cv)` 
- `objectives: (3, P)` 包含 stealth, destructiveness, cv

#### 2. 关闭 DNC-aware mask（默认）
- `use_dnc_aware_mask = False`
- 保留代码，可重新启用

#### 3. 构造约束安全的 guidance (g_safe)
- `sign_safe_weight = 0.1`: 削弱违反符号的维度
- `subspace_repair_strength = 0.5`: 移除子空间投影
- 应用于：精英种子、directional push、objective evaluation

#### 4. Sign 约束归一化
- `layer_loss = violation_norm / reference_norm`
- 避免大层主导
- `sign_layer_quantile = 0.9` 可配置

#### 5. 每 10 代打印 CV 统计
- population min CV
- population mean CV
- selected individual CV
- feasible ratio

### ✅ 本地验收结果

```
✓ 语法检查通过
✓ 静态代码检查通过 (19/19 项)
✓ 函数调用一致性验证通过
✓ 关键修改点已实现
```

### ⚠️ 服务器验证（待用户执行）

以下指标需在真实联邦学习环境验证：

- [ ] CV 随进化下降（min, mean）
- [ ] feasible ratio 上升
- [ ] Sign ratio 提高
- [ ] Subspace ratio 提高
- [ ] 测试准确率和 ASR
- [ ] GPU 显存使用

### 📁 交付文件

- `algorithms/attack/mos.py` - 主修改文件
- `verify_mos_cv_modifications.py` - 静态检查脚本
- `MOS_CV_OBJECTIVE_MODIFICATIONS.md` - 详细文档
- `MOS_LOCAL_ACCEPTANCE_REPORT.txt` - 完整验收报告
- `README_CV_MODIFICATIONS.md` - 本文件

### 🚀 服务器运行步骤

1. 备份原代码
2. 复制修改后的 `mos.py` 到服务器
3. 运行实验（建议先小规模测试 30 代）
4. 观察日志中的 CV 统计

### 📊 预期效果

```
Generation 1:   min_cv=0.500, feasible=0.00%
Generation 10:  min_cv=0.350, feasible=5.00%
Generation 20:  min_cv=0.200, feasible=15.00%
Generation 30:  min_cv=0.120, feasible=25.00%
```

### 🔧 参数调优

如果 CV 下降不明显：
- 增加 `sign_safe_weight` (0.1 → 0.2)
- 增加 `subspace_repair_strength` (0.5 → 0.7)

如果攻击效果不足：
- 降低 `subspace_repair_strength` (0.5 → 0.3)

---

**本地验收结论**: ✅ 通过

已通过语法检查和静态代码检查。代码就绪，可部署到服务器。
