# 📁 梯度冲突验证系统 - 文件索引

## 🎯 根据你的需求快速查找文件

### 我想快速开始使用
→ 查看 **`quick_start.md`** (5分钟快速入门)  
→ 运行 **`python check_gradient_conflict_system.py`** (系统检查)  
→ 运行 **`python demo_gradient_conflict.py`** (30秒演示)

### 我想了解完整功能
→ 阅读 **`GRADIENT_CONFLICT_README.md`** (完整文档，包含技术细节)  
→ 查看 **`TASK_COMPLETION_REPORT.md`** (任务完成报告)

### 我想运行实验
→ 使用 **`test_gradient_conflict.py`** (完整测试脚本)
```bash
# 快速测试
python test_gradient_conflict.py --mode quick

# 微观分析
python test_gradient_conflict.py --mode microview

# 消融实验
python test_gradient_conflict.py --mode macroview

# 完整测试
python test_gradient_conflict.py --mode full
```

### 我想集成到现有代码
→ 查看 **`GRADIENT_CONFLICT_GUIDE.py`** (使用指南，第2部分)  
→ 主模块: **`algorithms/attack/gradient_conflict_analyzer.py`**

### 我想撰写论文
→ 查看 **`GRADIENT_CONFLICT_GUIDE.py`** (论文撰写建议)  
→ 查看 **`GRADIENT_CONFLICT_SUMMARY.md`** (论文填写模板)  
→ 查看 **`TASK_COMPLETION_REPORT.md`** (实验结果示例)

### 我遇到了问题
→ 查看 **`GRADIENT_CONFLICT_README.md`** 的 FAQ 部分  
→ 查看 **`GRADIENT_CONFLICT_GUIDE.py`** 的常见问题部分  
→ 运行 **`python check_gradient_conflict_system.py`** 诊断问题

---

## 📂 完整文件列表

### 核心代码文件

| 文件 | 描述 | 行数 | 用途 |
|------|------|------|------|
| `algorithms/attack/gradient_conflict_analyzer.py` | 主模块 | ~336 | 梯度冲突分析器和消融实验运行器 |
| `algorithms/attack/mos.py` | MOS攻击（已更新） | ~800 | 集成了梯度冲突分析功能 |

### 测试和演示文件

| 文件 | 描述 | 行数 | 用途 |
|------|------|------|------|
| `test_gradient_conflict.py` | 完整测试脚本 | ~284 | 运行完整的梯度冲突验证实验 |
| `demo_gradient_conflict.py` | 简化演示 | ~237 | 快速演示核心功能，无需完整框架 |
| `check_gradient_conflict_system.py` | 系统检查脚本 | ~165 | 检查环境配置和运行基本测试 |

### 文档文件

| 文件 | 描述 | 适合对象 | 阅读时间 |
|------|------|----------|----------|
| `quick_start.md` | 5分钟快速入门 | 所有用户 | 5分钟 |
| `GRADIENT_CONFLICT_README.md` | 完整文档 | 深度用户 | 15分钟 |
| `TASK_COMPLETION_REPORT.md` | 任务完成报告 | 项目管理者 | 10分钟 |
| `GRADIENT_CONFLICT_SUMMARY.md` | 任务总结 | 开发者 | 10分钟 |
| `GRADIENT_CONFLICT_GUIDE.py` | 使用指南和FAQ | 所有用户 | 10分钟 |
| `FILE_INDEX.md` | 本文件 | 所有用户 | 3分钟 |

---

## 🚀 推荐的使用流程

### 第一次使用（预计15分钟）

```
1. 阅读 quick_start.md (5分钟)
   ↓
2. 运行 python check_gradient_conflict_system.py (1分钟)
   ↓
3. 运行 python demo_gradient_conflict.py (30秒)
   ↓
4. 运行 python test_gradient_conflict.py --mode quick (1分钟)
   ↓
5. 查看 GRADIENT_CONFLICT_GUIDE.py 了解更多用法 (5分钟)
```

### 运行论文实验（预计1小时）

```
1. 准备数据集和环境
   ↓
2. 运行微观分析（15分钟）
   python test_gradient_conflict.py --mode microview \
       --dataset mnist --nsga_generations 100
   ↓
3. 运行宏观消融实验（30分钟）
   python test_gradient_conflict.py --mode macroview \
       --dataset mnist --nsga_generations 50
   ↓
4. 保存结果并填入论文（15分钟）
   - 复制余弦相似度矩阵
   - 复制 LaTeX 表格
   - 使用论文撰写模板
```

### 集成到现有项目（预计30分钟）

```
1. 阅读 GRADIENT_CONFLICT_GUIDE.py 的集成部分 (10分钟)
   ↓
2. 在你的代码中添加一行
   args.enable_conflict_analysis = True
   ↓
3. 运行你的训练脚本，查看冲突分析报告 (10分钟)
   ↓
4. 根据需要调整参数和输出格式 (10分钟)
```

---

## 📊 各文件的关系图

```
                    快速入门
                       ↓
            quick_start.md
                       ↓
        ┌──────────────┴──────────────┐
        ↓                              ↓
  检查环境                        运行演示
        ↓                              ↓
check_gradient_conflict_system.py  demo_gradient_conflict.py
        ↓                              ↓
        └──────────────┬──────────────┘
                       ↓
              运行完整测试
                       ↓
        test_gradient_conflict.py
                       ↓
        ┌──────────────┴──────────────┐
        ↓                              ↓
   微观分析                       宏观分析
        ↓                              ↓
 compute_objective_gradients    run_ablation_study
        ↓                              ↓
        └──────────────┬──────────────┘
                       ↓
          gradient_conflict_analyzer.py
                   (核心模块)
                       ↓
                   mos.py
              (可选集成)
```

---

## 🔍 按功能查找文件

### 梯度提取和余弦相似度计算
- **主实现**: `algorithms/attack/gradient_conflict_analyzer.py`
  - 类: `GradientConflictAnalyzer`
  - 方法: `compute_objective_gradients()`, `compute_cosine_similarity_matrix()`
- **测试**: `test_gradient_conflict.py` (--mode microview)
- **演示**: `demo_gradient_conflict.py` (demonstrate_gradient_conflict)

### 消融实验
- **主实现**: `algorithms/attack/gradient_conflict_analyzer.py`
  - 类: `AblationTestRunner`
  - 方法: `run_ablation_study()`
- **测试**: `test_gradient_conflict.py` (--mode macroview)
- **演示**: `demo_gradient_conflict.py` (demonstrate_simple_ablation)

### 冲突分析报告
- **主实现**: `algorithms/attack/gradient_conflict_analyzer.py`
  - 方法: `log_conflict_analysis()`, `analyze_conflicts()`
- **输出**: 终端打印（带彩色标注）

### MOS 攻击集成
- **修改文件**: `algorithms/attack/mos.py`
- **集成方式**: 通过 `args.enable_conflict_analysis` 参数
- **调用位置**: 第 282-293 行

---

## 📖 文档内容索引

### quick_start.md 包含：
- ✅ 3步开始使用
- ✅ 常用命令速查
- ✅ 结果解读表格
- ✅ 性能优化技巧
- ✅ 论文写作模板

### GRADIENT_CONFLICT_README.md 包含：
- ✅ 系统概述
- ✅ 核心功能介绍
- ✅ 快速开始指南
- ✅ 高级用法
- ✅ 技术细节
- ✅ 参数说明
- ✅ FAQ 常见问题

### TASK_COMPLETION_REPORT.md 包含：
- ✅ 任务完成情况
- ✅ 文件列表
- ✅ 使用方法
- ✅ 预期输出示例
- ✅ 论文填写建议
- ✅ 验证清单

### GRADIENT_CONFLICT_SUMMARY.md 包含：
- ✅ 详细的任务完成清单
- ✅ 论文撰写详细建议
- ✅ 实验结果分析模板
- ✅ 图表建议
- ✅ 讨论部分模板

### GRADIENT_CONFLICT_GUIDE.py 包含：
- ✅ 3种使用方式
- ✅ 预期输出示例
- ✅ 论文撰写建议
- ✅ FAQ 常见问题
- ✅ 示例代码

---

## 🎯 特定场景文件推荐

### 场景 1: 我是第一次使用，想快速了解
**推荐顺序**:
1. `quick_start.md` - 5分钟快速入门
2. `python check_gradient_conflict_system.py` - 系统检查
3. `python demo_gradient_conflict.py` - 30秒演示

### 场景 2: 我要写论文，需要实验结果
**推荐顺序**:
1. `TASK_COMPLETION_REPORT.md` - 查看预期输出示例
2. `python test_gradient_conflict.py --mode full` - 运行完整实验
3. `GRADIENT_CONFLICT_SUMMARY.md` - 查看论文模板
4. `GRADIENT_CONFLICT_GUIDE.py` - 查看撰写建议

### 场景 3: 我要集成到现有代码
**推荐顺序**:
1. `GRADIENT_CONFLICT_README.md` (高级用法部分)
2. `GRADIENT_CONFLICT_GUIDE.py` (使用方式2)
3. `algorithms/attack/gradient_conflict_analyzer.py` (查看API)

### 场景 4: 我遇到了错误
**推荐顺序**:
1. `python check_gradient_conflict_system.py` - 诊断问题
2. `GRADIENT_CONFLICT_README.md` (FAQ部分)
3. `GRADIENT_CONFLICT_GUIDE.py` (常见问题部分)

### 场景 5: 我想了解技术细节
**推荐顺序**:
1. `GRADIENT_CONFLICT_README.md` (技术细节部分)
2. `algorithms/attack/gradient_conflict_analyzer.py` (查看源码)
3. `GRADIENT_CONFLICT_SUMMARY.md` (实现细节)

---

## 💡 小贴士

### 快速查找命令

```bash
# 查看所有文档文件
ls *.md GRADIENT_CONFLICT_*.py

# 查看核心代码
ls algorithms/attack/gradient_conflict_analyzer.py

# 查看测试脚本
ls test_gradient_conflict.py demo_gradient_conflict.py

# 搜索特定内容（例如：余弦相似度）
grep -r "cosine_similarity" *.md *.py
```

### 打开文档的推荐方式

```bash
# Markdown 文件
cat quick_start.md | less

# Python 文件（带语法高亮，如果安装了 pygments）
python GRADIENT_CONFLICT_GUIDE.py

# 或者在浏览器中查看 Markdown
# 使用 VSCode、Typora 或 GitHub 预览
```

---

## 📞 获取帮助

如果你在使用过程中遇到任何问题：

1. **首先**: 运行 `python check_gradient_conflict_system.py` 检查环境
2. **然后**: 查看对应文档的 FAQ 部分
3. **最后**: 检查错误日志并根据提示修复

所有文件都包含详细的注释和错误处理，应该能够解决大部分问题。

---

**最后更新**: 2024-01-XX  
**版本**: v1.0.0  
**维护者**: 梯度冲突验证系统团队
