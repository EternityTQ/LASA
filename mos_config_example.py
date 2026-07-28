"""
MOS 攻击推荐参数配置示例
修复版本：v2.0 (Unified Constraints)
日期：2026-07-28

使用方法：
将这些参数添加到你的 args 对象中，或在配置文件中设置
"""

# ============================================================================
# 推荐默认配置（稳定起点）
# ============================================================================

class MOSConfigDefault:
    """推荐的默认配置 - 适用于大多数场景"""

    # --- 变异参数 ---
    mos_mutation_mode = "benign_std"        # 变异模式：benign_std（推荐）或 unit_norm
    mos_mutation_scale = 0.05               # benign_std 模式的尺度系数
    mos_mutation_radius_ratio = 0.01        # unit_norm 模式的半径比例（不使用时可忽略）
    mos_dir_step_ratio = 0.02               # 方向性推动步长比例

    # --- 约束阈值参数 ---
    constraint_quantile = 0.95              # 良性约束loss的分位数阈值
    constraint_score_temperature = 0.5      # 约束得分映射的温度参数（越小越陡）

    # --- 攻击预算参数 ---
    radius_quantile = 0.95                  # 径向距离的分位数阈值
    attack_budget_ratio = 1.0               # 攻击预算比例（相对于径向阈值）

    # --- 约束聚合参数 ---
    sign_layer_reduce = "quantile"          # Sign约束层间聚合：quantile（推荐）、max、mean
    subspace_reduce = "max"                 # Subspace约束聚合：max（推荐）、mean

    # --- DNC参数 ---
    use_dnc_aware_mask = True               # 是否启用DNC-aware mask（当DNC防御启用时）
    enable_subspace_constraint = True       # 是否启用子空间约束

    # --- 数值稳定性参数 ---
    logit_warning_threshold = 100.0         # Logit警告阈值

    # --- 进化参数 ---
    nsga_generations = 100                  # NSGA-II进化代数
    evo_pop_size = 10                       # 固定进化种群大小
    sbx_crossover_prob = 0.9                # SBX交叉概率
    sbx_eta = 15.0                          # SBX分布参数

    # --- 打分系统参数 ---
    score_mode = "smooth"                   # 约束得分映射模式：smooth（推荐）、relu、linear


# ============================================================================
# 保守配置（高隐蔽性优先）
# ============================================================================

class MOSConfigConservative:
    """保守配置 - 优先保证高隐蔽性，牺牲部分破坏力"""

    # 较小的变异尺度
    mos_mutation_mode = "benign_std"
    mos_mutation_scale = 0.01               # 更小的变异
    mos_dir_step_ratio = 0.01               # 更保守的方向推动

    # 更严格的约束阈值
    constraint_quantile = 0.90              # 更低的分位数（更严格）
    constraint_score_temperature = 0.25     # 更陡的评分曲线（惩罚更重）

    # 更小的攻击预算
    radius_quantile = 0.90
    attack_budget_ratio = 0.75              # 限制攻击范围

    # 其他与默认配置相同
    sign_layer_reduce = "quantile"
    subspace_reduce = "max"
    use_dnc_aware_mask = True
    enable_subspace_constraint = True
    logit_warning_threshold = 100.0
    nsga_generations = 100
    evo_pop_size = 10
    sbx_crossover_prob = 0.9
    sbx_eta = 15.0
    score_mode = "smooth"


# ============================================================================
# 激进配置（高破坏性优先）
# ============================================================================

class MOSConfigAggressive:
    """激进配置 - 优先破坏性，接受较低隐蔽性"""

    # 较大的变异尺度
    mos_mutation_mode = "benign_std"
    mos_mutation_scale = 0.1                # 更大的变异
    mos_dir_step_ratio = 0.05               # 更强的方向推动

    # 更宽松的约束阈值
    constraint_quantile = 0.98              # 更高的分位数（更宽松）
    constraint_score_temperature = 1.0      # 更平滑的评分曲线（惩罚较轻）

    # 更大的攻击预算
    radius_quantile = 0.98
    attack_budget_ratio = 1.5               # 扩大攻击范围

    # 其他与默认配置相同
    sign_layer_reduce = "quantile"
    subspace_reduce = "max"
    use_dnc_aware_mask = True
    enable_subspace_constraint = True
    logit_warning_threshold = 100.0
    nsga_generations = 100
    evo_pop_size = 10
    sbx_crossover_prob = 0.9
    sbx_eta = 15.0
    score_mode = "smooth"


# ============================================================================
# 快速测试配置
# ============================================================================

class MOSConfigQuickTest:
    """快速测试配置 - 用于调试和初步验证"""

    # 使用默认的变异和约束参数
    mos_mutation_mode = "benign_std"
    mos_mutation_scale = 0.05
    mos_dir_step_ratio = 0.02
    constraint_quantile = 0.95
    constraint_score_temperature = 0.5
    radius_quantile = 0.95
    attack_budget_ratio = 1.0
    sign_layer_reduce = "quantile"
    subspace_reduce = "max"
    use_dnc_aware_mask = True
    enable_subspace_constraint = True
    logit_warning_threshold = 100.0
    score_mode = "smooth"

    # 减少进化代数和种群大小以加速
    nsga_generations = 20                   # 大幅减少代数
    evo_pop_size = 5                        # 减少种群大小
    sbx_crossover_prob = 0.9
    sbx_eta = 15.0


# ============================================================================
# 实验扫描配置生成器
# ============================================================================

def generate_experiment_configs():
    """
    生成建议的实验配置矩阵

    返回配置字典列表，每个字典包含一组参数
    """
    configs = []

    # 基础配置
    base_config = {
        'mos_mutation_mode': 'benign_std',
        'mos_dir_step_ratio': 0.02,
        'constraint_quantile': 0.95,
        'radius_quantile': 0.95,
        'sign_layer_reduce': 'quantile',
        'subspace_reduce': 'max',
        'use_dnc_aware_mask': True,
        'enable_subspace_constraint': True,
        'logit_warning_threshold': 100.0,
        'nsga_generations': 100,
        'evo_pop_size': 10,
        'score_mode': 'smooth',
    }

    # 变量1：attack_budget_ratio
    budget_ratios = [0.75, 1.0, 1.25, 1.5]

    # 变量2：constraint_score_temperature
    temperatures = [0.25, 0.5, 1.0]

    # 变量3：mos_mutation_scale
    mutation_scales = [0.01, 0.05, 0.1]

    # 生成所有组合
    for budget in budget_ratios:
        for temp in temperatures:
            for scale in mutation_scales:
                config = base_config.copy()
                config['attack_budget_ratio'] = budget
                config['constraint_score_temperature'] = temp
                config['mos_mutation_scale'] = scale
                config['name'] = f"budget{budget}_temp{temp}_scale{scale}"
                configs.append(config)

    return configs


# ============================================================================
# 使用示例
# ============================================================================

def apply_config_to_args(args, config_class):
    """
    将配置类应用到 args 对象

    Args:
        args: 训练参数对象
        config_class: 配置类（例如 MOSConfigDefault）
    """
    for key, value in config_class.__dict__.items():
        if not key.startswith('_'):
            setattr(args, key, value)
    return args


# 使用示例：
if __name__ == "__main__":
    import argparse

    # 创建 args 对象
    args = argparse.Namespace()

    # 应用默认配置
    args = apply_config_to_args(args, MOSConfigDefault)

    print("默认配置已应用：")
    print(f"  变异模式: {args.mos_mutation_mode}")
    print(f"  变异尺度: {args.mos_mutation_scale}")
    print(f"  攻击预算比例: {args.attack_budget_ratio}")
    print(f"  约束温度: {args.constraint_score_temperature}")

    print("\n生成实验配置...")
    configs = generate_experiment_configs()
    print(f"共生成 {len(configs)} 组实验配置")
    print(f"示例配置 1: {configs[0]['name']}")
    print(f"  - attack_budget_ratio: {configs[0]['attack_budget_ratio']}")
    print(f"  - constraint_score_temperature: {configs[0]['constraint_score_temperature']}")
    print(f"  - mos_mutation_scale: {configs[0]['mos_mutation_scale']}")
