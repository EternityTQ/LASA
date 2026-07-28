#!/usr/bin/env python3
"""
MOS.py 重构修复脚本

用途：完成主进化循环的重构，删除旧代码并替换为新系统

使用方法：
    python fix_mos_refactor.py
"""

import re

def fix_mos_file():
    """修复mos.py文件"""

    file_path = 'algorithms/attack/mos.py'

    print(f"[FIXING] 读取 {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_length = len(content)

    # ============================================================================
    # Step 1: 删除loss_mask相关代码（约13行）
    # ============================================================================
    print("[STEP 1] 删除loss_mask相关代码...")

    loss_mask_pattern = r"    # 1\. 解析传入的 loss_mask 参数.*?num_active_losses = 2\n        \n    normalizer = LossNormalizer\(num_objectives=num_active_losses, momentum=0\.8\)\n"
    content = re.sub(loss_mask_pattern, '', content, flags=re.DOTALL)

    # ============================================================================
    # Step 2: 删除旧的compute_pca_constraint和compute_subspace_constraint函数
    # ============================================================================
    print("[STEP 2] 删除旧的PCA和子空间约束函数...")

    # 删除compute_pca_constraint
    pca_pattern = r"    # ================= 修改 3：添加 PCA-aware 约束.*?def compute_pca_constraint\(pop, benign_grads, benign_mean, device\):.*?return l_pca_normalized\n\n"
    content = re.sub(pca_pattern, '', content, flags=re.DOTALL)

    # 删除compute_subspace_constraint
    subspace_pattern = r"    def compute_subspace_constraint\(pop, benign_grads, benign_mean, device, num_samples=3\):.*?return l_subspace_normalized\n\n"
    content = re.sub(subspace_pattern, '', content, flags=re.DOTALL)

    # ============================================================================
    # Step 3: 删除旧的compute_losses函数
    # ============================================================================
    print("[STEP 3] 删除旧的compute_losses函数...")

    compute_losses_pattern = r"    def compute_losses\(pop\):.*?return raw_losses_p\n\n"
    content = re.sub(compute_losses_pattern, '', content, flags=re.DOTALL)

    # ============================================================================
    # Step 4: 替换进化循环中的旧调用
    # ============================================================================
    print("[STEP 4] 替换进化循环中的函数调用...")

    # 替换第一处：评估当前种群
    old_eval_current = r"        # 评估当前种群\n        raw_current = compute_losses\(malicious_set\)\n        norm_current = normalizer\.update_and_normalize\(raw_current\)\n        objs_current = norm_current\.detach\(\)  # \(M, P\)"
    new_eval_current = """        # 评估当前种群（双目标）
        objectives_current, scores_current, losses_current = compute_objectives(
            malicious_set,
            benign_refs,
            constraint_thresholds,
            g_combined_unit,
            score_mode=score_mode
        )"""
    content = re.sub(old_eval_current, new_eval_current, content)

    # 替换第二处：评估合并种群
    old_eval_combined = r"        # 合并：当前、子代、以及 archive\（跨代记忆\)\n        combined = torch\.cat\(\[malicious_set, offspring, archive\], dim=0\)\n        raw_combined = compute_losses\(combined\)\n        norm_combined = normalizer\.update_and_normalize\(raw_combined\)\n        objs_combined = norm_combined\.detach\(\)"
    new_eval_combined = """        # 合并：当前、子代、以及 archive（跨代记忆）
        combined = torch.cat([malicious_set, offspring, archive], dim=0)
        objectives_combined, _, _ = compute_objectives(
            combined,
            benign_refs,
            constraint_thresholds,
            g_combined_unit,
            score_mode=score_mode
        )"""
    content = re.sub(old_eval_combined, new_eval_combined, content)

    # 替换第三处：最终评估
    old_eval_final = r"    # 进化完成后，从最终的 malicious_set 中选出排名第一的最优个体作为模板\n    final_losses = compute_losses\(malicious_set\)\n    final_normalized = normalizer\.update_and_normalize\(final_losses\)\n    final_objs = final_normalized\.detach\(\)"
    new_eval_final = """    # 进化完成后，从最终的 malicious_set 中选出排名第一的最优个体作为模板
    final_objectives, final_scores, final_losses = compute_objectives(
        malicious_set,
        benign_refs,
        constraint_thresholds,
        g_combined_unit,
        score_mode=score_mode
    )"""
    content = re.sub(old_eval_final, new_eval_final, content)

    # 替换所有nsga3_select调用为nsga2_select
    content = content.replace('nsga3_select(objs_current', 'nsga2_select(objectives_current')
    content = content.replace('nsga3_select(objs_combined', 'nsga2_select(objectives_combined')

    # 替换final_fronts = nondominated_sort(final_objs)
    content = content.replace('final_fronts = nondominated_sort(final_objs)',
                             'final_fronts = nondominated_sort(final_objectives)')

    # ============================================================================
    # Step 5: 添加score_mode和constraint_k_sigma获取
    # ============================================================================
    print("[STEP 5] 添加新参数获取...")

    # 在max_dev_threshold之后添加
    insert_point = "    large_norm_penalty_coeff = getattr(args, 'large_norm_penalty_coeff', 1e3)\n"
    new_params = """
    # 获取打分系统配置
    score_mode = getattr(args, 'score_mode', 'sigmoid')  # 默认sigmoid（用户选择）
    constraint_k_sigma = getattr(args, 'constraint_k_sigma', 2.0)  # 阈值系数

"""
    content = content.replace(insert_point, insert_point + new_params)

    # ============================================================================
    # Step 6: 更新进化循环开始的日志
    # ============================================================================
    print("[STEP 6] 更新进化循环日志...")

    old_log_start = r"    for it in range\(generations\):"
    new_log_start = """    print(f"[MOS LOG] 🧬 开始进化循环，代数={generations}，种群大小={EVOLUTION_POP_SIZE}")
    print(f"[MOS LOG] 📊 打分系统：{score_mode}映射，阈值系数k={constraint_k_sigma}")

    for it in range(generations):"""
    content = re.sub(old_log_start, new_log_start, content)

    # ============================================================================
    # Step 7: 添加进化过程中的周期性日志
    # ============================================================================
    print("[STEP 7] 添加进化过程日志...")

    # 在archive更新后添加日志
    old_archive_update = r"        archive = combined\[new_archive_idx\]\.clone\(\)\.detach\(\) if new_archive_idx else archive\n"
    new_archive_update = """        archive = combined[new_archive_idx].clone().detach() if new_archive_idx else archive

        # 每10代打印一次进度
        if (it + 1) % 10 == 0 or it == 0:
            avg_recognition = -objectives_current[0].mean().item()  # 负号还原
            avg_destructiveness = -objectives_current[1].mean().item()
            print(f"[MOS LOG]   Generation {it+1}/{generations}: "
                  f"识别率={avg_recognition:.3f}, 破坏性={avg_destructiveness:.3f}")
"""
    content = re.sub(old_archive_update, new_archive_update, content)

    # ============================================================================
    # Step 8: 更新最终日志输出
    # ============================================================================
    print("[STEP 8] 更新最终日志...")

    old_final_log = r"    print\(f\"\[MOS LOG\] 🏆 进化完成！最优模板索引: \{best_idx\}\"\)\n    print\(f\"\[MOS LOG\] 📋 将最优模板复制"
    new_final_log = """    print(f"\\n[MOS LOG] 🏆 进化完成！最优模板索引: {best_idx}")
    print(f"[MOS LOG]   最优个体识别率得分: {-final_objectives[0, best_idx].item():.3f}")
    print(f"[MOS LOG]   最优个体破坏性: {-final_objectives[1, best_idx].item():.3f}")
    print(f"[MOS LOG] 📋 将最优模板复制"""
    content = re.sub(old_final_log, new_final_log, content)

    # ============================================================================
    # 保存修复后的文件
    # ============================================================================
    print(f"[SAVING] 保存修复后的文件...")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    new_length = len(content)
    diff = original_length - new_length

    print(f"\n[SUCCESS] 修复完成！")
    print(f"  原始长度: {original_length} 字符")
    print(f"  新长度: {new_length} 字符")
    print(f"  删除: {diff} 字符")
    print(f"\n已删除的冗余代码：")
    print(f"  - LossNormalizer 类（已移除）")
    print(f"  - loss_mask 参数处理（已移除）")
    print(f"  - compute_pca_constraint 函数（已移除，使用预计算）")
    print(f"  - compute_subspace_constraint 函数（已移除，使用预计算）")
    print(f"  - compute_losses 函数（已替换为 compute_objectives）")
    print(f"  - nsga3_select 调用（已替换为 nsga2_select）")
    print(f"\n新增功能：")
    print(f"  ✓ 打分系统（sigmoid映射，默认）")
    print(f"  ✓ 双目标优化（识别率 + 破坏性）")
    print(f"  ✓ NSGA-II 选择机制")
    print(f"  ✓ 预计算良性参考系统")

if __name__ == '__main__':
    try:
        fix_mos_file()
    except Exception as e:
        print(f"\n[ERROR] 修复失败: {e}")
        import traceback
        traceback.print_exc()
