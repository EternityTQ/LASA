#!/usr/bin/env python3
"""
修复 mos.py 的结构问题：
1. 将 crowding_distance 和 nsga2_select 函数移到文件开头（模块级）
2. 将进化循环体代码正确地放在 for it in range(generations): 内部
3. 删除重复的代码
"""

import re

def fix_mos_structure():
    file_path = 'algorithms/attack/mos.py'

    print(f"[STEP 1] 读取文件...")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"[STEP 2] 分析文件结构...")
    print(f"  总行数: {len(lines)}")

    # 找到关键位置
    for_loop_line = None
    crowding_def_line = None
    nsga2_def_line = None
    compute_objectives_def_line = None
    nondominated_def_line = None
    mos_attack_def_line = None

    for i, line in enumerate(lines):
        if 'for it in range(generations):' in line:
            for_loop_line = i
            print(f"  找到 for 循环: 行 {i+1}")
        if line.strip().startswith('def crowding_distance('):
            crowding_def_line = i
            print(f"  找到 crowding_distance: 行 {i+1}")
        if line.strip().startswith('def nsga2_select('):
            nsga2_def_line = i
            print(f"  找到 nsga2_select: 行 {i+1}")
        if line.strip().startswith('def compute_objectives('):
            compute_objectives_def_line = i
            print(f"  找到 compute_objectives: 行 {i+1}")
        if line.strip().startswith('def nondominated_sort('):
            nondominated_def_line = i
            print(f"  找到 nondominated_sort: 行 {i+1}")
        if line.strip().startswith('def mos_attack('):
            mos_attack_def_line = i
            print(f"  找到 mos_attack: 行 {i+1}")

    if for_loop_line and crowding_def_line and for_loop_line < crowding_def_line:
        print(f"\n[问题确认] for 循环(行{for_loop_line+1})后面紧跟函数定义(行{crowding_def_line+1})")
        print(f"  需要：将函数定义移到模块级，补充for循环体")

    print(f"\n[STEP 3] 构建修复方案...")
    print(f"  方案：从备份文件读取正确的进化循环体代码")
    print(f"  由于结构太混乱，建议手动从模板重建关键部分")

    return lines, {
        'for_loop': for_loop_line,
        'crowding': crowding_def_line,
        'nsga2': nsga2_def_line,
        'compute_objectives': compute_objectives_def_line,
        'nondominated': nondominated_def_line,
        'mos_attack': mos_attack_def_line
    }

if __name__ == '__main__':
    lines, positions = fix_mos_structure()

    print(f"\n" + "="*60)
    print("诊断报告")
    print("="*60)
    print(f"结论：文件结构混乱，函数定义和代码块位置错乱")
    print(f"\n建议：")
    print(f"1. 备份当前文件")
    print(f"2. 从 Git 恢复到重构前的版本")
    print(f"3. 重新按正确顺序应用修改")
    print(f"\n或者：")
    print(f"手动调整第{positions['for_loop']+1}行附近的代码结构")
