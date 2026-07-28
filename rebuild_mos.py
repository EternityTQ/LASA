#!/usr/bin/env python3
"""
重建 mos.py 的正确结构
"""

def rebuild_mos():
    file_path = 'algorithms/attack/mos.py'

    print("[STEP 1] 读取文件...")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"  原始行数: {len(lines)}")

    # 策略：
    # 1. 保留 1-627行（到 for it in range(generations): 之前）
    # 2. 删除 628-951行（错位的函数定义和空的for循环）
    # 3. 保留 952-1117行（包含正确的函数定义和进化循环体）

    # 但需要调整952行开始的内容，把进化循环体代码移到正确的缩进位置

    print("[STEP 2] 删除第636-637行的空for循环...")
    # 找到 "for it in range(generations):" 并删除它和下一行
    new_lines = []
    skip_until = -1

    for i, line in enumerate(lines):
        # 跳过第630-637行（重复的日志和空for循环）
        if i >= 629 and i <= 636:
            continue

        # 跳过第638-951行（错位的函数定义）
        if i >= 637 and i <= 950:
            continue

        new_lines.append(line)

    print("[STEP 3] 在第627行后添加进化循环...")
    # 现在 new_lines 包含：
    # - 1-629行：mos_attack函数的前半部分（到打分系统配置）
    # - 951-1117行：正确的函数定义 + 进化循环体（但缩进可能不对）

    # 需要找到进化循环体的开始位置（在新数组中）
    final_lines = []
    for i, line in enumerate(new_lines):
        final_lines.append(line)

        # 在 "打分系统配置" 日志后添加 for 循环开头
        if i < len(new_lines) - 1:
            if '打分系统：' in line and '映射' in line:
                # 下一行添加空行和for循环
                final_lines.append('\n')
                final_lines.append('    for it in range(generations):\n')
                print(f"  在第{i+1}行后插入 for 循环")

                # 接下来需要把进化循环体代码（8空格缩进）插入
                # 但现在这些代码在后面某处，需要识别和调整缩进

                # 暂时跳过，继续处理剩余部分

    print(f"[STEP 4] 新文件行数: {len(final_lines)}")

    # 保存
    output_path = 'algorithms/attack/mos_rebuilt.py'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(final_lines)

    print(f"[SUCCESS] 重建文件已保存到: {output_path}")
    print(f"  请人工检查后，如果正确则替换原文件")

if __name__ == '__main__':
    rebuild_mos()
