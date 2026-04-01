#!/bin/bash

# ================= 配置区 =================
DATASET="cifar"         # 使用的数据集
ATTACK_RATIO=20         # 攻击者比例 20%
REPEAT_TIMES=1         # 每个组合重复跑 5 次

# 定义要跑的攻击和防御算法池
ATTACKS=("non_attack" "lie_attack" "agrAgnosticMinMax" "agrAgnosticMinSum")
DEFENSES=("fedavg" "multi_krum" "dnc" "rlr" "lasa")

# 创建一个专门存放终端日志的文件夹，防止主目录太乱
mkdir -p experiment_logs

# ================= 运行区 =================
echo "开始批量运行联邦学习对比实验..."

# 三层循环：攻击 -> 防御 -> 重复次数(Seed)
for atk in "${ATTACKS[@]}"; do
    for def in "${DEFENSES[@]}"; do
        for (( seed=1; seed<=REPEAT_TIMES; seed++ )); do
            
            # 定义当前日志文件的名字，例如：non_attack_fedavg_seed1.log
            LOG_FILE="experiment_logs/${atk}_${def}_seed${seed}.log"
            
            echo "正在运行: 攻击=[$atk], 防御=[$def], 第 $seed 次实验..."
            echo "日志将追加保存在: $LOG_FILE"
            
            # ---------------- 新增：写入时间戳和分隔符 ----------------
            # 获取当前时间，格式为 YYYY-MM-DD HH:MM:SS
            CURRENT_TIME=$(date '+%Y-%m-%d %H:%M:%S')
            
            # 将时间戳和当前实验配置追加写入日志（>> 表示追加）
            echo -e "\n===================================================" >> "$LOG_FILE"
            echo "实验启动时间: $CURRENT_TIME" >> "$LOG_FILE"
            echo "当前参数配置: 攻击=[$atk], 防御=[$def], 随机种子=[$seed]" >> "$LOG_FILE"
            echo "===================================================" >> "$LOG_FILE"
            # ----------------------------------------------------------
            
            # 执行 Python 命令。串行排队执行，防止把显存一次性撑爆
            # 注意：把 > 改成了 >>，实现追加写入；2>&1 保持不变，确保报错信息也写进日志
            python main.py \
                --dataset $DATASET \
                --num_attackers $ATTACK_RATIO \
                --attack $atk \
                --defend $def \
                --seed $seed \
                --gpu 2 \
                --repeat 5 \
                >> "$LOG_FILE" 2>&1
                
            echo "完成: 攻击=[$atk], 防御=[$def], 第 $seed 次实验。"
            echo "---------------------------------------------------"
            
        done
    done
done

echo "所有实验已全部跑完，辛苦了！"