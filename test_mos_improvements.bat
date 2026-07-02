@echo off
REM MOS 攻击改进测试脚本
REM 用于验证 Priority 1 修改的效果

echo ========================================
echo MOS Attack Priority 1 Improvements Test
echo ========================================
echo.

REM 测试 1: 基础配置（只测试覆盖率提升）
echo [Test 1] 基础配置 - 测试梯度覆盖率提升
echo 配置: loss_mask=11000 (只有 CE + CW + 幅度)
echo.
python main.py --dataset cifar --attack mos_attack --defend1 rlr --defend2 tr_mean --defend3 dnc --num_attackers 25 --gpu 4 --loss_mask 11000 --mos_conv_sparsity 0.3

echo.
echo ========================================
echo.

REM 测试 2: 添加 DNC-aware 掩码
echo [Test 2] DNC-aware 掩码测试
echo 配置: loss_mask=11000 + DNC-aware mask
echo.
python main.py --dataset cifar --attack mos_attack --defend1 rlr --defend2 tr_mean --defend3 dnc --num_attackers 25 --gpu 4 --loss_mask 11000 --mos_conv_sparsity 0.3 --use_dnc_aware_mask 1

echo.
echo ========================================
echo.

REM 测试 3: 添加 PCA 约束
echo [Test 3] PCA 约束测试
echo 配置: loss_mask=1100100 (添加 PCA)
echo.
python main.py --dataset cifar --attack mos_attack --defend1 rlr --defend2 tr_mean --defend3 dnc --num_attackers 25 --gpu 4 --loss_mask 1100100 --mos_conv_sparsity 0.3 --use_dnc_aware_mask 1

echo.
echo ========================================
echo.

REM 测试 4: 完整配置（所有改进）
echo [Test 4] 完整配置 - 所有 Priority 1 改进
echo 配置: loss_mask=1100110 (CE + CW + 幅度 + PCA + 子空间)
echo.
python main.py --dataset cifar --attack mos_attack --defend1 rlr --defend2 tr_mean --defend3 dnc --num_attackers 25 --gpu 4 --loss_mask 1100110 --mos_conv_sparsity 0.3 --use_dnc_aware_mask 1 --enable_subspace_constraint 1

echo.
echo ========================================
echo 所有测试完成！
echo ========================================
pause
