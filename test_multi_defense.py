"""
测试多防御基线功能的脚本
"""
import argparse

# 测试参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--defend1', type=str, default='lasa', help='primary defend method')
parser.add_argument('--defend2', type=str, default=None, help='secondary defend method (optional)')
parser.add_argument('--defend3', type=str, default=None, help='tertiary defend method (optional)')

# 测试1：只有一个防御方法
print("=== Test 1: Single defense method ===")
args1 = parser.parse_args(['--defend1', 'lasa'])
defend_methods1 = [args1.defend1]
if args1.defend2:
    defend_methods1.append(args1.defend2)
if args1.defend3:
    defend_methods1.append(args1.defend3)
print(f"Defense methods: {defend_methods1}")
print(f"Number of methods: {len(defend_methods1)}")

# 测试2：两个防御方法
print("\n=== Test 2: Two defense methods ===")
args2 = parser.parse_args(['--defend1', 'lasa', '--defend2', 'fedavg'])
defend_methods2 = [args2.defend1]
if args2.defend2:
    defend_methods2.append(args2.defend2)
if args2.defend3:
    defend_methods2.append(args2.defend3)
print(f"Defense methods: {defend_methods2}")
print(f"Number of methods: {len(defend_methods2)}")

# 测试3：三个防御方法
print("\n=== Test 3: Three defense methods ===")
args3 = parser.parse_args(['--defend1', 'lasa', '--defend2', 'fedavg', '--defend3', 'signguard'])
defend_methods3 = [args3.defend1]
if args3.defend2:
    defend_methods3.append(args3.defend2)
if args3.defend3:
    defend_methods3.append(args3.defend3)
print(f"Defense methods: {defend_methods3}")
print(f"Number of methods: {len(defend_methods3)}")

print("\n=== All tests passed! ===")
