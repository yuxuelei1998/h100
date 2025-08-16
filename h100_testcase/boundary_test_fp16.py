import numpy as np

# 操作符定义
OP_DEF = {
    "ADD": 0x10,
    "SUB": 0x12,
    "MUL": 0x14,
    "MPDPA": 0x16,
    "CMPEQ": 0x22,
    "CMPLT": 0x24,
    "CMPLE": 0x26,
    "CMPGT": 0x28,
    "CMPLTNUM": 0x34,
    "CMPLENUM": 0x36,
    "CMPGTNUM": 0x38,
    "UNORDERED": 0x2a,
    "FMA": 0x40,
    "FMS": 0x42,
    "FNMA": 0x44,
    "FNMS": 0x46,
    "CMPEQTRUE": 0x62,
    "CMPLTTRUE": 0x64,
    "CMPLETRUE": 0x66,
    "CMPGTTRUE": 0x68,
    "UNORDEREDTRUE": 0x6a,
    "TSEL": 0x32
}

# 舍入模式定义
RND_DEF = {
    "RND_PLUS_INF": 1,
    "RND_MINUS_INF": 2,
    "RND_NEAREST": 3,
    "RND_ZERO": 4
}

# 定义16个关键边界值（FP16）
hex_values = [
    # 零和无穷大
    0x0000,  # 正零
    0x8000,  # 负零
    0x7C00,  # 正无穷
    0xFC00,  # 负无穷
    
    # 非规格化数
    0x0001,  # 最小正非规格化数
    0x03FF,  # 最大正非规格化数
    0x8001,  # 最小负非规格化数
    0x83FF,  # 最大负非规格化数
    
    # 规格化数
    0x0400,  # 最小正规格化数
    0x7BFF,  # 最大正规格化数
    0x8400,  # 最小负规格化数
    0xFBFF,  # 最大负规格化数
    
    # NaN
    0x7C01,  # 信号NaN
    0x7E00,  # 安静NaN
    0xFC01,  # 负信号NaN
    0xFFFF   # 最大NaN
]

def generate_fp16_boundary_test_cases():
    """生成FP16边界测试用例三元组"""
    test_cases = []
    for a in hex_values:
        for b in hex_values:
            for c in hex_values:
                test_cases.append((a, b, c))
    return test_cases

def save_fp16_boundary_test_cases(test_cases, filename="fp16_boundary_test_cases.txt"):
    """保存FP16边界测试激励到文件"""
    with open(filename, "w") as f:
        # 遍历所有操作符和舍入模式组合
        for op_name in OP_DEF.keys():
            for rnd_name in RND_DEF.keys():
                # 为每个组合写入所有测试用例
                for a, b, c in test_cases:
                    f.write(f"{op_name}, {rnd_name}, ")
                    f.write(f"0x{a:04X}, ")  # 4位十六进制格式
                    f.write(f"0x{b:04X}, ")
                    f.write(f"0x{c:04X}\n")
    
    # 计算统计数据
    unique_values = len(hex_values)
    cases_per_combo = unique_values ** 3
    total_combos = len(OP_DEF) * len(RND_DEF)
    total_cases = cases_per_combo * total_combos
    
    print(f"FP16边界值数量: {unique_values}")
    print(f"每个操作符-舍入组合的测试用例: {cases_per_combo}")
    print(f"操作符-舍入组合数: {total_combos}")
    print(f"总测试用例数: {total_cases}")
    print(f"测试激励已保存到 {filename}")

# 生成并保存测试激励
fp16_boundary_test_cases = generate_fp16_boundary_test_cases()
save_fp16_boundary_test_cases(fp16_boundary_test_cases)