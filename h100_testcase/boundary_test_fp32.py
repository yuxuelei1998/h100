import numpy as np

# 操作符定义
OP_DEF = {
    "ADD": 0x10,
    "SUB": 0x12,
    "MUL": 0x14,
    #"MPDPA": 0x16,
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
    #"TSEL": 0x32
}

# 舍入模式定义
RND_DEF = {
    "RND_PLUS_INF": 1,
    "RND_MINUS_INF": 2,
    "RND_NEAREST": 3,
    "RND_ZERO": 4
}

# 定义16个边界值
hex_values = [
    0x00000000, 0x80000000, 0x7F800000, 0xFF800000,  # 0, -0, +∞, -∞
    0x00000001, 0x007FFFFF, 0x80000001, 0x807FFFFF,  # 非规格化数
    0x00800000, 0x7F7FFFFF, 0x80800000, 0xFF7FFFFF,  # 规格化数
    0x7F800001, 0x7FFFFFFF, 0xFF800001, 0xFFFFFFFF   # NaN
]

def hex_to_float(hex_val):
    """将16进制转换为浮点数"""
    return np.frombuffer(np.uint32(hex_val).tobytes(), dtype=np.float32)[0]

def float_to_hex(f):
    """将浮点数转换为16进制"""
    return np.frombuffer(np.float32(f).tobytes(), dtype=np.uint32)[0]

def generate_boundary_test_cases():
    """生成边界测试用例"""
    test_cases = []
    for a in hex_values:
        for b in hex_values:
            for c in hex_values:
                test_cases.append((a, b, c))
    return test_cases

def save_boundary_test_cases(test_cases, filename="fp32_boundary_test_cases.txt"):
    """保存边界测试激励到文件"""
    with open(filename, "w") as f:
        # 遍历所有操作符和舍入模式组合
        for op_name in OP_DEF.keys():
            for rnd_name in RND_DEF.keys():
                # 为每个组合写入所有测试用例
                for a, b, c in test_cases:
                    f.write(f"{op_name}, {rnd_name}, ")
                    f.write(f"0x{a:08X}, ")
                    f.write(f"0x{b:08X}, ")
                    f.write(f"0x{c:08X}\n")
    
    # 计算统计数据
    unique_values = len(hex_values)
    cases_per_combo = unique_values ** 3
    total_combos = len(OP_DEF) * len(RND_DEF)
    total_cases = cases_per_combo * total_combos
    
    print(f"边界值数量: {unique_values}")
    print(f"每个操作符-舍入组合的测试用例: {cases_per_combo}")
    print(f"操作符-舍入组合数: {total_combos}")
    print(f"总测试用例数: {total_cases}")
    print(f"测试激励已保存到 {filename}")

# 生成并保存测试激励
boundary_test_cases = generate_boundary_test_cases()
save_boundary_test_cases(boundary_test_cases)