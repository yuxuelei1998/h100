import random
import struct
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

def gen_operand_fp64():
    """生成符合比例要求的FP64操作数"""
    r = random.random()
    
    # 20% 为正负零和正负无穷大
    if r < 0.2:
        return random.choice([
            0.0,                   # 正零
            -0.0,                  # 负零
            float('inf'),          # 正无穷大
            -float('inf')           # 负无穷大
        ])
    
    # 20% 为非数 (NaN)
    elif r < 0.4:
        # 生成不同类型的NaN
        return random.choice([
            float('nan'),          # 标准NaN
            np.nan,                # numpy NaN
            struct.unpack('d', struct.pack('Q', 0x7FF8000000000001))[0],  # 安静NaN
            struct.unpack('d', struct.pack('Q', 0x7FF0000000000001))[0]   # 信号NaN
        ])
    
    # 60% 为常规数
    else:
        # 80% 规格化数，20% 非规格化数
        if random.random() < 0.8:
            # 规格化数 - 覆盖正常范围和边界
            exp_range = random.choice([
                (-1022, 1023),     # 全范围
                (-500, 500),        # 中间范围
                (-1022, -100),      # 小指数
                (100, 1023)         # 大指数
            ])
            exp = random.randint(*exp_range)
            mantissa = random.getrandbits(52)
            value = (1 + mantissa / (2**52)) * (2**exp)
            return value if random.choice([True, False]) else -value
        else:
            # 非规格化数 - 极小值
            bits = random.getrandbits(63) | (random.getrandbits(1) << 63)
            bits &= 0x000FFFFFFFFFFFFF  # 确保指数部分为0
            bits |= random.getrandbits(52)  # 随机尾数
            return struct.unpack('d', struct.pack('Q', bits))[0]

def float64_to_hex(f):
    """将FP64浮点数转换为十六进制表示"""
    return hex(struct.unpack('Q', struct.pack('d', f))[0])

def generate_fp64_test_cases(num_cases=1000):
    """生成FP64测试激励三元组"""
    test_cases = []
    for _ in range(num_cases):
        a = gen_operand_fp64()
        b = gen_operand_fp64()
        c = gen_operand_fp64()
        test_cases.append((a, b, c))
    return test_cases

def save_fp64_test_cases(test_cases, filename="fp64_test_cases.txt"):
    with open(filename, "w") as f:
        for op_name, op_code in OP_DEF.items():
            for rnd_name, rnd_code in RND_DEF.items():
                for a, b, c in test_cases:
                    f.write(f"{op_name}, {rnd_name}, ")
                    f.write(f"{float64_to_hex(a)}, ")
                    f.write(f"{float64_to_hex(b)}, ")
                    f.write(f"{float64_to_hex(c)}\n")

# 生成并保存测试激励
test_cases = generate_fp64_test_cases()
save_fp64_test_cases(test_cases)
print(f"已生成 {len(test_cases)} 个FP64测试激励并保存到 fp64_test_cases.txt")