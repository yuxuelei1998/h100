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

def gen_operand_fp16():
    """生成符合比例要求的FP16操作数"""
    r = random.random()
    
    # 20% 为正负零和正负无穷大
    if r < 0.2:
        return random.choice([
            np.float16(0.0),          # 正零
            np.float16(-0.0),         # 负零
            np.float16('inf'),        # 正无穷大
            np.float16('-inf')        # 负无穷大
        ])
    
    # 20% 为非数 (NaN)
    elif r < 0.4:
        # 生成不同类型的NaN
        nan_types = [
            np.float16('nan'),        # 标准NaN
            # 安静NaN (尾数最高位=1)
            np.frombuffer(struct.pack('H', 0x7E00), dtype=np.float16)[0],
            # 信号NaN (尾数最高位=0且尾数≠0)
            np.frombuffer(struct.pack('H', 0x7C01), dtype=np.float16)[0],
            # 随机NaN
            np.frombuffer(struct.pack('H', 0x7C00 | random.randint(1, 0x3FF)), dtype=np.float16)[0]
        ]
        return random.choice(nan_types)
    
    # 60% 为常规数
    else:
        # 80% 规格化数，20% 非规格化数
        if random.random() < 0.8:
            # 规格化数
            exp_range = random.choice([
                (-14, 15),           # 全范围
                (-6, 6),             # 中间范围
                (-14, -3),           # 小指数
                (3, 15)              # 大指数
            ])
            exp = random.randint(*exp_range)
            mantissa = random.getrandbits(10)
            value = (1 + mantissa / (2**10)) * (2**exp)
            return np.float16(value) if random.choice([True, False]) else np.float16(-value)
        else:
            # 非规格化数 - 确保尾数非零
            while True:
                # 生成随机符号位
                sign_bit = random.getrandbits(1) << 15
                # 生成非零尾数(0x0001 - 0x03FF)
                mantissa = random.getrandbits(10)
                if mantissa != 0:  # 确保非零
                    bits = sign_bit | mantissa
                    return np.frombuffer(struct.pack('H', bits), dtype=np.float16)[0]

def float16_to_hex(f):
    """将FP16浮点数转换为十六进制表示"""
    # 特殊处理NaN和无穷大以确保一致性
    if np.isnan(f):
        return '0x7E00'  # 标准安静NaN
    elif np.isposinf(f):
        return '0x7C00'
    elif np.isneginf(f):
        return '0xFC00'
    elif f == np.float16(0.0):
        return '0x0000'
    elif f == np.float16(-0.0):
        return '0x8000'
    
    # 转换常规数字
    uint_val = np.frombuffer(f.data, dtype=np.uint16)[0]
    return f"0x{uint_val:04X}"

def generate_fp16_test_cases(num_cases=100):
    """生成FP16测试激励三元组"""
    test_cases = []
    for _ in range(num_cases):
        a = gen_operand_fp16()
        b = gen_operand_fp16()
        c = gen_operand_fp16()
        test_cases.append((a, b, c))
    return test_cases

def save_fp16_test_cases(test_cases, filename="fp16_test_cases.txt"):
    """保存FP16测试激励到文件"""
    with open(filename, "w") as f:
        # 遍历所有操作符和舍入模式组合
        for op_name, op_code in OP_DEF.items():
            for rnd_name, rnd_code in RND_DEF.items():
                # 为每个组合写入所有测试用例
                for a, b, c in test_cases:
                    f.write(f"{op_name}, {rnd_name}, ")
                    f.write(f"{float16_to_hex(a)}, ")
                    f.write(f"{float16_to_hex(b)}, ")
                    f.write(f"{float16_to_hex(c)}\n")
    
    print(f"已生成 {len(test_cases)} 个FP16原始测试用例")
    total_cases = len(test_cases) * len(OP_DEF) * len(RND_DEF)
    print(f"组合后共生成 {total_cases} 个测试激励")

# 生成并保存测试激励
fp16_test_cases = generate_fp16_test_cases(2000)  # 100个原始测试用例
save_fp16_test_cases(fp16_test_cases)