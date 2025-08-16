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

def bf16_to_float(bf16_bits):
    """将BF16位模式转换为32位浮点数"""
    # BF16位模式左移16位形成32位浮点数位模式
    f32_bits = bf16_bits << 16
    return struct.unpack('>f', struct.pack('>I', f32_bits))[0]

def float_to_bf16(f):
    """将32位浮点数转换为BF16位模式"""
    # 获取32位浮点数的位模式
    f32_bits = struct.unpack('>I', struct.pack('>f', f))[0]
    # 取高16位作为BF16位模式
    return (f32_bits >> 16) & 0xFFFF

def gen_operand_bf16():
    """生成符合比例要求的BF16操作数"""
    r = random.random()
    
    # 20% 为正负零和正负无穷大
    if r < 0.2:
        return bf16_to_float(random.choice([
            0x0000,  # 正零
            0x8000,  # 负零
            0x7F80,  # 正无穷大
            0xFF80   # 负无穷大
        ]))
    
    # 20% 为非数 (NaN)
    elif r < 0.4:
        # 生成不同类型的NaN
        return bf16_to_float(random.choice([
            0x7FC0,  # 安静NaN
            0x7F81,  # 信号NaN1
            0x7FA0,  # 信号NaN2
            0xFFC0,  # 负NaN
            0x7F80 | random.randint(1, 0x7F)  # 随机NaN
        ]))
    
    # 60% 为常规数
    else:
        # 80% 规格化数，20% 非规格化数
        if random.random() < 0.8:
            # 规格化数
            exp_range = random.choice([
                (-126, 127),   # 全范围指数
                (-60, 60),     # 中间范围
                (-126, -30),   # 小指数
                (30, 127)      # 大指数
            ])
            exp = random.randint(*exp_range)
            # 尾数7位
            mantissa = random.getrandbits(7)
            # 计算值 (1 + 尾数/128) * 2^(指数)
            value = (1 + mantissa / 128.0) * (2.0 ** exp)
            return value if random.choice([True, False]) else -value
        else:
            # 非规格化数 - 确保尾数非零
            while True:
                # 生成随机符号位
                sign_bit = random.getrandbits(1) << 15
                # 指数部分为0
                exp_bits = 0
                # 生成非零尾数(0x01 - 0x7F)
                mantissa = random.getrandbits(7)
                if mantissa != 0:  # 确保非零
                    # 组合位模式: 符号位 + 指数(0) + 尾数
                    bf16_bits = sign_bit | exp_bits | mantissa
                    return bf16_to_float(bf16_bits)

def bf16_to_hex(bf16_bits):
    """将BF16位模式转换为十六进制表示"""
    return f"0x{bf16_bits:04X}"

def float_as_bf16(f):
    """将浮点数表示为BF16位模式"""
    # 特殊处理NaN和无穷大以确保一致性
    if np.isnan(f):
        return 0x7FC0  # 标准安静NaN
    elif np.isposinf(f):
        return 0x7F80
    elif np.isneginf(f):
        return 0xFF80
    elif f == 0.0:
        return 0x0000
    elif f == -0.0:
        return 0x8000
    
    # 转换常规数字
    return float_to_bf16(f)

def generate_bf16_test_cases(num_cases=100):
    """生成BF16测试激励三元组"""
    test_cases = []
    for _ in range(num_cases):
        a = gen_operand_bf16()
        b = gen_operand_bf16()
        c = gen_operand_bf16()
        test_cases.append((a, b, c))
    return test_cases

def save_bf16_test_cases(test_cases, filename="bf16_test_cases.txt"):
    """保存BF16测试激励到文件"""
    with open(filename, "w") as f:
        # 遍历所有操作符和舍入模式组合
        for op_name, op_code in OP_DEF.items():
            for rnd_name, rnd_code in RND_DEF.items():
                # 为每个组合写入所有测试用例
                for a, b, c in test_cases:
                    f.write(f"{op_name}, {rnd_name}, ")
                    f.write(f"{bf16_to_hex(float_as_bf16(a))}, ")
                    f.write(f"{bf16_to_hex(float_as_bf16(b))}, ")
                    f.write(f"{bf16_to_hex(float_as_bf16(c))}\n")
    
    print(f"已生成 {len(test_cases)} 个BF16原始测试用例")
    total_cases = len(test_cases) * len(OP_DEF) * len(RND_DEF)
    print(f"组合后共生成 {total_cases} 个测试激励")

# 生成并保存测试激励
bf16_test_cases = generate_bf16_test_cases(2000)  # 100个原始测试用例
save_bf16_test_cases(bf16_test_cases)