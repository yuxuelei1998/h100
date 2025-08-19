#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cctype>
#include <cmath>
#include <map>
#include <cstdint>
#include <cfenv>

// 使用CUDA特定的浮点异常标志定义
#define CUDA_FE_INVALID     0x01
#define CUDA_FE_DIVBYZERO   0x02
#define CUDA_FE_OVERFLOW    0x04
#define CUDA_FE_UNDERFLOW   0x08
#define CUDA_FE_INEXACT     0x10
#define CUDA_FE_DENORMAL    0x20

// 操作码枚举
enum Opcode {
    ADD, SUB, MUL, FMA, FMS, FNMA, FNMS,
    CMPEQ, CMPLT, CMPLE, CMPGT,
    CMPLTNUM, CMPLENUM, CMPGTNUM, UNORDERED
};

// 舍入模式枚举
enum RoundMode {
    RND_ZERO, RND_MINUS_INF, RND_PLUS_INF, RND_NEAREST
};

// 测试用例结构
struct TestCase {
    Opcode opcode;
    RoundMode roundMode;
    uint32_t operandA;
    uint32_t operandB;
    uint32_t operandC;
};

// 结果结构
struct Result {
    uint32_t result;
    uint32_t flags;
};

// 字符串到操作码映射
std::map<std::string, Opcode> opcodeMap = {
    {"ADD", ADD}, {"SUB", SUB}, {"MUL", MUL}, {"FMA", FMA}, {"FMS", FMS},
    {"FNMA", FNMA}, {"FNMS", FNMS}, {"CMPEQ", CMPEQ}, {"CMPLT", CMPLT},
    {"CMPLE", CMPLE}, {"CMPGT", CMPGT}, {"CMPLTNUM", CMPLTNUM},
    {"CMPLENUM", CMPLENUM}, {"CMPGTNUM", CMPGTNUM}, {"UNORDERED", UNORDERED}
};

// 字符串到舍入模式映射
std::map<std::string, RoundMode> roundModeMap = {
    {"RND_ZERO", RND_ZERO}, {"RND_MINUS_INF", RND_MINUS_INF},
    {"RND_PLUS_INF", RND_PLUS_INF}, {"RND_NEAREST", RND_NEAREST}
};

// 异常标志名称映射
std::map<uint32_t, std::string> flagNames = {
    {CUDA_FE_INVALID, "INVALID"},
    {CUDA_FE_DIVBYZERO, "DIVBYZERO"},
    {CUDA_FE_OVERFLOW, "OVERFLOW"},
    {CUDA_FE_UNDERFLOW, "UNDERFLOW"},
    {CUDA_FE_INEXACT, "INEXACT"},
    {CUDA_FE_DENORMAL, "DENORMAL"}
};

// 辅助函数：将数值转为十六进制字符串
std::string toHexString(uint32_t value, int width = 8) {
    std::stringstream ss;
    ss << "0x" << std::hex << std::setw(width) << std::setfill('0') << value;
    return ss.str();
}

// 格式化异常标志为可读字符串
std::string formatFlags(uint32_t flags) {
    if (flags == 0) return "NONE";
    
    std::string result;
    for (const auto& pair : flagNames) {
        if (flags & pair.first) {
            if (!result.empty()) result += "|";
            result += pair.second;
            flags &= ~pair.first;
        }
    }
    
    if (flags != 0) {
        if (!result.empty()) result += "|";
        result += "UNKNOWN(0x" + toHexString(flags, 1) + ")";
    }
    
    return result;
}

// CUDA内核：执行测试用例（添加异常检测）
__global__ void executeTests(const TestCase* __restrict__ testCases, 
                             Result* __restrict__ results, 
                             int numTests) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numTests) return;
    
    // 保存当前的浮点状态寄存器
    unsigned int initial_fpsr;
    asm("mov.u32 %0, %fpsr;" : "=r"(initial_fpsr));
    
    // 清除所有异常标志
    asm("mov.u32 %fpsr, 0;");
    
    TestCase tc = testCases[idx];
    const float a = __uint_as_float(tc.operandA);
    const float b = __uint_as_float(tc.operandB);
    const float c = __uint_as_float(tc.operandC);
    float res = 0.0f;
    
    // 分支预测优化
    switch (tc.opcode) {
        case ADD:
            switch (tc.roundMode) {
                case RND_ZERO: res = __fadd_rz(a, c); break;
                case RND_MINUS_INF: res = __fadd_rd(a, c); break;
                case RND_PLUS_INF: res = __fadd_ru(a, c); break;
                default: res = __fadd_rn(a, c);
            }
            break;
        case SUB:
            switch (tc.roundMode) {
                case RND_ZERO: res = __fsub_rz(a, c); break;
                case RND_MINUS_INF: res = __fsub_rd(a, c); break;
                case RND_PLUS_INF: res = __fsub_ru(a, c); break;
                default: res = __fsub_rn(a, c);
            }
            break;
        case MUL:
            switch (tc.roundMode) {
                    case RND_ZERO: res = __fmul_rz(a, b); break;
                    case RND_MINUS_INF: res = __fmul_rd(a, b); break;
                    case RND_PLUS_INF: res = __fmul_ru(a, b); break;
                    default: res = __fmul_rn(a, b);
            }
            break;
        case FMA:
            switch (tc.roundMode) {
                case RND_ZERO: res = __fmaf_rz(a, b, c); break;
                case RND_MINUS_INF: res = __fmaf_rd(a, b, c); break;
                case RND_PLUS_INF: res = __fmaf_ru(a, b, c); break;
                default: res = __fmaf_rn(a, b, c);
            }
            break;
        case FMS:
            switch (tc.roundMode) {
                case RND_ZERO: res = __fmaf_rz(a, b, -c); break;
                case RND_MINUS_INF: res = __fmaf_rd(a, b, -c); break;
                case RND_PLUS_INF: res = __fmaf_ru(a, b, -c); break;
                default: res = __fmaf_rn(a, b, -c);
            }
            break;
        case FNMA:
            switch (tc.roundMode) {
                case RND_ZERO: res = __fmaf_rz(-a, b, c); break;
                case RND_MINUS_INF: res = __fmaf_rd(-a, b, c); break;
                case RND_PLUS_INF: res = __fmaf_ru(-a, b, c); break;
                default: res = __fmaf_rn(-a, b, c);
            }
            break;
        case FNMS:
            switch (tc.roundMode) {
                case RND_ZERO: res = __fmaf_rz(-a, b, -c); break;
                case RND_MINUS_INF: res = __fmaf_rd(-a, b, -c); break;
                case RND_PLUS_INF: res = __fmaf_ru(-a, b, -c); break;
                default: res = __fmaf_rn(-a, b, -c);
            }
            break;
        case CMPEQ:
            res = (a == c) ? __int_as_float(0xFFFFFFFF) : 0.0f;
            break;
        case CMPLT:
            res = (a < c) ? __int_as_float(0xFFFFFFFF) : 0.0f;
            break;
        case CMPLE:
            res = (a <= c) ? __int_as_float(0xFFFFFFFF) : 0.0f;
            break;
        case CMPGT:
            res = (a > c) ? __int_as_float(0xFFFFFFFF) : 0.0f;
            break;
        case CMPLTNUM:
            res = fminf(a, c);
            break;
        case CMPLENUM:
            res = fminf(a, c);
            break;
        case CMPGTNUM:
            res = fmaxf(a, c);
            break;
        case UNORDERED:
            res = (isnan(a) || isnan(c)) ? __int_as_float(0xFFFFFFFF) : 0.0f;
            break;
    }
    
    // 获取并存储异常标志
    unsigned int flags;
    asm("mov.u32 %0, %fpsr;" : "=r"(flags));
    results[idx].result = __float_as_uint(res);
    
    // 提取异常标志位（低5位）
    results[idx].flags = flags & 0x1F;
    
    // 恢复原始浮点状态寄存器
    asm("mov.u32 %fpsr, %0;" : : "r"(initial_fpsr));
}

// 解析十六进制字符串
uint32_t parseHex(const std::string& hexStr) {
    return std::stoul(hexStr, nullptr, 16);
}

// 读取输入文件
std::vector<TestCase> readInputFile(const std::string& filename) {
    std::vector<TestCase> testCases;
    std::ifstream file(filename);
    
    // 检查文件是否成功打开
    if (!file.is_open()) {
        std::cerr << "错误：无法打开输入文件 " << filename << std::endl;
        return testCases;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(iss, token, ',')) {
            token.erase(0, token.find_first_not_of(' '));
            token.erase(token.find_last_not_of(' ') + 1);
            tokens.push_back(token);
        }
        
        if (tokens.size() == 5) {
            TestCase tc;
            if (opcodeMap.find(tokens[0]) != opcodeMap.end()) {
                tc.opcode = opcodeMap[tokens[0]];
            } else {
                std::cerr << "未知操作码: " << tokens[0] << std::endl;
                continue;
            }
            
            if (roundModeMap.find(tokens[1]) != roundModeMap.end()) {
                tc.roundMode = roundModeMap[tokens[1]];
            } else {
                std::cerr << "未知舍入模式: " << tokens[1] << std::endl;
                continue;
            }
            
            tc.operandA = parseHex(tokens[2]);
            tc.operandB = parseHex(tokens[3]);
            tc.operandC = parseHex(tokens[4]);
            testCases.push_back(tc);
        }
    }
    
    if (testCases.empty()) {
        std::cerr << "警告：输入文件中没有找到有效测试用例\n";
    }
    
    return testCases;
}

// 写输出文件（添加异常标志输出）
void writeOutputFile(const std::string& filename, 
                    const std::vector<TestCase>& testCases,
                    const std::vector<Result>& results) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "错误：无法创建输出文件 " << filename << std::endl;
        return;
    }
    
    file << "Opcode, Rnd, Operand A, Operand B, Operand C, Result, Flags, Flag Descriptions\n";
    
    // 反向映射用于输出
    std::map<Opcode, std::string> opcodeStr;
    for (const auto& p : opcodeMap) opcodeStr[p.second] = p.first;
    
    std::map<RoundMode, std::string> roundModeStr;
    for (const auto& p : roundModeMap) roundModeStr[p.second] = p.first;
    
    for (size_t i = 0; i < testCases.size(); ++i) {
        const TestCase& tc = testCases[i];
        const Result& res = results[i];
        
        file << opcodeStr[tc.opcode] << ", "
             << roundModeStr[tc.roundMode] << ", "
             << toHexString(tc.operandA) << ", "
             << toHexString(tc.operandB) << ", "
             << toHexString(tc.operandC) << ", "
             << toHexString(res.result) << ", "
             << toHexString(res.flags, 2) << ", "
             << formatFlags(res.flags) << "\n";
    }
}

int main() {
    std::string inputFilename, outputFilename;
    
    // 获取输入文件名
    while (true) {
        std::cout << "请输入输入文件名 (默认: fp32_input.txt): ";
        std::getline(std::cin, inputFilename);
        
        if (inputFilename.empty()) {
            inputFilename = "fp32_input.txt";
        }
        
        // 检查文件是否存在
        std::ifstream testFile(inputFilename);
        if (testFile.good()) {
            testFile.close();
            break;
        }
        
        std::cout << "文件 " << inputFilename << " 不存在，请重新输入。\n";
    }
    
    // 获取输出文件名
    std::cout << "请输入输出文件名 (默认: h100_fp32_output.txt): ";
    std::getline(std::cin, outputFilename);
    if (outputFilename.empty()) {
        outputFilename = "h100_fp32_output.txt";
    }
    
    // 读取输入文件
    std::vector<TestCase> testCases = readInputFile(inputFilename);
    if (testCases.empty()) {
        std::cerr << "错误：无有效测试用例，程序终止。\n";
        return 1;
    }
    
    int numTests = testCases.size();
    std::cout << "找到 " << numTests << " 个测试用例，开始处理...\n";
    
    // 分配设备内存
    TestCase* d_testCases;
    Result* d_results;
    cudaMalloc(&d_testCases, numTests * sizeof(TestCase));
    cudaMalloc(&d_results, numTests * sizeof(Result));
    
    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // 异步拷贝数据到设备
    cudaMemcpyAsync(d_testCases, testCases.data(), numTests * sizeof(TestCase), 
                   cudaMemcpyHostToDevice, stream);
    
    // 启动内核
    int blockSize = 512;
    int gridSize = (numTests + blockSize - 1) / blockSize;
    executeTests<<<gridSize, blockSize, 0, stream>>>(d_testCases, d_results, numTests);
    
    // 异步拷贝结果回主机
    std::vector<Result> results(numTests);
    cudaMemcpyAsync(results.data(), d_results, numTests * sizeof(Result), 
                   cudaMemcpyDeviceToHost, stream);
    
    // 同步流
    cudaStreamSynchronize(stream);
    
    // 写输出文件
    writeOutputFile(outputFilename, testCases, results);
    
    // 清理资源
    cudaFree(d_testCases);
    cudaFree(d_results);
    cudaStreamDestroy(stream);
    
    std::cout << "H100 FP32 测试完成，结果已写入 " << outputFilename << std::endl;
    //std::cout << "编译建议: nvcc -arch=sm_90 -o FP32H100FLAG FP32H100FLAG.cu\n";
    return 0;
}