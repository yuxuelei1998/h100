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

// CUDA内核：执行测试用例
__global__ void executeTests(const TestCase* __restrict__ testCases, 
                             Result* __restrict__ results, 
                             int numTests) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numTests) return;
    
    // 使用寄存器存储减少内存访问
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
    
    // 存储结果
    results[idx].result = __float_as_uint(res);
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

// 写输出文件
void writeOutputFile(const std::string& filename, 
                    const std::vector<TestCase>& testCases,
                    const std::vector<Result>& results) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "错误：无法创建输出文件 " << filename << std::endl;
        return;
    }
    
    file << "Opcode, Rnd, Operand A, Operand B, Operand C, Result\n";
    
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
             << "0x" << std::hex << std::setw(8) << std::setfill('0') << tc.operandA << ", "
             << "0x" << std::hex << std::setw(8) << std::setfill('0') << tc.operandB << ", "
             << "0x" << std::hex << std::setw(8) << std::setfill('0') << tc.operandC << ", "
             << "0x" << std::hex << std::setw(8) << std::setfill('0') << res.result << "\n";
    }
}

int main(int argc, char* argv[]) {
    std::string inputFilename = "fp32_input.txt";
    std::string outputFilename = "h100_fp32_output.txt";
    
    // 从命令行获取文件名
    if (argc >= 2) {
        inputFilename = argv[1];
    }
    if (argc >= 3) {
        outputFilename = argv[2];
    }
    
    // 检查输入文件是否存在
    std::ifstream testFile(inputFilename);
    if (!testFile.good()) {
        std::cerr << "错误：文件 " << inputFilename << " 不存在，程序终止。\n";
        return 1;
    }
    testFile.close();
    
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
    return 0;
}