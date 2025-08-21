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
    DOT8
};

// 舍入模式枚举
enum RoundMode {
    RND_ZERO, RND_MINUS_INF, RND_PLUS_INF, RND_NEAREST
};

// 测试用例结构
struct TestCase {
    Opcode opcode;
    RoundMode roundMode;
    uint32_t vectorA[8];
    uint32_t vectorB[8];
    uint32_t scalarC;
};

// 结果结构
struct Result {
    uint32_t result;
};

// 字符串到操作码映射
std::map<std::string, Opcode> opcodeMap = {
    {"DOT8", DOT8}
};

// 字符串到舍入模式映射
std::map<std::string, RoundMode> roundModeMap = {
    {"RND_ZERO", RND_ZERO}, {"RND_MINUS_INF", RND_MINUS_INF},
    {"RND_PLUS_INF", RND_PLUS_INF}, {"RND_NEAREST", RND_NEAREST}
};

// CUDA内核：执行八点积加操作
__global__ void executeTests(const TestCase* __restrict__ testCases, 
                             Result* __restrict__ results, 
                             int numTests) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numTests) return;
    
    TestCase tc = testCases[idx];
    float res = 0.0f;
    
    // 将操作数转换为浮点数
    float a[8], b[8], c;
    for (int i = 0; i < 8; i++) {
        a[i] = __uint_as_float(tc.vectorA[i]);
        b[i] = __uint_as_float(tc.vectorB[i]);
    }
    c = __uint_as_float(tc.scalarC);
    
    // 执行八点积加操作: result = Σ(a[i]*b[i]) + c
    // 根据舍入模式选择不同的计算方法
    switch (tc.roundMode) {
        case RND_ZERO:
            res = __fmaf_rz(a[0], b[0], 
                   __fmaf_rz(a[1], b[1], 
                   __fmaf_rz(a[2], b[2], 
                   __fmaf_rz(a[3], b[3], 
                   __fmaf_rz(a[4], b[4], 
                   __fmaf_rz(a[5], b[5], 
                   __fmaf_rz(a[6], b[6], 
                   __fmul_rz(a[7], b[7]))))))));
            res = __fadd_rz(res, c);
            break;
        case RND_MINUS_INF:
            res = __fmaf_rd(a[0], b[0], 
                   __fmaf_rd(a[1], b[1], 
                   __fmaf_rd(a[2], b[2], 
                   __fmaf_rd(a[3], b[3], 
                   __fmaf_rd(a[4], b[4], 
                   __fmaf_rd(a[5], b[5], 
                   __fmaf_rd(a[6], b[6], 
                   __fmul_rd(a[7], b[7]))))))));
            res = __fadd_rd(res, c);
            break;
        case RND_PLUS_INF:
            res = __fmaf_ru(a[0], b[0], 
                   __fmaf_ru(a[1], b[1], 
                   __fmaf_ru(a[2], b[2], 
                   __fmaf_ru(a[3], b[3], 
                   __fmaf_ru(a[4], b[4], 
                   __fmaf_ru(a[5], b[5], 
                   __fmaf_ru(a[6], b[6], 
                   __fmul_ru(a[7], b[7]))))))));
            res = __fadd_ru(res, c);
            break;
        default: // RND_NEAREST
            res = __fmaf_rn(a[0], b[0], 
                   __fmaf_rn(a[1], b[1], 
                   __fmaf_rn(a[2], b[2], 
                   __fmaf_rn(a[3], b[3], 
                   __fmaf_rn(a[4], b[4], 
                   __fmaf_rn(a[5], b[5], 
                   __fmaf_rn(a[6], b[6], 
                   __fmul_rn(a[7], b[7]))))))));
            res = __fadd_rn(res, c);
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
        
        if (tokens.size() == 19) {
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
            
            // 读取向量A的8个元素
            for (int i = 0; i < 8; i++) {
                tc.vectorA[i] = parseHex(tokens[2 + i]);
            }
            
            // 读取向量B的8个元素
            for (int i = 0; i < 8; i++) {
                tc.vectorB[i] = parseHex(tokens[10 + i]);
            }
            
            // 读取标量C
            tc.scalarC = parseHex(tokens[18]);
            
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
    
    file << "Opcode, Rnd, A0, A1, A2, A3, A4, A5, A6, A7, B0, B1, B2, B3, B4, B5, B6, B7, C, Result\n";
    
    // 反向映射用于输出
    std::map<Opcode, std::string> opcodeStr;
    for (const auto& p : opcodeMap) opcodeStr[p.second] = p.first;
    
    std::map<RoundMode, std::string> roundModeStr;
    for (const auto& p : roundModeMap) roundModeStr[p.second] = p.first;
    
    for (size_t i = 0; i < testCases.size(); ++i) {
        const TestCase& tc = testCases[i];
        const Result& res = results[i];
        
        file << opcodeStr[tc.opcode] << ", "
             << roundModeStr[tc.roundMode] << ", ";
        
        // 输出向量A
        for (int j = 0; j < 8; j++) {
            file << "0x" << std::hex << std::setw(8) << std::setfill('0') << tc.vectorA[j];
            if (j < 7) file << ", ";
        }
        
        file << ", ";
        
        // 输出向量B
        for (int j = 0; j < 8; j++) {
            file << "0x" << std::hex << std::setw(8) << std::setfill('0') << tc.vectorB[j];
            if (j < 7) file << ", ";
        }
        
        // 输出标量C和结果
        file << ", 0x" << std::hex << std::setw(8) << std::setfill('0') << tc.scalarC
             << ", 0x" << std::hex << std::setw(8) << std::setfill('0') << res.result << "\n";
    }
}

int main() {
    std::string inputFilename, outputFilename;
    
    // 获取输入文件名
    while (true) {
        std::cout << "请输入输入文件名 (默认: fp32_dot8_input.txt): ";
        std::getline(std::cin, inputFilename);
        
        if (inputFilename.empty()) {
            inputFilename = "fp32_dot8_input.txt";
        }
        
        std::ifstream testFile(inputFilename);
        if (testFile.good()) {
            testFile.close();
            break;
        }
        
        std::cout << "文件 " << inputFilename << " 不存在，请重新输入。\n";
    }
    
    // 获取输出文件名
    std::cout << "请输入输出文件名 (默认: h100_fp32_dot8_output.txt): ";
    std::getline(std::cin, outputFilename);
    if (outputFilename.empty()) {
        outputFilename = "h100_fp32_dot8_output.txt";
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
    
    std::cout << "H100 FP32 八点积加测试完成，结果已写入 " << outputFilename << std::endl;
    return 0;
}
