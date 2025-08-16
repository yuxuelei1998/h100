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
#include <cuda_bf16.h>

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
    uint16_t operandA;
    uint16_t operandB;
    uint16_t operandC;
};

// 结果结构
struct Result {
    uint16_t result;
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

// CUDA内核：执行测试用例（BF16专用）
__global__ void executeTests(const TestCase* __restrict__ testCases, 
                             Result* __restrict__ results, 
                             int numTests) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numTests) return;
    
    TestCase tc = testCases[idx];
    
    // 直接从位模式创建BF16值
    __nv_bfloat16 a = __nv_bfloat16(__nv_bfloat16_raw{tc.operandA});
    __nv_bfloat16 b = __nv_bfloat16(__nv_bfloat16_raw{tc.operandB});
    __nv_bfloat16 c = __nv_bfloat16(__nv_bfloat16_raw{tc.operandC});
    
    __nv_bfloat16 res = __float2bfloat16(0.0f);
    
    // BF16原生指令计算
    switch (tc.opcode) {
        case ADD:
            res = __hadd(a, c);
            break;
        case SUB:
            res = __hsub(a, c);
            break;
        case MUL:
            res = __hmul(a, b);
            break;
        case FMA:
            res = __hfma(a, b, c);
            break;
        case FMS:
            res = __hfma(a, b, __hneg(c));
            break;
        case FNMA:
            res = __hfma(__hneg(a), b, c);
            break;
        case FNMS:
            res = __hfma(__hneg(a), b, __hneg(c));
            break;
        case CMPEQ:
            // BF16比较返回位模式
            res = (a == c) ? __nv_bfloat16(__nv_bfloat16_raw{0xFFFF}) : __nv_bfloat16(__nv_bfloat16_raw{0x0000});
            break;
        case CMPLT:
            res = (a < c) ? __nv_bfloat16(__nv_bfloat16_raw{0xFFFF}) : __nv_bfloat16(__nv_bfloat16_raw{0x0000});
            break;
        case CMPLE:
            res = (a <= c) ? __nv_bfloat16(__nv_bfloat16_raw{0xFFFF}) : __nv_bfloat16(__nv_bfloat16_raw{0x0000});
            break;
        case CMPGT:
            res = (a > c) ? __nv_bfloat16(__nv_bfloat16_raw{0xFFFF}) : __nv_bfloat16(__nv_bfloat16_raw{0x0000});
            break;
        case CMPLTNUM:
            res = (a < c) ? a : c;
            break;
        case CMPLENUM:
            res = (a <= c) ? a : c;
            break;
        case CMPGTNUM:
            res = (a > c) ? a : c;
            break;
        case UNORDERED:
            res = (__hisnan(__bfloat162float(a)) || __hisnan(__bfloat162float(c))) 
                  ? __nv_bfloat16(__nv_bfloat16_raw{0xFFFF}) 
                  : __nv_bfloat16(__nv_bfloat16_raw{0x0000});
            break;
    }
    
    // 存储结果
    results[idx].result = static_cast<uint16_t>(__bfloat16_as_ushort(res));
}

// 解析十六进制字符串
uint16_t parseHex(const std::string& hexStr) {
    return static_cast<uint16_t>(std::stoul(hexStr, nullptr, 16) & 0xFFFF;
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
             << "0x" << std::hex << std::setw(4) << std::setfill('0') << tc.operandA << ", "
             << "0x" << std::hex << std::setw(4) << std::setfill('0') << tc.operandB << ", "
             << "0x" << std::hex << std::setw(4) << std::setfill('0') << tc.operandC << ", "
             << "0x" << std::hex << std::setw(4) << std::setfill('0') << res.result << "\n";
    }
}

int main() {
    std::string inputFilename, outputFilename;
    
    // 获取输入文件名
    while (true) {
        std::cout << "请输入输入文件名 (默认: bf16_input.txt): ";
        std::getline(std::cin, inputFilename);
        
        if (inputFilename.empty()) {
            inputFilename = "bf16_input.txt";
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
    std::cout << "请输入输出文件名 (默认: h100_bf16_output.txt): ";
    std::getline(std::cin, outputFilename);
    if (outputFilename.empty()) {
        outputFilename = "h100_bf16_output.txt";
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
    cudaError_t err;
    
    err = cudaMalloc(&d_testCases, numTests * sizeof(TestCase));
    if (err != cudaSuccess) {
        std::cerr << "设备内存分配错误 (测试用例): " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    err = cudaMalloc(&d_results, numTests * sizeof(Result));
    if (err != cudaSuccess) {
        std::cerr << "设备内存分配错误 (结果): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_testCases);
        return 1;
    }
    
    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // 异步拷贝数据到设备
    err = cudaMemcpyAsync(d_testCases, testCases.data(), numTests * sizeof(TestCase), 
                         cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::cerr << "主机到设备内存拷贝错误: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_testCases);
        cudaFree(d_results);
        cudaStreamDestroy(stream);
        return 1;
    }
    
    // 启动内核
    int blockSize = 512;
    int gridSize = (numTests + blockSize - 1) / blockSize;
    executeTests<<<gridSize, blockSize, 0, stream>>>(d_testCases, d_results, numTests);
    
    // 检查内核执行错误
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cerr << "CUDA内核执行错误: " << cudaGetErrorString(kernelErr) << std::endl;
        cudaFree(d_testCases);
        cudaFree(d_results);
        cudaStreamDestroy(stream);
        return 1;
    }
    
    // 异步拷贝结果回主机
    std::vector<Result> results(numTests);
    err = cudaMemcpyAsync(results.data(), d_results, numTests * sizeof(Result), 
                         cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        std::cerr << "设备到主机内存拷贝错误: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_testCases);
        cudaFree(d_results);
        cudaStreamDestroy(stream);
        return 1;
    }
    
    // 同步流
    cudaStreamSynchronize(stream);
    
    // 检查异步操作错误
    cudaError_t asyncErr = cudaGetLastError();
    if (asyncErr != cudaSuccess) {
        std::cerr << "CUDA异步操作错误: " << cudaGetErrorString(asyncErr) << std::endl;
        cudaFree(d_testCases);
        cudaFree(d_results);
        cudaStreamDestroy(stream);
        return 1;
    }
    
    // 写输出文件
    writeOutputFile(outputFilename, testCases, results);
    
    // 清理资源
    cudaFree(d_testCases);
    cudaFree(d_results);
    cudaStreamDestroy(stream);
    
    std::cout << "H100 BF16 测试完成，结果已写入 " << outputFilename << std::endl;
    //std::cout << "编译建议: nvcc -arch=sm_90 -o BF16H100 BF16H100.cu\n";
    return 0;
}