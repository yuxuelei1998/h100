#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
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
    uint16_t vectorA[8]; // 使用16位存储FP16值
    uint16_t vectorB[8]; // 使用16位存储FP16值
    uint32_t scalarC;    // 使用32位存储FP32值
};

// 结果结构
struct Result {
    uint32_t result; // 使用32位存储FP32结果
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

// 检查CUDA错误
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA Runtime Error at %s:%d : %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// 检查cuBLAS错误
#define checkCublasErrors(err) __checkCublasErrors (err, __FILE__, __LINE__)
inline void __checkCublasErrors(cublasStatus_t err, const char *file, const int line) {
    if (CUBLAS_STATUS_SUCCESS != err) {
        fprintf(stderr, "cuBLAS Error at %s:%d : %d\n", file, line, err);
        exit(EXIT_FAILURE);
    }
}

// 使用cuBLASLt执行八点积加操作
void executeDot8WithCublasLt(const TestCase* testCases, Result* results, int numTests) {
    cublasLtHandle_t handle;
    checkCublasErrors(cublasLtCreate(&handle));
    
    // 设置矩阵描述符
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
    cublasLtMatmulDesc_t operationDesc;
    
    // 创建矩阵描述符
    checkCublasErrors(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, 1, 8, 8));
    checkCublasErrors(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, 8, 1, 1));
    checkCublasErrors(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, 1, 1, 1));
    checkCublasErrors(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, 1, 1, 1));
    
    // 创建操作描述符
    checkCublasErrors(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    
    // 设置操作属性
    checkCublasErrors(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &CUBLAS_OP_N, sizeof(CUBLAS_OP_N)));
    checkCublasErrors(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &CUBLAS_OP_N, sizeof(CUBLAS_OP_N)));
    
    // 设置标量值
    float alpha = 1.0f;
    float beta = 1.0f;
    
    // 为所有测试用例分配设备内存
    __half* d_A;
    __half* d_B;
    float* d_C;
    float* d_D;
    
    checkCudaErrors(cudaMalloc(&d_A, numTests * 8 * sizeof(__half)));
    checkCudaErrors(cudaMalloc(&d_B, numTests * 8 * sizeof(__half)));
    checkCudaErrors(cudaMalloc(&d_C, numTests * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_D, numTests * sizeof(float)));
    
    // 准备输入数据
    std::vector<__half> h_A(numTests * 8);
    std::vector<__half> h_B(numTests * 8);
    std::vector<float> h_C(numTests);
    
    for (int i = 0; i < numTests; i++) {
        for (int j = 0; j < 8; j++) {
            h_A[i * 8 + j] = __ushort_as_half(testCases[i].vectorA[j]);
            h_B[i * 8 + j] = __ushort_as_half(testCases[i].vectorB[j]);
        }
        h_C[i] = __uint_as_float(testCases[i].scalarC);
    }
    
    // 拷贝数据到设备
    checkCudaErrors(cudaMemcpy(d_A, h_A.data(), numTests * 8 * sizeof(__half), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B.data(), numTests * 8 * sizeof(__half), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_C, h_C.data(), numTests * sizeof(float), cudaMemcpyHostToDevice));
    
    // 执行矩阵乘法 (1x8) * (8x1) + (1x1) = (1x1)
    for (int i = 0; i < numTests; i++) {
        checkCublasErrors(cublasLtMatmul(
            handle,
            operationDesc,
            &alpha,
            d_A + i * 8, Adesc,
            d_B + i * 8, Bdesc,
            &beta,
            d_C + i, Cdesc,
            d_D + i, Ddesc,
            nullptr,  // 使用默认算法
            nullptr,  // 不使用工作空间
            0,        // 工作空间大小为0
            0         // 流ID
        ));
    }
    
    // 同步等待所有操作完成
    checkCudaErrors(cudaDeviceSynchronize());
    
    // 拷贝结果回主机
    std::vector<float> h_D(numTests);
    checkCudaErrors(cudaMemcpy(h_D.data(), d_D, numTests * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 转换结果为uint32_t格式
    for (int i = 0; i < numTests; i++) {
        results[i].result = __float_as_uint(h_D[i]);
    }
    
    // 清理资源
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_D));
    
    checkCublasErrors(cublasLtMatmulDescDestroy(operationDesc));
    checkCublasErrors(cublasLtMatrixLayoutDestroy(Adesc));
    checkCublasErrors(cublasLtMatrixLayoutDestroy(Bdesc));
    checkCublasErrors(cublasLtMatrixLayoutDestroy(Cdesc));
    checkCublasErrors(cublasLtMatrixLayoutDestroy(Ddesc));
    checkCublasErrors(cublasLtDestroy(handle));
}

// 解析十六进制字符串（16位）
uint16_t parseHex16(const std::string& hexStr) {
    return static_cast<uint16_t>(std::stoul(hexStr, nullptr, 16));
}

// 解析十六进制字符串（32位）
uint32_t parseHex32(const std::string& hexStr) {
    return static_cast<uint32_t>(std::stoul(hexStr, nullptr, 16));
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
    int lineNum = 0;
    while (std::getline(file, line)) {
        lineNum++;
        
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(iss, token, ',')) {
            token.erase(0, token.find_first_not_of(' '));
            token.erase(token.find_last_not_of(' ') + 1);
            tokens.push_back(token);
        }
        
        // 正确的字段数量应该是19: 操作码 + 舍入模式 + 8个A向量元素 + 8个B向量元素 + 1个标量C
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
            
            // 读取向量A的8个元素 (索引2到9)
            for (int i = 0; i < 8; i++) {
                tc.vectorA[i] = parseHex16(tokens[2 + i]);
            }
            
            // 读取向量B的8个元素 (索引10到17)
            for (int i = 0; i < 8; i++) {
                tc.vectorB[i] = parseHex16(tokens[10 + i]);
            }
            
            // 读取标量C (索引18)
            tc.scalarC = parseHex32(tokens[18]);
            
            testCases.push_back(tc);
        } else {
            std::cerr << "行 " << lineNum << " 字段数量不正确，应为19，实际为" << tokens.size() << std::endl;
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
        
        // 输出向量A (16位，4位十六进制)
        for (int j = 0; j < 8; j++) {
            file << "0x" << std::hex << std::setw(4) << std::setfill('0') << tc.vectorA[j];
            if (j < 7) file << ", ";
        }
        
        file << ", ";
        
        // 输出向量B (16位，4位十六进制)
        for (int j = 0; j < 8; j++) {
            file << "0x" << std::hex << std::setw(4) << std::setfill('0') << tc.vectorB[j];
            if (j < 7) file << ", ";
        }
        
        // 输出标量C (32位，8位十六进制)和结果 (32位，8位十六进制)
        file << ", 0x" << std::hex << std::setw(8) << std::setfill('0') << tc.scalarC
             << ", 0x" << std::hex << std::setw(8) << std::setfill('0') << res.result << "\n";
    }
}

int main() {
    std::string inputFilename, outputFilename;
    
    // 获取输入文件名
    while (true) {
        std::cout << "请输入输入文件名 (默认: fp16a16b_fp32c_dot8_input.txt): ";
        std::getline(std::cin, inputFilename);
        
        if (inputFilename.empty()) {
            inputFilename = "fp16a16b_fp32c_dot8_input.txt";
        }
        
        std::ifstream testFile(inputFilename);
        if (testFile.good()) {
            testFile.close();
            break;
        }
        
        std::cout << "文件 " << inputFilename << " 不存在，请重新输入。\n";
    }
    
    // 获取输出文件名
    std::cout << "请输入输出文件名 (默认: h100_fp16a16b_fp32c_dot8_output.txt): ";
    std::getline(std::cin, outputFilename);
    if (outputFilename.empty()) {
        outputFilename = "h100_fp16a16b_fp32c_dot8_output.txt";
    }
    
    // 读取输入文件
    std::vector<TestCase> testCases = readInputFile(inputFilename);
    if (testCases.empty()) {
        std::cerr << "错误：无有效测试用例，程序终止。\n";
        return 1;
    }
    
    int numTests = testCases.size();
    std::cout << "找到 " << numTests << " 个测试用例，开始处理...\n";
    
    // 分配结果内存
    std::vector<Result> results(numTests);
    
    // 使用cuBLASLt执行测试
    executeDot8WithCublasLt(testCases.data(), results.data(), numTests);
    
    // 写输出文件
    writeOutputFile(outputFilename, testCases, results);
    
    std::cout << "H100 FP16(A,B) FP32(C) 八点积加测试完成，结果已写入 " << outputFilename << std::endl;
    return 0;
}