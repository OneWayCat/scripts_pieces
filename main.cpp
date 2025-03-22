#include "eigen3/Eigen/Dense"

// np.roots() 采用伴随矩阵求特征值（Companion Matrix Eigenvalues）
// eigen 3.3.9
std::vector<std::complex<double>> solveQuartic(const std::vector<double>& coef) {
    if (coef.size() != 5) {
        throw std::invalid_argument("Quartic equation must have exactly 5 coefficients.");
    }

    double a = coef[0], b = coef[1], c = coef[2], d = coef[3], e = coef[4];
    if (a == 0) {
        throw std::invalid_argument("Coefficient 'a' cannot be zero.");
    }

    // Normalize coefficients
    b /= a;
    c /= a;
    d /= a;
    e /= a;

    // Construct the companion matrix
    Eigen::Matrix4d C;
    C << 0, 0, 0, -e,
        1, 0, 0, -d,
        0, 1, 0, -c,
        0, 0, 1, -b;

    // Compute eigenvalues
    Eigen::EigenSolver<Eigen::Matrix4d> solver(C);
    Eigen::Vector4cd roots = solver.eigenvalues();

    // Convert results to std::vector<std::complex<double>>
    std::vector<std::complex<double>> result;
    for (int i = 0; i < 4; ++i) {
        result.push_back(roots[i]);
    }

    return result;
}

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

template<std::size_t... N>
constexpr auto create_array(std::index_sequence<N...>) {
    return std::array<float, sizeof...(N)>{ (2 * N * 1.0 / 100)... };
}

int main() {
    constexpr auto arr = create_array(std::make_index_sequence<100>{});
    // 步骤1: 假设已有OpenCL源代码在"kernel.cl"文件中

    // 步骤2和3: 创建程序对象并编译
    cl_int err;
    cl_context context;
    cl_device_id device;
    cl_program program;
    size_t sourceSize;
    char *sourceStr;

    // 初始化OpenCL环境，获取context和device（略去细节）
    // ...

    // 读取OpenCL源代码文件
    FILE *file = fopen("kernel.cl", "rb");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    sourceSize = ftell(file);
    fseek(file, 0, SEEK_SET);
    sourceStr = (char *)malloc(sourceSize + 1);
    fread(sourceStr, sourceSize, 1, file);
    fclose(file);
    sourceStr[sourceSize] = '\0';

    // 创建程序对象
    program = clCreateProgramWithSource(context, 1, (const char**)&sourceStr, &sourceSize, &err);
    free(sourceStr); // 释放源代码字符串
    if (err != CL_SUCCESS) {
        printf("Failed to create program with source.\n");
        exit(EXIT_FAILURE);
    }

    // 编译程序对象
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // 处理编译错误（略）
        exit(EXIT_FAILURE);
    }

    // 步骤4: 提取二进制代码
    size_t binarySize;
    err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binarySize, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to get binary size.\n");
        exit(EXIT_FAILURE);
    }

    unsigned char *binary = (unsigned char *)malloc(binarySize);
    err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &binary, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to get program binaries.\n");
        free(binary);
        exit(EXIT_FAILURE);
    }

    // 步骤5: 写入二进制文件
    FILE *binaryFile = fopen("kernel.bin", "wb");
    if (!binaryFile) {
        perror("Error opening binary file");
        free(binary);
        exit(EXIT_FAILURE);
    }
    fwrite(binary, binarySize, 1, binaryFile);
    fclose(binaryFile);

    // 清理资源
    free(binary);
    clReleaseProgram(program);
    // 清理其他OpenCL资源（略去细节）

    return 0;
}

/******************************加载编译好的二进制文件************************************/
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    // 假设已经初始化了OpenCL环境，并且有一个有效的context和device

    cl_int err;
    cl_context context;
    cl_device_id device;
    // 初始化OpenCL环境和获取context/device的代码（略去细节）

    // 步骤1: 读取二进制文件内容
    FILE *binaryFile = fopen("kernel.bin", "rb");
    if (!binaryFile) {
        perror("Error opening binary file");
        exit(EXIT_FAILURE);
    }
    fseek(binaryFile, 0, SEEK_END);
    size_t binarySize = ftell(binaryFile);
    fseek(binaryFile, 0, SEEK_SET);
    unsigned char *binary = (unsigned char *)malloc(binarySize);
    fread(binary, binarySize, 1, binaryFile);
    fclose(binaryFile);

    // 步骤2: 使用clCreateProgramWithBinary创建程序对象
    const unsigned char **binaryPtr = &binary;
    size_t *binarySizes = &binarySize;
    cl_program program = clCreateProgramWithBinary(context, 1, &device, binarySizes, &binaryPtr, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create program with binary.\n");
        free(binary);
        exit(EXIT_FAILURE);
    }

    // 步骤3: 构建程序（如果二进制是IR，则需要构建）
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // 处理构建错误（略）
    }

    // 清理资源
    free(binary);
    clReleaseProgram(program);
    // 清理其他OpenCL资源（略去细节）

    return 0;
}

// 特征值分解计算多项式的解
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    // 选择方程的系数
    double a = 1.0;
    double b = -1.0;
    double c = 2.0;
    double d = -1.0;

    // 转换为标准形式
    double p = b / a;
    double q = c / a;
    double r = d / a;

    // 构造矩阵 A
    cv::Mat A = (cv::Mat_<double>(3, 3) <<
        0, 1, 0,
        0, 0, 1,
        -r / p, -q / p, -p / p);

    // 计算特征值
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(A, eigenvalues, eigenvectors);

    // 输出特征值，包括实部和虚部
    std::cout << "Complex solutions: " << std::endl;
    for (int i = 0; i < eigenvalues.rows; ++i) {
        std::complex<double> root(eigenvalues.at<double>(i), 0.0);
        std::cout << "x_" << i + 1 << " = " << root << std::endl;
    }

    return 0;
}



// 利用AVX对mat数据进行加速
#include <opencv2/opencv.hpp>
#include <immintrin.h>  // 包含AVX头文件

void processImageAVX(cv::Mat& input, cv::Mat& output) {
    int rows = input.rows;
    int cols = input.cols;

    // 检查图像尺寸是否为4的倍数，以确保可以使用AVX指令集
    if (cols % 4 != 0) {
        std::cerr << "Error: Image width must be a multiple of 4 for AVX optimization." << std::endl;
        return;
    }

    // 遍历图像的每一行
    for (int row = 0; row < rows; ++row) {
        double* inputPtr = input.ptr<double>(row);
        double* outputPtr = output.ptr<double>(row);

        // 使用AVX指令处理每4个double值
        for (int col = 0; col < cols; col += 4) {
            // 读取4个double值
            __m256d data = _mm256_loadu_pd(inputPtr + col);

            // 示例：将每个double值乘以常数
            __m256d constant = _mm256_set1_pd(2.0);
            data = _mm256_mul_pd(data, constant);

            // 将处理后的double值存回输出图像
            _mm256_storeu_pd(outputPtr + col, data);
        }
    }
}

int main() {
    // 读取图像
    cv::Mat inputImage = cv::imread("input_image.jpg", cv::IMREAD_GRAYSCALE);

    if (inputImage.empty()) {
        std::cerr << "Error: Unable to read the input image." << std::endl;
        return -1;
    }

    // 创建输出图像
    cv::Mat outputImage(inputImage.size(), CV_64F);

    // 处理图像（使用AVX优化）
    processImageAVX(inputImage, outputImage);

    // 显示结果或保存输出图像
    // 请注意，如果输出图像为CV_64F类型，可能需要转换为合适的范围以便显示
    cv::imshow("Output Image (AVX)", outputImage);
    cv::waitKey(0);

    return 0;
}

// omp与avx进行结合
#pragma omp parallel for
for (int row = 0; row < rows; ++row) {
    double* inputPtr = input.ptr<double>(row);
    double* outputPtr = output.ptr<double>(row);

    // 使用AVX指令处理每4个double值
    for (int col = 0; col < cols; col += 4) {
        // 读取4个double值
        __m256d data = _mm256_loadu_pd(inputPtr + col);
        // AVX操作逻辑
        // ...

        // 将处理后的double值存回输出图像
        _mm256_storeu_pd(outputPtr + col, data);
    }

    // 处理剩余部分（非4的整数倍）
    for (int col = cols - cols % 4; col < cols; ++col) {
        // 处理剩余部分的逻辑
        outputPtr[col] = inputPtr[col];
    }
}


// 方法1：枷锁对myFunction调用进行保护
#include <Windows.h>
#include <iostream>
#include <vector>
#include <future>
#include <mutex>

// 声明一个函数指针类型，用于保存动态库中的函数地址
typedef void (*MyFunction)();

// 全局变量用于存储函数地址
MyFunction myFunction = nullptr;
std::mutex myFunctionMutex; // 用于保护对myFunction的访问

// 使用std::async异步调用myFunction
void asyncCall() {
    std::lock_guard<std::mutex> lock(myFunctionMutex); // 保护对myFunction的访问
    if (myFunction != nullptr) {
        myFunction();
    }
}

int main() {
    // 加载动态库
    HMODULE hDll = LoadLibrary(L"YourLibrary.dll");

    if (hDll == NULL) {
        std::cerr << "Failed to load the library." << std::endl;
        return 1;
    }

    // 获取函数地址
    {
        std::lock_guard<std::mutex> lock(myFunctionMutex);
        myFunction = (MyFunction)GetProcAddress(hDll, "YourFunction");
    }

    if (myFunction == nullptr) {
        std::cerr << "Failed to get the function address." << std::endl;
        FreeLibrary(hDll);
        return 1;
    }

    // 创建4个异步任务
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 4; ++i) {
        futures.push_back(std::async(std::launch::async, asyncCall));
    }

    // 等待所有异步任务完成
    for (auto& future : futures) {
        future.wait();
    }

    // 释放库
    FreeLibrary(hDll);

    return 0;
}
// 方法2 使用call_once
#include <Windows.h>
#include <iostream>
#include <vector>
#include <mutex>

// 声明一个函数指针类型，用于保存动态库中的函数地址
typedef void (*MyFunction)();

// 全局变量用于存储函数地址
MyFunction myFunction = nullptr;
std::once_flag onceFlag; // 一次性初始化标志

// 获取函数地址的函数
void initializeFunction() {
    myFunction = (MyFunction)GetProcAddress(GetModuleHandle(L"YourLibrary.dll"), "YourFunction");
}

// 使用std::async异步调用myFunction
void asyncCall() {
    std::call_once(onceFlag, initializeFunction); // 一次性初始化
    if (myFunction != nullptr) {
        myFunction();
    }
}

int main() {
    // 创建4个异步任务
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 4; ++i) {
        futures.push_back(std::async(std::launch::async, asyncCall));
    }

    // 等待所有异步任务完成
    for (auto& future : futures) {
        future.wait();
    }

    return 0;
}

