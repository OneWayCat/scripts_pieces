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

