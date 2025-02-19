/*
// Basic warp usage
import cv2
import numpy as np
import matplotlib.pyplot as plt

def getPoints(event,x,y,flags,param):
    global click_times,row,col,points
    if event==cv2.EVENT_LBUTTONDOWN:
        click_times+=1
        points[click_times-1]=[x,y]#将点击点的左边传给我们的points，用来变换的第一批参数
        cv2.circle(img,(x,y),4,(25,25,255),-1)#标记我们点击的位置信息
        cv2.rectangle(img,(col-60,row-20),(col,row),(255,255,255),-1)#空出一小块地方用来现实鼠标点击的位置
        cv2.putText(img,'%d,%d'%(x,y),(col-60,row-8),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1)
		

path = r'D:\PythonScripts\test.png'
img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

resized_img = cv2.resize(img, None, fx=0.5, fy=0.5)

cv2.namedWindow('image')
cv2.setMouseCallback('image', getPoints)

click_times = 0
points=np.float32([[0,0],[0,0],[0,0],[0,0]])
row, col, _ = img.shape

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(10)& 0xFF==ord('q') or click_times>=4:#当按下q或者满4个点退出，进行变换
        break
cv2.destroyAllWindows()

pts2=np.float32([[0,0],[300,0],[0,300],[300,300]])#输出图像的分辨率
M=cv2.getPerspectiveTransform(points,pts2)
dst=cv2.warpPerspective(img,M,(300,300))#透视变换
print(points)
cv2.imwrite(r'D:\PythonScripts\test1.png', dst)
*/

/*
// 在不阻塞线程的前提下读取日志详细
import subprocess
import threading

def read_stderr(proc, queue):
    """线程函数，用于读取 stderr 输出并将其放入队列中"""
    for line in proc.stderr:
        queue.append(line.decode('utf-8').strip())

def run_command(command):
    """执行命令并异步捕获 stderr 输出"""
    queue = []
    p = subprocess.Popen(command, shell=True, text=True, bufsize=1, 
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 创建一个线程来读取 stderr 输出
    stderr_thread = threading.Thread(target=read_stderr, args=(p, queue))
    stderr_thread.start()
    
    # 获取命令的输出结果
    stdout, stderr = p.communicate()
    
    # 等待 stderr 线程结束
    stderr_thread.join()
    
    # 从队列中获取 stderr 输出
    errors = queue
    return stdout, errors

# 使用示例
if __name__ == "__main__":
    command = "your-command-here 2>&1"  # 确保命令的 stderr 输出被捕获
    stdout, errors = run_command(command)
    
    print("STDOUT:")
    print(stdout)
    
    print("\nSTDERR:")
    for error in errors:
        print(error)
*/
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

