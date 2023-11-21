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

