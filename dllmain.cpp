// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "framework.h"
#include <DbgHelp.h>
#include <tchar.h>
#include <cstdio>
#include <stdlib.h>

#pragma comment(lib, "Dbghelp.lib")

LONG WINAPI exceptionFilter(EXCEPTION_POINTERS* pExceptionInfo)
{
    printf("Err dumped..................");
    TCHAR szDumpPath[MAX_PATH] = TEXT("D:\\Dump\\");
    TCHAR szDumpFile[MAX_PATH];

    // 创建dump目录
    CreateDirectory(szDumpPath, NULL);

    // 生成dump文件名
    _stprintf_s(szDumpFile, MAX_PATH, TEXT("%sMiniDump_%d.dmp"), szDumpPath, GetCurrentProcessId());

    // 配置MiniDump
    MINIDUMP_EXCEPTION_INFORMATION mdei = { 0 };
    mdei.ThreadId = GetCurrentThreadId();
    mdei.ExceptionPointers = pExceptionInfo;
    mdei.ClientPointers = FALSE;

    // 创建MiniDump
    HANDLE hFile = CreateFile(szDumpFile, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    MINIDUMP_EXCEPTION_INFORMATION* pMdei = &mdei;
    MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), hFile, MiniDumpNormal, pMdei, NULL, NULL);
    CloseHandle(hFile);

    // 这里可以添加其他异常处理代码
    // ...

    return EXCEPTION_EXECUTE_HANDLER;
}

void MyInvalidParameterHandler(const wchar_t* expression, const wchar_t* function, const wchar_t* file, unsigned int line, uintptr_t pReserved)
{
    fwprintf(stderr, L"Invalid parameter error:\n");
    fwprintf(stderr, L"Expression: %ls\n", expression);
    fwprintf(stderr, L"Function: %ls\n", function);
    fwprintf(stderr, L"File: %ls\n", file);
    fwprintf(stderr, L"Line: %u\n", line);
    fwprintf(stderr, L"Reserved: %lu\n", pReserved);
    printf("Error .....");
    // 可以在这里调用MiniDumpWriteDump或其他日志记录函数
}

// 异常过滤器函数
LONG WINAPI VectoredExceptionHandler(PEXCEPTION_POINTERS pExceptionInfo) {
    CONTEXT* pContext = pExceptionInfo->ContextRecord;
    EXCEPTION_RECORD* pExceptionRecord = pExceptionInfo->ExceptionRecord;

    // 打印异常代码和异常地址
    printf("Exception code: %lx\n", pExceptionRecord->ExceptionCode);
    printf("Exception address: %p\n", pExceptionRecord->ExceptionAddress);

    // 初始化堆栈跟踪
    STACKFRAME64 stackFrame;
    memset(&stackFrame, 0, sizeof(stackFrame));
#ifdef _AMD64_
    stackFrame.AddrPC.Offset = pContext->Rip;
    stackFrame.AddrFrame.Offset = pContext->Rsp;
    stackFrame.AddrStack.Offset = pContext->Rsp;
#else
    stackFrame.AddrPC.Offset = pContext->Eip;
    stackFrame.AddrFrame.Offset = pContext->Ebp;
    stackFrame.AddrStack.Offset = pContext->Esp;
#endif
    stackFrame.AddrPC.Mode = AddrModeFlat;
    stackFrame.AddrFrame.Mode = AddrModeFlat;
    stackFrame.AddrStack.Mode = AddrModeFlat;

    // 准备符号信息
    HANDLE hProcess = GetCurrentProcess();
    HANDLE hThread = GetCurrentThread();
    SymInitialize(hProcess, NULL, TRUE);

    while (StackWalk64(
#ifdef _AMD64_
        IMAGE_FILE_MACHINE_AMD64,
#else
        IMAGE_FILE_MACHINE_I386,
#endif
        hProcess, hThread, &stackFrame, pContext, NULL,
        SymFunctionTableAccess64, SymGetModuleBase64, NULL)) {
        // 获取函数名
        SYMBOL_INFO symbol;
        memset(&symbol, 0, sizeof(SYMBOL_INFO));
        symbol.SizeOfStruct = sizeof(SYMBOL_INFO);
        symbol.MaxNameLen = 255;
        if (SymFromAddr(hProcess, stackFrame.AddrPC.Offset, 0, &symbol)) {
            printf("Function: %s\n", symbol.Name);
        }
        // 获取源代码文件和行号
        IMAGEHLP_LINE64 line;
        DWORD displacement; // 用于接收行位移的变量

        // 初始化 IMAGEHLP_LINE64 结构体
        line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
        line.FileName = NULL; // 如果不关心文件名，可以设置为 NULL
        line.LineNumber = 0;   // 初始化行号

        // 调用 SymGetLineFromAddr64 函数
        if (SymGetLineFromAddr64(hProcess, stackFrame.AddrPC.Offset, &displacement, &line)) {
            // 如果成功，打印文件名和行号
            printf("File: %s, Line: %lu\n", line.FileName, line.LineNumber);
        }
        else {
            // 如果失败，打印错误信息
            printf("SymGetLineFromAddr64 failed with error %lu.\n", GetLastError());
        }
    }

    SymCleanup(hProcess);
    return EXCEPTION_EXECUTE_HANDLER;
}

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        printf("Set exception handler.......");
        AddVectoredExceptionHandler(TRUE, VectoredExceptionHandler);
        SetUnhandledExceptionFilter(exceptionFilter);
        // 设置无效参数处理器
        _set_invalid_parameter_handler(MyInvalidParameterHandler);
        break;
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

