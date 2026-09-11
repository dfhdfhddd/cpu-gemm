#pragma once
// 1. 定义函数指针别名：统一所有 GEMM 函数的签名
typedef void (*gemm_func_t)(const float*, const float*, float*, int);

// 2. 定义结构体：用于打包返回所有统计信息
struct BenchmarkStats {
    double median_ms;
    double min_ms;
    double mean_ms;
    double stddev_ms;
    double gflops;
    float checksum;
};
BenchmarkStats run_benchmark(gemm_func_t gemm_func, int N, int num_iters = 10);