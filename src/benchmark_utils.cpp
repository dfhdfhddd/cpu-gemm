#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <string>
#include "benchmark_utils.h"

// 1. 定义函数指针别名：统一所有 GEMM 函数的签名
typedef void (*gemm_func_t)(const float*, const float*, float*, int);

// 3. 核心测速引擎
BenchmarkStats run_benchmark(gemm_func_t gemm_func, int N, int num_iters) {
    // 【步骤 1】A/B/C 分配与随机数生成
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C(N * N, 0.0f);

    std::mt19937 rng(12345); // 固定种子，确保每次基准测试输入的数据相同
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N * N; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    // 【步骤 2】预热 (Warm-up) - 唤醒 CPU 缓存并解除降频，不计入成绩
    for (int i = 0; i < 3; i++)
        gemm_func(A.data(), B.data(), C.data(), N);

    // 【步骤 3】重复计时
    std::vector<double> times_ms(num_iters);
    for (int iter = 0; iter < num_iters; ++iter) {
        // 注意：清零矩阵 C 必须放在计时器外面！防止干扰纯计算耗时
        std::fill(C.begin(), C.end(), 0.0f);

        auto start = std::chrono::high_resolution_clock::now();
        
        gemm_func(A.data(), B.data(), C.data(), N); // 真正的测试目标
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end - start;
        times_ms[iter] = diff.count();
    }

    // 【步骤 4】统计分析 (排序以获取中位数和最小值)
    std::sort(times_ms.begin(), times_ms.end());
    
    double min_ms = times_ms.front();
    
    // 中位数计算
    double median_ms = 0.0;
    if (num_iters % 2 == 0) {
        median_ms = (times_ms[num_iters / 2 - 1] + times_ms[num_iters / 2]) / 2.0;
    } else {
        median_ms = times_ms[num_iters / 2];
    }

    // 平均值计算
    double sum_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
    double mean_ms = sum_ms / num_iters;

    // 标准差计算
    double sq_sum = 0.0;
    for (double t : times_ms) {
        sq_sum += (t - mean_ms) * (t - mean_ms);
    }
    double stddev_ms = std::sqrt(sq_sum / num_iters);

    // 【步骤 5】GFLOPS 计算
    // 矩阵乘法浮点操作数 = 2 * N^3
    // 业界标准：基于【中位数】耗时来计算代表性 GFLOPS
    double seconds = median_ms / 1000.0; 
    double flops = 2.0 * static_cast<double>(N) * N * N;
    double gflops = (flops / seconds) / 1e9;

    // 【步骤 6】Checksum 计算 (放在计时区间外，防死代码优化)
    float checksum = 0.0f;
    for (float v : C) {
        checksum += v;
    }

    return {median_ms, min_ms, mean_ms, stddev_ms, gflops, checksum};
}