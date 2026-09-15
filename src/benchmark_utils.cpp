#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <string>
#include <functional>
#include "benchmark_utils.h"

namespace {

// 普通 GEMM 和参数化分块 GEMM 共用同一套计时与统计流程。
template <typename Call>
BenchmarkStats run_benchmark_common(int N, int num_iters, Call call) {
    // 【步骤 1】A/B/C 分配与随机数生成
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C(N * N, 0.0f);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N * N; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    // 【步骤 2】预热
    for (int i = 0; i < 3; i++)
        call(A.data(), B.data(), C.data(), N);

    // 【步骤 3】重复计时
    std::vector<double> times_ms(num_iters);
    for (int iter = 0; iter < num_iters; ++iter) {
        std::fill(C.begin(), C.end(), 0.0f);

        auto start = std::chrono::high_resolution_clock::now();
        call(A.data(), B.data(), C.data(), N);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> diff = end - start;
        times_ms[iter] = diff.count();
    }

    // 【步骤 4】统计分析
    std::sort(times_ms.begin(), times_ms.end());
    double min_ms = times_ms.front();
    double median_ms = 0.0;
    if (num_iters % 2 == 0) {
        median_ms = (times_ms[num_iters / 2 - 1] + times_ms[num_iters / 2]) / 2.0;
    } else {
        median_ms = times_ms[num_iters / 2];
    }

    double sum_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
    double mean_ms = sum_ms / num_iters;
    double sq_sum = 0.0;
    for (double t : times_ms) {
        sq_sum += (t - mean_ms) * (t - mean_ms);
    }
    double stddev_ms = std::sqrt(sq_sum / num_iters);

    // 【步骤 5】GFLOPS 和 checksum
    double seconds = median_ms / 1000.0;
    double flops = 2.0 * static_cast<double>(N) * N * N;
    double gflops = (flops / seconds) / 1e9;

    float checksum = 0.0f;
    for (float v : C) {
        checksum += v;
    }

    return {median_ms, min_ms, mean_ms, stddev_ms, gflops, checksum};
}

}

BenchmarkStats run_benchmark(gemm_func_t gemm_func, int N, int num_iters) {
    return run_benchmark_common(N, num_iters,
        [gemm_func](const float* A, const float* B, float* C, int size) {
            gemm_func(A, B, C, size);
        });
}

BenchmarkStats run_benchmark_blocked(gemm_blocked_func_t gemm_func, int N, int num_iters,
                                      int Mc, int Nc, int Kc) {
    return run_benchmark_common(N, num_iters,
        [gemm_func, Mc, Nc, Kc](const float* A, const float* B, float* C, int size) {
            gemm_func(A, B, C, size, Mc, Nc, Kc);
        });
}
