#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <numeric>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <sstream> // 新增：用于字符串流拼接时间戳

#include "gemm.h"
#include "benchmark_utils.h"

namespace fs = std::filesystem;

// 新增：获取当前时间戳并格式化为 YYYYMMDD_HHMMSS 的函数
std::string get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
    return ss.str();
}

void log_result(std::ofstream& csv_file, const std::string& name, int N, const BenchmarkStats& stats) {
    std::cout << std::left << std::setw(15) << name 
              << "| N=" << std::setw(5) << N 
              << "| GFLOPS: " << std::setw(8) << stats.gflops 
              << "| Median: " << std::setw(8) << stats.median_ms << " ms "
              << "| Min: " << std::setw(8) << stats.min_ms << " ms "
              << "| StdDev: " << std::setw(6) << stats.stddev_ms << " ms "
              << "| Chk: " << stats.checksum << "\n";

    if (csv_file.is_open()) {
        csv_file << name << "," << N << "," << stats.gflops << "," 
                 << stats.median_ms << "," << stats.min_ms << "," 
                 << stats.stddev_ms << "," << stats.checksum << "\n";
    }
}

int main(){
    std::vector<int> test_sizes = {64, 128, 256, 512, 1024};
    int num_iters = 20;

    // 阶段三当前重点比较的四组候选参数。
    // 结构体只保存配置，不复制分块算法；三个微内核复用同一套参数。
    struct BlockConfig {
        int Mc;
        int Nc;
        int Kc;
    };
    const std::vector<BlockConfig> block_configs = {
        {48, 48, 48},
        {48, 48, 128},
        {128, 128, 48},
        {128, 128, 128}
    };

    std::string output_dir = "results";
    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir); 
    }

    // 修改：将固定的文件名替换为带有时间戳的动态文件名
    std::string file_path = output_dir + "/benchmark_" + get_current_timestamp() + ".csv";
    std::ofstream csv_file(file_path);

    if (csv_file.is_open()) {
        csv_file << "Algorithm,N,GFLOPS,Median_ms,Min_ms,StdDev_ms,Checksum\n";
        std::cout << "Data will be saved to: " << file_path << "\n\n"; // 运行时提醒文件存哪了
    } else {
        std::cerr << "Warning: Could not open CSV file for writing.\n";
    }

    std::cout << "Starting GEMM Benchmarks (Iters: " << num_iters << ")...\n";
    std::cout << std::string(120, '-') << "\n";

    for (int N : test_sizes) {
        // 测试第一版：朴素 IJK
        // auto stats_ijk = run_benchmark(gemm_ijk, N, num_iters);
        // log_result(csv_file, "gemm_ijk", N, stats_ijk);

        // 测试第二版：IKJ
        // auto stats_ikj = run_benchmark(gemm_ikj, N, num_iters);
        // log_result(csv_file, "gemm_ikj", N, stats_ikj);

        // 测试第三版：KIJ
        // auto stats_kij = run_benchmark(gemm_kij, N, num_iters);
        // log_result(csv_file, "gemm_kij", N, stats_kij);

        //表面分块
        // auto stats_block_ikj = run_benchmark(gemm_blocked_basic_ikj, N, num_iters);
        // log_result(csv_file, "gemm_blocked_basic_ikj128x128", N, stats_block_ikj);

        // auto stats_block_kij = run_benchmark(gemm_blocked_basic_kij, N, num_iters);
        // log_result(csv_file, "gemm_blocked_basic_kij128x128", N, stats_block_kij);

        // auto stats_block_ikj_ikj = run_benchmark(gemm_blocked_basic_ikj_ikj, N, num_iters);
        // log_result(csv_file, "gemm_blocked_basic_ikj_ikj128x128", N, stats_block_ikj_ikj);

        // auto stats_block_ikj_kij = run_benchmark(gemm_blocked_basic_ikj_kij, N, num_iters);
        // log_result(csv_file, "gemm_blocked_basic_ikj_kij128x128", N, stats_block_ikj_kij);

        // 在相同分块参数下比较 4x4、3x4 和 2x4 微内核。
        for (const BlockConfig& config : block_configs) {
            std::string name4x4 = "gemm_blocked_Mc" + std::to_string(config.Mc)
                + "_Nc" + std::to_string(config.Nc)
                + "_Kc" + std::to_string(config.Kc) + "_4x4";
            auto stats4x4 = run_benchmark_blocked(
                gemm_blocked_4x4_impl, N, num_iters,
                config.Mc, config.Nc, config.Kc);
            log_result(csv_file, name4x4, N, stats4x4);

            std::string name3x4 = "gemm_blocked_Mc" + std::to_string(config.Mc)
                + "_Nc" + std::to_string(config.Nc)
                + "_Kc" + std::to_string(config.Kc) + "_3x4";
            auto stats3x4 = run_benchmark_blocked(
                gemm_blocked_3x4_impl, N, num_iters,
                config.Mc, config.Nc, config.Kc);
            log_result(csv_file, name3x4, N, stats3x4);

            std::string name2x4 = "gemm_blocked_Mc" + std::to_string(config.Mc)
                + "_Nc" + std::to_string(config.Nc)
                + "_Kc" + std::to_string(config.Kc) + "_2x4";
            auto stats2x4 = run_benchmark_blocked(
                gemm_blocked_2x4_impl, N, num_iters,
                config.Mc, config.Nc, config.Kc);
            log_result(csv_file, name2x4, N, stats2x4);
        }
        std::cout << std::string(120, '-') << "\n";


    }

    return 0;
}

// #include <immintrin.h>
// #include <intrin.h>
// #include <iostream>

// void test_fma_throughput() {
//     __m128 a0 = _mm_set_ss(0.1f), a1 = _mm_set_ss(0.2f);
//     __m128 a2 = _mm_set_ss(0.3f), a3 = _mm_set_ss(0.4f);
//     __m128 a4 = _mm_set_ss(0.5f), a5 = _mm_set_ss(0.6f);
//     __m128 a6 = _mm_set_ss(0.7f), a7 = _mm_set_ss(0.8f);
//     __m128 b  = _mm_set_ss(1.0000001f);
//     __m128 c  = _mm_set_ss(1.0000001f);

//     const int iterations = 100000000;

//     // 预热 CPU，触发睿频
//     for (volatile int w = 0; w < 1000000; ++w);

//     _mm_mfence();
//     uint64_t start = __rdtsc();

//     for (int i = 0; i < iterations; ++i) {
//         a0 = _mm_fmadd_ss(b, c, a0);
//         a1 = _mm_fmadd_ss(b, c, a1);
//         a2 = _mm_fmadd_ss(b, c, a2);
//         a3 = _mm_fmadd_ss(b, c, a3);
//         a4 = _mm_fmadd_ss(b, c, a4);
//         a5 = _mm_fmadd_ss(b, c, a5);
//         a6 = _mm_fmadd_ss(b, c, a6);
//         a7 = _mm_fmadd_ss(b, c, a7);
//     }

//     _mm_mfence();
//     uint64_t end = __rdtsc();

//     // 强行使用计算结果，防止死代码消除
//     __m128 sum = _mm_add_ss(_mm_add_ss(_mm_add_ss(a0, a1), _mm_add_ss(a2, a3)),
//                             _mm_add_ss(_mm_add_ss(a4, a5), _mm_add_ss(a6, a7)));
//     float dummy = _mm_cvtss_f32(sum);

//     uint64_t total_cycles = end - start;
//     double cycles_per_loop = (double)total_cycles / iterations;

//     std::cout << "Cycles per loop: " << cycles_per_loop 
//               << " (dummy sink: " << dummy << ")" << std::endl;
// }

// int main() {
//     test_fma_throughput();
//     return 0;
// }
