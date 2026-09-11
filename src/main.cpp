#include<iostream>
#include<random>
#include<vector>
#include<chrono>
#include<numeric>
#include <filesystem>
#include <iomanip>
#include <fstream>

#include "gemm.h"
#include "benchmark_utils.h"

namespace fs = std::filesystem;

void gemm_ijk(const float* A, const float* B, float* C, int N);
BenchmarkStats run_benchmark(gemm_func_t gemm_func, int N, int num_iters);

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

    std::string output_dir = "results";
    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir); // 如果没有 results 文件夹，就自动建一个
    }

    std::string file_path = output_dir + "/benchmark_results.csv";
    std::ofstream csv_file(file_path);

    if (csv_file.is_open()) {
        csv_file << "Algorithm,N,GFLOPS,Median_ms,Min_ms,StdDev_ms,Checksum\n";
    } else {
        std::cerr << "Warning: Could not open CSV file for writing.\n";
    }

    std::cout << "Starting GEMM Benchmarks (Iters: " << num_iters << ")...\n";
    std::cout << std::string(120, '-') << "\n";


    for (int N : test_sizes) {
        // 测试第一版：朴素 IJK
        auto stats_ijk = run_benchmark(gemm_ijk, N, num_iters);
        log_result(csv_file, "gemm_ijk", N, stats_ijk);

        // 未来测试第二版：只需加两行！
        // auto stats_ikj = run_benchmark(gemm_ikj, N, num_iters);
        // log_result("gemm_ikj", N, stats_ikj);
        
        std::cout << std::string(120, '-') << "\n";
    }

    return 0;
}
