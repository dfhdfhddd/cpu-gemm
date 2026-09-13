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

void gemm_ijk(const float* A, const float* B, float* C, int N);
void gemm_ikj(const float *A, const float *B, float *C, int N);
void gemm_kij(const float *A, const float *B, float *C, int N);

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
        auto stats_ijk = run_benchmark(gemm_ijk, N, num_iters);
        log_result(csv_file, "gemm_ijk", N, stats_ijk);

        // 测试第二版：IKJ
        auto stats_ikj = run_benchmark(gemm_ikj, N, num_iters);
        log_result(csv_file, "gemm_ikj", N, stats_ikj);
        
        // 测试第三版：KIJ
        auto stats_kij = run_benchmark(gemm_kij, N, num_iters);
        log_result(csv_file, "gemm_kij", N, stats_kij);

        std::cout << std::string(120, '-') << "\n";
    }

    return 0;
}