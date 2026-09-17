#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <cstdlib> // std::exit

#include "gemm.h"

// 浮点数比对辅助函数
bool check_equal(const std::vector<float>& actual, const std::vector<float>& expected, float eps = 1e-4f) {
    if (actual.size() != expected.size()) return false;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > eps) {
            std::cerr << "Mismatch at index " << i << ": expected " 
                      << expected[i] << ", got " << actual[i] << std::endl;
            return false;
        }
    }
    return true;
}

// 1. N = 1 标量乘法
void test_scalar() {
    std::vector<float> A = {3.5f};
    std::vector<float> B = {2.0f};
    std::vector<float> C = {0.0f};
    
    gemm_ijk(A.data(), B.data(), C.data(), 1);
    
    if (!check_equal(C, {7.0f})) {
        std::cerr << "[FAIL] Case 1: N=1 Scalar Multiplication Failed!" << std::endl;
        std::exit(1);
    }
    std::cout << "[PASS] Case 1: N=1 Scalar Multiplication" << std::endl;
}

// 2. 全 1 矩阵: 每个元素是 N 个 1 相乘后累加，理论结果全为 N
void test_all_ones() {
    int N = 4;
    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 1.0f);
    std::vector<float> C(N * N, 0.0f);
    std::vector<float> expected(N * N, static_cast<float>(N));
    
    gemm_ijk(A.data(), B.data(), C.data(), N);
    
    if (!check_equal(C, expected)) {
        std::cerr << "[FAIL] Case 2: All-ones Matrix Failed!" << std::endl;
        std::exit(1);
    }
    std::cout << "[PASS] Case 2: All-ones Matrix (N=4)" << std::endl;
}

// 3. 单位矩阵 I: A * I = A
void test_identity() {
    int N = 3;
    std::vector<float> A = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    std::vector<float> I(N * N, 0.0f);
    for (int i = 0; i < N; ++i) I[i * N + i] = 1.0f; // 对角线置 1

    std::vector<float> C(N * N, 0.0f);
    gemm_ijk(A.data(), I.data(), C.data(), N);
    
    if (!check_equal(C, A)) {
        std::cerr << "[FAIL] Case 3: Identity Matrix Failed!" << std::endl;
        std::exit(1);
    }
    std::cout << "[PASS] Case 3: Identity Matrix (A * I = A)" << std::endl;
}

// 4. N = 2 手算矩阵
void test_hand_calculated() {
    std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> B = {2.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> C(4, 0.0f);
    std::vector<float> expected = {4.0f, 4.0f, 10.0f, 8.0f};

    gemm_ijk(A.data(), B.data(), C.data(), 2);
    
    if (!check_equal(C, expected)) {
        std::cerr << "[FAIL] Case 4: N=2 Hand-calculated Matrix Failed!" << std::endl;
        std::exit(1);
    }
    std::cout << "[PASS] Case 4: N=2 Hand-calculated Matrix" << std::endl;
}

// 5. 固定种子随机矩阵
void test_random_matrix() {
    int N = 8;
    std::vector<float> A(N * N), B(N * N), C(N * N, 0.0f);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N * N; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    gemm_ijk(A.data(), B.data(), C.data(), N);
    
    for (float v : C) {
        if (std::isnan(v) || std::isinf(v)) {
            std::cerr << "[FAIL] Case 5: Detected NaN or Inf in output!" << std::endl;
            std::exit(1);
        }
    }
    std::cout << "[PASS] Case 5: Fixed Random Matrix (N=8)" << std::endl;
}

// 6. 逐元素比对
void verify_matrix(const std::vector<float>& C_test, const std::vector<float>& C_ref, int N) {
    float max_abs_diff = 0.0f;
    const float eps = 1e-4f;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            float diff = std::fabs(C_test[idx] - C_ref[idx]);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
            }
            if (diff > eps) {
                std::cerr << "[FAIL] Verification failed at (" << i << ", " << j << "): "
                          << "expected " << C_ref[idx] << ", got " << C_test[idx] 
                          << ", diff: " << diff << std::endl;
                std::exit(1);
            }
        }
    }
    std::cout << "[PASS] Case 6: Element-wise Reference Verification (Max Diff: " 
              << max_abs_diff << ")" << std::endl;
}

void test_elementwise_reference(int N) {
    std::vector<float> A(N * N), B(N * N), C(N * N, 0.0f), C_ref(N * N, 0.0f);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < N * N; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    // 独立计算 Ground Truth
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C_ref[i * N + j] = sum;
        }
    }

    gemm_blocked_64x64_4x4(A.data(), B.data(), C.data(), N);
    verify_matrix(C, C_ref, N);

    std::fill(C.begin(), C.end(), 0.0f);
    gemm_blocked_2x4_impl(A.data(), B.data(), C.data(), N, 64, 64, 64);
    verify_matrix(C, C_ref, N);

    std::fill(C.begin(), C.end(), 0.0f);
    gemm_blocked_3x4_impl(A.data(), B.data(), C.data(), N, 64, 64, 64);
    verify_matrix(C, C_ref, N);

    std::fill(C.begin(), C.end(), 0.0f);
    gemm_blocked_128x128_4x4(A.data(), B.data(), C.data(), N);
    verify_matrix(C, C_ref, N);

    // N=130 专门验证 32、64、128 三种块大小的全部组合。
    if (N == 130) {
        const std::vector<int> block_sizes = {32, 64, 128};
        for (int Mc : block_sizes) {
            for (int Nc : block_sizes) {
                for (int Kc : block_sizes) {
                    std::fill(C.begin(), C.end(), 0.0f);
                    gemm_blocked_4x4_impl(A.data(), B.data(), C.data(), N, Mc, Nc, Kc);
                    verify_matrix(C, C_ref, N);

                    std::fill(C.begin(), C.end(), 0.0f);
                    gemm_blocked_2x4_impl(A.data(), B.data(), C.data(), N, Mc, Nc, Kc);
                    verify_matrix(C, C_ref, N);

                    std::fill(C.begin(), C.end(), 0.0f);
                    gemm_blocked_3x4_impl(A.data(), B.data(), C.data(), N, Mc, Nc, Kc);
                    verify_matrix(C, C_ref, N);
                }
            }
        }
        std::cout << "[PASS] All 27 Mc/Nc/Kc combinations for 4x4, 3x4 and 2x4 (N=130)" << std::endl;
    }

}

int main() {
    test_scalar();
    test_all_ones();
    test_identity();
    test_hand_calculated();
    test_random_matrix();
    std::vector<int> q = {1, 3, 5, 100, 127, 128, 130, 256};
    for(auto i:q)
        test_elementwise_reference(i);

    std::cout << "\nAll test cases PASSED! Ready for benchmark." << std::endl;
    return 0;
}
