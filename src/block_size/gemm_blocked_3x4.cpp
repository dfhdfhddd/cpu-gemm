#include "gemm.h"
#include <algorithm>

// 3x4 微内核：计算 C 的三行、四列，并覆盖 [k_start, k_end) 归约区间。
inline void core3x4_optimized(const float *A, const float *B, float *C,
                              int row, int col, int k_start, int k_end, int N){
    // 1. 初始化 12 个累加器，准备长期驻留
    float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
    float c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;
    float c20 = 0.0f, c21 = 0.0f, c22 = 0.0f, c23 = 0.0f;
    // float c30 = 0.0f, c31 = 0.0f, c32 = 0.0f, c33 = 0.0f;
    
    int n1 = row * N + k_start;
    int n2 = k_start * N + col;
    
    // 2. 核心大循环：把归约维度的遍历直接放在寄存器计算中
    for (int k = k_start; k < k_end; ++k){
        float a0 = A[n1];
        float a1 = A[n1 + N];
        float a2 = A[n1 + N * 2];
        // float a3 = A[n1 + N * 3];
        n1++;

        // 每次只保留一个 B 元素：用它依次更新三行，再处理下一列。
        // 这样 B 只占用一个临时寄存器，避免同时保留 b0~b3。
        float b = B[n2];
        c00 += a0 * b; c10 += a1 * b; c20 += a2 * b;

        b = B[n2 + 1];
        c01 += a0 * b; c11 += a1 * b; c21 += a2 * b;

        b = B[n2 + 2];
        c02 += a0 * b; c12 += a1 * b; c22 += a2 * b;

        b = B[n2 + 3];
        c03 += a0 * b; c13 += a1 * b; c23 += a2 * b;
        n2 += N;
        // 第四行不属于当前 3x4 微内核。
    }
    
    // 3. 循环彻底跑完（比如跑了 64 步），再进行唯一的一次内存写入！
    int n3 = row * N + col;
    C[n3] += c00, C[n3 + 1] += c01, C[n3 + 2] += c02, C[n3 + 3] += c03;
    C[n3 + N] += c10, C[n3 + 1 + N] += c11, C[n3 + 2 + N] += c12, C[n3 + 3 + N] += c13;
    C[n3 + N * 2] += c20, C[n3 + 1 + N * 2] += c21, C[n3 + 2 + N * 2] += c22, C[n3 + 3 + N * 2] += c23;
    // C[n3 + N * 3] += c30, C[n3 + 1 + N * 3] += c31, C[n3 + 2 + N * 3] += c32, C[n3 + 3 + N * 3] += c33;
}

// 分块实现的参数化内核：Mc、Nc、Kc 分别表示行块、列块和归约块大小。
void gemm_blocked_3x4_impl(const float *A, const float *B, float *C, int N, int Mc, int Nc, int Kc){
    for(int i=0;i<N*N;i++){
        C[i] = 0.0f;
    }

    for(int i=0; i<N; i+=Mc){
        int i_end = std::min(i + Mc, N);
        for(int col=0; col<N; col+=Nc){
            int col_end = std::min(col + Nc, N);
            for(int red=0; red<N; red+=Kc){
                int red_end = std::min(red + Kc, N);
                
                // 1. 计算能被 3 整除的安全主体边界
                int i_main = i + ((i_end - i) / 3) * 3;
                int col_main = col + ((col_end - col) / 4) * 4;

                // 2. 满血输出区：跑 3x4 微内核
                for (int ii = i; ii < i_main; ii+=3){
                    for (int jj = col; jj < col_main; jj+=4){
                        core3x4_optimized(A, B, C, ii, jj, red, red_end, N);
                    }
                }
                
                // 3. 扫尾区 A：处理右侧残缺列（行数满足主体，列数不够 4）
                for (int ii = i; ii < i_main; ++ii) {
                    for (int kk = red; kk < red_end; ++kk) {
                        float a_val = A[ii * N + kk];
                        for (int jj = col_main; jj < col_end; ++jj) {
                            C[ii * N + jj] += a_val * B[kk * N + jj];
                        }
                    }
                }
                
                // 4. 扫尾区 B：处理底部残缺行（包含右下角残缺部分）
                for (int ii = i_main; ii < i_end; ++ii) {
                    for (int kk = red; kk < red_end; ++kk) {
                        float a_val = A[ii * N + kk];
                        for (int jj = col; jj < col_end; ++jj) {
                            C[ii * N + jj] += a_val * B[kk * N + jj];
                        }
                    }
                }
                
            }
        }
    }
}

// 下面两个函数保持和其他 GEMM 实现相同的四参数接口，
// 这样可以直接传给 run_benchmark。
void gemm_blocked_64x64_3x4(const float* A, const float* B, float* C, int N){
    gemm_blocked_3x4_impl(A, B, C, N, 64, 64, 64);
}

void gemm_blocked_128x128_3x4(const float* A, const float* B, float* C, int N){
    gemm_blocked_3x4_impl(A, B, C, N, 128, 128, 128);
}
