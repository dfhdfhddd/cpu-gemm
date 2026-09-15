#pragma once
void gemm_ijk(const float* A, const float* B, float* C, int N);
void gemm_ikj(const float *A, const float *B, float *C, int N);
void gemm_kij(const float *A, const float *B, float *C, int N);

void gemm_blocked_basic_ikj(const float *A, const float *B, float *C, int N);
void gemm_blocked_basic_kij(const float *A, const float *B, float *C, int N);
void gemm_blocked_basic_ikj_ikj(const float *A, const float *B, float *C, int N);
void gemm_blocked_basic_ikj_kij(const float *A, const float *B, float *C, int N);

void gemm_blocked_64x64_4x4(const float *A, const float *B, float *C, int N);
void gemm_blocked_128x128_4x4(const float *A, const float *B, float *C, int N);

// 参数化分块实现：Mc、Nc、Kc 分别控制行块、列块和归约块大小。
void gemm_blocked_4x4_impl(const float *A, const float *B, float *C, int N,
                           int Mc, int Nc, int Kc);
