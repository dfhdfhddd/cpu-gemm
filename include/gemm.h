#pragma once
void gemm_ijk(const float* A, const float* B, float* C, int N);
void gemm_ikj(const float *A, const float *B, float *C, int N);
void gemm_kij(const float *A, const float *B, float *C, int N);
