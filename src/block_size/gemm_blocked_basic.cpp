#include "gemm.h"
#include <algorithm>
const int block = 128;

void gemm_blocked_basic_ikj(const float* A, const float* B, float* C, int N){
    for(int i=0;i<N*N;i++){
        C[i] = 0.0f;
    }
    for(int i=0;i<N;i+=block){
        for(int j=0;j<N;j+=block){
            for (int k = 0; k < N;k+=block){
                int mini = std::min(i + block, N);
                int minj = std::min(j + block, N);
                int mink = std::min(k + block, N);
                for (int ii = i; ii < mini;ii++){
                    for (int kk = k; kk < mink;kk++){
                        float sum = A[ii * N + kk];
                        for (int jj = j; jj < minj;jj++){
                            C[ii * N + jj] += sum * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
}

void gemm_blocked_basic_kij(const float* A, const float* B, float* C, int N){
    for(int i=0;i<N*N;i++){
        C[i] = 0.0f;
    }
    for(int i=0;i<N;i+=block){
        for(int j=0;j<N;j+=block){
            for (int k = 0; k < N;k+=block){
                int mini = std::min(i + block, N);
                int minj = std::min(j + block, N);
                int mink = std::min(k + block, N);
                for (int kk = k; kk < mink;kk++){
                    for (int ii = i; ii < mini;ii++){
                        float sum = A[ii * N + kk];
                        for (int jj = j; jj < minj;jj++){
                            C[ii * N + jj] += sum * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
}

void gemm_blocked_basic_ikj_ikj(const float* A, const float* B, float* C, int N){
    for(int i=0;i<N*N;i++){
        C[i] = 0.0f;
    }
    for(int i=0;i<N;i+=block){
        for(int k=0;k<N;k+=block){
            for (int j = 0; j < N;j+=block){
                int mini = std::min(i + block, N);
                int minj = std::min(j + block, N);
                int mink = std::min(k + block, N);
                for (int ii = i; ii < mini;ii++){
                    for (int kk = k; kk < mink;kk++){
                        float sum = A[ii * N + kk];
                        for (int jj = j; jj < minj;jj++){
                            C[ii * N + jj] += sum * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
}

void gemm_blocked_basic_ikj_kij(const float* A, const float* B, float* C, int N){
    for(int i=0;i<N*N;i++){
        C[i] = 0.0f;
    }
    for(int k=0;k<N;k+=block){
        for(int i=0;i<N;i+=block){
            for (int j = 0; j < N;j+=block){
                int mini = std::min(i + block, N);
                int minj = std::min(j + block, N);
                int mink = std::min(k + block, N);
                for (int ii = i; ii < mini;ii++){
                    for (int kk = k; kk < mink;kk++){
                        float sum = A[ii * N + kk];
                        for (int jj = j; jj < minj;jj++){
                            C[ii * N + jj] += sum * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
}