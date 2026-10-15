#pragma once

// All take host pointers, row-major: C[M×N] = A[M×K] * B[K×N]
#ifdef HAVE_CUDA
bool cuda_available();
void cuda_matmul(const float* A, const float* B, float* C, int M, int N, int K);
#endif

#ifdef HAVE_OPENCL
bool opencl_available();
void opencl_matmul(const float* A, const float* B, float* C, int M, int N, int K);
#endif