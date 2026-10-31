#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <chrono>

#define BLOCK_SIZE 16


bool cuda_available()
{
    int num_devices = 0;

    cudaError_t err = cudaGetDeviceCount(&num_devices);

    if (err != cudaSuccess)
    {
        cudaGetLastError();
        return false;
    }

    return num_devices > 0;
}

namespace cuHelpers
{

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}

static inline void check(cudaError_t err, const char* context)
{
    if (err != cudaSuccess) 
    {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)
} // namespace helpers


typedef struct
{
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;


__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[A.stride * row + col];
}

__device__ void SetElement(Matrix A, int row, int col, float value)
{
    A.elements[A.stride * row + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix subMat;
    subMat.width    = BLOCK_SIZE;
    subMat.height   = BLOCK_SIZE;
    subMat.stride   = A.stride;
    subMat.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];

    return subMat;
}

template<typename T>
void printResult(T *r, const int ny, const int nx)
{
    std::cout << std::fixed << std::setprecision(1);

    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j < nx; ++j)
        {
            std::cout << std::setw(8) << r[nx * i + j];
        }
        std::cout << '\n';
    }
        std::cout << '\n';
}


template <typename T>
__global__ void matMulCu(int M, int N, int K, const T *M1, const T *M2, T *result)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i >= M || j >= N)
    {
        return;
    }

    T sum = {};

    for (int k = 0; k < K; ++k)
    {
        sum += M1[K * i + k] * M2[N * j + k];
    }

    result[N * i + j] = sum;
}

template<typename T>
void cuda_matmul(const T* M1, const T* M2, T* result, int M, int N, int K)
{
    // 1. Allocate memory & copy data to GPU
    T* A_GPU = NULL;
    T* B_GPU = NULL;
    T* C_GPU = NULL;

    cuHelpers::CHECK( cudaMalloc((void**)&A_GPU, M * N * sizeof(T)) );
    cuHelpers::CHECK( cudaMalloc((void**)&B_GPU, N * K * sizeof(T)) );
    cuHelpers::CHECK( cudaMalloc((void**)&C_GPU, M * K * sizeof(T)) );

    cuHelpers::CHECK(cudaMemcpy(A_GPU, M1, M * N * sizeof(T), cudaMemcpyHostToDevice));
    cuHelpers::CHECK(cudaMemcpy(B_GPU, M2, N * K * sizeof(T), cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(C_GPU, M1, ny * nx * sizeof(float), cudaMemcpyHostToDevice));

    // 2. define dimensions and launch kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(
        cuHelpers::divup(M, dimBlock.x),
        cuHelpers::divup(N, dimBlock.y)
    );
    
    matMulCu<T><<<dimBlock, dimGrid>>>(M, N, K, A_GPU, B_GPU, C_GPU);
    cuHelpers::CHECK(cudaGetLastError());

    
    // 3. copy data back to host AND release memory
    cuHelpers::CHECK(cudaMemcpy(result, C_GPU, M * K * sizeof(T), cudaMemcpyDeviceToHost));

    cuHelpers::CHECK(cudaFree(A_GPU));
    cuHelpers::CHECK(cudaFree(B_GPU));
    cuHelpers::CHECK(cudaFree(C_GPU));

}


__global__ void matMulCuBlocked(const Matrix M1, const Matrix M2, Matrix result)
{

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int row = threadIdx.y;
    int col = threadIdx.x;
    
    Matrix subResult = GetSubMatrix(result, blockRow, blockCol);
    
    float subSum = 0.0f;

    for (int i = 0; i < (M1.width / BLOCK_SIZE); ++i)
    {
        Matrix M1Sub = GetSubMatrix(M1, blockRow, i);
        Matrix M2Sub = GetSubMatrix(M2, i, blockCol);

        __shared__ float M1SubShared[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float M2SubShared[BLOCK_SIZE][BLOCK_SIZE];
    
        M1SubShared[row][col] = GetElement(M1Sub, row, col);
        M2SubShared[row][col] = GetElement(M2Sub, row, col);

        __syncthreads();
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
            subSum += M1SubShared[row][j] * M2SubShared[j][col];
        }
        __syncthreads();
    }

    SetElement(subResult, row, col, subSum);
}

void gpuSetupMatMulBlocked(const int ny, const int nx, const float* M1, const float* M2, float* result)
{
    int fullRows = BLOCK_SIZE * cuHelpers::divup(ny, BLOCK_SIZE);
    int fullCols = BLOCK_SIZE * cuHelpers::divup(nx, BLOCK_SIZE);

    // Allocate memory & copy data to GPU
    float* M1GPU = NULL;
    float* M2GPU = NULL;

    cuHelpers::CHECK( cudaMalloc( (void**)&M1GPU, fullRows * fullCols * sizeof( float ) ) );
    cuHelpers::CHECK( cudaMalloc( (void**)&M2GPU, fullRows * fullCols * sizeof( float ) ) );

    cuHelpers::CHECK( cudaMemset( M1GPU, 0, fullRows * fullCols * sizeof( float ) ) );
    cuHelpers::CHECK( cudaMemset( M2GPU, 0, fullRows * fullCols * sizeof( float ) ) );
    
    cuHelpers::CHECK( 
        cudaMemcpy2D( 
            M1GPU,
            fullCols * sizeof( float ),
            M1,
            nx * sizeof( float ),
            nx * sizeof( float ),
            ny,
            cudaMemcpyHostToDevice 
        ) 
    );

    cuHelpers::CHECK( 
        cudaMemcpy2D( 
            M2GPU,
            fullCols * sizeof( float ),
            M2,
            nx * sizeof( float ),
            nx * sizeof( float ),
            ny,
            cudaMemcpyHostToDevice 
        ) 
    );


    float* resultGPU = NULL;
    cuHelpers::CHECK( cudaMalloc( ( void** )&resultGPU, fullRows * fullRows * sizeof( float ) ) );
    cuHelpers::CHECK( cudaMemset( resultGPU, 0,         fullRows * fullRows * sizeof( float ) ) );


    Matrix M1_d;
    M1_d.elements = M1GPU;
    M1_d.height   = fullRows;
    M1_d.width    = fullCols;
    M1_d.stride   = fullCols;

    Matrix M2_d;
    M2_d.elements = M2GPU;
    M2_d.height   = fullRows;
    M2_d.width    = fullCols;
    M2_d.stride   = fullCols;

    Matrix R_d;
    R_d.elements = resultGPU;
    R_d.height   = fullRows;
    R_d.width    = fullRows;
    R_d.stride   = fullRows;

    // define dimensions and launch kernel
    dim3 dimBlock(
        BLOCK_SIZE, 
        BLOCK_SIZE
    );
    
    dim3 dimGrid(
        cuHelpers::divup( fullRows, dimBlock.x ), 
        cuHelpers::divup( fullRows, dimBlock.y ) 
    );
    
    
    matMulCuBlocked<<<dimGrid, dimBlock>>>(M1_d, M2_d, R_d);
    cuHelpers::CHECK( cudaGetLastError() );

    // Copy data back to CPU & release memory

    cuHelpers::CHECK( 
        cudaMemcpy2D( 
            result,
            ny * sizeof( float ),
            resultGPU,
            fullRows * sizeof( float ),
            ny * sizeof( float ),
            ny,
            cudaMemcpyDeviceToHost 
        ) 
    );
    
    cuHelpers::CHECK(cudaFree(M1GPU));
    cuHelpers::CHECK(cudaFree(M2GPU));
    cuHelpers::CHECK(cudaFree(resultGPU));
}

/*
enum class DType { F32, F64, I32 };

template <typename T>
void launch_add(const void* a, const void* b, void* out, size_t n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<T><<<blocks, threads>>>(
        static_cast<const T*>(a), static_cast<const T*>(b),
        static_cast<T*>(out), n);
}

void add(DType dt, const void* a, const void* b, void* out, size_t n) {
    switch (dt) {
        case DType::F32: launch_add<float>(a, b, out, n);   break;
        case DType::F64: launch_add<double>(a, b, out, n);  break;
        case DType::I32: launch_add<int32_t>(a, b, out, n); break;
        default: throw std::runtime_error("unsupported dtype");
    }
}
*/

template void cuda_matmul<float>(
    const float*,
    const float*,
    float*,
    int,
    int,
    int
);