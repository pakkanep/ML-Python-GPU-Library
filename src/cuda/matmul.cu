#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <chrono>

#define BLOCK_SIZE 16

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


__global__ void matMulCu(int ny, int nx, const float *M1, const float *M2, float *result)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i >= ny || j >= nx)
    {
        return;
    }

    float sum = 0.0f;

    for (int k = 0; k < nx; ++k)
    {
        sum += M1[nx * i + k] * M1[nx * j + k];
    }

    result[nx * i + j] = sum;
}

void gpuSetupMatMulRegular(const int ny, const int nx, const float* M1, const float* M2, float* result)
{
    // 1. Allocate memory & copy data to GPU
    float* A_GPU = NULL;
    float* B_GPU = NULL;
    float* C_GPU = NULL;

    CHECK( cudaMalloc((void**)&A_GPU, ny * nx * sizeof(float)) );
    CHECK( cudaMalloc((void**)&B_GPU, ny * nx * sizeof(float)) );
    CHECK( cudaMalloc((void**)&C_GPU, ny * nx * sizeof(float)) );

    // 2. define dimensions and launch kernel
    // 3. copy data back to host (cpu) AND release memory
    CHECK(cudaFree(A_GPU));
    CHECK(cudaFree(B_GPU));
    CHECK(cudaFree(C_GPU));

}


void step(float* r, const float* d, int n) {
    // Allocate memory & copy data to GPU
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, n * n * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, n * n * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, d, n * n * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(n, dimBlock.x), divup(n, dimBlock.y));
    mykernel<<<dimGrid, dimBlock>>>(rGPU, dGPU, n);
    CHECK(cudaGetLastError());

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(r, rGPU, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
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


// int main()
// {
//     int ny = 10;
//     int nx = 10;
//     std::vector<float> M1(ny * nx, 0.0f);
//     std::vector<float> M2(ny * nx, 0.0f);
//     std::vector<float> result(ny * ny, 0.0f);
//     gpuSetupMatMulBlocked(ny, nx, M1.data(), M2.data(), result.data());
// }
