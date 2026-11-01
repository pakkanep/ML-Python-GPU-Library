#define BLOCK_SIZE 16

typedef struct
{
    int width;
    int height;
    int stride;
    __global float* elements;
} Matrix;

inline float GetElement(const Matrix A, int row, int col)
{
    return A.elements[A.stride * row + col];
}

inline void SetElement(Matrix A, int row, int col, float value)
{
    A.elements[A.stride * row + col] = value;
}

inline Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix sub;
    sub.width    = BLOCK_SIZE;
    sub.height   = BLOCK_SIZE;
    sub.stride   = A.stride;
    sub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return sub;
}

__kernel void matmul_opencl(
    const int M,
    const int N,
    const int K,
    __global const float* A,
    __global const float* B,
    __global float* C
)
{
    const int j = get_global_id(0);
    const int i = get_global_id(1);

    if (i >= M || j >= N)
    {
        return;
    }
    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
    {
        sum += A[K * i + k] * B[N * k + j];
    }
    C[N * i + j] = sum;
}