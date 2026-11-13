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