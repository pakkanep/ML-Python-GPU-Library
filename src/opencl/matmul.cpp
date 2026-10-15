#include <cstdlib>
#include <iostream>
#include <vector>
#include "../backend.h" 

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif


bool opencl_available()
{
    cl_uint num_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0)
        return false;

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    for (cl_platform_id p : platforms)
    {
        cl_uint num_devices = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) == CL_SUCCESS
            && num_devices > 0)
            return true;
    }
    return false;
}

static inline int divup(int a, int b)
{
    return (a + b - 1) / b;
}

static inline int roundup(int a, int b)
{
    return divup(a, b) * b;
}

static inline void check(cl_int err, const char* context)
{
    if (err != CL_SUCCESS)
    {
        std::cerr << "OpenCL error: " << context << ": " << err << std::endl;
        // std::exit(EXIT_FAILURE); // this would kill the entire python process
        throw std::runtime_error(std::string("OpenCL error: ") + context + ": " + std::to_string(err));
    }
}

static inline void check_build(cl_program program, cl_device_id device, cl_int err)
{
    if (err == CL_BUILD_PROGRAM_FAILURE)
    {
        std::cerr << "OpenCL build failed:" << std::endl;
        size_t len;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        char* log = new char[len];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, log, NULL);
        std::cout << log << std::endl;
        delete[] log;
        // std::exit(EXIT_FAILURE);
        throw std::runtime_error(std::string("OpenCL error: ") + ": " + std::to_string(err));
    }
    else if (err != CL_SUCCESS)
    {
        std::cerr << "OpenCL build failed: " << err << std::endl;
        // std::exit(EXIT_FAILURE);
        throw std::runtime_error(std::string("OpenCL error: ") + ": " + std::to_string(err));
    }
}

#define CHECK(x) check(x, #x)

typedef struct
{
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;


const char* matmul_kernel_sourceV1 =
R""(
__kernel void matmulkernel(
            int M,
            int N,
            int K,
            __global const float* M1,
            __global const float* M2,
            __global float* r)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= M || j >= N)
    {
        return;
    }
    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
    {
        sum += M1[K * i + k] * M2[N * k + j];
    }
    r[N * i + j] = sum;
}
)"";

const char* matmul_kernel_sourceV2 =
R""(
__kernel void matmulkernel(
            int M,
            int N,
            int K,
            __global const float* M1,
            __global const float* M2,
            __global float* r)
{
    int j = get_global_id(0);
    int i = get_global_id(1);
    if (i >= M || j >= N)
    {
        return;
    }
    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
    {
        sum += M1[K * i + k] * M2[N * k + j];
    }
    r[N * i + j] = sum;
}
)"";

// MatMul: C (M x N) = A (M x K) * B (K x N)
void opencl_matmul(const float* A, const float* B, float* C, int M, int N, int K)
{
    // M = rows of A = rows of C
    // K = cols of A = rows of B
    // N = cols of B = cols of C

    // ___setup device____
    cl_int err;
    cl_platform_id platform;
    CHECK(clGetPlatformIDs(1, &platform, NULL));

    cl_device_id device;
    CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check(err, "clCreateContext");

#ifdef CL_VERSION_2_0
    cl_command_queue queue
        = clCreateCommandQueueWithProperties(context, device, NULL, &err);
#else
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
#endif
    check(err, "clCreateCommandQueue");

    // ___Compile kernel___
    cl_program program = clCreateProgramWithSource(
        context,
        1,
        &matmul_kernel_sourceV1,
        NULL,
        &err
    );
    check(err, "clCreateProgramWithSource");

    check_build(program, device, clBuildProgram(program, 1, &device, NULL, NULL, NULL));
    cl_kernel kernel = clCreateKernel(program, "matmulkernel", &err);
    check(err, "clCreateKernel");

    // Allocate memory & copy data to GPU
    // ALLOC A
    cl_mem A_GPU = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, 
        M * K * sizeof(float),
        (void*)A,
        &err
    );
    check(err, "clCreateBuffer");

    // ALLOC B
    cl_mem B_GPU = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, 
        K * N * sizeof(float),
        (void*)B,
        &err
    );
    check(err, "clCreateBuffer");

    // ALLOC C
    cl_mem C_GPU = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        M * N * sizeof(float),
        NULL,
        &err
    );
    check(err, "clCreateBuffer");

    // Run kernel
    size_t wlsize[2] = {16, 16};
    size_t wgsize[2] = {
        size_t( roundup(M, wlsize[0]) ),
        size_t( roundup(N, wlsize[1]) )
    };
    
    CHECK(clSetKernelArg(kernel, 0, sizeof(int), &M));
    CHECK(clSetKernelArg(kernel, 1, sizeof(int), &N));
    CHECK(clSetKernelArg(kernel, 2, sizeof(int), &K));
    CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &A_GPU));
    CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &B_GPU));
    CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem), &C_GPU));

    CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, wgsize, wlsize, 0, NULL, NULL));
    CHECK(clFinish(queue));

    // Copy data back to CPU & release memory
    CHECK(clEnqueueReadBuffer(
        queue,
        C_GPU,
        true,
        0,
        M * N * sizeof(float),
        C,
        0,
        NULL,
        NULL)
    );

    CHECK(clReleaseMemObject(A_GPU));
    CHECK(clReleaseMemObject(B_GPU));
    CHECK(clReleaseMemObject(C_GPU));


    // Release everything else
    CHECK(clReleaseKernel(kernel));
    CHECK(clReleaseProgram(program));
    CHECK(clReleaseCommandQueue(queue));
    CHECK(clReleaseContext(context));

}