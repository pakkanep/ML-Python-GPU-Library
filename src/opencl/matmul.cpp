#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include "../backend.h" 

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/opencl.hpp"
#endif


/*
TODOS:
openCL available does the same thing as pick GPU
move CL runtime code in own header and source files 
move device matrix code in own header and source files
*/

static inline int divup(int a, int b)
{
    return (a + b - 1)/b;
}

static inline int roundup(int a, int b)
{
    return divup(a, b) * b;
}

constexpr int TILE = 16;

static std::string readFile(const std::string& path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("could not open " + path);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

class ClRuntime
{
public:
    static ClRuntime& get()
    {
        static ClRuntime instance;
        return instance;
    }

    ClRuntime(const ClRuntime&) = delete;
    ClRuntime& operator=(const ClRuntime&) = delete;

    static cl::Device pickGpu()
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        for (auto& p : platforms)
        {
            std::vector<cl::Device> devs;

            try
            {
                p.getDevices(CL_DEVICE_TYPE_GPU, &devs);
            }
            catch (const cl::Error&)
            {
                continue;
            }

            if (!devs.empty())
                return devs.front();
        }

        throw std::runtime_error("No OpenCL GPU found");
    }

    cl::Context& context() { return context_; }
    cl::CommandQueue& queue() { return queue_; }

    cl::Kernel& kernel(const std::string& name)
    {
        auto it = kernels_.find(name);

        if (it == kernels_.end())
            throw std::runtime_error("Kernel not found: " + name);

        return it->second;
    }

private:
    ClRuntime()
    {
        device_ = pickGpu();

        context_ = cl::Context(device_);

        queue_ = cl::CommandQueue(
            context_,
            device_,
            CL_QUEUE_PROFILING_ENABLE
        );

        std::string src;

        src += readFile(OPENCL_KERNEL_DIR "/matmul.cl");
        src += "\n";
        src += readFile(OPENCL_KERNEL_DIR "/blocked_matmul.cl");


        program_ = cl::Program(context_, src);

        try
        {
            std::string options =
                "-DTILE=" + std::to_string(TILE);

            program_.build(options.c_str());
        }
        catch (const cl::Error&)
        {
            std::cerr
                << program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_)
                << "\n";

            throw;
        }

        kernels_["matmul_opencl"] = cl::Kernel(program_, "matmul_opencl");
        kernels_["matrixMulKernel"] = cl::Kernel(program_, "matrixMulKernel");
    }

    cl::Device device_;
    cl::Context context_;
    cl::CommandQueue queue_;
    cl::Program program_;

    std::unordered_map<std::string, cl::Kernel> kernels_;
};


class DeviceMatrix
{
public:
    DeviceMatrix(int rows, int cols)
        : rows_(rows), cols_(cols),
          buf_(ClRuntime::get().context(), CL_MEM_READ_WRITE, sizeof(float) * rows * cols) {}

    void upload(const float* src)
    {
        auto& queue = ClRuntime::get().queue();

        queue.enqueueWriteBuffer(
            buf_,
            CL_TRUE,
            0,
            sizeof(float) * rows_ * cols_,
            src
        );
    }
    
    void download(float* dst) const
    {
        auto& queue = ClRuntime::get().queue();

        queue.enqueueReadBuffer(
            buf_,
            CL_TRUE,
            0,
            sizeof(float) * rows_ * cols_,
            dst
        );
    }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    const cl::Buffer& buffer() const { return buf_; }

private:
    int rows_, cols_;
    cl::Buffer buf_;
};



void matmul(const DeviceMatrix& A, const DeviceMatrix& B, DeviceMatrix& C)
{
    if (A.cols() != B.rows() || C.rows() != A.rows() || C.cols() != B.cols())
        throw std::invalid_argument("matmul: shape mismatch");

    const int M = A.rows(), K = A.cols(), N = B.cols();

    auto& rt = ClRuntime::get();                    // first call initializes, later calls just return it
    cl::Kernel k = rt.kernel("matrixMulKernel");

    // A: M x K
    k.setArg(0, K);
    k.setArg(1, M);
    k.setArg(2, K);
    k.setArg(3, A.buffer());

    // B: K x N
    k.setArg(4, N);
    k.setArg(5, K);
    k.setArg(6, N);
    k.setArg(7, B.buffer());

    // C: M x N
    k.setArg(8, N);
    k.setArg(9, M);
    k.setArg(10, N);
    k.setArg(11, C.buffer());

    auto roundUp = [](size_t x) {
        return ((x + TILE - 1) / TILE) * TILE;
    };

    cl::Event event;

    rt.queue().enqueueNDRangeKernel(
        k,
        cl::NullRange,
        cl::NDRange(roundUp(N), roundUp(M)),
        cl::NDRange(TILE, TILE),
        nullptr,
        &event
    );

    event.wait();

    cl_ulong start =
        event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

    cl_ulong end =
        event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    double kernel_ms = (end - start) * 1e-6;

    std::cout << "Kernel time: " << kernel_ms << " ms\n";
}

// Convenience wrapper for raw host arrays
void opencl_matmul(const float* A, const float* B, float* C, int M, int N, int K)
{
    DeviceMatrix dA(M, K), dB(K, N), dC(M, N);
    dA.upload(A);
    dB.upload(B);
    matmul(dA, dB, dC);
    dC.download(C);   // blocking read, so it also waits for the kernel to finish
}


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
void opencl_matmulOLD(const float* A, const float* B, float* C, int M, int N, int K)
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
        &matmul_kernel_sourceV2,
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

    size_t maxWorkGroup;
    clGetKernelWorkGroupInfo(
        kernel,
        device,
        CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(maxWorkGroup),
        &maxWorkGroup,
        nullptr);

    std::cout << "maxWorkGroup: " << maxWorkGroup << "\n";

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

