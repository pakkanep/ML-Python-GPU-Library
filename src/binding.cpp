#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// #include <nanobind/stl/string.h>
#include <stdexcept>
#include <string>
#include <memory>

#include "backend.h"

namespace nb = nanobind;
using Mat = nb::ndarray<float, nb::numpy, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

enum class Backend {None, CUDA, OpenCL};
static Backend g_backend = Backend::None;

static Backend detect_backend()
{
#ifdef HAVE_CUDA // only means that CMake found OpenCL headers and lib when compiling
    if (cuda_available())
    {
        return Backend::CUDA;
    }
#endif
#ifdef HAVE_OPENCL // only means that CMake found OpenCL headers and lib when compiling
    if (opencl_available()) // Real runtime check
    {
        return Backend::OpenCL;
    }
#endif
    return Backend::None;
}

static nb::ndarray<nb::numpy, float, nb::ndim<2>> matmul(Mat A, Mat B)
{
    const size_t M = A.shape(0);
    const size_t K = A.shape(1);
    const size_t N = A.shape(1);
    
    if (B.shape(0) != K)
        throw std::invalid_argument(
            "matmul: shape mismatch, A is " + std::to_string(M) + "x" + std::to_string(K) +
            " but B is " + std::to_string(B.shape(0)) + "x" + std::to_string(N));

    // 2. Allocate output (unique_ptr so it's freed if the backend throws)
    std::unique_ptr<float[]> C(new float[M * N]);

    // 3. Dispatch (release the GIL so Python threads aren't blocked during GPU work)
    {
        nb::gil_scoped_release release;

        switch (g_backend) 
        {
#ifdef HAVE_CUDA
            case Backend::CUDA:
                cuda_matmul( A.data(), B.data(), C.get(), int(M), int(N), int(K) );
                break;
#endif
#ifdef HAVE_OPENCL
            case Backend::OpenCL:
                opencl_matmul( A.data(), B.data(), C.get(), int(M), int(N), int(K) );
                break;
#endif
            default:
                throw std::runtime_error("matmul: no GPU backend available");
        }
    }

    // 4. Hand ownership to Python via a capsule
    float* data = C.release();
    nb::capsule owner(data, [](void* p) noexcept
    {
        delete[] static_cast<float*>(p);
    });

    return nb::ndarray<nb::numpy, float, nb::ndim<2>>(data, {M, N}, owner);

}

NB_MODULE(ml_gpu_library, m)
{
    g_backend = detect_backend();
    
    m.def("matmul", &matmul);

    m.def("backend", []() -> const char*
    {
        switch (g_backend)
        {
            case Backend::CUDA:   return "cuda";
            case Backend::OpenCL: return "opencl";
            default:              return "none";
        }
    });

    m.def("test", []() { return "success"; });
}

// NB_MODULE(ml_gpu_library, m)
// {
//     m.def("hello", []() { return "success"; });
// }

/*

*/