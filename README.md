# Machine Learning Python Library utilizing a GPU

A small Python library for running machine-learning operations on the GPU.

The compute kernels are written in C++ (OpenCL and CUDA) and exposed to Python with [nanobind](https://github.com/wjakob/nanobind). The backend is chosen at runtime: CUDA if available, otherwise OpenCL.

Currently implemented:
- Matmul

## Requirements
- Python 3.8+ and NumPy
- CMake 3.18+ and a C++17 compiler
- nanobind (`pip install nanobind`)
- An OpenCL driver and/or the CUDA toolkit

## Building
 
From the `src/` directory:
 
```bash
cmake -B build          # configure (Python, nanobind, OpenCL/CUDA)
cmake --build build     # compile
```

## Project structure
 
```
src/
├── CMakeLists.txt      # build config
├── binding.cpp         # Python bindings and backend dispatch
├── backend.h           # backend function declarations
├── opencl/matmul.cpp   # OpenCL implementation
├── cuda/matmul.cu      # CUDA implementation
└── main.py             # test script
```


## Benchmarks

 
Matrix multiplication of two 4000×4000 `float32` matrices.
 
| Implementation        | `matmul` call | Whole script (`time`) |
|-----------------------|--------------:|----------------------:|
| plain Python (todo)   |         inf s |                 inf s |
| plain C/C++ (todo)    |         --- s |                 --- s |
| NumPy (BLAS)          |       0.486 s |               1.141 s |
| myLib matmulV1  (OpenCL) |     16.08 s |               16.71 s |
| myLib matmulV2  (OpenCL) |     3.545 s |               4.021 s |
 
- Unfortunately the OpenCL version is currently very slow, about 7× slower than NumPy. Even though the numpy version is definitely not a weak baseline, as it uses OpenBLAS, which likely uses the whole shabang such as cache/register blocking, SIMD, and all the available CPU cores.  
- We managed to get a huge speedup by 'only' changing the memory access pattern.
- In V1 the threads reads addresses in the memory that are a whole row apart and write to the     result in the same way.
- in V2 we only swap the row (i) and col(j) indexing inside the kernel, so the neighbouring threads access neighbouring memory instead of kilobytes apart in the memory.
- The kernel is still a simple implementation without shared memory tiling
  and the OpenCL context is created and the kernel compiled on every call,
  so there is room for optimization.