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

## TODO
- First benchmark the naive kernels with vtune and Nsight 
- The OpenCL kernel will run on a laptop, so the RAM is shared between CPU and GPU without the neef of a PCIe transfer 
- Try allocating with CL_MEM_ALLOC_HOST_PTR, or CL_MEM_USE_HOST_PTR, to let the GPU use the memory without a copy

vtune -collect xpu-offload -r r_offload -- python3 src/main.py   host <-> GPU timeline, transfers, kernel times
vtune -collect gpu-hotspots -r r_hot -- python3 src/main.py     #inside the kernel: EU utilization, memory bandwidth, occupancy

coalesced V1 bechmark:
Elapsed Time: 15.171s

Hottest Host Tasks
Host Task               Task Time  % of Elapsed Time(%)  Task Count
----------------------  ---------  --------------------  ----------
clFinish                  12.321s                 81.2%           1
clBuildProgram             0.342s                  2.3%           1
clCreateBuffer             0.227s                  1.5%           3
clEnqueueReadBuffer        0.030s                  0.2%           1
clEnqueueNDRangeKernel     0.007s                  0.0%           1
[Others]                   0.001s                  0.0%           7

Hottest GPU Computing Tasks
Computing Task  Total Time  Instance Count  SIMD Utilization(%)
--------------  ----------  --------------  -------------------
matmulkernel       12.321s               1               100.0%

Elapsed Time: 16.685s
    GPU Time: 14.541s
EU Array Stalled/Idle: 92.5% of Elapsed time with GPU busy
 | The percentage of time when the EUs were stalled or idle is high, which has a
 | negative impact on compute-bound applications.
 |
    GPU L3 Bandwidth Bound: 87.2% of peak value
     | L3 bandwidth was high when EUs were stalled or idle. Consider improving
     | cache reuse.
     |

        Hottest GPU Computing Tasks Bound by GPU L3 Bandwidth
        Computing Task  Total Time
        --------------  ----------
        matmulkernel       14.385s
    Occupancy: 99.2% of peak value

        Hottest GPU Computing Tasks with Low Occupancy
        Computing Task  Total Time  SIMD Width  Peak EU Threads Occupancy(%)  Occupancy(%)  SIMD Utilization(%)
        --------------  ----------  ----------  ----------------------------  ------------  -------------------
    Sampler Busy: 0.1% of peak value

        Hottest GPU Computing Tasks with High Sampler Usage
        Computing Task  Total Time