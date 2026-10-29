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
- In V1 the threads reads addresses in the memory that are a whole row apart and write to the result in the same way.
- in V2 we only swap the row (i) and col(j) indexing inside the kernel, so the neighbouring threads access neighbouring memory instead of kilobytes apart in the    memory.

- The kernel is still a simple implementation without shared memory tiling
  and the OpenCL context is created and the kernel compiled on every call,
  so there is room for optimization.


## GPU Benchmarks

**Setup:** 4000×4000 `float32` matmul, OpenCL on Intel Iris Xe (80 EUs @ 1.3 GHz, peak 1664 GFLOP/s FP32)

| Version | Description | Kernel Time (s) | GFLOP/s | % of Peak | Speedup vs. V1 |
|:--------|:------------|----------------:|--------:|----------:|---------------:|
| V1 | Naive | 12.321 | 10.39 | 0.62% | 1.00× |
| V2 | Coalesced memory access | 2.631 | 48.65 | 2.92% | 4.68× |
| V3 | SLM tiling (16×16) | | | | |
| V4 | SLM tiling + register blocking | | | | |

### Host Tasks (XPU Offload)

| Host Task | V1 | V2 | V3 | V4 |
|:----------|---:|---:|---:|---:|
| Elapsed time | 15.171 s | 4.854 s | | |
| `clFinish` (waiting for kernel) | 12.321 s (81.2%) | 2.631 s (54.2%) | | |
| `clBuildProgram` | 0.342 s (2.3%) | 0.174 s (3.6%) | | |
| `clCreateBuffer` (×3) | 0.227 s (1.5%) | 0.052 s (1.1%) | | |
| `clEnqueueReadBuffer` | 0.030 s (0.2%) | 0.021 s (0.4%) | | |
| `clEnqueueNDRangeKernel` | 0.007 s (0.0%) | 0.001 s (0.0%) | | |

### GPU Hotspots

| Metric | V1 | V2 | V3 | V4 |
|:-------|---:|---:|---:|---:|
| Elapsed time | 16.685 s | 4.549 s | | |
| GPU time | 14.541 s | 2.908 s | | |
| GPU time, % of elapsed | 87.2% | 63.9% | | |
| EU Array Stalled/Idle | 92.5% | 64.0% | | |
| GPU L3 Bandwidth Bound | 87.2% | 52.0% | | |
| Occupancy | 99.2% | 98.4% | | |
| Sampler Busy | 0.1% | 0.3% | | |

### Takeaways

- **V1 (naive):** L3-bandwidth bound. Neighboring work-items access memory far apart, so reads can't be combined and EUs spend 92.5% of the time stalled or idle. Occupancy is near peak, so the launch configuration is not the problem.
- **V2 (coalesced):** Coalescing cut L3 pressure from 87% to 52% and gave a 4.7× speedup, but EUs are still stalled or idle 64% of the time. Every value is still loaded once per use, so the next step is data reuse via SLM tiling. Host overhead now takes about a third of the elapsed time, so measure kernel time separately from start-up costs.
- **V3 (SLM tiling):** 
- **V4 (register blocking):**

## TODO
- finish the cuda side
- benchmark the kernels with Nsight
- build the OpenCL kernel only once

## notes
vtune -collect xpu-offload -r r_offload -- python3 src/main.py   host <-> GPU timeline, transfers, kernel times
vtune -collect gpu-hotspots -r r_hot -- python3 src/main.py     #inside the kernel: EU utilization, memory bandwidth, occupancy