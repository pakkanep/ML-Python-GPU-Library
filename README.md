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
cmake -B build          # configure
cmake --build build     # compile
```

## Project structure

```text
src/
├── CMakeLists.txt
├── binding.cpp
├── backend.h
├── opencl/
│   ├── matmul.cpp
│   └── kernels/
├── cuda/
│   └── matmul.cu
└── main.py
```

## Benchmarks

Matrix multiplication of two 4000×4000 `float32` matrices.

| Implementation | `matmul` call | Whole script (`time`) |
|:---------------|--------------:|----------------------:|
| Plain Python (TODO) | — | — |
| Plain C/C++ (TODO) | — | — |
| NumPy (BLAS), original | 0.486 s | 1.141 s |
| myLib V1 (OpenCL, naive) | 16.080 s | 16.710 s |
| myLib V2 (OpenCL, coalesced) | 3.545 s | 4.021 s |
| myLib V3 (OpenCL, SLM tiled 16×16) | 1.451 s | — |

The latest measurements use a Python timer around the individual matrix multiplication call. Whole-script timings were not recorded for the latest benchmark.

### Optimization history

- **V1 → V2:** Changed the work-item indexing so neighboring work-items access neighboring memory addresses. This improved memory coalescing and substantially reduced memory bandwidth pressure.
- **V2 → V3:** Introduced shared local memory (SLM) tiling with 16×16 tiles. Work-items cooperatively load matrix tiles into local memory and reuse them during multiplication.
- **V3:** Introduced a persistent OpenCL runtime that caches the context, command queue, compiled program, and kernel objects rather than rebuilding everything on every call.
- **V4 (planned):** Register blocking, where each work-item computes multiple output elements to increase data reuse and instruction-level parallelism.

## GPU Benchmarks

**Setup:** 4000×4000 `float32` matrix multiplication, Intel Iris Xe Graphics (80 EUs, up to 1.3 GHz, theoretical FP32 peak approximately 1664 GFLOP/s).

The operation requires approximately:

\[
2(4000)^3 = 128 \times 10^9
\]



| Host Task | V1 | V2 | V3 | V4 |
|:----------|---:|---:|---:|---:|
| Elapsed time | 15.171 s | 4.854 s | 4.205 s | — |
| `clFinish` (waiting for kernel) | 12.321 s (81.2%) | 2.631 s (54.2%) | — | — |
| `clEnqueueReadBuffer` | 0.030 s (0.2%) | 0.021 s (0.4%) | 2.175 s (51.7%) | — |
| `clBuildProgram` | 0.342 s (2.3%) | 0.174 s (3.6%) | 0.178 s (4.2%) | — |
| `clCreateBuffer` | 0.227 s (1.5%) | 0.052 s (1.1%) | 0.062 s (1.5%) | — |
| `clEnqueueWriteBuffer` | — | — | 0.027 s (0.6%) | — |
| `clEnqueueNDRangeKernel` | 0.007 s | 0.001 s | — | — |



### GPU Hotspots

| Metric | V1 | V2 | V3 | V4 |
|:-------|---:|---:|---:|---:|
| Elapsed time | 16.685 s | 4.549 s | 4.673 s | — |
| GPU time | 14.541 s | 2.908 s | 2.548 s | — |
| GPU time, % of elapsed | 87.2% | 63.9% | 54.5% | — |
| EU Array Stalled/Idle | 92.5% | 64.0% | 71.2% | — |
| GPU L3 Bandwidth Bound | 87.2% | 52.0% | 7.6% | — |
| Occupancy | 99.2% | 98.4% | 98.2% | — |
| Sampler Busy | 0.1% | 0.3% | 0.6% | — |

These are VTune collection-level measurements.


## TODOs

- Finish the CUDA backend.
- Benchmark CUDA kernels with NVIDIA Nsight.
- Implement register blocking for OpenCL.
- Compare tile sizes and loop unrolling.
- Embed OpenCL kernel sources at build time with CMake.
- Add more GPU-accelerated machine-learning operations.

## Notes

```bash
rm -rf r_offload
vtune -collect xpu-offload -r r_offload -- python3 main.py
```

VTune GPU hotspots profiling:

```bash
rm -rf r_hot
vtune -collect gpu-hotspots -r r_hot -- python3 main.py
```

