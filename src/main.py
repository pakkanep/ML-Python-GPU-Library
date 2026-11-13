import sys
import numpy as np
import time
from build import ml_gpu_library as gpulib

if __name__ == "__main__":
    A = np.random.rand(4000, 4000).astype(np.float32)
    B = np.random.rand(4000, 4000).astype(np.float32)

    # GPU benchmark
    t = time.perf_counter()
    C_gpu = gpulib.matmul(A, B)
    gpu_time = time.perf_counter() - t

    print(f"GPU time: {gpu_time:.3f} s")

    # NumPy reference
    t = time.perf_counter()
    C_numpy = A @ B
    numpy_time = time.perf_counter() - t

    print(f"NumPy time: {numpy_time:.3f} s")

    # Correctness
    print("Correct:", np.allclose(
        C_gpu, C_numpy,
        rtol=1e-5,
        atol=1e-4
    ))

    # Performance
    operations = 2 * 4000**3

    print(f"GPU throughput: {operations / gpu_time / 1e9:.2f} GFLOP/s")
    print(f"NumPy throughput: {operations / numpy_time / 1e9:.2f} GFLOP/s")