import sys
import numpy as np
import time
from build import ml_gpu_library as gpulib # build/ml_gpu_library.cpython-310-x86_64-linux-gnu.so

def numpyMatmul(A, B):
    result = A @ B

def openCLMatmul(A, B):
    print(gpulib.backend())
    result = gpulib.matmul(A, B)
    print(type(result))



if __name__ == "__main__":
    print(gpulib.test())
    A = np.random.rand(4000, 4000).astype(np.float32)
    B = np.random.rand(4000, 4000).astype(np.float32)
    t = time.perf_counter()
    # numpyMatmul(A, B)
    openCLMatmul(A, B)
    print(f"{time.perf_counter() - t:.3f} s")
    #print(np.allclose(ml_gpu_library.matmul(A, B), A @ B, atol=1e-4))