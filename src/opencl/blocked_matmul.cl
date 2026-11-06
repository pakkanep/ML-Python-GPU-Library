#define BLOCK_SIZE 16

typedef struct
{
    int width;
    int height;
    int stride;
    __global float* elements;
} Matrix;

inline float GetElement(const Matrix A, int row, int col)
{
    return A.elements[A.stride * row + col];
}

inline void SetElement(Matrix A, int row, int col, float value)
{
    A.elements[A.stride * row + col] = value;
}

inline Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix sub;
    sub.width    = BLOCK_SIZE;
    sub.height   = BLOCK_SIZE;
    sub.stride   = A.stride;
    sub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return sub;
}

// why pass ASub and BSub as params
void __kernel blocked_matmul(__global const Matrix A, __global conts Matrix B, __global Matrix C)
{
    // Block row and column
    int blockRow = get_global_id(1);
    int blockCol = get_global_id(0);
    
    // Each thread block computes one sub-matrix Csub of C
    __local float ASub[BLOCK_SIZE][BLOCK_SIZE];
    __local float BSub[BLOCK_SIZE][BLOCK_SIZE];
    Matrix CSub = GetSubMatrix(C, blockRow, blockCol);

    // each thread in the block needs its own local accumulator
    float Cval = 0.0f;
    int row = get_local_id(1);
    int col = get_local_id(0);

    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m)
    {   
        Matrix ASub = GetSubMatrix(A, blockRow, m);
        Matrix BSub = GetSubMatrix(B, m, blockCol);

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix 
        As[row][col] = GetElement(ASub, row, col); // if dims are not a multiple of blocksize set remaining ones to zero
        Bs[row][col] = GetElement(BSub, row, col); // if dims are not a multiple of blocksize set remaining ones to zero

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Cval += ASub[row][k] * BSub[k][col];
            // Synchronize to make sure that the preceding
            // computation is done before loading two new
            // sub-matrices of A and B in the next iteration
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    // Write Csub to device memory
    // Each thread writes one element 
    SetElement(Csub, row, col, Cvalue);
}