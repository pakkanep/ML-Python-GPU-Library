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


inline void blockedMatmul(
    Matrix A, Matrix B, Matrix C,
    __local float (*As)[BLOCK_SIZE],
    __local float (*Bs)[BLOCK_SIZE]
)
{
    const int blockRow = get_group_id(1);
    const int blockCol = get_group_id(0);
    const int row = get_local_id(1);
    const int col = get_local_id(0);

    const int globalRow = blockRow * BLOCK_SIZE + row;
    const int globalCol = blockCol * BLOCK_SIZE + col;
    

    // each thread in the block needs its own local accumulator
    float Cval = 0.0f;
    const int numTiles = (A.width + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int m = 0; m < numTiles; ++m)
    {   
        Matrix ASub = GetSubMatrix(A, blockRow, m);
        Matrix BSub = GetSubMatrix(B, m, blockCol);

        const int aCol = m * BLOCK_SIZE + col;   // global column in A
        const int bRow = m * BLOCK_SIZE + row;   // global row in B
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix

        As[row][col] = (globalRow < A.height && aCol < A.width) ? GetElement(ASub, row, col) : 0.0f;
        Bs[row][col] = (bRow < B.height && globalCol < B.width) ? GetElement(BSub, row, col) : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Cval += As[row][k] * Bs[k][col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write Csub to device memory
    // Each thread writes one element 
    if (globalRow < C.height && globalCol < C.width)
    {
        SetElement(C, globalRow, globalCol, Cval);
    }
}

__kernel void matrixMulKernel(
    int Awidth, int Aheight, int Astride, __global float* Aelements,
    int Bwidth, int Bheight, int Bstride, __global float* Belements,
    int Cwidth, int Cheight, int Cstride, __global float* Celements
)
{
    Matrix A = { Awidth, Aheight, Astride, Aelements };
    Matrix B = { Bwidth, Bheight, Bstride, Belements };
    Matrix C = { Cwidth, Cheight, Cstride, Celements };
    
    // do this here as:
    // non-kernel function variable cannot be declared in local address space
    __local float As[BLOCK_SIZE][BLOCK_SIZE+1];
    __local float Bs[BLOCK_SIZE][BLOCK_SIZE+1];
    blockedMatmul(A, B, C, As, Bs);
} 