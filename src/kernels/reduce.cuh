#pragma once
#include <cuda_runtime.h>

template <int BLOCK_SIZE>
__device__ inline double blockReduceSum(double val) {
    __shared__ double shm[BLOCK_SIZE];
    int tid = threadIdx.x;
    shm[tid] = val;
    __syncthreads();
    for (int offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) shm[tid] += shm[tid + offset];
        __syncthreads();
    }
    return shm[0];
}


