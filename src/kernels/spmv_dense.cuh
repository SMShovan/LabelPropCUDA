#pragma once
#include <cuda_runtime.h>
#include "reduce.cuh"

// y_out[u] = sum_j A[u,j] * x[j]
template <int BLOCK_SIZE>
__global__ void spmv_dense_row_kernel(const double* __restrict__ A,
                                      int n,
                                      const int* __restrict__ rows,
                                      int num_rows,
                                      const int* __restrict__ cols,
                                      int num_cols,
                                      const double* __restrict__ x,
                                      double* __restrict__ y_out) {
    int ri = blockIdx.x;
    if (ri >= num_rows) return;
    int u = rows ? rows[ri] : ri;
    double partial = 0.0;
    for (int j = threadIdx.x; j < num_cols; j += blockDim.x) {
        int c = cols ? cols[j] : j;
        partial += A[u * n + c] * x[j];
    }
    double sum = blockReduceSum<BLOCK_SIZE>(partial);
    if (threadIdx.x == 0) y_out[ri] = sum;
}


