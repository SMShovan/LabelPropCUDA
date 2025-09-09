#pragma once
#include <cuda_runtime.h>

__global__ inline void k_init_array_int(int* a, int n, int value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = value;
}

__global__ inline void k_scatter_delta_pos(const int* __restrict__ delta_ids, int K, int* __restrict__ delta_pos) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < K) delta_pos[delta_ids[i]] = i;
}

__global__ inline void k_count_delta_edges(const int* __restrict__ row,
                                           const int* __restrict__ col,
                                           const double* __restrict__ val,
                                           const int* __restrict__ delta_ids,
                                           const int* __restrict__ delta_pos,
                                           int K, double tau,
                                           int* __restrict__ counts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K) return;
    int g = delta_ids[i];
    int start = row[g], end = row[g+1];
    int c = 0;
    for (int e = start; e < end; ++e) {
        int v = col[e];
        if (delta_pos[v] >= 0 && val[e] > tau) ++c;
    }
    counts[i] = c;
}

__global__ inline void k_fill_delta_csr(const int* __restrict__ row,
                                        const int* __restrict__ col,
                                        const double* __restrict__ val,
                                        const int* __restrict__ delta_ids,
                                        const int* __restrict__ delta_pos,
                                        const int* __restrict__ d_row,
                                        int K, double tau,
                                        int* __restrict__ d_col,
                                        double* __restrict__ d_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K) return;
    int g = delta_ids[i];
    int start = row[g], end = row[g+1];
    int w = d_row[i];
    for (int e = start; e < end; ++e) {
        int v = col[e];
        int j = delta_pos[v];
        if (j >= 0) {
            double we = val[e];
            if (we > tau) { d_col[w] = j; d_val[w] = we; ++w; }
        }
    }
}


