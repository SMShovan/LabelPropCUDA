#pragma once
#include <cuda_runtime.h>

__device__ __forceinline__ int __cc_find_root(int x, int* parent) {
    while (true) {
        int p = parent[x];
        int gp = parent[p];
        if (p == gp) return p;
        parent[x] = gp; // path halving
        x = gp;
    }
}

__global__ inline void cc_hook_edges_kernel(const int* __restrict__ d_row,
                                            const int* __restrict__ d_col,
                                            int K,
                                            int* __restrict__ parent,
                                            int* __restrict__ d_changed) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= K) return;
    int row_start = d_row[u];
    int row_end   = d_row[u+1];
    for (int e = row_start; e < row_end; ++e) {
        int v = d_col[e];
        int ru = __cc_find_root(u, parent);
        int rv = __cc_find_root(v, parent);
        if (ru != rv) {
            int hi = ru > rv ? ru : rv;
            int lo = ru ^ rv ^ hi; // min without branch
            int old = atomicMin(parent + hi, lo);
            if (old != lo) atomicExch(d_changed, 1);
        }
    }
}

__global__ inline void cc_jump_kernel(int K, int* __restrict__ parent) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < K) parent[i] = parent[parent[i]];
}


