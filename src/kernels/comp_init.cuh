#pragma once
#include <cuda_runtime.h>

__global__ inline void k_comp_accumulate_seed_weights(const int* __restrict__ row,
                                                      const int* __restrict__ col,
                                                      const double* __restrict__ val,
                                                      const int* __restrict__ delta_ids,
                                                      const int* __restrict__ node_comp,
                                                      const unsigned char* __restrict__ isL0,
                                                      const unsigned char* __restrict__ isL1,
                                                      int K,
                                                      double* __restrict__ comp_sum0,
                                                      double* __restrict__ comp_sum1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K) return;
    int g = delta_ids[i];
    int cid = node_comp[i];
    double s0 = 0.0, s1 = 0.0;
    int start = row[g], end = row[g+1];
    for (int e = start; e < end; ++e) {
        int v = col[e]; double w = val[e];
        if (isL0[v]) s0 += w; else if (isL1[v]) s1 += w;
    }
    if (s0 != 0.0) atomicAdd(&comp_sum0[cid], s0);
    if (s1 != 0.0) atomicAdd(&comp_sum1[cid], s1);
}

__global__ inline void k_comp_compute_labels(const double* __restrict__ comp_sum0,
                                             const double* __restrict__ comp_sum1,
                                             int num_comp,
                                             double* __restrict__ comp_label) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_comp) return;
    double s0 = comp_sum0[c];
    double s1 = comp_sum1[c];
    double den = s0 + s1;
    double Lc = 0.5;
    if (den > 0.0) Lc = 0.5 + (s1 - s0) / (2.0 * den);
    comp_label[c] = Lc;
}

__global__ inline void k_assign_delta_labels(const int* __restrict__ delta_ids,
                                             const int* __restrict__ node_comp,
                                             const double* __restrict__ comp_label,
                                             int K,
                                             double* __restrict__ Ld) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K) return;
    int g = delta_ids[i];
    int cid = node_comp[i];
    Ld[g] = comp_label[cid];
}


