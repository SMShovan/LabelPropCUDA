#pragma once
#include <cuda_runtime.h>

__global__ inline void propagate_kernel(const int* __restrict__ row_g,
                                        const int* __restrict__ col_g,
                                        const double* __restrict__ val_g,
                                        const uint8_t* __restrict__ isL0_d,
                                        const uint8_t* __restrict__ isL1_d,
                                        double* __restrict__ Ld,
                                        const uint8_t* __restrict__ active,
                                        uint8_t* __restrict__ nextActive,
                                        int* __restrict__ any,
                                        double delta_thr,
                                        int n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;
    if (!active[u]) return;
    if (isL0_d[u] || isL1_d[u]) return;
    int s = row_g[u], e = row_g[u+1];
    double Lu = Ld[u];
    double Omega = 0.0, W0 = 0.0, W1 = 0.0, S = 0.0;
    for (int p = s; p < e; ++p) {
        int v = col_g[p]; double w = val_g[p];
        Omega += w;
        if (isL0_d[v]) W0 += w; else if (isL1_d[v]) W1 += w; else S += (Ld[v] - Lu) * (w);
    }
    double Lnew = Lu;
    if (Omega > 0.0) Lnew = Lu + (0.0 - Lu)*(W0/Omega) + (1.0 - Lu)*(W1/Omega) + (S/Omega);
    double diff = fabs(Lnew - Lu);
    Ld[u] = Lnew;
    if (diff > delta_thr) {
        atomicExch(any, 1);
        nextActive[u] = 1;
        for (int p = s; p < e; ++p) { int v = col_g[p]; if (!isL0_d[v] && !isL1_d[v]) nextActive[v] = 1; }
    }
}


