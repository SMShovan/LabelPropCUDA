#pragma once
#include <vector>

inline void denseToCsr(const std::vector<double>& A, int n,
                       std::vector<int>& row_offsets, std::vector<int>& col_indices, std::vector<double>& values) {
    row_offsets.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        int nnz_row = 0; const double* row = &A[i * n];
        for (int j = 0; j < n; ++j) if (row[j] != 0.0) ++nnz_row;
        row_offsets[i + 1] = row_offsets[i] + nnz_row;
    }
    int nnz = row_offsets[n];
    col_indices.resize(nnz); values.resize(nnz);
    for (int i = 0; i < n; ++i) {
        int idx = row_offsets[i]; const double* row = &A[i * n];
        for (int j = 0; j < n; ++j) if (row[j] != 0.0) { col_indices[idx] = j; values[idx] = row[j]; ++idx; }
    }
}

inline std::vector<double> csrRowSums(const std::vector<int>& row_offsets, const std::vector<double>& values, int n) {
    std::vector<double> deg(n, 0.0);
    for (int i = 0; i < n; ++i) { double s = 0.0; for (int k = row_offsets[i]; k < row_offsets[i + 1]; ++k) s += values[k]; deg[i] = s; }
    return deg;
}


