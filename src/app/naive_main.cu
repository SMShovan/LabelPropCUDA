#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>
// cuSPARSE for CSR SpMV
#include <cusparse.h>
#include "../util/cuda_checks.cuh"
#include "../graph/graph_builder.hpp"
#include "../graph/graph_csr.hpp"

// Simple ER generators (duplicated from baseline for now)

// Iterative LP using CSR and cuSPARSE csrmv on full vector (labels clamped)
static void iterative_lp_csr(int n,
                             const std::vector<int>& row_offsets_h,
                             const std::vector<int>& col_indices_h,
                             const std::vector<double>& values_h,
                             const std::vector<int>& labeled_ids,
                             const std::vector<double>& labeled_vals,
                             int max_iters, double tol, double alpha,
                             std::vector<double>& out_labels) {
    // Device allocations
    int nnz = static_cast<int>(values_h.size());
    int *d_row_offsets = nullptr, *d_col_indices = nullptr;
    double *d_values = nullptr, *d_x_full = nullptr, *d_y_full = nullptr;
    checkCudaStatus(cudaMalloc((void**)&d_row_offsets, sizeof(int) * (n + 1)), "malloc row_offsets");
    checkCudaStatus(cudaMalloc((void**)&d_col_indices, sizeof(int) * nnz), "malloc col_indices");
    checkCudaStatus(cudaMalloc((void**)&d_values, sizeof(double) * nnz), "malloc values");
    checkCudaStatus(cudaMemcpy(d_row_offsets, row_offsets_h.data(), sizeof(int) * (n + 1), cudaMemcpyHostToDevice), "copy row_offsets");
    checkCudaStatus(cudaMemcpy(d_col_indices, col_indices_h.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice), "copy col_indices");
    checkCudaStatus(cudaMemcpy(d_values, values_h.data(), sizeof(double) * nnz, cudaMemcpyHostToDevice), "copy values");

    checkCudaStatus(cudaMalloc((void**)&d_x_full, sizeof(double) * n), "malloc x_full");
    checkCudaStatus(cudaMalloc((void**)&d_y_full, sizeof(double) * n), "malloc y_full");

    // cuSPARSE handle and descriptors (new or legacy API)
    cusparseHandle_t cusparseH = nullptr;
    checkCusparseStatus(cusparseCreate(&cusparseH), "create cusparse");
#if defined(CUSPARSE_VERSION) && (CUSPARSE_VERSION >= 11000)
    cusparseSpMatDescr_t matA;
    checkCusparseStatus(cusparseCreateCsr(&matA,
                                          n, n, nnz,
                                          (void*)d_row_offsets, (void*)d_col_indices, (void*)d_values,
                                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F), "create csr mat");
    cusparseDnVecDescr_t vecX, vecY;
    checkCusparseStatus(cusparseCreateDnVec(&vecX, n, (void*)d_x_full, CUDA_R_64F), "create dnvec x");
    checkCusparseStatus(cusparseCreateDnVec(&vecY, n, (void*)d_y_full, CUDA_R_64F), "create dnvec y");
    size_t spmvBufferSize = 0;
    const double one = 1.0, zero = 0.0;
    checkCusparseStatus(cusparseSpMV_bufferSize(cusparseH,
                                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &one, matA, vecX, &zero, vecY,
                                                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                                &spmvBufferSize), "spmv bufferSize");
    void* dSpmvBuffer = nullptr;
    checkCudaStatus(cudaMalloc(&dSpmvBuffer, spmvBufferSize), "malloc spmv buffer");
#else
    cusparseMatDescr_t descrA;
    checkCusparseStatus(cusparseCreateMatDescr(&descrA), "create descr");
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    const double one = 1.0, zero = 0.0;
#endif

    // Degree and constant term c = D^{-1} A * xL_full at U positions
    std::vector<double> deg = csrRowSums(row_offsets_h, values_h, n);

    std::vector<char> is_labeled(n, 0);
    for (int id : labeled_ids) if (id >= 0 && id < n) is_labeled[id] = 1;
    std::vector<int> U;
    for (int i = 0; i < n; ++i) if (!is_labeled[i]) U.push_back(i);

    // Build xL_full
    std::vector<double> xL_full(n, 0.0);
    for (size_t i = 0; i < labeled_ids.size(); ++i) xL_full[labeled_ids[i]] = labeled_vals[i];
    checkCudaStatus(cudaMemcpy(d_x_full, xL_full.data(), sizeof(double) * n, cudaMemcpyHostToDevice), "copy xL_full");
#if defined(CUSPARSE_VERSION) && (CUSPARSE_VERSION >= 11000)
    checkCusparseStatus(cusparseSpMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &one, matA, vecX, &zero, vecY,
                                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dSpmvBuffer), "SpMV A*xL");
#else
    checkCusparseStatus(cusparseDcsrmv(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       n, n, nnz, &one,
                                       descrA, d_values, d_row_offsets, d_col_indices,
                                       d_x_full, &zero, d_y_full), "csrmv A*xL");
#endif

    std::vector<double> y_full_h(n, 0.0);
    checkCudaStatus(cudaMemcpy(y_full_h.data(), d_y_full, sizeof(double) * n, cudaMemcpyDeviceToHost), "copy y_full");
    std::vector<double> cU(U.size(), 0.0);
    for (size_t i = 0; i < U.size(); ++i) {
        int u = U[i];
        cU[i] = (deg[u] > 0.0) ? (y_full_h[u] / deg[u]) : 0.0;
    }

    // Initialize yU (zeros)
    std::vector<double> yU(U.size(), 0.0), yU_new(U.size(), 0.0);

    // Iteration loop
    for (int it = 0; it < max_iters; ++it) {
        // Build xU_full
        std::fill(xL_full.begin(), xL_full.end(), 0.0);
        for (size_t i = 0; i < U.size(); ++i) xL_full[U[i]] = yU[i];
        checkCudaStatus(cudaMemcpy(d_x_full, xL_full.data(), sizeof(double) * n, cudaMemcpyHostToDevice), "copy xU_full");
        // y_full = A * xU_full
#if defined(CUSPARSE_VERSION) && (CUSPARSE_VERSION >= 11000)
        checkCusparseStatus(cusparseSpMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &one, matA, vecX, &zero, vecY,
                                         CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dSpmvBuffer), "SpMV A*xU");
#else
        checkCusparseStatus(cusparseDcsrmv(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           n, n, nnz, &one,
                                           descrA, d_values, d_row_offsets, d_col_indices,
                                           d_x_full, &zero, d_y_full), "csrmv A*xU");
#endif
        checkCudaStatus(cudaMemcpy(y_full_h.data(), d_y_full, sizeof(double) * n, cudaMemcpyDeviceToHost), "copy y_full U");
        // Update
        double max_diff = 0.0;
        for (size_t i = 0; i < U.size(); ++i) {
            int u = U[i];
            double yhat = (deg[u] > 0.0) ? (y_full_h[u] / deg[u]) : 0.0;
            double next = alpha * (yhat + cU[i]) + (1.0 - alpha) * yU[i];
            max_diff = std::max(max_diff, std::abs(next - yU[i]));
            yU_new[i] = next;
        }
        yU.swap(yU_new);
        if (max_diff < tol) break;
    }

    // Assemble full label vector
    out_labels.assign(n, 0.0);
    for (size_t i = 0; i < labeled_ids.size(); ++i) out_labels[labeled_ids[i]] = labeled_vals[i];
    for (size_t i = 0; i < U.size(); ++i) out_labels[U[i]] = yU[i];

    // Cleanup
    // Destroy descriptors and handle
#if defined(CUSPARSE_VERSION) && (CUSPARSE_VERSION >= 11000)
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cudaFree(dSpmvBuffer);
#else
    cusparseDestroyMatDescr(descrA);
#endif
    cusparseDestroy(cusparseH);
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_x_full);
    cudaFree(d_y_full);
}

int main(int argc, char** argv) {
    // CLI
    int n = 10;
    double avgdeg = 5.0;
    bool connected = true;
    int num_batches = 1;
    int batch_size = 0;
    double delta_labeled_pct = 0.1;
    int labeled_count = 3;
    unsigned seed = 1234;
    int max_iters = 500;
    double tol = 1e-6;
    double alpha = 1.0;
    std::string format = "dense"; // dense|csr (dense will convert to CSR internally)
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--n" || arg == "--num-vertices") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (arg == "--avgdeg" && i + 1 < argc) avgdeg = std::atof(argv[++i]);
        else if (arg == "--connected") connected = true;
        else if (arg == "--num-batches" && i + 1 < argc) num_batches = std::atoi(argv[++i]);
        else if (arg == "--batch-size" && i + 1 < argc) batch_size = std::atoi(argv[++i]);
        else if (arg == "--delta-labeled-pct" && i + 1 < argc) delta_labeled_pct = std::atof(argv[++i]);
        else if ((arg == "--labeled" || arg == "--num-labeled") && i + 1 < argc) labeled_count = std::atoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = static_cast<unsigned>(std::stoul(argv[++i]));
        else if (arg == "--max-iters" && i + 1 < argc) max_iters = std::atoi(argv[++i]);
        else if (arg == "--tol" && i + 1 < argc) tol = std::atof(argv[++i]);
        else if (arg == "--alpha" && i + 1 < argc) alpha = std::atof(argv[++i]);
        else if (arg == "--format" && i + 1 < argc) format = argv[++i];
    }
    labeled_count = std::max(0, std::min(labeled_count, n));

    // Initial graph (dense), then CSR
    std::vector<double> A = connected ? generateConnectedAdjacencyWithSpanningTree(n, avgdeg, seed)
                                      : generateErdosRenyiAdjacency(n, avgdeg, seed);
    std::vector<int> row_offsets, col_indices; std::vector<double> values;
    // For now, we use CSR iterative backend in both modes; dense is converted
    denseToCsr(A, n, row_offsets, col_indices, values);

    // Initial labeled set
    std::vector<int> labeled_ids; labeled_ids.reserve(n);
    std::vector<double> labeled_vals; labeled_vals.reserve(n);
    for (int i = 0; i < labeled_count; ++i) { labeled_ids.push_back(i); labeled_vals.push_back((i % 2 == 0) ? 1.0 : 0.0); }

    // Solve for t
    std::vector<double> labels;
    iterative_lp_csr(n, row_offsets, col_indices, values, labeled_ids, labeled_vals, max_iters, tol, alpha, labels);
    std::cout << "labels_t [" << labels.size() << "]:" << std::endl;
    for (double v : labels) std::cout << v << " "; std::cout << std::endl;

    // Batches
    for (int b = 0; b < num_batches; ++b) {
        if (batch_size <= 0) break;
        int oldN = n;
        int add = batch_size;
        int newN = n + add;
        // Extend dense and generate connections for new vertices, then reconvert to CSR
        std::vector<double> Anew(newN * newN, 0.0);
        // Copy old
        for (int i = 0; i < n; ++i) std::copy_n(&A[i * n], n, &Anew[i * newN]);
        // Connect new vertices
        std::mt19937 rng(seed + 1000 + b);
        std::uniform_real_distribution<double> wdist(0.0, 1.0);
        std::uniform_real_distribution<double> udist(0.0, 1.0);
        if (connected) {
            for (int v = n; v < newN; ++v) {
                std::uniform_int_distribution<int> parentDist(0, v - 1);
                int p = parentDist(rng);
                double w = wdist(rng);
                Anew[p * newN + v] = w; Anew[v * newN + p] = w;
            }
        }
        double p = std::max(0.0, std::min(1.0, avgdeg / static_cast<double>(newN - 1)));
        for (int i = 0; i < newN; ++i) {
            for (int j = i + 1; j < newN; ++j) {
                if (Anew[i * newN + j] == 0.0 && udist(rng) < p) {
                    double w = wdist(rng);
                    Anew[i * newN + j] = w; Anew[j * newN + i] = w;
                }
            }
        }
        A.swap(Anew); n = newN;
        denseToCsr(A, n, row_offsets, col_indices, values);

        // Label delta fraction
        int k_gt = static_cast<int>(delta_labeled_pct * static_cast<double>(add) + 0.5);
        if (k_gt > add) k_gt = add;
        for (int j = 0; j < k_gt; ++j) {
            int v = oldN + j;
            labeled_ids.push_back(v);
            labeled_vals.push_back((j % 2 == 0) ? 1.0 : 0.0);
        }

        // Re-solve
        iterative_lp_csr(n, row_offsets, col_indices, values, labeled_ids, labeled_vals, max_iters, tol, alpha, labels);
        std::cout << "labels_t+" << (b + 1) << " [" << labels.size() << "]:" << std::endl;
        for (double v : labels) std::cout << v << " "; std::cout << std::endl;
    }

    return 0;
}


