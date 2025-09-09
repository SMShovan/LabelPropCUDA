#include <iostream>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include <set>
#include <random>
#include <type_traits>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
using namespace std;

// === Harmonic label propagation helpers (dense, double) ===
static void checkCudaStatus(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << ": " << cudaGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static void checkCublasStatus(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error: " << msg << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static void checkCusolverStatus(cusolverStatus_t status, const char* msg) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cuSOLVER error: " << msg << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static std::vector<double> generateErdosRenyiAdjacency(int n, double avg_degree, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> wdist(0.0, 1.0);
    std::uniform_real_distribution<double> udist(0.0, 1.0);
    std::vector<double> A(n * n, 0.0);
    if (n <= 1) return A;
    double p = std::max(0.0, std::min(1.0, avg_degree / static_cast<double>(n - 1)));
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (udist(rng) < p) {
                double w = wdist(rng);
                A[i * n + j] = w;
                A[j * n + i] = w;
            }
        }
    }
    return A;
}

// Count undirected edges (i<j with positive weight)
static int countUndirectedEdges(const std::vector<double>& M, int n) {
    int e = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) if (M[i * n + j] > 0.0) ++e;
    }
    return e;
}

// Generate connected graph: random spanning tree + ER fill to match expected avg degree
static std::vector<double> generateConnectedAdjacencyWithSpanningTree(int n, double avg_degree, unsigned seed) {
    std::vector<double> A(n * n, 0.0);
    if (n <= 1) return A;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> wdist(0.0, 1.0);
    std::uniform_int_distribution<int> parentDist;

    // Spanning tree: for v=1..n-1, connect to random parent in [0..v-1]
    for (int v = 1; v < n; ++v) {
        parentDist = std::uniform_int_distribution<int>(0, v - 1);
        int p = parentDist(rng);
        double w = wdist(rng);
        A[p * n + v] = w;
        A[v * n + p] = w;
    }

    // Add extra edges with probability to reach target expected edges
    const double target_edges = std::max(0.0, (avg_degree * n) / 2.0);
    int current_edges = countUndirectedEdges(A, n);
    const int total_pairs = (n * (n - 1)) / 2;
    int remaining_pairs = total_pairs - current_edges;
    double remaining_needed = std::max(0.0, target_edges - static_cast<double>(current_edges));
    double p_extra = (remaining_pairs > 0) ? std::min(1.0, remaining_needed / static_cast<double>(remaining_pairs)) : 0.0;

    std::uniform_real_distribution<double> udist(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (A[i * n + j] == 0.0 && udist(rng) < p_extra) {
                double w = wdist(rng);
                A[i * n + j] = w;
                A[j * n + i] = w;
            }
        }
    }
    return A;
}

static void extendAdjacencyWithNewVertices(std::vector<double>& adjacency, int& num_vertices, int num_new_vertices, double target_avg_degree, unsigned seed, bool ensure_connected) {
    if (num_new_vertices <= 0) return;
    int oldN = num_vertices;
    int newN = num_vertices + num_new_vertices;
    std::vector<double> B(newN * newN, 0.0);
    for (int i = 0; i < oldN; ++i) {
        std::copy_n(&adjacency[i * oldN], oldN, &B[i * newN]);
    }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> wdist(0.0, 1.0);
    std::uniform_real_distribution<double> udist(0.0, 1.0);
    if (ensure_connected) {
        // Ensure connectivity by attaching each new vertex to an existing/previous vertex (spanning tree extension)
        for (int v = oldN; v < newN; ++v) {
            std::uniform_int_distribution<int> parentDist(0, v - 1);
            int p = parentDist(rng);
            double w = wdist(rng);
            B[p * newN + v] = w;
            B[v * newN + p] = w;
        }
        // Fill extra edges to target expected avg degree
        int current_edges = countUndirectedEdges(B, newN);
        const double target_edges = std::max(0.0, (target_avg_degree * newN) / 2.0);
        const int total_pairs = (newN * (newN - 1)) / 2;
        int remaining_pairs = total_pairs - current_edges;
        double remaining_needed = std::max(0.0, target_edges - static_cast<double>(current_edges));
        double p_extra = (remaining_pairs > 0) ? std::min(1.0, remaining_needed / static_cast<double>(remaining_pairs)) : 0.0;
        for (int i = 0; i < newN; ++i) {
            for (int j = i + 1; j < newN; ++j) {
                if (B[i * newN + j] == 0.0 && udist(rng) < p_extra) {
                    double w = wdist(rng);
                    B[i * newN + j] = w;
                    B[j * newN + i] = w;
                }
            }
        }
    } else {
        // Original ER addition respecting expected degree
        double p = (newN > 1) ? std::max(0.0, std::min(1.0, target_avg_degree / static_cast<double>(newN - 1))) : 0.0;
        for (int i = 0; i < oldN; ++i) {
            for (int j = oldN; j < newN; ++j) {
                if (udist(rng) < p) {
                    double w = wdist(rng);
                    B[i * newN + j] = w;
                    B[j * newN + i] = w;
                }
            }
        }
        for (int i = oldN; i < newN; ++i) {
            for (int j = i + 1; j < newN; ++j) {
                if (udist(rng) < p) {
                    double w = wdist(rng);
                    B[i * newN + j] = w;
                    B[j * newN + i] = w;
                }
            }
        }
    }
    adjacency.swap(B);
    num_vertices = newN;
}

static std::vector<double> buildGraphLaplacian(const std::vector<double>& adjacency, int num_vertices) {
    std::vector<double> laplacian(num_vertices * num_vertices, 0.0);
    for (int i = 0; i < num_vertices; ++i) {
        double rowSum = 0.0;
        for (int j = 0; j < num_vertices; ++j) rowSum += adjacency[i * num_vertices + j];
        for (int j = 0; j < num_vertices; ++j) {
            if (i == j) laplacian[i * num_vertices + j] = rowSum;
            else laplacian[i * num_vertices + j] = -adjacency[i * num_vertices + j];
        }
    }
    return laplacian;
}

static std::vector<double> extractSubmatrix(const std::vector<double>& L, int n, const std::vector<int>& rows, const std::vector<int>& cols) {
    int r = static_cast<int>(rows.size());
    int c = static_cast<int>(cols.size());
    std::vector<double> M(r * c, 0.0);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            M[i * c + j] = L[rows[i] * n + cols[j]];
        }
    }
    return M;
}

// Forward declaration for CUDA kernel launched in solveHarmonicLabels
__global__ void aggregate_labeled_rhs_kernel(const double* __restrict__ adjacency,
                                             int num_vertices,
                                             const int* __restrict__ unlabeled_idx,
                                             int num_unlabeled,
                                             const int* __restrict__ labeled_idx,
                                             const double* __restrict__ labeled_vals,
                                             int num_labeled,
                                             double* __restrict__ out_rhs);

static std::vector<double> solveHarmonicLabels(const std::vector<double>& laplacian,
                                               const std::vector<double>& adjacency,
                                               int num_vertices,
                                               const std::vector<int>& labeled_vertex_ids,
                                               const std::vector<double>& labeled_label_values,
                                               bool use_agg_kernel) {
    std::vector<char> is_labeled(num_vertices, 0);
    for (int idx : labeled_vertex_ids) if (idx >= 0 && idx < num_vertices) is_labeled[idx] = 1;
    std::vector<int> unlabeled_indices, labeled_indices;
    for (int i = 0; i < num_vertices; ++i) {
        if (is_labeled[i]) labeled_indices.push_back(i); else unlabeled_indices.push_back(i);
    }
    int num_unlabeled = static_cast<int>(unlabeled_indices.size());
    int num_labeled = static_cast<int>(labeled_indices.size());
    if (num_labeled != static_cast<int>(labeled_label_values.size())) {
        std::cerr << "xl size does not match number of labeled nodes" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (num_unlabeled == 0) return {};

    std::vector<double> laplacian_uu = extractSubmatrix(laplacian, num_vertices, unlabeled_indices, unlabeled_indices);

    double *device_luu = nullptr, *device_rhs = nullptr;
    checkCudaStatus(cudaMalloc((void**)&device_luu, sizeof(double) * num_unlabeled * num_unlabeled), "malloc device_luu");
    checkCudaStatus(cudaMalloc((void**)&device_rhs, sizeof(double) * num_unlabeled), "malloc device_rhs");
    checkCudaStatus(cudaMemcpy(device_luu, laplacian_uu.data(), sizeof(double) * num_unlabeled * num_unlabeled, cudaMemcpyHostToDevice), "copy Luu");

    if (use_agg_kernel) {
        double* device_adjacency = nullptr;
        int *device_unlabeled_idx = nullptr, *device_labeled_idx = nullptr;
        double* device_xl = nullptr;
        checkCudaStatus(cudaMalloc((void**)&device_adjacency, sizeof(double) * num_vertices * num_vertices), "malloc adjacency");
        checkCudaStatus(cudaMalloc((void**)&device_unlabeled_idx, sizeof(int) * num_unlabeled), "malloc unlabeled_idx");
        checkCudaStatus(cudaMalloc((void**)&device_labeled_idx, sizeof(int) * num_labeled), "malloc labeled_idx");
        checkCudaStatus(cudaMalloc((void**)&device_xl, sizeof(double) * num_labeled), "malloc xl");
        checkCudaStatus(cudaMemcpy(device_adjacency, adjacency.data(), sizeof(double) * num_vertices * num_vertices, cudaMemcpyHostToDevice), "copy adjacency");
        checkCudaStatus(cudaMemcpy(device_unlabeled_idx, unlabeled_indices.data(), sizeof(int) * num_unlabeled, cudaMemcpyHostToDevice), "copy unlabeled_idx");
        checkCudaStatus(cudaMemcpy(device_labeled_idx, labeled_indices.data(), sizeof(int) * num_labeled, cudaMemcpyHostToDevice), "copy labeled_idx");
        checkCudaStatus(cudaMemcpy(device_xl, labeled_label_values.data(), sizeof(double) * num_labeled, cudaMemcpyHostToDevice), "copy xl");
        dim3 grid(num_unlabeled);
        dim3 block(256);
        aggregate_labeled_rhs_kernel<<<grid, block>>>(device_adjacency, num_vertices, device_unlabeled_idx, num_unlabeled,
                                                      device_labeled_idx, device_xl, num_labeled, device_rhs);
        checkCudaStatus(cudaDeviceSynchronize(), "aggregate kernel sync");
        cudaFree(device_adjacency);
        cudaFree(device_unlabeled_idx);
        cudaFree(device_labeled_idx);
        cudaFree(device_xl);
    } else {
        std::vector<double> laplacian_ul = extractSubmatrix(laplacian, num_vertices, unlabeled_indices, labeled_indices);
        double *device_lul = nullptr, *device_xl = nullptr;
        checkCudaStatus(cudaMalloc((void**)&device_lul, sizeof(double) * num_unlabeled * num_labeled), "malloc device_lul");
        checkCudaStatus(cudaMalloc((void**)&device_xl, sizeof(double) * num_labeled), "malloc device_xl");
        checkCudaStatus(cudaMemcpy(device_lul, laplacian_ul.data(), sizeof(double) * num_unlabeled * num_labeled, cudaMemcpyHostToDevice), "copy Lul");
        checkCudaStatus(cudaMemcpy(device_xl, labeled_label_values.data(), sizeof(double) * num_labeled, cudaMemcpyHostToDevice), "copy xl");
        cublasHandle_t cublas_handle = nullptr;
        checkCublasStatus(cublasCreate(&cublas_handle), "create cublas");
        const double alpha = -1.0, beta = 0.0;
        checkCublasStatus(cublasDgemv(cublas_handle,
                                      CUBLAS_OP_T,
                                      num_labeled,
                                      num_unlabeled,
                                      &alpha,
                                      device_lul,
                                      num_labeled,
                                      device_xl,
                                      1,
                                      &beta,
                                      device_rhs,
                                      1), "gemv b = -Lul*xl");
        cublasDestroy(cublas_handle);
        cudaFree(device_lul);
        cudaFree(device_xl);
    }

    cusolverDnHandle_t cusolver_handle = nullptr;
    checkCusolverStatus(cusolverDnCreate(&cusolver_handle), "create cusolver");
    int lwork = 0, *device_pivots = nullptr, *device_info = nullptr;
    checkCudaStatus(cudaMalloc((void**)&device_pivots, sizeof(int) * num_unlabeled), "malloc pivots");
    checkCudaStatus(cudaMalloc((void**)&device_info, sizeof(int)), "malloc info");
    checkCusolverStatus(cusolverDnDgetrf_bufferSize(cusolver_handle, num_unlabeled, num_unlabeled, device_luu, num_unlabeled, &lwork), "bufferSize");
    double* device_workspace = nullptr;
    checkCudaStatus(cudaMalloc((void**)&device_workspace, sizeof(double) * lwork), "malloc workspace");
    checkCusolverStatus(cusolverDnDgetrf(cusolver_handle, num_unlabeled, num_unlabeled, device_luu, num_unlabeled, device_workspace, device_pivots, device_info), "getrf");
    checkCusolverStatus(cusolverDnDgetrs(cusolver_handle, CUBLAS_OP_N, num_unlabeled, 1, device_luu, num_unlabeled, device_pivots, device_rhs, num_unlabeled, device_info), "getrs");

    std::vector<double> xu(num_unlabeled, 0.0);
    checkCudaStatus(cudaMemcpy(xu.data(), device_rhs, sizeof(double) * num_unlabeled, cudaMemcpyDeviceToHost), "copy xu");

    cudaFree(device_luu);
    cudaFree(device_rhs);
    cudaFree(device_pivots);
    cudaFree(device_info);
    cudaFree(device_workspace);
    cusolverDnDestroy(cusolver_handle);

    std::vector<double> labels(num_vertices, 0.0);
    for (int i = 0; i < num_labeled; ++i) labels[labeled_indices[i]] = labeled_label_values[i];
    for (int i = 0; i < num_unlabeled; ++i) labels[unlabeled_indices[i]] = xu[i];
    return labels;
}

static void printVectorDouble(const std::vector<double>& v, const char* name) {
    std::cout << name << " [" << v.size() << "]:" << std::endl;
    for (double x : v) std::cout << x << " ";
    std::cout << std::endl;
}

// Aggregate labeled influence into RHS for each unlabeled node:
// b[u_i] = sum_{j in L} adjacency[u_i, labeled_idx[j]] * labeled_vals[j]
__global__ void aggregate_labeled_rhs_kernel(const double* __restrict__ adjacency,
                                             int num_vertices,
                                             const int* __restrict__ unlabeled_idx,
                                             int num_unlabeled,
                                             const int* __restrict__ labeled_idx,
                                             const double* __restrict__ labeled_vals,
                                             int num_labeled,
                                             double* __restrict__ out_rhs) {
    int ui = blockIdx.x;
    if (ui >= num_unlabeled) return;
    int u = unlabeled_idx[ui];
    double sum = 0.0;
    for (int j = threadIdx.x; j < num_labeled; j += blockDim.x) {
        int l = labeled_idx[j];
        sum += adjacency[u * num_vertices + l] * labeled_vals[j];
    }
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    if (tid < 256) sdata[tid] = 0.0;
    __syncthreads();
    if (tid < 256) sdata[tid] = sum;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) sdata[tid] += sdata[tid + offset];
        __syncthreads();
    }
    if (tid == 0) out_rhs[ui] = sdata[0];
}

template <typename T>
struct is_vector : std::false_type {};

template <typename T, typename A>
struct is_vector<std::vector<T, A>> : std::true_type {};

// Function to transpose a vector
template <typename T>
typename std::enable_if<!is_vector<typename T::value_type>::value, std::vector<std::vector<typename T::value_type>>>::type
transpose(const T& container) {
    std::vector<std::vector<typename T::value_type>> result(container.size(), std::vector<typename T::value_type>(1));
    for (size_t i = 0; i < container.size(); ++i) {
        result[i][0] = container[i];
    }
    return result;
}

// Function to transpose a matrix
template <typename T>
typename std::enable_if<is_vector<typename T::value_type>::value, std::vector<std::vector<typename T::value_type::value_type>>>::type
transpose(const T& container) {
    if (container.empty()) return {};

    size_t rows = container.size();
    size_t cols = container[0].size();
    std::vector<std::vector<typename T::value_type::value_type>> result(cols, std::vector<typename T::value_type::value_type>(rows));
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = container[i][j];
        }
    }
    
    return result;
}


// Function to get the dimension of a vector
template <typename T>
typename std::enable_if<!is_vector<typename T::value_type>::value, std::pair<int, int>>::type
getDimension(const T& container) {
    return {static_cast<int>(container.size()), 1}; // Return size and 1 (indicating it's a vector)
}

// Function to get the dimension of a matrix
template <typename T>
typename std::enable_if<is_vector<typename T::value_type>::value, std::pair<int, int>>::type
getDimension(const T& container) {
    int rows = static_cast<int>(container.size());
    int cols = rows > 0 ? static_cast<int>(container[0].size()) : 0; // If there are rows, get the column size from the first row
    return {rows, cols}; // Return number of rows and columns
}

std::vector<int> getDifference(const std::vector<int>& vec1, int n) {
    std::set<int> excludeSet(vec1.begin(), vec1.end());
    std::vector<int> result;
    
    for (int i = 0; i < n; ++i) {
        if (excludeSet.find(i) == excludeSet.end()) {
            result.push_back(i);
        }
    }
    
    return result;
}
std::vector<int> generateRandomBinaryVector(int n) {
    std::vector<int> result(n);
    std::random_device rd;  // Random number generator device
    std::mt19937 gen(rd()); // Mersenne Twister random number generator
    std::uniform_int_distribution<> dis(0, 1); // Distribution for 0 and 1

    for (int i = 0; i < n; ++i) {
        result[i] = dis(gen);
    }
    
    return result;
}


__global__ void generateRandomMatrixKernel(int* matrix, int n, int m, int r1, int r2, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * m;
    if (idx < total_elements) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        matrix[idx] = r1 + curand(&state) % (r2 - r1 + 1);
    }
}

__global__ void fetchSubMatrixKernel(int* matrix, int* submatrix, int* rows, int* cols, int sub_n, int sub_m, int n, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = sub_n * sub_m;
    if (idx < total_elements) {
        int row = rows[idx / sub_m];
        int col = cols[idx % sub_m];
        submatrix[idx] = matrix[row * m + col];
    }
}

void generateRandomMatrix(int* d_matrix, int n, int m, int r1, int r2) {
    int total_elements = n * m;
    unsigned long seed = time(0);

    // Define block and grid sizes
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;

    // Launch the kernel
    generateRandomMatrixKernel<<<numBlocks, blockSize>>>(d_matrix, n, m, r1, r2, seed);
    cudaDeviceSynchronize();
}

void fetchSubMatrix(int* d_matrix, int* d_submatrix, const std::vector<int>& rows, const std::vector<int>& cols, int sub_n, int sub_m, int n, int m) {
    int* d_rows;
    int* d_cols;
    int total_elements = sub_n * sub_m;

    // Allocate device memory for row and column indices
    cudaMalloc(&d_rows, sub_n * sizeof(int));
    cudaMalloc(&d_cols, sub_m * sizeof(int));

    // Copy row and column indices to device
    cudaMemcpy(d_rows, rows.data(), sub_n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, cols.data(), sub_m * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;

    // Launch the kernel to fetch the submatrix
    fetchSubMatrixKernel<<<numBlocks, blockSize>>>(d_matrix, d_submatrix, d_rows, d_cols, sub_n, sub_m, n, m);
    cudaDeviceSynchronize();

    // Free device memory for row and column indices
    cudaFree(d_rows);
    cudaFree(d_cols);
}

__global__ void matrixMultiplyKernel(int* A, int* B, int* C, int n, int m, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < p) {
        int value = 0;
        for (int k = 0; k < m; ++k) {
            value += A[row * m + k] * B[k * p + col];
        }
        C[row * p + col] = value;
    }
}

void parallelMatrixMult(int* d_A, int* d_B, int* d_C, int n, int m, int p) {
    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 numBlocks((p + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    // Launch the kernel to perform matrix multiplication
    matrixMultiplyKernel<<<numBlocks, blockSize>>>(d_A, d_B, d_C, n, m, p);
    cudaDeviceSynchronize();
}

__global__ void invertMatrixKernel(int* d_matrix, int* d_inv_matrix, int n) {
    extern __shared__ float shared[];

    int i = threadIdx.x;
    int j = threadIdx.y;
    int index = i * n + j;

    shared[index] = d_matrix[index];
    __syncthreads();

    for (int k = 0; k < n; ++k) {
        if (i == k) {
            for (int l = 0; l < n; ++l) {
                d_inv_matrix[index * n + l] /= shared[k * n + k];
            }
        }
        __syncthreads();

        if (i != k) {
            float factor = shared[i * n + k];
            for (int l = 0; l < n; ++l) {
                d_inv_matrix[index * n + l] -= factor * d_inv_matrix[k * n + l];
            }
        }
        __syncthreads();
    }
}

bool isInvertible(int* d_matrix, int n) {
    // Host memory for the determinant
    float det;
    cudaMemcpy(&det, d_matrix, sizeof(float), cudaMemcpyDeviceToHost);
    return det != 0;
}

void invertMatrix(int* d_matrix, int* d_inv_matrix, int n) {
    dim3 blockSize(n, n);
    int sharedSize = n * n * sizeof(float);

    invertMatrixKernel<<<1, blockSize, sharedSize>>>(d_matrix, d_inv_matrix, n);
    cudaDeviceSynchronize();
}

void printMatrix(const std::vector<std::vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}


int main(int argc, char** argv) {
    int num_vertices = 10;
    int num_labeled = 3;
    int num_new_vertices = 0;
    unsigned seed = 1234;
    double target_avg_degree = 0.0; // 0 -> dense random weights, >0 -> ER with expected degree
    bool ensure_connected = false;
    double delta_labeled_pct = 0.1; // fraction of new vertices labeled per delta
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--num-vertices" || arg == "--n") && i + 1 < argc) num_vertices = std::atoi(argv[++i]);
        else if ((arg == "--num-labeled" || arg == "--labeled") && i + 1 < argc) num_labeled = std::atoi(argv[++i]);
        else if ((arg == "--num-new" || arg == "--delta") && i + 1 < argc) num_new_vertices = std::atoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = static_cast<unsigned>(std::stoul(argv[++i]));
        else if ((arg == "--avg-degree" || arg == "--avgdeg") && i + 1 < argc) target_avg_degree = std::atof(argv[++i]);
        else if (arg == "--connected") ensure_connected = true;
        else if (arg == "--delta-labeled-pct" && i + 1 < argc) delta_labeled_pct = std::atof(argv[++i]);
    }
    num_labeled = std::max(0, std::min(num_labeled, num_vertices));

    std::vector<double> adjacency = ensure_connected ? generateConnectedAdjacencyWithSpanningTree(num_vertices, target_avg_degree, seed)
                                                     : generateErdosRenyiAdjacency(num_vertices, target_avg_degree, seed);
    std::vector<double> laplacian = buildGraphLaplacian(adjacency, num_vertices);

    std::vector<int> labeled_vertex_ids;
    for (int i = 0; i < num_labeled; ++i) labeled_vertex_ids.push_back(i);
    std::vector<double> labeled_label_values(num_labeled, 0.0);
    for (int i = 0; i < num_labeled; ++i) labeled_label_values[i] = (i % 2 == 0) ? 1.0 : 0.0;

    std::vector<double> labels_t = solveHarmonicLabels(laplacian, adjacency, num_vertices, labeled_vertex_ids, labeled_label_values, true);
    std::cout << "labels_t [" << labels_t.size() << "]:" << std::endl;
    for (double v : labels_t) std::cout << v << " ";
    std::cout << std::endl;

    if (num_new_vertices > 0) {
        int prev_num_vertices = num_vertices;
        extendAdjacencyWithNewVertices(adjacency, num_vertices, num_new_vertices, target_avg_degree, seed + 1, ensure_connected);
        laplacian = buildGraphLaplacian(adjacency, num_vertices);
        // Build t+1 labeled set: keep old labeled, plus a fraction of new vertices
        std::vector<int> labeled_vertex_ids_t1;
        for (int idx : labeled_vertex_ids) if (idx < num_vertices) labeled_vertex_ids_t1.push_back(idx);
        std::vector<double> labeled_label_values_t1 = labeled_label_values;
        int new_count = num_vertices - prev_num_vertices;
        if (new_count > 0 && delta_labeled_pct > 0.0) {
            int k_gt = static_cast<int>(delta_labeled_pct * static_cast<double>(new_count) + 0.5);
            if (k_gt > new_count) k_gt = new_count;
            // Deterministic selection: first k_gt new vertices; assign balanced 1/0 labels
            int num_pos = k_gt / 2;
            int num_zero = k_gt - num_pos;
            for (int j = 0; j < k_gt; ++j) {
                int v = prev_num_vertices + j;
                labeled_vertex_ids_t1.push_back(v);
                double lbl = (j < num_pos) ? 1.0 : 0.0;
                labeled_label_values_t1.push_back(lbl);
            }
        }
        std::vector<double> labels_t_plus_1 = solveHarmonicLabels(laplacian, adjacency, num_vertices, labeled_vertex_ids_t1, labeled_label_values_t1, true);
        std::cout << "labels_t+1 [" << labels_t_plus_1.size() << "]:" << std::endl;
        for (double v : labels_t_plus_1) std::cout << v << " ";
        std::cout << std::endl;
    }
    return 0;
}
