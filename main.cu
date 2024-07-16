#include <iostream>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>

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

int main() {
    int n = 10, m = 15, r1 = 0, r2 = 50;
    int total_elements = n * m;

    // Allocate host memory for the main matrix
    std::vector<std::vector<int>> h_matrix(n, std::vector<int>(m));
    int* h_flat_matrix = new int[total_elements];

    // Allocate device memory for the main matrix
    int* d_matrix;
    cudaMalloc(&d_matrix, total_elements * sizeof(int));

    // Generate the random matrix on the GPU
    generateRandomMatrix(d_matrix, n, m, r1, r2);

    // Copy the matrix back to host
    cudaMemcpy(h_flat_matrix, d_matrix, total_elements * sizeof(int), cudaMemcpyDeviceToHost);

    // Convert flat matrix to 2D vector for printing
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            h_matrix[i][j] = h_flat_matrix[i * m + j];
        }
    }

    // Print the generated matrix
    std::cout << "Generated Matrix:" << std::endl;
    printMatrix(h_matrix);

    // Define the first submatrix row and column indices
    std::vector<int> rows1 = {1, 3, 5};
    std::vector<int> cols1 = {2, 4, 6};
    int sub_n1 = rows1.size();
    int sub_m1 = cols1.size();
    int sub_total_elements1 = sub_n1 * sub_m1;

    // Allocate host memory for the first submatrix
    std::vector<std::vector<int>> h_submatrix1(sub_n1, std::vector<int>(sub_m1));
    int* h_flat_submatrix1 = new int[sub_total_elements1];

    // Allocate device memory for the first submatrix
    int* d_submatrix1;
    cudaMalloc(&d_submatrix1, sub_total_elements1 * sizeof(int));

    // Fetch the first submatrix on the GPU
    fetchSubMatrix(d_matrix, d_submatrix1, rows1, cols1, sub_n1, sub_m1, n, m);

    // Copy the first submatrix back to host
    cudaMemcpy(h_flat_submatrix1, d_submatrix1, sub_total_elements1 * sizeof(int), cudaMemcpyDeviceToHost);

    // Convert flat submatrix to 2D vector for printing
    for (int i = 0; i < sub_n1; ++i) {
        for (int j = 0; j < sub_m1; ++j) {
            h_submatrix1[i][j] = h_flat_submatrix1[i * sub_m1 + j];
        }
    }

    // Print the first fetched submatrix
    std::cout << "First Fetched Submatrix:" << std::endl;
    printMatrix(h_submatrix1);

    // Define the second submatrix row and column indices
    std::vector<int> rows2 = {0, 2, 4};
    std::vector<int> cols2 = {1, 3, 5};
    int sub_n2 = rows2.size();
    int sub_m2 = cols2.size();
    int sub_total_elements2 = sub_n2 * sub_m2;

    // Allocate host memory for the second submatrix
    std::vector<std::vector<int>> h_submatrix2(sub_n2, std::vector<int>(sub_m2));
    int* h_flat_submatrix2 = new int[sub_total_elements2];

    // Allocate device memory for the second submatrix
    int* d_submatrix2;
    cudaMalloc(&d_submatrix2, sub_total_elements2 * sizeof(int));

    // Fetch the second submatrix on the GPU
    fetchSubMatrix(d_matrix, d_submatrix2, rows2, cols2, sub_n2, sub_m2, n, m);

    // Copy the second submatrix back to host
    cudaMemcpy(h_flat_submatrix2, d_submatrix2, sub_total_elements2 * sizeof(int), cudaMemcpyDeviceToHost);

    // Convert flat submatrix to 2D vector for printing
    for (int i = 0; i < sub_n2; ++i) {
        for (int j = 0; j < sub_m2; ++j) {
            h_submatrix2[i][j] = h_flat_submatrix2[i * sub_m2 + j];
        }
    }

    // Print the second fetched submatrix
    std::cout << "Second Fetched Submatrix:" << std::endl;
    printMatrix(h_submatrix2);

    // Ensure the number of columns in the first submatrix equals the number of rows in the second submatrix
    if (sub_m1 != sub_n2) {
        std::cerr << "Error: Submatrix dimensions do not allow multiplication." << std::endl;
        return -1;
    }

    // Allocate host memory for the result submatrix
    std::vector<std::vector<int>> h_result(sub_n1, std::vector<int>(sub_m2));
    int* h_flat_result = new int[sub_n1 * sub_m2];

    // Allocate device memory for the result submatrix
    int* d_result;
    cudaMalloc(&d_result, sub_n1 * sub_m2 * sizeof(int));

    // Perform matrix multiplication on the GPU
    parallelMatrixMult(d_submatrix1, d_submatrix2, d_result, sub_n1, sub_m1, sub_m2);

    // Copy the result submatrix back to host
    cudaMemcpy(h_flat_result, d_result, sub_n1 * sub_m2 * sizeof(int), cudaMemcpyDeviceToHost);

    // Convert flat result submatrix to 2D vector for printing
    for (int i = 0; i < sub_n1; ++i) {
        for (int j = 0; j < sub_m2; ++j) {
            h_result[i][j] = h_flat_result[i * sub_m2 + j];
        }
    }

    // Print the result submatrix
    std::cout << "Result of Submatrix Multiplication:" << std::endl;
    printMatrix(h_result);

    // Check if the result matrix is invertible
    if (isInvertible(d_result, sub_n1)) {
        std::cout << "The result matrix is invertible." << std::endl;

        // Allocate device memory for the inverted matrix
        int* d_inv_result;
        cudaMalloc(&d_inv_result, sub_n1 * sub_m2 * sizeof(int));

        // Invert the result matrix on the GPU
        invertMatrix(d_result, d_inv_result, sub_n1);

        // Allocate host memory for the inverted matrix
        int* h_flat_inv_result = new int[sub_n1 * sub_m2];
        std::vector<std::vector<int>> h_inv_result(sub_n1, std::vector<int>(sub_m2));

        // Copy the inverted matrix back to host
        cudaMemcpy(h_flat_inv_result, d_inv_result, sub_n1 * sub_m2 * sizeof(int), cudaMemcpyDeviceToHost);

        // Convert flat inverted matrix to 2D vector for printing
        for (int i = 0; i < sub_n1; ++i) {
            for (int j = 0; j < sub_m2; ++j) {
                h_inv_result[i][j] = h_flat_inv_result[i * sub_m2 + j];
            }
        }

        // Print the inverted matrix
        std::cout << "Inverted Result Matrix:" << std::endl;
        printMatrix(h_inv_result);

        // Free memory for the inverted matrix
        delete[] h_flat_inv_result;
        cudaFree(d_inv_result);
    } else {
        std::cout << "The result matrix is not invertible." << std::endl;
    }

    // Free memory
    delete[] h_flat_matrix;
    delete[] h_flat_submatrix1;
    delete[] h_flat_submatrix2;
    delete[] h_flat_result;
    cudaFree(d_matrix);
    cudaFree(d_submatrix1);
    cudaFree(d_submatrix2);
    cudaFree(d_result);

    return 0;
}
