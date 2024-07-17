#include <iostream>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include <set>
#include <random>
#include <type_traits>
using namespace std;

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

    

    // // Allocate host memory for the result submatrix
    // std::vector<std::vector<int>> h_result(sub_n1, std::vector<int>(sub_m2));
    // int* h_flat_result = new int[sub_n1 * sub_m2];

    // // Allocate device memory for the result submatrix
    // int* d_result;
    // cudaMalloc(&d_result, sub_n1 * sub_m2 * sizeof(int));

    // // Perform matrix multiplication on the GPU
    // parallelMatrixMult(d_submatrix1, d_submatrix2, d_result, sub_n1, sub_m1, sub_m2);

    // // Copy the result submatrix back to host
    // cudaMemcpy(h_flat_result, d_result, sub_n1 * sub_m2 * sizeof(int), cudaMemcpyDeviceToHost);

    // // Convert flat result submatrix to 2D vector for printing
    // for (int i = 0; i < sub_n1; ++i) {
    //     for (int j = 0; j < sub_m2; ++j) {
    //         h_result[i][j] = h_flat_result[i * sub_m2 + j];
    //     }
    // }

    // // Print the result submatrix
    // std::cout << "Result of Submatrix Multiplication:" << std::endl;
    // printMatrix(h_result);

    // // Check if the result matrix is invertible
    // if (isInvertible(d_result, sub_n1)) {
    //     std::cout << "The result matrix is invertible." << std::endl;

    //     // Allocate device memory for the inverted matrix
    //     int* d_inv_result;
    //     cudaMalloc(&d_inv_result, sub_n1 * sub_m2 * sizeof(int));

    //     // Invert the result matrix on the GPU
    //     invertMatrix(d_result, d_inv_result, sub_n1);

    //     // Allocate host memory for the inverted matrix
    //     int* h_flat_inv_result = new int[sub_n1 * sub_m2];
    //     std::vector<std::vector<int>> h_inv_result(sub_n1, std::vector<int>(sub_m2));

    //     // Copy the inverted matrix back to host
    //     cudaMemcpy(h_flat_inv_result, d_inv_result, sub_n1 * sub_m2 * sizeof(int), cudaMemcpyDeviceToHost);

    //     // Convert flat inverted matrix to 2D vector for printing
    //     for (int i = 0; i < sub_n1; ++i) {
    //         for (int j = 0; j < sub_m2; ++j) {
    //             h_inv_result[i][j] = h_flat_inv_result[i * sub_m2 + j];
    //         }
    //     }

    //     // Print the inverted matrix
    //     std::cout << "Inverted Result Matrix:" << std::endl;
    //     printMatrix(h_inv_result);

    //     // Free memory for the inverted matrix
    //     delete[] h_flat_inv_result;
    //     cudaFree(d_inv_result);
    // } else {
    //     std::cout << "The result matrix is not invertible." << std::endl;
    // }


    //std::vector<int> result = getDifference(rows2, n);
    
    std::vector<int> randomVector = generateRandomBinaryVector(n);

    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::vector<std::vector<int>> mat = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    
    auto vecDim = getDimension(vec);
    auto matDim = getDimension(mat);
    
    auto transposedVec = transpose(vec);
    auto transposedMat = transpose(mat);

    cout<< "******************************* Testing Done *******************************"<<endl;

    std::vector<int> u = {3, 5, 8};
    std::vector<int> l = getDifference(u, n);


    std::cout << "dim of u :" << getDimension(u).first << " X "<< getDimension(u).second<< endl;
    std::cout << "dim of l :" << getDimension(l).first << " X "<< getDimension(l).second<< endl;
    

    // Define the first submatrix row and column indices
    
    int sub_n1 = u.size();
    int sub_m1 = l.size();
    int sub_total_elements1 = sub_n1 * sub_m1;

    // Allocate host memory for the first submatrix
    std::vector<std::vector<int>> h_submatrix1(sub_n1, std::vector<int>(sub_m1));
    int* h_flat_submatrix1 = new int[sub_total_elements1];

    // Allocate device memory for the first submatrix
    int* d_submatrix1;
    cudaMalloc(&d_submatrix1, sub_total_elements1 * sizeof(int));

    // Fetch the first submatrix on the GPU
    fetchSubMatrix(d_matrix, d_submatrix1, u, l, sub_n1, sub_m1, n, m);

    // Copy the first submatrix back to host
    cudaMemcpy(h_flat_submatrix1, d_submatrix1, sub_total_elements1 * sizeof(int), cudaMemcpyDeviceToHost);

    // Convert flat submatrix to 2D vector for printing
    for (int i = 0; i < sub_n1; ++i) {
        for (int j = 0; j < sub_m1; ++j) {
            h_submatrix1[i][j] = h_flat_submatrix1[i * sub_m1 + j];
        }
    }


    std::cout << "dim of ul :" << getDimension(h_submatrix1).first << " X "<< getDimension(h_submatrix1).second<< endl;

    int sub_n2 = u.size();
    int sub_m2 = u.size();
    int sub_total_elements2 = sub_n2 * sub_m2;

    // Allocate host memory for the first submatrix
    std::vector<std::vector<int>> h_submatrix2(sub_n2, std::vector<int>(sub_m2));
    int* h_flat_submatrix2 = new int[sub_total_elements2];

    // Allocate device memory for the first submatrix
    int* d_submatrix2;
    cudaMalloc(&d_submatrix2, sub_total_elements2 * sizeof(int));

    // Fetch the first submatrix on the GPU
    fetchSubMatrix(d_matrix, d_submatrix2, u, u, sub_n2, sub_m2, n, m);

    // Copy the first submatrix back to host
    cudaMemcpy(h_flat_submatrix2, d_submatrix2, sub_total_elements2 * sizeof(int), cudaMemcpyDeviceToHost);

    // Convert flat submatrix to 2D vector for printing
    for (int i = 0; i < sub_n2; ++i) {
        for (int j = 0; j < sub_m2; ++j) {
            h_submatrix2[i][j] = h_flat_submatrix2[i * sub_m2 + j];
        }
    }


    std::cout << "dim of uu :" << getDimension(h_submatrix2).first << " X "<< getDimension(h_submatrix2).second<< endl;

    int inv_n2 = getDimension(h_submatrix2).first;
    int inv_m2 = getDimension(h_submatrix2).second;
    int* d_inv_submatrix2;
    std::vector<std::vector<int>> h_inv_submatrix2(inv_n2, std::vector<int>(inv_m2));
    // // Check if the result matrix is invertible
    if (isInvertible(d_submatrix2, getDimension(h_submatrix2).first)) {
        std::cout << "The result matrix is invertible." << std::endl;
    
        // Allocate device memory for the inverted matrix
        
        cudaMalloc(&d_inv_submatrix2, inv_n2 * inv_m2 * sizeof(int));

        // Invert the result matrix on the GPU
        invertMatrix(d_submatrix2, d_inv_submatrix2, inv_n2);

        // Allocate host memory for the inverted matrix
        int* h_flat_inv_submatrix2 = new int[inv_n2 * inv_m2];
        

        // Copy the inverted matrix back to host
        cudaMemcpy(h_flat_inv_submatrix2, d_inv_submatrix2, inv_n2 * inv_m2 * sizeof(int), cudaMemcpyDeviceToHost);

        // Convert flat inverted matrix to 2D vector for printing
        for (int i = 0; i < inv_n2; ++i) {
            for (int j = 0; j < inv_m2; ++j) {
                h_inv_submatrix2[i][j] = h_flat_inv_submatrix2[i * inv_m2 + j];
            }
        }

        // // Print the inverted matrix
        // std::cout << "Inverted Result Matrix:" << std::endl;
        // printMatrix(h_inv_submatrix2);

    } else {
        std::cout << "The result matrix is not invertible." << std::endl;
    }

    std::cout << "dim of inv_uu :" << getDimension(h_inv_submatrix2).first << " X "<< getDimension(h_inv_submatrix2).second<< endl;


    // Allocate host memory for the result submatrix
    std::vector<std::vector<int>> h_result(sub_n2, std::vector<int>(sub_m1 ));
    int* h_flat_result = new int[sub_m1 * sub_n2];

    // Allocate device memory for the result submatrix
    int* d_result;
    cudaMalloc(&d_result, sub_m1 * sub_n2 * sizeof(int));

    // Perform matrix multiplication on the GPU
    parallelMatrixMult(d_inv_submatrix2, d_submatrix1, d_result, sub_n2, sub_n1, sub_m1);

    // Copy the result submatrix back to host
    cudaMemcpy(h_flat_result, d_result, sub_m1 * sub_n2 * sizeof(int), cudaMemcpyDeviceToHost);

    // Convert flat result submatrix to 2D vector for printing
    for (int i = 0; i < sub_n2; ++i) {
        for (int j = 0; j < sub_m1; ++j) {
            h_result[i][j] = h_flat_result[i * sub_m1 + j];
        }
    }

    std::cout << "dim of result :" << getDimension(h_result).first << " X "<< getDimension(h_result).second<< endl;


    // Allocate host memory for the result submatrix
    int res_n = getDimension(h_result).first;
    int res_m = getDimension(h_result).second;
    int vec_n = getDimension(l).first;
    int vec_m = getDimension(l).second;

    std::vector<std::vector<int>> h_result2(res_n, std::vector<int>(vec_m ));
    int* h_flat_result2 = new int[vec_m * res_n];

    // Allocate device memory for the result submatrix
    int* d_result2;
    int * d_vec;
    cudaMalloc(&d_vec, vec_m * vec_n * sizeof(int));
    cudaMemcpy(d_vec, l.data(), vec_m * vec_n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_result2, vec_m * res_n * sizeof(int));
    

    // Perform matrix multiplication on the GPU
    parallelMatrixMult(d_result, d_vec, d_result2, res_n, res_m, vec_m);

    // Copy the result submatrix back to host
    cudaMemcpy(h_flat_result2, d_result2, vec_m * res_n * sizeof(int), cudaMemcpyDeviceToHost);

    // Convert flat result submatrix to 2D vector for printing
    for (int i = 0; i < res_n; ++i) {
        for (int j = 0; j < vec_m; ++j) {
            h_result2[i][j] = h_flat_result2[i * vec_m + j];
        }
    }

    std::cout << "dim of final result :" << getDimension(h_result2).first << " X "<< getDimension(h_result2).second<< endl;

    std::cout << "Final result u:" << std::endl;
    for (const auto& row : h_result2) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    // // Free memory
    // delete[] h_flat_matrix;
    // delete[] h_flat_submatrix1;
    // delete[] h_flat_submatrix2;
    // delete[] h_flat_result;
    // cudaFree(d_matrix);
    // cudaFree(d_submatrix1);
    // cudaFree(d_submatrix2);
    // cudaFree(d_result);

    return 0;
}
