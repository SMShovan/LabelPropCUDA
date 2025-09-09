#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse_v2.h>

inline void checkCudaStatus(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << ": " << cudaGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline void checkCublasStatus(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error: " << msg << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline void checkCusolverStatus(cusolverStatus_t status, const char* msg) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cuSOLVER error: " << msg << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline void checkCusparseStatus(cusparseStatus_t status, const char* msg) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "cuSPARSE error: " << msg << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


