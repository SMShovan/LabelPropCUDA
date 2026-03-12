/**
 * @file validation.cu
 * @brief IrLP (Iterative Label Propagation) baseline implementation (CUDA).
 *
 * See validation.h for full documentation.
 *
 * ============================================================================
 * KERNEL DESIGN
 * ============================================================================
 *
 * irLPKernel:
 *   - One block per vertex (block-per-row model, same as DynLP propagation).
 *   - Each block cooperatively computes the weighted average of neighbors'
 *     labels using shared memory reduction.
 *   - Uses double-buffering to avoid read-write races.
 *
 * Unlike DynLP's Step 3, IrLP processes ALL unlabeled vertices in every
 * iteration — not just affected ones. This is why IrLP is slower but
 * provides the optimal solution for accuracy comparison.
 *
 * ============================================================================
 */

#include "validation.h"

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

static const int IRLP_BLOCK_SIZE = 256;

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief IrLP iteration kernel: weighted average of neighbor labels.
 *
 * One block per vertex. Threads cooperatively sum neighbor weights and
 * weighted labels, then compute F'_u = weightedSum / totalWeight.
 *
 * Double-buffered: reads from d_labelsOld, writes to d_labelsNew.
 */
__global__ void irLPKernel(const int *d_rowPtr, const int *d_colInd,
                            const float *d_values, const float *d_labelsOld,
                            float *d_labelsNew, const int *d_vertexStatus,
                            const int *d_gtClass, int numNodes,
                            int *d_changed, float delta) {
  int u = blockIdx.x;
  if (u >= numNodes) return;

  // Ground-truth vertices keep their labels
  if (d_gtClass[u] >= 0) {
    if (threadIdx.x == 0) {
      d_labelsNew[u] = d_labelsOld[u];
    }
    return;
  }

  // Deleted/inactive vertices keep neutral labels
  if (d_vertexStatus[u] != VERTEX_ACTIVE) {
    if (threadIdx.x == 0) {
      d_labelsNew[u] = d_labelsOld[u];
    }
    return;
  }

  int rowStart = d_rowPtr[u];
  int rowEnd = d_rowPtr[u + 1];

  // Shared memory for reduction
  __shared__ float s_wAll[IRLP_BLOCK_SIZE];
  __shared__ float s_weightedLabel[IRLP_BLOCK_SIZE];

  float localWAll = 0.0f;
  float localWeightedLabel = 0.0f;

  // Block-strided neighbor traversal
  for (int e = rowStart + threadIdx.x; e < rowEnd; e += blockDim.x) {
    int v = d_colInd[e];
    if (v < 0) continue;
    if (d_vertexStatus[v] == VERTEX_DELETED) continue;

    float w = d_values[e];
    float Fv = d_labelsOld[v];

    localWAll += w;
    localWeightedLabel += w * Fv;
  }

  s_wAll[threadIdx.x] = localWAll;
  s_weightedLabel[threadIdx.x] = localWeightedLabel;
  __syncthreads();

  // Block-level reduction
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_wAll[threadIdx.x] += s_wAll[threadIdx.x + stride];
      s_weightedLabel[threadIdx.x] += s_weightedLabel[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float wAll = s_wAll[0];
    float weightedLabel = s_weightedLabel[0];
    float oldLabel = d_labelsOld[u];

    float newLabel;
    if (wAll > 1e-10f) {
      newLabel = weightedLabel / wAll;
    } else {
      newLabel = oldLabel; // Isolated vertex
    }

    // Clamp to [0, 1]
    newLabel = fminf(fmaxf(newLabel, 0.0f), 1.0f);
    d_labelsNew[u] = newLabel;

    if (fabsf(newLabel - oldLabel) > delta) {
      atomicMax(d_changed, 1);
    }
  }
}

/**
 * @brief Enforce ground-truth labels after each iteration.
 */
__global__ void irLPEnforceGTKernel(float *d_labels, const int *d_gtClass,
                                     int numNodes) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= numNodes) return;

  if (d_gtClass[v] == 0) d_labels[v] = 0.0f;
  else if (d_gtClass[v] == 1) d_labels[v] = 1.0f;
}

// ============================================================================
// Host Launch Functions
// ============================================================================

void launchIrLP(const int *d_rowPtr, const int *d_colInd,
                const float *d_values, float *d_labels,
                const int *d_vertexStatus, const int *d_gtClass,
                int numNodes, float delta, int maxIterations,
                int &iterationsUsed, cudaStream_t stream) {

  int threadsPerBlock = 256;
  int numBlocksAll = (numNodes + threadsPerBlock - 1) / threadsPerBlock;

  // Allocate buffer for double-buffering
  float *d_labelsBuffer;
  CUDA_CHECK(cudaMalloc(&d_labelsBuffer, numNodes * sizeof(float)));
  CUDA_CHECK(cudaMemcpyAsync(d_labelsBuffer, d_labels,
                              numNodes * sizeof(float),
                              cudaMemcpyDeviceToDevice, stream));

  int *d_changed;
  CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

  float *d_labelsOld = d_labels;
  float *d_labelsNew = d_labelsBuffer;

  iterationsUsed = 0;

  for (int iter = 0; iter < maxIterations; ++iter) {
    CUDA_CHECK(cudaMemsetAsync(d_changed, 0, sizeof(int), stream));

    // One block per vertex
    irLPKernel<<<numNodes, IRLP_BLOCK_SIZE, 0, stream>>>(
        d_rowPtr, d_colInd, d_values, d_labelsOld, d_labelsNew,
        d_vertexStatus, d_gtClass, numNodes, d_changed, delta);
    CUDA_KERNEL_CHECK();

    // Enforce GT labels
    irLPEnforceGTKernel<<<numBlocksAll, threadsPerBlock, 0, stream>>>(
        d_labelsNew, d_gtClass, numNodes);
    CUDA_KERNEL_CHECK();

    // Swap buffers
    float *temp = d_labelsOld;
    d_labelsOld = d_labelsNew;
    d_labelsNew = temp;

    iterationsUsed = iter + 1;

    // Check convergence
    int changed = 0;
    CUDA_CHECK(cudaMemcpyAsync(&changed, d_changed, sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (changed == 0) break;
  }

  // Ensure final labels are in d_labels
  if (d_labelsOld != d_labels) {
    CUDA_CHECK(cudaMemcpyAsync(d_labels, d_labelsOld,
                                numNodes * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
  }

  CUDA_CHECK(cudaFree(d_labelsBuffer));
  CUDA_CHECK(cudaFree(d_changed));
}

bool runIrLPValidation(const CSRGraph &activeGraph,
                       const vector<int> &h_vertexStatus,
                       const vector<int> &h_gtClass,
                       vector<float> &h_labels, int numNodes,
                       float delta, int maxIterations, int &iterationsUsed) {

  if (numNodes <= 0 || activeGraph.numEdges <= 0) {
    cout << "Warning: IrLP validation skipped (empty graph).\n";
    return true;
  }

  // Allocate device memory
  int *d_rowPtr, *d_colInd;
  float *d_values, *d_labels;
  int *d_vertexStatus, *d_gtClass;

  CUDA_CHECK(cudaMalloc(&d_rowPtr, (numNodes + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_colInd, activeGraph.numEdges * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_values, activeGraph.numEdges * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_labels, numNodes * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_vertexStatus, numNodes * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_gtClass, numNodes * sizeof(int)));

  // Transfer data
  CUDA_CHECK(cudaMemcpy(d_rowPtr, activeGraph.rowPtr.data(),
                         (numNodes + 1) * sizeof(int),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_colInd, activeGraph.colInd.data(),
                         activeGraph.numEdges * sizeof(int),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, activeGraph.values.data(),
                         activeGraph.numEdges * sizeof(float),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(),
                         numNodes * sizeof(float),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vertexStatus, h_vertexStatus.data(),
                         numNodes * sizeof(int),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gtClass, h_gtClass.data(),
                         numNodes * sizeof(int),
                         cudaMemcpyHostToDevice));

  // Run IrLP
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  launchIrLP(d_rowPtr, d_colInd, d_values, d_labels, d_vertexStatus,
             d_gtClass, numNodes, delta, maxIterations, iterationsUsed,
             stream);

  // Retrieve results
  CUDA_CHECK(cudaMemcpy(h_labels.data(), d_labels,
                         numNodes * sizeof(float),
                         cudaMemcpyDeviceToHost));

  // Cleanup
  CUDA_CHECK(cudaFree(d_rowPtr));
  CUDA_CHECK(cudaFree(d_colInd));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_labels));
  CUDA_CHECK(cudaFree(d_vertexStatus));
  CUDA_CHECK(cudaFree(d_gtClass));
  CUDA_CHECK(cudaStreamDestroy(stream));

  return true;
}
