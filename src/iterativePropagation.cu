/**
 * @file iterativePropagation.cu
 * @brief Step 3 of DynLP: Iterative Label Propagation (CUDA).
 *
 * See iterativePropagation.h for full documentation.
 *
 * ============================================================================
 * KERNEL DESIGN — Block-Per-Row Execution Model
 * ============================================================================
 *
 * Following the MOSPOpenMP parallelization style and the paper's Figure 3:
 *
 * propagateLabelsKernel:
 *   - Grid: one block per affected vertex
 *   - Block: 256 threads cooperatively process the neighbor list
 *   - Each thread handles a strided subset of neighbors
 *   - Shared memory is used for partial sums (W_all, W_L0, W_L1, peer_sum)
 *   - After block-level reduction, thread 0 computes the new label
 *
 * collectAffectedKernel:
 *   - One thread per vertex
 *   - Checks if the vertex is currently flagged as affected
 *   - Compacts affected vertex IDs into a dense array using atomicAdd
 *
 * markNeighborsAffectedKernel:
 *   - One block per changed vertex
 *   - Marks all neighbors as affected for the next iteration
 *
 * ============================================================================
 * DOUBLE BUFFERING (ISSUE 5 FIX)
 * ============================================================================
 *
 * To avoid read-write races during parallel label updates:
 *   - Iteration reads from d_labelsOld (current labels)
 *   - Writes to d_labelsNew (next labels)
 *   - After the kernel, swap pointers
 *
 * This ensures all threads read consistent label values from the same
 * iteration, regardless of execution order.
 *
 * ============================================================================
 */

#include "iterativePropagation.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Block size for the propagation kernel
static const int PROPAGATE_BLOCK_SIZE = 256;

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief Collect affected vertex IDs into a dense array.
 *
 * One thread per vertex. If d_isAffected[v] == 1 and vertex is active
 * and not ground-truth, add to the affected list using atomicAdd.
 */
__global__ void collectAffectedKernel(const int *d_isAffected,
                                       const int *d_vertexStatus,
                                       const int *d_gtClass,
                                       int *d_affectedList,
                                       int *d_numAffected,
                                       int numNodes) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= numNodes) return;

  // Only process active, non-GT, affected vertices
  if (d_isAffected[v] == 1 &&
      d_vertexStatus[v] == VERTEX_ACTIVE &&
      d_gtClass[v] < 0) {
    int idx = atomicAdd(d_numAffected, 1);
    d_affectedList[idx] = v;
  }
}

/**
 * @brief Propagate labels for affected vertices using block-per-row model.
 *
 * Grid: one block per affected vertex.
 * Block: PROPAGATE_BLOCK_SIZE threads cooperatively process neighbors.
 *
 * Each thread accumulates partial sums for:
 *   - W_all: total weight of all active neighbors
 *   - weightedLabelSum: Σ w(u,v) * F_v for all active neighbors
 *
 * Then the new label is: F'_u = weightedLabelSum / W_all
 *
 * This is equivalent to the paper's update rule (proven in Section 5).
 *
 * ISSUE 2 FIX: If W_all == 0 (isolated vertex), skip the update.
 * ISSUE 7 FIX: Use absolute difference |F'_u - F_u| > δ.
 */
__global__ void propagateLabelsKernel(const int *d_rowPtr,
                                       const int *d_colInd,
                                       const float *d_values,
                                       const float *d_labelsOld,
                                       float *d_labelsNew,
                                       const int *d_vertexStatus,
                                       const int *d_gtClass,
                                       const int *d_affectedList,
                                       int numAffected,
                                       int *d_isChanged,
                                       float delta) {
  // Each block processes one affected vertex
  int affIdx = blockIdx.x;
  if (affIdx >= numAffected) return;

  int u = d_affectedList[affIdx];
  int rowStart = d_rowPtr[u];
  int rowEnd = d_rowPtr[u + 1];
  int rowLen = rowEnd - rowStart;

  // Shared memory for partial sums
  __shared__ float s_wAll[PROPAGATE_BLOCK_SIZE];
  __shared__ float s_weightedLabel[PROPAGATE_BLOCK_SIZE];

  float localWAll = 0.0f;
  float localWeightedLabel = 0.0f;

  // Block-strided traversal of the neighbor list
  for (int e = rowStart + threadIdx.x; e < rowEnd; e += blockDim.x) {
    int v = d_colInd[e];
    if (v < 0) continue;  // Skip soft-deleted edges
    if (d_vertexStatus[v] == VERTEX_DELETED) continue; // Skip deleted vertices

    float w = d_values[e];
    float Fv = d_labelsOld[v]; // Read from old buffer (double-buffering)

    localWAll += w;
    localWeightedLabel += w * Fv;
  }

  // Store in shared memory
  s_wAll[threadIdx.x] = localWAll;
  s_weightedLabel[threadIdx.x] = localWeightedLabel;
  __syncthreads();

  // Block-level reduction (power-of-two reduction tree)
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_wAll[threadIdx.x] += s_wAll[threadIdx.x + stride];
      s_weightedLabel[threadIdx.x] += s_weightedLabel[threadIdx.x + stride];
    }
    __syncthreads();
  }

  // Thread 0 computes the final label update
  if (threadIdx.x == 0) {
    float wAll = s_wAll[0];
    float weightedLabel = s_weightedLabel[0];
    float oldLabel = d_labelsOld[u];

    float newLabel;
    if (wAll > 1e-10f) {
      // F'_u = Σ α_{u,v} * F_v = (Σ w(u,v) * F_v) / (Σ w(u,v))
      newLabel = weightedLabel / wAll;
    } else {
      // ISSUE 2 FIX: Isolated vertex — keep current label
      newLabel = oldLabel;
    }

    // Clamp to [0, 1] for numerical stability
    newLabel = fminf(fmaxf(newLabel, 0.0f), 1.0f);

    // Write to new buffer
    d_labelsNew[u] = newLabel;

    // ISSUE 7 FIX: Check absolute difference for convergence
    float diff = fabsf(newLabel - oldLabel);
    if (diff > delta) {
      atomicMax(d_isChanged, 1);
    }
  }
}

/**
 * @brief Mark neighbors of changed vertices as affected for next iteration.
 *
 * One block per affected vertex. Checks if the vertex's label changed
 * significantly, and if so, marks all its neighbors as affected.
 */
__global__ void markNeighborsKernel(const int *d_rowPtr, const int *d_colInd,
                                     const float *d_labelsOld,
                                     const float *d_labelsNew,
                                     const int *d_vertexStatus,
                                     const int *d_gtClass,
                                     int *d_isAffectedNext,
                                     const int *d_affectedList,
                                     int numAffected,
                                     float delta) {
  int affIdx = blockIdx.x;
  if (affIdx >= numAffected) return;

  int u = d_affectedList[affIdx];
  float diff = fabsf(d_labelsNew[u] - d_labelsOld[u]);

  if (diff > delta) {
    // Mark all neighbors as affected for next iteration
    int rowStart = d_rowPtr[u];
    int rowEnd = d_rowPtr[u + 1];
    for (int e = rowStart + threadIdx.x; e < rowEnd; e += blockDim.x) {
      int v = d_colInd[e];
      if (v >= 0 && d_vertexStatus[v] != VERTEX_DELETED && d_gtClass[v] < 0) {
        atomicMax(&d_isAffectedNext[v], 1);
      }
    }
    // Also mark u itself in case it needs further refinement
    if (threadIdx.x == 0 && d_gtClass[u] < 0) {
      atomicMax(&d_isAffectedNext[u], 1);
    }
  }
}

/**
 * @brief Copy labels from new buffer to old buffer for non-affected vertices.
 *
 * After swapping, ensure GT vertices retain their fixed labels.
 */
__global__ void enforceGroundTruthKernel(float *d_labels,
                                          const int *d_gtClass,
                                          int numNodes) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= numNodes) return;

  if (d_gtClass[v] == 0) {
    d_labels[v] = 0.0f;
  } else if (d_gtClass[v] == 1) {
    d_labels[v] = 1.0f;
  }
}

// ============================================================================
// Host Launch Function
// ============================================================================

void launchIterativePropagation(const int *d_rowPtr, const int *d_colInd,
                                 const float *d_values, float *d_labels,
                                 float *d_labelsBuffer,
                                 const int *d_vertexStatus,
                                 const int *d_gtClass, int *d_isAffected,
                                 int numNodes, float delta,
                                 int maxIterations, int &iterationsUsed,
                                 cudaStream_t stream) {
  int threadsPerBlock = 256;
  int numBlocksAll = (numNodes + threadsPerBlock - 1) / threadsPerBlock;

  // --- Allocate working arrays ---
  int *d_affectedList;
  int *d_numAffected;
  int *d_isChanged;
  int *d_isAffectedNext;

  CUDA_CHECK(cudaMalloc(&d_affectedList, numNodes * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_numAffected, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_isChanged, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_isAffectedNext, numNodes * sizeof(int)));

  // Pointers for double buffering
  float *d_labelsOld = d_labels;
  float *d_labelsNew = d_labelsBuffer;

  // Copy current labels to buffer to initialize both buffers
  CUDA_CHECK(cudaMemcpyAsync(d_labelsNew, d_labelsOld,
                              numNodes * sizeof(float),
                              cudaMemcpyDeviceToDevice, stream));

  iterationsUsed = 0;

  for (int iter = 0; iter < maxIterations; ++iter) {
    // --- Collect affected vertices into a dense list ---
    CUDA_CHECK(cudaMemsetAsync(d_numAffected, 0, sizeof(int), stream));
    collectAffectedKernel<<<numBlocksAll, threadsPerBlock, 0, stream>>>(
        d_isAffected, d_vertexStatus, d_gtClass, d_affectedList,
        d_numAffected, numNodes);
    CUDA_KERNEL_CHECK();

    int numAffected = 0;
    CUDA_CHECK(cudaMemcpyAsync(&numAffected, d_numAffected, sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (numAffected == 0) break; // Converged — no affected vertices remain

    // --- Propagate labels (block-per-row) ---
    CUDA_CHECK(cudaMemsetAsync(d_isChanged, 0, sizeof(int), stream));

    propagateLabelsKernel<<<numAffected, PROPAGATE_BLOCK_SIZE, 0, stream>>>(
        d_rowPtr, d_colInd, d_values, d_labelsOld, d_labelsNew,
        d_vertexStatus, d_gtClass, d_affectedList, numAffected,
        d_isChanged, delta);
    CUDA_KERNEL_CHECK();

    // --- Mark neighbors of changed vertices for next iteration ---
    CUDA_CHECK(
        cudaMemsetAsync(d_isAffectedNext, 0, numNodes * sizeof(int), stream));

    markNeighborsKernel<<<numAffected, PROPAGATE_BLOCK_SIZE, 0, stream>>>(
        d_rowPtr, d_colInd, d_labelsOld, d_labelsNew, d_vertexStatus,
        d_gtClass, d_isAffectedNext, d_affectedList, numAffected, delta);
    CUDA_KERNEL_CHECK();

    // --- Swap label buffers ---
    float *temp = d_labelsOld;
    d_labelsOld = d_labelsNew;
    d_labelsNew = temp;

    // --- Swap affected arrays ---
    // Copy d_isAffectedNext to d_isAffected for the next iteration
    CUDA_CHECK(cudaMemcpyAsync(d_isAffected, d_isAffectedNext,
                                numNodes * sizeof(int),
                                cudaMemcpyDeviceToDevice, stream));

    iterationsUsed = iter + 1;

    // --- Check convergence ---
    int changed = 0;
    CUDA_CHECK(cudaMemcpyAsync(&changed, d_isChanged, sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (changed == 0) break;
  }

  // --- Ensure final labels are in d_labels (the primary buffer) ---
  if (d_labelsOld != d_labels) {
    CUDA_CHECK(cudaMemcpyAsync(d_labels, d_labelsOld,
                                numNodes * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
  }

  // --- Enforce ground-truth labels (safety net) ---
  enforceGroundTruthKernel<<<numBlocksAll, threadsPerBlock, 0, stream>>>(
      d_labels, d_gtClass, numNodes);
  CUDA_KERNEL_CHECK();

  // Cleanup
  CUDA_CHECK(cudaFree(d_affectedList));
  CUDA_CHECK(cudaFree(d_numAffected));
  CUDA_CHECK(cudaFree(d_isChanged));
  CUDA_CHECK(cudaFree(d_isAffectedNext));
}
