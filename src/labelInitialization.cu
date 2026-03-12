/**
 * @file labelInitialization.cu
 * @brief Step 2 of DynLP: Label Initialization (CUDA).
 *
 * See labelInitialization.h for full documentation.
 *
 * ============================================================================
 * KERNEL DESIGN
 * ============================================================================
 *
 * Phase 1: computeSupernodeWeightsKernel
 *   - One thread per new vertex.
 *   - Each thread walks the full adjacency of its vertex in the main graph.
 *   - For each neighbor that is a ground-truth vertex:
 *       atomicAdd to the component's W_L0 or W_L1 accumulator.
 *   - Uses global memory atomics (float atomicAdd, available since sm_20).
 *
 * Phase 2: initializeLabelsKernel
 *   - One thread per new vertex.
 *   - Reads the component's W_L0 and W_L1.
 *   - Computes F_u = W_L1 / (W_L0 + W_L1), or 0.5 if denominator is zero.
 *   - Ground-truth vertices are set to their fixed labels.
 *
 * ============================================================================
 * EDGE CASES
 * ============================================================================
 *
 * 1. Component with no edges to any GT vertex: F_u = 0.5 (neutral).
 *    This is ISSUE 1 from the analysis — the paper's formula divides by
 *    zero in this case.
 *
 * 2. Single-vertex component with no neighbors: F_u = 0.5.
 *
 * 3. GT vertex in a component: Its label is set to the GT value (0.0 or 1.0)
 *    regardless of the component initialization formula.
 *
 * ============================================================================
 */

#include "labelInitialization.h"

#include <cuda_runtime.h>
#include <cstdio>

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief Compute aggregate edge weights from each component to L0 and L1.
 *
 * One thread per newly inserted vertex. Each thread iterates over its
 * neighbors in the main graph and accumulates weights to GT class 0 and
 * GT class 1 supernodes using atomic float addition.
 *
 * d_compWeightL0[compId] = Σ_{u∈comp, v∈L0} w(u, v)
 * d_compWeightL1[compId] = Σ_{u∈comp, v∈L1} w(u, v)
 */
__global__ void computeSupernodeWeightsKernel(
    const int *d_rowPtr, const int *d_colInd, const float *d_values,
    const int *d_vertexStatus, const int *d_componentId,
    const int *d_gtClass, const int *d_insertedNodes, int numInserted,
    float *d_compWeightL0, float *d_compWeightL1) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numInserted) return;

  int u = d_insertedNodes[tid];
  int compId = d_componentId[u];
  if (compId < 0) return; // Vertex not assigned to a component

  // Walk the adjacency list of u in the main graph
  int rowStart = d_rowPtr[u];
  int rowEnd = d_rowPtr[u + 1];

  for (int e = rowStart; e < rowEnd; ++e) {
    int v = d_colInd[e];
    if (v < 0) continue; // Soft-deleted edge

    // Check if neighbor v is a ground-truth vertex
    int gtClass = d_gtClass[v];
    if (gtClass == 0) {
      atomicAdd(&d_compWeightL0[compId], d_values[e]);
    } else if (gtClass == 1) {
      atomicAdd(&d_compWeightL1[compId], d_values[e]);
    }
  }
}

/**
 * @brief Initialize labels for new vertices based on component weights.
 *
 * One thread per newly inserted vertex.
 * F_u = W_L1 / (W_L0 + W_L1) if denominator > 0, else 0.5.
 *
 * ISSUE 1 FIX: Division by zero handled with neutral initialization.
 * ISSUE 6 FIX: Simplified formula used directly.
 */
__global__ void initializeLabelsKernel(float *d_labels,
                                        const int *d_componentId,
                                        const int *d_gtClass,
                                        const float *d_compWeightL0,
                                        const float *d_compWeightL1,
                                        const int *d_insertedNodes,
                                        int numInserted) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numInserted) return;

  int u = d_insertedNodes[tid];

  // Ground-truth vertices get fixed labels
  int gtClass = d_gtClass[u];
  if (gtClass == 0) {
    d_labels[u] = 0.0f;
    return;
  }
  if (gtClass == 1) {
    d_labels[u] = 1.0f;
    return;
  }

  // Non-GT vertex: use component-based initialization
  int compId = d_componentId[u];
  if (compId < 0) {
    // Not assigned to a component (shouldn't happen, but handle gracefully)
    d_labels[u] = 0.5f;
    return;
  }

  float wL0 = d_compWeightL0[compId];
  float wL1 = d_compWeightL1[compId];
  float denominator = wL0 + wL1;

  if (denominator > 1e-10f) {
    // Simplified formula: F_u = W_L1 / (W_L0 + W_L1)
    // Values close to 0 → class 0, close to 1 → class 1
    d_labels[u] = wL1 / denominator;
  } else {
    // ISSUE 1 FIX: No edges to any GT vertex → neutral initialization
    d_labels[u] = 0.5f;
  }
}

// ============================================================================
// Host Launch Function
// ============================================================================

void launchLabelInitialization(const int *d_rowPtr, const int *d_colInd,
                                const float *d_values, float *d_labels,
                                const int *d_vertexStatus,
                                const int *d_componentId, int numComponents,
                                const int *d_insertedNodes, int numInserted,
                                const int *d_gtClass, int numNodes,
                                cudaStream_t stream) {
  if (numInserted == 0) return;

  int threadsPerBlock = 256;
  int numBlocks = (numInserted + threadsPerBlock - 1) / threadsPerBlock;

  // --- Allocate component weight accumulators ---
  float *d_compWeightL0 = nullptr;
  float *d_compWeightL1 = nullptr;

  int numComp = max(numComponents, 1);
  CUDA_CHECK(cudaMalloc(&d_compWeightL0, numComp * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_compWeightL1, numComp * sizeof(float)));
  CUDA_CHECK(cudaMemsetAsync(d_compWeightL0, 0, numComp * sizeof(float),
                              stream));
  CUDA_CHECK(cudaMemsetAsync(d_compWeightL1, 0, numComp * sizeof(float),
                              stream));

  // --- Phase 1: Compute supernode weights ---
  computeSupernodeWeightsKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
      d_rowPtr, d_colInd, d_values, d_vertexStatus, d_componentId, d_gtClass,
      d_insertedNodes, numInserted, d_compWeightL0, d_compWeightL1);
  CUDA_KERNEL_CHECK();

  // --- Phase 2: Initialize labels ---
  initializeLabelsKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
      d_labels, d_componentId, d_gtClass, d_compWeightL0, d_compWeightL1,
      d_insertedNodes, numInserted);
  CUDA_KERNEL_CHECK();

  // Cleanup
  CUDA_CHECK(cudaFree(d_compWeightL0));
  CUDA_CHECK(cudaFree(d_compWeightL1));
}
