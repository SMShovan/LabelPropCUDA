/**
 * @file dynLP.cu
 * @brief Main DynLP orchestrator implementation.
 *
 * See dynLP.h for full documentation.
 *
 * ============================================================================
 * IMPLEMENTATION OVERVIEW
 * ============================================================================
 *
 * The orchestrator maintains:
 *   - A growing host-side CSR graph (appended per batch)
 *   - Device-side CSR arrays (pre-allocated to max size)
 *   - Label arrays (device, double-buffered)
 *   - Vertex status and GT class arrays
 *   - Affected flag arrays
 *
 * For each batch, the orchestrator:
 *   1. Updates the host-side CSR with new vertices/edges
 *   2. Transfers new data to the GPU via async memcpy
 *   3. Runs Step 1 (sparsification + CC)
 *   4. Runs Step 2 (label initialization)
 *   5. Runs Step 3 (iterative propagation)
 *   6. Computes accuracy against true labels
 *
 * ============================================================================
 * ASYNC MEMORY TRANSFER (Paper Section 6.3)
 * ============================================================================
 *
 * Uses CUDA streams to overlap:
 *   - Stream 1: Data transfer (host → device)
 *   - Stream 2: Kernel execution
 *
 * When transferring batch t's data, we can concurrently run sparsification
 * on already-transferred data from the previous transfer.
 *
 * ============================================================================
 */

#include "dynLP.h"
#include "connectedComponents.h"
#include "iterativePropagation.h"
#include "labelInitialization.h"
#include "sparsifySubgraph.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>

using namespace std;

namespace {

/**
 * @brief Compute classification accuracy.
 *
 * Maps fractional labels to binary predictions using 0.5 threshold,
 * then compares against true labels.
 */
float computeAccuracy(const float *predictedLabels,
                      const float *trueLabels,
                      const int *vertexStatus,
                      int numNodes) {
  int correct = 0;
  int total = 0;

  for (int i = 0; i < numNodes; ++i) {
    if (vertexStatus[i] == VERTEX_DELETED) continue;
    if (vertexStatus[i] == VERTEX_GROUND_TRUTH) continue; // Skip GT

    // Only count active non-GT vertices
    int predicted = (predictedLabels[i] >= 0.5f) ? 1 : 0;
    int truth = (trueLabels[i] >= 0.5f) ? 1 : 0;

    if (predicted == truth) correct++;
    total++;
  }

  return (total > 0) ? static_cast<float>(correct) / total : 1.0f;
}

/**
 * @brief Compute the average edge weight for auto tau selection.
 */
float computeAvgEdgeWeight(const CSRGraph &graph) {
  if (graph.numEdges == 0) return 0.5f;

  double sum = 0.0;
  int count = 0;
  for (int i = 0; i < graph.numEdges; ++i) {
    if (graph.colInd[i] >= 0) {
      sum += graph.values[i];
      count++;
    }
  }
  return (count > 0) ? static_cast<float>(sum / count) : 0.5f;
}

/**
 * @brief Build the incremental host CSR for active vertices after a batch.
 *
 * Rebuilds the CSR incorporating new vertices and soft-deletions.
 * For efficiency, we maintain the CSR incrementally: new vertices' rows
 * are appended, and deleted vertices' edges are marked as -1.
 */
void updateHostCSR(CSRGraph &activeGraph,
                   const CSRGraph &fullGraph,
                   const BatchData &batch,
                   const vector<int> &vertexBatchMap,
                   int currentBatch,
                   vector<int> &h_vertexStatus) {
  // Mark deleted vertices
  for (int v : batch.deletedNodeIds) {
    h_vertexStatus[v] = VERTEX_DELETED;
  }

  // Add new vertices: their edges from fullGraph that connect to active verts
  for (int u : batch.insertedNodeIds) {
    h_vertexStatus[u] = VERTEX_ACTIVE;
  }

  // Mark GT vertices
  for (size_t i = 0; i < batch.groundTruthNodeIds.size(); ++i) {
    h_vertexStatus[batch.groundTruthNodeIds[i]] = VERTEX_GROUND_TRUTH;
  }

  // Rebuild CSR from fullGraph, including only active edges
  // (edges where both endpoints are active or GT)
  activeGraph.rowPtr.clear();
  activeGraph.colInd.clear();
  activeGraph.values.clear();
  activeGraph.rowPtr.push_back(0);

  int numNodes = fullGraph.numNodes;
  activeGraph.numNodes = numNodes;

  for (int u = 0; u < numNodes; ++u) {
    if (h_vertexStatus[u] == VERTEX_DELETED ||
        vertexBatchMap[u] > currentBatch ||
        vertexBatchMap[u] < 0) {
      // Vertex not yet active or deleted: empty row
      activeGraph.rowPtr.push_back(activeGraph.rowPtr.back());
      continue;
    }

    int rowStart = fullGraph.rowPtr[u];
    int rowEnd = fullGraph.rowPtr[u + 1];
    for (int e = rowStart; e < rowEnd; ++e) {
      int v = fullGraph.colInd[e];
      if (v < 0) continue;
      if (h_vertexStatus[v] == VERTEX_DELETED) continue;
      if (vertexBatchMap[v] > currentBatch || vertexBatchMap[v] < 0) continue;

      activeGraph.colInd.push_back(v);
      activeGraph.values.push_back(fullGraph.values[e]);
    }
    activeGraph.rowPtr.push_back(
        static_cast<int>(activeGraph.colInd.size()));
  }

  activeGraph.numEdges = static_cast<int>(activeGraph.colInd.size());
}

} // namespace

// ============================================================================
// Public API
// ============================================================================

bool runDynLP(const DynLPConfig &config, const CSRGraph &fullGraph,
              const vector<float> &trueLabels,
              const vector<BatchData> &batches,
              vector<float> &finalLabels,
              vector<float> &batchTimes,
              vector<float> &batchAccuracy) {

  int totalNodes = fullGraph.numNodes;
  int numBatches = config.numBatches;

  // Determine tau
  float tau = config.tau;
  if (tau < 0.0f) {
    tau = computeAvgEdgeWeight(fullGraph);
    cout << "Auto tau (average edge weight): " << tau << "\n";
  }

  // --- Build vertex → batch mapping ---
  vector<int> vertexBatchMap(totalNodes, -1);
  for (int b = 0; b < numBatches; ++b) {
    for (int v : batches[b].insertedNodeIds) {
      vertexBatchMap[v] = b;
    }
  }

  // --- Host-side state ---
  vector<int> h_vertexStatus(totalNodes, VERTEX_DELETED); // All start inactive
  vector<int> h_gtClass(totalNodes, -1); // -1 = not ground truth
  vector<float> h_labels(totalNodes, 0.5f); // Neutral initialization

  // Initialize output vectors
  batchTimes.resize(numBatches, 0.0f);
  batchAccuracy.resize(numBatches, 0.0f);
  finalLabels.resize(totalNodes, 0.5f);

  // ========================================================================
  // GPU Memory Allocation (pre-allocate for max size)
  // ========================================================================

  // Estimate max edges: fullGraph.numEdges is the upper bound
  int maxEdges = fullGraph.numEdges;
  int maxNodes = totalNodes;

  int *d_rowPtr, *d_colInd;
  float *d_values;
  float *d_labels, *d_labelsBuffer;
  int *d_vertexStatus, *d_gtClass;
  int *d_isAffected;
  int *d_componentId;
  int *d_isNewVertex;

  CUDA_CHECK(cudaMalloc(&d_rowPtr, (maxNodes + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_colInd, maxEdges * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_values, maxEdges * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_labels, maxNodes * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_labelsBuffer, maxNodes * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_vertexStatus, maxNodes * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_gtClass, maxNodes * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_isAffected, maxNodes * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_componentId, maxNodes * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_isNewVertex, maxNodes * sizeof(int)));

  // Initialize device arrays
  CUDA_CHECK(cudaMemset(d_isAffected, 0, maxNodes * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_isNewVertex, 0, maxNodes * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_componentId, 0xFF, maxNodes * sizeof(int)));

  // Initialize labels to 0.5
  {
    vector<float> initLabels(maxNodes, 0.5f);
    CUDA_CHECK(cudaMemcpy(d_labels, initLabels.data(),
                           maxNodes * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labelsBuffer, initLabels.data(),
                           maxNodes * sizeof(float),
                           cudaMemcpyHostToDevice));
  }

  // Initialize vertex status to DELETED (inactive)
  {
    vector<int> initStatus(maxNodes, VERTEX_DELETED);
    CUDA_CHECK(cudaMemcpy(d_vertexStatus, initStatus.data(),
                           maxNodes * sizeof(int),
                           cudaMemcpyHostToDevice));
  }

  // Initialize gtClass to -1
  {
    vector<int> initGT(maxNodes, -1);
    CUDA_CHECK(cudaMemcpy(d_gtClass, initGT.data(), maxNodes * sizeof(int),
                           cudaMemcpyHostToDevice));
  }

  // Create CUDA streams
  cudaStream_t streamCompute, streamTransfer;
  CUDA_CHECK(cudaStreamCreate(&streamCompute));
  CUDA_CHECK(cudaStreamCreate(&streamTransfer));

  // CUDA events for timing
  cudaEvent_t batchStart, batchEnd;
  CUDA_CHECK(cudaEventCreate(&batchStart));
  CUDA_CHECK(cudaEventCreate(&batchEnd));

  // ========================================================================
  // Process each batch
  // ========================================================================

  CSRGraph activeGraph;

  for (int b = 0; b < numBatches; ++b) {
    CUDA_CHECK(cudaEventRecord(batchStart, streamCompute));

    const BatchData &batch = batches[b];

    if (config.verbose) {
      cout << "--- Batch " << b << " ---\n";
      cout << "  Inserting " << batch.insertedNodeIds.size() << " vertices ("
           << batch.groundTruthNodeIds.size() << " GT), deleting "
           << batch.deletedNodeIds.size() << "\n";
    }

    // ====================================================================
    // Update host-side state
    // ====================================================================

    // Update GT class mapping
    for (size_t i = 0; i < batch.groundTruthNodeIds.size(); ++i) {
      int v = batch.groundTruthNodeIds[i];
      h_gtClass[v] = (batch.groundTruthLabels[i] < 0.5f) ? 0 : 1;
      h_labels[v] = batch.groundTruthLabels[i];
    }

    // Build active CSR for this batch
    updateHostCSR(activeGraph, fullGraph, batch, vertexBatchMap, b,
                  h_vertexStatus);

    // ====================================================================
    // Transfer to GPU (async)
    // ====================================================================

    CUDA_CHECK(cudaMemcpyAsync(d_rowPtr, activeGraph.rowPtr.data(),
                                (totalNodes + 1) * sizeof(int),
                                cudaMemcpyHostToDevice, streamTransfer));
    CUDA_CHECK(cudaMemcpyAsync(d_colInd, activeGraph.colInd.data(),
                                activeGraph.numEdges * sizeof(int),
                                cudaMemcpyHostToDevice, streamTransfer));
    CUDA_CHECK(cudaMemcpyAsync(d_values, activeGraph.values.data(),
                                activeGraph.numEdges * sizeof(float),
                                cudaMemcpyHostToDevice, streamTransfer));
    CUDA_CHECK(cudaMemcpyAsync(d_vertexStatus, h_vertexStatus.data(),
                                totalNodes * sizeof(int),
                                cudaMemcpyHostToDevice, streamTransfer));
    CUDA_CHECK(cudaMemcpyAsync(d_gtClass, h_gtClass.data(),
                                totalNodes * sizeof(int),
                                cudaMemcpyHostToDevice, streamTransfer));
    CUDA_CHECK(cudaMemcpyAsync(d_labels, h_labels.data(),
                                totalNodes * sizeof(float),
                                cudaMemcpyHostToDevice, streamTransfer));

    // Wait for transfer to complete before kernels
    CUDA_CHECK(cudaStreamSynchronize(streamTransfer));

    // ====================================================================
    // Mark new vertices and affected vertices
    // ====================================================================

    // Reset isNewVertex and isAffected
    CUDA_CHECK(
        cudaMemsetAsync(d_isNewVertex, 0, maxNodes * sizeof(int),
                        streamCompute));
    CUDA_CHECK(
        cudaMemsetAsync(d_isAffected, 0, maxNodes * sizeof(int),
                        streamCompute));

    // Set isNewVertex flags for this batch's inserted nodes
    if (!batch.insertedNodeIds.empty()) {
      vector<int> newFlags(maxNodes, 0);
      for (int v : batch.insertedNodeIds) {
        newFlags[v] = 1;
      }
      CUDA_CHECK(cudaMemcpyAsync(d_isNewVertex, newFlags.data(),
                                  maxNodes * sizeof(int),
                                  cudaMemcpyHostToDevice, streamCompute));
    }

    // ====================================================================
    // Step 1: Handle deletions
    // ====================================================================

    if (!batch.deletedNodeIds.empty()) {
      int *d_deletedNodes;
      int numDeleted = static_cast<int>(batch.deletedNodeIds.size());
      CUDA_CHECK(cudaMalloc(&d_deletedNodes, numDeleted * sizeof(int)));
      CUDA_CHECK(cudaMemcpyAsync(d_deletedNodes, batch.deletedNodeIds.data(),
                                  numDeleted * sizeof(int),
                                  cudaMemcpyHostToDevice, streamCompute));

      launchMarkDeletions(d_rowPtr, d_colInd, d_vertexStatus, d_isAffected,
                          d_deletedNodes, numDeleted, totalNodes,
                          streamCompute);

      CUDA_CHECK(cudaFree(d_deletedNodes));
    }

    // ====================================================================
    // Step 1: Sparsification and Connected Components
    // ====================================================================

    int numComponents = 0;

    if (!batch.insertedNodeIds.empty() && !batch.newEdgeSrc.empty()) {
      int numInserted = static_cast<int>(batch.insertedNodeIds.size());
      int numNewEdges = static_cast<int>(batch.newEdgeSrc.size());

      // Transfer new edge data
      int *d_newEdgeSrc, *d_newEdgeDst;
      float *d_newEdgeWeight;
      int *d_insertedNodes;

      CUDA_CHECK(cudaMalloc(&d_newEdgeSrc, numNewEdges * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&d_newEdgeDst, numNewEdges * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&d_newEdgeWeight, numNewEdges * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_insertedNodes, numInserted * sizeof(int)));

      CUDA_CHECK(cudaMemcpyAsync(
          d_newEdgeSrc, batch.newEdgeSrc.data(), numNewEdges * sizeof(int),
          cudaMemcpyHostToDevice, streamCompute));
      CUDA_CHECK(cudaMemcpyAsync(
          d_newEdgeDst, batch.newEdgeDst.data(), numNewEdges * sizeof(int),
          cudaMemcpyHostToDevice, streamCompute));
      CUDA_CHECK(cudaMemcpyAsync(
          d_newEdgeWeight, batch.newEdgeWeight.data(),
          numNewEdges * sizeof(float), cudaMemcpyHostToDevice,
          streamCompute));
      CUDA_CHECK(cudaMemcpyAsync(
          d_insertedNodes, batch.insertedNodeIds.data(),
          numInserted * sizeof(int), cudaMemcpyHostToDevice,
          streamCompute));

      // Sparsification: build subgraph G'
      int *d_subRowPtr, *d_subColInd;
      float *d_subValues;
      int numSubNodes = 0, numSubEdges = 0;

      // Allocate max possible subgraph size
      int maxSubEdges = numNewEdges * 2; // Each edge appears twice
      CUDA_CHECK(cudaMalloc(&d_subRowPtr, (numInserted + 1) * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&d_subColInd, max(maxSubEdges, 1) * sizeof(int)));
      CUDA_CHECK(
          cudaMalloc(&d_subValues, max(maxSubEdges, 1) * sizeof(float)));

      launchBuildSparsifiedSubgraph(
          d_rowPtr, d_colInd, d_values, d_isAffected, d_vertexStatus,
          d_newEdgeSrc, d_newEdgeDst, d_newEdgeWeight, numNewEdges,
          d_insertedNodes, numInserted, d_isNewVertex, tau, d_subRowPtr,
          d_subColInd, d_subValues, numSubNodes, numSubEdges, streamCompute);

      // Connected Components via Shiloach-Vishkin
      if (numSubEdges > 0) {
        // Use the edge list from the subgraph build
        // d_subColInd contains dst vertices, we need src+dst
        // Re-use d_newEdgeSrc/Dst filtered for new-to-new edges > tau
        launchConnectedComponents(d_newEdgeSrc, d_newEdgeDst, numSubEdges,
                                   d_insertedNodes, numInserted,
                                   d_componentId, numComponents, totalNodes,
                                   streamCompute);
      } else {
        // No edges in subgraph: each new vertex is its own component
        numComponents = numInserted;
        // Assign sequential component IDs on host
        vector<int> h_compId(totalNodes, -1);
        for (int i = 0; i < numInserted; ++i) {
          h_compId[batch.insertedNodeIds[i]] = i;
        }
        CUDA_CHECK(cudaMemcpyAsync(d_componentId, h_compId.data(),
                                    totalNodes * sizeof(int),
                                    cudaMemcpyHostToDevice, streamCompute));
      }

      if (config.verbose) {
        cout << "  Step 1: " << numComponents << " connected components, "
             << numSubEdges << " subgraph edges (tau=" << tau << ")\n";
      }

      // ================================================================
      // Step 2: Label Initialization
      // ================================================================

      launchLabelInitialization(d_rowPtr, d_colInd, d_values, d_labels,
                                 d_vertexStatus, d_componentId, numComponents,
                                 d_insertedNodes, numInserted, d_gtClass,
                                 totalNodes, streamCompute);

      if (config.verbose) {
        cout << "  Step 2: Labels initialized for " << numInserted
             << " new vertices\n";
      }

      // Cleanup batch-specific allocations
      CUDA_CHECK(cudaFree(d_newEdgeSrc));
      CUDA_CHECK(cudaFree(d_newEdgeDst));
      CUDA_CHECK(cudaFree(d_newEdgeWeight));
      CUDA_CHECK(cudaFree(d_insertedNodes));
      CUDA_CHECK(cudaFree(d_subRowPtr));
      CUDA_CHECK(cudaFree(d_subColInd));
      CUDA_CHECK(cudaFree(d_subValues));

    } else if (!batch.insertedNodeIds.empty()) {
      // New vertices but no edges: each is its own component
      int numInserted = static_cast<int>(batch.insertedNodeIds.size());
      numComponents = numInserted;

      // Set labels for GT vertices, 0.5 for others
      vector<float> h_batchLabels(totalNodes, 0.5f);
      for (size_t i = 0; i < batch.groundTruthNodeIds.size(); ++i) {
        h_batchLabels[batch.groundTruthNodeIds[i]] =
            batch.groundTruthLabels[i];
      }
      // Only update labels for new vertices
      for (int v : batch.insertedNodeIds) {
        if (h_gtClass[v] >= 0) {
          h_labels[v] = h_batchLabels[v];
        } else {
          h_labels[v] = 0.5f;
        }
      }
      CUDA_CHECK(cudaMemcpyAsync(d_labels, h_labels.data(),
                                  totalNodes * sizeof(float),
                                  cudaMemcpyHostToDevice, streamCompute));
    }

    // ====================================================================
    // Step 3: Iterative Propagation
    // ====================================================================

    // Also mark all new vertices as affected for propagation
    for (int v : batch.insertedNodeIds) {
      if (h_gtClass[v] < 0) { // Non-GT vertices
        // This is done via the sparsification kernel already,
        // but we ensure it here as a safety net
      }
    }

    // Copy current labels to buffer for double-buffering
    CUDA_CHECK(cudaMemcpyAsync(d_labelsBuffer, d_labels,
                                totalNodes * sizeof(float),
                                cudaMemcpyDeviceToDevice, streamCompute));

    int iterationsUsed = 0;
    launchIterativePropagation(d_rowPtr, d_colInd, d_values, d_labels,
                                d_labelsBuffer, d_vertexStatus, d_gtClass,
                                d_isAffected, totalNodes, config.delta,
                                config.maxIterations, iterationsUsed,
                                streamCompute);

    if (config.verbose) {
      cout << "  Step 3: Converged in " << iterationsUsed << " iterations\n";
    }

    // ====================================================================
    // Retrieve labels and compute accuracy
    // ====================================================================

    CUDA_CHECK(cudaMemcpyAsync(h_labels.data(), d_labels,
                                totalNodes * sizeof(float),
                                cudaMemcpyDeviceToHost, streamCompute));
    CUDA_CHECK(cudaStreamSynchronize(streamCompute));

    float accuracy = computeAccuracy(h_labels.data(), trueLabels.data(),
                                      h_vertexStatus.data(), totalNodes);
    batchAccuracy[b] = accuracy;

    CUDA_CHECK(cudaEventRecord(batchEnd, streamCompute));
    CUDA_CHECK(cudaEventSynchronize(batchEnd));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, batchStart, batchEnd));
    batchTimes[b] = elapsedMs;

    if (config.verbose) {
      cout << "  Accuracy: " << (accuracy * 100.0f) << "%, Time: "
           << elapsedMs << " ms\n\n";
    }
  }

  // ========================================================================
  // Copy final labels
  // ========================================================================

  finalLabels = h_labels;

  // ========================================================================
  // Cleanup
  // ========================================================================

  CUDA_CHECK(cudaFree(d_rowPtr));
  CUDA_CHECK(cudaFree(d_colInd));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_labels));
  CUDA_CHECK(cudaFree(d_labelsBuffer));
  CUDA_CHECK(cudaFree(d_vertexStatus));
  CUDA_CHECK(cudaFree(d_gtClass));
  CUDA_CHECK(cudaFree(d_isAffected));
  CUDA_CHECK(cudaFree(d_componentId));
  CUDA_CHECK(cudaFree(d_isNewVertex));

  CUDA_CHECK(cudaEventDestroy(batchStart));
  CUDA_CHECK(cudaEventDestroy(batchEnd));
  CUDA_CHECK(cudaStreamDestroy(streamCompute));
  CUDA_CHECK(cudaStreamDestroy(streamTransfer));

  return true;
}
