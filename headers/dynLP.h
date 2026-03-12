/**
 * @file dynLP.h
 * @brief Main DynLP orchestrator: manages batch-by-batch execution.
 *
 * This is the top-level module that ties together all components:
 *   - Graph construction and batch management
 *   - Memory management (host ↔ device transfers)
 *   - Step 1: Change Adjustment and Sparsification
 *   - Step 2: Label Initialization
 *   - Step 3: Iterative Propagation
 *
 * ============================================================================
 * EXECUTION FLOW PER BATCH
 * ============================================================================
 *
 * For each batch t from 0 to numBatches-1:
 *
 *   1. Host-side graph update:
 *      - Append new vertices' CSR rows to the host graph
 *      - Soft-delete removed vertices
 *
 *   2. Transfer to GPU:
 *      - Async copy of new CSR rows to device
 *      - Overlap with kernel execution where possible
 *
 *   3. Step 1: Sparsification + CC finding
 *      - Mark affected vertices from deletions
 *      - Build sparsified subgraph G'
 *      - Find connected components in G' via Shiloach-Vishkin
 *
 *   4. Step 2: Label initialization
 *      - Compute supernode weights per component
 *      - Initialize new vertex labels
 *
 *   5. Step 3: Iterative propagation
 *      - Update labels for all affected vertices until convergence
 *
 *   6. Record accuracy and timing
 *
 * ============================================================================
 * MEMORY MANAGEMENT
 * ============================================================================
 *
 * GPU memory layout follows the paper's Figure 4c: CSR rows are stored
 * contiguously and appended batch-by-batch. We pre-allocate for the
 * maximum expected graph size to avoid repeated cudaMalloc calls.
 *
 * ============================================================================
 */

#ifndef DYNLP_H
#define DYNLP_H

#include "graphTypes.h"
#include <vector>

/**
 * @brief Run the complete DynLP algorithm across all batches.
 *
 * @param config        Algorithm configuration
 * @param graph         Full pre-generated CSR graph (all nodes and edges)
 * @param trueLabels    True labels for all vertices (for accuracy computation)
 * @param batches       Vector of batch data (insertions, deletions, edges)
 * @param finalLabels   Output: final predicted labels for all vertices
 * @param batchTimes    Output: execution time per batch (ms)
 * @param batchAccuracy Output: accuracy per batch
 * @return true on success, false on error
 */
bool runDynLP(const DynLPConfig &config, const CSRGraph &graph,
              const std::vector<float> &trueLabels,
              const std::vector<BatchData> &batches,
              std::vector<float> &finalLabels,
              std::vector<float> &batchTimes,
              std::vector<float> &batchAccuracy);

#endif // DYNLP_H
