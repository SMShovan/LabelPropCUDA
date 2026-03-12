/**
 * @file labelInitialization.h
 * @brief Step 2 of DynLP: Connected-Component-Based Label Initialization.
 *
 * After connected components are found in the new-vertex subgraph G',
 * this step initializes labels for all vertices in each component using
 * the edge weights between the component and the two ground-truth
 * supernodes L0 (class 0) and L1 (class 1).
 *
 * ============================================================================
 * INITIALIZATION FORMULA (Simplified from Algorithm 2, Line 22)
 * ============================================================================
 *
 * For each vertex u in connected component c_i:
 *
 *   W_ci^L0 = Σ_{x∈c_i} Σ_{y∈L0} w(x, y)   (total weight to class 0)
 *   W_ci^L1 = Σ_{x∈c_i} Σ_{y∈L1} w(x, y)   (total weight to class 1)
 *
 *   if W_ci^L0 + W_ci^L1 > 0:
 *     F_u = W_ci^L1 / (W_ci^L0 + W_ci^L1)
 *   else:
 *     F_u = 0.5  (neutral — ISSUE 1 FIX)
 *
 * ISSUE 6 FIX: We use the algebraically simplified form directly instead
 * of the paper's expanded three-term formula, which is equivalent but
 * clearer and avoids redundant computation.
 *
 * ============================================================================
 */

#ifndef LABEL_INITIALIZATION_H
#define LABEL_INITIALIZATION_H

#include "graphTypes.h"

/**
 * @brief Initialize labels for newly inserted vertices using connected
 *        component membership and ground-truth supernode weights.
 *
 * @param d_rowPtr          Device CSR row pointers for the full graph
 * @param d_colInd          Device CSR column indices
 * @param d_values          Device CSR edge weights
 * @param d_labels          Device label array (modified for new vertices)
 * @param d_vertexStatus    Device vertex status array
 * @param d_componentId     Device component ID for each new vertex
 * @param numComponents     Number of connected components
 * @param d_insertedNodes   Device array of newly inserted node IDs
 * @param numInserted       Number of inserted nodes
 * @param d_gtClass         Device array: gtClass[v] = 0 or 1 for GT vertices,
 *                          -1 for non-GT vertices
 * @param numNodes          Total number of nodes
 * @param stream            CUDA stream
 */
void launchLabelInitialization(const int *d_rowPtr, const int *d_colInd,
                                const float *d_values, float *d_labels,
                                const int *d_vertexStatus,
                                const int *d_componentId, int numComponents,
                                const int *d_insertedNodes, int numInserted,
                                const int *d_gtClass, int numNodes,
                                cudaStream_t stream);

#endif // LABEL_INITIALIZATION_H
