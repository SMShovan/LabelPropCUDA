/**
 * @file sparsifySubgraph.h
 * @brief Step 1 of DynLP: Change Adjustment and Sparsification.
 *
 * Handles:
 *   (a) Marking affected vertices from deletions and insertions
 *   (b) Soft-deleting removed vertices from the CSR
 *   (c) Building the new-vertex subgraph G'(V', E') with threshold τ
 *   (d) Adding new-to-existing edges to the main graph
 *
 * All operations are GPU-parallelized using CUDA kernels with the
 * block-per-row execution model.
 */

#ifndef SPARSIFY_SUBGRAPH_H
#define SPARSIFY_SUBGRAPH_H

#include "graphTypes.h"
#include <vector>

/**
 * @brief Mark neighbors of deleted vertices as affected and soft-delete.
 *
 * For each vertex u in deletedNodeIds:
 *   1. Visit all neighbors N(u) in the current graph.
 *   2. Mark each neighbor as affected (needs label update in Step 3).
 *   3. Soft-delete u by setting its colInd entries to -1.
 *
 * ISSUE 3 FIX: We capture neighbors from the ORIGINAL adjacency (before
 * deletion) to ensure all affected vertices are properly identified.
 *
 * @param d_rowPtr          Device CSR row pointers
 * @param d_colInd          Device CSR column indices (modified: soft-delete)
 * @param d_vertexStatus    Device vertex status array (modified)
 * @param d_isAffected      Device affected flag array (modified)
 * @param d_deletedNodes    Device array of deleted node IDs
 * @param numDeleted        Number of deleted nodes
 * @param numNodes          Total number of nodes in the graph
 * @param stream            CUDA stream for async execution
 */
void launchMarkDeletions(int *d_rowPtr, int *d_colInd, int *d_vertexStatus,
                         int *d_isAffected, const int *d_deletedNodes,
                         int numDeleted, int numNodes, cudaStream_t stream);

/**
 * @brief Build the sparsified new-vertex subgraph and mark affected vertices.
 *
 * For new vertices, builds:
 *   - Subgraph G' for connected component finding (new-to-new edges only)
 *   - Adds new-to-existing edges to the main graph CSR
 *   - Marks all neighbors of new vertices as affected
 *
 * Sparsification: only edges with weight > τ are kept.
 *
 * ISSUE 4 FIX: new-to-new edges go into G' (for CC finding),
 * new-to-existing edges go into E_{t+1} (main graph).
 *
 * @param d_rowPtr          Device CSR row pointers for main graph
 * @param d_colInd          Device CSR column indices for main graph
 * @param d_values          Device CSR values for main graph
 * @param d_isAffected      Device affected flag array (modified)
 * @param d_vertexStatus    Device vertex status array
 * @param d_newEdgeSrc      Device new edge source array
 * @param d_newEdgeDst      Device new edge destination array
 * @param d_newEdgeWeight   Device new edge weight array
 * @param numNewEdges       Number of new edges
 * @param d_insertedNodes   Device array of newly inserted node IDs
 * @param numInserted       Number of inserted nodes
 * @param d_isNewVertex     Device flag: 1 if vertex is in current batch
 * @param tau               Sparsification threshold
 * @param d_subRowPtr       Output: device row pointers for subgraph G'
 * @param d_subColInd       Output: device column indices for subgraph G'
 * @param d_subValues       Output: device values for subgraph G'
 * @param numSubNodes       Output: number of nodes in subgraph
 * @param numSubEdges       Output: number of edges in subgraph
 * @param stream            CUDA stream
 */
void launchBuildSparsifiedSubgraph(
    int *d_rowPtr, int *d_colInd, float *d_values, int *d_isAffected,
    const int *d_vertexStatus, const int *d_newEdgeSrc,
    const int *d_newEdgeDst, const float *d_newEdgeWeight, int numNewEdges,
    const int *d_insertedNodes, int numInserted, const int *d_isNewVertex,
    float tau, int *d_subRowPtr, int *d_subColInd, float *d_subValues,
    int &numSubNodes, int &numSubEdges, cudaStream_t stream);

#endif // SPARSIFY_SUBGRAPH_H
