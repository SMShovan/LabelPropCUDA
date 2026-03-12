/**
 * @file connectedComponents.h
 * @brief Shiloach-Vishkin parallel connected component finding on GPU.
 *
 * Implements the Shiloach-Vishkin (SV) algorithm for finding connected
 * components in the sparsified new-vertex subgraph G'(V', E'). The
 * algorithm consists of two alternating phases:
 *
 *   Hook: Each vertex tries to attach to its neighbor with the smallest ID
 *         using atomicMin on the parent array.
 *
 *   Jump: Path compression — each vertex updates par[i] = par[par[i]]
 *         to shorten the path to the root.
 *
 * The algorithm terminates when no changes occur in a Jump phase,
 * indicating that all paths are fully compressed and components are
 * identified.
 *
 * After convergence, each vertex's par[i] value is the root of its
 * connected component. A prefix scan renumbers components sequentially.
 *
 * Reference: Y. Shiloach and U. Vishkin, "An O(log n) parallel
 * connectivity algorithm," J. Algorithms, 1982.
 */

#ifndef CONNECTED_COMPONENTS_H
#define CONNECTED_COMPONENTS_H

#include "graphTypes.h"
#include <vector>

/**
 * @brief Find connected components in the new-vertex subgraph using SV.
 *
 * Operates on the edge list produced by sparsification. Each new vertex
 * is assigned a component ID. Component IDs are sequential starting from 0.
 *
 * @param d_edgeSrc         Device array of edge source vertices
 * @param d_edgeDst         Device array of edge destination vertices
 * @param numEdges          Number of directed edges (each undirected edge
 *                          appears twice)
 * @param d_insertedNodes   Device array of new vertex IDs
 * @param numInserted       Number of newly inserted vertices
 * @param d_componentId     Output: device array mapping vertex → component ID
 *                          (size = total numNodes, only new vertices filled)
 * @param numComponents     Output: number of connected components found
 * @param numNodes          Total number of nodes in the graph
 * @param stream            CUDA stream
 */
void launchConnectedComponents(const int *d_edgeSrc, const int *d_edgeDst,
                                int numEdges, const int *d_insertedNodes,
                                int numInserted, int *d_componentId,
                                int &numComponents, int numNodes,
                                cudaStream_t stream);

#endif // CONNECTED_COMPONENTS_H
