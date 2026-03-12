/**
 * @file generateSparseGraph.h
 * @brief Erdős–Rényi sparse graph generator with similarity weights.
 *
 * Generates an undirected sparse graph stored in CSR format. Edge weights
 * represent pairwise similarity values in [0, 1]. The generator ensures
 * graph connectivity by first building a random spanning tree, then adding
 * random edges to reach the desired average degree.
 *
 * Each vertex is also assigned a "true" binary class label (0 or 1) which
 * is used to bias edge weights: vertices of the same class tend to have
 * higher similarity (edge weights), mimicking real-world feature-based
 * similarity graphs.
 */

#ifndef GENERATE_SPARSE_GRAPH_H
#define GENERATE_SPARSE_GRAPH_H

#include "graphTypes.h"
#include <string>
#include <vector>

/**
 * @brief Generate a sparse undirected weighted graph in CSR format.
 *
 * The generation process:
 *   1. Assign each vertex a random binary class label (balanced 50/50).
 *   2. Build a random spanning tree to guarantee connectivity.
 *   3. Add random edges until avgDegree is reached.
 *   4. Assign edge weights: same-class edges get higher weights (0.5-1.0),
 *      cross-class edges get lower weights (0.0-0.5).
 *   5. Store in CSR format.
 *
 * @param totalNodes    Total number of vertices
 * @param avgDegree     Target average degree
 * @param seed          RNG seed for reproducibility
 * @param graph         Output: CSR graph
 * @param trueLabels    Output: ground-truth labels for all vertices (0.0 or 1.0)
 * @param outputPrefix  Path prefix for writing CSR files (empty = don't write)
 * @return true on success, false on error
 */
bool generateSparseGraph(int totalNodes, int avgDegree, unsigned int seed,
                         CSRGraph &graph, std::vector<float> &trueLabels,
                         const std::string &outputPrefix = "");

#endif // GENERATE_SPARSE_GRAPH_H
