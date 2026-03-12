/**
 * @file generateBatches.h
 * @brief Dynamic batch generator for DynLP.
 *
 * Splits the full graph into a sequence of batches that simulate the
 * incremental arrival of data. The first batch establishes the initial
 * ground-truth vertices and some unlabeled vertices. Subsequent batches
 * add new vertices and edges, and optionally delete existing vertices.
 *
 * ============================================================================
 * BATCH COMPOSITION (per the paper's experimental setup)
 * ============================================================================
 *
 * Each batch (after the first) consists of:
 *   - 90% unlabeled new vertices (no ground truth)
 *   -  1% ground-truth new vertices (label known)
 *   -  9% deletions (existing non-GT vertices removed)
 *
 * The first batch (batch 0) is special: it contains only the initial
 * ground-truth vertices and establishes the seed labels V^L.
 *
 * ============================================================================
 */

#ifndef GENERATE_BATCHES_H
#define GENERATE_BATCHES_H

#include "graphTypes.h"
#include <string>
#include <vector>

/**
 * @brief Generate batches of dynamic changes from a pre-generated graph.
 *
 * The full graph is pre-generated with all nodes and edges. This function
 * partitions the vertices into batches and determines which edges belong
 * to each batch (an edge is active when both endpoints are active).
 *
 * Batch 0 (initialization):
 *   - Contains all initial ground-truth vertices (1% of total, balanced
 *     between class 0 and class 1).
 *   - No deletions.
 *
 * Batches 1..numBatches-1:
 *   - New vertices are drawn from the remaining vertex pool.
 *   - Ground-truth vertices are a small fraction of the new vertices.
 *   - Deletions are sampled from existing non-GT active vertices.
 *   - Edges between new and existing vertices are activated.
 *
 * @param graph         Full pre-generated CSR graph
 * @param trueLabels    True labels for all vertices (used to pick GT nodes)
 * @param config        Configuration parameters
 * @param batches       Output: vector of BatchData, one per batch
 * @return true on success, false on error
 */
bool generateBatches(const CSRGraph &graph,
                     const std::vector<float> &trueLabels,
                     const DynLPConfig &config,
                     std::vector<BatchData> &batches);

#endif // GENERATE_BATCHES_H
