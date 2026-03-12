/**
 * @file iterativePropagation.h
 * @brief Step 3 of DynLP: Iterative Label Propagation on GPU.
 *
 * Performs the iterative label update for all affected vertices until
 * convergence. This is the core computational step of DynLP.
 *
 * ============================================================================
 * UPDATE RULE (Algorithm 2, Line 28 — with fixes)
 * ============================================================================
 *
 * For each affected unlabeled vertex u:
 *
 *   W_all = Σ_{v∈N(u)} w(u, v)             (total neighbor weight)
 *   W_L0  = Σ_{v∈L0}   w(u, v)             (weight to class-0 GT)
 *   W_L1  = Σ_{v∈L1}   w(u, v)             (weight to class-1 GT)
 *
 *   F'_u = F_u
 *        + (0 - F_u) * W_L0 / W_all          (pull toward class 0)
 *        + (1 - F_u) * W_L1 / W_all          (pull toward class 1)
 *        + Σ_{v∈N(u)\L0\L1} (F_v - F_u) * w(u,v) / W_all  (peer influence)
 *
 * This is equivalent to the standard weighted neighborhood averaging:
 *   F'_u = Σ_{v∈N(u)} α_{u,v} * F_v
 * where α_{u,v} = w(u,v) / W_all.
 *
 * The proof of equivalence is given in Section 5 of the paper.
 *
 * ============================================================================
 * CONVERGENCE
 * ============================================================================
 *
 * If |F'_u - F_u| > δ, vertex u and its neighbors are marked as affected
 * for the next iteration (ISSUE 7 FIX: absolute difference).
 *
 * The process converges when no vertex changes by more than δ, or when
 * the maximum iteration count is reached.
 *
 * ============================================================================
 * RACE CONDITION AVOIDANCE (ISSUE 5 FIX)
 * ============================================================================
 *
 * We use a two-buffer approach: read labels from d_labelsOld, write to
 * d_labelsNew. After each iteration, swap the pointers. This avoids
 * read-write races where a thread reads a neighbor's label that was
 * already updated in the same iteration.
 *
 * ============================================================================
 */

#ifndef ITERATIVE_PROPAGATION_H
#define ITERATIVE_PROPAGATION_H

#include "graphTypes.h"

/**
 * @brief Run iterative label propagation until convergence.
 *
 * @param d_rowPtr          Device CSR row pointers
 * @param d_colInd          Device CSR column indices
 * @param d_values          Device CSR edge weights
 * @param d_labels          Device label array (read/write, current labels)
 * @param d_labelsBuffer    Device label buffer for double-buffering
 * @param d_vertexStatus    Device vertex status array
 * @param d_gtClass         Device GT class array (-1 for non-GT)
 * @param d_isAffected      Device affected flag array (input/output)
 * @param numNodes          Total number of nodes in the graph
 * @param delta             Convergence threshold
 * @param maxIterations     Maximum number of iterations
 * @param iterationsUsed    Output: actual iterations performed
 * @param stream            CUDA stream
 */
void launchIterativePropagation(const int *d_rowPtr, const int *d_colInd,
                                 const float *d_values, float *d_labels,
                                 float *d_labelsBuffer,
                                 const int *d_vertexStatus,
                                 const int *d_gtClass, int *d_isAffected,
                                 int numNodes, float delta,
                                 int maxIterations, int &iterationsUsed,
                                 cudaStream_t stream);

#endif // ITERATIVE_PROPAGATION_H
