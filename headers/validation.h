/**
 * @file validation.h
 * @brief IrLP (Iterative Label Propagation) baseline for accuracy validation.
 *
 * Implements the standard iterative label propagation algorithm (IrLP)
 * as a baseline to compare DynLP's accuracy against. IrLP recomputes
 * all labels from scratch using the full graph at each timestep.
 *
 * ============================================================================
 * IrLP UPDATE RULE (Equation 1 from the paper)
 * ============================================================================
 *
 * For each unlabeled vertex u ∈ V^U at iteration k+1:
 *
 *   F_u^(k+1) = (1/d(u)) * Σ_{v∈N(u)} w(u,v) * F_v^(k)
 *
 * where d(u) = Σ_{v∈N(u)} w(u,v) is the weighted degree.
 *
 * Ground-truth vertices retain their fixed labels:
 *   F_u^(k+1) = Y_u  for u ∈ V^L
 *
 * The iteration continues until max|F_u^(k+1) - F_u^(k)| < delta.
 *
 * ============================================================================
 * PURPOSE
 * ============================================================================
 *
 * IrLP serves as the "gold standard" for label accuracy. Since it performs
 * a full recomputation using the harmonic solution, its labels are optimal
 * given the current graph. DynLP aims to achieve similar accuracy with
 * much lower computation time by only updating affected vertices.
 *
 * ============================================================================
 */

#ifndef VALIDATION_H
#define VALIDATION_H

#include "graphTypes.h"
#include <vector>

/**
 * @brief Run IrLP baseline on the current graph snapshot.
 *
 * Performs full iterative label propagation from scratch on the given
 * CSR graph. Used to validate DynLP accuracy.
 *
 * @param d_rowPtr          Device CSR row pointers
 * @param d_colInd          Device CSR column indices
 * @param d_values          Device CSR edge weights
 * @param d_labels          Output: device label array (filled by IrLP)
 * @param d_vertexStatus    Device vertex status array
 * @param d_gtClass         Device GT class array (-1 for non-GT)
 * @param numNodes          Total number of nodes
 * @param delta             Convergence threshold
 * @param maxIterations     Maximum iterations
 * @param iterationsUsed    Output: actual iterations performed
 * @param stream            CUDA stream
 */
void launchIrLP(const int *d_rowPtr, const int *d_colInd,
                const float *d_values, float *d_labels,
                const int *d_vertexStatus, const int *d_gtClass,
                int numNodes, float delta, int maxIterations,
                int &iterationsUsed, cudaStream_t stream);

/**
 * @brief Run IrLP on host-side data for validation.
 *
 * This is a convenience wrapper that handles all GPU memory allocation
 * and transfer internally.
 *
 * @param activeGraph       Host CSR graph (active edges only)
 * @param h_vertexStatus    Host vertex status array
 * @param h_gtClass         Host GT class array
 * @param h_labels          Output: host label array
 * @param numNodes          Total number of nodes
 * @param delta             Convergence threshold
 * @param maxIterations     Maximum iterations
 * @param iterationsUsed    Output: iterations used
 * @return true on success
 */
bool runIrLPValidation(const CSRGraph &activeGraph,
                       const std::vector<int> &h_vertexStatus,
                       const std::vector<int> &h_gtClass,
                       std::vector<float> &h_labels, int numNodes,
                       float delta, int maxIterations, int &iterationsUsed);

#endif // VALIDATION_H
