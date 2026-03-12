/**
 * @file graphTypes.h
 * @brief Shared data types for the DynLP (Dynamic Label Propagation) project.
 *
 * Defines the CSR graph representation, configuration parameters,
 * batch data structures, and CUDA utility macros used throughout the project.
 *
 * ============================================================================
 * GRAPH REPRESENTATION
 * ============================================================================
 *
 * We store graphs in Compressed Sparse Row (CSR) format:
 *   - rowPtr[v] .. rowPtr[v+1]-1 gives the range of neighbors for vertex v
 *   - colInd[i] gives the destination vertex of the i-th edge
 *   - values[i] gives the similarity weight of the i-th edge
 *
 * The graph is undirected and weighted. Weights represent pairwise
 * similarity in [0, 1]. For dynamic updates, we use "soft deletion"
 * by setting colInd entries to -1 rather than removing them.
 *
 * ============================================================================
 * BINARY CLASSIFICATION
 * ============================================================================
 *
 * DynLP performs binary semi-supervised classification. Each vertex has a
 * fractional label F_u in [0, 1]:
 *   - F_u close to 0 → class 0
 *   - F_u close to 1 → class 1
 *   - F_u = 0.5       → neutral / uncertain
 *
 * Ground-truth vertices have fixed labels (0.0 or 1.0) that never change.
 *
 * ============================================================================
 */

#ifndef GRAPH_TYPES_H
#define GRAPH_TYPES_H

#include <cstdio>          // fprintf
#include <cstdlib>         // exit, EXIT_FAILURE
#include <cuda_runtime.h>  // cudaError_t, cudaGetErrorString, cudaGetLastError
#include <string>
#include <vector>

// ============================================================================
// CUDA Error Checking Macros
// ============================================================================

/**
 * @brief Check a CUDA runtime API call and exit on error.
 *
 * Usage: CUDA_CHECK(cudaMalloc(&ptr, size));
 */
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/**
 * @brief Check the last CUDA kernel launch for errors.
 *
 * Usage: call immediately after a kernel launch.
 */
#define CUDA_KERNEL_CHECK()                                                    \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                       \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Kernel Error at %s:%d - %s\n", __FILE__,          \
              __LINE__, cudaGetErrorString(err));                               \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ============================================================================
// CSR Graph Representation
// ============================================================================

/**
 * @brief Host-side CSR graph with similarity weights.
 *
 * The graph is stored on the host (CPU) and transferred to the device (GPU)
 * in batches. For dynamic updates, rows are appended as new batches arrive.
 * Deleted edges are soft-deleted by setting colInd entries to -1.
 */
struct CSRGraph {
  std::vector<int> rowPtr;     ///< Row pointer array (size = numNodes + 1)
  std::vector<int> colInd;     ///< Column index array (size = numEdges)
  std::vector<float> values;   ///< Edge weight (similarity) array
  int numNodes;                ///< Current number of active vertices
  int numEdges;                ///< Current number of edges (including soft-deleted)

  CSRGraph() : numNodes(0), numEdges(0) { rowPtr.push_back(0); }
};

/**
 * @brief Device-side CSR graph pointers for GPU kernels.
 *
 * This structure holds raw device pointers to the CSR arrays on the GPU.
 * Memory is managed externally by the DynLP orchestrator.
 */
struct DeviceCSR {
  int *d_rowPtr;     ///< Device row pointer array
  int *d_colInd;     ///< Device column index array
  float *d_values;   ///< Device edge weight array
  int numNodes;      ///< Number of nodes currently on device
  int numEdges;      ///< Number of edges currently on device
};

// ============================================================================
// Configuration Parameters
// ============================================================================

/**
 * @brief Configuration for a DynLP run.
 *
 * All thresholds and fractions are configurable via command-line arguments.
 * Default values are chosen based on the paper's experimental setup.
 */
struct DynLPConfig {
  // --- Graph generation parameters ---
  int totalNodes;              ///< Total number of vertices across all batches
  int avgDegree;               ///< Average degree for Erdős–Rényi generation
  int numBatches;              ///< Number of dynamic batches (default: 10)
  unsigned int seed;           ///< RNG seed for reproducibility

  // --- DynLP algorithm parameters ---
  float tau;                   ///< Sparsification threshold (default: avg edge weight)
  float delta;                 ///< Convergence threshold for iterative propagation
  int maxIterations;           ///< Maximum iterations per propagation step

  // --- Batch composition fractions ---
  float groundTruthFraction;   ///< Fraction of ground-truth nodes per batch
  float unlabeledFraction;     ///< Fraction of unlabeled nodes per batch
  float deletedFraction;       ///< Fraction of deleted nodes per batch

  // --- Output control ---
  std::string outputDir;       ///< Output directory path
  bool validate;               ///< Whether to run IrLP baseline for comparison
  bool verbose;                ///< Whether to print detailed per-batch info

  // --- Default constructor with paper-recommended values ---
  DynLPConfig()
      : totalNodes(50000), avgDegree(5), numBatches(10), seed(42),
        tau(-1.0f),  // -1 means "auto: use average edge weight"
        delta(0.0001f), maxIterations(100000),
        groundTruthFraction(0.01f), unlabeledFraction(0.90f),
        deletedFraction(0.09f), outputDir("output/"), validate(true),
        verbose(true) {}
};

// ============================================================================
// Batch Data
// ============================================================================

/**
 * @brief Data for a single dynamic batch of changes.
 *
 * Each batch contains new vertices (some with ground truth, most without)
 * and a list of vertices to delete from the existing graph. The edge data
 * for new vertices is stored as a local CSR subgraph.
 */
struct BatchData {
  // --- Inserted vertices ---
  std::vector<int> insertedNodeIds;        ///< Global IDs of newly inserted vertices
  std::vector<int> groundTruthNodeIds;     ///< Subset of inserted that have GT labels
  std::vector<float> groundTruthLabels;    ///< Ground truth labels (0.0 or 1.0)

  // --- Deleted vertices ---
  std::vector<int> deletedNodeIds;         ///< Global IDs of vertices to remove

  // --- Edges for new vertices (local CSR of the subgraph induced by new nodes) ---
  std::vector<int> newEdgeSrc;             ///< Source vertex of each new edge
  std::vector<int> newEdgeDst;             ///< Destination vertex of each new edge
  std::vector<float> newEdgeWeight;        ///< Similarity weight of each new edge

  int batchId;                             ///< Batch sequence number (0-indexed)
};

// ============================================================================
// Vertex State
// ============================================================================

/**
 * @brief Status of a vertex in the dynamic graph.
 */
enum VertexStatus : int {
  VERTEX_ACTIVE = 0,       ///< Active vertex in the graph
  VERTEX_DELETED = 1,      ///< Soft-deleted vertex
  VERTEX_GROUND_TRUTH = 2  ///< Ground-truth vertex (label is fixed)
};

#endif // GRAPH_TYPES_H
