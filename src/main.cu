/**
 * @file main.cu
 * @brief Entry point for the DynLP (Dynamic Label Propagation) application.
 *
 * This program implements the DynLP algorithm for GPU-parallel dynamic
 * batch label propagation in semi-supervised learning. It generates a
 * synthetic sparse graph, splits it into batches, runs DynLP, and
 * optionally validates accuracy against the IrLP baseline.
 *
 * ============================================================================
 * USAGE
 * ============================================================================
 *
 *   ./dynLP [options]
 *
 * Run with --help for full option listing. Key parameters:
 *   --totalNodes N    Total vertices (default: 50000)
 *   --numBatches N    Number of batches (default: 10)
 *   --avgDegree  N    Average degree (default: 5)
 *   --delta      F    Convergence threshold (default: 0.0001)
 *   --tau        F    Sparsification threshold (default: auto)
 *
 * ============================================================================
 * EXECUTION FLOW
 * ============================================================================
 *
 * 1. Parse command-line arguments
 * 2. Generate synthetic sparse graph (Erdős–Rényi with similarity weights)
 * 3. Generate batches of dynamic changes
 * 4. Run DynLP algorithm across all batches
 * 5. (Optional) Run IrLP baseline for accuracy comparison
 * 6. Write results to output files
 * 7. Print summary
 *
 * ============================================================================
 */

#include "dynLP.h"
#include "generateBatches.h"
#include "generateSparseGraph.h"
#include "utils.h"
#include "validation.h"

#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

int main(int argc, char *argv[]) {
  // ========================================================================
  // Step 1: Parse command-line arguments
  // ========================================================================

  DynLPConfig config;
  if (!parseArguments(argc, argv, config)) {
    return 1;
  }
  printConfig(config);

  // Ensure output directories exist
  if (!ensureDirectory(config.outputDir)) {
    return 1;
  }
  if (!ensureDirectory(config.outputDir + "/graph")) {
    return 1;
  }

  // ========================================================================
  // Step 2: Generate synthetic sparse graph
  // ========================================================================

  cout << "Generating sparse graph...\n";

  CSRGraph fullGraph;
  vector<float> trueLabels;
  string graphPrefix = config.outputDir + "/graph/csr";

  if (!generateSparseGraph(config.totalNodes, config.avgDegree, config.seed,
                           fullGraph, trueLabels, graphPrefix)) {
    cout << "Error: Graph generation failed.\n";
    return 1;
  }

  // ========================================================================
  // Step 3: Generate batches of dynamic changes
  // ========================================================================

  cout << "Generating batches...\n";

  vector<BatchData> batches;
  if (!generateBatches(fullGraph, trueLabels, config, batches)) {
    cout << "Error: Batch generation failed.\n";
    return 1;
  }

  // ========================================================================
  // Step 4: Run DynLP algorithm
  // ========================================================================

  cout << "Running DynLP...\n\n";

  vector<float> dynlpLabels;
  vector<float> batchTimes;
  vector<float> batchAccuracy;

  if (!runDynLP(config, fullGraph, trueLabels, batches, dynlpLabels,
                batchTimes, batchAccuracy)) {
    cout << "Error: DynLP execution failed.\n";
    return 1;
  }

  // ========================================================================
  // Step 5: (Optional) Run IrLP baseline validation
  // ========================================================================

  vector<float> irlpLabels;
  float irlpAccuracy = 0.0f;
  float irlpTimeMs = 0.0f;

  if (config.validate) {
    cout << "Running IrLP baseline for validation...\n";

    // Build the final active graph for IrLP
    // (IrLP gets the full graph snapshot at the final timestep)
    vector<int> h_vertexStatus(config.totalNodes, VERTEX_DELETED);
    vector<int> h_gtClass(config.totalNodes, -1);
    irlpLabels.resize(config.totalNodes, 0.5f);

    // Replay all batches to determine final vertex status
    for (int b = 0; b < config.numBatches; ++b) {
      for (int v : batches[b].insertedNodeIds) {
        h_vertexStatus[v] = VERTEX_ACTIVE;
      }
      for (int v : batches[b].deletedNodeIds) {
        h_vertexStatus[v] = VERTEX_DELETED;
      }
      for (size_t i = 0; i < batches[b].groundTruthNodeIds.size(); ++i) {
        int v = batches[b].groundTruthNodeIds[i];
        h_vertexStatus[v] = VERTEX_GROUND_TRUTH;
        h_gtClass[v] = (batches[b].groundTruthLabels[i] < 0.5f) ? 0 : 1;
        irlpLabels[v] = batches[b].groundTruthLabels[i];
      }
    }

    // Build active CSR for the final state
    CSRGraph activeGraph;
    vector<int> vertexBatchMap(config.totalNodes, -1);
    for (int b = 0; b < config.numBatches; ++b) {
      for (int v : batches[b].insertedNodeIds) {
        vertexBatchMap[v] = b;
      }
    }

    activeGraph.rowPtr.clear();
    activeGraph.colInd.clear();
    activeGraph.values.clear();
    activeGraph.rowPtr.push_back(0);
    activeGraph.numNodes = config.totalNodes;

    for (int u = 0; u < config.totalNodes; ++u) {
      if (h_vertexStatus[u] == VERTEX_DELETED || vertexBatchMap[u] < 0) {
        activeGraph.rowPtr.push_back(activeGraph.rowPtr.back());
        continue;
      }
      int rowStart = fullGraph.rowPtr[u];
      int rowEnd = fullGraph.rowPtr[u + 1];
      for (int e = rowStart; e < rowEnd; ++e) {
        int v = fullGraph.colInd[e];
        if (v < 0) continue;
        if (h_vertexStatus[v] == VERTEX_DELETED || vertexBatchMap[v] < 0)
          continue;
        activeGraph.colInd.push_back(v);
        activeGraph.values.push_back(fullGraph.values[e]);
      }
      activeGraph.rowPtr.push_back(
          static_cast<int>(activeGraph.colInd.size()));
    }
    activeGraph.numEdges = static_cast<int>(activeGraph.colInd.size());

    // Time the IrLP run
    cudaEvent_t irlpStart, irlpEnd;
    CUDA_CHECK(cudaEventCreate(&irlpStart));
    CUDA_CHECK(cudaEventCreate(&irlpEnd));
    CUDA_CHECK(cudaEventRecord(irlpStart));

    int irlpIter = 0;
    if (!runIrLPValidation(activeGraph, h_vertexStatus, h_gtClass,
                           irlpLabels, config.totalNodes, config.delta,
                           config.maxIterations, irlpIter)) {
      cout << "Error: IrLP validation failed.\n";
      return 1;
    }

    CUDA_CHECK(cudaEventRecord(irlpEnd));
    CUDA_CHECK(cudaEventSynchronize(irlpEnd));
    CUDA_CHECK(cudaEventElapsedTime(&irlpTimeMs, irlpStart, irlpEnd));

    // Compute IrLP accuracy
    int correct = 0, total = 0;
    for (int i = 0; i < config.totalNodes; ++i) {
      if (h_vertexStatus[i] == VERTEX_DELETED) continue;
      if (h_vertexStatus[i] == VERTEX_GROUND_TRUTH) continue;
      int predicted = (irlpLabels[i] >= 0.5f) ? 1 : 0;
      int truth = (trueLabels[i] >= 0.5f) ? 1 : 0;
      if (predicted == truth) correct++;
      total++;
    }
    irlpAccuracy = (total > 0) ? static_cast<float>(correct) / total : 1.0f;

    cout << "IrLP baseline: " << irlpIter << " iterations, "
         << irlpTimeMs << " ms, accuracy "
         << fixed << setprecision(2) << (irlpAccuracy * 100.0f) << "%\n\n";

    CUDA_CHECK(cudaEventDestroy(irlpStart));
    CUDA_CHECK(cudaEventDestroy(irlpEnd));
  }

  // ========================================================================
  // Step 6: Write results
  // ========================================================================

  // Write final DynLP labels
  if (!writeLabelsToFile(config.outputDir + "/dynlp_labels.txt",
                         dynlpLabels.data(), config.totalNodes)) {
    cout << "Error: Could not write DynLP labels.\n";
    return 1;
  }

  // Write timing and accuracy CSV
  if (!writeResultsCSV(config.outputDir + "/results.csv", batchTimes.data(),
                       batchAccuracy.data(), config.numBatches)) {
    cout << "Error: Could not write results CSV.\n";
    return 1;
  }

  // Write IrLP labels if validation was run
  if (config.validate) {
    if (!writeLabelsToFile(config.outputDir + "/irlp_labels.txt",
                           irlpLabels.data(), config.totalNodes)) {
      cout << "Error: Could not write IrLP labels.\n";
      return 1;
    }
  }

  // ========================================================================
  // Step 7: Print summary
  // ========================================================================

  float totalDynLPTime =
      accumulate(batchTimes.begin(), batchTimes.end(), 0.0f);
  float avgAccuracy =
      accumulate(batchAccuracy.begin(), batchAccuracy.end(), 0.0f) /
      config.numBatches;
  float finalAccuracy = batchAccuracy.back();

  cout << "============================================================\n"
       << " DynLP Results Summary\n"
       << "============================================================\n"
       << "  Total DynLP time:     " << fixed << setprecision(2)
       << totalDynLPTime << " ms\n"
       << "  Average accuracy:     " << fixed << setprecision(2)
       << (avgAccuracy * 100.0f) << "%\n"
       << "  Final batch accuracy: " << fixed << setprecision(2)
       << (finalAccuracy * 100.0f) << "%\n";

  if (config.validate) {
    float speedup = (irlpTimeMs > 0.0f) ? (irlpTimeMs / totalDynLPTime) : 0.0f;
    cout << "  IrLP baseline time:   " << fixed << setprecision(2)
         << irlpTimeMs << " ms\n"
         << "  IrLP accuracy:        " << fixed << setprecision(2)
         << (irlpAccuracy * 100.0f) << "%\n"
         << "  Speedup vs IrLP:      " << fixed << setprecision(1)
         << speedup << "x\n";
  }

  cout << "============================================================\n"
       << "\n"
       << "Per-batch results:\n";
  for (int b = 0; b < config.numBatches; ++b) {
    cout << "  Batch " << setw(2) << b << ": " << fixed << setprecision(2)
         << setw(10) << batchTimes[b] << " ms, accuracy " << fixed
         << setprecision(2) << (batchAccuracy[b] * 100.0f) << "%\n";
  }

  cout << "\nOutput files written to: " << config.outputDir << "/\n";
  cout << "  dynlp_labels.txt  — Final predicted labels\n";
  cout << "  results.csv       — Per-batch timing and accuracy\n";
  if (config.validate) {
    cout << "  irlp_labels.txt   — IrLP baseline labels\n";
  }
  cout << "  graph/            — Generated graph in CSR format\n";

  return 0;
}
