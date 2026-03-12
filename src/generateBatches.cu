/**
 * @file generateBatches.cu
 * @brief Dynamic batch generation for DynLP.
 *
 * See generateBatches.h for full documentation.
 *
 * ============================================================================
 * IMPLEMENTATION NOTES
 * ============================================================================
 *
 * The batch generation works in two phases:
 *
 * Phase A: Partition vertices into batches.
 *   - First, select initial ground-truth vertices (batch 0).
 *   - Distribute remaining vertices evenly across batches 1..N-1.
 *   - Within each batch, designate GT fraction as ground truth.
 *
 * Phase B: Determine edges per batch.
 *   - An edge (u, v) with weight w is activated in the batch where
 *     BOTH u and v are active. For a new vertex u arriving in batch t,
 *     its edge to an already-active vertex v is included in batch t.
 *   - Edges between two vertices arriving in the same batch are also
 *     included in that batch.
 *
 * Phase C: Select deletions.
 *   - For each batch t > 0, randomly select deletedFraction of the
 *     existing active non-GT vertices for deletion.
 *   - Deletion sampling uses replacement when needed (paper Section 7.1).
 *
 * ============================================================================
 */

#include "generateBatches.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <vector>

using namespace std;

bool generateBatches(const CSRGraph &graph, const vector<float> &trueLabels,
                     const DynLPConfig &config, vector<BatchData> &batches) {
  int totalNodes = graph.numNodes;
  int numBatches = config.numBatches;

  if (numBatches <= 0 || totalNodes <= 0) {
    cout << "Error: Invalid numBatches or totalNodes.\n";
    return false;
  }

  mt19937 rng(config.seed + 12345); // Different seed from graph gen

  // ========================================================================
  // PHASE A: Partition vertices into batches
  // ========================================================================

  // Separate vertices by class for balanced GT selection
  vector<int> class0Vertices;
  vector<int> class1Vertices;
  for (int i = 0; i < totalNodes; ++i) {
    if (trueLabels[i] < 0.5f) {
      class0Vertices.push_back(i);
    } else {
      class1Vertices.push_back(i);
    }
  }
  shuffle(class0Vertices.begin(), class0Vertices.end(), rng);
  shuffle(class1Vertices.begin(), class1Vertices.end(), rng);

  // --- Batch 0: Initial ground-truth vertices ---
  // Select gtFraction of totalNodes as initial GT, balanced between classes
  int numInitialGT = max(2, static_cast<int>(totalNodes *
                                              config.groundTruthFraction));
  int numGTPerClass = numInitialGT / 2;

  batches.resize(numBatches);
  batches[0].batchId = 0;

  // Take GT vertices from each class
  set<int> initialGTSet;
  for (int i = 0; i < numGTPerClass && i < static_cast<int>(
                                                class0Vertices.size()); ++i) {
    int v = class0Vertices[i];
    batches[0].insertedNodeIds.push_back(v);
    batches[0].groundTruthNodeIds.push_back(v);
    batches[0].groundTruthLabels.push_back(0.0f);
    initialGTSet.insert(v);
  }
  for (int i = 0; i < numGTPerClass && i < static_cast<int>(
                                                class1Vertices.size()); ++i) {
    int v = class1Vertices[i];
    batches[0].insertedNodeIds.push_back(v);
    batches[0].groundTruthNodeIds.push_back(v);
    batches[0].groundTruthLabels.push_back(1.0f);
    initialGTSet.insert(v);
  }

  // Collect remaining (non-initial-GT) vertices and shuffle
  vector<int> remainingVertices;
  for (int i = 0; i < totalNodes; ++i) {
    if (initialGTSet.find(i) == initialGTSet.end()) {
      remainingVertices.push_back(i);
    }
  }
  shuffle(remainingVertices.begin(), remainingVertices.end(), rng);

  // Distribute remaining vertices evenly across batches 1..N-1
  int numRemaining = static_cast<int>(remainingVertices.size());
  int perBatch = numRemaining / max(1, numBatches - 1);
  int leftover = numRemaining - perBatch * max(1, numBatches - 1);

  int idx = 0;
  for (int b = 1; b < numBatches; ++b) {
    batches[b].batchId = b;
    int count = perBatch + (b <= leftover ? 1 : 0);
    for (int j = 0; j < count && idx < numRemaining; ++j) {
      int v = remainingVertices[idx++];
      batches[b].insertedNodeIds.push_back(v);
    }

    // Designate a small fraction as ground truth within this batch
    int numBatchGT = max(1, static_cast<int>(
                                batches[b].insertedNodeIds.size() *
                                config.groundTruthFraction));
    // Pick from the batch's inserted nodes
    vector<int> batchShuffled = batches[b].insertedNodeIds;
    shuffle(batchShuffled.begin(), batchShuffled.end(), rng);
    for (int j = 0; j < numBatchGT &&
                    j < static_cast<int>(batchShuffled.size()); ++j) {
      int v = batchShuffled[j];
      batches[b].groundTruthNodeIds.push_back(v);
      batches[b].groundTruthLabels.push_back(trueLabels[v]);
    }
  }

  // ========================================================================
  // PHASE B: Determine edges per batch
  // ========================================================================

  // Build a map: vertex → batch it arrives in
  vector<int> vertexBatch(totalNodes, -1);
  for (int b = 0; b < numBatches; ++b) {
    for (int v : batches[b].insertedNodeIds) {
      vertexBatch[v] = b;
    }
  }

  // For each edge (u, v, w) in the CSR, assign it to max(batch[u], batch[v])
  // because the edge activates when the later vertex arrives
  for (int u = 0; u < totalNodes; ++u) {
    if (vertexBatch[u] < 0) continue;
    for (int e = graph.rowPtr[u]; e < graph.rowPtr[u + 1]; ++e) {
      int v = graph.colInd[e];
      if (v < 0 || vertexBatch[v] < 0) continue;
      if (u < v) { // Process each undirected edge once
        float w = graph.values[e];
        int activeBatch = max(vertexBatch[u], vertexBatch[v]);
        batches[activeBatch].newEdgeSrc.push_back(u);
        batches[activeBatch].newEdgeDst.push_back(v);
        batches[activeBatch].newEdgeWeight.push_back(w);
      }
    }
  }

  // ========================================================================
  // PHASE C: Select deletions for batches 1..N-1
  // ========================================================================

  // Track active non-GT vertices across batches
  set<int> gtSet(initialGTSet); // All GT vertices (never deleted)
  set<int> activeNonGT;         // Currently active non-GT vertices

  // After batch 0, all batch-0 non-GT vertices are active (none in this case,
  // since batch 0 only has GT)

  for (int b = 1; b < numBatches; ++b) {
    // Add this batch's GT to the GT set
    for (int v : batches[b].groundTruthNodeIds) {
      gtSet.insert(v);
    }

    // Add this batch's non-GT inserted vertices to active set
    for (int v : batches[b].insertedNodeIds) {
      if (gtSet.find(v) == gtSet.end()) {
        activeNonGT.insert(v);
      }
    }

    // Select deletion candidates from activeNonGT
    int numToDelete = static_cast<int>(
        batches[b].insertedNodeIds.size() * config.deletedFraction);
    numToDelete = min(numToDelete, static_cast<int>(activeNonGT.size()));

    if (numToDelete > 0) {
      vector<int> candidates(activeNonGT.begin(), activeNonGT.end());
      shuffle(candidates.begin(), candidates.end(), rng);

      for (int j = 0; j < numToDelete; ++j) {
        int v = candidates[j];
        batches[b].deletedNodeIds.push_back(v);
        activeNonGT.erase(v);
      }
    }
  }

  // ========================================================================
  // Print summary
  // ========================================================================

  cout << "\nBatch generation summary:\n";
  for (int b = 0; b < numBatches; ++b) {
    cout << "  Batch " << b << ": " << batches[b].insertedNodeIds.size()
         << " inserted (" << batches[b].groundTruthNodeIds.size() << " GT), "
         << batches[b].deletedNodeIds.size() << " deleted, "
         << batches[b].newEdgeSrc.size() << " edges\n";
  }
  cout << "\n";

  return true;
}
