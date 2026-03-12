/**
 * @file generateSparseGraph.cu
 * @brief Erdős–Rényi sparse graph generator with similarity-based weights.
 *
 * Generates a sparse undirected weighted graph suitable for testing DynLP.
 * Edge weights mimic feature-based cosine similarity: vertices of the same
 * class have higher similarity (0.5–1.0) while cross-class edges have
 * lower similarity (0.0–0.5).
 *
 * ============================================================================
 * GENERATION STRATEGY
 * ============================================================================
 *
 * 1. Assign each vertex a random binary class label (balanced).
 * 2. Build a random spanning tree via random permutation to ensure
 *    connectivity. This guarantees every vertex is reachable.
 * 3. Add random edges until the target average degree is reached.
 *    Edges are sampled uniformly; duplicates and self-loops are rejected.
 * 4. Assign weights based on class agreement:
 *    - Same class:  weight ~ Uniform(0.5, 1.0)
 *    - Cross class: weight ~ Uniform(0.0, 0.5)
 * 5. Convert the edge list to undirected CSR format.
 *
 * ============================================================================
 */

#include "generateSparseGraph.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace std;

namespace {

/**
 * @brief Convert an edge list to undirected CSR format.
 *
 * For each edge (u, v, w) in the input, both (u→v) and (v→u) are stored.
 * The CSR arrays are sorted by source vertex, then by destination vertex
 * within each row for cache-friendly access patterns.
 */
void edgeListToCSR(const vector<pair<int, int>> &edges,
                   const vector<float> &weights, int numNodes,
                   CSRGraph &graph) {
  // Count degree of each vertex (undirected: each edge contributes twice)
  vector<int> degree(numNodes, 0);
  for (const auto &edge : edges) {
    degree[edge.first]++;
    degree[edge.second]++;
  }

  // Build rowPtr using prefix sum
  graph.numNodes = numNodes;
  graph.rowPtr.resize(numNodes + 1);
  graph.rowPtr[0] = 0;
  for (int i = 0; i < numNodes; ++i) {
    graph.rowPtr[i + 1] = graph.rowPtr[i] + degree[i];
  }

  int totalEntries = graph.rowPtr[numNodes];
  graph.numEdges = totalEntries;
  graph.colInd.resize(totalEntries);
  graph.values.resize(totalEntries);

  // Fill using insertion offsets
  vector<int> offset(numNodes, 0);
  for (size_t e = 0; e < edges.size(); ++e) {
    int u = edges[e].first;
    int v = edges[e].second;
    float w = weights[e];

    int posU = graph.rowPtr[u] + offset[u];
    graph.colInd[posU] = v;
    graph.values[posU] = w;
    offset[u]++;

    int posV = graph.rowPtr[v] + offset[v];
    graph.colInd[posV] = u;
    graph.values[posV] = w;
    offset[v]++;
  }

  // Sort each row by column index for coalesced GPU access
  for (int u = 0; u < numNodes; ++u) {
    int start = graph.rowPtr[u];
    int end = graph.rowPtr[u + 1];
    // Simple insertion sort (rows are small for sparse graphs)
    for (int i = start + 1; i < end; ++i) {
      int keyCol = graph.colInd[i];
      float keyVal = graph.values[i];
      int j = i - 1;
      while (j >= start && graph.colInd[j] > keyCol) {
        graph.colInd[j + 1] = graph.colInd[j];
        graph.values[j + 1] = graph.values[j];
        j--;
      }
      graph.colInd[j + 1] = keyCol;
      graph.values[j + 1] = keyVal;
    }
  }
}

} // namespace

// ============================================================================
// Public API
// ============================================================================

bool generateSparseGraph(int totalNodes, int avgDegree, unsigned int seed,
                         CSRGraph &graph, vector<float> &trueLabels,
                         const string &outputPrefix) {
  if (totalNodes <= 1) {
    cout << "Error: totalNodes must be > 1.\n";
    return false;
  }
  if (avgDegree <= 0) {
    cout << "Error: avgDegree must be positive.\n";
    return false;
  }

  mt19937 rng(seed);

  // --- Step 1: Assign binary class labels (balanced) ---
  trueLabels.resize(totalNodes);
  for (int i = 0; i < totalNodes; ++i) {
    trueLabels[i] = (i < totalNodes / 2) ? 0.0f : 1.0f;
  }
  // Shuffle to randomize class assignment
  shuffle(trueLabels.begin(), trueLabels.end(), rng);

  // --- Step 2: Build a random spanning tree for connectivity ---
  // Create a random permutation of vertices
  vector<int> perm(totalNodes);
  iota(perm.begin(), perm.end(), 0);
  shuffle(perm.begin(), perm.end(), rng);

  set<pair<int, int>> edgeSet; // Use ordered pairs (min, max) for dedup
  vector<pair<int, int>> edgeList;
  vector<float> weightList;

  uniform_real_distribution<float> sameClassDist(0.5f, 1.0f);
  uniform_real_distribution<float> crossClassDist(0.01f, 0.5f);

  // Connect perm[i] to perm[i+1] for i = 0..n-2 → spanning tree
  for (int i = 0; i < totalNodes - 1; ++i) {
    int u = min(perm[i], perm[i + 1]);
    int v = max(perm[i], perm[i + 1]);
    if (u == v) continue; // skip self-loops (shouldn't happen)
    edgeSet.insert({u, v});

    // Weight depends on class agreement
    float w;
    if (trueLabels[u] == trueLabels[v]) {
      w = sameClassDist(rng);
    } else {
      w = crossClassDist(rng);
    }
    edgeList.push_back({u, v});
    weightList.push_back(w);
  }

  // --- Step 3: Add random edges until target total is reached ---
  // Target total undirected edges = (totalNodes * avgDegree) / 2
  long long targetEdges = static_cast<long long>(totalNodes) * avgDegree / 2;
  if (targetEdges < static_cast<long long>(totalNodes) - 1) {
    targetEdges = totalNodes - 1; // At minimum, keep spanning tree
  }

  uniform_int_distribution<int> nodeDist(0, totalNodes - 1);
  int attempts = 0;
  int maxAttempts = static_cast<int>(targetEdges) * 10; // Avoid infinite loop

  while (static_cast<long long>(edgeList.size()) < targetEdges &&
         attempts < maxAttempts) {
    int u = nodeDist(rng);
    int v = nodeDist(rng);
    if (u == v) {
      attempts++;
      continue;
    }
    int a = min(u, v);
    int b = max(u, v);
    if (edgeSet.count({a, b})) {
      attempts++;
      continue;
    }

    edgeSet.insert({a, b});

    float w;
    if (trueLabels[a] == trueLabels[b]) {
      w = sameClassDist(rng);
    } else {
      w = crossClassDist(rng);
    }
    edgeList.push_back({a, b});
    weightList.push_back(w);
    attempts = 0; // Reset on success
  }

  cout << "Generated graph: " << totalNodes << " nodes, " << edgeList.size()
       << " undirected edges (avg degree ~"
       << (2.0 * edgeList.size() / totalNodes) << ")\n";

  // --- Step 4: Convert to CSR ---
  edgeListToCSR(edgeList, weightList, totalNodes, graph);

  // --- Step 5: Optionally write CSR files ---
  if (!outputPrefix.empty()) {
    if (!ensureDirectory(outputPrefix.substr(
            0, outputPrefix.find_last_of("/")))) {
      return false;
    }

    // Write RowPtr
    {
      ofstream file(outputPrefix + "RowPtr.txt");
      if (!file.is_open()) {
        cout << "Error: Could not write RowPtr file.\n";
        return false;
      }
      for (int i = 0; i <= graph.numNodes; ++i) {
        file << graph.rowPtr[i] << "\n";
      }
    }

    // Write ColInd
    {
      ofstream file(outputPrefix + "ColInd.txt");
      if (!file.is_open()) {
        cout << "Error: Could not write ColInd file.\n";
        return false;
      }
      for (int i = 0; i < graph.numEdges; ++i) {
        file << graph.colInd[i] << "\n";
      }
    }

    // Write Values
    {
      ofstream file(outputPrefix + "Values.txt");
      if (!file.is_open()) {
        cout << "Error: Could not write Values file.\n";
        return false;
      }
      file << fixed;
      for (int i = 0; i < graph.numEdges; ++i) {
        file << setprecision(6) << graph.values[i] << "\n";
      }
    }

    // Write true labels
    {
      ofstream file(outputPrefix + "TrueLabels.txt");
      if (!file.is_open()) {
        cout << "Error: Could not write TrueLabels file.\n";
        return false;
      }
      for (int i = 0; i < totalNodes; ++i) {
        file << i << " " << trueLabels[i] << "\n";
      }
    }

    cout << "CSR graph written to: " << outputPrefix << "*.txt\n";
  }

  return true;
}
