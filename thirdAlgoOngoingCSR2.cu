#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <set>
#include <map>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;

// ------------------- Graph Structure (CSR) -------------------
struct CSRGraph {
    vector<int> row_ptr;
    vector<int> col_ind;
    vector<int> val;
    int nNodes;
};

// ------------------- Graph Generation -------------------
CSRGraph generateGraph(int nNodes, int sRand, int eRand) {
    srand(time(0));
    CSRGraph graph;
    graph.nNodes = nNodes;

    vector<pair<int, int>> edges;
    vector<int> edge_vals;

    for (int i = 0; i < nNodes; ++i) {
        for (int j = 0; j < nNodes; ++j) {
            if (i != j) {
                int weight = sRand + rand() % (eRand - sRand + 1);
                edges.push_back({i, j});
                edge_vals.push_back(weight);
            }
        }
    }

    graph.row_ptr.resize(nNodes + 1, 0);
    for (auto& edge : edges) {
        graph.row_ptr[edge.first + 1]++;
    }
    for (int i = 1; i <= nNodes; ++i) {
        graph.row_ptr[i] += graph.row_ptr[i - 1];
    }

    graph.col_ind.resize(edges.size());
    graph.val.resize(edges.size());

    vector<int> counter = graph.row_ptr;
    for (int idx = 0; idx < edges.size(); ++idx) {
        int u = edges[idx].first;
        int v = edges[idx].second;
        int pos = counter[u]++;
        graph.col_ind[pos] = v;
        graph.val[pos] = edge_vals[idx];
    }

    return graph;
}

void saveGraphToFile(const CSRGraph& graph, const string& filename) {
    ofstream outFile(filename);
    if (outFile.is_open()) {
        outFile << graph.nNodes << "\n";
        for (auto v : graph.row_ptr) outFile << v << " ";
        outFile << "\n";
        for (auto v : graph.col_ind) outFile << v << " ";
        outFile << "\n";
        for (auto v : graph.val) outFile << v << " ";
        outFile << "\n";
        outFile.close();
    } else {
        cerr << "Unable to open file " << filename << endl;
    }
}

CSRGraph readGraphFromFile(const string& filename) {
    ifstream inFile(filename);
    CSRGraph graph;
    if (inFile.is_open()) {
        inFile >> graph.nNodes;
        graph.row_ptr.resize(graph.nNodes + 1);
        for (int i = 0; i <= graph.nNodes; ++i) inFile >> graph.row_ptr[i];
        int edgeCount = graph.row_ptr.back();
        graph.col_ind.resize(edgeCount);
        graph.val.resize(edgeCount);
        for (int i = 0; i < edgeCount; ++i) inFile >> graph.col_ind[i];
        for (int i = 0; i < edgeCount; ++i) inFile >> graph.val[i];
        inFile.close();
    } else {
        cerr << "Unable to open file " << filename << endl;
    }
    return graph;
}

void printGraph(const CSRGraph& graph) {
    cout << "RowPtr: ";
    for (auto v : graph.row_ptr) cout << v << " ";
    cout << "\nColInd: ";
    for (auto v : graph.col_ind) cout << v << " ";
    cout << "\nVal: ";
    for (auto v : graph.val) cout << v << " ";
    cout << "\n";
}
// ------------------- Sparsify CUDA -------------------
__global__ void sparsifyKernel(int* val, int nEdges, int threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nEdges) return;

    if (val[idx] < threshold) {
        val[idx] = 0;
    }
}

CSRGraph sparsifyCUDA(const CSRGraph& inputGraph, int threshold) {
    CSRGraph graph = inputGraph;
    int nEdges = graph.val.size();

    int* d_val;
    cudaMalloc(&d_val, nEdges * sizeof(int));
    cudaMemcpy(d_val, graph.val.data(), nEdges * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (nEdges + threadsPerBlock - 1) / threadsPerBlock;

    sparsifyKernel<<<blocks, threadsPerBlock>>>(d_val, nEdges, threshold);
    cudaDeviceSynchronize();

    cudaMemcpy(graph.val.data(), d_val, nEdges * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_val);

    // Remove zero-weight edges
    CSRGraph sparseGraph;
    sparseGraph.nNodes = graph.nNodes;
    sparseGraph.row_ptr.resize(graph.nNodes + 1, 0);
    for (int i = 0; i < graph.nNodes; ++i) {
        for (int j = graph.row_ptr[i]; j < graph.row_ptr[i + 1]; ++j) {
            if (graph.val[j] != 0) {
                sparseGraph.col_ind.push_back(graph.col_ind[j]);
                sparseGraph.val.push_back(graph.val[j]);
                sparseGraph.row_ptr[i + 1]++;
            }
        }
    }
    for (int i = 1; i <= graph.nNodes; ++i) {
        sparseGraph.row_ptr[i] += sparseGraph.row_ptr[i - 1];
    }
    return sparseGraph;
}

// ------------------- Label Generation -------------------
void generateLabels(int nNodes, double percentage, vector<pair<int, int>>& labeledNodes, vector<pair<int, int>>& unlabeledNodes) {
    int numLabeled = static_cast<int>(nNodes * (percentage / 100.0));
    set<int> labeledSet;
    srand(time(0));

    while (labeledSet.size() < numLabeled) {
        int node = rand() % nNodes;
        if (labeledSet.find(node) == labeledSet.end()) {
            int label = rand() % 2;
            labeledNodes.push_back({node, label});
            labeledSet.insert(node);
        }
    }

    for (int i = 0; i < nNodes; ++i) {
        if (labeledSet.find(i) == labeledSet.end()) {
            unlabeledNodes.push_back({i, -1});
        }
    }
}

void saveLabelsToFile(const vector<pair<int, int>>& nodes, const string& filename) {
    ofstream outFile(filename);
    if (outFile.is_open()) {
        for (const auto& node : nodes) {
            outFile << node.first << " " << node.second << "\n";
        }
        outFile.close();
    } else {
        cerr << "Unable to open file " << filename << endl;
    }
}

vector<pair<int, int>> readLabelsFromFile(const string& filename) {
    ifstream inFile(filename);
    vector<pair<int, int>> nodes;
    if (inFile.is_open()) {
        int node, label;
        while (inFile >> node >> label) {
            nodes.push_back({node, label});
        }
        inFile.close();
    } else {
        cerr << "Unable to open file " << filename << endl;
    }
    return nodes;
}

void printLabels(const vector<pair<int, int>>& nodes, const string& labelType) {
    cout << labelType << " Nodes: \n";
    for (const auto& node : nodes) {
        cout << "(" << node.first << ", " << node.second << ") ";
    }
    cout << "\n";
}

// ------------------- Subgraph Extraction (Unlabeled Nodes only) -------------------
CSRGraph unlabeledSubSparseGraph(const CSRGraph& graph, const vector<pair<int, int>>& unlabeledNodes) {
    set<int> unlabeledSet;
    for (const auto& node : unlabeledNodes) {
        unlabeledSet.insert(node.first);
    }

    CSRGraph subGraph;
    subGraph.nNodes = unlabeledSet.size();
    map<int, int> nodeMapping;
    int idx = 0;
    for (auto node : unlabeledSet) nodeMapping[node] = idx++;

    subGraph.row_ptr.resize(subGraph.nNodes + 1, 0);
    vector<int> temp_col_ind, temp_val;

    for (auto& p : nodeMapping) {
        int origNode = p.first;
        int mappedNode = p.second;
        for (int j = graph.row_ptr[origNode]; j < graph.row_ptr[origNode + 1]; ++j) {
            int neighbor = graph.col_ind[j];
            if (unlabeledSet.count(neighbor)) {
                temp_col_ind.push_back(nodeMapping[neighbor]);
                temp_val.push_back(graph.val[j]);
                subGraph.row_ptr[mappedNode + 1]++;
            }
        }
    }

    for (int i = 1; i <= subGraph.nNodes; ++i) {
        subGraph.row_ptr[i] += subGraph.row_ptr[i - 1];
    }
    subGraph.col_ind = temp_col_ind;
    subGraph.val = temp_val;

    return subGraph;
}

// ------------------- Upper Triangular Extraction -------------------
CSRGraph upperTriangularWithDiagonal(const CSRGraph& matrix) {
    CSRGraph upperMatrix;
    upperMatrix.nNodes = matrix.nNodes;
    upperMatrix.row_ptr.resize(matrix.nNodes + 1, 0);

    vector<int> up_col_ind, up_val;
    for (int i = 0; i < matrix.nNodes; ++i) {
        for (int j = matrix.row_ptr[i]; j < matrix.row_ptr[i + 1]; ++j) {
            if (matrix.col_ind[j] >= i) {
                up_col_ind.push_back(matrix.col_ind[j]);
                up_val.push_back(matrix.val[j]);
                upperMatrix.row_ptr[i + 1]++;
            }
        }
    }

    for (int i = 1; i <= matrix.nNodes; ++i) {
        upperMatrix.row_ptr[i] += upperMatrix.row_ptr[i - 1];
    }
    upperMatrix.col_ind = up_col_ind;
    upperMatrix.val = up_val;

    return upperMatrix;
}

// ------------------- Connected Components CUDA -------------------
__global__ void bfsExpandKernel(
    const int* row_ptr, const int* col_ind,
    int* visited, int* componentId,
    int* currentFrontier, int currentFrontierSize,
    int* nextFrontier, int* nextFrontierSize,
    int nNodes, int componentLabel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= currentFrontierSize) return;

    int node = currentFrontier[idx];
    for (int j = row_ptr[node]; j < row_ptr[node + 1]; ++j) {
        int neighbor = col_ind[j];
        if (atomicCAS(&visited[neighbor], 0, 1) == 0) {
            componentId[neighbor] = componentLabel;
            int pos = atomicAdd(nextFrontierSize, 1);
            nextFrontier[pos] = neighbor;
        }
    }
}

vector<vector<int>> findConnectedComponent(const CSRGraph& graph) {
    int nNodes = graph.nNodes;
    int nEdges = graph.col_ind.size();

    int* d_row_ptr;
    int* d_col_ind;
    int* d_visited;
    int* d_componentId;
    int* d_currentFrontier;
    int* d_nextFrontier;
    int* d_nextFrontierSize;

    cudaMalloc(&d_row_ptr, (nNodes + 1) * sizeof(int));
    cudaMalloc(&d_col_ind, nEdges * sizeof(int));
    cudaMalloc(&d_visited, nNodes * sizeof(int));
    cudaMalloc(&d_componentId, nNodes * sizeof(int));
    cudaMalloc(&d_currentFrontier, nNodes * sizeof(int));
    cudaMalloc(&d_nextFrontier, nNodes * sizeof(int));
    cudaMalloc(&d_nextFrontierSize, sizeof(int));

    cudaMemcpy(d_row_ptr, graph.row_ptr.data(), (nNodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, graph.col_ind.data(), nEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_visited, 0, nNodes * sizeof(int));
    cudaMemset(d_componentId, -1, nNodes * sizeof(int));

    int* h_currentFrontier = new int[nNodes];
    int* h_nextFrontier = new int[nNodes];

    int componentLabel = 0;

    for (int startNode = 0; startNode < nNodes; ++startNode) {
        int visitedVal;
        cudaMemcpy(&visitedVal, d_visited + startNode, sizeof(int), cudaMemcpyDeviceToHost);
        if (visitedVal == 0) {
            cudaMemset(d_nextFrontierSize, 0, sizeof(int));
            cudaMemcpy(d_currentFrontier, &startNode, sizeof(int), cudaMemcpyHostToDevice);

            int one = 1;
            cudaMemcpy(d_visited + startNode, &one, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_componentId + startNode, &componentLabel, sizeof(int), cudaMemcpyHostToDevice);

            int currentFrontierSize = 1;

            while (currentFrontierSize > 0) {
                int threadsPerBlock = 256;
                int blocks = (currentFrontierSize + threadsPerBlock - 1) / threadsPerBlock;

                bfsExpandKernel<<<blocks, threadsPerBlock>>>(
                    d_row_ptr, d_col_ind,
                    d_visited, d_componentId,
                    d_currentFrontier, currentFrontierSize,
                    d_nextFrontier, d_nextFrontierSize,
                    nNodes, componentLabel
                );
                cudaDeviceSynchronize();

                cudaMemcpy(&currentFrontierSize, d_nextFrontierSize, sizeof(int), cudaMemcpyDeviceToHost);
                swap(d_currentFrontier, d_nextFrontier);
                cudaMemset(d_nextFrontierSize, 0, sizeof(int));
            }
            componentLabel++;
        }
    }

    vector<int> componentId(nNodes);
    cudaMemcpy(componentId.data(), d_componentId, nNodes * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_visited);
    cudaFree(d_componentId);
    cudaFree(d_currentFrontier);
    cudaFree(d_nextFrontier);
    cudaFree(d_nextFrontierSize);
    delete[] h_currentFrontier;
    delete[] h_nextFrontier;

    map<int, vector<int>> componentsMap;
    for (int i = 0; i < nNodes; ++i) {
        if (componentId[i] >= 0)
            componentsMap[componentId[i]].push_back(i);
    }

    vector<vector<int>> components;
    for (auto& kv : componentsMap) {
        components.push_back(kv.second);
    }

    return components;
}

// ------------------- Classification Voting -------------------
void labelToClass(const vector<pair<int, int>>& readLabeledNodes, set<int>& class0, set<int>& class1) {
    for (const auto& node : readLabeledNodes) {
        if (node.second == 0) {
            class0.insert(node.first);
        } else if (node.second == 1) {
            class1.insert(node.first);
        }
    }
}

struct ComponentProperties {
    vector<int> nodes;
    int index;
    int class0Sum;
    int class1Sum;
    int total;
    float label;
    ComponentProperties() : index(0), class0Sum(0), class1Sum(0), total(0), label(0.5) {}
};

void parallelEdgeComponent(const CSRGraph& graph, const vector<vector<int>>& connectedComponents, const set<int>& class0, const set<int>& class1, vector<ComponentProperties>& componentsProperties, const vector<pair<int,int>>& readUnlabeledNodes) {
    map<int, int> componentMap;
    for (int i = 0; i < connectedComponents.size(); ++i) {
        for (const auto& node : connectedComponents[i]) {
            componentMap[node] = i;
        }
    }

    for (int i = 0; i < connectedComponents.size(); ++i) {
        ComponentProperties cp;
        cp.index = i;
        cp.nodes = connectedComponents[i];
        for (const auto& node : connectedComponents[i]) {
            int realNode = readUnlabeledNodes[node].first;
            for (int j = graph.row_ptr[realNode]; j < graph.row_ptr[realNode + 1]; ++j) {
                int neighbor = graph.col_ind[j];
                if (class0.count(neighbor)) {
                    cp.class0Sum += graph.val[j];
                } else if (class1.count(neighbor)) {
                    cp.class1Sum += graph.val[j];
                }
            }
        }
        cp.total = cp.class0Sum + cp.class1Sum;
        componentsProperties.push_back(cp);
    }
}
int main() {
    srand(time(0));  // Random seed ONCE at beginning

    int nNodes = 15;
    int sRand = 1;
    int eRand = 10;
    int threshold = 8;
    double percentage = 40.0;

    // 1. Generate Graph and Save
    CSRGraph graph = generateGraph(nNodes, sRand, eRand);
    saveGraphToFile(graph, "graph.txt");

    CSRGraph readGraph = readGraphFromFile("graph.txt");
    cout << "Original Graph (CSR)\n";
    printGraph(readGraph);

    // 2. Sparsify
    cout << "\nSparse Graph (CSR)\n";
    CSRGraph sparseGraph = sparsifyCUDA(readGraph, threshold);
    saveGraphToFile(sparseGraph, "sparseGraph.txt");
    printGraph(sparseGraph);

    // 3. Label Generation
    vector<pair<int, int>> labeledNodes, unlabeledNodes;
    generateLabels(nNodes, percentage, labeledNodes, unlabeledNodes);
    saveLabelsToFile(labeledNodes, "labeledNodes.txt");
    saveLabelsToFile(unlabeledNodes, "unlabeledNodes.txt");

    vector<pair<int, int>> readLabeledNodes = readLabelsFromFile("labeledNodes.txt");
    vector<pair<int, int>> readUnlabeledNodes = readLabelsFromFile("unlabeledNodes.txt");

    cout << "\n";
    printLabels(readLabeledNodes, "Labeled");
    cout << "\n";
    printLabels(readUnlabeledNodes, "Unlabeled");

    // 4. Extract Unlabeled Subgraph
    CSRGraph subSparseGraph = unlabeledSubSparseGraph(sparseGraph, readUnlabeledNodes);
    saveGraphToFile(subSparseGraph, "unlabeledSubSparseGraph.txt");
    cout << "\nUnlabeled Subgraph (CSR)\n";
    printGraph(subSparseGraph);

    // 5. Upper Triangular
    CSRGraph upperSubSparseGraph = upperTriangularWithDiagonal(subSparseGraph);
    saveGraphToFile(upperSubSparseGraph, "upperSubSparseGraph.txt");
    cout << "\nUpper Triangular Subgraph (CSR)\n";
    printGraph(upperSubSparseGraph);

    // 6. Find Connected Components
    vector<vector<int>> connectedComponents = findConnectedComponent(upperSubSparseGraph);
    cout << "Number of connected components: " << connectedComponents.size() << "\n";
    for (auto& component : connectedComponents) {
        cout << "Component: ";
        for (auto& node : component) {
            cout << readUnlabeledNodes[node].first << " ";
        }
        cout << "\n";
    }

    // 7. Prepare class0/class1 sets
    set<int> class0, class1;
    labelToClass(readLabeledNodes, class0, class1);

    cout << "Class 0 Nodes: ";
    for (auto node : class0) cout << node << " ";
    cout << "\n";
    cout << "Class 1 Nodes: ";
    for (auto node : class1) cout << node << " ";
    cout << "\n";

    // 8. Parallel Edge Classification
    cout << "Predicting Labels for Unlabeled Nodes...\n";

    vector<ComponentProperties> componentsProperties;
    vector<pair<int, int>> resultUnlabeledNodes;

    parallelEdgeComponent(readGraph, connectedComponents, class0, class1, componentsProperties, readUnlabeledNodes);

    for (auto& cp : componentsProperties) {
        for (auto& nodeIdx : cp.nodes) {
            int realNode = readUnlabeledNodes[nodeIdx].first;
            int predictedLabel = 0;
            if (cp.total != 0) {
                predictedLabel = (cp.label + (0 - cp.label) * 1.0 * cp.class0Sum / cp.total / 2
                                  + (1 - cp.label) * 1.0 * cp.class1Sum / cp.total / 2) >= 0.5 ? 1 : 0;
            }
            resultUnlabeledNodes.push_back({realNode, predictedLabel});
        }
    }

    for (auto& p : resultUnlabeledNodes) {
        cout << "(" << p.first << ", " << p.second << ")\n";
    }

    return 0;
}
