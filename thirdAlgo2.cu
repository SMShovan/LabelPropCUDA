#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <set>
#include <map>
#include <cstdlib>

using namespace std;

vector<vector<int>> generateGraph(int nNodes, int sRand, int eRand) {
    vector<vector<int>> adjacencyMatrix(nNodes, vector<int>(nNodes));
    srand(time(0));
    for (int i = 0; i < nNodes; ++i) {
        for (int j = 0; j < nNodes; ++j) {
            if (i == j) {
                adjacencyMatrix[i][j] = 0; // No self-edges
            } else {
                adjacencyMatrix[i][j] = sRand + rand() % (eRand - sRand + 1);
            }
        }
    }
    return adjacencyMatrix;
}

void saveGraphToFile(const vector<vector<int>>& graph, const string& filename) {
    ofstream outFile(filename);
    if (outFile.is_open()) {
        for (const auto& row : graph) {
            for (const auto& cell : row) {
                outFile << cell << " ";
            }
            outFile << "\n";
        }
        outFile.close();
    } else {
        cerr << "Unable to open file " << filename << endl;
    }
}

vector<vector<int>> readGraphFromFile(const string& filename) {
    ifstream inFile(filename);
    vector<vector<int>> graph;
    if (inFile.is_open()) {
        string line;
        while (getline(inFile, line)) {
            vector<int> row;
            istringstream iss(line);
            int value;
            while (iss >> value) {
                row.push_back(value);
            }
            graph.push_back(row);
        }
        inFile.close();
    } else {
        cerr << "Unable to open file " << filename << endl;
    }
    return graph;
}

void printGraph(const vector<vector<int>>& graph) {
    for (const auto& row : graph) {
        for (const auto& cell : row) {
            cout << cell << " ";
        }
        cout << "\n";
    }
}
__global__ void sparsifyKernel(int* graph, int nNodes, int threshold) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < nNodes && col < nNodes) {
        int index = row * nNodes + col;
        if (graph[index] < threshold) {
            graph[index] = 0;
        }
    }
}
vector<vector<int>> sparsifyCUDA(vector<vector<int>>& graph, int threshold) {
    int nNodes = graph.size();
    
    // Flatten the graph to 1D array for CUDA
    vector<int> flatGraph(nNodes * nNodes);
    for (int i = 0; i < nNodes; ++i) {
        for (int j = 0; j < nNodes; ++j) {
            flatGraph[i * nNodes + j] = graph[i][j];
        }
    }

    // Allocate memory on GPU
    int* d_graph;
    cudaMalloc((void**)&d_graph, nNodes * nNodes * sizeof(int));

    // Copy graph data to GPU
    cudaMemcpy(d_graph, flatGraph.data(), nNodes * nNodes * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid size
    dim3 blockSize(16, 16);  // Each block has 16x16 threads
    dim3 gridSize((nNodes + blockSize.x - 1) / blockSize.x, (nNodes + blockSize.y - 1) / blockSize.y);

    // Launch kernel on the GPU
    sparsifyKernel<<<gridSize, blockSize>>>(d_graph, nNodes, threshold);
    cudaDeviceSynchronize();
    // Copy the result back to CPU
    cudaMemcpy(flatGraph.data(), d_graph, nNodes * nNodes * sizeof(int), cudaMemcpyDeviceToHost);

    // Reshape flatGraph back to 2D
    for (int i = 0; i < nNodes; ++i) {
        for (int j = 0; j < nNodes; ++j) {
            graph[i][j] = flatGraph[i * nNodes + j];
        }
    }

    // Free GPU memory
    cudaFree(d_graph);
    return graph;
}

vector<vector<int>> sparsify(const vector<vector<int>>& graph, int threshold) {
    vector<vector<int>> sparseGraph = graph;
    for (auto& row : sparseGraph) {
        for (auto& cell : row) {
            if (cell < threshold) {
                cell = 0;
            }
        }
    }
    return sparseGraph;
}

void generateLabels(int nNodes, double percentage, vector<pair<int, int>>& labeledNodes, vector<pair<int, int>>& unlabeledNodes) {
    int numLabeled = static_cast<int>(nNodes * (percentage / 100.0));
    set<int> labeledSet;
    srand(time(0));

    // Generate labeled nodes
    while (labeledSet.size() < numLabeled) {
        int node = rand() % nNodes;
        if (labeledSet.find(node) == labeledSet.end()) {
            int label = rand() % 2;
            labeledNodes.push_back({node, label});
            labeledSet.insert(node);
        }
    }

    // Generate unlabeled nodes
    for (int i = 0; i < nNodes; ++i) {
        if (labeledSet.find(i) == labeledSet.end()) {
            unlabeledNodes.push_back({i, -1}); // Using -1 to indicate null
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

vector<vector<int>> unlabeledSubSparseGraph(const vector<vector<int>>& sparseGraph, const vector<pair<int, int>>& unlabeledNodes) {
    vector<vector<int>> subSparseGraph;
    set<int> unlabeledSet;
    for (const auto& node : unlabeledNodes) {
        unlabeledSet.insert(node.first);
    }

    for (int i = 0; i < sparseGraph.size(); ++i) {
        if (unlabeledSet.find(i) != unlabeledSet.end()) {
            vector<int> newRow;
            for (int j = 0; j < sparseGraph.size(); ++j) {
                if (unlabeledSet.find(j) != unlabeledSet.end()) {
                    newRow.push_back(sparseGraph[i][j]);
                }
            }
            subSparseGraph.push_back(newRow);
        }
    }
    return subSparseGraph;
}

void dfs(int node, const vector<vector<int>>& graph, vector<bool>& visited, vector<int>& component) {
    visited[node] = true;
    component.push_back(node);
    for (int i = 0; i < graph[node].size(); ++i) {
        if (graph[node][i] != 0 && !visited[i]) {
            dfs(i, graph, visited, component);
        }
    }
}

vector<vector<int>> findConnectedComponent(const vector<vector<int>>& graph) {
    vector<vector<int>> components;
    vector<bool> visited(graph.size(), false);

    for (int i = 0; i < graph.size(); ++i) {
        if (!visited[i]) {
            vector<int> component;
            dfs(i, graph, visited, component);
            components.push_back(component);
        }
    }

    return components;
}

vector<vector<int>> upperTriangularWithDiagonal(const vector<vector<int>>& matrix) {
    vector<vector<int>> upperMatrix = matrix;
    for (int i = 0; i < matrix.size(); ++i) {
        for (int j = 0; j < i; ++j) {
            upperMatrix[i][j] = 0;
        }
    }
    return upperMatrix;
}

void labelToClass(const vector<pair<int, int>>& readLabeledNodes, vector<int>& class0, vector<int>& class1) {
    for (const auto& node : readLabeledNodes) {
        if (node.second == 0) {
            class0.push_back(node.first);
        } else if (node.second == 1) {
            class1.push_back(node.first);
        }
    }
}

class ComponentProperties {
public:
    vector<int> nodes;
    int index;
    int class0Sum;
    int class1Sum;
    int total;
    float label;

    ComponentProperties() : index(0), class0Sum(0), class1Sum(0), total(0), label(0.5) {}
};

void parallelEdgeComponent(const vector<vector<int>>& readUpperSubSparseGraph, const vector<vector<int>>& connectedComponents, const vector<int>& class0, const vector<int>& class1, vector<ComponentProperties>& componentsProperties) {
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
            for (const auto& target : class0) {
                cp.class0Sum += readUpperSubSparseGraph[node][target];
            }
            for (const auto& target : class1) {
                cp.class1Sum += readUpperSubSparseGraph[node][target];
            }
        }
        cp.total = cp.class0Sum + cp.class1Sum;
        componentsProperties.push_back(cp);
    }
}

__global__ void computeEdgeWeightSum(const float* adjMatrix, float* vertexWeights, int nRows, int nCols) {
    // Global row index
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < nRows) {
        float sum = 0.0f;
        for (int col = 0; col < nCols; ++col) {
            sum += adjMatrix[row * nCols + col];
        }
        vertexWeights[row] = sum;
    }
}

vector<float> computeEdgeWeightSumCUDA(const vector<vector<int>>& graph) {
    int nRows = graph.size();
    int nCols = graph[0].size(); // Assuming all rows have the same number of columns

    // Flatten the graph
    vector<float> flatGraph(nRows * nCols);
    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            flatGraph[i * nCols + j] = static_cast<float>(graph[i][j]);
        }
    }

    // Allocate memory on the GPU
    float* d_graph;
    float* d_vertexWeights;
    cudaMalloc((void**)&d_graph, nRows * nCols * sizeof(float));
    cudaMalloc((void**)&d_vertexWeights, nRows * sizeof(float));

    // Copy the graph to the GPU
    cudaMemcpy(d_graph, flatGraph.data(), nRows * nCols * sizeof(float), cudaMemcpyHostToDevice);

    // Define the grid and block size
    int blockSize = 256; // Number of threads per block
    int gridSize = (nRows + blockSize - 1) / blockSize; // One thread per row

    // Launch the kernel
    computeEdgeWeightSum<<<gridSize, blockSize>>>(d_graph, d_vertexWeights, nRows, nCols);
    cudaDeviceSynchronize();

    // Copy the results back to the CPU
    vector<float> vertexWeights(nRows);
    cudaMemcpy(vertexWeights.data(), d_vertexWeights, nRows * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(d_graph);
    cudaFree(d_vertexWeights);

    return vertexWeights;
}


__global__ void filterSubgraphKernel(
    const int* graph,        // Original adjacency matrix (flattened)
    int* subgraph,           // Output subgraph (flattened)
    const int* unlabeledNodes, // Indices of unlabeled nodes
    const int* classNodes,    // Indices of class nodes (0 or 1)
    int nNodes,              // Total number of nodes
    int nUnlabeled,          // Number of unlabeled nodes
    int nClass               // Number of class nodes
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Index in unlabeledNodes
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Index in classNodes

    if (row < nUnlabeled && col < nClass) {
        int origRow = unlabeledNodes[row];
        int origCol = classNodes[col];
        subgraph[row * nClass + col] = graph[origRow * nNodes + origCol];
    }
}

vector<vector<int>> filterSubgraphCUDA(
    const vector<vector<int>>& graph, 
    const vector<int>& unlabeledNodes, 
    const vector<int>& classNodes
) {
    int nNodes = graph.size();
    int nUnlabeled = unlabeledNodes.size();
    int nClass = classNodes.size();

    // Flatten the graph
    vector<int> flatGraph(nNodes * nNodes);
    for (int i = 0; i < nNodes; ++i) {
        for (int j = 0; j < nNodes; ++j) {
            flatGraph[i * nNodes + j] = graph[i][j];
        }
    }

    // Allocate device memory
    int* d_graph;
    int* d_unlabeledNodes;
    int* d_classNodes;
    int* d_subgraph;

    cudaMalloc((void**)&d_graph, nNodes * nNodes * sizeof(int));
    cudaMalloc((void**)&d_unlabeledNodes, nUnlabeled * sizeof(int));
    cudaMalloc((void**)&d_classNodes, nClass * sizeof(int));
    cudaMalloc((void**)&d_subgraph, nUnlabeled * nClass * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_graph, flatGraph.data(), nNodes * nNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_unlabeledNodes, unlabeledNodes.data(), nUnlabeled * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_classNodes, classNodes.data(), nClass * sizeof(int), cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((nClass + blockSize.x - 1) / blockSize.x, 
                  (nUnlabeled + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    filterSubgraphKernel<<<gridSize, blockSize>>>(
        d_graph, d_subgraph, d_unlabeledNodes, d_classNodes, nNodes, nUnlabeled, nClass);
    cudaDeviceSynchronize();
    // Copy result back to host
    vector<int> flatSubgraph(nUnlabeled * nClass);
    cudaMemcpy(flatSubgraph.data(), d_subgraph, nUnlabeled * nClass * sizeof(int), cudaMemcpyDeviceToHost);

    // Convert flat subgraph to 2D matrix
    vector<vector<int>> subgraph(nUnlabeled, vector<int>(nClass));
    for (int i = 0; i < nUnlabeled; ++i) {
        for (int j = 0; j < nClass; ++j) {
            subgraph[i][j] = flatSubgraph[i * nClass + j];
        }
    }

    // Free device memory
    cudaFree(d_graph);
    cudaFree(d_unlabeledNodes);
    cudaFree(d_classNodes);
    cudaFree(d_subgraph);

    return subgraph;
}

vector<int> getNodeIndices(const vector<pair<int, int>>& nodes, int labelFilter) {
    vector<int> indices;
    for (const auto& node : nodes) {
        if (node.second == labelFilter) {
            indices.push_back(node.first);
        }
    }
    return indices;
}

__global__ void computeAverageEdgeWeights(const float* subgraph0, const float* subgraph1, float* avgSubgraph, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {float denominator = subgraph0[idx] + subgraph1[idx];
        if (denominator == 0.0f) {
            avgSubgraph[idx] = 0.5f; // Set to 0.5 if denominator is zero
        } else {
            avgSubgraph[idx] = 0.5f - (subgraph0[idx] / denominator) / 2.0f + (subgraph1[idx] / denominator) / 2.0f;
        }
    }
}

void computeAverageEdgeWeightsCUDA(
    const vector<float>& edgeWeightSumsSubgraph0,
    const vector<float>& edgeWeightSumsSubgraph1,
    vector<float>& edgeWeightSumsSubgraphAvg) {

    int size = edgeWeightSumsSubgraph0.size();

    // Allocate memory on GPU
    float *d_edgeWeightSumsSubgraph0, *d_edgeWeightSumsSubgraph1, *d_avgSubgraph;
    cudaMalloc((void**)&d_edgeWeightSumsSubgraph0, size * sizeof(float));
    cudaMalloc((void**)&d_edgeWeightSumsSubgraph1, size * sizeof(float));
    cudaMalloc((void**)&d_avgSubgraph, size * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_edgeWeightSumsSubgraph0, edgeWeightSumsSubgraph0.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeWeightSumsSubgraph1, edgeWeightSumsSubgraph1.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid size
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Launch kernel
    computeAverageEdgeWeights<<<gridSize, blockSize>>>(
        d_edgeWeightSumsSubgraph0, d_edgeWeightSumsSubgraph1, d_avgSubgraph, size);
    cudaDeviceSynchronize();

    // Copy result back to host
    edgeWeightSumsSubgraphAvg.resize(size);
    cudaMemcpy(edgeWeightSumsSubgraphAvg.data(), d_avgSubgraph, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_edgeWeightSumsSubgraph0);
    cudaFree(d_edgeWeightSumsSubgraph1);
    cudaFree(d_avgSubgraph);
}



int main() {
    int nNodes = 15;
    int sRand = 1;
    int eRand = 10;
    int threshold = 8;
    double percentage = 40.0; 
    
    vector<vector<int>> graph = generateGraph(nNodes, sRand, eRand);
    saveGraphToFile(graph, "graph.txt");

    vector<vector<int>> readGraph = readGraphFromFile("graph.txt");
    cout<<"Original Graph"<<endl;
    printGraph(readGraph);

    cout<<endl;
    cout<<"Sparse Graph"<<endl;
    vector<vector<int>> sparseGraph = sparsifyCUDA(readGraph, threshold);
    saveGraphToFile(sparseGraph, "sparseGraph.txt");
    printGraph(sparseGraph);

    
    vector<pair<int, int>> labeledNodes;
    vector<pair<int, int>> unlabeledNodes;
    
    generateLabels(nNodes, percentage, labeledNodes, unlabeledNodes);
    saveLabelsToFile(labeledNodes, "labeledNodes.txt");
    saveLabelsToFile(unlabeledNodes, "unlabeledNodes.txt");

    


    vector<pair<int, int>> readLabeledNodes = readLabelsFromFile("labeledNodes.txt");
    vector<pair<int, int>> readUnlabeledNodes = readLabelsFromFile("unlabeledNodes.txt");

    
    cout<<endl;
    printLabels(readLabeledNodes, "Labeled"); 
    cout<<endl;
    printLabels(readUnlabeledNodes, "Unlabeled");

    vector<vector<int>> subSparseGraph = unlabeledSubSparseGraph(sparseGraph, readUnlabeledNodes);
    saveGraphToFile(subSparseGraph, "unlabeledSubSparseGraph.txt");

    vector<vector<int>> readSubSparseGraph = readGraphFromFile("unlabeledSubSparseGraph.txt");
    printGraph(readSubSparseGraph);

    vector<vector<int>> upperSubSparseGraph = upperTriangularWithDiagonal(readSubSparseGraph);
    saveGraphToFile(upperSubSparseGraph, "upperSubSparseGraph.txt");

    vector<vector<int>> readUpperSubSparseGraph = readGraphFromFile("upperSubSparseGraph.txt");
    cout << "Upper Triangular Sub Sparse Graph" << endl;
    printGraph(readUpperSubSparseGraph);

    std::string inputFile = "sparseGraph.txt";
    std::string outputFile = "connected_components.txt";

    // Construct the Python command
    std::string command = "python3 test3.py " + inputFile + " " + outputFile;

    // Call the Python script
    int result = system(command.c_str());

    if (result != 0) {
        std::cerr << "Error: Python script execution failed with code " << result << std::endl;
        return 1;
    }

    // Read the output file and print the connected components
    std::ifstream inFile(outputFile);
    if (inFile.is_open()) {
        std::string line;
        std::cout << "\nConnected Components (from Python):\n";
        while (std::getline(inFile, line)) {
            std::cout << line << std::endl;
        }
        inFile.close();
    } else {
        std::cerr << "Unable to open connected_components.txt" << std::endl;
    }

    vector<float> edgeWeightSums = computeEdgeWeightSumCUDA(sparseGraph);
    cout << "\nEdge Weight Sums for Each Vertex (CUDA):" << endl;
    for (int i = 0; i < edgeWeightSums.size(); ++i) {
        cout << "Vertex " << i << ": " << edgeWeightSums[i] << endl;
    }

   
    // Get node indices for unlabeled and class nodes
    vector<int> unlabeledIndices = getNodeIndices(readUnlabeledNodes, -1);
    vector<int> class0Indices = getNodeIndices(readLabeledNodes, 0);
    vector<int> class1Indices = getNodeIndices(readLabeledNodes, 1);

    // Create subgraph0 (Unlabeled + Class 0)
    vector<vector<int>> subgraph0 = filterSubgraphCUDA(sparseGraph, unlabeledIndices, class0Indices);
    cout << "\nSubgraph (Unlabeled + Label 0):\n";
    for (const auto& row : subgraph0) {
        for (const auto& val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    // Compute edge weight sums for subgraph0
    vector<float> edgeWeightSumsSubgraph0 = computeEdgeWeightSumCUDA(subgraph0);
    cout << "\nEdge Weight Sums for Subgraph0 (Unlabeled + Label 0):" << endl;
    for (int i = 0; i < edgeWeightSumsSubgraph0.size(); ++i) {
        cout << "Vertex " << i << ": " << edgeWeightSumsSubgraph0[i] << endl;
    }

    // Create subgraph1 (Unlabeled + Label 1)
    vector<vector<int>> subgraph1 = filterSubgraphCUDA(sparseGraph, unlabeledIndices, class1Indices);
    cout << "\nSubgraph (Unlabeled + Label 1):\n";
    for (const auto& row : subgraph1) {
        for (const auto& val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    // Compute edge weight sums for subgraph1
    vector<float> edgeWeightSumsSubgraph1 = computeEdgeWeightSumCUDA(subgraph1);
    cout << "\nEdge Weight Sums for Subgraph1 (Unlabeled + Label 1):" << endl;
    for (int i = 0; i < edgeWeightSumsSubgraph1.size(); ++i) {
        cout << "Vertex " << i << ": " << edgeWeightSumsSubgraph1[i] << endl;
    }

    vector<float> edgeWeightSumsSubgraphAvg;

    // Call the new function
    computeAverageEdgeWeightsCUDA(edgeWeightSumsSubgraph0, edgeWeightSumsSubgraph1, edgeWeightSumsSubgraphAvg);

    // Print the results
    cout << "\nAverage Edge Weight Sums (Subgraph0 and Subgraph1):" << endl;
    for (int i = 0; i < edgeWeightSumsSubgraphAvg.size(); ++i) {
        cout << "Vertex " << i << ": " << edgeWeightSumsSubgraphAvg[i] << endl;
    }


    std::string pythonCommand = "python3 rowSum.py sparseGraph.txt row_sums.txt";
    int pyResult = system(pythonCommand.c_str());

    if (pyResult != 0) {
        cerr << "Error: Python script execution failed with code " << pyResult << endl;
        return 1;
    }

    // Read the row sums from the file generated by the Python script
    ifstream rowSumsFile("row_sums.txt");
    vector<float> rowSums;
    if (rowSumsFile.is_open()) {
        float rowSum;
        while (rowSumsFile >> rowSum) {
            rowSums.push_back(rowSum);
        }
        rowSumsFile.close();
    } else {
        cerr << "Unable to open row_sums.txt" << endl;
        return 1;
    }

    // Print the row sums
    cout << "\nRow Sums (from Python/CuPy):" << endl;
    for (int i = 0; i < rowSums.size(); ++i) {
        cout << "Node " << i << ": " << rowSums[i] << endl;
    }


    return 0;
    

}
