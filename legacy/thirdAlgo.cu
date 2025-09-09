#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <set>
#include <map>

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

    vector<vector<int>> connectedComponents = findConnectedComponent(readUpperSubSparseGraph);
    cout << "Number of connected components: " << connectedComponents.size() << "\n";
    for ( auto& component : connectedComponents) {
        cout << "Component: ";
        for ( auto& node : component) {
            // cout << readUnlabeledNodes[node].first << " ";
            node = readUnlabeledNodes[node].first;
            cout << node << " ";
        }
        cout << "\n";
    }

    vector<int> class0, class1;
    labelToClass(readLabeledNodes, class0, class1);
    cout << "Class 0 Nodes: ";
    for (const auto& node : class0) {
        cout << node << " ";
    }
    cout << "\n";

    cout << "Class 1 Nodes: ";
    for (const auto& node : class1) {
        cout << node << " ";
    }
    cout << "\n";

    cout << "Original Graph" <<"\n";
    printGraph(upperTriangularWithDiagonal(readGraph));

    vector<ComponentProperties> componentsProperties;
    vector<pair<int, int>> resultUnlabeledNodes(nNodes);
    parallelEdgeComponent(upperTriangularWithDiagonal(readGraph), connectedComponents, class0, class1, componentsProperties);
    int resIdx = 0;
    for (auto& cp : componentsProperties) {
        //cout << "Component " << cp.index << ": Nodes = ";
        for (auto& node : cp.nodes) {
            cout << node << " " << ((cp.label + (0 - cp.label)*1.0*cp.class0Sum/cp.total/2 + (1 - cp.label)*1.0*cp.class1Sum/cp.total/2) >= 0.5)? 1 : 0 ;
            cout << "\n";
            
            resultUnlabeledNodes[resIdx] = make_pair(node, (cp.label + (0 - cp.label)*1.0*cp.class0Sum/cp.total/2 + (1 - cp.label)*1.0*cp.class1Sum/cp.total/2 >= 0.5)? 1 : 0);

            resIdx++;
        }
        // cout << ", Class 0 Sum = " << cp.class0Sum;
        // cout << ", Class 1 Sum = " << cp.class1Sum;
        // cout << ", Total = " << cp.total;
        // cp.label = cp.label + (0 - cp.label)*1.0*cp.class0Sum/cp.total/2 + (1 - cp.label)*1.0*cp.class1Sum/cp.total/2;
        // cout << ", Label = " << (cp.label >= 0.5)? 1 : 0;
        // cout << "\n";
    }



}
