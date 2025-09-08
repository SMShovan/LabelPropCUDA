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
    vector<vector<int>> adjacencyMatrix(nNodes, vector<int>(nNodes, 0));
    srand(time(0));
    
    for (int i = 0; i < nNodes; ++i) {
        for (int j = i + 1; j < nNodes; ++j) {
            int weight = sRand + rand() % (eRand - sRand + 1); // Random weight for edge
            adjacencyMatrix[i][j] = weight; // Set weight for edge (i, j)
            adjacencyMatrix[j][i] = weight; // Mirror weight for edge (j, i)
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

void labelToClass(const vector<pair<int, int>>& readUnlabeledNodes, vector<int>& classU) {
    for (const auto& node : readUnlabeledNodes) {
        if (node.second == -1) {
            classU.push_back(node.first);
        }
    }
}

class unlabelledNodeProperties {
public:
    int node;
    int class0Sum;
    int class1Sum;
    int classUSum;
    int total;
    float prevLabel;
    float curLabel;

    unlabelledNodeProperties() : node(-1), class0Sum(0), class1Sum(0), classUSum(0), prevLabel(-1.0), curLabel(0.5), total(0) {}
};




void parallelEdgeUnlabeled(const vector<vector<int>>& readUpperSubSparseGraph, 
                           const vector<int>& readUnlabeledNodesFirst, 
                           const vector<int>& class0, 
                           const vector<int>& class1,
                           const vector<int>& classU, 
                           vector<unlabelledNodeProperties>& unlabellednodeProperties) {
    for (const auto& node : readUnlabeledNodesFirst) {
        unlabelledNodeProperties cp;
        cp.node = node;
        for (const auto& target : class0) {
            cp.class0Sum += readUpperSubSparseGraph[node][target];
        }
        for (const auto& target : class1) {
            cp.class1Sum += readUpperSubSparseGraph[node][target];
        }
        for (const auto& target : classU) {
            cp.classUSum += readUpperSubSparseGraph[node][target];
        }
        cp.total = cp.class0Sum + cp.class1Sum + cp.classUSum;
        unlabellednodeProperties.push_back(cp);
    }
}

vector<int> flattenGraph(const vector<vector<int>>& graph) {
    vector<int> flattened;
    for (const auto& row : graph) {
        for (const auto& cell : row) {
            flattened.push_back(cell);
        }
    }
    return flattened;
}


__global__ void kernelProcessNodes(const int* flattenedGraph, int* c0, int* c1, int* cA,  float* lab, int nNodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < nNodes) {
        // Only process if cA[tid] == 1
        if (cA[tid] == 1) {
            int class0Sum = 0;
            int class1Sum = 0;
            int classUSum = 0;
            int classUCount = 0;

            // Flattened graph indexing: flattenedGraph[i * nNodes + j] = weight between node i and node j
            for (int j = 0; j < nNodes; j++) {
                int weight = flattenedGraph[tid * nNodes + j];
                if (c0[j] == 1) {
                    class0Sum += weight;
                }
                if (c1[j] == 1) {
                    class1Sum += weight;
                }
                if (c0[j] == 0 && c1[j] == 0){
                    classUSum += weight;
                    classUCount++;
                }
            }
            int totalSum = class0Sum + class1Sum + classUSum;
            float newVal;
            float oldVal = lab[tid];
            int classU;
            if (totalSum > 0) {
                // Update lab[tid]
                classU = (float)classUSum/classUCount;
                if (classU < 0.5)
                    classUSum = -1 * classUSum;


                newVal = lab[tid] - ((float)class0Sum / (float)totalSum *0.5) + ((float)class1Sum / (float)totalSum * 0.5) + ((float)classUCount / (float) totalSum * 0.5);
                lab[tid] = newVal;
            }
            cA[tid] = 0;
            if (abs(oldVal - newVal) >= 0.1)
            {
                for (int j = 0; j < nNodes; j++) {
                    int weight = flattenedGraph[tid * nNodes + j];
                    if (weight != 0) {
                        // Check if neighbor j is not in class0 or class1
                        if (c0[j] == 0 && c1[j] == 0) {
                            cA[j] = 1;
                        }
                    }
                }
            }
            printf("Thread for node %d (old cA=1): Class0Sum=%d, Class1Sum=%d, Updated lab[%d]=%f, Updated cA[%d]=%d\n",
                   tid, class0Sum, class1Sum, tid, lab[tid], tid, cA[tid]);
        }
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

    vector<int> flattenedGraph = flattenGraph(graph);

    
    vector<pair<int, int>> labeledNodes;
    vector<pair<int, int>> unlabeledNodes;
    generateLabels(nNodes, percentage, labeledNodes, unlabeledNodes);
    saveLabelsToFile(labeledNodes, "labeledNodes.txt");
    saveLabelsToFile(unlabeledNodes, "unlabeledNodes.txt");

    


    vector<pair<int, int>> readLabeledNodes = readLabelsFromFile("labeledNodes.txt");
    vector<pair<int, int>> readUnlabeledNodes = readLabelsFromFile("unlabeledNodes.txt");

    

    vector<int> class0, class1, classU;
    labelToClass(readLabeledNodes, class0, class1);
    labelToClass(readUnlabeledNodes, classU);

    vector<int> c0(nNodes, 0);
    vector<int> c1(nNodes, 0);
    vector<int> cA(nNodes, 0);
    vector<float> lab(nNodes, 0);

    cout << "Class 0 Nodes: ";
    for (const auto& node : class0) {
        c0[node] = 1;  
        lab[node] = 0.0;
    }
    for(int i = 0; i < nNodes; i++)
    {
        cout << c0[i] << " ";
    }
    cout << endl;
    cout << "\n";


    cout << "Class 1 Nodes: ";
    for (const auto& node : class1) {
        c1[node] = 1;
        lab[node] = 1.0;
    }
    for(int i = 0; i < nNodes; i++)
    {
        cout << c1[i] << " ";
    }
    cout << endl;
    cout << "\n";

    cout << "Class Unlabelled Nodes: ";
    for (const auto& node : classU) {
        cA[node] = 1;
        lab[node] = 0.5;
    }
    for(int i = 0; i < nNodes; i++)
    {
        cout << cA[i] << " ";
    }
    cout << endl;
    cout << "\n";
    cout << "Labels: ";
    for(int i = 0; i < nNodes; i++)
    {
        cout << lab[i] << " ";
    }




    int* d_flattenedGraph;
    int* d_c0;
    int* d_c1;
    int* d_cA;
    float* d_lab;
    size_t graphSize = flattenedGraph.size() * sizeof(int);
    size_t nodeSize = nNodes * sizeof(int);
    size_t labSize = nNodes * sizeof(float);

    cudaMalloc(&d_flattenedGraph, graphSize);
    cudaMalloc(&d_c0, nodeSize);
    cudaMalloc(&d_c1, nodeSize);
    cudaMalloc(&d_cA, nodeSize);
    cudaMalloc(&d_lab, labSize);


    // Copy data to the device
    cudaMemcpy(d_flattenedGraph, flattenedGraph.data(), graphSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c0, c0.data(), nodeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c1, c1.data(), nodeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cA, cA.data(), nodeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lab, lab.data(), labSize, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (nNodes + blockSize - 1) / blockSize;
    int iterations = 10;
    for (int it = 0; it < iterations; it++) {
        kernelProcessNodes<<<gridSize, blockSize>>>(d_flattenedGraph, d_c0, d_c1, d_cA, d_lab, nNodes);
        cudaDeviceSynchronize();
    }

    // Synchronize and finish
    cudaDeviceSynchronize();

    cudaMemcpy(lab.data(), d_lab, labSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(cA.data(), d_cA, nodeSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_flattenedGraph);
    cudaFree(d_c0);
    cudaFree(d_c1);
    cudaFree(d_cA);
    cudaFree(d_lab);

    cout << "Updated Labels: ";
    for (int i = 0; i < nNodes; i++) {
        cout << lab[i] << " ";
    }
    cout << endl;

    cout << "Updated cA: ";
    for (int i = 0; i < nNodes; i++) {
        cout << cA[i] << " ";
    }
    cout << endl;


    return 0;
}


