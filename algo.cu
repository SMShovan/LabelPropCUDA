#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <set>

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


int main() {
    int nNodes = 15;
    int sRand = 1;
    int eRand = 10;
    int threshold = 5;
    double percentage = 40.0; 
    
    vector<vector<int>> graph = generateGraph(nNodes, sRand, eRand);
    saveGraphToFile(graph, "graph.txt");

    vector<vector<int>> readGraph = readGraphFromFile("graph.txt");
    cout<<"Original Graph"<<endl;
    printGraph(readGraph);

    cout<<endl;
    cout<<"Sparse Graph"<<endl;
    vector<vector<int>> sparseGraph = sparsify(readGraph, threshold);
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

    vector<vector<int>> connectedComponents = findConnectedComponent(readSubSparseGraph);
    cout << "Number of connected components: " << connectedComponents.size() << "\n";
    for (const auto& component : connectedComponents) {
        cout << "Component: ";
        for (const auto& node : component) {
            cout << node << " ";
        }
        cout << "\n";
    }

    return 0;
}
