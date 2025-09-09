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

    // cout<<endl;
    // cout<<"Sparse Graph"<<endl;
    // vector<vector<int>> sparseGraph = sparsify(readGraph, threshold);
    // saveGraphToFile(sparseGraph, "sparseGraph.txt");
    // printGraph(sparseGraph);

    
    vector<pair<int, int>> labeledNodes;
    vector<pair<int, int>> unlabeledNodes;
    generateLabels(nNodes, percentage, labeledNodes, unlabeledNodes);
    saveLabelsToFile(labeledNodes, "labeledNodes.txt");
    saveLabelsToFile(unlabeledNodes, "unlabeledNodes.txt");

    


    vector<pair<int, int>> readLabeledNodes = readLabelsFromFile("labeledNodes.txt");
    vector<pair<int, int>> readUnlabeledNodes = readLabelsFromFile("unlabeledNodes.txt");

    
    // cout<<endl;
    // printLabels(readLabeledNodes, "Labeled"); 
    // cout<<endl;
    // printLabels(readUnlabeledNodes, "Unlabeled");

    // vector<vector<int>> subSparseGraph = unlabeledSubSparseGraph(sparseGraph, readUnlabeledNodes);
    // saveGraphToFile(subSparseGraph, "unlabeledSubSparseGraph.txt");

    // vector<vector<int>> readSubSparseGraph = readGraphFromFile("unlabeledSubSparseGraph.txt");
    // printGraph(readSubSparseGraph);

    // vector<vector<int>> upperSubSparseGraph = upperTriangularWithDiagonal(readSubSparseGraph);
    // saveGraphToFile(upperSubSparseGraph, "upperSubSparseGraph.txt");

    // vector<vector<int>> readUpperSubSparseGraph = readGraphFromFile("upperSubSparseGraph.txt");
    // cout << "Upper Triangular Sub Sparse Graph" << endl;
    // printGraph(readUpperSubSparseGraph);

    // vector<vector<int>> connectedComponents = findConnectedComponent(readUpperSubSparseGraph);
    // cout << "Number of connected components: " << connectedComponents.size() << "\n";
    // for ( auto& component : connectedComponents) {
    //     cout << "Component: ";
    //     for ( auto& node : component) {
    //         // cout << readUnlabeledNodes[node].first << " ";
    //         node = readUnlabeledNodes[node].first;
    //         cout << node << " ";
    //     }
    //     cout << "\n";
    // }

    vector<int> class0, class1, classU;
    labelToClass(readLabeledNodes, class0, class1);
    labelToClass(readUnlabeledNodes, classU);

    vector<int> c0(nNodes, 0);
    vector<int> c1(nNodes, 0);
    vector<int> cA(nNodes, 0);

    cout << "Class 0 Nodes: ";
    for (const auto& node : class0) {
        c0[node] = 1;  
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
    }
    for(int i = 0; i < nNodes; i++)
    {
        cout << cA[i] << " ";
    }
    cout << endl;
    cout << "\n";





    // cout << "Original Graph" <<"\n";
    // printGraph(upperTriangularWithDiagonal(readGraph));

    // vector<unlabelledNodeProperties> unlabellednodeProperties;
    // parallelEdgeComponent(upperTriangularWithDiagonal(readGraph), connectedComponents, class0, class1, unlabellednodeProperties);
    // for (const auto& cp : unlabellednodeProperties) {
    //     cout << "Component " << cp.index << ": Nodes = ";
    //     for (const auto& node : cp.nodes) {
    //         cout << node << " ";
    //     }
    //     cout << ", Class 0 Sum = " << cp.class0Sum;
    //     cout << ", Class 1 Sum = " << cp.class1Sum;
    //     cout << ", Total = " << cp.total;
    //     cout << "\n";
    // }

    // vector<int> readUnlabeledNodesFirst;
    // for (const auto& node : readUnlabeledNodes) {
    //     readUnlabeledNodesFirst.push_back(node.first);
    // }

    // vector<unlabelledNodeProperties> unlabellednodeProperties;

    // // Call the updated function
    // parallelEdgeUnlabeled(upperTriangularWithDiagonal(readGraph), readUnlabeledNodesFirst, class0, class1, classU, unlabellednodeProperties);
    // vector<pair<int, int>> resultUnlabeledNodes(nNodes);
    // int idx = 0; 
    // // Print the results from main
    // for (int i = 0; i < 10; i++)
    //     for ( auto& cp : unlabellednodeProperties) {
            
    //         if (cp.total == 0)
    //             continue;

    //         float unContribute = 0;
    //         for (int u = 0; u < classU.size(); u++)
    //         {
    //             for ( auto& un : unlabellednodeProperties)
    //             {
    //                 if (un.node == classU[u])
    //                 {
    //                     unContribute+= (un.curLabel - cp.curLabel) * 1.0 * upperTriangularWithDiagonal(readGraph)[cp.node][un.node]/cp.total; 
    //                 }
    //             }
    //         }

    //         cp.curLabel = (cp.curLabel + (0 - cp.curLabel)*1.0*cp.class0Sum/cp.total/2 + (1 - cp.curLabel)*1.0*cp.class1Sum/cp.total/2 + unContribute/2);
            
            
            
    //         if (i == 9)
    //         {
    //             cout << "Node " << cp.node << ": Class 0 Sum = " << cp.class0Sum 
    //             << ", Class 1 Sum = " << cp.class1Sum 
    //             << ", Total = " << cp.total 
    //             << ", Label = " << ((cp.curLabel >= 0.5)? 1 : 0)
    //             << "\n";
    //             resultUnlabeledNodes[idx] = make_pair(cp.node, (cp.curLabel >= 0.5)? 1 : 0);
    //             idx++;
    //         }
            
    //     }
        
    




    return 0;
}
