#include <vector>
#include <iostream>

using namespace std;

// DFS function to mark reachable nodes
void dfs(int node, const vector<vector<int>>& graph, vector<bool>& visited, vector<int>& reachableNodes) {
    visited[node] = true; // Mark the current node as visited
    reachableNodes.push_back(node); // Add this node to the list of reachable nodes

    // Visit all neighbors of the current node
    for (size_t i = 0; i < graph[node].size(); ++i) {
        if (graph[node][i] != 0 && !visited[i]) { // If there's an edge and the neighbor isn't visited
            dfs(i, graph, visited, reachableNodes); // Recursively visit the neighbor
        }
    }
}

// Function to compute the reachability matrix
vector<vector<int>> computeReachabilityMatrix(const vector<vector<int>>& graph) {
    int n = graph.size(); // Number of nodes in the graph
    vector<vector<int>> reachable(n, vector<int>(n, 0)); // Initialize the reachability matrix with 0s

    // For each node, perform DFS and mark reachable nodes
    for (int i = 0; i < n; ++i) {
        vector<bool> visited(n, false); // To keep track of visited nodes
        vector<int> reachableNodes;     // List to store reachable nodes from the current node

        // Perform DFS starting from node `i`
        dfs(i, graph, visited, reachableNodes);

        // Mark reachable nodes in the matrix
        for (const auto& node : reachableNodes) {
            reachable[i][node] = 1; // Set reachable[i][node] to 1
        }
    }

    return reachable; // Return the reachability matrix
}

// Function to remove redundancy from the reachability matrix
void removeRedundancy(vector<vector<int>>& reachabilityMatrix) {
    int n = reachabilityMatrix.size(); // Number of nodes

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = n; j > 0; --j) {
            if (reachabilityMatrix[i][j - 1] != 0)
                reachabilityMatrix[i][j] = 0;
        }
    vector<int> cumCount(n);
    #pragma omp parallel for
    for(int j = 0; j < n; j++)
    {
        cumCount[j] = 0;
        for(int i = 0; i < n; i++)
        {
            if (reachabilityMatrix[i][j] != 0)
                cumCount[j]++;
        }
    }

    for(int j = 0; j < n; j++)
    {   
        if (j != 0)
            cumCount[j] = cumCount[j] + cumCount[j - 1];
        cout<<cumCount[j]<< " ";
    }
    cout<<" Cummulative Sum"<<endl;
    vector<int> partialResult(n);
    int startIdx;
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        if (j != 0 && cumCount[j] - cumCount[j - 1] == 0)
            continue;
        
        if (j == 0)
            startIdx = 0;
        else 
            startIdx = cumCount[j - 1];

        for (int i = 0; i < n; i++)
        {
            if (reachabilityMatrix[i][j] != 0)
                partialResult[startIdx++] = i;
        }
    }

    for (int i = 0; i < n; i++)
        cout<<partialResult[i]<<" ";
    cout<<" Flattened result"<<endl;
}

int main() {
    // Updated adjacency matrix (graph)
    vector<vector<int>> graph = {
        {0, 1, 0, 0, 0}, // Node 0 connects to node 1
        {1, 0, 1, 0, 0}, // Node 1 connects to node 0 and node 2
        {0, 1, 0, 0, 0}, // Node 2 connects to node 1
        {0, 0, 0, 0, 1}, // Node 3 connects to node 4
        {0, 0, 0, 1, 0}  // Node 4 connects to node 3
    };

    // Get the reachability matrix
    vector<vector<int>> reachabilityMatrix = computeReachabilityMatrix(graph);

    // Print the original reachability matrix
    cout << "Original Reachability Matrix:" << endl;
    for (const auto& row : reachabilityMatrix) {
        for (const auto& cell : row) {
            cout << cell << " ";
        }
        cout << endl;
    }

    // Remove redundancy
    removeRedundancy(reachabilityMatrix);

    // Print the updated reachability matrix
    cout << "Updated Reachability Matrix (after removing redundancy):" << endl;
    for (const auto& row : reachabilityMatrix) {
        for (const auto& cell : row) {
            cout << cell << " ";
        }
        cout << endl;
    }

    return 0;
}
