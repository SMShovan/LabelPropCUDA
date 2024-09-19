#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <climits>
#include <tuple>
#include <sstream>

using namespace std;

// Function to read the graph from "graph.txt"
vector<tuple<int, int, int, int>> readGraphFromFile(const string& filename) {
    vector<tuple<int, int, int, int>> edgeList;
    ifstream inFile(filename);
    if (inFile.is_open()) {
        string line;
        while (getline(inFile, line)) {
            istringstream iss(line);
            int u, v, obj1, obj2;
            if (iss >> u >> v >> obj1 >> obj2) {
                edgeList.push_back({u, v, obj1, obj2});
            }
        }
        inFile.close();
    } else {
        cerr << "Unable to open file " << filename << endl;
    }
    return edgeList;
}

// Function to read the tree from "tree.txt"
vector<vector<int>> readTreeFromFile(const string& filename, int& nNodes) {
    vector<vector<int>> tree;
    ifstream inFile(filename);
    if (inFile.is_open()) {
        string line;
        while (getline(inFile, line)) {
            istringstream iss(line);
            vector<int> neighbors;
            int node;
            while (iss >> node) {
                neighbors.push_back(node);
            }
            tree.push_back(neighbors);
        }
        nNodes = tree.size(); // Set the number of nodes based on the number of lines
        inFile.close();
    } else {
        cerr << "Unable to open file " << filename << endl;
    }
    return tree;
}

// Function to store the bi-objective distances from node 0 to all nodes
void saveDistancesToFile(const vector<pair<int, int>>& distances, const string& filename) {
    ofstream outFile(filename);
    if (outFile.is_open()) {
        for (int i = 0; i < distances.size(); ++i) {
            outFile << "" << i << " " << distances[i].first << " " << distances[i].second << "\n";
        }
        outFile.close();
    } else {
        cerr << "Unable to open file " << filename << endl;
    }
}


int main() {
    // Read the graph and tree from the respective files
    vector<tuple<int, int, int, int>> edgeList = readGraphFromFile("undirected_graph.txt");
    int nNodes = 0;
    vector<vector<int>> tree = readTreeFromFile("combinedGraphTree.txt", nNodes);

    // Distance vector initialized with large values for both objectives
    vector<pair<int, int>> distances(nNodes, {INT_MAX, INT_MAX});
    
    // Priority queue for Dijkstra-like algorithm: {current objective 1, current objective 2, node}
    priority_queue<tuple<int, int, int>, vector<tuple<int, int, int>>, greater<tuple<int, int, int>>> pq;
    
    // Start from node 0, set its distance to {0, 0}
    distances[0] = {0, 0};
    pq.push({0, 0, 0});
    
    // Run modified Dijkstra's algorithm
    while (!pq.empty()) {
        auto [dist1, dist2, node] = pq.top();
        pq.pop();
        
        // Traverse all edges connected to the current node
        for (const auto& [u, v, obj1, obj2] : edgeList) {
            if (u == node || v == node) {
                int neighbor = (u == node) ? v : u;  // Find the other end of the edge
                int newDist1 = dist1 + obj1;
                int newDist2 = dist2 + obj2;
                
                // If the new distance is better, update and push to the queue
                if (newDist1 < distances[neighbor].first || newDist2 < distances[neighbor].second) {
                    distances[neighbor] = {newDist1, newDist2};
                    pq.push({newDist1, newDist2, neighbor});
                }
            }
        }
    }
    
    // Save distances to a file
    saveDistancesToFile(distances, "bi_objective_distances.txt");
    
    cout << "Bi-objective distances from node 0 have been saved to bi_objective_distances.txt" << endl;

    vector<pair<int, int>> distances1 = readDistancesFromFile("tree.txt");
    vector<pair<int, int>> distances2 = readDistancesFromFile("bi_objective_distances.txt");
    
    // Check that both files have the same number of nodes
    if (distances1.size() != distances2.size()) {
        cerr << "Error: Files contain a different number of nodes." << endl;
        return 1;
    }

    // Compute the difference of distances item-wise
    vector<pair<int, int>> diffDistances(distances1.size());
    for (int i = 0; i < distances1.size(); ++i) {
        diffDistances[i].first = distances2[i].first - distances1[i].first;
        diffDistances[i].second = distances2[i].second - distances1[i].second;
    }

    // Save the result to "diff_bi_objective_distances.txt"
    saveDiffToFile(diffDistances, "diff_bi_objective_distances.txt");
    
    cout << "Difference of bi-objective distances has been saved to diff_bi_objective_distances.txt" << endl;


    return 0;
}
