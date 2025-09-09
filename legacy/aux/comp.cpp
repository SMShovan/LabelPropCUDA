#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <set>
#include <climits>
#include <algorithm> 
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <cstdlib>
#include <ctime>
#include <set>
#include <fstream>
#include <queue>
#include <limits>
#include <sstream>


using namespace std;

struct Edge {
    int u, v;
    vector<int> weight; // Weight is now a vector of two random values
};

// Function to generate a random number in the range [minVal, maxVal]
int random(int minVal, int maxVal) {
    return minVal + rand() % (maxVal - minVal + 1);
}

// Union-Find (Disjoint Set) to manage connected components
class DisjointSet {
public:
    DisjointSet(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; ++i)
            parent[i] = i;
    }

    int find(int u) {
        if (u != parent[u])
            parent[u] = find(parent[u]);
        return parent[u];
    }

    void unite(int u, int v) {
        int rootU = find(u);
        int rootV = find(v);
        if (rootU != rootV) {
            if (rank[rootU] > rank[rootV])
                parent[rootV] = rootU;
            else if (rank[rootU] < rank[rootV])
                parent[rootU] = rootV;
            else {
                parent[rootV] = rootU;
                rank[rootU]++;
            }
        }
    }

private:
    vector<int> parent, rank;
};

// Function to generate a directed graph with n nodes and m edges
vector<Edge> generateDirectedGraph(int n, int m, int r1, int r2) {
    vector<Edge> edges;
    DisjointSet ds(n);
    srand(time(0));
    
    // Step 1: Create a directed tree (n-1 edges to ensure the graph is connected)
    set<pair<int, int>> usedEdges; // Track used edges to avoid duplicates
    for (int i = 1; i < n; ++i) {
        int u = i;
        int v = random(0, i - 1); // Ensure v is a previous node to form a tree
        vector<int> weight = {random(r1, r2), random(r1, r2)};
        edges.push_back({u, v, weight}); // Add a directed edge from u to v
        ds.unite(u, v);
        usedEdges.insert({u, v}); // Track the directed edge (u -> v)
    }

    // Step 2: Add remaining (m - (n - 1)) random directed edges
    while (edges.size() < m) {
        int u = random(0, n - 1);
        int v = random(0, n - 1);
        if (u != v && usedEdges.find({u, v}) == usedEdges.end()) {
            vector<int> weight = {random(r1, r2), random(r1, r2)};
            edges.push_back({u, v, weight}); // Add a directed edge from u to v
            usedEdges.insert({u, v});
        }
    }

    return edges;
}



// Function to print the graph
void printGraph(const vector<Edge>& edges) {
    cout << "Graph edges (u <-> v, weights):" << endl;
    for (const auto& edge : edges) {
        cout << "(" << edge.u << " <-> " << edge.v << ", [" << edge.weight[0] << ", " << edge.weight[1] << "])" << endl;
    }
}

// Function to save the undirected graph in edge list format to a file
void saveGraphToFile(const vector<Edge>& edges, const string& filename) {
    ofstream outFile(filename);
    if (outFile.is_open()) {
        // Format: u v weight1 weight2
        // Each line represents an undirected edge (u <-> v)
        for (const auto& edge : edges) {
            outFile << edge.u << " " << edge.v << " " << edge.weight[0] << " " << edge.weight[1] << endl;
        }
        outFile.close();
        cout << "Graph saved to " << filename << endl;
    } else {
        cerr << "Unable to open file " << filename << endl;
    }
}






// Structure to store a path cost (bi-objective)
struct PathCost {
    int weight1, weight2;

    // Compare two PathCosts for Pareto optimality (non-dominance)
    bool dominates(const PathCost& other) const {
        return (weight1 <= other.weight1 && weight2 <= other.weight2)
            && (weight1 < other.weight1 || weight2 < other.weight2);
    }
};

// Structure for priority queue (min-heap)
struct QueueNode {
    int node;
    PathCost cost;

    // Custom comparator for priority queue (minimize on weight1 and weight2)
    bool operator>(const QueueNode& other) const {
        return cost.weight1 > other.cost.weight1 || 
               (cost.weight1 == other.cost.weight1 && cost.weight2 > other.cost.weight2);
    }
};

// Function to perform the bi-objective Dijkstra algorithm with Pareto optimality
void biObjectiveDijkstra(int n, int start, const vector<Edge>& graph, vector<vector<PathCost>>& paretoFront) {
    priority_queue<QueueNode, vector<QueueNode>, greater<QueueNode>> pq;
    
    // Initialize the Pareto front with a large value (infinity) for all nodes except the start node
    paretoFront[start].push_back({0, 0});  // Starting node has cost (0, 0)
    pq.push({start, {0, 0}});
    
    while (!pq.empty()) {
        QueueNode current = pq.top();
        pq.pop();
        
        int u = current.node;
        PathCost currentCost = current.cost;

        // Process all neighbors of the current node
        for (const auto& edge : graph) {
            if (edge.u == u) {
                int v = edge.v;
                PathCost newCost = {currentCost.weight1 + edge.weight[0], currentCost.weight2 + edge.weight[1]};
                
                // Check if the new path is Pareto optimal for node v
                bool isParetoOptimal = true;
                for (const auto& existingCost : paretoFront[v]) {
                    if (existingCost.dominates(newCost)) {
                        isParetoOptimal = false;  // The new path is dominated by an existing path
                        break;
                    }
                }

                if (isParetoOptimal) {
                    // Remove dominated paths from the Pareto front
                    paretoFront[v].erase(
                        remove_if(paretoFront[v].begin(), paretoFront[v].end(),
                                  [&](const PathCost& cost) { return newCost.dominates(cost); }),
                        paretoFront[v].end()
                    );

                    // Add the new Pareto-optimal path
                    paretoFront[v].push_back(newCost);
                    pq.push({v, newCost});
                }
            }
        }
    }
}

// Function to save the Pareto-optimal shortest path tree to a file
void saveShortestPathTree(const vector<vector<PathCost>>& paretoFront, const string& filename) {
    ofstream outFile(filename);
    if (outFile.is_open()) {
        for (int i = 0; i < paretoFront.size(); ++i) {
            outFile << "" << i << " ";
            for (const auto& cost : paretoFront[i]) {
                outFile << "" << cost.weight1 << " " << cost.weight2 << "\n";
                // Store only first pareto optimal data.
                break;
            }
        }
        outFile.close();
        cout << "Shortest path tree saved to " << filename << endl;
    } else {
        cerr << "Unable to open file " << filename << endl;
    }
}

// Function to convert a directed graph to an undirected graph
vector<Edge> convertToUndirected(const vector<Edge>& directedEdges) {
    vector<Edge> undirectedEdges;
    set<pair<int, int>> addedEdges;

    for (const auto& edge : directedEdges) {
        int u = edge.u;
        int v = edge.v;
        vector<int> weight = edge.weight;

        // Add edge (u -> v)
        if (addedEdges.find({u, v}) == addedEdges.end()) {
            undirectedEdges.push_back({u, v, weight});
            addedEdges.insert({u, v});
        }

        // Add edge (v -> u) to make it undirected
        if (addedEdges.find({v, u}) == addedEdges.end()) {
            undirectedEdges.push_back({v, u, weight});
            addedEdges.insert({v, u});
        }
    }

    return undirectedEdges;
}



// Structure for priority queue (min-heap) for single-objective Dijkstra
struct QueueNodeSingleObjective {
    int node;
    int cost; // Single-objective cost

    // Custom comparator for priority queue (minimize on cost)
    bool operator>(const QueueNodeSingleObjective& other) const {
        return cost > other.cost;
    }
};

// Function to perform single-objective Dijkstra algorithm
void singleObjectiveDijkstra(int n, int start, const vector<Edge>& graph, vector<vector<int>>& tree, int objective) {
    priority_queue<QueueNodeSingleObjective, vector<QueueNodeSingleObjective>, greater<QueueNodeSingleObjective>> pq;
    vector<int> dist(n, numeric_limits<int>::max());
    vector<int> parent(n, -1);  // Store parent of each node
    
    // Initialize the distances with a large value (infinity)
    dist[start] = 0;  // Starting node has cost 0
    pq.push({start, 0});
    
    while (!pq.empty()) {
        QueueNodeSingleObjective current = pq.top();
        pq.pop();
        
        int u = current.node;
        int currentCost = current.cost;

        // Process all neighbors of the current node
        for (const auto& edge : graph) {
            if (edge.u == u) {
                int v = edge.v;
                int edgeCost = edge.weight[objective]; // Use the selected objective

                if (currentCost + edgeCost < dist[v]) {
                    dist[v] = currentCost + edgeCost;
                    parent[v] = u;  // Set parent for v
                    pq.push({v, dist[v]});
                }
            }
        }
    }

    // Convert parent relationships into an adjacency list for the tree
    tree.resize(n);
    for (int v = 0; v < n; ++v) {
        if (parent[v] != -1) {
            tree[parent[v]].push_back(v);
        }
    }
}

// Function to save the adjacency list of a tree to a file
void saveTree(const vector<vector<int>>& tree, const string& filename) {
    ofstream outFile(filename);
    if (outFile.is_open()) {
        for (int i = 0; i < tree.size(); ++i) {
            outFile << i;
            for (int child : tree[i]) {
                outFile << " " << child;
            }
            outFile << "\n";
        }
        outFile.close();
        cout << "Tree saved to " << filename << endl;
    } else {
        cerr << "Unable to open file " << filename << endl;
    }
}



// Function to print the tree (adjacency list)
void printTree(const vector<vector<int>>& tree, const string& treeName) {
    cout << treeName << ":\n";
    for (int i = 0; i < tree.size(); ++i) {
        cout << "Node " << i << ":";
        for (int child : tree[i]) {
            cout << " " << child;
        }
        cout << endl;
    }
}

// Function to read a tree (adjacency list) from a file
vector<vector<int>> readTree(const string& filename) {
    ifstream infile(filename);
    vector<vector<int>> adjList;
    string line;

    if (!infile.is_open()) {
        cerr << "Error opening file " << filename << endl;
        return adjList;
    }

    // Read the file and build the adjacency list
    while (getline(infile, line)) {
        istringstream iss(line);
        int node;
        iss >> node;

        vector<int> children;
        int child;
        while (iss >> child) {
            children.push_back(child);
        }

        if (node >= adjList.size()) {
            adjList.resize(node + 1);
        }
        adjList[node] = children;
    }

    infile.close();
    return adjList;
}

// Function to save the combined graph to a file
void saveCombinedGraph(const unordered_map<int, unordered_map<int, int>>& graph, const string& filename) {
    ofstream outFile(filename);
    if (outFile.is_open()) {
        for (const auto& [node, neighbors] : graph) {
            outFile << node;
            for (const auto& [neighbor, weight] : neighbors) {
                outFile << " " << neighbor << ":" << weight;
            }
            outFile << "\n";
        }
        outFile.close();
        cout << "Combined graph saved to " << filename << endl;
    } else {
        cerr << "Unable to open file " << filename << endl;
    }
}

// Function to create the combined graph
unordered_map<int, unordered_map<int, int>> createCombinedGraph(
    const vector<vector<int>>& tree1, const vector<vector<int>>& tree2) {

    unordered_map<int, unordered_map<int, int>> combinedGraph;

    // Step 1: Add edges from treeObj1 with weight 2
    for (int u = 0; u < tree1.size(); ++u) {
        for (int v : tree1[u]) {
            combinedGraph[u][v] = 2;
            combinedGraph[v][u] = 2; // Since it's an undirected graph
        }
    }

    // Step 2: Add or update edges from treeObj2
    for (int u = 0; u < tree2.size(); ++u) {
        for (int v : tree2[u]) {
            if (combinedGraph[u].find(v) != combinedGraph[u].end()) {
                // If edge exists, reduce weight to 1
                combinedGraph[u][v] = 1;
                combinedGraph[v][u] = 1;
            } else {
                // If edge does not exist, add with weight 2
                combinedGraph[u][v] = 2;
                combinedGraph[v][u] = 2;
            }
        }
    }

    return combinedGraph;
}

void convertToEdgeList(const string& inputFile, const string& outputFile) {
    ifstream inFile(inputFile);
    ofstream outFile(outputFile);
    
    if (!inFile.is_open() || !outFile.is_open()) {
        cerr << "Error opening file!" << endl;
        return;
    }

    unordered_set<string> visitedEdges; // To track visited edges
    string line;

    while (getline(inFile, line)) {
        istringstream iss(line);
        int node1;
        iss >> node1;
        
        string neighbor;
        while (iss >> neighbor) {
            size_t colonPos = neighbor.find(':');
            int node2 = stoi(neighbor.substr(0, colonPos));
            int weight = stoi(neighbor.substr(colonPos + 1));

            // Create an edge identifier in the format "node1-node2"
            string edge = to_string(min(node1, node2)) + "-" + to_string(max(node1, node2));
            
            // If the edge hasn't been visited yet, write it to the output file
            if (visitedEdges.find(edge) == visitedEdges.end()) {
                outFile << node1 << " " << node2 << " " << weight << " " << weight << endl;
                outFile << node2 << " " << node1 << " " << weight << " " << weight << endl;
                visitedEdges.insert(edge);
            }
        }
    }

    inFile.close();
    outFile.close();
}
vector<Edge> readGraphFromFile(const string& filename) {
    ifstream inFile(filename);
    vector<Edge> graph;
    if (!inFile.is_open()) {
        cerr << "Error opening file " << filename << endl;
        return graph;
    }
    
    int u, v, w1, w2;
    while (inFile >> u >> v >> w1 >> w2) {
        graph.push_back({u, v, {w1, w2}});
    }
    
    inFile.close();
    return graph;
}
vector<Edge> readEdgeList(const string& filename) {
    ifstream inFile(filename);
    vector<Edge> edges;
    if (!inFile.is_open()) {
        cerr << "Error opening file " << filename << endl;
        return edges;
    }

    int u, v, weight;
    while (inFile >> u >> v >> weight) {
        edges.push_back({u, v, {weight, weight}}); // For now, use the same weight for both objectives
    }

    inFile.close();
    return edges;
}

// Function to read the graph from "graph.txt"
vector<tuple<int, int, int, int>> readOriginalGraphFromFile(const string& filename) {
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
vector<pair<int, int>> readDistancesFromFile(const string& filename) {
    vector<pair<int, int>> distances;
    ifstream inFile(filename);
    if (inFile.is_open()) {
        string line;
        while (getline(inFile, line)) {
            istringstream iss(line);
            int node, dist1, dist2;
            if (iss >> node >> dist1 >> dist2) {
                distances.push_back({dist1, dist2});
            }
        }
        inFile.close();
    } else {
        cerr << "Unable to open file " << filename << endl;
    }
    return distances;
}

// Function to save the difference of bi-objective distances to a file
void saveDiffToFile(const vector<pair<int, int>>& diffDistances, const string& filename) {
    ofstream outFile(filename);
    if (outFile.is_open()) {
        for (int i = 0; i < diffDistances.size(); ++i) {
            outFile << i << " " << diffDistances[i].first << " " << diffDistances[i].second << "\n";
        }
        outFile.close();
    } else {
        cerr << "Unable to open file " << filename << endl;
    }
}

int main() {
    int n = 10, m = 50, r1 = 1, r2 = 10;

    if (m < n - 1) {
        cout << "Number of edges must be at least n-1 to create a connected graph." << endl;
        return 1;
    }

    // Step 1: Generate a directed graph
    vector<Edge> directedGraph = generateDirectedGraph(n, m, r1, r2);

    // Step 2: Convert the directed graph to undirected
    vector<Edge> undirectedGraph = convertToUndirected(directedGraph);

    // Step 3: Print the undirected graph
    printGraph(undirectedGraph);

    // Step 4: Save the undirected graph to a file
    saveGraphToFile(undirectedGraph, "undirected_graph.txt");

    // Step 5: Perform bi-objective Dijkstra to obtain Pareto-optimal shortest path tree
    vector<vector<PathCost>> paretoFront(n);
    biObjectiveDijkstra(n, 0, undirectedGraph, paretoFront);  // Root at node 0

    // Step 6: Save the shortest path tree to a file
    saveShortestPathTree(paretoFront, "tree.txt");

    string graphFile = "undirected_graph.txt"; // The input file that contains the graph with two objectives
    string treeFile1 = "treeObj1.txt"; // Tree generated by first objective
    string treeFile2 = "treeObj2.txt"; // Tree generated by second objective

    // Step 1: Read the graph from file
    vector<Edge> graph = readGraphFromFile(graphFile);
    if (graph.empty()) {
        cerr << "Failed to read graph from " << graphFile << endl;
        return 1;
    }

    // Step 2: Generate treeObj1.txt based on the first objective
    vector<vector<int>> treeObj1;
    singleObjectiveDijkstra(n, 0, graph, treeObj1, 0); // Objective 0 is the first weight
    saveTree(treeObj1, treeFile1);

    // Step 3: Generate treeObj2.txt based on the second objective
    vector<vector<int>> treeObj2;
    singleObjectiveDijkstra(n, 0, graph, treeObj2, 1); // Objective 1 is the second weight
    saveTree(treeObj2, treeFile2);

    // Step 7: Read and print treeObj1
    treeObj1 = readTree("treeObj1.txt");
    printTree(treeObj1, "Tree Object 1");

    // Step 8: Read and print treeObj2
    treeObj2 = readTree("treeObj2.txt");
    printTree(treeObj2, "Tree Object 2");

    // Step 9: Create the combined graph based on the rules
    unordered_map<int, unordered_map<int, int>> combinedGraph = createCombinedGraph(treeObj1, treeObj2);

    // Step 10: Save the combined graph to a file
    saveCombinedGraph(combinedGraph, "combinedGraph.txt");

    string inputFile = "combinedGraph.txt";
    string outputFile = "combinedGraphEL.txt";

    convertToEdgeList(inputFile, outputFile);

    cout << "Edge list saved to " << outputFile << endl;

    inputFile = "combinedGraphEL.txt";  // The edge list file to read from
    string treeFile = "combinedGraphTree.txt"; // The output file for the tree
    
    // Step 1: Read the combined graph from the edge list file
    vector<Edge> combinedGraphRead = readGraphFromFile(inputFile);
    if (combinedGraphRead.empty()) {
        cerr << "Failed to read combined graph from " << inputFile << endl;
        return 1;
    }

    // Step 2: Run single-objective Dijkstra's algorithm on the combined graph (use objective 0)
    vector<vector<int>> combinedGraphTree;
    singleObjectiveDijkstra(n, 0, combinedGraphRead, combinedGraphTree, 0); // Use objective 0 for simplicity

    // Step 3: Save the resulting tree to a file
    saveTree(combinedGraphTree, treeFile);
    
    cout << "Combined graph tree saved to " << treeFile << endl;

    vector<tuple<int, int, int, int>> edgeList = readOriginalGraphFromFile("undirected_graph.txt");
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