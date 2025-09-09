#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

int main() {
    string inputFile = "/mnt/stor/ceph/csc/sdas-lab/Shovan/adjacency_matrix10000_p0.1.txt";
    ifstream inFile(inputFile);
    vector<vector<int>> adjacencyMatrix;

    if (!inFile.is_open()) {
        cerr << "Failed to open " << inputFile << endl;
        return 1;
    }

    // Read matrix from file
    string line;
    while (getline(inFile, line)) {
        istringstream iss(line);
        vector<int> row;
        int val;
        while (iss >> val) {
            row.push_back(val);
        }
        adjacencyMatrix.push_back(row);
    }
    inFile.close();

    int n = adjacencyMatrix.size();
    vector<int> row_ptr(n + 1, 0);
    vector<int> col_ind;
    vector<int> val;

    // Convert to CSR
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < adjacencyMatrix[i].size(); ++j) {
            if (adjacencyMatrix[i][j] != 0) {
                col_ind.push_back(j);
                val.push_back(adjacencyMatrix[i][j]);
                row_ptr[i + 1]++;
            }
        }
    }

    for (int i = 1; i <= n; ++i) {
        row_ptr[i] += row_ptr[i - 1];
    }

    // Output files
    ofstream outRow("/mnt/stor/ceph/csc/sdas-lab/Shovan/row_ptr.txt"), outCol("/mnt/stor/ceph/csc/sdas-lab/Shovan/col_ind.txt"), outVal("/mnt/stor/ceph/csc/sdas-lab/Shovan/val.txt");

    for (int r : row_ptr) outRow << r << " ";
    for (int c : col_ind) outCol << c << " ";
    for (int v : val) outVal << v << " ";

    outRow.close();
    outCol.close();
    outVal.close();

    cout << "CSR written to row_ptr.txt, col_ind.txt, val.txt\n";

    return 0;
}
