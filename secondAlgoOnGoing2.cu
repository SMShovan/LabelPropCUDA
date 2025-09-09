#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <set>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

using namespace std;

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

void generateLabels(int nNodes, double percentage, vector<int>& class0, vector<int>& class1, vector<int>& classU) {
    int numClass0 = static_cast<int>(nNodes * (percentage / 100.0));
    set<int> labeledSet;
    srand(time(0));

    while (labeledSet.size() < numClass0) {
        int node = rand() % nNodes;
        if (labeledSet.insert(node).second) {
            class0.push_back(node);
        }
    }
    while (labeledSet.size() < 2 * numClass0) {
        int node = rand() % nNodes;
        if (labeledSet.insert(node).second) {
            class1.push_back(node);
        }
    }
    for (int i = 0; i < nNodes; ++i) {
        if (labeledSet.find(i) == labeledSet.end()) {
            classU.push_back(i);
        }
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

__global__ void kernelProcessNodes(const int* flattenedGraph, int* c0, int* c1, int* cA, float* lab, int nNodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nNodes && cA[tid] == 1) {
        int class0Sum = 0, class1Sum = 0, classUSum = 0, classUCount = 0;
        for (int j = 0; j < nNodes; j++) {
            int weight = flattenedGraph[tid * nNodes + j];
            if (c0[j] == 1) class0Sum += weight;
            if (c1[j] == 1) class1Sum += weight;
            if (c0[j] == 0 && c1[j] == 0) {
                classUSum += weight;
                classUCount++;
            }
        }
        int totalSum = class0Sum + class1Sum + classUSum;
        if (totalSum > 0) {
            float newVal = lab[tid] - ((float)class0Sum / totalSum * 0.5) + ((float)class1Sum / totalSum * 0.5) + ((float)classUCount / totalSum * 0.5);
            lab[tid] = newVal;
        }
        cA[tid] = 0;
    }
}

int main() {
    vector<vector<int>> graph = readGraphFromFile("/mnt/stor/ceph/csc/sdas-lab/Shovan/adjacency_matrix10000_p0.1.txt");
    int nNodes = graph.size();
    vector<int> class0, class1, classU;
    generateLabels(nNodes, 10.0, class0, class1, classU);
    
    vector<int> c0(nNodes, 0), c1(nNodes, 0), cA(nNodes, 0);
    vector<float> lab(nNodes, 0.5);
    for (int node : class0) { c0[node] = 1; lab[node] = 0.0; }
    for (int node : class1) { c1[node] = 1; lab[node] = 1.0; }
    for (int node : classU) { cA[node] = 1; }
    
    vector<int> flattenedGraph = flattenGraph(graph);
    int* d_flattenedGraph, * d_c0, * d_c1, * d_cA;
    float* d_lab;
    size_t size = nNodes * nNodes * sizeof(int), nodeSize = nNodes * sizeof(int), labSize = nNodes * sizeof(float);
    cudaMalloc(&d_flattenedGraph, size);
    cudaMalloc(&d_c0, nodeSize);
    cudaMalloc(&d_c1, nodeSize);
    cudaMalloc(&d_cA, nodeSize);
    cudaMalloc(&d_lab, labSize);
    
    cudaMemcpy(d_flattenedGraph, flattenedGraph.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c0, c0.data(), nodeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c1, c1.data(), nodeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cA, cA.data(), nodeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lab, lab.data(), labSize, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (nNodes + blockSize - 1) / blockSize;
    int iterCount = 0;
    int maxIters = 1000;
    
    while (iterCount < maxIters) {
        kernelProcessNodes<<<gridSize, blockSize>>>(d_flattenedGraph, d_c0, d_c1, d_cA, d_lab, nNodes);
        cudaDeviceSynchronize();

        // Efficient sum of d_cA using thrust
        thrust::device_ptr<int> d_cA_ptr(d_cA);
        int sum_cA = thrust::reduce(d_cA_ptr, d_cA_ptr + nNodes);

        // if (sum_cA == 0) {
        //     cout << "Converged in " << iterCount << " iterations." << endl;
        //     break;
        // }

        iterCount++;
    }

    if (iterCount == maxIters) {
        cout << "infinity" << endl;
    }

    
    cudaMemcpy(lab.data(), d_lab, labSize, cudaMemcpyDeviceToHost);
    
    cudaFree(d_flattenedGraph);
    cudaFree(d_c0);
    cudaFree(d_c1);
    cudaFree(d_cA);
    cudaFree(d_lab);
    
    cout << "Updated Labels: ";
    for (float l : lab) cout << l << " ";
    cout << endl;
    
    return 0;
}
/*
adjacency_matrix10000_p0.1.txt  adjacency_matrix10000_p0.9.txt  adjacency_matrix1000_p0.5.txt  adjacency_matrix.txt
adjacency_matrix10000_p0.3.txt  adjacency_matrix10000.txt       adjacency_matrix1000_p0.7.txt
adjacency_matrix10000_p0.5.txt  adjacency_matrix1000_p0.1.txt   adjacency_matrix1000_p0.9.txt
adjacency_matrix10000_p0.7.txt  adjacency_matrix1000_p0.3.txt   adjacency_matrix1000.txt
*/