#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <set>
#include <cuda_runtime.h>

using namespace std;

// --- Define a tile size for shared memory tiling ---
#define TILE_SIZE 32

// --- Error checking macro ---
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess){
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// --- Read a graph from a file (each line is a row of the adjacency matrix) ---
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

// --- Randomly generate labels: class0, class1, and unlabeled (classU) ---
void generateLabels(int nNodes, double percentage, vector<int>& class0, vector<int>& class1, vector<int>& classU) {
    int numClass0 = static_cast<int>(nNodes * (percentage / 100.0));
    set<int> labeledSet;
    srand(static_cast<unsigned int>(time(0)));

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

// --- Flatten a 2D vector (graph) into a 1D vector (row‐major order) ---
vector<int> flattenGraph(const vector<vector<int>>& graph) {
    vector<int> flattened;
    for (const auto& row : graph) {
        for (const auto& cell : row) {
            flattened.push_back(cell);
        }
    }
    return flattened;
}

/*
  --- Optimized CUDA kernel ---
  
  Each thread is responsible for one “active” node (i.e. if cA[tid]==1).
  For that node, we must sum over the row in the graph.
  To accelerate the inner loop over j, we:
    - Use pitched memory for the flattened graph (for better alignment).
    - Tile the loop over j and load small portions of c0 and c1 into shared memory.
    - Mark read-only arrays with __restrict__ and const.
*/
__global__ void kernelProcessNodes(const int* __restrict__ d_flattenedGraph, size_t pitch, 
                                     const int* __restrict__ d_c0, const int* __restrict__ d_c1, 
                                     int* d_cA, float* d_lab, int nNodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Only process “active” nodes:
    if (tid < nNodes && d_cA[tid] == 1) {
        int class0Sum = 0, class1Sum = 0, classUSum = 0, classUCount = 0;

        // Use pitched memory to get the pointer to the beginning of this row.
        const int* row = (const int*)((const char*)d_flattenedGraph + tid * pitch);

        // Allocate shared memory for a tile of d_c0 and d_c1.
        // The total shared memory size is 2*TILE_SIZE*sizeof(int).
        extern __shared__ int shared[];
        int* s_c0 = shared;              // Tile for d_c0
        int* s_c1 = shared + TILE_SIZE;    // Tile for d_c1

        // Process the row in tiles.
        int nTiles = (nNodes + TILE_SIZE - 1) / TILE_SIZE;
        for (int tile = 0; tile < nTiles; tile++) {
            int tileStart = tile * TILE_SIZE;
            // Load one tile of d_c0 and d_c1 into shared memory.
            for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
                int col = tileStart + i;
                if (col < nNodes) {
                    s_c0[i] = d_c0[col];
                    s_c1[i] = d_c1[col];
                } else {
                    s_c0[i] = 0;
                    s_c1[i] = 0;
                }
            }
            __syncthreads();

            // Process this tile:
            #pragma unroll
            for (int i = 0; i < TILE_SIZE; i++) {
                int col = tileStart + i;
                if (col < nNodes) {
                    int weight = row[col]; // Using pitched memory pointer
                    if (s_c0[i] == 1)
                        class0Sum += weight;
                    if (s_c1[i] == 1)
                        class1Sum += weight;
                    if (s_c0[i] == 0 && s_c1[i] == 0) {
                        classUSum += weight;
                        classUCount++;
                    }
                }
            }
            __syncthreads();
        }
        int totalSum = class0Sum + class1Sum + classUSum;
        if (totalSum > 0) {
            float newVal = d_lab[tid]
                           - ((float)class0Sum / totalSum * 0.5f)
                           + ((float)class1Sum / totalSum * 0.5f)
                           + ((float)classUCount / totalSum * 0.5f);
            d_lab[tid] = newVal;
        }
        d_cA[tid] = 0;
    }
}

int main() {
    // --- Read in the graph ---
    string filename = "/mnt/stor/ceph/csc/sdas-lab/Shovan/adjacency_matrix10000_p0.1.txt";
    vector<vector<int>> graph = readGraphFromFile(filename);
    int nNodes = graph.size();
    if(nNodes == 0) {
        cerr << "Graph is empty or could not be read." << endl;
        return -1;
    }

    // --- Generate labels ---
    vector<int> class0, class1, classU;
    generateLabels(nNodes, 10.0, class0, class1, classU);
    
    // --- Prepare host arrays ---
    vector<int> h_c0(nNodes, 0), h_c1(nNodes, 0), h_cA(nNodes, 0);
    vector<float> h_lab(nNodes, 0.5f);
    for (int node : class0) { h_c0[node] = 1; h_lab[node] = 0.0f; }
    for (int node : class1) { h_c1[node] = 1; h_lab[node] = 1.0f; }
    for (int node : classU) { h_cA[node] = 1; }
    
    // --- Flatten the graph ---
    vector<int> flattenedGraph = flattenGraph(graph);

    // --- Allocate device memory for c0, c1, cA, and lab ---
    int *d_c0, *d_c1, *d_cA;
    float *d_lab;
    size_t nodeSize = nNodes * sizeof(int);
    size_t labSize  = nNodes * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_c0, nodeSize));
    CUDA_CHECK(cudaMalloc(&d_c1, nodeSize));
    CUDA_CHECK(cudaMalloc(&d_cA, nodeSize));
    CUDA_CHECK(cudaMalloc(&d_lab, labSize));

    // --- Allocate pitched memory for the flattened graph ---
    int* d_flattenedGraph;
    size_t pitch;
    size_t widthInBytes = nNodes * sizeof(int);
    CUDA_CHECK(cudaMallocPitch(&d_flattenedGraph, &pitch, widthInBytes, nNodes));
    // Copy host flattened graph into pitched device memory
    CUDA_CHECK(cudaMemcpy2D(d_flattenedGraph, pitch, 
                            flattenedGraph.data(), widthInBytes,
                            widthInBytes, nNodes,
                            cudaMemcpyHostToDevice));

    // --- Copy the other arrays to the device ---
    CUDA_CHECK(cudaMemcpy(d_c0, h_c0.data(), nodeSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c1, h_c1.data(), nodeSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cA, h_cA.data(), nodeSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lab, h_lab.data(), labSize, cudaMemcpyHostToDevice));

    // --- Launch kernel ---
    int blockSize = 256;
    int gridSize  = (nNodes + blockSize - 1) / blockSize;
    // Shared memory: we need space for 2 * TILE_SIZE integers.
    size_t sharedMemSize = 2 * TILE_SIZE * sizeof(int);
    
    for (int it = 0; it < 10; it++) {
        kernelProcessNodes<<<gridSize, blockSize, sharedMemSize>>>(d_flattenedGraph, pitch,
                                                                     d_c0, d_c1, d_cA, d_lab,
                                                                     nNodes);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // --- Copy results back to host ---
    CUDA_CHECK(cudaMemcpy(h_lab.data(), d_lab, labSize, cudaMemcpyDeviceToHost));
    
    // --- Free device memory ---
    CUDA_CHECK(cudaFree(d_flattenedGraph));
    CUDA_CHECK(cudaFree(d_c0));
    CUDA_CHECK(cudaFree(d_c1));
    CUDA_CHECK(cudaFree(d_cA));
    CUDA_CHECK(cudaFree(d_lab));
    
    // --- Output updated labels ---
    cout << "Updated Labels: ";
    for (float l : h_lab)
        cout << l << " ";
    cout << endl;
    
    return 0;
}
