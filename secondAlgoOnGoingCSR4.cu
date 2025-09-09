#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <set>
#include <cuda_runtime.h>

using namespace std;

// --- CSR Loading ---
vector<int> loadVectorFromFile(const string& filename) {
    vector<int> vec;
    ifstream inFile(filename);
    int val;
    while (inFile >> val) vec.push_back(val);
    return vec;
}

// --- Label generation ---
void generateLabels(int nNodes, double percentage, vector<int>& class0, vector<int>& class1, vector<int>& classU) {
    int numClass0 = static_cast<int>(nNodes * (percentage / 100.0));
    set<int> labeledSet;
    srand(static_cast<unsigned int>(time(0)));

    while (labeledSet.size() < numClass0) {
        int node = rand() % nNodes;
        if (labeledSet.insert(node).second)
            class0.push_back(node);
    }
    while (labeledSet.size() < 2 * numClass0) {
        int node = rand() % nNodes;
        if (labeledSet.insert(node).second)
            class1.push_back(node);
    }
    for (int i = 0; i < nNodes; ++i) {
        if (labeledSet.find(i) == labeledSet.end())
            classU.push_back(i);
    }
}

// --- Error check ---
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess){
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// --- CUDA Kernel ---
__global__ void processNodesCSR(
    const int* row_ptr, const int* col_ind, const int* val,
    const int* d_c0, const int* d_c1, int* d_cA, float* d_lab,
    int nNodes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nNodes && d_cA[tid] == 1) {
        int class0Sum = 0, class1Sum = 0, classUSum = 0, classUCount = 0;
        for (int idx = row_ptr[tid]; idx < row_ptr[tid + 1]; ++idx) {
            int col = col_ind[idx];
            int weight = val[idx];

            if (d_c0[col] == 1) class0Sum += weight;
            else if (d_c1[col] == 1) class1Sum += weight;
            else { classUSum += weight; classUCount++; }
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
    // --- Load CSR from files ---
    string basePath = "/mnt/stor/ceph/csc/sdas-lab/Shovan/";
    vector<int> row_ptr = loadVectorFromFile(basePath + "row_ptr.txt");
    vector<int> col_ind = loadVectorFromFile(basePath + "col_ind.txt");
    vector<int> val     = loadVectorFromFile(basePath + "val.txt");

    int nNodes = row_ptr.size() - 1;
    int nEdges = col_ind.size();
    cout << "Loaded CSR graph with " << nNodes << " nodes and " << nEdges << " edges.\n";

    // --- Label generation ---
    vector<int> class0, class1, classU;
    generateLabels(nNodes, 10.0, class0, class1, classU);

    // --- Host arrays ---
    vector<int> h_c0(nNodes, 0), h_c1(nNodes, 0), h_cA(nNodes, 0);
    vector<float> h_lab(nNodes, 0.5f);

    for (int i : class0) { h_c0[i] = 1; h_lab[i] = 0.0f; }
    for (int i : class1) { h_c1[i] = 1; h_lab[i] = 1.0f; }
    for (int i : classU) { h_cA[i] = 1; }

    // --- Device memory ---
    int *d_row_ptr, *d_col_ind, *d_val;
    int *d_c0, *d_c1, *d_cA;
    float *d_lab;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, row_ptr.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_ind, col_ind.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val, val.size() * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_c0, nNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_c1, nNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cA, nNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lab, nNodes * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_ind, col_ind.data(), col_ind.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val, val.data(), val.size() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_c0, h_c0.data(), nNodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c1, h_c1.data(), nNodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cA, h_cA.data(), nNodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lab, h_lab.data(), nNodes * sizeof(float), cudaMemcpyHostToDevice));

    // --- Kernel launch ---
    int blockSize = 256;
    int gridSize = (nNodes + blockSize - 1) / blockSize;

    for (int it = 0; it < 10; it++) {
        processNodesCSR<<<gridSize, blockSize>>>(
            d_row_ptr, d_col_ind, d_val,
            d_c0, d_c1, d_cA, d_lab, nNodes
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // --- Copy results back ---
    CUDA_CHECK(cudaMemcpy(h_lab.data(), d_lab, nNodes * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Free ---
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_ind));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_c0));
    CUDA_CHECK(cudaFree(d_c1));
    CUDA_CHECK(cudaFree(d_cA));
    CUDA_CHECK(cudaFree(d_lab));

    // --- Output ---
    cout << "Updated Labels:\n";
    for (float l : h_lab) cout << l << " ";
    cout << "\n";

    return 0;
}
