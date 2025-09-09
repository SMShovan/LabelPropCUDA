#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <cuda_runtime.h>

#include "../util/cuda_checks.cuh"

using std::cout;
using std::endl;
using std::string;
using std::vector;

// ------------------------
// Helpers: read vectors
// ------------------------
static vector<int> readIntVector(const string &path) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open " << path << endl; std::exit(EXIT_FAILURE); }
    vector<int> a; a.reserve(1024);
    int x; while (f >> x) a.push_back(x);
    return a;
}
static vector<double> readDoubleVector(const string &path) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open " << path << endl; std::exit(EXIT_FAILURE); }
    vector<double> a; a.reserve(1024);
    double x; while (f >> x) a.push_back(x);
    return a;
}

// ------------------------
// CLI parse (very light)
// ------------------------
struct Options {
    string csr_row, csr_col, csr_val;
    string l0_file, l1_file;
    string delta_file;
    string delta_labels_file; // optional: labels for selected delta GT (0/1 per line)
    double tau = 0.0;
    double delta_threshold = 1e-4;
    int max_iters = 50;
    double alpha = 1.0; // unused for now
    double delta_labeled_pct = 0.0; // fraction of delta to clamp as GT
};

static Options parseArgs(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--csr-row" && i+1 < argc) o.csr_row = argv[++i];
        else if (a == "--csr-col" && i+1 < argc) o.csr_col = argv[++i];
        else if (a == "--csr-val" && i+1 < argc) o.csr_val = argv[++i];
        else if (a == "--l0" && i+1 < argc) o.l0_file = argv[++i];
        else if (a == "--l1" && i+1 < argc) o.l1_file = argv[++i];
        else if (a == "--delta" && i+1 < argc) o.delta_file = argv[++i];
        else if (a == "--delta-labels" && i+1 < argc) o.delta_labels_file = argv[++i];
        else if (a == "--tau" && i+1 < argc) o.tau = std::atof(argv[++i]);
        else if (a == "--delta-threshold" && i+1 < argc) o.delta_threshold = std::atof(argv[++i]);
        else if (a == "--max-iters" && i+1 < argc) o.max_iters = std::atoi(argv[++i]);
        else if (a == "--alpha" && i+1 < argc) o.alpha = std::atof(argv[++i]);
        else if (a == "--delta-labeled-pct" && i+1 < argc) o.delta_labeled_pct = std::atof(argv[++i]);
    }
    if (o.csr_row.empty() || o.csr_col.empty() || o.csr_val.empty() || o.delta_file.empty()) {
        std::cerr << "Usage: proposed --csr-row ROW --csr-col COL --csr-val VAL --delta DELTA_IDS --l0 L0 --l1 L1 --tau T --delta-threshold D --max-iters N\n";
        std::exit(EXIT_FAILURE);
    }
    return o;
}

// --------------------------------------------------
// Step 1: Build Delta CSR (GPU) with threshold tau
// --------------------------------------------------
#include "../kernels/delta_csr.cuh"

static void buildDeltaCsrGpu(
    const vector<int>& row, const vector<int>& col, const vector<double>& val,
    const vector<int>& delta_ids, int n,
    double tau,
    vector<int>& d_row_h, vector<int>& d_col_h, vector<double>& d_val_h,
    vector<int>& delta_pos_h
) {
    int K = static_cast<int>(delta_ids.size());
    d_row_h.assign(K+1, 0);
    delta_pos_h.assign(n, -1);
    if (K == 0) { d_col_h.clear(); d_val_h.clear(); return; }

    int *d_row = nullptr, *d_col = nullptr, *d_delta_ids = nullptr, *d_delta_pos = nullptr;
    double *d_val = nullptr; int *d_counts = nullptr;
    checkCudaStatus(cudaMalloc((void**)&d_row, sizeof(int)*(n+1)), "malloc row");
    checkCudaStatus(cudaMalloc((void**)&d_col, sizeof(int)*col.size()), "malloc col");
    checkCudaStatus(cudaMalloc((void**)&d_val, sizeof(double)*val.size()), "malloc val");
    checkCudaStatus(cudaMemcpy(d_row, row.data(), sizeof(int)*(n+1), cudaMemcpyHostToDevice), "copy row");
    checkCudaStatus(cudaMemcpy(d_col, col.data(), sizeof(int)*col.size(), cudaMemcpyHostToDevice), "copy col");
    checkCudaStatus(cudaMemcpy(d_val, val.data(), sizeof(double)*val.size(), cudaMemcpyHostToDevice), "copy val");

    checkCudaStatus(cudaMalloc((void**)&d_delta_ids, sizeof(int)*K), "malloc delta_ids");
    checkCudaStatus(cudaMemcpy(d_delta_ids, delta_ids.data(), sizeof(int)*K, cudaMemcpyHostToDevice), "copy delta_ids");
    checkCudaStatus(cudaMalloc((void**)&d_delta_pos, sizeof(int)*n), "malloc delta_pos");

    dim3 block(256), gridDelta((K + block.x - 1) / block.x), gridN((n + block.x - 1) / block.x);
    k_init_array_int<<<gridN, block>>>(d_delta_pos, n, -1);
    checkCudaStatus(cudaDeviceSynchronize(), "init delta_pos");
    k_scatter_delta_pos<<<gridDelta, block>>>(d_delta_ids, K, d_delta_pos);
    checkCudaStatus(cudaDeviceSynchronize(), "scatter delta_pos");

    checkCudaStatus(cudaMalloc((void**)&d_counts, sizeof(int)*K), "malloc counts");
    k_count_delta_edges<<<gridDelta, block>>>(d_row, d_col, d_val, d_delta_ids, d_delta_pos, K, tau, d_counts);
    checkCudaStatus(cudaDeviceSynchronize(), "count edges");

    // exclusive scan on device counts -> d_row_h
    // Copy counts to host and do a simple CPU scan (K is small relative to n)
    vector<int> counts_h(K);
    checkCudaStatus(cudaMemcpy(counts_h.data(), d_counts, sizeof(int)*K, cudaMemcpyDeviceToHost), "copy counts");
    d_row_h[0] = 0;
    for (int i = 0; i < K; ++i) d_row_h[i+1] = d_row_h[i] + counts_h[i];
    int m = d_row_h.back();
    d_col_h.resize(m); d_val_h.resize(m);

    // Copy d_row_h to device for fill pass
    int *d_row_delta = nullptr;
    checkCudaStatus(cudaMalloc((void**)&d_row_delta, sizeof(int)*(K+1)), "malloc d_row_delta");
    checkCudaStatus(cudaMemcpy(d_row_delta, d_row_h.data(), sizeof(int)*(K+1), cudaMemcpyHostToDevice), "copy d_row_delta");

    int *d_col_delta = nullptr; double *d_val_delta = nullptr;
    checkCudaStatus(cudaMalloc((void**)&d_col_delta, sizeof(int)*m), "malloc d_col_delta");
    checkCudaStatus(cudaMalloc((void**)&d_val_delta, sizeof(double)*m), "malloc d_val_delta");

    k_fill_delta_csr<<<gridDelta, block>>>(d_row, d_col, d_val, d_delta_ids, d_delta_pos, d_row_delta, K, tau, d_col_delta, d_val_delta);
    checkCudaStatus(cudaDeviceSynchronize(), "fill delta csr");

    // Copy back results
    checkCudaStatus(cudaMemcpy(d_col_h.data(), d_col_delta, sizeof(int)*m, cudaMemcpyDeviceToHost), "copy d_col_h");
    checkCudaStatus(cudaMemcpy(d_val_h.data(), d_val_delta, sizeof(double)*m, cudaMemcpyDeviceToHost), "copy d_val_h");
    checkCudaStatus(cudaMemcpy(delta_pos_h.data(), d_delta_pos, sizeof(int)*n, cudaMemcpyDeviceToHost), "copy delta_pos_h");

    // cleanup
    cudaFree(d_row); cudaFree(d_col); cudaFree(d_val);
    cudaFree(d_delta_ids); cudaFree(d_delta_pos); cudaFree(d_counts);
    cudaFree(d_row_delta); cudaFree(d_col_delta); cudaFree(d_val_delta);
}

// --------------------------------------------------
// Step 2: Hook-and-Jump CC on Delta-CSR (GPU)
// --------------------------------------------------
// kernels
#include "../kernels/cc_hook_jump.cuh"

// --------------------------------------------------
// Propagation kernel (Step 4)
// --------------------------------------------------
// kernels
#include "../kernels/propagate.cuh"
// --------------------------------------------------
// Component init kernels (Step 3)
// --------------------------------------------------
#include "../kernels/comp_init.cuh"

static int runCCOnDevice(const vector<int>& d_row_h, const vector<int>& d_col_h, int K, vector<int>& node_comp_out) {
    if (K == 0) return 0;
    int *d_row = nullptr, *d_col = nullptr, *parent = nullptr, *d_changed = nullptr;
    checkCudaStatus(cudaMalloc((void**)&d_row, sizeof(int) * (K+1)), "malloc d_row");
    checkCudaStatus(cudaMalloc((void**)&d_col, sizeof(int) * d_col_h.size()), "malloc d_col");
    checkCudaStatus(cudaMemcpy(d_row, d_row_h.data(), sizeof(int) * (K+1), cudaMemcpyHostToDevice), "copy d_row");
    checkCudaStatus(cudaMemcpy(d_col, d_col_h.data(), sizeof(int) * d_col_h.size(), cudaMemcpyHostToDevice), "copy d_col");
    checkCudaStatus(cudaMalloc((void**)&parent, sizeof(int) * K), "malloc parent");
    checkCudaStatus(cudaMalloc((void**)&d_changed, sizeof(int)), "malloc changed");
    // init parent[i]=i
    vector<int> parent_h(K); std::iota(parent_h.begin(), parent_h.end(), 0);
    checkCudaStatus(cudaMemcpy(parent, parent_h.data(), sizeof(int) * K, cudaMemcpyHostToDevice), "copy parent init");

    dim3 block(256), grid((K + block.x - 1) / block.x);
    for (int it = 0; it < 32; ++it) { // cap iterations
        int zero = 0; checkCudaStatus(cudaMemcpy(d_changed, &zero, sizeof(int), cudaMemcpyHostToDevice), "reset changed");
        cc_hook_edges_kernel<<<grid, block>>>(d_row, d_col, K, parent, d_changed);
        checkCudaStatus(cudaDeviceSynchronize(), "hook sync");
        cc_jump_kernel<<<grid, block>>>(K, parent);
        checkCudaStatus(cudaDeviceSynchronize(), "jump sync");
        int changed = 0; checkCudaStatus(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost), "get changed");
        if (!changed) break;
    }

    checkCudaStatus(cudaMemcpy(parent_h.data(), parent, sizeof(int) * K, cudaMemcpyDeviceToHost), "copy parent out");
    // Compress and count unique roots, and output compact comp ids
    for (int i = 0; i < K; ++i) { while (parent_h[i] != parent_h[parent_h[i]]) parent_h[i] = parent_h[parent_h[i]]; }
    std::unordered_map<int,int> root_to_comp; root_to_comp.reserve(K*2);
    int comp_count = 0;
    node_comp_out.assign(K, -1);
    for (int i = 0; i < K; ++i) {
        int r = parent_h[i];
        auto it = root_to_comp.find(r);
        if (it == root_to_comp.end()) {
            int cid = comp_count++;
            root_to_comp.emplace(r, cid);
            node_comp_out[i] = cid;
        } else {
            node_comp_out[i] = it->second;
        }
    }

    cudaFree(d_row); cudaFree(d_col); cudaFree(parent); cudaFree(d_changed);
    return comp_count;
}

int main(int argc, char** argv) {
    Options opt = parseArgs(argc, argv);

    // Load base CSR
    vector<int> row = readIntVector(opt.csr_row);
    vector<int> col = readIntVector(opt.csr_col);
    vector<double> val = readDoubleVector(opt.csr_val);
    int n = static_cast<int>(row.size()) - 1;
    if ((int)col.size() != row.back() || (int)val.size() != row.back()) {
        std::cerr << "CSR size mismatch\n"; return 1;
    }

    // Load seeds
    vector<int> L0 = opt.l0_file.empty() ? vector<int>{} : readIntVector(opt.l0_file);
    vector<int> L1 = opt.l1_file.empty() ? vector<int>{} : readIntVector(opt.l1_file);
    vector<uint8_t> isL0(n, 0), isL1(n, 0);
    for (int v : L0) if (v>=0 && v<n) isL0[v]=1; for (int v : L1) if (v>=0 && v<n) isL1[v]=1;

    // Load delta IDs
    vector<int> delta_ids = readIntVector(opt.delta_file);
    // Filter invalids
    delta_ids.erase(std::remove_if(delta_ids.begin(), delta_ids.end(), [&](int v){return v<0 || v>=n;}), delta_ids.end());
    int K = static_cast<int>(delta_ids.size());
    cout << "Loaded graph n=" << n << ", nnz=" << col.size() << ", delta K=" << K << endl;

    // Optional: clamp a fraction of delta as new GT seeds
    if (K > 0 && opt.delta_labeled_pct > 0.0) {
        int Kgt = static_cast<int>(opt.delta_labeled_pct * static_cast<double>(K) + 0.5);
        Kgt = std::max(0, std::min(Kgt, K));
        if (Kgt > 0) {
            vector<int> gt_ids(delta_ids.begin(), delta_ids.begin() + Kgt);
            if (!opt.delta_labels_file.empty()) {
                vector<int> gt_lbls = readIntVector(opt.delta_labels_file);
                if ((int)gt_lbls.size() < Kgt) gt_lbls.resize(Kgt, 0);
                for (int i = 0; i < Kgt; ++i) {
                    int g = gt_ids[i]; int lbl = gt_lbls[i] ? 1 : 0;
                    if (lbl) { isL1[g] = 1; isL0[g] = 0; }
                    else { isL0[g] = 1; isL1[g] = 0; }
                }
            } else {
                // Balanced policy: first half -> L1, rest -> L0 (customize if needed)
                int half = Kgt / 2;
                for (int i = 0; i < Kgt; ++i) {
                    int g = gt_ids[i]; bool lbl1 = (i < half);
                    if (lbl1) { isL1[g] = 1; isL0[g] = 0; }
                    else { isL0[g] = 1; isL1[g] = 0; }
                }
            }
            cout << "Delta GT clamped: " << Kgt << " nodes (pct=" << opt.delta_labeled_pct << ")" << endl;
        }
    }

    // Step 1: Build Delta CSR on GPU (threshold tau)
    vector<int> d_row, d_col; vector<double> d_val; vector<int> delta_pos;
    buildDeltaCsrGpu(row, col, val, delta_ids, n, opt.tau, d_row, d_col, d_val, delta_pos);
    cout << "Delta CSR: K=" << K << ", m_delta=" << d_col.size() << " (tau=" << opt.tau << ")" << endl;

    // Step 2: Connected components via hook-and-jump on GPU
    vector<int> node_comp;
    int num_comp = runCCOnDevice(d_row, d_col, K, node_comp);
    cout << "Delta components (hook&jump): " << num_comp << endl;

    // Copy base CSR to device (used by Step 3 and Step 4)
    int *d_row_g = nullptr, *d_col_g = nullptr; double *d_val_g = nullptr;
    checkCudaStatus(cudaMalloc((void**)&d_row_g, sizeof(int)*(n+1)), "malloc d_row_g");
    checkCudaStatus(cudaMalloc((void**)&d_col_g, sizeof(int)*col.size()), "malloc d_col_g");
    checkCudaStatus(cudaMalloc((void**)&d_val_g, sizeof(double)*val.size()), "malloc d_val_g");
    checkCudaStatus(cudaMemcpy(d_row_g, row.data(), sizeof(int)*(n+1), cudaMemcpyHostToDevice), "copy row_g");
    checkCudaStatus(cudaMemcpy(d_col_g, col.data(), sizeof(int)*col.size(), cudaMemcpyHostToDevice), "copy col_g");
    checkCudaStatus(cudaMemcpy(d_val_g, val.data(), sizeof(double)*val.size(), cudaMemcpyHostToDevice), "copy val_g");

    // -----------------------------
    // Step 3: Component label init (GPU)
    // -----------------------------
    // Prepare device arrays for comp init
    int *d_node_comp = nullptr, *d_delta_ids_for_init = nullptr;
    checkCudaStatus(cudaMalloc((void**)&d_node_comp, sizeof(int)*K), "malloc node_comp");
    checkCudaStatus(cudaMemcpy(d_node_comp, node_comp.data(), sizeof(int)*K, cudaMemcpyHostToDevice), "copy node_comp");
    checkCudaStatus(cudaMalloc((void**)&d_delta_ids_for_init, sizeof(int)*K), "malloc delta_ids for init");
    checkCudaStatus(cudaMemcpy(d_delta_ids_for_init, delta_ids.data(), sizeof(int)*K, cudaMemcpyHostToDevice), "copy delta ids for init");

    unsigned char *d_isL0_u8 = nullptr, *d_isL1_u8 = nullptr;
    checkCudaStatus(cudaMalloc((void**)&d_isL0_u8, sizeof(unsigned char)*n), "malloc isL0_u8");
    checkCudaStatus(cudaMalloc((void**)&d_isL1_u8, sizeof(unsigned char)*n), "malloc isL1_u8");
    checkCudaStatus(cudaMemcpy(d_isL0_u8, isL0.data(), sizeof(unsigned char)*n, cudaMemcpyHostToDevice), "copy isL0 to u8");
    checkCudaStatus(cudaMemcpy(d_isL1_u8, isL1.data(), sizeof(unsigned char)*n, cudaMemcpyHostToDevice), "copy isL1 to u8");

    double *d_comp_sum0 = nullptr, *d_comp_sum1 = nullptr, *d_comp_label = nullptr;
    checkCudaStatus(cudaMalloc((void**)&d_comp_sum0, sizeof(double)*num_comp), "malloc comp_sum0");
    checkCudaStatus(cudaMalloc((void**)&d_comp_sum1, sizeof(double)*num_comp), "malloc comp_sum1");
    checkCudaStatus(cudaMalloc((void**)&d_comp_label, sizeof(double)*num_comp), "malloc comp_label");
    checkCudaStatus(cudaMemset(d_comp_sum0, 0, sizeof(double)*num_comp), "zero comp_sum0");
    checkCudaStatus(cudaMemset(d_comp_sum1, 0,  sizeof(double)*num_comp), "zero comp_sum1");

    {
        dim3 block(256), gridK((K + block.x - 1) / block.x);
        k_comp_accumulate_seed_weights<<<gridK, block>>>(d_row_g, d_col_g, d_val_g,
                                                         d_delta_ids_for_init, d_node_comp,
                                                         d_isL0_u8, d_isL1_u8, K,
                                                         d_comp_sum0, d_comp_sum1);
        checkCudaStatus(cudaDeviceSynchronize(), "comp accumulate");
        dim3 gridC((num_comp + block.x - 1) / block.x);
        k_comp_compute_labels<<<gridC, block>>>(d_comp_sum0, d_comp_sum1, num_comp, d_comp_label);
        checkCudaStatus(cudaDeviceSynchronize(), "comp compute labels");
    }

    // Initialize label vector: seeds clamped on host for reuse, and assign Δ values on device
    vector<double> L_h(n, 0.5);
    for (int v : L0) if (v>=0 && v<n) L_h[v] = 0.0;
    for (int v : L1) if (v>=0 && v<n) L_h[v] = 1.0;

    // d_row_g/d_col_g/d_val_g are already prepared above

    uint8_t *d_isL0 = nullptr, *d_isL1 = nullptr, *d_active = nullptr, *d_next_active = nullptr;
    checkCudaStatus(cudaMalloc((void**)&d_isL0, sizeof(uint8_t)*n), "malloc isL0");
    checkCudaStatus(cudaMalloc((void**)&d_isL1, sizeof(uint8_t)*n), "malloc isL1");
    checkCudaStatus(cudaMemcpy(d_isL0, isL0.data(), sizeof(uint8_t)*n, cudaMemcpyHostToDevice), "copy isL0");
    checkCudaStatus(cudaMemcpy(d_isL1, isL1.data(), sizeof(uint8_t)*n, cudaMemcpyHostToDevice), "copy isL1");
    checkCudaStatus(cudaMalloc((void**)&d_active, sizeof(uint8_t)*n), "malloc active");
    checkCudaStatus(cudaMalloc((void**)&d_next_active, sizeof(uint8_t)*n), "malloc next_active");
    checkCudaStatus(cudaMemset(d_active, 0, sizeof(uint8_t)*n), "memset active");
    checkCudaStatus(cudaMemset(d_next_active, 0, sizeof(uint8_t)*n), "memset next_active");

    double *d_L = nullptr;
    checkCudaStatus(cudaMalloc((void**)&d_L, sizeof(double)*n), "malloc L");
    checkCudaStatus(cudaMemcpy(d_L, L_h.data(), sizeof(double)*n, cudaMemcpyHostToDevice), "copy L");
    // Assign component labels to delta nodes on device
    if (K > 0 && num_comp > 0) {
        dim3 block2(256), gridK2((K + block2.x - 1) / block2.x);
        k_assign_delta_labels<<<gridK2, block2>>>(d_delta_ids_for_init, d_node_comp, d_comp_label, K, d_L);
        checkCudaStatus(cudaDeviceSynchronize(), "assign delta labels");
    }

    // Mark frontier: delta and their neighbors
    {
        vector<uint8_t> active_h(n, 0);
        for (int i = 0; i < K; ++i) { int g = delta_ids[i]; if (!isL0[g] && !isL1[g]) active_h[g] = 1; }
        for (int i = 0; i < K; ++i) {
            int g = delta_ids[i]; int s = row[g], e = row[g+1];
            for (int p = s; p < e; ++p) { int v = col[p]; if (!isL0[v] && !isL1[v]) active_h[v] = 1; }
        }
        checkCudaStatus(cudaMemcpy(d_active, active_h.data(), sizeof(uint8_t)*n, cudaMemcpyHostToDevice), "copy active");
    }

    // -----------------------------
    // Step 4: Propagation kernel(s)
    // -----------------------------
    // device flag for activity
    int *d_any = nullptr; checkCudaStatus(cudaMalloc((void**)&d_any, sizeof(int)), "malloc any");

    // Iterate
    {
        dim3 block(256), grid((n + block.x - 1) / block.x);
        for (int it = 0; it < opt.max_iters; ++it) {
            checkCudaStatus(cudaMemset(d_next_active, 0, sizeof(uint8_t)*n), "zero next_active");
            int zero = 0; checkCudaStatus(cudaMemcpy(d_any, &zero, sizeof(int), cudaMemcpyHostToDevice), "zero any");
            propagate_kernel<<<grid, block>>>(d_row_g, d_col_g, d_val_g, d_isL0, d_isL1, d_L, d_active, d_next_active, d_any, opt.delta_threshold, n);
            checkCudaStatus(cudaDeviceSynchronize(), "prop sync");
            int any = 0; checkCudaStatus(cudaMemcpy(&any, d_any, sizeof(int), cudaMemcpyDeviceToHost), "copy any");
            if (!any) break;
            // swap active
            checkCudaStatus(cudaMemcpy(d_active, d_next_active, sizeof(uint8_t)*n, cudaMemcpyDeviceToDevice), "swap active");
        }
    }

    // Copy back labels (optional print)
    checkCudaStatus(cudaMemcpy(L_h.data(), d_L, sizeof(double)*n, cudaMemcpyDeviceToHost), "copy labels back");
    cout << "proposed: finished propagation. sample labels: ";
    for (int i = 0; i < std::min(n, 10); ++i) cout << L_h[i] << ' '; cout << endl;

    // Cleanup
    cudaFree(d_row_g); cudaFree(d_col_g); cudaFree(d_val_g);
    cudaFree(d_isL0); cudaFree(d_isL1); cudaFree(d_active); cudaFree(d_next_active); cudaFree(d_L); cudaFree(d_any);
    // Step 3 temporaries
    if (K > 0) { cudaFree(d_node_comp); cudaFree(d_delta_ids_for_init); }
    cudaFree(d_isL0_u8); cudaFree(d_isL1_u8);
    cudaFree(d_comp_sum0); cudaFree(d_comp_sum1); cudaFree(d_comp_label);

    return 0;
}


