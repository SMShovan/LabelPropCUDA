#pragma once
#include <vector>
#include <random>
#include <algorithm>

inline std::vector<double> generateErdosRenyiAdjacency(int n, double avg_degree, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> wdist(0.0, 1.0);
    std::uniform_real_distribution<double> udist(0.0, 1.0);
    std::vector<double> A(n * n, 0.0);
    if (n <= 1) return A;
    double p = std::max(0.0, std::min(1.0, avg_degree / static_cast<double>(n - 1)));
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (udist(rng) < p) {
                double w = wdist(rng);
                A[i * n + j] = w;
                A[j * n + i] = w;
            }
        }
    }
    return A;
}

inline std::vector<double> generateConnectedAdjacencyWithSpanningTree(int n, double avg_degree, unsigned seed) {
    std::vector<double> A(n * n, 0.0);
    if (n <= 1) return A;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> wdist(0.0, 1.0);
    std::uniform_int_distribution<int> parentDist;
    for (int v = 1; v < n; ++v) {
        parentDist = std::uniform_int_distribution<int>(0, v - 1);
        int p = parentDist(rng);
        double w = wdist(rng);
        A[p * n + v] = w;
        A[v * n + p] = w;
    }
    const double target_edges = std::max(0.0, (avg_degree * n) / 2.0);
    int current_edges = 0;
    for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) if (A[i * n + j] > 0.0) ++current_edges;
    const int total_pairs = (n * (n - 1)) / 2;
    int remaining_pairs = total_pairs - current_edges;
    double remaining_needed = std::max(0.0, target_edges - static_cast<double>(current_edges));
    double p_extra = (remaining_pairs > 0) ? std::min(1.0, remaining_needed / static_cast<double>(remaining_pairs)) : 0.0;
    std::uniform_real_distribution<double> udist(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (A[i * n + j] == 0.0 && udist(rng) < p_extra) {
                double w = wdist(rng);
                A[i * n + j] = w;
                A[j * n + i] = w;
            }
        }
    }
    return A;
}

inline void extendAdjacencyWithNewVertices(std::vector<double>& adjacency, int& num_vertices, int num_new_vertices, double avg_degree, unsigned seed, bool ensure_connected) {
    if (num_new_vertices <= 0) return;
    int oldN = num_vertices;
    int newN = num_vertices + num_new_vertices;
    std::vector<double> B(newN * newN, 0.0);
    for (int i = 0; i < oldN; ++i) std::copy_n(&adjacency[i * oldN], oldN, &B[i * newN]);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> wdist(0.0, 1.0);
    std::uniform_real_distribution<double> udist(0.0, 1.0);
    if (ensure_connected) {
        for (int v = oldN; v < newN; ++v) {
            std::uniform_int_distribution<int> parentDist(0, v - 1);
            int p = parentDist(rng);
            double w = wdist(rng);
            B[p * newN + v] = w; B[v * newN + p] = w;
        }
    }
    double p = (newN > 1) ? std::max(0.0, std::min(1.0, avg_degree / static_cast<double>(newN - 1))) : 0.0;
    for (int i = 0; i < newN; ++i) {
        for (int j = i + 1; j < newN; ++j) {
            if (B[i * newN + j] == 0.0 && udist(rng) < p) {
                double w = wdist(rng);
                B[i * newN + j] = w; B[j * newN + i] = w;
            }
        }
    }
    adjacency.swap(B);
    num_vertices = newN;
}


