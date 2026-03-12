# DynLP: Parallel Dynamic Batch Label Propagation on GPU

A CUDA implementation of the DynLP algorithm for efficient semi-supervised learning on dynamically evolving sparse graphs. Based on the paper *"DynLP: Parallel Dynamic Batch Update for Label Propagation in Semi-Supervised Learning"* (ICS 2026).

## Overview

DynLP performs **binary classification** on graph-structured data where:
- A small fraction of vertices have known labels (**ground truth**)
- New vertices arrive in **batches** over time
- Some existing vertices may be **deleted** per batch
- The goal is to predict labels for all unlabeled vertices efficiently

Instead of recomputing all labels from scratch after each batch (which is expensive), DynLP only updates the **affected** vertices using three steps:

1. **Change Adjustment & Sparsification** — Identify affected vertices, build a sparsified subgraph of new vertices, find connected components using the Shiloach-Vishkin algorithm
2. **Label Initialization** — Initialize new vertex labels using aggregate edge weights to ground-truth supernodes
3. **Iterative Propagation** — Update affected vertex labels using weighted neighborhood averaging until convergence

## Build

### Prerequisites

- NVIDIA GPU with compute capability >= 7.0 (e.g., V100, A100, H100)
- CUDA Toolkit >= 11.0
- C++17 compatible host compiler (GCC >= 9)

### Compile

```bash
# Default build (sm_70 architecture)
make

# For specific GPU architecture
make ARCH=sm_80     # A100
make ARCH=sm_90     # H100

# Debug build (with device-side debugging)
make DEBUG=1

# Clean build artifacts
make clean
```

## Run

### Quick Start

```bash
# Default: 50K nodes, 10 batches, average degree 5
make run

# Small test case (5K nodes, fast)
make run-small

# Large benchmark (500K nodes)
make run-large

# Without IrLP validation (faster)
make run-fast
```

### Custom Parameters

```bash
./bin/dynLP [options]
```

| Parameter       | Default   | Description                                      |
|-----------------|-----------|--------------------------------------------------|
| `--totalNodes`  | 50000     | Total number of vertices across all batches      |
| `--numBatches`  | 10        | Number of dynamic batches                        |
| `--avgDegree`   | 5         | Average degree for graph generation (3, 5, or 7) |
| `--tau`         | auto      | Sparsification threshold (auto = avg edge weight)|
| `--delta`       | 0.0001    | Convergence threshold for label propagation      |
| `--maxIter`     | 100000    | Maximum propagation iterations per batch         |
| `--gtFraction`  | 0.01      | Fraction of ground-truth vertices per batch (1%) |
| `--seed`        | 42        | Random number generator seed                     |
| `--output`      | output/   | Output directory                                 |
| `--validate`    | enabled   | Run IrLP baseline for accuracy comparison        |
| `--noValidate`  |           | Skip IrLP baseline (faster execution)            |
| `--verbose`     | enabled   | Print per-batch details                          |
| `--quiet`       |           | Only print final summary                         |

### Example

```bash
./bin/dynLP --totalNodes 100000 --numBatches 10 --avgDegree 5 \
            --delta 0.0001 --seed 42 --output results/ --verbose
```

## Output

All output files are written to the specified output directory:

| File                  | Description                                    |
|-----------------------|------------------------------------------------|
| `dynlp_labels.txt`   | Final predicted labels (vertex_id label)       |
| `results.csv`        | Per-batch timing (ms) and accuracy             |
| `irlp_labels.txt`    | IrLP baseline labels (if --validate enabled)   |
| `graph/csrRowPtr.txt` | Generated graph CSR row pointers              |
| `graph/csrColInd.txt` | Generated graph CSR column indices            |
| `graph/csrValues.txt` | Generated graph CSR edge weights (similarity) |
| `graph/csrTrueLabels.txt` | True binary labels for all vertices       |

### Label Format

Labels are fractional values in [0, 1]:
- Values close to 0.0 predict **class 0**
- Values close to 1.0 predict **class 1**
- A threshold of 0.5 maps fractional labels to binary predictions

## Algorithm Details

### Batch Composition

Each batch (after batch 0) contains:
- **90%** unlabeled new vertices
- **1%** ground-truth new vertices (label known)
- **9%** deletions of existing non-ground-truth vertices

Batch 0 is special: it contains only the initial ground-truth vertices.

### Graph Generation

The synthetic graph is generated using the Erdős-Rényi model:
- A random spanning tree ensures connectivity
- Random edges are added to reach the target average degree
- Edge weights represent similarity: same-class pairs get weights in [0.5, 1.0], cross-class pairs get [0.0, 0.5]

### Key Parameters

- **tau (sparsification threshold)**: Controls which edges are kept in the new-vertex subgraph. Higher tau = more sparsification = more connected components = better label initialization but potentially fewer edges for propagation. Default: average edge weight.

- **delta (convergence threshold)**: Controls when iterative propagation stops. Smaller delta = more iterations = higher accuracy but slower. Recommended: 0.0001 (from the paper's experiments).

### Issues Identified and Resolved

Several issues in the original paper's algorithm were identified and fixed:

1. **Division by zero in initialization** (Algorithm 2, Line 22): When a connected component has no edges to either ground-truth supernode, the formula's denominator is zero. Fixed by defaulting to 0.5 (neutral).

2. **Division by zero in propagation** (Line 28): When a vertex has no active neighbors, W_all = 0. Fixed by skipping the update for isolated vertices.

3. **Underspecified affected set**: The paper doesn't clarify that V_aff must include neighbors from the original adjacency before deletion. Fixed by capturing all neighbors before edge removal.

4. **One-sided convergence check** (Line 29): The paper checks F'_u - F_u > delta (unsigned), but labels can also decrease. Fixed by using |F'_u - F_u| > delta (absolute difference).

5. **Race condition in parallel updates** (Lines 22, 28): Concurrent read/write of labels during propagation. Fixed using double-buffering: read from old buffer, write to new buffer, swap after each iteration.

6. **Misleading initialization formula**: The paper's three-term expansion simplifies algebraically to W_L1 / (W_L0 + W_L1). We use the simplified form directly.

### CUDA Optimizations

- **Block-per-row execution model**: One thread block per vertex, threads cooperatively process neighbors with strided access
- **Shared memory reduction**: For edge-weight summation within each block
- **Double-buffered label updates**: Avoid read-write races without explicit synchronization
- **Coalesced memory access**: CSR rows stored contiguously in batch order
- **Async memory transfer**: Overlaps host-to-device copy with kernel execution via CUDA streams
- **Atomic operations**: atomicMin for connected component merging, atomicAdd for weight accumulation, atomicMax for convergence detection

## Project Structure

```
DynLP/
├── Makefile                        # Build system
├── README.md                       # This file
├── headers/
│   ├── graphTypes.h                # CSR graph, config, batch data types
│   ├── generateSparseGraph.h       # Graph generator interface
│   ├── generateBatches.h           # Batch generator interface
│   ├── sparsifySubgraph.h          # Step 1: Sparsification
│   ├── connectedComponents.h       # Step 1: Shiloach-Vishkin CC
│   ├── labelInitialization.h       # Step 2: Label init
│   ├── iterativePropagation.h      # Step 3: Iterative update
│   ├── dynLP.h                     # Main orchestrator
│   ├── validation.h                # IrLP baseline
│   └── utils.h                     # Utilities (CLI, I/O, timing)
├── src/
│   ├── main.cu                     # Entry point
│   ├── generateSparseGraph.cu      # Graph generation
│   ├── generateBatches.cu          # Batch creation
│   ├── sparsifySubgraph.cu         # CUDA: sparsification kernels
│   ├── connectedComponents.cu      # CUDA: Shiloach-Vishkin kernels
│   ├── labelInitialization.cu      # CUDA: label init kernels
│   ├── iterativePropagation.cu     # CUDA: propagation kernels
│   ├── dynLP.cu                    # Orchestrator
│   ├── validation.cu               # IrLP baseline
│   └── utils.cu                    # Utility implementations
├── data/                           # Generated graph data
├── output/                         # Execution results
└── tests/                          # Test cases
```

## Testing

### Varying Graph Size

```bash
# Small graph
./bin/dynLP --totalNodes 5000 --numBatches 5 --verbose

# Medium graph
./bin/dynLP --totalNodes 50000 --numBatches 10 --verbose

# Large graph
./bin/dynLP --totalNodes 500000 --numBatches 10 --verbose
```

### Varying Average Degree

```bash
make run-deg3    # Average degree 3 (sparser)
make run         # Average degree 5 (default)
make run-deg7    # Average degree 7 (denser)
```

### Varying Convergence Threshold

```bash
./bin/dynLP --delta 0.1     # Fast but less accurate
./bin/dynLP --delta 0.001   # Balanced
./bin/dynLP --delta 0.0001  # High accuracy (default)
./bin/dynLP --delta 0.00001 # Maximum accuracy
```

## References

- DynLP paper: "DynLP: Parallel Dynamic Batch Update for Label Propagation in Semi-Supervised Learning" (ICS 2026)
- Shiloach-Vishkin algorithm: Y. Shiloach and U. Vishkin, "An O(log n) parallel connectivity algorithm," J. Algorithms, 1982
- Label Propagation: X. Zhu et al., "Semi-supervised learning using gaussian fields and harmonic functions," ICML 2003
