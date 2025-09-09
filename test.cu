// combined_async.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

//---------------------------------------------------------
// CUDA Error Checking Macro and Function
//---------------------------------------------------------
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)  {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

//---------------------------------------------------------
// CUDA Kernel: Compute Row Sum for a Tile
//---------------------------------------------------------
/*
 * For a tile of size tileRows x tileCols (stored in row-major order),
 * each thread computes the sum for one row of the tile.
 */
__global__ void rowSumKernel(const float *tile, float *partialRowSum,
                             int tileRows, int tileCols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < tileRows) {
        float sum = 0.0f;
        for (int col = 0; col < tileCols; col++) {
            sum += tile[row * tileCols + col];
        }
        partialRowSum[row] = sum;
    }
}

//---------------------------------------------------------
// Function: Create Matrix File (matrix.bin)
//---------------------------------------------------------
/*
 * This function creates a binary file with an N x N matrix (row-major order)
 * and then appends its row-sum vector. The matrix is generated as:
 *
 *    matrix[i][j] = i + j + 1,
 *
 * so that the row sum for row i is:
 *
 *    rowSum[i] = sum_{j=0}^{N-1}(i+j+1).
 */
void createMatrixFile(const char *filename, int N) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error opening file %s for writing.\n", filename);
        exit(EXIT_FAILURE);
    }

    float *matrix = (float*)malloc(N * N * sizeof(float));
    if (!matrix) {
        fprintf(stderr, "Error allocating memory for the matrix.\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    float *rowSum = (float*)malloc(N * sizeof(float));
    if (!rowSum) {
        fprintf(stderr, "Error allocating memory for rowSum.\n");
        free(matrix);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Fill the matrix and compute row sums.
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            float value = (float)(i + j + 1);
            matrix[i * N + j] = value;
            sum += value;
        }
        rowSum[i] = sum;
    }

    // Write matrix data.
    size_t written = fwrite(matrix, sizeof(float), N * N, fp);
    if (written != (size_t)(N * N)) {
        fprintf(stderr, "Error writing matrix to file.\n");
        free(matrix);
        free(rowSum);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Append the ground-truth row sums.
    written = fwrite(rowSum, sizeof(float), N, fp);
    if (written != (size_t)N) {
        fprintf(stderr, "Error writing row-sum vector to file.\n");
        free(matrix);
        free(rowSum);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    fclose(fp);
    free(matrix);
    free(rowSum);
    printf("File \"%s\" created successfully.\n", filename);
    printf("Matrix size: %d x %d, and row sums appended at the end.\n", N, N);
}

//---------------------------------------------------------
// Main Function
//---------------------------------------------------------
int main(void)
{
    //===========================================================================
    // PARAMETERS
    //===========================================================================
    int N = 8;                // Full matrix size (N x N). Adjust as needed.
    const int TILE_SIZE = 4;  // Tile size (TILE_SIZE x TILE_SIZE). Assume TILE_SIZE <= N.
    const char *filename = "matrix.bin";

    //===========================================================================
    // STEP 1: Create the test matrix file.
    //===========================================================================
    createMatrixFile(filename, N);

    //===========================================================================
    // STEP 2: Allocate global accumulator for the row sums (size N).
    //===========================================================================
    float *globalRowSum = (float*)malloc(N * sizeof(float));
    if (!globalRowSum) {
        fprintf(stderr, "Failed to allocate memory for globalRowSum.\n");
        return -1;
    }
    for (int i = 0; i < N; i++) {
        globalRowSum[i] = 0.0f;
    }

    //===========================================================================
    // STEP 3: Prepare for tiled processing with asynchronous transfers.
    //===========================================================================
    // We use 2 streams with double-buffering.
    cudaStream_t streams[2];
    cudaCheckError(cudaStreamCreate(&streams[0]));
    cudaCheckError(cudaStreamCreate(&streams[1]));

    // For each stream, allocate pinned host buffers and device buffers.
    // Maximum tile size is TILE_SIZE x TILE_SIZE (even if last tiles are smaller).
    float *h_tile[2], *h_partial[2];
    float *d_tile[2], *d_partial[2];
    for (int s = 0; s < 2; s++) {
        cudaCheckError(cudaHostAlloc((void**)&h_tile[s],
                                     TILE_SIZE * TILE_SIZE * sizeof(float),
                                     cudaHostAllocDefault));
        cudaCheckError(cudaHostAlloc((void**)&h_partial[s],
                                     TILE_SIZE * sizeof(float),
                                     cudaHostAllocDefault));
        cudaCheckError(cudaMalloc((void**)&d_tile[s],
                                  TILE_SIZE * TILE_SIZE * sizeof(float)));
        cudaCheckError(cudaMalloc((void**)&d_partial[s],
                                  TILE_SIZE * sizeof(float)));
    }
    // We'll keep track of the parameters for the last tile launched on each stream
    // so that when we reuse that stream we can accumulate its partial results.
    int prevTileRow[2] = { -1, -1 };   // Starting row index of the last tile in that stream.
    int prevTileRows[2] = { 0, 0 };      // Number of rows in the last tile for that stream.
    // A flag to know if a stream has already been used.
    bool streamUsed[2] = { false, false };

    // Open the file to read the matrix (only the matrix part, not the appended row sums).
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open file %s for reading.\n", filename);
        return -1;
    }

    //===========================================================================
    // STEP 4: Process the matrix in tiles using asynchronous operations.
    //===========================================================================
    int tileCounter = 0; // Counts total tiles processed.
    // Loop over tile rows.
    for (int tileRow = 0; tileRow < N; tileRow += TILE_SIZE) {
        // Determine number of rows in this tile.
        int currentTileRows = (tileRow + TILE_SIZE <= N) ? TILE_SIZE : (N - tileRow);

        // Loop over tile columns.
        for (int tileCol = 0; tileCol < N; tileCol += TILE_SIZE) {
            // Determine number of columns in this tile.
            int currentTileCols = (tileCol + TILE_SIZE <= N) ? TILE_SIZE : (N - tileCol);

            // Select one of the 2 streams (double buffering).
            int streamIdx = tileCounter % 2;

            // If this stream has been used before, synchronize and accumulate its prior tile.
            if (streamUsed[streamIdx]) {
                cudaCheckError(cudaStreamSynchronize(streams[streamIdx]));
                // Accumulate the previous tile's partial row sums into the global row sum.
                // The previous tile started at row index prevTileRow[streamIdx] and had prevTileRows[streamIdx] rows.
                for (int i = 0; i < prevTileRows[streamIdx]; i++) {
                    globalRowSum[prevTileRow[streamIdx] + i] += h_partial[streamIdx][i];
                }
            } else {
                streamUsed[streamIdx] = true;
            }

            // Read the current tile from file.
            // Because the full matrix is stored in row-major order, each row of the tile is read separately.
            for (int i = 0; i < currentTileRows; i++) {
                // Compute the file offset for row (tileRow + i) and column tileCol.
                long offset = (long)((tileRow + i) * N + tileCol) * sizeof(float);
                if (fseek(fp, offset, SEEK_SET) != 0) {
                    fprintf(stderr, "fseek failed.\n");
                    fclose(fp);
                    return -1;
                }
                size_t itemsRead = fread(&h_tile[streamIdx][i * currentTileCols],
                                           sizeof(float),
                                           currentTileCols,
                                           fp);
                if (itemsRead != (size_t)currentTileCols) {
                    fprintf(stderr, "fread failed: expected %d, got %zu\n",
                            currentTileCols, itemsRead);
                    fclose(fp);
                    return -1;
                }
            }

            // Asynchronously copy the tile from host (pinned) to device.
            size_t tileSizeBytes = currentTileRows * currentTileCols * sizeof(float);
            cudaCheckError(cudaMemcpyAsync(d_tile[streamIdx], h_tile[streamIdx],
                                           tileSizeBytes, cudaMemcpyHostToDevice,
                                           streams[streamIdx]));

            // Launch the kernel asynchronously.
            int threadsPerBlock = 256;
            int blocksPerGrid = (currentTileRows + threadsPerBlock - 1) / threadsPerBlock;
            rowSumKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[streamIdx]>>>(
                d_tile[streamIdx], d_partial[streamIdx], currentTileRows, currentTileCols);
            cudaCheckError(cudaGetLastError());

            // Asynchronously copy the partial row sums back to host (pinned).
            size_t partialSizeBytes = currentTileRows * sizeof(float);
            cudaCheckError(cudaMemcpyAsync(h_partial[streamIdx], d_partial[streamIdx],
                                           partialSizeBytes, cudaMemcpyDeviceToHost,
                                           streams[streamIdx]));

            // Save the parameters of the current tile for later accumulation.
            prevTileRow[streamIdx] = tileRow;
            prevTileRows[streamIdx] = currentTileRows;

            tileCounter++;
        } // end tileCol loop
    } // end tileRow loop

    // After processing all tiles, synchronize and accumulate any pending results from each stream.
    for (int s = 0; s < 2; s++) {
        if (streamUsed[s]) {
            cudaCheckError(cudaStreamSynchronize(streams[s]));
            for (int i = 0; i < prevTileRows[s]; i++) {
                globalRowSum[prevTileRow[s] + i] += h_partial[s][i];
            }
        }
    }

    fclose(fp);

    //===========================================================================
    // STEP 5: Read the ground–truth row sums from the file and compare.
    //===========================================================================
    FILE* fp2 = fopen(filename, "rb");
    if (!fp2) {
        fprintf(stderr, "Failed to open file %s for reading ground truth row sums.\n", filename);
        free(globalRowSum);
        return -1;
    }
    // Ground–truth row sums are stored after the N x N matrix.
    if (fseek(fp2, N * N * sizeof(float), SEEK_SET) != 0) {
        fprintf(stderr, "fseek failed while reading ground truth row sums.\n");
        fclose(fp2);
        free(globalRowSum);
        return -1;
    }
    float *groundTruthRowSum = (float*)malloc(N * sizeof(float));
    if (!groundTruthRowSum) {
        fprintf(stderr, "Failed to allocate memory for groundTruthRowSum.\n");
        fclose(fp2);
        free(globalRowSum);
        return -1;
    }
    size_t itemsRead = fread(groundTruthRowSum, sizeof(float), N, fp2);
    if (itemsRead != (size_t)N) {
        fprintf(stderr, "Failed to read all ground truth row sums.\n");
        fclose(fp2);
        free(globalRowSum);
        free(groundTruthRowSum);
        return -1;
    }
    fclose(fp2);

    //===========================================================================
    // STEP 6: Compare and print the results.
    //===========================================================================
    printf("\nComparison of computed row sums and ground truth:\n");
    printf("Row\tComputedSum\tGroundTruth\tDifference\n");
    for (int i = 0; i < N; i++) {
        float diff = globalRowSum[i] - groundTruthRowSum[i];
        if (diff < 0) diff = -diff;
        printf("%d\t%f\t%f\t%f\n", i, globalRowSum[i], groundTruthRowSum[i], diff);
    }

    //===========================================================================
    // Cleanup: Free resources.
    //===========================================================================
    free(globalRowSum);
    free(groundTruthRowSum);
    for (int s = 0; s < 2; s++) {
        cudaFree(d_tile[s]);
        cudaFree(d_partial[s]);
        cudaFreeHost(h_tile[s]);
        cudaFreeHost(h_partial[s]);
        cudaStreamDestroy(streams[s]);
    }

    return 0;
}
