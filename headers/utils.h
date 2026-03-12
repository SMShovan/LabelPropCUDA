/**
 * @file utils.h
 * @brief Utility functions for CLI parsing, timing, and file I/O.
 *
 * Provides:
 *   - Command-line argument parsing into DynLPConfig
 *   - CUDA event-based timer for kernel profiling
 *   - CSV export for timing and accuracy results
 *   - Directory creation helpers
 */

#ifndef UTILS_H
#define UTILS_H

#include "graphTypes.h"
#include <string>

/**
 * @brief Parse command-line arguments into a DynLPConfig structure.
 *
 * Supported flags:
 *   --totalNodes N       Total number of vertices (default: 50000)
 *   --numBatches N       Number of dynamic batches (default: 10)
 *   --avgDegree  N       Average degree for graph generation (default: 5)
 *   --tau        F       Sparsification threshold (default: auto)
 *   --delta      F       Convergence threshold (default: 0.0001)
 *   --maxIter    N       Maximum propagation iterations (default: 100000)
 *   --gtFraction F       Ground-truth fraction per batch (default: 0.01)
 *   --seed       N       RNG seed (default: 42)
 *   --output     DIR     Output directory (default: "output/")
 *   --validate           Run IrLP baseline comparison (default: true)
 *   --noValidate         Skip IrLP baseline
 *   --verbose            Print per-batch details (default: true)
 *   --quiet              Suppress per-batch details
 *
 * @param argc  Argument count from main()
 * @param argv  Argument vector from main()
 * @param config Output configuration structure
 * @return true on success, false if arguments are invalid
 */
bool parseArguments(int argc, char *argv[], DynLPConfig &config);

/**
 * @brief Print the configuration to stdout.
 */
void printConfig(const DynLPConfig &config);

/**
 * @brief Ensure a directory exists, creating it recursively if needed.
 *
 * @param dirPath Path to the directory
 * @return true on success
 */
bool ensureDirectory(const std::string &dirPath);

/**
 * @brief Write a vector of labels to a text file.
 *
 * Format: one line per vertex "vertexId label\n"
 *
 * @param path    Output file path
 * @param labels  Label vector (index = vertex ID)
 * @param numNodes Number of vertices
 * @return true on success
 */
bool writeLabelsToFile(const std::string &path, const float *labels,
                       int numNodes);

/**
 * @brief Write timing and accuracy results to a CSV file.
 *
 * @param path          Output CSV path
 * @param batchTimes    Time in ms per batch
 * @param batchAccuracy Accuracy per batch (fraction of correct predictions)
 * @param numBatches    Number of batches
 * @return true on success
 */
bool writeResultsCSV(const std::string &path, const float *batchTimes,
                     const float *batchAccuracy, int numBatches);

#endif // UTILS_H
