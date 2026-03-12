/**
 * @file utils.cu
 * @brief Implementation of utility functions for CLI parsing, timing, I/O.
 *
 * See utils.h for full documentation of each function.
 */

#include "utils.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace std;
namespace fs = std::filesystem;

// ============================================================================
// CLI Argument Parsing
// ============================================================================

bool parseArguments(int argc, char *argv[], DynLPConfig &config) {
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];

    if (arg == "--totalNodes" && i + 1 < argc) {
      config.totalNodes = atoi(argv[++i]);
      if (config.totalNodes <= 0) {
        cout << "Error: --totalNodes must be positive.\n";
        return false;
      }
    } else if (arg == "--numBatches" && i + 1 < argc) {
      config.numBatches = atoi(argv[++i]);
      if (config.numBatches <= 0) {
        cout << "Error: --numBatches must be positive.\n";
        return false;
      }
    } else if (arg == "--avgDegree" && i + 1 < argc) {
      config.avgDegree = atoi(argv[++i]);
      if (config.avgDegree <= 0) {
        cout << "Error: --avgDegree must be positive.\n";
        return false;
      }
    } else if (arg == "--tau" && i + 1 < argc) {
      config.tau = atof(argv[++i]);
    } else if (arg == "--delta" && i + 1 < argc) {
      config.delta = atof(argv[++i]);
      if (config.delta <= 0.0f) {
        cout << "Error: --delta must be positive.\n";
        return false;
      }
    } else if (arg == "--maxIter" && i + 1 < argc) {
      config.maxIterations = atoi(argv[++i]);
    } else if (arg == "--gtFraction" && i + 1 < argc) {
      config.groundTruthFraction = atof(argv[++i]);
      if (config.groundTruthFraction <= 0.0f ||
          config.groundTruthFraction >= 1.0f) {
        cout << "Error: --gtFraction must be in (0, 1).\n";
        return false;
      }
    } else if (arg == "--seed" && i + 1 < argc) {
      config.seed = static_cast<unsigned int>(atoi(argv[++i]));
    } else if (arg == "--output" && i + 1 < argc) {
      config.outputDir = argv[++i];
    } else if (arg == "--validate") {
      config.validate = true;
    } else if (arg == "--noValidate") {
      config.validate = false;
    } else if (arg == "--verbose") {
      config.verbose = true;
    } else if (arg == "--quiet") {
      config.verbose = false;
    } else if (arg == "--help" || arg == "-h") {
      cout << "Usage: dynLP [options]\n"
           << "\nOptions:\n"
           << "  --totalNodes N    Total vertices across all batches "
              "(default: 50000)\n"
           << "  --numBatches N    Number of dynamic batches (default: 10)\n"
           << "  --avgDegree  N    Average degree for graph gen (default: 5)\n"
           << "  --tau        F    Sparsification threshold (default: auto)\n"
           << "  --delta      F    Convergence threshold (default: 0.0001)\n"
           << "  --maxIter    N    Max propagation iterations "
              "(default: 100000)\n"
           << "  --gtFraction F    Ground-truth fraction (default: 0.01)\n"
           << "  --seed       N    RNG seed (default: 42)\n"
           << "  --output     DIR  Output directory (default: output/)\n"
           << "  --validate        Run IrLP baseline comparison\n"
           << "  --noValidate      Skip IrLP baseline\n"
           << "  --verbose         Print per-batch details\n"
           << "  --quiet           Suppress per-batch details\n"
           << "  --help, -h        Show this help message\n";
      return false;
    } else {
      cout << "Error: Unknown argument: " << arg << "\n";
      cout << "Run with --help for usage information.\n";
      return false;
    }
  }

  // Ensure batch fractions sum to <= 1.0
  float totalFraction = config.groundTruthFraction +
                         config.unlabeledFraction + config.deletedFraction;
  if (totalFraction > 1.001f) {
    cout << "Error: Batch fractions (GT + unlabeled + deleted) exceed 1.0.\n";
    return false;
  }

  return true;
}

// ============================================================================
// Configuration Printing
// ============================================================================

void printConfig(const DynLPConfig &config) {
  cout << "\n"
       << "============================================================\n"
       << " DynLP Configuration\n"
       << "============================================================\n"
       << "  Total nodes:        " << config.totalNodes << "\n"
       << "  Number of batches:  " << config.numBatches << "\n"
       << "  Average degree:     " << config.avgDegree << "\n"
       << "  Tau (sparsify):     "
       << (config.tau < 0 ? "auto (avg edge weight)" : to_string(config.tau))
       << "\n"
       << "  Delta (converge):   " << config.delta << "\n"
       << "  Max iterations:     " << config.maxIterations << "\n"
       << "  GT fraction:        " << config.groundTruthFraction << "\n"
       << "  Unlabeled fraction: " << config.unlabeledFraction << "\n"
       << "  Deleted fraction:   " << config.deletedFraction << "\n"
       << "  Seed:               " << config.seed << "\n"
       << "  Output directory:   " << config.outputDir << "\n"
       << "  Validate (IrLP):    " << (config.validate ? "yes" : "no") << "\n"
       << "  Verbose:            " << (config.verbose ? "yes" : "no") << "\n"
       << "============================================================\n\n";
}

// ============================================================================
// Directory Helpers
// ============================================================================

bool ensureDirectory(const string &dirPath) {
  try {
    if (!fs::exists(dirPath)) {
      fs::create_directories(dirPath);
    }
    return true;
  } catch (const fs::filesystem_error &e) {
    cout << "Error: Could not create directory " << dirPath << ": " << e.what()
         << "\n";
    return false;
  }
}

// ============================================================================
// File I/O
// ============================================================================

bool writeLabelsToFile(const string &path, const float *labels, int numNodes) {
  fs::path filePath(path);
  if (!filePath.parent_path().empty()) {
    ensureDirectory(filePath.parent_path().string());
  }

  ofstream file(path);
  if (!file.is_open()) {
    cout << "Error: Could not open file for writing: " << path << "\n";
    return false;
  }

  for (int i = 0; i < numNodes; ++i) {
    file << i << " " << fixed << setprecision(6) << labels[i] << "\n";
  }

  return true;
}

bool writeResultsCSV(const string &path, const float *batchTimes,
                     const float *batchAccuracy, int numBatches) {
  fs::path filePath(path);
  if (!filePath.parent_path().empty()) {
    ensureDirectory(filePath.parent_path().string());
  }

  ofstream file(path);
  if (!file.is_open()) {
    cout << "Error: Could not open CSV file for writing: " << path << "\n";
    return false;
  }

  file << "batch,time_ms,accuracy\n";
  for (int i = 0; i < numBatches; ++i) {
    file << i << "," << fixed << setprecision(2) << batchTimes[i] << ","
         << fixed << setprecision(6) << batchAccuracy[i] << "\n";
  }

  return true;
}
