# ==============================================================================
# DynLP Makefile
# ==============================================================================
#
# Builds the DynLP (Dynamic Label Propagation) CUDA application.
#
# Usage:
#   make              Build the dynLP executable
#   make clean        Remove all build artifacts
#   make run          Build and run with default parameters
#   make run-small    Build and run with a small test case
#   make run-large    Build and run with a larger test case
#
# Configuration:
#   Override GPU architecture:  make ARCH=sm_80
#   Override compiler:          make NVCC=/usr/local/cuda/bin/nvcc
#   Enable debug mode:          make DEBUG=1
#
# ==============================================================================

# Compiler
NVCC        ?= nvcc

# GPU architecture (override for your hardware)
# Common values: sm_70 (V100), sm_80 (A100), sm_90 (H100)
ARCH        ?= sm_70

# Directories
SRC_DIR     := src
HDR_DIR     := headers
OBJ_DIR     := obj
BIN_DIR     := bin

# Source files
SRCS        := $(wildcard $(SRC_DIR)/*.cu)
OBJS        := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRCS))

# Target executable
TARGET      := $(BIN_DIR)/dynLP

# Compiler flags
NVCC_FLAGS  := -std=c++17 -arch=$(ARCH) -I$(HDR_DIR)
NVCC_FLAGS  += -Xcompiler -Wall

# Link flags
LDFLAGS     :=

# Debug vs Release
ifdef DEBUG
  NVCC_FLAGS += -G -g -O0 -DDEBUG
else
  NVCC_FLAGS += -O3 --use_fast_math
endif

# ==============================================================================
# Build Rules
# ==============================================================================

.PHONY: all clean run run-small run-large directories

all: directories $(TARGET)

directories:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $@ $(LDFLAGS)
	@echo ""
	@echo "Build complete: $(TARGET)"
	@echo ""

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(wildcard $(HDR_DIR)/*.h)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "Clean complete."

# ==============================================================================
# Run Targets
# ==============================================================================

# Default: 50K nodes, 10 batches, degree 5
run: all
	@mkdir -p output
	$(TARGET) --totalNodes 50000 --numBatches 10 --avgDegree 5 \
	          --delta 0.0001 --seed 42 --output output --verbose

# Small test: 5K nodes, 5 batches (fast, for quick validation)
run-small: all
	@mkdir -p output
	$(TARGET) --totalNodes 5000 --numBatches 5 --avgDegree 5 \
	          --delta 0.0001 --seed 42 --output output --verbose

# Large test: 500K nodes, 10 batches (for performance benchmarking)
run-large: all
	@mkdir -p output
	$(TARGET) --totalNodes 500000 --numBatches 10 --avgDegree 5 \
	          --delta 0.0001 --seed 42 --output output --verbose

# Quick run without IrLP validation (faster)
run-fast: all
	@mkdir -p output
	$(TARGET) --totalNodes 50000 --numBatches 10 --avgDegree 5 \
	          --delta 0.0001 --seed 42 --output output --noValidate --verbose

# Varying average degrees (for paper-style experiments)
run-deg3: all
	@mkdir -p output
	$(TARGET) --totalNodes 50000 --numBatches 10 --avgDegree 3 \
	          --delta 0.0001 --seed 42 --output output --verbose

run-deg7: all
	@mkdir -p output
	$(TARGET) --totalNodes 50000 --numBatches 10 --avgDegree 7 \
	          --delta 0.0001 --seed 42 --output output --verbose
