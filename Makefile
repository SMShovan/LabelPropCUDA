# Simple Makefile for building and running src/baseline.cu

# Usage examples:
#   make                # build baseline with defaults
#   make SM=80          # build for sm_80 (A100)
#   make run            # build then run ./baseline
#   make clean          # remove build artifacts

NVCC ?= nvcc
SM ?= 80
ARCH_FLAG ?= -arch=sm_$(SM)

SRC_DIR := src
TARGET := baseline
SRC := $(SRC_DIR)/baseline.cu

CXXFLAGS ?= -O3 -std=c++14 $(ARCH_FLAG) -Xcompiler -Wno-deprecated-declarations
LDFLAGS ?= -lcurand -lcublas -lcusolver -lcusparse

# Default arguments for `make run` (override with: make run ARGS="...")
ARGS ?= --n 10 --labeled 3 --avgdeg 5 --connected --delta 100 --delta-labeled-pct 0.1

.PHONY: all run clean help

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) $(ARGS)

.PHONY: run-baseline run-naive run-all

run-baseline: $(TARGET)
	./$(TARGET) $(ARGS)

run-naive: naive
	./naive $(ARGS)

run-all: $(TARGET) naive
	@echo "--- Running baseline ---"
	./$(TARGET) $(ARGS)
	@echo "\n--- Running naive ---"
	./naive $(ARGS)

clean:
	rm -f $(TARGET)
	rm -f naive

help:
	@echo "Targets:"
	@echo "  all     - build $(TARGET) (default)"
	@echo "  run     - build and run ./$(TARGET)"
	@echo "  clean   - remove build artifacts"
	@echo "Variables:"
	@echo "  NVCC    - nvcc compiler (default: nvcc)"
	@echo "  SM      - compute capability (e.g., 70, 75, 80, 90)"
	@echo "  ARCH_FLAG - override arch flag (default: -arch=sm_$(SM))"
	@echo "  CXXFLAGS - extra compile flags (default includes $(ARCH_FLAG))"
	@echo "  LDFLAGS  - link flags (default: -lcurand -lcublas -lcusolver)"
	@echo "  ARGS     - runtime args for run target (default: $(ARGS))"

# Naive iterative solver target (placeholder for now)
naive: $(SRC_DIR)/app/naive_main.cu $(SRC_DIR)/util/cuda_checks.cuh $(SRC_DIR)/kernels/spmv_dense.cuh $(SRC_DIR)/kernels/reduce.cuh
	$(NVCC) $(CXXFLAGS) -o naive $(SRC_DIR)/app/naive_main.cu $(LDFLAGS)


