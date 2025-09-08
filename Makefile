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

CXXFLAGS ?= -O3 -std=c++14 $(ARCH_FLAG)
LDFLAGS ?= -lcurand

.PHONY: all run clean help

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

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
	@echo "  LDFLAGS  - link flags (default: -lcurand)"


