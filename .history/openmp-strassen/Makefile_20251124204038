# Makefile for OpenMP Divide & Conquer Matrix Multiplication
# Parallel Computing Assignment 1

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++11 -O3 -fopenmp -Wall -Wextra
LDFLAGS = -fopenmp

# Target executable
TARGET = optimized_main

# Source files
SOURCES = optimized_main.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Test parameters (as per assignment requirements)
SIZES = 100 1000 10000
THREADS = 1 2 4 8 16
THRESHOLDS = 32 64 128 256

# Default target
all: $(TARGET)
	@echo "============================================"
	@echo "Build complete!"
	@echo "Executable: $(TARGET)"
	@echo "============================================"

# Build executable
$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $(TARGET) $(OBJECTS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET) *.txt *.csv
	@echo "Clean complete"

# ==========================================
# ASSIGNMENT REQUIRED TESTS
# ==========================================

# Test Case 1: Matrix sizes (100, 1000, 10000) as per assignment
test-assignment-sizes: $(TARGET)
	@echo "=========================================="
	@echo "TEST CASE 1: Assignment Matrix Sizes"
	@echo "Testing: 100x100, 1000x1000, 10000x10000"
	@echo "=========================================="
	@echo ""
	@echo "Size: 100x100 (with verification)"
	@echo "------------------------------------------"
	./$(TARGET) 100 1 8 128
	@echo ""
	@echo "Size: 1000x1000 (with verification)"
	@echo "------------------------------------------"
	./$(TARGET) 1000 1 8 128
	@echo ""
	@echo "Size: 10000x10000 (NO verification - too slow!)"
	@echo "------------------------------------------"
	./$(TARGET) 10000 0 8 128
	@echo ""

# Test Case 2: Scalability Study - varying thread counts (strong scaling)
test-scalability-100: $(TARGET)
	@echo "=========================================="
	@echo "TEST CASE 2A: Scalability Study - 100x100"
	@echo "Strong Scaling (fixed problem size)"
	@echo "=========================================="
	@echo ""
	@for threads in 1 2 4 8 16; do \
		echo "Threads: $$threads"; \
		./$(TARGET) 100 0 $$threads 64; \
		echo ""; \
	done

test-scalability-1000: $(TARGET)
	@echo "=========================================="
	@echo "TEST CASE 2B: Scalability Study - 1000x1000"
	@echo "Strong Scaling (fixed problem size)"
	@echo "=========================================="
	@echo ""
	@for threads in 1 2 4 8 16; do \
		echo "Threads: $$threads"; \
		./$(TARGET) 1000 0 $$threads 128; \
		echo ""; \
	done

test-scalability-10000: $(TARGET)
	@echo "=========================================="
	@echo "TEST CASE 2C: Scalability Study - 10000x10000"
	@echo "Strong Scaling (fixed problem size)"
	@echo "=========================================="
	@echo ""
	@for threads in 1 2 4 8 16; do \
		echo "Threads: $$threads"; \
		./$(TARGET) 10000 0 $$threads 128; \
		echo ""; \
	done

test-scalability-all: test-scalability-100 test-scalability-1000 test-scalability-10000

# Test Case 3: Threshold Optimization Study
test-threshold-100: $(TARGET)
	@echo "=========================================="
	@echo "TEST CASE 3A: Threshold Study - 100x100"
	@echo "Finding optimal base case size"
	@echo "=========================================="
	@echo ""
	@for thresh in 32 64 128 256; do \
		echo "Threshold: $$thresh"; \
		./$(TARGET) 100 0 8 $$thresh; \
		echo ""; \
	done

test-threshold-1000: $(TARGET)
	@echo "=========================================="
	@echo "TEST CASE 3B: Threshold Study - 1000x1000"
	@echo "Finding optimal base case size"
	@echo "=========================================="
	@echo ""
	@for thresh in 32 64 128 256 512; do \
		echo "Threshold: $$thresh"; \
		./$(TARGET) 1000 0 8 $$thresh; \
		echo ""; \
	done

test-threshold-10000: $(TARGET)
	@echo "=========================================="
	@echo "TEST CASE 3C: Threshold Study - 10000x10000"
	@echo "Finding optimal base case size"
	@echo "=========================================="
	@echo ""
	@for thresh in 64 128 256 512; do \
		echo "Threshold: $$thresh"; \
		./$(TARGET) 10000 0 8 $$thresh; \
		echo ""; \
	done

test-threshold-all: test-threshold-100 test-threshold-1000 test-threshold-10000

# Test Case 4: Correctness Verification for all sizes
test-correctness: $(TARGET)
	@echo "=========================================="
	@echo "TEST CASE 4: Correctness Verification"
	@echo "Verifying results against serial version"
	@echo "=========================================="
	@echo ""
	@echo "Verifying 100x100..."
	@echo "------------------------------------------"
	./$(TARGET) 100 1 8 128
	@echo ""
	@echo "Verifying 500x500..."
	@echo "------------------------------------------"
	./$(TARGET) 500 1 8 128
	@echo ""
	@echo "Verifying 1000x1000..."
	@echo "------------------------------------------"
	./$(TARGET) 1000 1 8 128
	@echo ""

# Test Case 5: Power-of-2 vs Non-Power-of-2 Comparison
test-power2-comparison: $(TARGET)
	@echo "=========================================="
	@echo "TEST CASE 5: Power-of-2 vs Non-Power-of-2"
	@echo "Comparing optimal vs suboptimal sizes"
	@echo "=========================================="
	@echo ""
	@echo "Non-Power-of-2: 100x100"
	./$(TARGET) 100 0 8 64
	@echo ""
	@echo "Power-of-2: 128x128"
	./$(TARGET) 128 0 8 64
	@echo ""
	@echo "Non-Power-of-2: 1000x1000"
	./$(TARGET) 1000 0 8 128
	@echo ""
	@echo "Power-of-2: 1024x1024"
	./$(TARGET) 1024 0 8 128
	@echo ""
	@echo "Non-Power-of-2: 10000x10000"
	./$(TARGET) 10000 0 8 128
	@echo ""
	@echo "Power-of-2: 8192x8192"
	./$(TARGET) 8192 0 8 128
	@echo ""

# Test Case 6: Performance across different matrix sizes
test-performance-range: $(TARGET)
	@echo "=========================================="
	@echo "TEST CASE 6: Performance Across Size Range"
	@echo "Testing multiple sizes for performance curve"
	@echo "=========================================="
	@echo ""
	@for size in 100 200 500 1000 2000 5000 10000; do \
		echo "Size: $$size x $$size"; \
		./$(TARGET) $$size 0 8 128; \
		echo ""; \
	done

# Test Case 7: Weak Scaling (problem size scales with threads)
test-weak-scaling: $(TARGET)
	@echo "=========================================="
	@echo "TEST CASE 7: Weak Scaling Study"
	@echo "Problem size increases with thread count"
	@echo "=========================================="
	@echo ""
	@echo "1 thread, 512x512 matrix"
	./$(TARGET) 512 0 1 128
	@echo ""
	@echo "2 threads, 724x724 matrix (≈√2 scaling)"
	./$(TARGET) 724 0 2 128
	@echo ""
	@echo "4 threads, 1024x1024 matrix (2x scaling)"
	./$(TARGET) 1024 0 4 128
	@echo ""
	@echo "8 threads, 1448x1448 matrix (≈2√2 scaling)"
	./$(TARGET) 1448 0 8 128
	@echo ""

# Test Case 8: Comparing different thread configurations
test-thread-configs: $(TARGET)
	@echo "=========================================="
	@echo "TEST CASE 8: Thread Configuration Study"
	@echo "Testing various thread counts on 1000x1000"
	@echo "=========================================="
	@echo ""
	@for threads in 1 2 3 4 6 8 10 12 16 20 24; do \
		echo "Threads: $$threads"; \
		./$(TARGET) 1000 0 $$threads 128; \
		echo ""; \
	done

# Test Case 9: Threshold vs Size Interaction
test-threshold-size-matrix: $(TARGET)
	@echo "=========================================="
	@echo "TEST CASE 9: Threshold-Size Interaction"
	@echo "Testing different threshold-size combinations"
	@echo "=========================================="
	@echo ""
	@echo "Small Matrix (100x100) with various thresholds:"
	@echo "------------------------------------------"
	@for thresh in 16 32 64 128; do \
		echo "  Threshold: $$thresh"; \
		./$(TARGET) 100 0 8 $$thresh; \
	done
	@echo ""
	@echo "Medium Matrix (1000x1000) with various thresholds:"
	@echo "------------------------------------------"
	@for thresh in 32 64 128 256 512; do \
		echo "  Threshold: $$thresh"; \
		./$(TARGET) 1000 0 8 $$thresh; \
	done
	@echo ""
	@echo "Large Matrix (5000x5000) with various thresholds:"
	@echo "------------------------------------------"
	@for thresh in 64 128 256 512 1024; do \
		echo "  Threshold: $$thresh"; \
		./$(TARGET) 5000 0 8 $$thresh; \
	done
	@echo ""

# ==========================================
# COMPREHENSIVE TEST SUITES
# ==========================================

# Run ALL test cases (use this for complete report data!)
test-all: $(TARGET)
	@echo "=========================================="
	@echo "RUNNING ALL TEST CASES"
	@echo "This will take a while..."
	@echo "=========================================="
	@echo ""
	@make test-assignment-sizes
	@make test-scalability-all
	@make test-threshold-all
	@make test-correctness
	@make test-power2-comparison
	@make test-performance-range

# Quick test suite (essential tests only)
test-quick: $(TARGET)
	@echo "=========================================="
	@echo "QUICK TEST SUITE"
	@echo "=========================================="
	@make test-assignment-sizes
	@make test-scalability-1000

# Complete benchmark for report
test-report: $(TARGET)
	@echo "=========================================="
	@echo "GENERATING COMPLETE REPORT DATA"
	@echo "=========================================="
	@echo ""
	@echo "Starting comprehensive benchmark..."
	@echo "Output will be saved to: benchmark_results.txt"
	@echo ""
	@make test-all > benchmark_results.txt 2>&1
	@echo ""
	@echo "=========================================="
	@echo "BENCHMARK COMPLETE!"
	@echo "Results saved to: benchmark_results.txt"
	@echo "=========================================="

# ==========================================
# DATA EXPORT FOR ANALYSIS
# ==========================================

# Generate CSV data for plotting
export-csv: $(TARGET)
	@echo "Generating CSV data for analysis..."
	@echo "size,threads,threshold,time,gflops" > scalability_data.csv
	@echo "# Scalability data for 1000x1000"
	@for threads in 1 2 4 8 16; do \
		./$(TARGET) 1000 0 $$threads 128 | grep "Time:" | \
		awk -v t=$$threads '{print "1000,"t",128,"$$4","$$6}' >> scalability_data.csv; \
	done
	@echo "Data saved to: scalability_data.csv"

# ==========================================
# PLATFORM-SPECIFIC TESTS
# ==========================================

# Test on current machine
test-current-machine: $(TARGET)
	@echo "=========================================="
	@echo "TESTING ON CURRENT MACHINE"
	@echo "=========================================="
	@echo ""
	@echo "System Information:"
	@echo "------------------------------------------"
	@make info
	@echo ""
	@echo "Running benchmarks..."
	@echo "------------------------------------------"
	@make test-assignment-sizes
	@make test-scalability-1000

# Multiple machine test (for assignment requirement)
test-machine-1: test-current-machine
	@echo "Machine 1 tests complete"
	@echo "Results for Machine 1" > machine1_results.txt
	@make test-assignment-sizes >> machine1_results.txt 2>&1
	@make test-scalability-1000 >> machine1_results.txt 2>&1

test-machine-2: test-current-machine
	@echo "Machine 2 tests complete"
	@echo "Results for Machine 2" > machine2_results.txt
	@make test-assignment-sizes >> machine2_results.txt 2>&1
	@make test-scalability-1000 >> machine2_results.txt 2>&1

test-machine-3: test-current-machine
	@echo "Machine 3 tests complete"
	@echo "Results for Machine 3" > machine3_results.txt
	@make test-assignment-sizes >> machine3_results.txt 2>&1
	@make test-scalability-1000 >> machine3_results.txt 2>&1

# ==========================================
# UTILITY TARGETS
# ==========================================

# Quick test with small matrix and verification
test: $(TARGET)
	@echo "==================================="
	@echo "Quick Test (100x100, verify=1)"
	@echo "==================================="
	./$(TARGET) 100 1 4 64

# Run with default parameters
run: $(TARGET)
	@echo "Running with default parameters (1000x1000)..."
	./$(TARGET) 1000

# Check OpenMP support
check-openmp:
	@echo "Checking OpenMP support..."
	@echo "#include <omp.h>" > check_omp.cpp
	@echo "#include <iostream>" >> check_omp.cpp
	@echo "int main() { std::cout << \"Max threads: \" << omp_get_max_threads() << std::endl; return 0; }" >> check_omp.cpp
	@$(CXX) -fopenmp check_omp.cpp -o check_omp 2>/dev/null && \
		(./check_omp && echo "✓ OpenMP is supported") || \
		echo "✗ OpenMP is NOT supported"
	@rm -f check_omp.cpp check_omp

# Show system information
info:
	@echo "=========================================="
	@echo "System Information"
	@echo "=========================================="
	@echo "Compiler: $(CXX)"
	@$(CXX) --version | head -n 1
	@echo ""
	@echo "CPU Info:"
	@if [ -f /proc/cpuinfo ]; then \
		echo "  Model: $$(grep 'model name' /proc/cpuinfo | head -n 1 | cut -d: -f2 | sed 's/^[ \t]*//')"; \
		echo "  Cores: $$(grep -c ^processor /proc/cpuinfo)"; \
		echo "  Threads per core: $$(lscpu | grep 'Thread(s) per core' | awk '{print $$4}')"; \
	else \
		echo "  (CPU info not available)"; \
	fi
	@echo ""
	@echo "Memory Info:"
	@if [ -f /proc/meminfo ]; then \
		echo "  Total: $$(grep MemTotal /proc/meminfo | awk '{print $$2/1024/1024 \" GB"}')"; \
	else \
		echo "  (Memory info not available)"; \
	fi
	@echo ""
	@echo "OpenMP Info:"
	@make check-openmp 2>/dev/null || echo "  Unable to check OpenMP"

# Help message
help:
	@echo "=========================================="
	@echo "OpenMP Divide & Conquer Matrix Multiplication"
	@echo "=========================================="
	@echo ""
	@echo "BUILD:"
	@echo "  make              - Build the program"
	@echo "  make clean        - Remove all build and result files"
	@echo ""
	@echo "ASSIGNMENT TESTS (Required for report):"
	@echo "  make test-assignment-sizes      - Test 100, 1000, 10000 (REQUIRED)"
	@echo "  make test-scalability-all       - Thread scaling for all sizes (REQUIRED)"
	@echo "  make test-threshold-all         - Threshold optimization (REQUIRED)"
	@echo "  make test-correctness           - Verify results (REQUIRED)"
	@echo "  make test-power2-comparison     - Compare power-of-2 vs non-power-of-2"
	@echo ""
	@echo "INDIVIDUAL TEST CASES:"
	@echo "  make test-scalability-100       - Thread scaling for 100x100"
	@echo "  make test-scalability-1000      - Thread scaling for 1000x1000"
	@echo "  make test-scalability-10000     - Thread scaling for 10000x10000"
	@echo "  make test-threshold-100         - Threshold tuning for 100x100"
	@echo "  make test-threshold-1000        - Threshold tuning for 1000x1000"
	@echo "  make test-threshold-10000       - Threshold tuning for 10000x10000"
	@echo "  make test-performance-range     - Performance across multiple sizes"
	@echo "  make test-weak-scaling          - Weak scaling study"
	@echo "  make test-thread-configs        - Various thread configurations"
	@echo "  make test-threshold-size-matrix - Threshold-size interactions"
	@echo ""
	@echo "COMPREHENSIVE SUITES:"
	@echo "  make test-all                   - Run ALL test cases"
	@echo "  make test-quick                 - Essential tests only"
	@echo "  make test-report                - Generate complete report data"
	@echo ""
	@echo "MULTI-MACHINE TESTING:"
	@echo "  make test-machine-1             - Test on machine 1 (save to file)"
	@echo "  make test-machine-2             - Test on machine 2 (save to file)"
	@echo "  make test-machine-3             - Test on machine 3 (save to file)"
	@echo ""
	@echo "DATA EXPORT:"
	@echo "  make export-csv                 - Export data in CSV format"
	@echo ""
	@echo "UTILITIES:"
	@echo "  make test                       - Quick test (100x100)"
	@echo "  make run                        - Default run (1000x1000)"
	@echo "  make check-openmp               - Check OpenMP availability"
	@echo "  make info                       - Show system information"
	@echo "  make help                       - Show this help"
	@echo ""
	@echo "MANUAL EXECUTION:"
	@echo "  ./$(TARGET) <size> [verify] [threads] [threshold]"
	@echo ""
	@echo "  Examples:"
	@echo "    ./$(TARGET) 1000              - 1000x1000, verify, max threads"
	@echo "    ./$(TARGET) 2000 0            - 2000x2000, no verify"
	@echo "    ./$(TARGET) 1000 1 8          - 1000x1000, verify, 8 threads"
	@echo "    ./$(TARGET) 1000 0 16 64      - Full control"
	@echo ""
	@echo "RECOMMENDED FOR ASSIGNMENT:"
	@echo "  1. make test-report             - Generate all data"
	@echo "  2. make test-machine-1          - Test on your laptop"
	@echo "  3. make test-machine-2          - Test on lab machine"
	@echo "  4. make test-machine-3          - Test on cluster"

.PHONY: all clean run test test-assignment-sizes test-scalability-100 \
        test-scalability-1000 test-scalability-10000 test-scalability-all \
        test-threshold-100 test-threshold-1000 test-threshold-10000 \
        test-threshold-all test-correctness test-power2-comparison \
        test-performance-range test-weak-scaling test-thread-configs \
        test-threshold-size-matrix test-all test-quick test-report \
        export-csv test-current-machine test-machine-1 test-machine-2 \
        test-machine-3 check-openmp info help