CXX = mpicxx
CXXFLAGS = -std=c++17 -O3 -march=native
EXECUTABLE = strassen_mpi
SOURCES = main.cpp
NP = 7

.PHONY: all
all: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCES)
	@echo "Compiling Strassen MPI with $(CXXFLAGS)..."
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(EXECUTABLE)
	@echo "Compilation successful. Executable: $(EXECUTABLE)"


.PHONY: test100
test100: $(EXECUTABLE)
	@echo ""
	@echo "--- Running Test: N=100 (Check ON) ---"
	mpirun --hostfile host.txt -np $(NP) ./$(EXECUTABLE) 100 1


.PHONY: test1000
test1000: $(EXECUTABLE)
	@echo ""
	@echo "--- Running Test: N=1000 (Check ON) ---"
	mpirun --hostfile host.txt -np $(NP) ./$(EXECUTABLE) 1000 1


.PHONY: test10000
test10000: $(EXECUTABLE)
	@echo ""
	@echo "--- Running Test: N=10000 (Check ON) ---"
	mpirun --hostfile host.txt -np $(NP) ./$(EXECUTABLE) 10000 1

.PHONY: test_all
test_all: test100 test1000 test10000
	@echo ""
	@echo "--- All tests completed. ---"


.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -f $(EXECUTABLE)