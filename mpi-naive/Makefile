CXX = mpicxx
CXXFLAGS = -O2 -std=c++11
TARGET = main
SRC = main.cpp
N ?= 1000
NP ?= 4

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	mpirun -np $(NP) --use-hwthread-cpus ./$(TARGET) $(N)

clean:
	rm -f $(TARGET)