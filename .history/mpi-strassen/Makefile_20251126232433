CXX = mpicxx
CXXFLAGS = -O2 -std=c++11
TARGET = main
SOURCES := $(wildcard *.cpp)
OBJECTS := $(SOURCES:.cpp=.o)
N ?= 1000
NP ?= 7

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^


run: $(TARGET)
	mpirun -np $(NP)  ./$(TARGET) $(N) 0

clean:
	rm -f $(TARGET)