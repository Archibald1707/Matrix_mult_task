CXX = g++
NVCC = nvcc

CXXFLAGS = -std=c++17 -O2 -Wall -fopenmp
NVCCFLAGS = -std=c++17

TARGET = matrix_mul

SRC_CPP = main.cpp
SRC_CU = cuda_mul.cu

OBJ = main.o cuda.o

all: $(TARGET)

main.o: $(SRC_CPP)
	$(CXX) $(CXXFLAGS) -c $(SRC_CPP) -o main.o

cuda.o: $(SRC_CU)
	$(NVCC) $(NVCCFLAGS) -c $(SRC_CU) -o cuda.o

$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(OBJ) -Xcompiler -fopenmp

clean:
	rm -f *.o $(TARGET)
