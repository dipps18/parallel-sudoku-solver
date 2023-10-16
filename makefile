NVCC=nvcc
CUDAFLAGS=-arch=sm_30
OPT=-g -G
INC=-I./Common

.PHONY: all clean

all: sudoku_gpu sudoku_cpu

main: sudoku_gpu.o sudoku_cpu.o
	${NVCC} ${INC} ${OPT} -o main sudoku_gpu.o

sudoku_gpu.o: sudoku_gpu.cu
	$(NVCC) ${INC} ${OPT} ${CUDAFLAGS} -std=c++11 -c sudoku_gpu.cu

sudoku_cpu.o: sudoku_cpu.cu
	$(NVCC) ${INC} ${OPT} ${CUDAFLAGS} -std=c++11 -c sudoku_cpu.cu

sudoku_gpu: sudoku_gpu.o
	${NVCC} ${INC} ${CUDAFLAGS} -o sudoku_gpu sudoku_gpu.o

sudoku_cpu: sudoku_cpu.o
	${NVCC} ${INC} ${CUDAFLAGS} -o sudoku_cpu sudoku_cpu.o

clean:
	rm -f *.o sudoku_gpu, sudoku_cpu
