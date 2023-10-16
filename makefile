NVCC=nvcc -I/home/purechar/assignment1/parallel-sudoku-solver/Common
CUDAFLAGS=-arch=sm_30
OPT=-g -G

.PHONY: all clean

all: sudoku_gpu sudoku_cpu

main: sudoku_gpu.o sudoku_cpu.o
	${NVCC} ${OPT} -o main sudoku_gpu.o

sudoku_gpu.o: sudoku_gpu.cu
	$(NVCC) ${OPT} ${CUDAFLAGS} -std=c++11 -c sudoku_gpu.cu

sudoku_cpu.o: sudoku_cpu.cu
	$(NVCC) ${OPT} ${CUDAFLAGS} -std=c++11 -c sudoku_cpu.cu

sudoku_gpu: sudoku_gpu.o
	${NVCC} ${CUDAFLAGS} -o sudoku_gpu sudoku_gpu.o

sudoku_cpu: sudoku_cpu.o
	${NVCC} ${CUDAFLAGS} -o sudoku_cpu sudoku_cpu.o

clean:
	rm -f *.o sudoku_gpu, sudoku_cpu
