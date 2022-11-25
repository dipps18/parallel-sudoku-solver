#include<iostream>
#include<stdio.h>
#include<cstdlib>
#include<algorithm>
#include<thrust/swap.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

const int grid_width = 81;

typedef int grid[grid_width];

void display_grid(int *h_grid)
{
    for(int i = 0; i < 9; i++)
    {
        for(int j = 0; j < 9; j++)
            std::cout << h_grid[9*i + j] << " ";
        std::cout << std::endl;
    }
}

__device__ bool is_safe(grid h_grid, int val, int row, int col)
{
    for(int i = 0; i < 9; i++)
    {
        if(i != row && h_grid[9*i + col] == val) 
            return false;
        if(i != col && h_grid[row*9 + i] == val)
            return false;
    }

    int start_row = row - row % 3;
    int start_col = col - col % 3;
    for(int i = start_row; i < start_row + 3; i++)
    {
        for(int j = start_col; j < start_col + 3; j++)
        {
            if((i != row || j != col) && h_grid[9*i + j] == val)
                return false;
        }
    }
    return true;
}

__global__ void initNewGrids(grid *curGrids, grid *newGrids, int r, int c, int curGridsSize, int* elems)
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    int idx = 0;
    if(tid < curGridsSize)
    {
        grid* localGrids = (grid*)malloc(elems[tid]*sizeof(grid));
        for(int i = 1; i <= 9; i++)
        {
            if(is_safe(curGrids[tid], i, r, c))
            {
                curGrids[tid][r*9 + c] = i;
                memcpy(localGrids[idx++], curGrids[tid], sizeof(grid));
            }
        }
    }
    __syncthreads();

}

__global__ void possibleGrids(grid* grids, int row, int col, int curGridCount, int* newGridCount, int* startPositions)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if(tid < curGridCount)
    {
        grid localGrid;
        int count; //new grids that will be formed from the current grids
        memcpy(localGrid, grids[tid], sizeof(grid));
        int mp[10] = {0};
        for(int i = 0; i < 9; i++)
        {
            int row_cell = localGrid[9*i + col];
            int col_cell = localGrid[9*row + i];
            mp[row_cell]++;
            mp[col_cell]++;
        }
        int start_row = row - row % 3;
        int start_col = col - col % 3;
        for(int i = start_row; i < start_row + 3; i++)
        {
            for(int j = start_col; j < start_col + 3; j++)
            {
                int cell = grids[tid][9*i + j];
                mp[cell]++;
            }
        }
        for(int i = 1; i < 10; i++)
        {
            if(mp[i] == 0)
            { 
                count++;
                atomicAdd(newGridCount, 1);
            }
        }
        startPositions[tid] = count;
    }
    
}

void solve(grid &h_grid)
{
    int threadsPerBlock = 256;
    //Stores the current grids
    grid *h_curGrids;
    //Stores the current and new grids
    grid *d_curGrids, *d_newGrids;
    //Stores the number of current grids and the number of new grids
    int h_curGridCount = 1, h_newGridCount = 0;
    int *d_curGridCount, *d_newGridCount;
    int *d_startPositions;
    checkCudaErrors(cudaMalloc(&d_curGridCount, sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_newGridCount, sizeof(int)));

    checkCudaErrors(cudaMemset(d_newGridCount, 0, sizeof(int)));

    checkCudaErrors(cudaMalloc(&d_curGrids, sizeof(grid)));
    checkCudaErrors(cudaMemcpy(d_curGrids, &h_grid, sizeof(grid), cudaMemcpyHostToDevice));
    for(int i = 0; i < 9; i++)
    {
        for(int j = 0; j < 9; j++)
        {
            if(h_grid[9*i + j] == 0)
            {
                int numBlocks = (h_curGridCount + threadsPerBlock - 1)/threadsPerBlock;
                checkCudaErrors(cudaMalloc(&d_startPositions, sizeof(int)*h_curGridCount));
                checkCudaErrors(cudaMemset(d_startPositions, 0, sizeof(int)*h_curGridCount));
                possibleGrids<<<numBlocks, threadsPerBlock>>>(d_curGrids, i, j, h_curGridCount, d_newGridCount, d_startPositions);
                
                checkCudaErrors(cudaMemcpy(&h_newGridCount, d_newGridCount, sizeof(int), cudaMemcpyDeviceToHost));
                checkCudaErrors(cudaMalloc(&d_newGrids, sizeof(grid) * h_newGridCount));
                checkCudaErrors(cudaMemset(d_newGrids, 0, sizeof(grid) * h_newGridCount));

                initNewGrids<<<numBlocks, threadsPerBlock>>>(d_curGrids, d_newGrids, i, j, h_curGridCount, d_startPositions);
                

                checkCudaErrors(cudaFreeHost(h_curGrids));
                checkCudaErrors(cudaMallocHost(&h_curGrids, sizeof(grid)*h_newGridCount));
                if(h_curGrids == NULL) {
                    fprintf(stderr, "Failed to allocate host grid curGrids!\n");
                    exit(EXIT_FAILURE);
                }
                cudaMemset(h_curGrids, 0, sizeof(grid));
                checkCudaErrors(cudaMemcpy(h_curGrids, d_newGrids, sizeof(grid)*h_newGridCount, cudaMemcpyDeviceToHost));
                thrust::swap(d_newGrids, d_curGrids);
                checkCudaErrors(cudaFree(d_newGrids));
                h_curGridCount = h_newGridCount;
                checkCudaErrors(cudaMemset(d_newGridCount, 0, sizeof(int)));
                for(int k = 0; k < h_curGridCount; k++)
                {
                    display_grid(h_curGrids[k]);
                    std::cout << std::endl;
                }
                checkCudaErrors(cudaFree(d_startPositions));

            }
        }
    }

}


int main(void)
{
    grid myGrid = {3, 0, 6, 5, 0, 8, 4, 0, 0, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 8, 7, 0, 0, 0, 0, 3, 1, 0, 0, 3, 0, 1, 0, 0, 8, 0, 9, 0, 0, 8, 6, 3, 0, 0, 5, 0, 5, 0, 0, 9, 0, 6, 0, 0, 1, 3, 0, 0, 0, 0, 2, 5, 0 ,0, 0, 0, 0, 0, 0, 0, 7, 4, 0, 0, 5, 2, 0, 6, 3, 0, 0};
    solve(myGrid);
}