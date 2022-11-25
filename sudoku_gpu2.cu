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

// #define checkCudaErrors(msg) \
//     do { \
//         cudaError_t __err = cudaGetLastError(); \
//         if (__err != cudaSuccess) { \
//             fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
//                 msg, cudaGetErrorString(__err), \
//                 __FILE__, __LINE__); \
//             fprintf(stderr, "*** FAILED - ABORTING\n"); \
//             exit(1); \
//         } \
//     } while (0)

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

__global__ void initNewGrids(grid *curGrids, grid *newGrids, int r, int c, int curGridsSize, int* starPosition)
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid < curGridsSize)
    {
        int idx = starPosition[tid];
        for(int i = 1; i <= 9; i++)
        {
            if(is_safe(curGrids[tid], i, r, c))
            {
                curGrids[tid][r*9 + c] = i;
                memcpy(newGrids[idx++], curGrids[tid], sizeof(grid));
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
        int count = 0; //new grids that will be formed from the current grids
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

void solve(grid &initGrid)
{
    int threadsPerBlock = 256;
    //Stores the current grids
    grid *curGrids, *newGrids;
    //Stores the number of current grids and the number of new grids
    int *curCount, *newCount;
    int *startPositions;
    checkCudaErrors(cudaMallocManaged(&curCount, sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&newCount, sizeof(int)));

    checkCudaErrors(cudaMemset(newCount, 0, sizeof(int)));
    checkCudaErrors(cudaMemset(curCount, 0, sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&curGrids, sizeof(grid)));
    *curCount = 1;
    memcpy(curGrids, initGrid, sizeof(grid));
    for(int i = 0; i < 9; i++)
    {
        for(int j = 0; j < 9; j++)
        {
            if(initGrid[9*i + j] == 0)
            {
                int numBlocks = (*curCount + threadsPerBlock - 1)/threadsPerBlock;
                checkCudaErrors(cudaMallocManaged(&startPositions, sizeof(int)*(*curCount)));
                possibleGrids<<<numBlocks, threadsPerBlock>>>(curGrids, i, j, *curCount, newCount, startPositions);
                cudaDeviceSynchronize();
                checkCudaErrors(cudaMallocManaged(&newGrids, sizeof(grid) * (*newCount)));
                checkCudaErrors(cudaMemset(newGrids, 0, sizeof(grid) * (*newCount)));
                //thrust::inclusive_scan(thrust::host, startPositions, startPositions + *curCount - 1, startPositions);
                int prefixSum = startPositions[0];
                for(int k = 1; k < *curCount; k++) 
                {
                    int temp = startPositions[k];
                    startPositions[k] = prefixSum;
                    prefixSum += temp;
                }
                startPositions[0] = 0;

                initNewGrids<<<numBlocks, threadsPerBlock>>>(curGrids, newGrids, i, j, *curCount, startPositions);
                cudaDeviceSynchronize();
                checkCudaErrors(cudaFree(curGrids));
                checkCudaErrors(cudaMallocManaged(&curGrids, sizeof(grid)*(*newCount)));

                cudaMemset(curGrids, 0, sizeof(grid));
                thrust::swap(newGrids, curGrids);
                std::swap(curCount, newCount);
                *newCount = 0;
                for(int k = 0; k < *curCount; k++)
                {
                    display_grid(curGrids[k]);
                    std::cout << std::endl;
                }

                checkCudaErrors(cudaFree(startPositions));
                checkCudaErrors(cudaFree(newGrids));

            }
        }
    }

}


int main(void)
{
    grid myGrid = {3, 0, 6, 5, 0, 8, 4, 0, 0, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 8, 7, 0, 0, 0, 0, 3, 1, 0, 0, 3, 0, 1, 0, 0, 8, 0, 9, 0, 0, 8, 6, 3, 0, 0, 5, 0, 5, 0, 0, 9, 0, 6, 0, 0, 1, 3, 0, 0, 0, 0, 2, 5, 0 ,0, 0, 0, 0, 0, 0, 0, 7, 4, 0, 0, 5, 2, 0, 6, 3, 0, 0};
    solve(myGrid);
}