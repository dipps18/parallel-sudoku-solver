#include<iostream>
#include<stdio.h>
#include<cstdlib>
#include<algorithm>
#include<thrust/swap.h>
#include <thrust/scan.h>
#include<thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

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

__global__ void initNewGrids(grid *curGrids, grid *newGrids, int r, int c, int curGridsSize, int* startPositions)
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid < curGridsSize)
    {
        grid newGridsTid[9] = {0};
        int idx = 0;
        for(int i = 1; i <= 9; i++)
        {
            if(is_safe(curGrids[tid], i, r, c))
            {
                curGrids[tid][r*9 + c] = i;
                memcpy(newGridsTid[idx++], curGrids[tid], sizeof(grid));
            }
        }
        memcpy(newGrids + startPositions[tid], newGridsTid, sizeof(grid)*idx);
    }
}

__global__ void possibleGrids(grid* grids, int row, int col, int curGridCount, int* startPositions)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if(tid < curGridCount)
    {
        int count = 0;
        for(int i = 1; i <= 9; i++)
        {
            if(is_safe(grids[tid], i, row, col))
                count++;
        }
        startPositions[tid] = count;
    }
}

float solve(grid &initGrid, grid* &curGrids, grid* &newGrids, int* startPositions)
{
    cudaEvent_t algo_start, algo_stop;
    float algo_time = 0;
    int threadsPerBlock = 256, newCount = 0, *curCount;
    checkCudaErrors(cudaMallocManaged(&curCount, sizeof(int)));
    *curCount = 1;
    memcpy(curGrids[0], initGrid, sizeof(grid));

    cudaEventCreate(&algo_start);
    cudaEventCreate(&algo_stop);
    cudaEventRecord(algo_start);

    for(int i = 0; i < 9; i++)
    {
        for(int j = 0; j < 9; j++)
        {
            if(initGrid[9*i + j] == 0)
            {
                int numBlocks = (*curCount + threadsPerBlock - 1)/threadsPerBlock;
                possibleGrids<<<numBlocks, threadsPerBlock>>>(curGrids, i, j, *curCount, startPositions);
                cudaDeviceSynchronize();

                newCount = thrust::reduce(startPositions, startPositions + *curCount);       
                thrust::exclusive_scan(startPositions, startPositions + *curCount, startPositions);

                initNewGrids<<<numBlocks, threadsPerBlock>>>(curGrids, newGrids, i, j, *curCount, startPositions);
                cudaDeviceSynchronize();

                thrust::swap(newGrids, curGrids);
                *curCount = newCount;
            }
        }
    }
    cudaEventRecord(algo_stop);
    cudaEventSynchronize(algo_stop);
    cudaEventElapsedTime(&algo_time, algo_start, algo_stop);

    checkCudaErrors(cudaFree(curCount));
    return algo_time;
}

int main(void)
{
    int n = 1000000, *startPositions;
    grid *curGrids, *newGrids;
    cudaEvent_t start_init, stop_init;
    float init_time = 0, total_algo_time = 0;

    cudaEventCreate(&start_init);
    cudaEventCreate(&stop_init);
    cudaEventRecord(start_init);

    checkCudaErrors(cudaMallocManaged(&curGrids, n*sizeof(grid)));
    checkCudaErrors(cudaMallocManaged(&newGrids, n*sizeof(grid)));
    checkCudaErrors(cudaMallocManaged(&startPositions, n*sizeof(int)));

    grid myGrids[6] ={ 
        {5,8,6,0,7,0,0,0,0,0,0,0,9,0,1,6,0,0,0,0,0,6,0,0,0,0,0,0,0,7,0,0,0,0,0,0,9,0,2,0,1,0,3,0,5,0,0,5,0,9,0,0,0,0,0,9,0,0,4,0,0,0,8,0,0,3,5,0,0,0,0,6,0,0,0,0,2,0,4,7,0},
        {0,7,0,0,0,0,0,4,3,0,4,0,0,0,9,6,1,0,8,0,0,6,3,4,9,0,0,0,9,4,0,5,2,0,0,0,3,5,8,4,6,0,0,2,0,0,0,0,8,0,0,5,3,0,0,8,0,0,7,0,0,9,1,9,0,2,1,0,0,0,0,5,0,0,7,0,4,0,8,0,2},
        {3,0,1,0,8,6,5,0,4,0,4,6,5,2,1,0,7,0,5,0,0,0,0,0,0,0,1,4,0,0,8,0,0,0,0,2,0,8,0,3,4,7,9,0,0,0,0,9,0,5,0,0,3,8,0,0,4,0,9,0,2,0,0,0,0,8,7,3,4,0,9,0,0,0,7,2,0,8,1,0,3},
        {0,4,8,3,0,1,5,6,0,3,6,0,0,0,8,0,9,0,9,1,0,6,7,0,0,0,3,0,2,0,0,0,0,9,3,5,5,0,9,0,1,0,2,0,0,6,7,0,0,2,0,0,1,0,0,0,4,0,0,2,1,0,7,0,9,0,1,0,0,0,0,8,1,5,0,8,3,4,0,2,9},
        {0,0,0,0,0,0,0,0,8,0,0,3,0,0,0,4,0,0,0,9,0,0,2,0,0,6,0,0,0,0,0,7,9,0,0,0,0,0,0,0,6,1,2,0,0,0,6,0,5,0,2,0,7,0,0,0,8,0,0,0,5,0,0,0,1,0,0,0,0,0,2,0,4,0,5,0,0,0,0,0,3},
        {0,0,0,0,0,0,0,0,2,0,0,8,0,1,0,9,0,0,5,0,0,0,0,3,0,4,0,0,0,0,1,0,9,3,0,0,0,6,0,0,3,0,0,8,0,0,0,3,7,0,0,0,0,0,0,4,0,0,0,0,0,0,5,3,0,1,0,7,0,8,0,0,2,0,0,0,0,0,0,0,0}};

    cudaEventRecord(stop_init);
    cudaEventSynchronize(stop_init);
    cudaEventElapsedTime(&init_time, start_init, stop_init);

    for(int i = 0; i < 6; i++)
    {
        std::cout <<"Grid "<<i + 1<< ": \n";
        display_grid(myGrids[i]);
        total_algo_time += solve(myGrids[i], curGrids, newGrids, startPositions);
        std::cout << "\nSolved Grid:\n";
        display_grid(curGrids[0]);
        std::cout << std::endl;
    }

    std::cout << "\nTime taken for data generation: " << init_time << "ms"<< std::endl;
    std::cout << "Total time taken by the main algorithm: " << total_algo_time << "ms"<< std::endl;

    checkCudaErrors(cudaFree(curGrids));
    checkCudaErrors(cudaFree(newGrids));
    checkCudaErrors(cudaFree(startPositions));
}