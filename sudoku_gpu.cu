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


typedef char grid[81];

void displayGrid(grid curGrid)
{
    for(int i = 0; i < 9; i++)
    {
        for(int j = 0; j < 9; j++)
            std::cout << (int)curGrid[9*i + j] << " ";
        std::cout << std::endl;
    }
}

__device__ bool isSafe(grid curGrid, int val, int row, int col)
{
    for(int i = 0; i < 9; i++)
    {
        if(i != row && curGrid[9*i + col] == val) 
            return false;
        if(i != col && curGrid[row*9 + i] == val)
            return false;
    }

    int start_row = row - row % 3;
    int start_col = col - col % 3;
    for(int i = start_row; i < start_row + 3; i++)
    {
        for(int j = start_col; j < start_col + 3; j++)
        {
            if((i != row || j != col) && curGrid[9*i + j] == val)
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
            if(isSafe(curGrids[tid], i, r, c))
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
            if(isSafe(grids[tid], i, row, col))
                count++;
        }
        startPositions[tid] = count;
    }
}

float solve(grid &initGrid, grid* &curGrids, grid* &newGrids, int* startPositions)
{
    cudaEvent_t algo_start, algo_stop;
    float algo_time = 0;
    int threadsPerBlock = 64, newCount = 0, *curCount;
    checkCudaErrors(cudaMallocManaged(&curCount, sizeof(int)));
    *curCount = 1;
    memcpy(curGrids[0], initGrid, sizeof(grid));

    checkCudaErrors(cudaEventCreate(&algo_start));
    checkCudaErrors(cudaEventCreate(&algo_stop));
    checkCudaErrors(cudaEventRecord(algo_start));

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
    checkCudaErrors(cudaEventRecord(algo_stop));
    checkCudaErrors(cudaEventSynchronize(algo_stop));
    checkCudaErrors(cudaEventElapsedTime(&algo_time, algo_start, algo_stop));
    checkCudaErrors(cudaFree(curCount));
    return algo_time;
}

int main(void)
{
    int n = 1000000, *startPositions;
    grid *curGrids, *newGrids;
    //Measures time taken for initialization
    cudaEvent_t startInit, stopInit;
    float initTime = 0, totalAlgoTime = 0;

    cudaEventCreate(&startInit);
    cudaEventCreate(&stopInit);
    cudaEventRecord(startInit);

    checkCudaErrors(cudaMallocManaged(&curGrids, n*sizeof(grid)));
    checkCudaErrors(cudaMallocManaged(&newGrids, n*sizeof(grid)));
    checkCudaErrors(cudaMallocManaged(&startPositions, n*sizeof(int)));

    grid myGrids[16] ={ 
        {5,8,6,0,7,0,0,0,0,0,0,0,9,0,1,6,0,0,0,0,0,6,0,0,0,0,0,0,0,7,0,0,0,0,0,0,9,0,2,0,1,0,3,0,5,0,0,5,0,9,0,0,0,0,0,9,0,0,4,0,0,0,8,0,0,3,5,0,0,0,0,6,0,0,0,0,2,0,4,7,0},
        {0,0,0,0,0,0,0,0,2,0,0,8,0,1,0,9,0,0,5,0,0,0,0,3,0,4,0,0,0,0,1,0,9,3,0,0,0,6,0,0,3,0,0,8,0,0,0,3,7,0,0,0,0,0,0,4,0,0,0,0,0,0,5,3,0,1,0,7,0,8,0,0,2,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,7,0,0,4,0,2,0,6,0,0,8,0,0,0,0,0,3,1,0,0,0,0,0,0,2,9,0,0,0,4,0,0,9,0,0,3,0,0,0,9,5,0,6,0,0,0,0,1,0,0,0,0,0,0,8,0,0,6,0,5,0,2,0,0,7,0,0,0,0,0,0,6,0},
        {0,0,2,0,0,0,7,0,0,0,1,0,0,0,0,0,6,0,5,0,0,0,0,0,0,1,8,0,0,0,0,3,7,0,0,0,0,0,0,0,4,9,0,0,0,0,0,4,1,0,2,3,0,0,0,0,3,0,2,0,9,0,0,0,8,0,0,0,0,0,5,0,6,0,0,0,0,0,0,0,2},
        {0,0,4,0,0,3,0,0,0,0,7,0,0,8,0,0,0,0,2,0,8,1,0,0,0,0,6,0,0,3,0,0,0,0,9,0,0,8,0,0,2,0,0,0,0,1,0,0,7,0,0,0,0,3,0,0,0,0,0,0,4,5,0,0,0,0,8,0,0,9,0,0,0,0,9,0,0,5,0,0,8},
        {0,0,6,0,0,1,0,0,0,0,5,0,0,3,0,0,0,0,9,0,0,4,0,0,0,0,7,0,0,1,0,0,0,0,2,0,0,3,0,0,9,0,0,0,0,4,0,0,5,0,0,0,0,1,3,0,0,0,0,0,6,8,0,0,0,0,3,0,0,2,0,0,0,0,2,0,0,8,0,0,3},
        {0,0,0,0,0,0,0,0,3,0,0,1,0,0,9,0,6,0,0,5,0,0,8,0,4,0,0,0,0,0,9,0,0,0,8,0,0,0,8,6,7,0,0,0,0,0,1,0,0,0,0,2,0,0,0,0,6,0,0,7,0,2,0,0,3,0,8,0,0,5,0,0,4,0,0,0,0,0,0,0,8},
        {0,0,0,0,0,0,0,0,5,0,0,6,0,0,8,7,0,0,3,0,0,0,0,0,0,9,0,0,0,0,1,0,7,0,4,0,0,0,7,0,0,0,8,0,0,0,4,0,0,0,6,0,0,0,0,9,0,0,8,0,0,0,3,0,0,1,6,0,0,4,0,0,5,0,0,0,2,0,0,0,0},                     
        {0,0,0,0,0,0,0,0,9,0,0,6,0,1,0,7,0,2,4,0,0,0,0,0,0,3,0,0,0,0,0,0,1,2,0,0,0,6,0,0,2,0,0,5,0,0,0,2,8,0,7,0,0,0,0,3,0,0,0,0,0,0,4,0,0,8,0,7,0,6,0,0,9,0,0,1,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,6,0,0,5,0,0,1,8,0,0,0,9,0,0,0,8,0,7,0,0,0,0,8,0,2,0,0,0,0,0,3,0,1,0,2,0,0,4,0,0,5,0,3,0,0,0,0,6,0,0,0,0,0,9,0,0,0,8,3,0,0,1,0,0,7,0,0,0,0,0,0,0,4},
        {0,0,0,0,0,5,0,0,4,0,9,0,0,0,0,0,2,0,0,0,6,0,7,0,3,0,0,0,0,0,7,0,0,8,0,0,0,0,8,6,0,0,0,0,0,1,3,0,0,8,0,0,0,0,0,0,3,0,1,0,6,0,0,0,2,0,0,0,0,0,0,5,4,0,0,0,0,0,0,9,0},
        {0,0,4,0,0,3,0,0,0,0,7,0,0,8,0,0,0,0,2,0,8,1,0,0,0,0,6,0,0,3,0,0,0,0,9,0,0,8,0,0,2,0,0,0,0,1,0,0,7,0,0,0,0,3,0,0,0,0,0,0,4,5,0,0,0,0,8,0,0,9,0,0,0,0,9,0,0,5,0,0,8},
        {0,0,6,0,0,1,0,0,0,0,5,0,0,3,0,0,0,0,9,0,0,4,0,0,0,0,7,0,0,1,0,0,0,0,2,0,0,3,0,0,9,0,0,0,0,4,0,0,5,0,0,0,0,1,3,0,0,0,0,0,6,8,0,0,0,0,3,0,0,2,0,0,0,0,2,0,0,8,0,0,3},
        {0,0,0,0,0,0,0,0,3,0,0,1,0,0,9,0,6,0,0,5,0,0,8,0,4,0,0,0,0,0,9,0,0,0,8,0,0,0,8,6,7,0,0,0,0,0,1,0,0,0,0,2,0,0,0,0,6,0,0,7,0,2,0,0,3,0,8,0,0,5,0,0,4,0,0,0,0,0,0,0,8},
        {0,0,0,0,0,0,0,0,5,0,0,6,0,0,8,7,0,0,3,0,0,0,0,0,0,9,0,0,0,0,1,0,7,0,4,0,0,0,7,0,0,0,8,0,0,0,4,0,0,0,6,0,0,0,0,9,0,0,8,0,0,0,3,0,0,1,6,0,0,4,0,0,5,0,0,0,2,0,0,0,0},
        {0,0,0,0,0,5,0,0,3,0,0,9,0,0,0,0,4,0,0,8,1,0,4,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0,4,0,0,2,0,0,6,8,0,0,0,1,4,0,3,0,0,0,0,0,0,0,2,0,0,0,4,0,0,0,6,0,0,7,9,0,0,0,5,0,0,1,0}};
    cudaEventRecord(stopInit);
    cudaEventSynchronize(stopInit);
    cudaEventElapsedTime(&initTime, startInit, stopInit);

    for(int i = 0; i < 16; i++)
    {
        std::cout <<"Grid "<<i + 1<< ": \n";
        displayGrid(myGrids[i]);
        totalAlgoTime += solve(myGrids[i], curGrids, newGrids, startPositions);
        std::cout << "\nSolved Grid:\n";
        displayGrid(curGrids[0]);
        std::cout << std::endl;
    }

    std::cout << "\nTime taken for data generation: " << initTime << "ms"<< std::endl;
    std::cout << "Total time taken by the main algorithm: " << totalAlgoTime << "ms"<< std::endl;

    checkCudaErrors(cudaFree(curGrids));
    checkCudaErrors(cudaFree(newGrids));
    checkCudaErrors(cudaFree(startPositions));
}