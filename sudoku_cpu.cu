#include<iostream>
#include<stdio.h>
#include<cstdlib>
#include<algorithm>
#include<string>
void display_grid(int *initalGrid)
{
    for(int i = 0; i < 9; i++)
    {
        for(int j = 0; j < 9; j++)
            std::cout << initalGrid[9*i + j] << " ";
        std::cout << std::endl;
    }
}

bool is_safe(int* initalGrid, int val, int row, int col)
{
    for(int i = 0; i < 9; i++)
    {
        if(i != row && initalGrid[9*i + col] == val) 
            return false;
        if(i != col && initalGrid[row*9 + i] == val)
            return false;
    }

    int start_row = row - row % 3;
    int start_col = col - col % 3;
    for(int i = start_row; i < start_row + 3; i++)
    {
        for(int j = start_col; j < start_col + 3; j++)
        {
            if((i != row || j != col) && initalGrid[9*i + j] == val)
                return false;
        }
    }
    return true;
}

void update_grids(int **curGrids, int **newGrids, int r, int c, int grids)
{
    int idx = 0;
    for(int k = 0; k < grids; k++)
    {  
        for(int i = 1; i <= 9; i++)
        {
            if(is_safe(curGrids[k], i, r, c))
            {
                curGrids[k][r*9 + c] = i;
                std::copy(curGrids[k], curGrids[k] + 81, newGrids[idx++]);
            }
        }
    }
}

int no_of_possible_grids(int** grids, int row, int col, int gridCount)
{
    int res = 0;
    for(int k = 0; k < gridCount; k++)
    {
        for(int i = 1; i <= 9; i++)
            if(is_safe(grids[k], i, row, col))
                res++;
    }
    return res;
    
}

void dealloc_mem(int** &grid, int size)
{
    for(int i = 0; i < size; i++)
            delete[] grid[i];
        delete [] grid;
}

void alloc_mem(int** &grid, int size)
{
    grid = new int* [size];
    for(int k = 0; k < size; k++)
        grid[k] = new int[81];
}

float solve(int (&initalGrid)[81], int**&curGrids, int**&newGrids)
{
    //To measure time the by the main algorithm
    cudaEvent_t startAlgo, stopAlgo;
    cudaEventCreate(&startAlgo);
    cudaEventCreate(&stopAlgo);
    float algoTime = 0;
    int curGridCount = 1;
    std::copy(initalGrid, initalGrid + 81, curGrids[0]);
    cudaEventRecord(startAlgo);
    for(int i = 0; i < curGridCount; i++)
    {
        for(int r = 0; r < 9; r++)
        {
            for(int c = 0; c < 9; c++)
            {
                if(curGrids[i][r*9 + c] == 0)
                {
                    int num = no_of_possible_grids(curGrids, r, c, curGridCount);
                    update_grids(curGrids, newGrids, r, c, curGridCount);
                    std::swap(curGrids, newGrids);
                    curGridCount = num;
                }
            }
        }
    }
    cudaEventRecord(stopAlgo);
    cudaEventSynchronize(stopAlgo);
    cudaEventElapsedTime(&algoTime, startAlgo, stopAlgo);
    return algoTime;
}


int main(void)
{
    //To measure time taken for initialization of data 
    cudaEvent_t startInit, stopInit;
    float initTime = 0, totalAlgoTime = 0;
    cudaEventCreate(&startInit);
    cudaEventCreate(&stopInit);
    cudaEventRecord(startInit);
    int n = 1000000;
    int** curGrids, **newGrids;
    alloc_mem(curGrids, n);
    alloc_mem(newGrids, n);
    int myGrids[16][81] ={{0,2,0,4,0,0,0,8,0,0,0,6,0,8,0,0,0,0,7,0,0,0,0,3,0,0,6,0,0,0,9,0,0,0,0,0,6,0,0,0,0,7,0,0,1,0,0,4,0,2,0,9,0,0,0,6,7,0,0,0,0,0,5,5,0,0,0,0,0,3,1,0,0,1,0,0,0,5,0,0,0},
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
    cudaEventElapsedTime(&initTime, startInit, stopInit);

    for(int i = 0; i < 16; i++)
    {
        std::cout <<"\nGrid "<<i + 1<< ": \n";
        display_grid(myGrids[i]);
        totalAlgoTime += solve(myGrids[i], curGrids, newGrids);
        std::cout << "\nSolved Grid:\n";
        display_grid(curGrids[0]);
    }
    std::cout << "\nTime taken for data generation: " << initTime << "ms"<< std::endl;
    std::cout << "Total time taken by the main algorithm: " << totalAlgoTime << "ms"<< std::endl;
    
    dealloc_mem(curGrids, n);
    dealloc_mem(newGrids, n);
}