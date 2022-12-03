#include<iostream>
#include<stdio.h>
#include<cstdlib>
#include<algorithm>
#include<string>
void display_grid(int *h_grid)
{
    for(int i = 0; i < 9; i++)
    {
        for(int j = 0; j < 9; j++)
            std::cout << h_grid[9*i + j] << " ";
        std::cout << std::endl;
    }
}

bool is_safe(int* h_grid, int val, int row, int col)
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

void update_grids(int **cur_grids, int **new_grids, int r, int c, int grids)
{
    int idx = 0;
    for(int k = 0; k < grids; k++)
    {  
        for(int i = 1; i <= 9; i++)
        {
            if(is_safe(cur_grids[k], i, r, c))
            {
                cur_grids[k][r*9 + c] = i;
                std::copy(cur_grids[k], cur_grids[k] + 81, new_grids[idx++]);
            }
        }
    }
}

int no_of_possible_grids(int** h_grids, int row, int col, int grids)
{
    int res = 0;
    for(int k = 0; k < grids; k++)
    {
        int mp[10] = {0};
        for(int i = 0; i < 9; i++)
        {
            int row_cell = h_grids[k][9*i + col];
            int col_cell = h_grids[k][9*row + i];
            mp[row_cell]++;
            mp[col_cell]++;
        }
        int start_row = row - row % 3;
        int start_col = col - col % 3;
        for(int i = start_row; i < start_row + 3; i++)
        {
            for(int j = start_col; j < start_col + 3; j++)
            {
                int cell = h_grids[k][9*i + j];
                mp[cell]++;
            }
        }
        for(int i = 1; i < 10; i++)
            if(mp[i] == 0) res++;
    }
    return res;
    
}
bool is_filled(int* h_grid)
{
    for(int r = 0; r < 9; r++)
    {
        for(int c = 0; c < 9; c++)
        {
            if(h_grid[r*9 + c] == 0) 
                return false;
        }
    }
    return true;
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

float solve(int (&h_grid)[81], int**&cur_grids, int**&new_grids)
{
    //To measure time the by the main algorithm
    cudaEvent_t start_algo, stop_algo;
    cudaEventCreate(&start_algo);
    cudaEventCreate(&stop_algo);
    float algo_time = 0;
    int no_of_grids = 1;
    std::copy(h_grid, h_grid + 81, cur_grids[0]);
    cudaEventRecord(start_algo);
    for(int i = 0; i < no_of_grids; i++)
    {
        for(int r = 0; r < 9; r++)
        {
            for(int c = 0; c < 9; c++)
            {
                if(cur_grids[i][r*9 + c] == 0)
                {
                    int num = no_of_possible_grids(cur_grids, r, c, no_of_grids);
                    update_grids(cur_grids, new_grids, r, c, no_of_grids);
                    std::swap(cur_grids, new_grids);
                    no_of_grids = num;
                }
            }
        }
    }
    cudaEventRecord(stop_algo);
    cudaEventSynchronize(stop_algo);
    cudaEventElapsedTime(&algo_time, start_algo, stop_algo);
    return algo_time;
}


int main(void)
{
    //To measure time taken for initialization of data 
    cudaEvent_t start_init, stop_init;
    float init_time = 0, total_algo_time = 0;
    cudaEventCreate(&start_init);
    cudaEventCreate(&stop_init);
    cudaEventRecord(start_init);
    int n = 1000000;
    int** cur_grids, **new_grids;
    int* startPositions;
    alloc_mem(cur_grids, n);
    alloc_mem(new_grids, n);
    startPositions = new int[n];
    int myGrids[6][81] = {{5,8,6,0,7,0,0,0,0,0,0,0,9,0,1,6,0,0,0,0,0,6,0,0,0,0,0,0,0,7,0,0,0,0,0,0,9,0,2,0,1,0,3,0,5,0,0,5,0,9,0,0,0,0,0,9,0,0,4,0,0,0,8,0,0,3,5,0,0,0,0,6,0,0,0,0,2,0,4,7,0},
                         {0,7,0,0,0,0,0,4,3,0,4,0,0,0,9,6,1,0,8,0,0,6,3,4,9,0,0,0,9,4,0,5,2,0,0,0,3,5,8,4,6,0,0,2,0,0,0,0,8,0,0,5,3,0,0,8,0,0,7,0,0,9,1,9,0,2,1,0,0,0,0,5,0,0,7,0,4,0,8,0,2},
                         {3,0,1,0,8,6,5,0,4,0,4,6,5,2,1,0,7,0,5,0,0,0,0,0,0,0,1,4,0,0,8,0,0,0,0,2,0,8,0,3,4,7,9,0,0,0,0,9,0,5,0,0,3,8,0,0,4,0,9,0,2,0,0,0,0,8,7,3,4,0,9,0,0,0,7,2,0,8,1,0,3},
                         {0,4,8,3,0,1,5,6,0,3,6,0,0,0,8,0,9,0,9,1,0,6,7,0,0,0,3,0,2,0,0,0,0,9,3,5,5,0,9,0,1,0,2,0,0,6,7,0,0,2,0,0,1,0,0,0,4,0,0,2,1,0,7,0,9,0,1,0,0,0,0,8,1,5,0,8,3,4,0,2,9},
                         {0,0,0,0,0,0,0,0,8,0,0,3,0,0,0,4,0,0,0,9,0,0,2,0,0,6,0,0,0,0,0,7,9,0,0,0,0,0,0,0,6,1,2,0,0,0,6,0,5,0,2,0,7,0,0,0,8,0,0,0,5,0,0,0,1,0,0,0,0,0,2,0,4,0,5,0,0,0,0,0,3},
                         {0,0,0,0,0,0,0,0,2,0,0,8,0,1,0,9,0,0,5,0,0,0,0,3,0,4,0,0,0,0,1,0,9,3,0,0,0,6,0,0,3,0,0,8,0,0,0,3,7,0,0,0,0,0,0,4,0,0,0,0,0,0,5,3,0,1,0,7,0,8,0,0,2,0,0,0,0,0,0,0,0}};
    cudaEventRecord(stop_init);
    cudaEventElapsedTime(&init_time, start_init, stop_init);

    for(int i = 0; i < 6; i++)
    {
        std::cout <<"\nGrid "<<i + 1<< ": \n";
        display_grid(myGrids[i]);
        total_algo_time += solve(myGrids[i], cur_grids, new_grids);
        std::cout << "\nSolved Grid:\n";
        display_grid(cur_grids[0]);
    }
    std::cout << "\nTime taken for data generation: " << init_time << "ms"<< std::endl;
    std::cout << "Total time taken by the main algorithm: " << total_algo_time << "ms"<< std::endl;
    
    dealloc_mem(cur_grids, n);
    dealloc_mem(new_grids, n);
    delete[] startPositions;
}