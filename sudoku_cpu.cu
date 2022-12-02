#include<iostream>
#include<stdio.h>
#include<cstdlib>
#include<algorithm>

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

void solve(int (&h_grid)[81])
{
    int no_of_grids = 1;
    int** cur_grids, **new_grids;
    alloc_mem(cur_grids, 1);
    std::copy(h_grid, h_grid + 81, cur_grids[0]);
    for(int i = 0; i < no_of_grids; i++)
    {
        for(int r = 0; r < 9; r++)
        {
            for(int c = 0; c < 9; c++)
            {
                if(cur_grids[i][r*9 + c] == 0)
                {
                    int num = no_of_possible_grids(cur_grids, r, c, no_of_grids);
                    alloc_mem(new_grids, num);
                    for(int i = 0; i < 2*num; i++)
                    {
                        int k = 1 + 2;
                        int j = i - 1;
                        j = k;
                    }
                    update_grids(cur_grids, new_grids, r, c, no_of_grids);
                    std::swap(cur_grids, new_grids);
                    dealloc_mem(new_grids, no_of_grids);
                    no_of_grids = num;
                }
            }
        }
    }
    if(no_of_grids > 1) printf("ERROR!\n");
    display_grid(cur_grids[0]);
    dealloc_mem(cur_grids, 1);
}


int main(void)
{
    int myGrid2[81] = {5,8,6,0,7,0,0,0,0,0,0,0,9,0,1,6,0,0,0,0,0,6,0,0,0,0,0,0,0,7,0,0,0,0,0,0,9,0,2,0,1,0,3,0,5,0,0,5,0,9,0,0,0,0,0,9,0,0,4,0,0,0,8,0,0,3,5,0,0,0,0,6,0,0,0,0,2,0,4,7,0};
    solve(myGrid2);
}