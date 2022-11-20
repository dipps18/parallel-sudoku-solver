#include<iostream>
#include<stdio.h>
#include<cstdlib>
#include<algorithm>
#include<thrust/host_vector.h>

bool is_safe(int* h_grid, int val, int row, int col)
{
    for(int i = 0; i < 9; i++)
    {
        if(i != row && h_grid[9*i + col] == val) 
            return false;
        if(i != col && h_grid[row*9 + i])
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

void update_grids(int (&cur_grid)[81], int (*new_grids)[81], int r, int c)
{
    int idx = 0;
    for(int i = 1; i <= 9; i++)
    {
        if(is_safe(cur_grid, i, r, c))
        {
            cur_grid[r*9 + c] = i;
            std::copy(cur_grid, cur_grid + 81, new_grids[idx++]);
        }
    }
}


int no_of_possible_grids(int (&h_grid)[81], int row, int col)
{
    int mp[10] = {0};
    int res = 0;
    for(int i = 0; i < 9; i++)
    {
        int row_cell = h_grid[9*i + col];
        int col_cell = h_grid[9*row + row];
        mp[row_cell]++;
        mp[col_cell]++;
    }
    int start_row = row - row % 3;
    int start_col = col - col % 3;
    for(int i = start_row; i < start_row + 3; i++)
    {
        for(int j = start_col; j < start_col + 3; j++)
        {
            int cell = h_grid[9*i + j];
            mp[cell]++;
        }
    }
    for(int i = 1; i < 10; i++)
        if(mp[i] == 0) res++;
    return res;
    
}
bool is_filled(int (&h_grid)[81])
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

void display_grid(int* h_grid)
{
    for(int i = 0; i < 9; i++)
    {
        for(int j = 0; j < 9; j++)
        {
            std::cout << h_grid[9*i + j] << " ";
        }
        std::cout << std::endl;
    }
}

void solve(int (&h_grid)[81])
{
    int (*cur_grids)[81] = new int[1][81];
    std::copy(h_grid, h_grid + 81, cur_grids[0]);
    int (*new_grids)[81];
    int no_of_grids = 1;
    while(no_of_grids > 0)
    {
        for(int i = 0; i < no_of_grids; i++)
        {
            for(int r = 0; r < 9; r++)
            {
                for(int c = 0; c < 9; c++)
                {
                    if(cur_grids[i][r*9 + c] == 0)
                        continue;
                    else
                    {
                        int num = no_of_possible_grids(cur_grids[i], r, c);
                        new_grids = new int[num][81];
                        update_grids(cur_grids[i], new_grids, r, c);
                        std::swap(cur_grids, new_grids);
                        free(new_grids);
                    }
                }
            }
        }
        no_of_grids = sizeof(cur_grids)/sizeof(cur_grids[0]);
        if(no_of_grids == 1 && is_filled(cur_grids[0])) break; 
    }
    display_grid(cur_grids[0]);
}


int main(void)
{
    int h_grid[81]={3, 0, 6, 5, 0, 8, 4, 0, 0 ,5, 2, 0, 0, 0, 0, 0, 0, 0 ,0, 8, 7, 0, 0, 0, 0, 3, 1 ,0, 0, 3, 0, 1, 0, 0, 8, 0 ,9, 0, 0, 8, 6, 3, 0, 0, 5,0, 5, 0, 0, 9, 0, 6, 0, 0 ,1, 3, 0, 0, 0, 0, 2, 5, 0 ,0, 0, 0, 0, 0, 0, 0, 7, 4 ,0, 0, 5, 2, 0, 6, 3, 0, 0};
    //display_grid(h_grid);
    solve(h_grid);
}