# Sudoku Solver

## Description
This is project consists of two implementations of a sudoku solver, one uses only the CPU wheras the other uses both the CPU and the GPU.

## Time difference
The CPU version takes around 25 seconds whereas the parallel version takes around 4 seconds, to solve 16 of the hardest puzzles collected from [this](http://forum.enjoysudoku.com/the-hardest-sudokus-new-thread-t6539.html#p65791) dataset.

## Algorithm and implementation
The algorithm uses Breadth First Search(BFS), where based on the initial grid, new possible grids are computed and stored in a data structure and those new grids in turn generate more grids and are stored and the process repeats. 
The new grids are computed by taking the first empty cell of the grid, computing the different possible numbers that could be inserted to that cell and just saving those grids.

One thread is assigned to each existing grid which adds the new grids formed its grid to the list of new grids. The challenging part about this is figuring out how to store the new grids in a data structure asynchonously. The way I did this, was to calculate the new grids formed by each existing grid and do an exclusive scan (prefix sum - value of element in the cell) which gives us the start positions of where the new grids would have to be inserted for the current grid.