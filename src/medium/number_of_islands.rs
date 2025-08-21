//! # Problem 200: Number of Islands
//!
//! Given an m x n 2D binary grid which represents a map of '1's (land) and '0's (water), 
//! return the number of islands.
//!
//! An island is surrounded by water and is formed by connecting adjacent lands horizontally 
//! or vertically. You may assume all four edges of the grid are all surrounded by water.
//!
//! ## Examples
//!
//! ```text
//! Input: grid = [
//!   ["1","1","1","1","0"],
//!   ["1","1","0","1","0"],
//!   ["1","1","0","0","0"],
//!   ["0","0","0","0","0"]
//! ]
//! Output: 1
//! ```
//!
//! ```text
//! Input: grid = [
//!   ["1","1","0","0","0"],
//!   ["1","1","0","0","0"],
//!   ["0","0","1","0","0"],
//!   ["0","0","0","1","1"]
//! ]
//! Output: 3
//! ```
//!
//! ## Constraints
//!
//! * m == grid.length
//! * n == grid[i].length
//! * 1 <= m, n <= 300
//! * grid[i][j] is '0' or '1'

use std::collections::VecDeque;

/// Solution for Number of Islands problem
pub struct Solution;

impl Solution {
    /// Creates a new instance of Solution
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: DFS with Grid Modification (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Iterate through each cell in the grid
    /// 2. When we find a '1', increment island count and start DFS
    /// 3. DFS marks all connected '1's as visited by changing them to '0'
    /// 4. Continue until all cells are processed
    /// 
    /// **Time Complexity:** O(m * n) - Visit each cell at most twice
    /// **Space Complexity:** O(min(m, n)) - DFS recursion depth in worst case
    /// 
    /// **Key Insights:**
    /// - Each '1' cell is visited exactly once across all DFS calls
    /// - Modifying grid in-place saves memory for visited tracking
    /// - 4-directional connectivity forms connected components
    /// 
    /// **Why this works:**
    /// - DFS explores entire connected component in one go
    /// - Marking visited cells prevents double counting
    /// - Each DFS call corresponds to exactly one island
    /// 
    /// **DFS traversal pattern:**
    /// ```text
    /// Start at (i,j) if grid[i][j] == '1'
    /// Mark current cell as '0' (visited)
    /// Recursively visit: up, down, left, right neighbors
    /// ```
    pub fn num_islands(&self, grid: Vec<Vec<char>>) -> i32 {
        if grid.is_empty() || grid[0].is_empty() {
            return 0;
        }
        
        let mut grid = grid;  // Make mutable for in-place modification
        let mut count = 0;
        let rows = grid.len();
        let cols = grid[0].len();
        
        for i in 0..rows {
            for j in 0..cols {
                if grid[i][j] == '1' {
                    count += 1;
                    self.dfs(&mut grid, i, j);
                }
            }
        }
        
        count
    }
    
    fn dfs(&self, grid: &mut Vec<Vec<char>>, i: usize, j: usize) {
        // Check bounds and if current cell is water or already visited
        if i >= grid.len() || j >= grid[0].len() || grid[i][j] != '1' {
            return;
        }
        
        // Mark current cell as visited
        grid[i][j] = '0';
        
        // Explore 4 directions
        if i > 0 { self.dfs(grid, i - 1, j); }  // Up
        if i + 1 < grid.len() { self.dfs(grid, i + 1, j); }  // Down
        if j > 0 { self.dfs(grid, i, j - 1); }  // Left
        if j + 1 < grid[0].len() { self.dfs(grid, i, j + 1); }  // Right
    }

    /// # Approach 2: BFS with Queue
    /// 
    /// **Algorithm:**
    /// 1. For each unvisited '1', start BFS to explore the entire island
    /// 2. Use queue to process cells level by level
    /// 3. Mark cells as visited by changing to '0'
    /// 4. Add all unvisited neighboring '1's to queue
    /// 
    /// **Time Complexity:** O(m * n) - Each cell visited once
    /// **Space Complexity:** O(min(m, n)) - Queue size at most min(m, n)
    /// 
    /// **Advantages:**
    /// - Iterative approach avoids recursion stack overflow
    /// - Level-by-level exploration
    /// - Better memory usage for very large grids
    /// 
    /// **When to use:** Very large grids where recursion depth is a concern
    pub fn num_islands_bfs(&self, grid: Vec<Vec<char>>) -> i32 {
        if grid.is_empty() || grid[0].is_empty() {
            return 0;
        }
        
        let mut grid = grid;
        let mut count = 0;
        let rows = grid.len();
        let cols = grid[0].len();
        
        for i in 0..rows {
            for j in 0..cols {
                if grid[i][j] == '1' {
                    count += 1;
                    self.bfs(&mut grid, i, j);
                }
            }
        }
        
        count
    }
    
    fn bfs(&self, grid: &mut Vec<Vec<char>>, start_i: usize, start_j: usize) {
        let mut queue = VecDeque::new();
        queue.push_back((start_i, start_j));
        grid[start_i][start_j] = '0';  // Mark as visited
        
        let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];  // right, down, left, up
        
        while let Some((i, j)) = queue.pop_front() {
            for (di, dj) in directions.iter() {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                
                if ni >= 0 && ni < grid.len() as i32 && nj >= 0 && nj < grid[0].len() as i32 {
                    let ni = ni as usize;
                    let nj = nj as usize;
                    
                    if grid[ni][nj] == '1' {
                        grid[ni][nj] = '0';  // Mark as visited
                        queue.push_back((ni, nj));
                    }
                }
            }
        }
    }

    /// # Approach 3: Union-Find (Disjoint Set)
    /// 
    /// **Algorithm:**
    /// 1. Create union-find structure for all land cells
    /// 2. Union adjacent land cells
    /// 3. Count number of distinct components
    /// 
    /// **Time Complexity:** O(m * n * α(m * n)) - Nearly linear with inverse Ackermann
    /// **Space Complexity:** O(m * n) - Union-find structure
    /// 
    /// **Advantages:**
    /// - Good for dynamic connectivity queries
    /// - Supports efficient union and find operations
    /// - Extensible to more complex connectivity problems
    /// 
    /// **When useful:** When you need to support dynamic updates to the grid
    pub fn num_islands_union_find(&self, grid: Vec<Vec<char>>) -> i32 {
        if grid.is_empty() || grid[0].is_empty() {
            return 0;
        }
        
        let rows = grid.len();
        let cols = grid[0].len();
        let mut uf = UnionFind::new(rows * cols);
        
        // Connect adjacent land cells
        for i in 0..rows {
            for j in 0..cols {
                if grid[i][j] == '1' {
                    let current = i * cols + j;
                    
                    // Check right neighbor
                    if j + 1 < cols && grid[i][j + 1] == '1' {
                        uf.union(current, i * cols + j + 1);
                    }
                    
                    // Check down neighbor
                    if i + 1 < rows && grid[i + 1][j] == '1' {
                        uf.union(current, (i + 1) * cols + j);
                    }
                }
            }
        }
        
        // Count distinct components among land cells
        let mut unique_roots = std::collections::HashSet::new();
        for i in 0..rows {
            for j in 0..cols {
                if grid[i][j] == '1' {
                    unique_roots.insert(uf.find(i * cols + j));
                }
            }
        }
        
        unique_roots.len() as i32
    }

    /// # Approach 4: DFS with Separate Visited Array
    /// 
    /// **Algorithm:**
    /// 1. Use separate boolean array to track visited cells
    /// 2. DFS explores connected components without modifying original grid
    /// 3. Preserves original grid structure
    /// 
    /// **Time Complexity:** O(m * n) - Each cell visited once
    /// **Space Complexity:** O(m * n) - Visited array + recursion stack
    /// 
    /// **Advantages:**
    /// - Preserves original grid
    /// - Clear separation of algorithm and data
    /// - Easier to debug and test
    /// 
    /// **When to use:** When original grid must remain unchanged
    pub fn num_islands_visited_array(&self, grid: Vec<Vec<char>>) -> i32 {
        if grid.is_empty() || grid[0].is_empty() {
            return 0;
        }
        
        let rows = grid.len();
        let cols = grid[0].len();
        let mut visited = vec![vec![false; cols]; rows];
        let mut count = 0;
        
        for i in 0..rows {
            for j in 0..cols {
                if grid[i][j] == '1' && !visited[i][j] {
                    count += 1;
                    self.dfs_visited(&grid, &mut visited, i, j);
                }
            }
        }
        
        count
    }
    
    fn dfs_visited(&self, grid: &Vec<Vec<char>>, visited: &mut Vec<Vec<bool>>, i: usize, j: usize) {
        if i >= grid.len() || j >= grid[0].len() || visited[i][j] || grid[i][j] != '1' {
            return;
        }
        
        visited[i][j] = true;
        
        // Explore 4 directions
        if i > 0 { self.dfs_visited(grid, visited, i - 1, j); }
        if i + 1 < grid.len() { self.dfs_visited(grid, visited, i + 1, j); }
        if j > 0 { self.dfs_visited(grid, visited, i, j - 1); }
        if j + 1 < grid[0].len() { self.dfs_visited(grid, visited, i, j + 1); }
    }

    /// # Approach 5: Iterative DFS with Stack
    /// 
    /// **Algorithm:**
    /// 1. Use explicit stack instead of recursion
    /// 2. Push neighboring land cells to stack
    /// 3. Process stack until empty for each island
    /// 
    /// **Time Complexity:** O(m * n) - Each cell visited once
    /// **Space Complexity:** O(min(m, n)) - Stack size
    /// 
    /// **Advantages:**
    /// - Avoids recursion stack overflow
    /// - More control over traversal order
    /// - Can be optimized for specific patterns
    /// 
    /// **When useful:** Very deep islands or when recursion is not preferred
    pub fn num_islands_iterative_dfs(&self, grid: Vec<Vec<char>>) -> i32 {
        if grid.is_empty() || grid[0].is_empty() {
            return 0;
        }
        
        let mut grid = grid;
        let mut count = 0;
        let rows = grid.len();
        let cols = grid[0].len();
        
        for i in 0..rows {
            for j in 0..cols {
                if grid[i][j] == '1' {
                    count += 1;
                    
                    let mut stack = Vec::new();
                    stack.push((i, j));
                    
                    while let Some((x, y)) = stack.pop() {
                        if x >= rows || y >= cols || grid[x][y] != '1' {
                            continue;
                        }
                        
                        grid[x][y] = '0';  // Mark as visited
                        
                        // Add neighbors to stack
                        if x > 0 { stack.push((x - 1, y)); }
                        if x + 1 < rows { stack.push((x + 1, y)); }
                        if y > 0 { stack.push((x, y - 1)); }
                        if y + 1 < cols { stack.push((x, y + 1)); }
                    }
                }
            }
        }
        
        count
    }

    /// # Approach 6: Flood Fill with Component Labeling
    /// 
    /// **Algorithm:**
    /// 1. Assign unique labels to each connected component
    /// 2. Use flood fill to propagate labels
    /// 3. Count distinct labels used
    /// 
    /// **Time Complexity:** O(m * n) - Each cell visited once
    /// **Space Complexity:** O(m * n) - Label array
    /// 
    /// **Additional benefits:**
    /// - Provides component identification beyond just counting
    /// - Useful for subsequent queries about specific islands
    /// - Can track island sizes and properties
    /// 
    /// **When useful:** When you need to identify or query specific islands later
    pub fn num_islands_component_labeling(&self, grid: Vec<Vec<char>>) -> i32 {
        if grid.is_empty() || grid[0].is_empty() {
            return 0;
        }
        
        let rows = grid.len();
        let cols = grid[0].len();
        let mut labels = vec![vec![0; cols]; rows];
        let mut label_count = 0;
        
        for i in 0..rows {
            for j in 0..cols {
                if grid[i][j] == '1' && labels[i][j] == 0 {
                    label_count += 1;
                    self.flood_fill(&grid, &mut labels, i, j, label_count);
                }
            }
        }
        
        label_count
    }
    
    fn flood_fill(&self, grid: &Vec<Vec<char>>, labels: &mut Vec<Vec<i32>>, 
                  i: usize, j: usize, label: i32) {
        if i >= grid.len() || j >= grid[0].len() || 
           grid[i][j] != '1' || labels[i][j] != 0 {
            return;
        }
        
        labels[i][j] = label;
        
        // Flood fill in 4 directions
        if i > 0 { self.flood_fill(grid, labels, i - 1, j, label); }
        if i + 1 < grid.len() { self.flood_fill(grid, labels, i + 1, j, label); }
        if j > 0 { self.flood_fill(grid, labels, i, j - 1, label); }
        if j + 1 < grid[0].len() { self.flood_fill(grid, labels, i, j + 1, label); }
    }
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }
    
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);  // Path compression
        }
        self.parent[x]
    }
    
    fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);
        
        if root_x != root_y {
            match self.rank[root_x].cmp(&self.rank[root_y]) {
                std::cmp::Ordering::Less => self.parent[root_x] = root_y,
                std::cmp::Ordering::Greater => self.parent[root_y] = root_x,
                std::cmp::Ordering::Equal => {
                    self.parent[root_y] = root_x;
                    self.rank[root_x] += 1;
                }
            }
        }
    }
}

impl Default for Solution {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Solution {
        Solution::new()
    }

    #[test]
    fn test_basic_examples() {
        let solution = setup();
        
        // Example 1: Single large island
        let grid1 = vec![
            vec!['1','1','1','1','0'],
            vec!['1','1','0','1','0'],
            vec!['1','1','0','0','0'],
            vec!['0','0','0','0','0']
        ];
        assert_eq!(solution.num_islands(grid1), 1);
        
        // Example 2: Multiple islands
        let grid2 = vec![
            vec!['1','1','0','0','0'],
            vec!['1','1','0','0','0'],
            vec!['0','0','1','0','0'],
            vec!['0','0','0','1','1']
        ];
        assert_eq!(solution.num_islands(grid2), 3);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // All water
        let all_water = vec![
            vec!['0','0','0'],
            vec!['0','0','0']
        ];
        assert_eq!(solution.num_islands(all_water), 0);
        
        // All land
        let all_land = vec![
            vec!['1','1'],
            vec!['1','1']
        ];
        assert_eq!(solution.num_islands(all_land), 1);
        
        // Single cell water
        let single_water = vec![vec!['0']];
        assert_eq!(solution.num_islands(single_water), 0);
        
        // Single cell land
        let single_land = vec![vec!['1']];
        assert_eq!(solution.num_islands(single_land), 1);
        
        // Single row
        let single_row = vec![vec!['1','0','1','0','1']];
        assert_eq!(solution.num_islands(single_row), 3);
        
        // Single column
        let single_col = vec![
            vec!['1'],
            vec!['0'],
            vec!['1'],
            vec!['0'],
            vec!['1']
        ];
        assert_eq!(solution.num_islands(single_col), 3);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        let test_grids = vec![
            vec![
                vec!['1','1','1','1','0'],
                vec!['1','1','0','1','0'],
                vec!['1','1','0','0','0'],
                vec!['0','0','0','0','0']
            ],
            vec![
                vec!['1','1','0','0','0'],
                vec!['1','1','0','0','0'],
                vec!['0','0','1','0','0'],
                vec!['0','0','0','1','1']
            ],
            vec![
                vec!['1','0','1'],
                vec!['0','1','0'],
                vec!['1','0','1']
            ],
            vec![vec!['1']],
            vec![vec!['0']],
        ];
        
        for grid in test_grids {
            let result1 = solution.num_islands(grid.clone());
            let result2 = solution.num_islands_bfs(grid.clone());
            let result3 = solution.num_islands_union_find(grid.clone());
            let result4 = solution.num_islands_visited_array(grid.clone());
            let result5 = solution.num_islands_iterative_dfs(grid.clone());
            let result6 = solution.num_islands_component_labeling(grid.clone());
            
            assert_eq!(result1, result2, "DFS vs BFS mismatch");
            assert_eq!(result2, result3, "BFS vs Union-Find mismatch");
            assert_eq!(result3, result4, "Union-Find vs Visited Array mismatch");
            assert_eq!(result4, result5, "Visited Array vs Iterative DFS mismatch");
            assert_eq!(result5, result6, "Iterative DFS vs Component Labeling mismatch");
        }
    }

    #[test]
    fn test_complex_shapes() {
        let solution = setup();
        
        // L-shaped island
        let l_shape = vec![
            vec!['1','0','0'],
            vec!['1','0','0'],
            vec!['1','1','1']
        ];
        assert_eq!(solution.num_islands(l_shape), 1);
        
        // C-shaped island
        let c_shape = vec![
            vec!['1','1','1'],
            vec!['1','0','0'],
            vec!['1','1','1']
        ];
        assert_eq!(solution.num_islands(c_shape), 1);
        
        // Snake-like island
        let snake = vec![
            vec!['1','0','0','0'],
            vec!['1','1','0','0'],
            vec!['0','1','1','0'],
            vec!['0','0','1','1']
        ];
        assert_eq!(solution.num_islands(snake), 1);
    }

    #[test]
    fn test_diagonal_separation() {
        let solution = setup();
        
        // Diagonally adjacent but not connected
        let diagonal = vec![
            vec!['1','0','1'],
            vec!['0','1','0'],
            vec!['1','0','1']
        ];
        assert_eq!(solution.num_islands(diagonal), 5);
        
        // Cross pattern
        let cross = vec![
            vec!['0','1','0'],
            vec!['1','1','1'],
            vec!['0','1','0']
        ];
        assert_eq!(solution.num_islands(cross), 1);
    }

    #[test]
    fn test_large_grids() {
        let solution = setup();
        
        // Large grid with checkerboard pattern
        let size = 10;
        let mut checkerboard = vec![vec!['0'; size]; size];
        for i in 0..size {
            for j in 0..size {
                if (i + j) % 2 == 0 {
                    checkerboard[i][j] = '1';
                }
            }
        }
        // Checkerboard creates many single-cell islands
        assert_eq!(solution.num_islands(checkerboard), 50);
        
        // Large single island
        let large_island = vec![vec!['1'; 20]; 20];
        assert_eq!(solution.num_islands(large_island), 1);
    }

    #[test]
    fn test_boundary_islands() {
        let solution = setup();
        
        // Islands on edges
        let edge_islands = vec![
            vec!['1','0','0','0','1'],
            vec!['0','0','0','0','0'],
            vec!['0','0','1','0','0'],
            vec!['0','0','0','0','0'],
            vec!['1','0','0','0','1']
        ];
        assert_eq!(solution.num_islands(edge_islands), 5);
        
        // Border surrounding water
        let border = vec![
            vec!['1','1','1','1'],
            vec!['1','0','0','1'],
            vec!['1','0','0','1'],
            vec!['1','1','1','1']
        ];
        assert_eq!(solution.num_islands(border), 1);
    }

    #[test]
    fn test_narrow_strips() {
        let solution = setup();
        
        // Thin horizontal strip
        let horizontal = vec![vec!['1','1','1','1','1']];
        assert_eq!(solution.num_islands(horizontal), 1);
        
        // Thin vertical strip
        let vertical = vec![
            vec!['1'],
            vec!['1'],
            vec!['1'],
            vec!['1']
        ];
        assert_eq!(solution.num_islands(vertical), 1);
        
        // Broken strips
        let broken_horizontal = vec![vec!['1','1','0','1','1']];
        assert_eq!(solution.num_islands(broken_horizontal), 2);
    }

    #[test]
    fn test_disconnected_components() {
        let solution = setup();
        
        // Four corners
        let four_corners = vec![
            vec!['1','0','1'],
            vec!['0','0','0'],
            vec!['1','0','1']
        ];
        assert_eq!(solution.num_islands(four_corners), 4);
        
        // Multiple disconnected complex shapes
        let complex = vec![
            vec!['1','1','0','0','1','1'],
            vec!['1','0','0','0','0','1'],
            vec!['0','0','1','1','0','0'],
            vec!['0','0','1','1','0','0'],
            vec!['1','0','0','0','0','1'],
            vec!['1','1','0','0','1','1']
        ];
        assert_eq!(solution.num_islands(complex), 5);
    }

    #[test]
    fn test_performance_characteristics() {
        let solution = setup();
        
        // Dense grid (many small islands)
        let mut dense = vec![vec!['0'; 50]; 50];
        for i in (0..50).step_by(2) {
            for j in (0..50).step_by(2) {
                dense[i][j] = '1';
            }
        }
        let result = solution.num_islands(dense);
        assert_eq!(result, 625); // 25x25 islands
        
        // Sparse grid (few large islands)
        let mut sparse = vec![vec!['0'; 50]; 50];
        for i in 0..10 {
            for j in 0..10 {
                sparse[i][j] = '1';
            }
        }
        for i in 40..50 {
            for j in 40..50 {
                sparse[i][j] = '1';
            }
        }
        assert_eq!(solution.num_islands(sparse), 2);
    }

    #[test]
    fn test_mathematical_properties() {
        let solution = setup();
        
        // Property: Islands count <= total land cells
        let grid = vec![
            vec!['1','1','0'],
            vec!['0','1','1'],
            vec!['1','0','1']
        ];
        let islands = solution.num_islands(grid.clone());
        let land_cells = grid.iter()
            .flat_map(|row| row.iter())
            .filter(|&&cell| cell == '1')
            .count();
        assert!(islands <= land_cells as i32);
        
        // Property: Empty grid has 0 islands
        let empty = vec![vec!['0'; 5]; 5];
        assert_eq!(solution.num_islands(empty), 0);
        
        // Property: Single connected component
        let connected = vec![
            vec!['1','1','1'],
            vec!['1','1','1'],
            vec!['1','1','1']
        ];
        assert_eq!(solution.num_islands(connected), 1);
    }
}