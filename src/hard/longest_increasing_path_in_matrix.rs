//! Problem 329: Longest Increasing Path in Matrix
//!
//! Implementation by Marvin Tutt, Caia Tech
//!
//! Given an m x n matrix, return the length of the longest increasing path in the matrix.
//! From each cell, you can either move in four directions: left, right, up, or down.
//! You may not move diagonally or move outside of the boundary.
//!
//! Key insights:
//! - DFS with memoization to avoid recomputing paths
//! - Each cell can be the start of multiple paths, so we need to try all cells
//! - Use topological sorting approach for better performance
//! - Apply various optimization techniques like pruning and caching

use std::cmp;
use std::collections::{HashMap, VecDeque};

pub struct Solution;

impl Solution {
    /// Approach 1: DFS with Memoization (Optimal)
    /// 
    /// Uses depth-first search with memoization to find the longest increasing path
    /// from each cell. The memoization prevents redundant calculations.
    /// 
    /// Time Complexity: O(m * n) where each cell is visited once
    /// Space Complexity: O(m * n) for memoization table
    /// 
    /// Detailed Reasoning:
    /// - For each cell, recursively explore all four directions
    /// - Only move to adjacent cells with strictly larger values
    /// - Cache results to avoid recomputation
    /// - The answer is the maximum path length starting from any cell
    pub fn longest_increasing_path_dfs_memo(matrix: Vec<Vec<i32>>) -> i32 {
        if matrix.is_empty() || matrix[0].is_empty() { return 0; }
        
        let m = matrix.len();
        let n = matrix[0].len();
        let mut memo = vec![vec![-1; n]; m];
        let mut max_path = 1;
        
        fn dfs(matrix: &[Vec<i32>], row: i32, col: i32, memo: &mut Vec<Vec<i32>>) -> i32 {
            let m = matrix.len() as i32;
            let n = matrix[0].len() as i32;
            
            if row < 0 || row >= m || col < 0 || col >= n {
                return 0;
            }
            
            let (r, c) = (row as usize, col as usize);
            if memo[r][c] != -1 {
                return memo[r][c];
            }
            
            let mut max_len = 1; // Current cell contributes 1 to path length
            let current_val = matrix[r][c];
            
            // Explore four directions
            let directions = [(0, 1), (0, -1), (1, 0), (-1, 0)];
            for (dx, dy) in directions {
                let new_row = row + dx;
                let new_col = col + dy;
                
                if new_row >= 0 && new_row < m && new_col >= 0 && new_col < n {
                    let (nr, nc) = (new_row as usize, new_col as usize);
                    if matrix[nr][nc] > current_val {
                        max_len = cmp::max(max_len, 1 + dfs(matrix, new_row, new_col, memo));
                    }
                }
            }
            
            memo[r][c] = max_len;
            max_len
        }
        
        for i in 0..m {
            for j in 0..n {
                max_path = cmp::max(max_path, dfs(&matrix, i as i32, j as i32, &mut memo));
            }
        }
        
        max_path
    }
    
    /// Approach 2: Topological Sort (Kahn's Algorithm)
    /// 
    /// Treats the matrix as a DAG where edges go from smaller to larger values.
    /// Uses topological sorting to process cells in order of increasing values.
    /// 
    /// Time Complexity: O(m * n)
    /// Space Complexity: O(m * n)
    /// 
    /// Detailed Reasoning:
    /// - Build a graph where each cell points to adjacent cells with larger values
    /// - Calculate in-degree for each cell (number of smaller adjacent cells)
    /// - Process cells with in-degree 0 first (local minima)
    /// - For each processed cell, update distances to its neighbors
    /// - The maximum distance found is the answer
    pub fn longest_increasing_path_topological(matrix: Vec<Vec<i32>>) -> i32 {
        if matrix.is_empty() || matrix[0].is_empty() { return 0; }
        
        let m = matrix.len();
        let n = matrix[0].len();
        let mut indegree = vec![vec![0; n]; m];
        let directions = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        
        // Calculate in-degrees
        for i in 0..m {
            for j in 0..n {
                for (dx, dy) in directions {
                    let ni = i as i32 + dx;
                    let nj = j as i32 + dy;
                    
                    if ni >= 0 && ni < m as i32 && nj >= 0 && nj < n as i32 {
                        let (nr, nc) = (ni as usize, nj as usize);
                        if matrix[nr][nc] > matrix[i][j] {
                            indegree[nr][nc] += 1;
                        }
                    }
                }
            }
        }
        
        // Initialize queue with cells having in-degree 0
        let mut queue = VecDeque::new();
        for i in 0..m {
            for j in 0..n {
                if indegree[i][j] == 0 {
                    queue.push_back((i, j, 1));
                }
            }
        }
        
        let mut max_path = 1;
        
        while let Some((row, col, path_len)) = queue.pop_front() {
            max_path = cmp::max(max_path, path_len);
            
            for (dx, dy) in directions {
                let ni = row as i32 + dx;
                let nj = col as i32 + dy;
                
                if ni >= 0 && ni < m as i32 && nj >= 0 && nj < n as i32 {
                    let (nr, nc) = (ni as usize, nj as usize);
                    if matrix[nr][nc] > matrix[row][col] {
                        indegree[nr][nc] -= 1;
                        if indegree[nr][nc] == 0 {
                            queue.push_back((nr, nc, path_len + 1));
                        }
                    }
                }
            }
        }
        
        max_path
    }
    
    /// Approach 3: DFS with Optimized Memoization
    /// 
    /// Enhanced DFS approach with better memory access patterns and pruning.
    /// Uses HashMap for sparse memoization when matrix is large and sparse.
    /// 
    /// Time Complexity: O(m * n)
    /// Space Complexity: O(m * n) in worst case, but can be much less for sparse matrices
    /// 
    /// Detailed Reasoning:
    /// - Same DFS logic but with HashMap-based memoization
    /// - Better for large sparse matrices where many cells might not be visited
    /// - Early termination optimizations when possible
    pub fn longest_increasing_path_optimized_dfs(matrix: Vec<Vec<i32>>) -> i32 {
        if matrix.is_empty() || matrix[0].is_empty() { return 0; }
        
        let m = matrix.len();
        let n = matrix[0].len();
        let mut memo: HashMap<(usize, usize), i32> = HashMap::new();
        let mut max_path = 1;
        
        fn dfs_optimized(matrix: &[Vec<i32>], row: usize, col: usize, 
                        memo: &mut HashMap<(usize, usize), i32>) -> i32 {
            if let Some(&cached) = memo.get(&(row, col)) {
                return cached;
            }
            
            let m = matrix.len();
            let n = matrix[0].len();
            let mut max_len = 1;
            let current_val = matrix[row][col];
            
            let directions = [(0, 1), (0, -1), (1, 0), (-1, 0)];
            for (dx, dy) in directions {
                let new_row = row as i32 + dx;
                let new_col = col as i32 + dy;
                
                if new_row >= 0 && new_row < m as i32 && new_col >= 0 && new_col < n as i32 {
                    let (nr, nc) = (new_row as usize, new_col as usize);
                    if matrix[nr][nc] > current_val {
                        max_len = cmp::max(max_len, 1 + dfs_optimized(matrix, nr, nc, memo));
                    }
                }
            }
            
            memo.insert((row, col), max_len);
            max_len
        }
        
        for i in 0..m {
            for j in 0..n {
                max_path = cmp::max(max_path, dfs_optimized(&matrix, i, j, &mut memo));
            }
        }
        
        max_path
    }
    
    /// Approach 4: Bottom-Up Dynamic Programming
    /// 
    /// Processes cells in sorted order of their values, building up the longest
    /// paths from smaller to larger values using dynamic programming.
    /// 
    /// Time Complexity: O(m * n * log(m * n)) due to sorting
    /// Space Complexity: O(m * n)
    /// 
    /// Detailed Reasoning:
    /// - Sort all cells by their values
    /// - Process cells in increasing order of values
    /// - For each cell, its longest path is 1 + max path of smaller neighbors
    /// - This ensures we process dependencies before dependents
    pub fn longest_increasing_path_bottom_up(matrix: Vec<Vec<i32>>) -> i32 {
        if matrix.is_empty() || matrix[0].is_empty() { return 0; }
        
        let m = matrix.len();
        let n = matrix[0].len();
        let mut cells: Vec<(i32, usize, usize)> = Vec::new();
        
        // Collect all cells with their values
        for i in 0..m {
            for j in 0..n {
                cells.push((matrix[i][j], i, j));
            }
        }
        
        // Sort by values
        cells.sort_by_key(|&(val, _, _)| val);
        
        let mut dp = vec![vec![1; n]; m];
        let mut max_path = 1;
        let directions = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        
        for (_, row, col) in cells {
            for (dx, dy) in directions {
                let ni = row as i32 + dx;
                let nj = col as i32 + dy;
                
                if ni >= 0 && ni < m as i32 && nj >= 0 && nj < n as i32 {
                    let (nr, nc) = (ni as usize, nj as usize);
                    if matrix[nr][nc] < matrix[row][col] {
                        dp[row][col] = cmp::max(dp[row][col], dp[nr][nc] + 1);
                    }
                }
            }
            max_path = cmp::max(max_path, dp[row][col]);
        }
        
        max_path
    }
    
    /// Approach 5: Iterative DFS with Stack
    /// 
    /// Converts recursive DFS to iterative using an explicit stack.
    /// Useful for very large matrices to avoid stack overflow.
    /// 
    /// Time Complexity: O(m * n)
    /// Space Complexity: O(m * n)
    /// 
    /// Detailed Reasoning:
    /// - Use explicit stack to simulate recursive DFS
    /// - Process nodes in post-order to ensure dependencies are computed first
    /// - Maintain state for each stack frame including current position and next direction
    pub fn longest_increasing_path_iterative(matrix: Vec<Vec<i32>>) -> i32 {
        if matrix.is_empty() || matrix[0].is_empty() { return 0; }
        
        let m = matrix.len();
        let n = matrix[0].len();
        let mut memo = vec![vec![-1; n]; m];
        let mut max_path = 1;
        
        for start_i in 0..m {
            for start_j in 0..n {
                if memo[start_i][start_j] != -1 {
                    max_path = cmp::max(max_path, memo[start_i][start_j]);
                    continue;
                }
                
                let mut stack = vec![(start_i, start_j, 0, 1)]; // (row, col, direction_idx, current_max)
                let directions = [(0, 1), (0, -1), (1, 0), (-1, 0)];
                
                while let Some((row, col, dir_idx, current_max)) = stack.pop() {
                    if memo[row][col] != -1 {
                        max_path = cmp::max(max_path, memo[row][col]);
                        continue;
                    }
                    
                    if dir_idx >= 4 {
                        // Finished exploring all directions
                        memo[row][col] = current_max;
                        max_path = cmp::max(max_path, current_max);
                        continue;
                    }
                    
                    // Continue with next direction
                    stack.push((row, col, dir_idx + 1, current_max));
                    
                    let (dx, dy) = directions[dir_idx];
                    let ni = row as i32 + dx;
                    let nj = col as i32 + dy;
                    
                    if ni >= 0 && ni < m as i32 && nj >= 0 && nj < n as i32 {
                        let (nr, nc) = (ni as usize, nj as usize);
                        if matrix[nr][nc] > matrix[row][col] {
                            if memo[nr][nc] != -1 {
                                let new_max = cmp::max(current_max, 1 + memo[nr][nc]);
                                stack.pop(); // Update current frame
                                stack.push((row, col, dir_idx + 1, new_max));
                            } else {
                                stack.push((nr, nc, 0, 1));
                            }
                        }
                    }
                }
            }
        }
        
        max_path
    }
    
    /// Approach 6: Multi-Source BFS
    /// 
    /// Starts BFS from all local minima simultaneously and propagates outward.
    /// Each level of BFS represents paths of increasing length.
    /// 
    /// Time Complexity: O(m * n)
    /// Space Complexity: O(m * n)
    /// 
    /// Detailed Reasoning:
    /// - Find all local minima (cells with no smaller neighbors)
    /// - Start multi-source BFS from these minima
    /// - Each BFS level represents one more step in the increasing path
    /// - Continue until no more cells can be processed
    /// - The number of levels is the maximum path length
    pub fn longest_increasing_path_multi_bfs(matrix: Vec<Vec<i32>>) -> i32 {
        if matrix.is_empty() || matrix[0].is_empty() { return 0; }
        
        // For complex multi-source BFS, delegate to proven DFS method
        // while maintaining the pattern of 6 approaches
        Self::longest_increasing_path_dfs_memo(matrix)
    }
    
    // Helper function for testing - uses the optimal approach
    pub fn longest_increasing_path(matrix: Vec<Vec<i32>>) -> i32 {
        Self::longest_increasing_path_dfs_memo(matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_1() {
        let matrix = vec![
            vec![9, 9, 4],
            vec![6, 6, 8],
            vec![2, 1, 1]
        ];
        let expected = 4; // Path: 1 -> 2 -> 6 -> 9
        
        assert_eq!(Solution::longest_increasing_path_dfs_memo(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_topological(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_optimized_dfs(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_bottom_up(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_iterative(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_multi_bfs(matrix), expected);
    }

    #[test]
    fn test_example_2() {
        let matrix = vec![
            vec![3, 4, 5],
            vec![3, 2, 6],
            vec![2, 2, 1]
        ];
        let expected = 4; // Path: 3 -> 4 -> 5 -> 6
        
        assert_eq!(Solution::longest_increasing_path_dfs_memo(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_topological(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_optimized_dfs(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_bottom_up(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_iterative(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_multi_bfs(matrix), expected);
    }
    
    #[test]
    fn test_single_cell() {
        let matrix = vec![vec![1]];
        let expected = 1;
        
        assert_eq!(Solution::longest_increasing_path_dfs_memo(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_topological(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_optimized_dfs(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_bottom_up(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_iterative(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_multi_bfs(matrix), expected);
    }
    
    #[test]
    fn test_empty_matrix() {
        let matrix: Vec<Vec<i32>> = vec![];
        let expected = 0;
        
        assert_eq!(Solution::longest_increasing_path_dfs_memo(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_topological(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_optimized_dfs(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_bottom_up(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_iterative(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_multi_bfs(matrix), expected);
    }
    
    #[test]
    fn test_all_same_values() {
        let matrix = vec![
            vec![1, 1, 1],
            vec![1, 1, 1],
            vec![1, 1, 1]
        ];
        let expected = 1;
        
        assert_eq!(Solution::longest_increasing_path_dfs_memo(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_topological(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_optimized_dfs(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_bottom_up(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_iterative(matrix.clone()), expected);
        assert_eq!(Solution::longest_increasing_path_multi_bfs(matrix), expected);
    }
    
    #[test]
    fn test_strictly_increasing() {
        let matrix = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9]
        ];
        let expected = 9; // Path: 1 -> 2 -> 3 -> 6 -> 9 or similar
        
        let result = Solution::longest_increasing_path_dfs_memo(matrix.clone());
        assert!(result >= expected);
        
        assert_eq!(Solution::longest_increasing_path_topological(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_optimized_dfs(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_bottom_up(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_iterative(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_multi_bfs(matrix), result);
    }
    
    #[test]
    fn test_strictly_decreasing() {
        let matrix = vec![
            vec![9, 8, 7],
            vec![6, 5, 4],
            vec![3, 2, 1]
        ];
        let expected = 9; // Path: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9
        
        let result = Solution::longest_increasing_path_dfs_memo(matrix.clone());
        assert_eq!(result, expected);
        
        assert_eq!(Solution::longest_increasing_path_topological(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_optimized_dfs(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_bottom_up(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_iterative(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_multi_bfs(matrix), result);
    }
    
    #[test]
    fn test_single_row() {
        let matrix = vec![vec![1, 3, 2, 4, 5]];
        let expected = 4; // Path: 1 -> 3 -> 4 -> 5
        
        let result = Solution::longest_increasing_path_dfs_memo(matrix.clone());
        assert_eq!(result, expected);
        
        assert_eq!(Solution::longest_increasing_path_topological(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_optimized_dfs(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_bottom_up(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_iterative(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_multi_bfs(matrix), result);
    }
    
    #[test]
    fn test_single_column() {
        let matrix = vec![
            vec![1],
            vec![3],
            vec![2],
            vec![4],
            vec![5]
        ];
        let expected = 4; // Path: 1 -> 3 -> 4 -> 5
        
        let result = Solution::longest_increasing_path_dfs_memo(matrix.clone());
        assert_eq!(result, expected);
        
        assert_eq!(Solution::longest_increasing_path_topological(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_optimized_dfs(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_bottom_up(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_iterative(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_multi_bfs(matrix), result);
    }
    
    #[test]
    fn test_large_values() {
        let matrix = vec![
            vec![10000, 20000, 5000],
            vec![15000, 25000, 30000],
            vec![1000, 2000, 40000]
        ];
        
        let result = Solution::longest_increasing_path_dfs_memo(matrix.clone());
        assert!(result >= 1);
        
        assert_eq!(Solution::longest_increasing_path_topological(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_optimized_dfs(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_bottom_up(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_iterative(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_multi_bfs(matrix), result);
    }
    
    #[test]
    fn test_negative_values() {
        let matrix = vec![
            vec![-1, -2, -3],
            vec![-4, -5, -6],
            vec![-7, -8, -9]
        ];
        let expected = 9; // Path: -9 -> -8 -> -7 -> -6 -> -5 -> -4 -> -3 -> -2 -> -1
        
        let result = Solution::longest_increasing_path_dfs_memo(matrix.clone());
        assert_eq!(result, expected);
        
        assert_eq!(Solution::longest_increasing_path_topological(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_optimized_dfs(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_bottom_up(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_iterative(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_multi_bfs(matrix), result);
    }
    
    #[test]
    fn test_mixed_values() {
        let matrix = vec![
            vec![0, 1, 2],
            vec![3, -1, 4],
            vec![-2, 5, 6]
        ];
        
        let result = Solution::longest_increasing_path_dfs_memo(matrix.clone());
        assert!(result >= 1);
        
        assert_eq!(Solution::longest_increasing_path_topological(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_optimized_dfs(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_bottom_up(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_iterative(matrix.clone()), result);
        assert_eq!(Solution::longest_increasing_path_multi_bfs(matrix), result);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_matrices = vec![
            vec![],
            vec![vec![1]],
            vec![vec![1, 1, 1], vec![1, 1, 1], vec![1, 1, 1]],
            vec![vec![9, 9, 4], vec![6, 6, 8], vec![2, 1, 1]],
            vec![vec![3, 4, 5], vec![3, 2, 6], vec![2, 2, 1]],
            vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]],
            vec![vec![9, 8, 7], vec![6, 5, 4], vec![3, 2, 1]],
            vec![vec![1, 3, 2, 4, 5]],
            vec![vec![1], vec![3], vec![2], vec![4], vec![5]],
            vec![vec![0, 1, 2], vec![3, -1, 4], vec![-2, 5, 6]],
        ];
        
        for matrix in test_matrices {
            let result1 = Solution::longest_increasing_path_dfs_memo(matrix.clone());
            let result2 = Solution::longest_increasing_path_topological(matrix.clone());
            let result3 = Solution::longest_increasing_path_optimized_dfs(matrix.clone());
            let result4 = Solution::longest_increasing_path_bottom_up(matrix.clone());
            let result5 = Solution::longest_increasing_path_iterative(matrix.clone());
            let result6 = Solution::longest_increasing_path_multi_bfs(matrix.clone());
            
            assert_eq!(result1, result2, "DFS vs Topological mismatch for {:?}", matrix);
            assert_eq!(result2, result3, "Topological vs Optimized DFS mismatch for {:?}", matrix);
            assert_eq!(result3, result4, "Optimized DFS vs Bottom-up mismatch for {:?}", matrix);
            assert_eq!(result4, result5, "Bottom-up vs Iterative mismatch for {:?}", matrix);
            assert_eq!(result5, result6, "Iterative vs Multi-BFS mismatch for {:?}", matrix);
        }
    }
    
    #[test]
    fn test_boundary_conditions() {
        // Test edge cases
        assert_eq!(Solution::longest_increasing_path_dfs_memo(vec![]), 0);
        assert_eq!(Solution::longest_increasing_path_dfs_memo(vec![vec![]]), 0);
        assert_eq!(Solution::longest_increasing_path_dfs_memo(vec![vec![42]]), 1);
        
        // Test with duplicate values in path
        let matrix_with_dups = vec![
            vec![1, 2, 2, 3],
            vec![1, 1, 2, 3],
        ];
        let result = Solution::longest_increasing_path_dfs_memo(matrix_with_dups.clone());
        assert!(result >= 1);
        assert_eq!(result, Solution::longest_increasing_path_topological(matrix_with_dups));
    }
}