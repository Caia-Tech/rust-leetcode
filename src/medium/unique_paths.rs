//! # 62. Unique Paths
//!
//! There is a robot on an m x n grid. The robot is initially located at the top-left corner 
//! (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). 
//! The robot can only move either down or right at any point in time.
//!
//! Given the two integers m and n, return the number of possible unique paths that the robot can take 
//! to reach the bottom-right corner.
//!
//! The test cases are generated so that the answer will be less than or equal to 2 * 10^9.
//!
//! **Example 1:**
//! ```text
//! Input: m = 3, n = 7
//! Output: 28
//! ```
//!
//! **Example 2:**
//! ```text
//! Input: m = 3, n = 2
//! Output: 3
//! Explanation: From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
//! 1. Right -> Down -> Down
//! 2. Down -> Right -> Down
//! 3. Down -> Down -> Right
//! ```
//!
//! **Constraints:**
//! - 1 <= m, n <= 100

/// Solution for Unique Paths - 6 different approaches
pub struct Solution;

impl Solution {
    /// Approach 1: Mathematical Combination (Optimal)
    /// 
    /// The robot needs to make (m-1) down moves and (n-1) right moves, for a total of 
    /// (m+n-2) moves. This is a combination problem: C(m+n-2, m-1) or C(m+n-2, n-1).
    ///
    /// Time Complexity: O(min(m, n))
    /// Space Complexity: O(1)
    pub fn unique_paths_math(&self, m: i32, n: i32) -> i32 {
        if m == 1 || n == 1 {
            return 1;
        }
        
        // For large inputs, use DP approach to avoid overflow
        if m > 20 || n > 20 {
            return self.unique_paths_dp_1d(m, n);
        }
        
        let m = m as u64;
        let n = n as u64;
        
        // Calculate C(m+n-2, min(m-1, n-1)) to minimize iterations
        let total_moves = m + n - 2;
        let down_moves = m - 1;
        let right_moves = n - 1;
        let k = down_moves.min(right_moves);
        
        let mut result = 1u64;
        
        // Calculate C(total_moves, k) = total_moves! / (k! * (total_moves-k)!)
        // Optimized to avoid overflow: multiply and divide incrementally
        for i in 0..k {
            result = result * (total_moves - i) / (i + 1);
        }
        
        result as i32
    }
    
    /// Approach 2: Dynamic Programming 2D Array
    /// 
    /// Build a 2D DP table where dp[i][j] represents the number of ways
    /// to reach position (i, j) from (0, 0).
    ///
    /// Time Complexity: O(m * n)
    /// Space Complexity: O(m * n)
    pub fn unique_paths_dp_2d(&self, m: i32, n: i32) -> i32 {
        let m = m as usize;
        let n = n as usize;
        
        let mut dp = vec![vec![0; n]; m];
        
        // Initialize first row and first column
        for i in 0..m {
            dp[i][0] = 1;
        }
        for j in 0..n {
            dp[0][j] = 1;
        }
        
        // Fill the DP table
        for i in 1..m {
            for j in 1..n {
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        
        dp[m-1][n-1]
    }
    
    /// Approach 3: Space-Optimized DP (1D Array)
    /// 
    /// Since we only need the previous row to calculate the current row,
    /// we can optimize space by using a 1D array.
    ///
    /// Time Complexity: O(m * n)
    /// Space Complexity: O(n)
    pub fn unique_paths_dp_1d(&self, m: i32, n: i32) -> i32 {
        let m = m as usize;
        let n = n as usize;
        
        let mut dp = vec![1; n];
        
        for _ in 1..m {
            for j in 1..n {
                dp[j] = dp[j] + dp[j-1];
            }
        }
        
        dp[n-1]
    }
    
    /// Approach 4: Recursive with Memoization
    /// 
    /// Use recursion to explore all paths, with memoization to avoid
    /// recomputing the same subproblems.
    ///
    /// Time Complexity: O(m * n)
    /// Space Complexity: O(m * n)
    pub fn unique_paths_memo(&self, m: i32, n: i32) -> i32 {
        use std::collections::HashMap;
        let mut memo = HashMap::new();
        self.count_paths(0, 0, m as usize, n as usize, &mut memo)
    }
    
    fn count_paths(&self, row: usize, col: usize, m: usize, n: usize, 
                  memo: &mut std::collections::HashMap<(usize, usize), i32>) -> i32 {
        if row == m - 1 && col == n - 1 {
            return 1;
        }
        if row >= m || col >= n {
            return 0;
        }
        
        let key = (row, col);
        if let Some(&cached) = memo.get(&key) {
            return cached;
        }
        
        let paths = self.count_paths(row + 1, col, m, n, memo) + 
                    self.count_paths(row, col + 1, m, n, memo);
        
        memo.insert(key, paths);
        paths
    }
    
    /// Approach 5: BFS (Educational)
    /// 
    /// Use breadth-first search to count all possible paths.
    /// This approach is less efficient but demonstrates another way to think about the problem.
    ///
    /// Time Complexity: O(2^(m+n)) - exponential
    /// Space Complexity: O(2^(m+n))
    pub fn unique_paths_bfs(&self, m: i32, n: i32) -> i32 {
        use std::collections::VecDeque;
        
        let m = m as usize;
        let n = n as usize;
        
        let mut queue = VecDeque::new();
        queue.push_back((0, 0));
        
        let mut count = 0;
        
        while let Some((row, col)) = queue.pop_front() {
            if row == m - 1 && col == n - 1 {
                count += 1;
                continue;
            }
            
            // Move down
            if row + 1 < m {
                queue.push_back((row + 1, col));
            }
            
            // Move right
            if col + 1 < n {
                queue.push_back((row, col + 1));
            }
        }
        
        count
    }
    
    /// Approach 6: Pascal's Triangle Pattern
    /// 
    /// Recognize that this problem follows Pascal's triangle pattern.
    /// Each cell value is the sum of the cell above and the cell to the left.
    ///
    /// Time Complexity: O(m * n)
    /// Space Complexity: O(min(m, n))
    pub fn unique_paths_pascal(&self, m: i32, n: i32) -> i32 {
        let m = m as usize;
        let n = n as usize;
        
        // Use the smaller dimension for space optimization
        if m < n {
            return self.unique_paths_pascal_helper(m, n);
        } else {
            return self.unique_paths_pascal_helper(n, m);
        }
    }
    
    fn unique_paths_pascal_helper(&self, rows: usize, cols: usize) -> i32 {
        let mut prev_row = vec![1; rows];
        
        for _ in 1..cols {
            let mut curr_row = vec![1; rows];
            for i in 1..rows {
                curr_row[i] = prev_row[i] + curr_row[i-1];
            }
            prev_row = curr_row;
        }
        
        prev_row[rows-1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_examples() {
        let solution = Solution;
        
        // Example 1: m=3, n=7 -> 28
        assert_eq!(solution.unique_paths_math(3, 7), 28);
        assert_eq!(solution.unique_paths_dp_2d(3, 7), 28);
        assert_eq!(solution.unique_paths_dp_1d(3, 7), 28);
        assert_eq!(solution.unique_paths_memo(3, 7), 28);
        assert_eq!(solution.unique_paths_pascal(3, 7), 28);
        
        // Example 2: m=3, n=2 -> 3
        assert_eq!(solution.unique_paths_math(3, 2), 3);
        assert_eq!(solution.unique_paths_dp_2d(3, 2), 3);
        assert_eq!(solution.unique_paths_dp_1d(3, 2), 3);
        assert_eq!(solution.unique_paths_memo(3, 2), 3);
        assert_eq!(solution.unique_paths_pascal(3, 2), 3);
    }

    #[test]
    fn test_edge_cases() {
        let solution = Solution;
        
        // 1x1 grid
        assert_eq!(solution.unique_paths_math(1, 1), 1);
        assert_eq!(solution.unique_paths_dp_2d(1, 1), 1);
        assert_eq!(solution.unique_paths_dp_1d(1, 1), 1);
        assert_eq!(solution.unique_paths_memo(1, 1), 1);
        assert_eq!(solution.unique_paths_pascal(1, 1), 1);
        
        // Single row
        assert_eq!(solution.unique_paths_math(1, 5), 1);
        assert_eq!(solution.unique_paths_dp_2d(1, 5), 1);
        assert_eq!(solution.unique_paths_dp_1d(1, 5), 1);
        assert_eq!(solution.unique_paths_memo(1, 5), 1);
        assert_eq!(solution.unique_paths_pascal(1, 5), 1);
        
        // Single column
        assert_eq!(solution.unique_paths_math(5, 1), 1);
        assert_eq!(solution.unique_paths_dp_2d(5, 1), 1);
        assert_eq!(solution.unique_paths_dp_1d(5, 1), 1);
        assert_eq!(solution.unique_paths_memo(5, 1), 1);
        assert_eq!(solution.unique_paths_pascal(5, 1), 1);
    }

    #[test]
    fn test_small_grids() {
        let solution = Solution;
        
        // 2x2 grid
        assert_eq!(solution.unique_paths_math(2, 2), 2);
        assert_eq!(solution.unique_paths_dp_2d(2, 2), 2);
        assert_eq!(solution.unique_paths_dp_1d(2, 2), 2);
        assert_eq!(solution.unique_paths_memo(2, 2), 2);
        assert_eq!(solution.unique_paths_pascal(2, 2), 2);
        
        // 2x3 grid
        assert_eq!(solution.unique_paths_math(2, 3), 3);
        assert_eq!(solution.unique_paths_dp_2d(2, 3), 3);
        assert_eq!(solution.unique_paths_dp_1d(2, 3), 3);
        assert_eq!(solution.unique_paths_memo(2, 3), 3);
        assert_eq!(solution.unique_paths_pascal(2, 3), 3);
        
        // 3x3 grid
        assert_eq!(solution.unique_paths_math(3, 3), 6);
        assert_eq!(solution.unique_paths_dp_2d(3, 3), 6);
        assert_eq!(solution.unique_paths_dp_1d(3, 3), 6);
        assert_eq!(solution.unique_paths_memo(3, 3), 6);
        assert_eq!(solution.unique_paths_pascal(3, 3), 6);
    }

    #[test]
    fn test_rectangular_grids() {
        let solution = Solution;
        
        // 4x6 grid
        assert_eq!(solution.unique_paths_math(4, 6), 56);
        assert_eq!(solution.unique_paths_dp_2d(4, 6), 56);
        assert_eq!(solution.unique_paths_dp_1d(4, 6), 56);
        assert_eq!(solution.unique_paths_memo(4, 6), 56);
        assert_eq!(solution.unique_paths_pascal(4, 6), 56);
        
        // 6x4 grid (should be same as 4x6)
        assert_eq!(solution.unique_paths_math(6, 4), 56);
        assert_eq!(solution.unique_paths_dp_2d(6, 4), 56);
        assert_eq!(solution.unique_paths_dp_1d(6, 4), 56);
        assert_eq!(solution.unique_paths_memo(6, 4), 56);
        assert_eq!(solution.unique_paths_pascal(6, 4), 56);
    }

    #[test]
    fn test_symmetry() {
        let solution = Solution;
        
        // Test that unique_paths(m, n) == unique_paths(n, m)
        let test_cases = [(2, 5), (3, 7), (4, 6), (5, 8)];
        
        for (m, n) in test_cases {
            let result1 = solution.unique_paths_math(m, n);
            let result2 = solution.unique_paths_math(n, m);
            assert_eq!(result1, result2, "unique_paths({}, {}) should equal unique_paths({}, {})", m, n, n, m);
            
            // Test with other approaches too
            assert_eq!(solution.unique_paths_dp_2d(m, n), solution.unique_paths_dp_2d(n, m));
            assert_eq!(solution.unique_paths_dp_1d(m, n), solution.unique_paths_dp_1d(n, m));
            assert_eq!(solution.unique_paths_memo(m, n), solution.unique_paths_memo(n, m));
            assert_eq!(solution.unique_paths_pascal(m, n), solution.unique_paths_pascal(n, m));
        }
    }

    #[test]
    fn test_larger_grids() {
        let solution = Solution;
        
        // 10x10 grid
        assert_eq!(solution.unique_paths_math(10, 10), 48620);
        assert_eq!(solution.unique_paths_dp_2d(10, 10), 48620);
        assert_eq!(solution.unique_paths_dp_1d(10, 10), 48620);
        assert_eq!(solution.unique_paths_memo(10, 10), 48620);
        assert_eq!(solution.unique_paths_pascal(10, 10), 48620);
        
        // 15x15 grid
        let result_15x15 = solution.unique_paths_math(15, 15);
        assert_eq!(solution.unique_paths_dp_2d(15, 15), result_15x15);
        assert_eq!(solution.unique_paths_dp_1d(15, 15), result_15x15);
        assert_eq!(solution.unique_paths_memo(15, 15), result_15x15);
        assert_eq!(solution.unique_paths_pascal(15, 15), result_15x15);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = Solution;
        
        let test_cases = vec![
            (1, 1), (1, 10), (10, 1), (2, 2), (3, 3), (4, 5), (5, 4), (7, 8), (8, 7)
        ];
        
        for (m, n) in test_cases {
            let math_result = solution.unique_paths_math(m, n);
            let dp_2d_result = solution.unique_paths_dp_2d(m, n);
            let dp_1d_result = solution.unique_paths_dp_1d(m, n);
            let memo_result = solution.unique_paths_memo(m, n);
            let bfs_result = solution.unique_paths_bfs(m, n);
            let pascal_result = solution.unique_paths_pascal(m, n);
            
            assert_eq!(math_result, dp_2d_result, "Math vs DP_2D mismatch for ({}, {})", m, n);
            assert_eq!(math_result, dp_1d_result, "Math vs DP_1D mismatch for ({}, {})", m, n);
            assert_eq!(math_result, memo_result, "Math vs Memo mismatch for ({}, {})", m, n);
            assert_eq!(math_result, bfs_result, "Math vs BFS mismatch for ({}, {})", m, n);
            assert_eq!(math_result, pascal_result, "Math vs Pascal mismatch for ({}, {})", m, n);
        }
    }

    #[test]
    fn test_boundary_constraints() {
        let solution = Solution;
        
        // Test constraint boundaries
        assert_eq!(solution.unique_paths_math(1, 100), 1);
        assert_eq!(solution.unique_paths_dp_1d(1, 100), 1);  // Use space-optimized for large inputs
        
        assert_eq!(solution.unique_paths_math(100, 1), 1);
        assert_eq!(solution.unique_paths_dp_1d(100, 1), 1);
        
        // Test some larger cases to ensure no overflow
        let result = solution.unique_paths_math(20, 20);
        assert!(result > 0, "Result should be positive");
        assert!(result < 2_000_000_000, "Result should be within i32 range");
    }

    #[test]
    fn test_manual_calculation() {
        let solution = Solution;
        
        // Manual verification for small cases
        // 2x3: RRD, RDR, DRR = 3 paths
        assert_eq!(solution.unique_paths_math(2, 3), 3);
        
        // 3x2: DDR, DRD, RDD = 3 paths  
        assert_eq!(solution.unique_paths_math(3, 2), 3);
        
        // 2x4: RRRD, RRDR, RDRR, DRRR = 4 paths
        assert_eq!(solution.unique_paths_math(2, 4), 4);
        
        // 4x2: DDDR, DDRD, DRDD, RDDD = 4 paths
        assert_eq!(solution.unique_paths_math(4, 2), 4);
    }

    #[test]
    fn test_performance_comparison() {
        let solution = Solution;
        
        // Test that math approach works efficiently for moderately large inputs
        let result = solution.unique_paths_math(15, 15);
        assert!(result > 0);

        // For performance testing, use sizes that avoid overflow
        let large_results = vec![
            solution.unique_paths_math(20, 10),
            solution.unique_paths_math(10, 20),
            solution.unique_paths_math(16, 16),
        ];
        
        for result in large_results {
            assert!(result > 0, "Large grid should have positive number of paths");
        }
    }

    #[test] 
    fn test_mathematical_properties() {
        let solution = Solution;
        
        // Test that paths increase as grid size increases
        assert!(solution.unique_paths_math(2, 2) < solution.unique_paths_math(3, 3));
        assert!(solution.unique_paths_math(3, 3) < solution.unique_paths_math(4, 4));
        assert!(solution.unique_paths_math(4, 4) < solution.unique_paths_math(5, 5));
        
        // Test that adding a row or column increases paths
        assert!(solution.unique_paths_math(3, 4) < solution.unique_paths_math(4, 4));
        assert!(solution.unique_paths_math(3, 4) < solution.unique_paths_math(3, 5));
    }
}