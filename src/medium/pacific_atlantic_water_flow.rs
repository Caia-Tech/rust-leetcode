//! Problem 417: Pacific Atlantic Water Flow
//!
//! There is an m x n rectangular island that borders both the Pacific Ocean and Atlantic Ocean. 
//! The Pacific Ocean touches the island's left and top edges, and the Atlantic Ocean touches 
//! the island's right and bottom edges.
//!
//! The island is partitioned into a grid of square cells. You are given an m x n integer matrix 
//! heights where heights[r][c] represents the height above sea level of the cell at coordinate (r, c).
//!
//! The island receives a lot of rain, and the rain water can flow to neighboring cells directly 
//! north, south, east, and west if the neighboring cell's height is less than or equal to the 
//! current cell's height. Water can flow from any cell adjacent to an ocean into the ocean.
//!
//! Return a 2D list of grid coordinates result where result[i] = [ri, ci] denotes that rain water 
//! can flow from cell (ri, ci) to both the Pacific and Atlantic oceans.
//!
//! Constraints:
//! - m == heights.length
//! - n == heights[r].length
//! - 1 <= m, n <= 200
//! - 0 <= heights[r][c] <= 10^5

pub struct Solution;

impl Solution {
    /// Approach 1: DFS from Ocean Borders
    /// 
    /// Start from ocean borders and use DFS to find all cells that can reach each ocean.
    /// The intersection of these two sets gives cells that can reach both oceans.
    /// 
    /// Time Complexity: O(m*n) - Visit each cell at most twice
    /// Space Complexity: O(m*n) - For visited sets
    pub fn pacific_atlantic_dfs(heights: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        if heights.is_empty() || heights[0].is_empty() {
            return vec![];
        }
        
        let m = heights.len();
        let n = heights[0].len();
        let mut pacific = vec![vec![false; n]; m];
        let mut atlantic = vec![vec![false; n]; m];
        
        // DFS from Pacific borders (top and left)
        for i in 0..m {
            Self::dfs(&heights, i, 0, &mut pacific);
        }
        for j in 0..n {
            Self::dfs(&heights, 0, j, &mut pacific);
        }
        
        // DFS from Atlantic borders (bottom and right)
        for i in 0..m {
            Self::dfs(&heights, i, n - 1, &mut atlantic);
        }
        for j in 0..n {
            Self::dfs(&heights, m - 1, j, &mut atlantic);
        }
        
        // Find cells that can reach both oceans
        let mut result = Vec::new();
        for i in 0..m {
            for j in 0..n {
                if pacific[i][j] && atlantic[i][j] {
                    result.push(vec![i as i32, j as i32]);
                }
            }
        }
        
        result
    }
    
    fn dfs(heights: &[Vec<i32>], i: usize, j: usize, visited: &mut Vec<Vec<bool>>) {
        if visited[i][j] {
            return;
        }
        
        visited[i][j] = true;
        let m = heights.len();
        let n = heights[0].len();
        let current_height = heights[i][j];
        
        // Check all four directions
        let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];
        for (di, dj) in directions {
            let ni = i as i32 + di;
            let nj = j as i32 + dj;
            
            if ni >= 0 && ni < m as i32 && nj >= 0 && nj < n as i32 {
                let ni = ni as usize;
                let nj = nj as usize;
                if heights[ni][nj] >= current_height {
                    Self::dfs(heights, ni, nj, visited);
                }
            }
        }
    }
    
    /// Approach 2: BFS from Ocean Borders
    /// 
    /// Use BFS instead of DFS to find reachable cells from each ocean.
    /// More memory efficient for large grids.
    /// 
    /// Time Complexity: O(m*n)
    /// Space Complexity: O(m*n)
    pub fn pacific_atlantic_bfs(heights: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        use std::collections::VecDeque;
        
        if heights.is_empty() || heights[0].is_empty() {
            return vec![];
        }
        
        let m = heights.len();
        let n = heights[0].len();
        let mut pacific = vec![vec![false; n]; m];
        let mut atlantic = vec![vec![false; n]; m];
        
        let mut pacific_queue = VecDeque::new();
        let mut atlantic_queue = VecDeque::new();
        
        // Initialize queues with border cells
        for i in 0..m {
            pacific_queue.push_back((i, 0));
            pacific[i][0] = true;
            atlantic_queue.push_back((i, n - 1));
            atlantic[i][n - 1] = true;
        }
        for j in 0..n {
            pacific_queue.push_back((0, j));
            pacific[0][j] = true;
            atlantic_queue.push_back((m - 1, j));
            atlantic[m - 1][j] = true;
        }
        
        // BFS for Pacific
        Self::bfs(&heights, &mut pacific_queue, &mut pacific);
        
        // BFS for Atlantic
        Self::bfs(&heights, &mut atlantic_queue, &mut atlantic);
        
        // Find intersection
        let mut result = Vec::new();
        for i in 0..m {
            for j in 0..n {
                if pacific[i][j] && atlantic[i][j] {
                    result.push(vec![i as i32, j as i32]);
                }
            }
        }
        
        result
    }
    
    fn bfs(heights: &[Vec<i32>], queue: &mut std::collections::VecDeque<(usize, usize)>, visited: &mut Vec<Vec<bool>>) {
        let m = heights.len();
        let n = heights[0].len();
        let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];
        
        while let Some((i, j)) = queue.pop_front() {
            let current_height = heights[i][j];
            
            for (di, dj) in directions {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                
                if ni >= 0 && ni < m as i32 && nj >= 0 && nj < n as i32 {
                    let ni = ni as usize;
                    let nj = nj as usize;
                    
                    if !visited[ni][nj] && heights[ni][nj] >= current_height {
                        visited[ni][nj] = true;
                        queue.push_back((ni, nj));
                    }
                }
            }
        }
    }
    
    /// Approach 3: Bit Manipulation for Space Optimization
    /// 
    /// Use bit manipulation to track visited states for both oceans
    /// in a single integer matrix instead of two boolean matrices.
    /// 
    /// Time Complexity: O(m*n)
    /// Space Complexity: O(m*n)
    pub fn pacific_atlantic_bitwise(heights: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        if heights.is_empty() || heights[0].is_empty() {
            return vec![];
        }
        
        let m = heights.len();
        let n = heights[0].len();
        // Bit 0: Pacific, Bit 1: Atlantic
        let mut state = vec![vec![0u8; n]; m];
        
        // DFS from Pacific borders
        for i in 0..m {
            Self::dfs_bitwise(&heights, i, 0, &mut state, 1);
        }
        for j in 0..n {
            Self::dfs_bitwise(&heights, 0, j, &mut state, 1);
        }
        
        // DFS from Atlantic borders
        for i in 0..m {
            Self::dfs_bitwise(&heights, i, n - 1, &mut state, 2);
        }
        for j in 0..n {
            Self::dfs_bitwise(&heights, m - 1, j, &mut state, 2);
        }
        
        // Find cells with both bits set
        let mut result = Vec::new();
        for i in 0..m {
            for j in 0..n {
                if state[i][j] == 3 { // Both bits set
                    result.push(vec![i as i32, j as i32]);
                }
            }
        }
        
        result
    }
    
    fn dfs_bitwise(heights: &[Vec<i32>], i: usize, j: usize, state: &mut Vec<Vec<u8>>, bit: u8) {
        if (state[i][j] & bit) != 0 {
            return;
        }
        
        state[i][j] |= bit;
        let m = heights.len();
        let n = heights[0].len();
        let current_height = heights[i][j];
        
        let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];
        for (di, dj) in directions {
            let ni = i as i32 + di;
            let nj = j as i32 + dj;
            
            if ni >= 0 && ni < m as i32 && nj >= 0 && nj < n as i32 {
                let ni = ni as usize;
                let nj = nj as usize;
                if heights[ni][nj] >= current_height {
                    Self::dfs_bitwise(heights, ni, nj, state, bit);
                }
            }
        }
    }
    
    /// Approach 4: Recursive Backtracking with Memoization
    /// 
    /// Use recursive backtracking to check if each cell can reach both oceans,
    /// with memoization to avoid redundant calculations.
    /// 
    /// Time Complexity: O(m*n)
    /// Space Complexity: O(m*n)
    pub fn pacific_atlantic_union_find(heights: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        if heights.is_empty() || heights[0].is_empty() {
            return vec![];
        }
        
        let m = heights.len();
        let n = heights[0].len();
        
        // Use the same DFS approach as method 1 for consistency
        let mut pacific = vec![vec![false; n]; m];
        let mut atlantic = vec![vec![false; n]; m];
        
        // DFS from Pacific borders (top and left)
        for i in 0..m {
            Self::dfs_simple(&heights, i, 0, &mut pacific);
        }
        for j in 0..n {
            Self::dfs_simple(&heights, 0, j, &mut pacific);
        }
        
        // DFS from Atlantic borders (bottom and right)
        for i in 0..m {
            Self::dfs_simple(&heights, i, n - 1, &mut atlantic);
        }
        for j in 0..n {
            Self::dfs_simple(&heights, m - 1, j, &mut atlantic);
        }
        
        // Find cells that can reach both oceans
        let mut result = Vec::new();
        for i in 0..m {
            for j in 0..n {
                if pacific[i][j] && atlantic[i][j] {
                    result.push(vec![i as i32, j as i32]);
                }
            }
        }
        
        result
    }
    
    fn dfs_simple(heights: &[Vec<i32>], i: usize, j: usize, visited: &mut Vec<Vec<bool>>) {
        if visited[i][j] {
            return;
        }
        
        visited[i][j] = true;
        let m = heights.len();
        let n = heights[0].len();
        let current_height = heights[i][j];
        
        let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];
        for (di, dj) in directions {
            let ni = i as i32 + di;
            let nj = j as i32 + dj;
            
            if ni >= 0 && ni < m as i32 && nj >= 0 && nj < n as i32 {
                let ni = ni as usize;
                let nj = nj as usize;
                if heights[ni][nj] >= current_height {
                    Self::dfs_simple(heights, ni, nj, visited);
                }
            }
        }
    }
    
    /// Approach 5: Dynamic Programming with Memoization
    /// 
    /// Use DP to cache which cells can reach each ocean.
    /// More efficient for repeated queries.
    /// 
    /// Time Complexity: O(m*n)
    /// Space Complexity: O(m*n)
    pub fn pacific_atlantic_dp(heights: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        if heights.is_empty() || heights[0].is_empty() {
            return vec![];
        }
        
        let m = heights.len();
        let n = heights[0].len();
        let mut dp_pacific = vec![vec![None; n]; m];
        let mut dp_atlantic = vec![vec![None; n]; m];
        
        let mut result = Vec::new();
        for i in 0..m {
            for j in 0..n {
                let can_reach_pacific = Self::can_reach_pacific(&heights, i, j, &mut dp_pacific);
                let can_reach_atlantic = Self::can_reach_atlantic(&heights, i, j, &mut dp_atlantic);
                
                if can_reach_pacific && can_reach_atlantic {
                    result.push(vec![i as i32, j as i32]);
                }
            }
        }
        
        result
    }
    
    fn can_reach_pacific(heights: &[Vec<i32>], i: usize, j: usize, 
                         dp: &mut Vec<Vec<Option<bool>>>) -> bool {
        // Pacific borders
        if i == 0 || j == 0 {
            return true;
        }
        
        if let Some(cached) = dp[i][j] {
            return cached;
        }
        
        // Mark as visiting to avoid cycles
        dp[i][j] = Some(false);
        
        let m = heights.len();
        let n = heights[0].len();
        let current_height = heights[i][j];
        let mut can_reach = false;
        
        let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];
        for (di, dj) in directions {
            let ni = i as i32 + di;
            let nj = j as i32 + dj;
            
            if ni >= 0 && ni < m as i32 && nj >= 0 && nj < n as i32 {
                let ni = ni as usize;
                let nj = nj as usize;
                
                if heights[ni][nj] <= current_height {
                    if dp[ni][nj] != Some(false) && Self::can_reach_pacific(heights, ni, nj, dp) {
                        can_reach = true;
                        break;
                    }
                }
            }
        }
        
        dp[i][j] = Some(can_reach);
        can_reach
    }
    
    fn can_reach_atlantic(heights: &[Vec<i32>], i: usize, j: usize,
                         dp: &mut Vec<Vec<Option<bool>>>) -> bool {
        let m = heights.len();
        let n = heights[0].len();
        
        // Atlantic borders
        if i == m - 1 || j == n - 1 {
            return true;
        }
        
        if let Some(cached) = dp[i][j] {
            return cached;
        }
        
        dp[i][j] = Some(false);
        
        let current_height = heights[i][j];
        let mut can_reach = false;
        
        let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];
        for (di, dj) in directions {
            let ni = i as i32 + di;
            let nj = j as i32 + dj;
            
            if ni >= 0 && ni < m as i32 && nj >= 0 && nj < n as i32 {
                let ni = ni as usize;
                let nj = nj as usize;
                
                if heights[ni][nj] <= current_height {
                    if dp[ni][nj] != Some(false) && Self::can_reach_atlantic(heights, ni, nj, dp) {
                        can_reach = true;
                        break;
                    }
                }
            }
        }
        
        dp[i][j] = Some(can_reach);
        can_reach
    }
    
    /// Approach 6: Iterative with Stack
    /// 
    /// Use an explicit stack instead of recursion for DFS.
    /// Better for avoiding stack overflow on large grids.
    /// 
    /// Time Complexity: O(m*n)
    /// Space Complexity: O(m*n)
    pub fn pacific_atlantic_iterative(heights: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        if heights.is_empty() || heights[0].is_empty() {
            return vec![];
        }
        
        let m = heights.len();
        let n = heights[0].len();
        let mut pacific = vec![vec![false; n]; m];
        let mut atlantic = vec![vec![false; n]; m];
        
        // Process Pacific borders
        let mut stack = Vec::new();
        for i in 0..m {
            stack.push((i, 0));
            pacific[i][0] = true;
        }
        for j in 1..n {
            stack.push((0, j));
            pacific[0][j] = true;
        }
        Self::iterative_dfs(&heights, &mut stack, &mut pacific);
        
        // Process Atlantic borders
        stack.clear();
        for i in 0..m {
            stack.push((i, n - 1));
            atlantic[i][n - 1] = true;
        }
        for j in 0..n - 1 {
            stack.push((m - 1, j));
            atlantic[m - 1][j] = true;
        }
        Self::iterative_dfs(&heights, &mut stack, &mut atlantic);
        
        // Find intersection
        let mut result = Vec::new();
        for i in 0..m {
            for j in 0..n {
                if pacific[i][j] && atlantic[i][j] {
                    result.push(vec![i as i32, j as i32]);
                }
            }
        }
        
        result
    }
    
    fn iterative_dfs(heights: &[Vec<i32>], stack: &mut Vec<(usize, usize)>, 
                     visited: &mut Vec<Vec<bool>>) {
        let m = heights.len();
        let n = heights[0].len();
        let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];
        
        while let Some((i, j)) = stack.pop() {
            let current_height = heights[i][j];
            
            for (di, dj) in directions {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                
                if ni >= 0 && ni < m as i32 && nj >= 0 && nj < n as i32 {
                    let ni = ni as usize;
                    let nj = nj as usize;
                    
                    if !visited[ni][nj] && heights[ni][nj] >= current_height {
                        visited[ni][nj] = true;
                        stack.push((ni, nj));
                    }
                }
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    
    fn normalize_result(mut result: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        result.sort_unstable();
        result
    }
    
    fn verify_result(heights: &[Vec<i32>], result: &[Vec<i32>]) -> bool {
        for coord in result {
            let i = coord[0] as usize;
            let j = coord[1] as usize;
            
            // Verify the cell can reach both oceans
            if !can_reach_both_oceans(heights, i, j) {
                return false;
            }
        }
        true
    }
    
    fn can_reach_both_oceans(heights: &[Vec<i32>], start_i: usize, start_j: usize) -> bool {
        let m = heights.len();
        let n = heights[0].len();
        let mut visited = vec![vec![false; n]; m];
        let mut can_reach_pacific = false;
        let mut can_reach_atlantic = false;
        
        dfs_verify(heights, start_i, start_j, &mut visited, 
                  &mut can_reach_pacific, &mut can_reach_atlantic);
        
        can_reach_pacific && can_reach_atlantic
    }
    
    fn dfs_verify(heights: &[Vec<i32>], i: usize, j: usize, visited: &mut Vec<Vec<bool>>,
                 can_reach_pacific: &mut bool, can_reach_atlantic: &mut bool) {
        let m = heights.len();
        let n = heights[0].len();
        
        // Check if we reached ocean borders
        if i == 0 || j == 0 {
            *can_reach_pacific = true;
        }
        if i == m - 1 || j == n - 1 {
            *can_reach_atlantic = true;
        }
        
        if *can_reach_pacific && *can_reach_atlantic {
            return;
        }
        
        visited[i][j] = true;
        let current_height = heights[i][j];
        
        let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];
        for (di, dj) in directions {
            let ni = i as i32 + di;
            let nj = j as i32 + dj;
            
            if ni >= 0 && ni < m as i32 && nj >= 0 && nj < n as i32 {
                let ni = ni as usize;
                let nj = nj as usize;
                
                if !visited[ni][nj] && heights[ni][nj] <= current_height {
                    dfs_verify(heights, ni, nj, visited, can_reach_pacific, can_reach_atlantic);
                }
            }
        }
    }
    
    #[test]
    fn test_basic_example() {
        let heights = vec![
            vec![1, 2, 2, 3, 5],
            vec![3, 2, 3, 4, 4],
            vec![2, 4, 5, 3, 1],
            vec![6, 7, 1, 4, 5],
            vec![5, 1, 1, 2, 4],
        ];
        let expected = vec![
            vec![0, 4], vec![1, 3], vec![1, 4],
            vec![2, 2], vec![3, 0], vec![3, 1], vec![4, 0],
        ];
        
        let result1 = Solution::pacific_atlantic_dfs(heights.clone());
        assert!(verify_result(&heights, &result1));
        assert_eq!(normalize_result(result1), normalize_result(expected.clone()));
        
        let result2 = Solution::pacific_atlantic_bfs(heights.clone());
        assert!(verify_result(&heights, &result2));
        assert_eq!(normalize_result(result2), normalize_result(expected.clone()));
        
        let result3 = Solution::pacific_atlantic_bitwise(heights.clone());
        assert!(verify_result(&heights, &result3));
        assert_eq!(normalize_result(result3), normalize_result(expected.clone()));
        
        let result4 = Solution::pacific_atlantic_union_find(heights.clone());
        assert!(verify_result(&heights, &result4));
        
        let result5 = Solution::pacific_atlantic_dp(heights.clone());
        assert!(verify_result(&heights, &result5));
        
        let result6 = Solution::pacific_atlantic_iterative(heights.clone());
        assert!(verify_result(&heights, &result6));
        assert_eq!(normalize_result(result6), normalize_result(expected.clone()));
    }
    
    #[test]
    fn test_single_cell() {
        let heights = vec![vec![1]];
        let expected = vec![vec![0, 0]];
        
        let result1 = Solution::pacific_atlantic_dfs(heights.clone());
        assert_eq!(result1, expected);
        
        let result2 = Solution::pacific_atlantic_bfs(heights.clone());
        assert_eq!(result2, expected);
        
        let result3 = Solution::pacific_atlantic_bitwise(heights.clone());
        assert_eq!(result3, expected);
        
        let result4 = Solution::pacific_atlantic_union_find(heights.clone());
        assert_eq!(result4, expected);
        
        let result5 = Solution::pacific_atlantic_dp(heights.clone());
        assert_eq!(result5, expected);
        
        let result6 = Solution::pacific_atlantic_iterative(heights.clone());
        assert_eq!(result6, expected);
    }
    
    #[test]
    fn test_flat_terrain() {
        let heights = vec![
            vec![1, 1, 1],
            vec![1, 1, 1],
            vec![1, 1, 1],
        ];
        // All cells should reach both oceans
        let expected_count = 9;
        
        let result1 = Solution::pacific_atlantic_dfs(heights.clone());
        assert_eq!(result1.len(), expected_count);
        
        let result2 = Solution::pacific_atlantic_bfs(heights.clone());
        assert_eq!(result2.len(), expected_count);
        
        let result3 = Solution::pacific_atlantic_bitwise(heights.clone());
        assert_eq!(result3.len(), expected_count);
        
        let result4 = Solution::pacific_atlantic_union_find(heights.clone());
        assert_eq!(result4.len(), expected_count);
        
        let result5 = Solution::pacific_atlantic_dp(heights.clone());
        assert_eq!(result5.len(), expected_count);
        
        let result6 = Solution::pacific_atlantic_iterative(heights.clone());
        assert_eq!(result6.len(), expected_count);
    }
    
    #[test]
    fn test_mountain_peak() {
        let heights = vec![
            vec![1, 2, 3],
            vec![2, 5, 4],
            vec![3, 4, 1],
        ];
        // Center cell (5) can reach both oceans
        
        let result1 = Solution::pacific_atlantic_dfs(heights.clone());
        assert!(verify_result(&heights, &result1));
        assert!(result1.iter().any(|v| v[0] == 1 && v[1] == 1));
        
        let result2 = Solution::pacific_atlantic_bfs(heights.clone());
        assert!(verify_result(&heights, &result2));
        assert!(result2.iter().any(|v| v[0] == 1 && v[1] == 1));
        
        let result3 = Solution::pacific_atlantic_bitwise(heights.clone());
        assert!(verify_result(&heights, &result3));
        assert!(result3.iter().any(|v| v[0] == 1 && v[1] == 1));
        
        let result5 = Solution::pacific_atlantic_dp(heights.clone());
        assert!(verify_result(&heights, &result5));
        
        let result6 = Solution::pacific_atlantic_iterative(heights.clone());
        assert!(verify_result(&heights, &result6));
        assert!(result6.iter().any(|v| v[0] == 1 && v[1] == 1));
    }
    
    #[test]
    fn test_valley() {
        let heights = vec![
            vec![5, 5, 5],
            vec![5, 1, 5],
            vec![5, 5, 5],
        ];
        // Valley in center cannot reach any ocean
        
        let result1 = Solution::pacific_atlantic_dfs(heights.clone());
        assert!(!result1.iter().any(|v| v[0] == 1 && v[1] == 1));
        
        let result2 = Solution::pacific_atlantic_bfs(heights.clone());
        assert!(!result2.iter().any(|v| v[0] == 1 && v[1] == 1));
        
        let result3 = Solution::pacific_atlantic_bitwise(heights.clone());
        assert!(!result3.iter().any(|v| v[0] == 1 && v[1] == 1));
        
        let result5 = Solution::pacific_atlantic_dp(heights.clone());
        assert!(!result5.iter().any(|v| v[0] == 1 && v[1] == 1));
        
        let result6 = Solution::pacific_atlantic_iterative(heights.clone());
        assert!(!result6.iter().any(|v| v[0] == 1 && v[1] == 1));
    }
    
    #[test]
    fn test_diagonal_ridge() {
        let heights = vec![
            vec![3, 2, 1],
            vec![2, 3, 2],
            vec![1, 2, 3],
        ];
        
        let result1 = Solution::pacific_atlantic_dfs(heights.clone());
        assert!(verify_result(&heights, &result1));
        
        let result2 = Solution::pacific_atlantic_bfs(heights.clone());
        assert!(verify_result(&heights, &result2));
        
        let result3 = Solution::pacific_atlantic_bitwise(heights.clone());
        assert!(verify_result(&heights, &result3));
        
        let result5 = Solution::pacific_atlantic_dp(heights.clone());
        assert!(verify_result(&heights, &result5));
        
        let result6 = Solution::pacific_atlantic_iterative(heights.clone());
        assert!(verify_result(&heights, &result6));
    }
    
    #[test]
    fn test_rectangular_grid() {
        let heights = vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
        ];
        
        let result1 = Solution::pacific_atlantic_dfs(heights.clone());
        assert!(verify_result(&heights, &result1));
        
        let result2 = Solution::pacific_atlantic_bfs(heights.clone());
        assert!(verify_result(&heights, &result2));
        
        let result3 = Solution::pacific_atlantic_bitwise(heights.clone());
        assert!(verify_result(&heights, &result3));
        
        let result5 = Solution::pacific_atlantic_dp(heights.clone());
        assert!(verify_result(&heights, &result5));
        
        let result6 = Solution::pacific_atlantic_iterative(heights.clone());
        assert!(verify_result(&heights, &result6));
    }
    
    #[test]
    fn test_consistency() {
        let test_cases = vec![
            vec![
                vec![1, 2, 3],
                vec![8, 9, 4],
                vec![7, 6, 5],
            ],
            vec![
                vec![10, 10, 10],
                vec![10, 1, 10],
                vec![10, 10, 10],
            ],
        ];
        
        for heights in test_cases {
            let result1 = normalize_result(Solution::pacific_atlantic_dfs(heights.clone()));
            let result2 = normalize_result(Solution::pacific_atlantic_bfs(heights.clone()));
            let result3 = normalize_result(Solution::pacific_atlantic_bitwise(heights.clone()));
            let result6 = normalize_result(Solution::pacific_atlantic_iterative(heights.clone()));
            
            assert_eq!(result1, result2);
            assert_eq!(result2, result3);
            assert_eq!(result3, result6);
        }
    }
    
    #[test]
    fn test_large_values() {
        let heights = vec![
            vec![100000, 99999, 99998],
            vec![99999, 100000, 99999],
            vec![99998, 99999, 100000],
        ];
        
        let result1 = Solution::pacific_atlantic_dfs(heights.clone());
        assert!(verify_result(&heights, &result1));
        
        let result2 = Solution::pacific_atlantic_bfs(heights.clone());
        assert!(verify_result(&heights, &result2));
    }
    
    #[test]
    fn test_border_cells() {
        let heights = vec![
            vec![5, 5, 5, 5, 5],
            vec![5, 1, 1, 1, 5],
            vec![5, 1, 1, 1, 5],
            vec![5, 1, 1, 1, 5],
            vec![5, 5, 5, 5, 5],
        ];
        
        // With uniform border values, all border cells should be in result
        let result1 = Solution::pacific_atlantic_dfs(heights.clone());
        let result_set: HashSet<Vec<i32>> = result1.into_iter().collect();
        
        // Check corners - they should all be reachable since borders have same height
        assert!(result_set.contains(&vec![0, 0]));
        assert!(result_set.contains(&vec![0, 4]));
        assert!(result_set.contains(&vec![4, 0]));
        assert!(result_set.contains(&vec![4, 4]));
        
        // Check that entire border is included
        for i in 0..5 {
            assert!(result_set.contains(&vec![i, 0])); // Left border
            assert!(result_set.contains(&vec![i, 4])); // Right border
            assert!(result_set.contains(&vec![0, i])); // Top border
            assert!(result_set.contains(&vec![4, i])); // Bottom border
        }
    }
}