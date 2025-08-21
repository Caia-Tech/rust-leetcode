//! # 55. Jump Game
//!
//! You are given an integer array `nums`. You are initially positioned at the array's first index, 
//! and each element in the array represents your maximum jump length at that position.
//!
//! Return `true` if you can reach the last index, or `false` otherwise.
//!
//! **Example 1:**
//! ```
//! Input: nums = [2,3,1,1,4]
//! Output: true
//! Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
//! ```
//!
//! **Example 2:**
//! ```
//! Input: nums = [3,2,1,0,4]
//! Output: false
//! Explanation: You will always arrive at index 3 no matter what. Its maximum jump length 
//! is 0, which makes it impossible to reach the last index.
//! ```
//!
//! **Constraints:**
//! - 1 <= nums.length <= 10^4
//! - 0 <= nums[i] <= 10^5

/// Solution for Jump Game - 6 different approaches
pub struct Solution;

impl Solution {
    /// Approach 1: Greedy (Optimal)
    /// 
    /// Track the farthest position reachable so far. At each position,
    /// update the farthest reachable position and check if we can reach the end.
    ///
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn can_jump_greedy(&self, nums: Vec<i32>) -> bool {
        let n = nums.len();
        if n <= 1 { return true; }
        
        let mut farthest = 0;
        
        for i in 0..n {
            // If current position is beyond farthest reachable, we can't proceed
            if i > farthest {
                return false;
            }
            
            // Update farthest reachable position
            farthest = farthest.max(i + nums[i] as usize);
            
            // If we can reach or exceed the last index, return true
            if farthest >= n - 1 {
                return true;
            }
        }
        
        true
    }
    
    /// Approach 2: Dynamic Programming (Bottom-Up)
    /// 
    /// Use DP array where dp[i] represents if position i is reachable.
    /// For each position, mark all positions it can reach as reachable.
    ///
    /// Time Complexity: O(n²) in worst case
    /// Space Complexity: O(n)
    pub fn can_jump_dp(&self, nums: Vec<i32>) -> bool {
        let n = nums.len();
        if n <= 1 { return true; }
        
        let mut dp = vec![false; n];
        dp[0] = true;
        
        for i in 0..n {
            if !dp[i] { continue; }
            
            let max_jump = nums[i] as usize;
            for j in 1..=max_jump {
                if i + j >= n { break; }
                dp[i + j] = true;
                if i + j == n - 1 { return true; }
            }
        }
        
        dp[n - 1]
    }
    
    /// Approach 3: Backtracking with Memoization
    /// 
    /// Use recursion to try all possible jumps from each position,
    /// with memoization to avoid recomputing same subproblems.
    ///
    /// Time Complexity: O(n²)
    /// Space Complexity: O(n)
    pub fn can_jump_memo(&self, nums: Vec<i32>) -> bool {
        let n = nums.len();
        let mut memo = vec![None; n];
        self.can_jump_from(&nums, 0, &mut memo)
    }
    
    fn can_jump_from(&self, nums: &[i32], pos: usize, memo: &mut Vec<Option<bool>>) -> bool {
        if pos >= nums.len() - 1 {
            return true;
        }
        
        if let Some(cached) = memo[pos] {
            return cached;
        }
        
        let max_jump = nums[pos] as usize;
        for i in 1..=max_jump {
            if self.can_jump_from(nums, pos + i, memo) {
                memo[pos] = Some(true);
                return true;
            }
        }
        
        memo[pos] = Some(false);
        false
    }
    
    /// Approach 4: BFS (Breadth-First Search)
    /// 
    /// Treat each position as a node and each possible jump as an edge.
    /// Use BFS to explore all reachable positions level by level.
    ///
    /// Time Complexity: O(n²) in worst case
    /// Space Complexity: O(n)
    pub fn can_jump_bfs(&self, nums: Vec<i32>) -> bool {
        use std::collections::{VecDeque, HashSet};
        
        let n = nums.len();
        if n <= 1 { return true; }
        
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        
        queue.push_back(0);
        visited.insert(0);
        
        while let Some(pos) = queue.pop_front() {
            if pos >= n - 1 {
                return true;
            }
            
            let max_jump = nums[pos] as usize;
            for i in 1..=max_jump {
                let next_pos = pos + i;
                if next_pos >= n {
                    return true;
                }
                if !visited.contains(&next_pos) {
                    visited.insert(next_pos);
                    queue.push_back(next_pos);
                }
            }
        }
        
        false
    }
    
    /// Approach 5: Backward Greedy
    /// 
    /// Start from the end and work backwards. Track the leftmost position
    /// from which we can reach the target (initially the last position).
    ///
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn can_jump_backward(&self, nums: Vec<i32>) -> bool {
        let n = nums.len();
        if n <= 1 { return true; }
        
        let mut target = n - 1;
        
        for i in (0..n-1).rev() {
            if i + nums[i] as usize >= target {
                target = i;
            }
        }
        
        target == 0
    }
    
    /// Approach 6: Sliding Window Maximum
    /// 
    /// Use a sliding window approach where we track the maximum reachable
    /// position within the current window of reachable positions.
    ///
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn can_jump_sliding_window(&self, nums: Vec<i32>) -> bool {
        let n = nums.len();
        if n <= 1 { return true; }
        
        let mut start = 0;
        let mut end = 0;
        
        while end < n - 1 {
            let mut farthest = end;
            
            // Find the farthest position reachable from current window
            for i in start..=end {
                farthest = farthest.max(i + nums[i] as usize);
            }
            
            // If we can't make progress, we're stuck
            if farthest <= end {
                return false;
            }
            
            start = end + 1;
            end = farthest;
        }
        
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_examples() {
        let solution = Solution;
        
        // Example 1: [2,3,1,1,4] -> true
        assert_eq!(solution.can_jump_greedy(vec![2,3,1,1,4]), true);
        assert_eq!(solution.can_jump_dp(vec![2,3,1,1,4]), true);
        assert_eq!(solution.can_jump_memo(vec![2,3,1,1,4]), true);
        assert_eq!(solution.can_jump_bfs(vec![2,3,1,1,4]), true);
        assert_eq!(solution.can_jump_backward(vec![2,3,1,1,4]), true);
        assert_eq!(solution.can_jump_sliding_window(vec![2,3,1,1,4]), true);
        
        // Example 2: [3,2,1,0,4] -> false
        assert_eq!(solution.can_jump_greedy(vec![3,2,1,0,4]), false);
        assert_eq!(solution.can_jump_dp(vec![3,2,1,0,4]), false);
        assert_eq!(solution.can_jump_memo(vec![3,2,1,0,4]), false);
        assert_eq!(solution.can_jump_bfs(vec![3,2,1,0,4]), false);
        assert_eq!(solution.can_jump_backward(vec![3,2,1,0,4]), false);
        assert_eq!(solution.can_jump_sliding_window(vec![3,2,1,0,4]), false);
    }

    #[test]
    fn test_edge_cases() {
        let solution = Solution;
        
        // Single element
        assert_eq!(solution.can_jump_greedy(vec![0]), true);
        assert_eq!(solution.can_jump_dp(vec![0]), true);
        assert_eq!(solution.can_jump_memo(vec![0]), true);
        assert_eq!(solution.can_jump_bfs(vec![0]), true);
        assert_eq!(solution.can_jump_backward(vec![0]), true);
        assert_eq!(solution.can_jump_sliding_window(vec![0]), true);
        
        assert_eq!(solution.can_jump_greedy(vec![1]), true);
        assert_eq!(solution.can_jump_dp(vec![1]), true);
        assert_eq!(solution.can_jump_memo(vec![1]), true);
        assert_eq!(solution.can_jump_bfs(vec![1]), true);
        assert_eq!(solution.can_jump_backward(vec![1]), true);
        assert_eq!(solution.can_jump_sliding_window(vec![1]), true);
        
        // Two elements
        assert_eq!(solution.can_jump_greedy(vec![1,0]), true);
        assert_eq!(solution.can_jump_dp(vec![1,0]), true);
        assert_eq!(solution.can_jump_memo(vec![1,0]), true);
        assert_eq!(solution.can_jump_bfs(vec![1,0]), true);
        assert_eq!(solution.can_jump_backward(vec![1,0]), true);
        assert_eq!(solution.can_jump_sliding_window(vec![1,0]), true);
        
        assert_eq!(solution.can_jump_greedy(vec![0,1]), false);
        assert_eq!(solution.can_jump_dp(vec![0,1]), false);
        assert_eq!(solution.can_jump_memo(vec![0,1]), false);
        assert_eq!(solution.can_jump_bfs(vec![0,1]), false);
        assert_eq!(solution.can_jump_backward(vec![0,1]), false);
        assert_eq!(solution.can_jump_sliding_window(vec![0,1]), false);
    }

    #[test]
    fn test_all_zeros_except_first() {
        let solution = Solution;
        
        // Can reach with first jump
        assert_eq!(solution.can_jump_greedy(vec![1,0]), true);
        assert_eq!(solution.can_jump_dp(vec![1,0]), true);
        assert_eq!(solution.can_jump_memo(vec![1,0]), true);
        assert_eq!(solution.can_jump_bfs(vec![1,0]), true);
        assert_eq!(solution.can_jump_backward(vec![1,0]), true);
        assert_eq!(solution.can_jump_sliding_window(vec![1,0]), true);
        
        // Cannot reach due to zero in middle
        assert_eq!(solution.can_jump_greedy(vec![1,0,1]), false);
        assert_eq!(solution.can_jump_dp(vec![1,0,1]), false);
        assert_eq!(solution.can_jump_memo(vec![1,0,1]), false);
        assert_eq!(solution.can_jump_bfs(vec![1,0,1]), false);
        assert_eq!(solution.can_jump_backward(vec![1,0,1]), false);
        assert_eq!(solution.can_jump_sliding_window(vec![1,0,1]), false);
        
        // Can jump over zero
        assert_eq!(solution.can_jump_greedy(vec![2,0,1]), true);
        assert_eq!(solution.can_jump_dp(vec![2,0,1]), true);
        assert_eq!(solution.can_jump_memo(vec![2,0,1]), true);
        assert_eq!(solution.can_jump_bfs(vec![2,0,1]), true);
        assert_eq!(solution.can_jump_backward(vec![2,0,1]), true);
        assert_eq!(solution.can_jump_sliding_window(vec![2,0,1]), true);
    }

    #[test]
    fn test_large_jumps() {
        let solution = Solution;
        
        // Can reach in one jump
        assert_eq!(solution.can_jump_greedy(vec![5,1,1,1,1]), true);
        assert_eq!(solution.can_jump_dp(vec![5,1,1,1,1]), true);
        assert_eq!(solution.can_jump_memo(vec![5,1,1,1,1]), true);
        assert_eq!(solution.can_jump_bfs(vec![5,1,1,1,1]), true);
        assert_eq!(solution.can_jump_backward(vec![5,1,1,1,1]), true);
        assert_eq!(solution.can_jump_sliding_window(vec![5,1,1,1,1]), true);
        
        // Very large first jump
        assert_eq!(solution.can_jump_greedy(vec![100000,0,0,0,1]), true);
        assert_eq!(solution.can_jump_dp(vec![100000,0,0,0,1]), true);
        assert_eq!(solution.can_jump_memo(vec![100000,0,0,0,1]), true);
        assert_eq!(solution.can_jump_bfs(vec![100000,0,0,0,1]), true);
        assert_eq!(solution.can_jump_backward(vec![100000,0,0,0,1]), true);
        assert_eq!(solution.can_jump_sliding_window(vec![100000,0,0,0,1]), true);
    }

    #[test]
    fn test_decreasing_jumps() {
        let solution = Solution;
        
        // Decreasing but still reachable
        assert_eq!(solution.can_jump_greedy(vec![4,3,2,1,0]), true);
        assert_eq!(solution.can_jump_dp(vec![4,3,2,1,0]), true);
        assert_eq!(solution.can_jump_memo(vec![4,3,2,1,0]), true);
        assert_eq!(solution.can_jump_bfs(vec![4,3,2,1,0]), true);
        assert_eq!(solution.can_jump_backward(vec![4,3,2,1,0]), true);
        assert_eq!(solution.can_jump_sliding_window(vec![4,3,2,1,0]), true);
        
        // Decreasing and gets stuck
        assert_eq!(solution.can_jump_greedy(vec![3,2,1,0,4]), false);
        assert_eq!(solution.can_jump_dp(vec![3,2,1,0,4]), false);
        assert_eq!(solution.can_jump_memo(vec![3,2,1,0,4]), false);
        assert_eq!(solution.can_jump_bfs(vec![3,2,1,0,4]), false);
        assert_eq!(solution.can_jump_backward(vec![3,2,1,0,4]), false);
        assert_eq!(solution.can_jump_sliding_window(vec![3,2,1,0,4]), false);
    }

    #[test]
    fn test_all_ones() {
        let solution = Solution;
        
        // Array of all 1s should always be reachable
        assert_eq!(solution.can_jump_greedy(vec![1,1,1,1,1]), true);
        assert_eq!(solution.can_jump_dp(vec![1,1,1,1,1]), true);
        assert_eq!(solution.can_jump_memo(vec![1,1,1,1,1]), true);
        assert_eq!(solution.can_jump_bfs(vec![1,1,1,1,1]), true);
        assert_eq!(solution.can_jump_backward(vec![1,1,1,1,1]), true);
        assert_eq!(solution.can_jump_sliding_window(vec![1,1,1,1,1]), true);
    }

    #[test]
    fn test_complex_patterns() {
        let solution = Solution;
        
        // Complex reachable pattern
        assert_eq!(solution.can_jump_greedy(vec![2,3,1,1,4,0,0]), true);
        assert_eq!(solution.can_jump_dp(vec![2,3,1,1,4,0,0]), true);
        assert_eq!(solution.can_jump_memo(vec![2,3,1,1,4,0,0]), true);
        assert_eq!(solution.can_jump_bfs(vec![2,3,1,1,4,0,0]), true);
        assert_eq!(solution.can_jump_backward(vec![2,3,1,1,4,0,0]), true);
        assert_eq!(solution.can_jump_sliding_window(vec![2,3,1,1,4,0,0]), true);
        
        // Multiple possible paths
        assert_eq!(solution.can_jump_greedy(vec![1,2,3,0,4]), true);
        assert_eq!(solution.can_jump_dp(vec![1,2,3,0,4]), true);
        assert_eq!(solution.can_jump_memo(vec![1,2,3,0,4]), true);
        assert_eq!(solution.can_jump_bfs(vec![1,2,3,0,4]), true);
        assert_eq!(solution.can_jump_backward(vec![1,2,3,0,4]), true);
        assert_eq!(solution.can_jump_sliding_window(vec![1,2,3,0,4]), true);
        
        // Requires specific path
        assert_eq!(solution.can_jump_greedy(vec![2,0,1,0,4]), false);
        assert_eq!(solution.can_jump_dp(vec![2,0,1,0,4]), false);
        assert_eq!(solution.can_jump_memo(vec![2,0,1,0,4]), false);
        assert_eq!(solution.can_jump_bfs(vec![2,0,1,0,4]), false);
        assert_eq!(solution.can_jump_backward(vec![2,0,1,0,4]), false);
        assert_eq!(solution.can_jump_sliding_window(vec![2,0,1,0,4]), false);
    }

    #[test]
    fn test_boundary_values() {
        let solution = Solution;
        
        // Maximum jump value
        assert_eq!(solution.can_jump_greedy(vec![100000]), true);
        assert_eq!(solution.can_jump_dp(vec![100000]), true);
        assert_eq!(solution.can_jump_memo(vec![100000]), true);
        assert_eq!(solution.can_jump_bfs(vec![100000]), true);
        assert_eq!(solution.can_jump_backward(vec![100000]), true);
        assert_eq!(solution.can_jump_sliding_window(vec![100000]), true);
        
        // Mix of zeros and maximum values
        assert_eq!(solution.can_jump_greedy(vec![0,100000]), false);
        assert_eq!(solution.can_jump_dp(vec![0,100000]), false);
        assert_eq!(solution.can_jump_memo(vec![0,100000]), false);
        assert_eq!(solution.can_jump_bfs(vec![0,100000]), false);
        assert_eq!(solution.can_jump_backward(vec![0,100000]), false);
        assert_eq!(solution.can_jump_sliding_window(vec![0,100000]), false);
    }

    #[test]
    fn test_minimum_jumps_needed() {
        let solution = Solution;
        
        // Requires minimum jumps
        assert_eq!(solution.can_jump_greedy(vec![1,1,1,1]), true);
        assert_eq!(solution.can_jump_dp(vec![1,1,1,1]), true);
        assert_eq!(solution.can_jump_memo(vec![1,1,1,1]), true);
        assert_eq!(solution.can_jump_bfs(vec![1,1,1,1]), true);
        assert_eq!(solution.can_jump_backward(vec![1,1,1,1]), true);
        assert_eq!(solution.can_jump_sliding_window(vec![1,1,1,1]), true);
        
        // Can be done in fewer jumps
        assert_eq!(solution.can_jump_greedy(vec![2,1,1,1]), true);
        assert_eq!(solution.can_jump_dp(vec![2,1,1,1]), true);
        assert_eq!(solution.can_jump_memo(vec![2,1,1,1]), true);
        assert_eq!(solution.can_jump_bfs(vec![2,1,1,1]), true);
        assert_eq!(solution.can_jump_backward(vec![2,1,1,1]), true);
        assert_eq!(solution.can_jump_sliding_window(vec![2,1,1,1]), true);
    }

    #[test]
    fn test_performance_edge_cases() {
        let solution = Solution;
        
        // Large array that should be reachable
        let large_reachable = vec![1; 1000];
        assert_eq!(solution.can_jump_greedy(large_reachable.clone()), true);
        
        // Large array with early termination
        let mut large_unreachable = vec![1; 1000];
        large_unreachable[500] = 0;
        for i in 501..1000 {
            large_unreachable[i] = 0;
        }
        assert_eq!(solution.can_jump_greedy(large_unreachable), false);
    }
}