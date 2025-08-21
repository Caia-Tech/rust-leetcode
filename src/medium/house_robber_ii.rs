//! # 213. House Robber II
//!
//! You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. 
//! All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. 
//! Meanwhile, adjacent houses have security systems connected and it will automatically contact the police if two 
//! adjacent houses were broken into on the same night.
//!
//! Given an integer array nums representing the amount of money of each house, return the maximum amount of money 
//! you can rob tonight without alerting the police.
//!
//! **Example 1:**
//! ```
//! Input: nums = [2,3,2]
//! Output: 3
//! Explanation: You cannot rob house 0 and 2 (they are adjacent) since they are adjacent, so you rob house 1.
//! ```
//!
//! **Example 2:**
//! ```
//! Input: nums = [1,2,3,1]
//! Output: 4
//! Explanation: Rob house 1 (money = 2) and house 3 (money = 1). Total amount = 2 + 1 = 4.
//! ```
//!
//! **Constraints:**
//! - 1 <= nums.length <= 100
//! - 0 <= nums[i] <= 1000

/// Solution for House Robber II - 6 different approaches
pub struct Solution;

impl Solution {
    /// Approach 1: Two Linear House Robber Calls (Most Intuitive)
    /// 
    /// Since houses are arranged in a circle, we can't rob both first and last house.
    /// So we solve two linear problems:
    /// 1. Rob houses 0 to n-2 (exclude last house)
    /// 2. Rob houses 1 to n-1 (exclude first house)
    /// Return the maximum of both.
    ///
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn rob_two_linear_calls(&self, nums: Vec<i32>) -> i32 {
        let n = nums.len();
        if n == 1 { return nums[0]; }
        if n == 2 { return nums[0].max(nums[1]); }
        
        // Case 1: Rob houses 0 to n-2 (exclude last)
        let rob_first = self.rob_linear(&nums[0..n-1]);
        
        // Case 2: Rob houses 1 to n-1 (exclude first)
        let rob_last = self.rob_linear(&nums[1..n]);
        
        rob_first.max(rob_last)
    }
    
    /// Helper function for linear house robber problem
    fn rob_linear(&self, nums: &[i32]) -> i32 {
        if nums.is_empty() { return 0; }
        if nums.len() == 1 { return nums[0]; }
        
        let mut prev2 = 0;
        let mut prev1 = nums[0];
        
        for i in 1..nums.len() {
            let current = prev1.max(prev2 + nums[i]);
            prev2 = prev1;
            prev1 = current;
        }
        
        prev1
    }
    
    /// Approach 2: Single Pass with Two States
    /// 
    /// Track two scenarios simultaneously:
    /// - State 1: Rob first house, cannot rob last
    /// - State 2: Don't rob first house, can rob last
    ///
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn rob_single_pass(&self, nums: Vec<i32>) -> i32 {
        let n = nums.len();
        if n == 1 { return nums[0]; }
        if n == 2 { return nums[0].max(nums[1]); }
        
        // State 1: Include first house (exclude last)
        let mut include_first_prev2 = 0;
        let mut include_first_prev1 = nums[0];
        
        // State 2: Exclude first house (can include last)
        let mut exclude_first_prev2 = 0;
        let mut exclude_first_prev1 = 0;
        
        for i in 1..n {
            if i == n - 1 {
                // Last house: can't rob if we robbed first
                let new_exclude_first = exclude_first_prev1.max(exclude_first_prev2 + nums[i]);
                exclude_first_prev2 = exclude_first_prev1;
                exclude_first_prev1 = new_exclude_first;
            } else {
                // Regular house: update both states
                let new_include_first = include_first_prev1.max(include_first_prev2 + nums[i]);
                include_first_prev2 = include_first_prev1;
                include_first_prev1 = new_include_first;
                
                let new_exclude_first = exclude_first_prev1.max(exclude_first_prev2 + nums[i]);
                exclude_first_prev2 = exclude_first_prev1;
                exclude_first_prev1 = new_exclude_first;
            }
        }
        
        include_first_prev1.max(exclude_first_prev1)
    }
    
    /// Approach 3: Memoized Recursion with Range
    /// 
    /// Use recursion with memoization, but handle the circular constraint
    /// by solving two subproblems with different valid ranges.
    ///
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    pub fn rob_memoized_recursion(&self, nums: Vec<i32>) -> i32 {
        use std::collections::HashMap;
        
        let n = nums.len();
        if n == 1 { return nums[0]; }
        if n == 2 { return nums[0].max(nums[1]); }
        
        let mut memo1 = HashMap::new();
        let mut memo2 = HashMap::new();
        
        // Case 1: Can rob first house (0 to n-2)
        let rob_first = self.rob_memo_helper(&nums, 0, n - 2, &mut memo1);
        
        // Case 2: Cannot rob first house (1 to n-1)  
        let rob_last = self.rob_memo_helper(&nums, 1, n - 1, &mut memo2);
        
        rob_first.max(rob_last)
    }
    
    fn rob_memo_helper(&self, nums: &[i32], start: usize, end: usize, memo: &mut std::collections::HashMap<usize, i32>) -> i32 {
        if start > end { return 0; }
        if start == end { return nums[start]; }
        
        if let Some(&cached) = memo.get(&start) {
            return cached;
        }
        
        let rob_current = nums[start] + self.rob_memo_helper(nums, start + 2, end, memo);
        let skip_current = self.rob_memo_helper(nums, start + 1, end, memo);
        
        let result = rob_current.max(skip_current);
        memo.insert(start, result);
        result
    }
    
    /// Approach 4: DP Array with Two Cases
    /// 
    /// Create DP arrays for both scenarios and return the maximum.
    /// More explicit about the state transitions.
    ///
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    pub fn rob_dp_array(&self, nums: Vec<i32>) -> i32 {
        let n = nums.len();
        if n == 1 { return nums[0]; }
        if n == 2 { return nums[0].max(nums[1]); }
        
        // Case 1: Rob first house, cannot rob last (0 to n-2)
        let mut dp1 = vec![0; n - 1];
        dp1[0] = nums[0];
        if dp1.len() > 1 {
            dp1[1] = nums[0].max(nums[1]);
            for i in 2..dp1.len() {
                dp1[i] = dp1[i-1].max(dp1[i-2] + nums[i]);
            }
        }
        
        // Case 2: Don't rob first house, can rob last (1 to n-1)
        let mut dp2 = vec![0; n - 1];
        dp2[0] = nums[1];
        if dp2.len() > 1 {
            dp2[1] = nums[1].max(nums[2]);
            for i in 2..dp2.len() {
                dp2[i] = dp2[i-1].max(dp2[i-2] + nums[i+1]);
            }
        }
        
        dp1[dp1.len()-1].max(dp2[dp2.len()-1])
    }
    
    /// Approach 5: DP with Four States
    /// 
    /// Track four possible states at each position:
    /// - Rob first house, rob current house
    /// - Rob first house, skip current house  
    /// - Skip first house, rob current house
    /// - Skip first house, skip current house
    ///
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn rob_state_machine(&self, nums: Vec<i32>) -> i32 {
        let n = nums.len();
        if n == 1 { return nums[0]; }
        if n == 2 { return nums[0].max(nums[1]); }
        
        // Four states: [rob_first_rob_current, rob_first_skip_current, skip_first_rob_current, skip_first_skip_current]
        let mut dp = vec![vec![0; 4]; n];
        
        // Initialize first house
        dp[0][0] = nums[0];  // Rob first, rob current (same house)
        dp[0][1] = 0;        // Rob first, skip current (impossible)
        dp[0][2] = 0;        // Skip first, rob current (impossible)
        dp[0][3] = 0;        // Skip first, skip current
        
        for i in 1..n {
            if i == n - 1 {
                // Last house: can't rob if we robbed first
                dp[i][0] = dp[i-1][1];  // Can't rob last if robbed first
                dp[i][1] = dp[i-1][0].max(dp[i-1][1]); // Skip last
                dp[i][2] = dp[i-1][3] + nums[i]; // Rob last if didn't rob first
                dp[i][3] = dp[i-1][2].max(dp[i-1][3]); // Skip last
            } else {
                // Regular house: all transitions allowed
                dp[i][0] = dp[i-1][1] + nums[i]; // Rob current after skipping previous
                dp[i][1] = dp[i-1][0].max(dp[i-1][1]); // Skip current
                dp[i][2] = dp[i-1][3] + nums[i]; // Rob current after skipping previous
                dp[i][3] = dp[i-1][2].max(dp[i-1][3]); // Skip current
            }
        }
        
        dp[n-1].iter().cloned().max().unwrap_or(0)
    }
    
    /// Approach 6: Optimized Single Pass (Most Efficient)
    /// 
    /// Combines the logic of both linear cases into a single optimized pass.
    /// Tracks the maximum money for both scenarios simultaneously.
    ///
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn rob_optimized(&self, nums: Vec<i32>) -> i32 {
        let n = nums.len();
        if n == 1 { return nums[0]; }
        if n == 2 { return nums[0].max(nums[1]); }
        
        // For case 1: rob houses [0, n-2]
        let mut rob1 = 0;
        let mut not_rob1 = 0;
        
        // For case 2: rob houses [1, n-1]
        let mut rob2 = 0;
        let mut not_rob2 = 0;
        
        for i in 0..n {
            if i == 0 {
                // First house: only case 1 can rob it
                rob1 = nums[0];
                not_rob1 = 0;
            } else if i == n - 1 {
                // Last house: only case 2 can rob it
                let new_rob2 = not_rob2 + nums[i];
                let new_not_rob2 = rob2.max(not_rob2);
                rob2 = new_rob2;
                not_rob2 = new_not_rob2;
            } else {
                // Middle houses: both cases can rob
                let new_rob1 = not_rob1 + nums[i];
                let new_not_rob1 = rob1.max(not_rob1);
                rob1 = new_rob1;
                not_rob1 = new_not_rob1;
                
                let new_rob2 = not_rob2 + nums[i];
                let new_not_rob2 = rob2.max(not_rob2);
                rob2 = new_rob2;
                not_rob2 = new_not_rob2;
            }
        }
        
        let max1 = rob1.max(not_rob1);
        let max2 = rob2.max(not_rob2);
        max1.max(max2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_cases() {
        let solution = Solution;
        
        // Example 1
        assert_eq!(solution.rob_two_linear_calls(vec![2,3,2]), 3);
        assert_eq!(solution.rob_single_pass(vec![2,3,2]), 3);
        assert_eq!(solution.rob_memoized_recursion(vec![2,3,2]), 3);
        assert_eq!(solution.rob_dp_array(vec![2,3,2]), 3);
        assert_eq!(solution.rob_state_machine(vec![2,3,2]), 3);
        assert_eq!(solution.rob_optimized(vec![2,3,2]), 3);
        
        // Example 2
        assert_eq!(solution.rob_two_linear_calls(vec![1,2,3,1]), 4);
        assert_eq!(solution.rob_single_pass(vec![1,2,3,1]), 4);
        assert_eq!(solution.rob_memoized_recursion(vec![1,2,3,1]), 4);
        assert_eq!(solution.rob_dp_array(vec![1,2,3,1]), 4);
        assert_eq!(solution.rob_state_machine(vec![1,2,3,1]), 4);
        assert_eq!(solution.rob_optimized(vec![1,2,3,1]), 4);
    }

    #[test]
    fn test_edge_cases() {
        let solution = Solution;
        
        // Single house
        assert_eq!(solution.rob_two_linear_calls(vec![5]), 5);
        assert_eq!(solution.rob_single_pass(vec![5]), 5);
        assert_eq!(solution.rob_memoized_recursion(vec![5]), 5);
        assert_eq!(solution.rob_dp_array(vec![5]), 5);
        assert_eq!(solution.rob_state_machine(vec![5]), 5);
        assert_eq!(solution.rob_optimized(vec![5]), 5);
        
        // Two houses
        assert_eq!(solution.rob_two_linear_calls(vec![1,2]), 2);
        assert_eq!(solution.rob_single_pass(vec![1,2]), 2);
        assert_eq!(solution.rob_memoized_recursion(vec![1,2]), 2);
        assert_eq!(solution.rob_dp_array(vec![1,2]), 2);
        assert_eq!(solution.rob_state_machine(vec![1,2]), 2);
        assert_eq!(solution.rob_optimized(vec![1,2]), 2);
        
        // Two houses equal
        assert_eq!(solution.rob_two_linear_calls(vec![5,5]), 5);
        assert_eq!(solution.rob_single_pass(vec![5,5]), 5);
        assert_eq!(solution.rob_memoized_recursion(vec![5,5]), 5);
        assert_eq!(solution.rob_dp_array(vec![5,5]), 5);
        assert_eq!(solution.rob_state_machine(vec![5,5]), 5);
        assert_eq!(solution.rob_optimized(vec![5,5]), 5);
    }

    #[test]
    fn test_three_houses() {
        let solution = Solution;
        
        // Three houses ascending
        assert_eq!(solution.rob_two_linear_calls(vec![1,2,3]), 3);
        assert_eq!(solution.rob_single_pass(vec![1,2,3]), 3);
        assert_eq!(solution.rob_memoized_recursion(vec![1,2,3]), 3);
        assert_eq!(solution.rob_dp_array(vec![1,2,3]), 3);
        assert_eq!(solution.rob_state_machine(vec![1,2,3]), 3);
        assert_eq!(solution.rob_optimized(vec![1,2,3]), 3);
        
        // Three houses with middle being largest
        assert_eq!(solution.rob_two_linear_calls(vec![1,5,1]), 5);
        assert_eq!(solution.rob_single_pass(vec![1,5,1]), 5);
        assert_eq!(solution.rob_memoized_recursion(vec![1,5,1]), 5);
        assert_eq!(solution.rob_dp_array(vec![1,5,1]), 5);
        assert_eq!(solution.rob_state_machine(vec![1,5,1]), 5);
        assert_eq!(solution.rob_optimized(vec![1,5,1]), 5);
    }

    #[test]
    fn test_larger_cases() {
        let solution = Solution;
        
        // Five houses
        assert_eq!(solution.rob_two_linear_calls(vec![2,3,2,3,2]), 6);
        assert_eq!(solution.rob_single_pass(vec![2,3,2,3,2]), 6);
        assert_eq!(solution.rob_memoized_recursion(vec![2,3,2,3,2]), 6);
        assert_eq!(solution.rob_dp_array(vec![2,3,2,3,2]), 6);
        assert_eq!(solution.rob_state_machine(vec![2,3,2,3,2]), 6);
        assert_eq!(solution.rob_optimized(vec![2,3,2,3,2]), 6);
        
        // Six houses
        assert_eq!(solution.rob_two_linear_calls(vec![1,2,3,4,5,6]), 12);
        assert_eq!(solution.rob_single_pass(vec![1,2,3,4,5,6]), 12);
        assert_eq!(solution.rob_memoized_recursion(vec![1,2,3,4,5,6]), 12);
        assert_eq!(solution.rob_dp_array(vec![1,2,3,4,5,6]), 12);
        assert_eq!(solution.rob_state_machine(vec![1,2,3,4,5,6]), 12);
        assert_eq!(solution.rob_optimized(vec![1,2,3,4,5,6]), 12);
    }

    #[test]
    fn test_zeros_and_high_values() {
        let solution = Solution;
        
        // With zeros
        assert_eq!(solution.rob_two_linear_calls(vec![0,1,0,2,0]), 3);
        assert_eq!(solution.rob_single_pass(vec![0,1,0,2,0]), 3);
        assert_eq!(solution.rob_memoized_recursion(vec![0,1,0,2,0]), 3);
        assert_eq!(solution.rob_dp_array(vec![0,1,0,2,0]), 3);
        assert_eq!(solution.rob_state_machine(vec![0,1,0,2,0]), 3);
        assert_eq!(solution.rob_optimized(vec![0,1,0,2,0]), 3);
        
        // All zeros
        assert_eq!(solution.rob_two_linear_calls(vec![0,0,0]), 0);
        assert_eq!(solution.rob_single_pass(vec![0,0,0]), 0);
        assert_eq!(solution.rob_memoized_recursion(vec![0,0,0]), 0);
        assert_eq!(solution.rob_dp_array(vec![0,0,0]), 0);
        assert_eq!(solution.rob_state_machine(vec![0,0,0]), 0);
        assert_eq!(solution.rob_optimized(vec![0,0,0]), 0);
        
        // High values
        assert_eq!(solution.rob_two_linear_calls(vec![1000,500,1000]), 1000);
        assert_eq!(solution.rob_single_pass(vec![1000,500,1000]), 1000);
        assert_eq!(solution.rob_memoized_recursion(vec![1000,500,1000]), 1000);
        assert_eq!(solution.rob_dp_array(vec![1000,500,1000]), 1000);
        assert_eq!(solution.rob_state_machine(vec![1000,500,1000]), 1000);
        assert_eq!(solution.rob_optimized(vec![1000,500,1000]), 1000);
    }

    #[test]
    fn test_alternating_pattern() {
        let solution = Solution;
        
        // Alternating high-low
        assert_eq!(solution.rob_two_linear_calls(vec![5,1,5,1]), 10);
        assert_eq!(solution.rob_single_pass(vec![5,1,5,1]), 10);
        assert_eq!(solution.rob_memoized_recursion(vec![5,1,5,1]), 10);
        assert_eq!(solution.rob_dp_array(vec![5,1,5,1]), 10);
        assert_eq!(solution.rob_state_machine(vec![5,1,5,1]), 10);
        assert_eq!(solution.rob_optimized(vec![5,1,5,1]), 10);
        
        // Complex pattern
        assert_eq!(solution.rob_two_linear_calls(vec![2,1,1,2]), 3);
        assert_eq!(solution.rob_single_pass(vec![2,1,1,2]), 3);
        assert_eq!(solution.rob_memoized_recursion(vec![2,1,1,2]), 3);
        assert_eq!(solution.rob_dp_array(vec![2,1,1,2]), 3);
        assert_eq!(solution.rob_state_machine(vec![2,1,1,2]), 3);
        assert_eq!(solution.rob_optimized(vec![2,1,1,2]), 3);
    }

    #[test]
    fn test_boundary_constraint_cases() {
        let solution = Solution;
        
        // First house is maximum
        assert_eq!(solution.rob_two_linear_calls(vec![10,1,2,1]), 12);
        assert_eq!(solution.rob_single_pass(vec![10,1,2,1]), 12);
        assert_eq!(solution.rob_memoized_recursion(vec![10,1,2,1]), 12);
        assert_eq!(solution.rob_dp_array(vec![10,1,2,1]), 12);
        assert_eq!(solution.rob_state_machine(vec![10,1,2,1]), 12);
        assert_eq!(solution.rob_optimized(vec![10,1,2,1]), 12);
        
        // Last house is maximum
        assert_eq!(solution.rob_two_linear_calls(vec![1,2,1,10]), 12);
        assert_eq!(solution.rob_single_pass(vec![1,2,1,10]), 12);
        assert_eq!(solution.rob_memoized_recursion(vec![1,2,1,10]), 12);
        assert_eq!(solution.rob_dp_array(vec![1,2,1,10]), 12);
        assert_eq!(solution.rob_state_machine(vec![1,2,1,10]), 12);
        assert_eq!(solution.rob_optimized(vec![1,2,1,10]), 12);
        
        // Both first and last are high
        assert_eq!(solution.rob_two_linear_calls(vec![10,1,1,10]), 11);
        assert_eq!(solution.rob_single_pass(vec![10,1,1,10]), 11);
        assert_eq!(solution.rob_memoized_recursion(vec![10,1,1,10]), 11);
        assert_eq!(solution.rob_dp_array(vec![10,1,1,10]), 11);
        assert_eq!(solution.rob_state_machine(vec![10,1,1,10]), 11);
        assert_eq!(solution.rob_optimized(vec![10,1,1,10]), 11);
    }
}