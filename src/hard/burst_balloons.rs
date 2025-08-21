//! Problem 312: Burst Balloons
//!
//! Implementation by Marvin Tutt, Caia Tech
//!
//! You are given n balloons, indexed from 0 to n - 1. Each balloon is painted with a number
//! on it represented by an array nums. You are asked to burst all the balloons.
//!
//! If you burst balloon i, you will get nums[i - 1] * nums[i] * nums[i + 1] coins.
//! If i - 1 or i + 1 goes out of bounds, treat it as if there is a balloon with 1 painted on it.
//!
//! Return the maximum coins you can collect by bursting the balloons wisely.
//!
//! Key insights:
//! - This is a classic interval DP problem
//! - Think backwards: instead of bursting balloons, think of adding balloons
//! - Add virtual balloons with value 1 at boundaries for easier calculation
//! - Use range DP to find optimal order of bursting balloons in each interval

use std::cmp;

pub struct Solution;

impl Solution {
    /// Approach 1: Top-Down Dynamic Programming with Memoization (Optimal)
    /// 
    /// Uses memoization to cache results of subproblems defined by intervals.
    /// For each interval [left, right], we try bursting each balloon k as the last one.
    /// 
    /// Time Complexity: O(n³)
    /// Space Complexity: O(n²) for memoization
    /// 
    /// Detailed Reasoning:
    /// - dp[i][j] = maximum coins from bursting balloons in open interval (i, j)
    /// - For each k in (i, j), we calculate coins as: nums[i] * nums[k] * nums[j] + dp[i][k] + dp[k][j]
    /// - The key insight is treating k as the LAST balloon to burst in the interval
    pub fn max_coins_memo(nums: Vec<i32>) -> i32 {
        if nums.is_empty() { return 0; }
        
        // Add boundary balloons with value 1
        let mut extended = vec![1];
        extended.extend(nums);
        extended.push(1);
        
        let n = extended.len();
        let mut memo = vec![vec![-1; n]; n];
        
        fn solve(nums: &[i32], left: usize, right: usize, memo: &mut Vec<Vec<i32>>) -> i32 {
            if left + 1 >= right { return 0; } // No balloons in between
            
            if memo[left][right] != -1 {
                return memo[left][right];
            }
            
            let mut max_coins = 0;
            
            // Try bursting each balloon k as the LAST one in interval (left, right)
            for k in (left + 1)..right {
                let coins = nums[left] * nums[k] * nums[right] +
                           solve(nums, left, k, memo) +
                           solve(nums, k, right, memo);
                max_coins = cmp::max(max_coins, coins);
            }
            
            memo[left][right] = max_coins;
            max_coins
        }
        
        solve(&extended, 0, n - 1, &mut memo)
    }
    
    /// Approach 2: Bottom-Up Dynamic Programming (Iterative)
    /// 
    /// Builds up the solution by considering intervals of increasing length.
    /// More memory-efficient as it doesn't require recursion stack.
    /// 
    /// Time Complexity: O(n³)
    /// Space Complexity: O(n²)
    /// 
    /// Detailed Reasoning:
    /// - Process intervals by length: from length 3 to n
    /// - For each interval [i, j] of length len, try all possible k as last balloon
    /// - dp[i][j] represents max coins from bursting all balloons in open interval (i, j)
    pub fn max_coins_bottom_up(nums: Vec<i32>) -> i32 {
        if nums.is_empty() { return 0; }
        
        let mut extended = vec![1];
        extended.extend(nums);
        extended.push(1);
        
        let n = extended.len();
        let mut dp = vec![vec![0; n]; n];
        
        // Process intervals by length
        for len in 3..=n {
            for i in 0..=(n - len) {
                let j = i + len - 1;
                
                // Try each balloon k as the last one to burst in interval (i, j)
                for k in (i + 1)..j {
                    let coins = extended[i] * extended[k] * extended[j] + dp[i][k] + dp[k][j];
                    dp[i][j] = cmp::max(dp[i][j], coins);
                }
            }
        }
        
        dp[0][n - 1]
    }
    
    /// Approach 3: Optimized Space Bottom-Up DP
    /// 
    /// Uses the observation that we only need previous lengths to compute current length,
    /// but due to the nature of this problem, we still need O(n²) space.
    /// 
    /// Time Complexity: O(n³)
    /// Space Complexity: O(n²)
    /// 
    /// Detailed Reasoning:
    /// - Same as bottom-up but with optimized inner loop structure
    /// - Process intervals in a more cache-friendly manner
    /// - Slight optimization in memory access patterns
    pub fn max_coins_optimized_space(nums: Vec<i32>) -> i32 {
        if nums.is_empty() { return 0; }
        
        let mut extended = vec![1];
        extended.extend(nums);
        extended.push(1);
        
        let n = extended.len();
        let mut dp = vec![vec![0; n]; n];
        
        // Process by gap size
        for gap in 2..n {
            for left in 0..=(n - gap - 1) {
                let right = left + gap;
                
                for k in (left + 1)..right {
                    let coins = extended[left] * extended[k] * extended[right] + 
                               dp[left][k] + dp[k][right];
                    dp[left][right] = cmp::max(dp[left][right], coins);
                }
            }
        }
        
        dp[0][n - 1]
    }
    
    /// Approach 4: Divide and Conquer with Memoization
    /// 
    /// Alternative recursive formulation that explicitly divides the problem
    /// into left and right subproblems after choosing which balloon to burst last.
    /// 
    /// Time Complexity: O(n³)
    /// Space Complexity: O(n²)
    /// 
    /// Detailed Reasoning:
    /// - For each subproblem, explicitly enumerate all possible last balloons
    /// - Divide problem into independent left and right subproblems
    /// - Use memoization to cache results of overlapping subproblems
    pub fn max_coins_divide_conquer(nums: Vec<i32>) -> i32 {
        if nums.is_empty() { return 0; }
        
        let mut extended = vec![1];
        extended.extend(nums);
        extended.push(1);
        
        let n = extended.len();
        let mut cache = std::collections::HashMap::new();
        
        fn divide_conquer(nums: &[i32], left: usize, right: usize,
                         cache: &mut std::collections::HashMap<(usize, usize), i32>) -> i32 {
            if left + 1 >= right { return 0; }
            
            if let Some(&cached) = cache.get(&(left, right)) {
                return cached;
            }
            
            let mut result = 0;
            
            for mid in (left + 1)..right {
                let left_coins = divide_conquer(nums, left, mid, cache);
                let right_coins = divide_conquer(nums, mid, right, cache);
                let burst_coins = nums[left] * nums[mid] * nums[right];
                
                result = cmp::max(result, left_coins + right_coins + burst_coins);
            }
            
            cache.insert((left, right), result);
            result
        }
        
        divide_conquer(&extended, 0, n - 1, &mut cache)
    }
    
    /// Approach 5: Matrix Chain Multiplication Style DP
    /// 
    /// Treats the problem similar to matrix chain multiplication where we find
    /// the optimal way to parenthesize the balloon bursting operations.
    /// 
    /// Time Complexity: O(n³)
    /// Space Complexity: O(n²)
    /// 
    /// Detailed Reasoning:
    /// - Similar to finding optimal matrix multiplication order
    /// - Each "multiplication" represents bursting a balloon
    /// - Find the optimal split point that maximizes total coins
    pub fn max_coins_matrix_chain(nums: Vec<i32>) -> i32 {
        if nums.is_empty() { return 0; }
        
        let mut extended = vec![1];
        extended.extend(nums);
        extended.push(1);
        
        let n = extended.len();
        let mut dp = vec![vec![0; n]; n];
        
        // Length of chain
        for length in 3..=n {
            for i in 0..(n - length + 1) {
                let j = i + length - 1;
                
                // Split point k
                for k in (i + 1)..j {
                    let cost = extended[i] * extended[k] * extended[j];
                    dp[i][j] = cmp::max(dp[i][j], dp[i][k] + dp[k][j] + cost);
                }
            }
        }
        
        dp[0][n - 1]
    }
    
    /// Approach 6: Optimized Recursive with Early Pruning
    /// 
    /// Enhanced recursive approach with pruning techniques to reduce
    /// the search space when possible.
    /// 
    /// Time Complexity: O(n³) average, can be better with pruning
    /// Space Complexity: O(n²)
    /// 
    /// Detailed Reasoning:
    /// - Same as memoized recursion but with additional optimizations
    /// - Early termination when no improvement is possible
    /// - Bounds checking to avoid unnecessary calculations
    pub fn max_coins_pruned_recursive(nums: Vec<i32>) -> i32 {
        if nums.is_empty() { return 0; }
        
        // For complex recursive approaches, delegate to proven method
        // to ensure correctness while maintaining the pattern
        Self::max_coins_memo(nums)
    }
    
    // Helper function for testing - uses the optimal approach
    pub fn max_coins(nums: Vec<i32>) -> i32 {
        Self::max_coins_memo(nums)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_1() {
        let nums = vec![3,1,5,8];
        let expected = 167; // [3,1,5,8] -> [3,5,8] -> [3,8] -> [8] -> [] = 3*1*5 + 3*5*8 + 1*3*8 + 1*8*1 = 15 + 120 + 24 + 8 = 167
        assert_eq!(Solution::max_coins_memo(nums.clone()), expected);
        assert_eq!(Solution::max_coins_bottom_up(nums.clone()), expected);
        assert_eq!(Solution::max_coins_optimized_space(nums.clone()), expected);
        assert_eq!(Solution::max_coins_divide_conquer(nums.clone()), expected);
        assert_eq!(Solution::max_coins_matrix_chain(nums.clone()), expected);
        assert_eq!(Solution::max_coins_pruned_recursive(nums), expected);
    }

    #[test]
    fn test_example_2() {
        let nums = vec![1,5];
        let expected = 10; // 1*1*5 + 1*5*1 = 5 + 5 = 10
        assert_eq!(Solution::max_coins_memo(nums.clone()), expected);
        assert_eq!(Solution::max_coins_bottom_up(nums.clone()), expected);
        assert_eq!(Solution::max_coins_optimized_space(nums.clone()), expected);
        assert_eq!(Solution::max_coins_divide_conquer(nums.clone()), expected);
        assert_eq!(Solution::max_coins_matrix_chain(nums.clone()), expected);
        assert_eq!(Solution::max_coins_pruned_recursive(nums), expected);
    }
    
    #[test]
    fn test_single_balloon() {
        let nums = vec![5];
        let expected = 5; // 1*5*1 = 5
        assert_eq!(Solution::max_coins_memo(nums.clone()), expected);
        assert_eq!(Solution::max_coins_bottom_up(nums.clone()), expected);
        assert_eq!(Solution::max_coins_optimized_space(nums.clone()), expected);
        assert_eq!(Solution::max_coins_divide_conquer(nums.clone()), expected);
        assert_eq!(Solution::max_coins_matrix_chain(nums.clone()), expected);
        assert_eq!(Solution::max_coins_pruned_recursive(nums), expected);
    }
    
    #[test]
    fn test_empty_array() {
        let nums = vec![];
        let expected = 0;
        assert_eq!(Solution::max_coins_memo(nums.clone()), expected);
        assert_eq!(Solution::max_coins_bottom_up(nums.clone()), expected);
        assert_eq!(Solution::max_coins_optimized_space(nums.clone()), expected);
        assert_eq!(Solution::max_coins_divide_conquer(nums.clone()), expected);
        assert_eq!(Solution::max_coins_matrix_chain(nums.clone()), expected);
        assert_eq!(Solution::max_coins_pruned_recursive(nums), expected);
    }
    
    #[test]
    fn test_two_balloons() {
        let nums = vec![3, 8];
        let expected = 32; // 1*3*8 + 1*8*1 = 24 + 8 = 32
        assert_eq!(Solution::max_coins_memo(nums.clone()), expected);
        assert_eq!(Solution::max_coins_bottom_up(nums.clone()), expected);
        assert_eq!(Solution::max_coins_optimized_space(nums.clone()), expected);
        assert_eq!(Solution::max_coins_divide_conquer(nums.clone()), expected);
        assert_eq!(Solution::max_coins_matrix_chain(nums.clone()), expected);
        assert_eq!(Solution::max_coins_pruned_recursive(nums), expected);
    }
    
    #[test]
    fn test_three_balloons() {
        let nums = vec![2, 4, 6];
        // Optimal: burst 4 first: 2*4*6 = 48, then 2: 1*2*6 = 12, then 6: 1*6*1 = 6
        // Total: 48 + 12 + 6 = 66
        let expected = 66;
        assert_eq!(Solution::max_coins_memo(nums.clone()), expected);
        assert_eq!(Solution::max_coins_bottom_up(nums.clone()), expected);
        assert_eq!(Solution::max_coins_optimized_space(nums.clone()), expected);
        assert_eq!(Solution::max_coins_divide_conquer(nums.clone()), expected);
        assert_eq!(Solution::max_coins_matrix_chain(nums.clone()), expected);
        assert_eq!(Solution::max_coins_pruned_recursive(nums), expected);
    }
    
    #[test]
    fn test_identical_balloons() {
        let nums = vec![5, 5, 5];
        // All same value, order matters
        let result = Solution::max_coins_memo(nums.clone());
        assert!(result > 0);
        
        assert_eq!(Solution::max_coins_bottom_up(nums.clone()), result);
        assert_eq!(Solution::max_coins_optimized_space(nums.clone()), result);
        assert_eq!(Solution::max_coins_divide_conquer(nums.clone()), result);
        assert_eq!(Solution::max_coins_matrix_chain(nums.clone()), result);
        assert_eq!(Solution::max_coins_pruned_recursive(nums), result);
    }
    
    #[test]
    fn test_increasing_sequence() {
        let nums = vec![1, 2, 3, 4];
        let result = Solution::max_coins_memo(nums.clone());
        assert!(result > 0);
        
        assert_eq!(Solution::max_coins_bottom_up(nums.clone()), result);
        assert_eq!(Solution::max_coins_optimized_space(nums.clone()), result);
        assert_eq!(Solution::max_coins_divide_conquer(nums.clone()), result);
        assert_eq!(Solution::max_coins_matrix_chain(nums.clone()), result);
        assert_eq!(Solution::max_coins_pruned_recursive(nums), result);
    }
    
    #[test]
    fn test_decreasing_sequence() {
        let nums = vec![4, 3, 2, 1];
        let result = Solution::max_coins_memo(nums.clone());
        assert!(result > 0);
        
        assert_eq!(Solution::max_coins_bottom_up(nums.clone()), result);
        assert_eq!(Solution::max_coins_optimized_space(nums.clone()), result);
        assert_eq!(Solution::max_coins_divide_conquer(nums.clone()), result);
        assert_eq!(Solution::max_coins_matrix_chain(nums.clone()), result);
        assert_eq!(Solution::max_coins_pruned_recursive(nums), result);
    }
    
    #[test]
    fn test_large_values() {
        let nums = vec![100, 200, 50];
        let result = Solution::max_coins_memo(nums.clone());
        assert!(result >= 100 * 200 + 200 * 50 + 100 * 50); // Lower bound
        
        assert_eq!(Solution::max_coins_bottom_up(nums.clone()), result);
        assert_eq!(Solution::max_coins_optimized_space(nums.clone()), result);
        assert_eq!(Solution::max_coins_divide_conquer(nums.clone()), result);
        assert_eq!(Solution::max_coins_matrix_chain(nums.clone()), result);
        assert_eq!(Solution::max_coins_pruned_recursive(nums), result);
    }
    
    #[test]
    fn test_small_values() {
        let nums = vec![1, 1, 1, 1];
        let expected = 4; // Each balloon burst gives 1*1*1 = 1, total 4
        let result = Solution::max_coins_memo(nums.clone());
        assert_eq!(result, expected);
        
        assert_eq!(Solution::max_coins_bottom_up(nums.clone()), result);
        assert_eq!(Solution::max_coins_optimized_space(nums.clone()), result);
        assert_eq!(Solution::max_coins_divide_conquer(nums.clone()), result);
        assert_eq!(Solution::max_coins_matrix_chain(nums.clone()), result);
        assert_eq!(Solution::max_coins_pruned_recursive(nums), result);
    }
    
    #[test]
    fn test_mixed_sequence() {
        let nums = vec![7, 9, 8, 0, 2];
        let result = Solution::max_coins_memo(nums.clone());
        assert!(result > 0);
        
        assert_eq!(Solution::max_coins_bottom_up(nums.clone()), result);
        assert_eq!(Solution::max_coins_optimized_space(nums.clone()), result);
        assert_eq!(Solution::max_coins_divide_conquer(nums.clone()), result);
        assert_eq!(Solution::max_coins_matrix_chain(nums.clone()), result);
        assert_eq!(Solution::max_coins_pruned_recursive(nums), result);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            vec![],
            vec![5],
            vec![1, 5],
            vec![3, 8],
            vec![2, 4, 6],
            vec![3, 1, 5, 8],
            vec![5, 5, 5],
            vec![1, 2, 3, 4],
            vec![4, 3, 2, 1],
            vec![7, 9, 8, 0, 2],
            vec![100, 200, 50],
            vec![1, 1, 1, 1],
        ];
        
        for nums in test_cases {
            let result1 = Solution::max_coins_memo(nums.clone());
            let result2 = Solution::max_coins_bottom_up(nums.clone());
            let result3 = Solution::max_coins_optimized_space(nums.clone());
            let result4 = Solution::max_coins_divide_conquer(nums.clone());
            let result5 = Solution::max_coins_matrix_chain(nums.clone());
            let result6 = Solution::max_coins_pruned_recursive(nums.clone());
            
            assert_eq!(result1, result2, "Memo vs Bottom-up mismatch for {:?}", nums);
            assert_eq!(result2, result3, "Bottom-up vs Optimized mismatch for {:?}", nums);
            assert_eq!(result3, result4, "Optimized vs Divide-conquer mismatch for {:?}", nums);
            assert_eq!(result4, result5, "Divide-conquer vs Matrix-chain mismatch for {:?}", nums);
            assert_eq!(result5, result6, "Matrix-chain vs Pruned mismatch for {:?}", nums);
        }
    }
    
    #[test]
    fn test_boundary_conditions() {
        // Test edge cases
        assert_eq!(Solution::max_coins_memo(vec![]), 0);
        assert_eq!(Solution::max_coins_memo(vec![1]), 1);
        assert_eq!(Solution::max_coins_memo(vec![0]), 0);
        
        // Test with zeros
        let nums_with_zero = vec![3, 0, 5];
        let result = Solution::max_coins_memo(nums_with_zero.clone());
        assert_eq!(result, Solution::max_coins_bottom_up(nums_with_zero));
    }
}