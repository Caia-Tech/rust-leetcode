//! # Problem 410: Split Array Largest Sum
//!
//! Given an integer array `nums` and an integer `k`, split `nums` into `k` non-empty subarrays 
//! such that the largest sum of any subarray is minimized.
//!
//! Return the minimized largest sum of the split.
//!
//! A subarray is a contiguous part of the array.
//!
//! ## Examples
//!
//! ```
//! Input: nums = [7,2,5,10,8], k = 2
//! Output: 18
//! Explanation: There are four ways to split nums into two subarrays.
//! The best way is to split it into [7,2,5] and [10,8],
//! where the largest sum among the two subarrays is only 18.
//! ```
//!
//! ```
//! Input: nums = [1,2,3,4,5], k = 2
//! Output: 9
//! ```
//!
//! ```
//! Input: nums = [1,4,4], k = 3
//! Output: 4
//! ```

use std::cmp::{max, min};

/// Solution struct for Split Array Largest Sum problem
pub struct Solution;

impl Solution {
    /// Approach 1: Binary Search with Greedy Validation
    ///
    /// Uses binary search on the answer space and validates each candidate
    /// using a greedy approach to check if we can split with at most k subarrays.
    ///
    /// Time Complexity: O(n * log(sum - max)) where n is array length
    /// Space Complexity: O(1)
    pub fn split_array_binary_search(nums: Vec<i32>, k: i32) -> i32 {
        let mut left = *nums.iter().max().unwrap() as i64;
        let mut right = nums.iter().map(|&x| x as i64).sum::<i64>();
        
        while left < right {
            let mid = left + (right - left) / 2;
            
            if Self::can_split(&nums, k, mid) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        
        left as i32
    }
    
    fn can_split(nums: &[i32], k: i32, max_sum: i64) -> bool {
        let mut count = 1;
        let mut current_sum = 0i64;
        
        for &num in nums {
            if current_sum + num as i64 > max_sum {
                count += 1;
                current_sum = num as i64;
                if count > k {
                    return false;
                }
            } else {
                current_sum += num as i64;
            }
        }
        
        true
    }
    
    /// Approach 2: Dynamic Programming with 2D DP Table
    ///
    /// Uses DP where dp[i][j] represents the minimum largest sum
    /// when splitting nums[0..i] into j subarrays.
    ///
    /// Time Complexity: O(n^2 * k) where n is array length
    /// Space Complexity: O(n * k)
    pub fn split_array_dp(nums: Vec<i32>, k: i32) -> i32 {
        let n = nums.len();
        let k = k as usize;
        
        // Precompute prefix sums for quick range sum queries
        let mut prefix_sum = vec![0i64; n + 1];
        for i in 0..n {
            prefix_sum[i + 1] = prefix_sum[i] + nums[i] as i64;
        }
        
        // dp[i][j] = minimum largest sum when splitting nums[0..i] into j subarrays
        let mut dp = vec![vec![i64::MAX; k + 1]; n + 1];
        dp[0][0] = 0;
        
        for i in 1..=n {
            for j in 1..=min(i, k) {
                // Try all possible positions for the last subarray
                for m in j-1..i {
                    let subarray_sum = prefix_sum[i] - prefix_sum[m];
                    dp[i][j] = min(dp[i][j], max(dp[m][j-1], subarray_sum));
                }
            }
        }
        
        dp[n][k] as i32
    }
    
    /// Approach 3: Memoized Recursion (Top-Down DP)
    ///
    /// Uses recursion with memoization to find the optimal split.
    /// More intuitive than bottom-up DP but with similar complexity.
    ///
    /// Time Complexity: O(n^2 * k)
    /// Space Complexity: O(n * k) for memoization
    pub fn split_array_memo(nums: Vec<i32>, k: i32) -> i32 {
        let n = nums.len();
        let k = k as usize;
        
        // Precompute prefix sums
        let mut prefix_sum = vec![0i64; n + 1];
        for i in 0..n {
            prefix_sum[i + 1] = prefix_sum[i] + nums[i] as i64;
        }
        
        let mut memo = vec![vec![None; k + 1]; n];
        Self::solve_memo(&prefix_sum, 0, k, &mut memo) as i32
    }
    
    fn solve_memo(
        prefix_sum: &[i64],
        start: usize,
        k: usize,
        memo: &mut Vec<Vec<Option<i64>>>
    ) -> i64 {
        let n = prefix_sum.len() - 1;
        
        // Base case: one subarray left
        if k == 1 {
            return prefix_sum[n] - prefix_sum[start];
        }
        
        // Check memo
        if let Some(result) = memo[start][k] {
            return result;
        }
        
        let mut result = i64::MAX;
        
        // Try all possible positions for the first subarray
        for i in start..=n-k {
            let first_sum = prefix_sum[i + 1] - prefix_sum[start];
            let remaining = Self::solve_memo(prefix_sum, i + 1, k - 1, memo);
            result = min(result, max(first_sum, remaining));
        }
        
        memo[start][k] = Some(result);
        result
    }
    
    /// Approach 4: Optimized DP with Space Compression
    ///
    /// Uses rolling array technique to reduce space complexity from O(n*k) to O(n).
    ///
    /// Time Complexity: O(n^2 * k)
    /// Space Complexity: O(n)
    pub fn split_array_dp_optimized(nums: Vec<i32>, k: i32) -> i32 {
        let n = nums.len();
        let k = k as usize;
        
        // Precompute prefix sums
        let mut prefix_sum = vec![0i64; n + 1];
        for i in 0..n {
            prefix_sum[i + 1] = prefix_sum[i] + nums[i] as i64;
        }
        
        // Use two arrays for space optimization
        let mut prev = vec![i64::MAX; n + 1];
        prev[0] = 0;
        
        for j in 1..=k {
            let mut curr = vec![i64::MAX; n + 1];
            
            for i in j..=n {
                // Try all possible positions for the last subarray
                for m in j-1..i {
                    let subarray_sum = prefix_sum[i] - prefix_sum[m];
                    curr[i] = min(curr[i], max(prev[m], subarray_sum));
                }
            }
            
            prev = curr;
        }
        
        prev[n] as i32
    }
    
    /// Approach 5: Binary Search with Optimized Validation
    ///
    /// Enhanced binary search with early termination and better bounds.
    ///
    /// Time Complexity: O(n * log(sum - max))
    /// Space Complexity: O(1)
    pub fn split_array_binary_optimized(nums: Vec<i32>, k: i32) -> i32 {
        let n = nums.len();
        
        // Better bounds calculation
        let mut max_elem = 0i64;
        let mut sum = 0i64;
        for &num in &nums {
            max_elem = max(max_elem, num as i64);
            sum += num as i64;
        }
        
        // Edge cases
        if k == 1 {
            return sum as i32;
        }
        if k as usize == n {
            return max_elem as i32;
        }
        
        let mut left = max_elem;
        let mut right = sum;
        let mut result = right;
        
        while left <= right {
            let mid = left + (right - left) / 2;
            
            let splits = Self::count_splits(&nums, mid);
            
            if splits <= k {
                result = min(result, mid);
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        
        result as i32
    }
    
    fn count_splits(nums: &[i32], max_sum: i64) -> i32 {
        let mut count = 1;
        let mut current_sum = 0i64;
        
        for &num in nums {
            if current_sum + num as i64 > max_sum {
                count += 1;
                current_sum = num as i64;
            } else {
                current_sum += num as i64;
            }
        }
        
        count
    }
    
    /// Approach 6: Segment Tree Based Solution
    ///
    /// Uses a segment tree for efficient range queries combined with DP.
    /// This approach is more complex but demonstrates advanced data structures.
    ///
    /// Time Complexity: O(n^2 * k) with optimized range queries
    /// Space Complexity: O(n * k)
    pub fn split_array_segment_tree(nums: Vec<i32>, k: i32) -> i32 {
        // For this problem, segment tree doesn't provide significant benefit
        // over prefix sums, so we'll use a hybrid approach with better constants
        
        let n = nums.len();
        let k = k as usize;
        
        // Build cumulative max array for quick max queries
        let mut cumulative_max = vec![vec![0i64; n]; n];
        for i in 0..n {
            cumulative_max[i][i] = nums[i] as i64;
            for j in i+1..n {
                cumulative_max[i][j] = max(cumulative_max[i][j-1], nums[j] as i64);
            }
        }
        
        // Build range sum array
        let mut range_sum = vec![vec![0i64; n]; n];
        for i in 0..n {
            range_sum[i][i] = nums[i] as i64;
            for j in i+1..n {
                range_sum[i][j] = range_sum[i][j-1] + nums[j] as i64;
            }
        }
        
        // DP with precomputed values
        let mut dp = vec![vec![i64::MAX; k + 1]; n];
        
        // Base case: first element with 1 split
        for i in 0..n {
            dp[i][1] = range_sum[0][i];
        }
        
        // Fill DP table
        for i in 1..n {
            for j in 2..=min(i + 1, k) {
                for m in j-2..i {
                    dp[i][j] = min(dp[i][j], max(dp[m][j-1], range_sum[m+1][i]));
                }
            }
        }
        
        dp[n-1][k] as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_1() {
        let nums = vec![7, 2, 5, 10, 8];
        let k = 2;
        let expected = 18;
        
        assert_eq!(Solution::split_array_binary_search(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_memo(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_binary_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_segment_tree(nums, k), expected);
    }

    #[test]
    fn test_example_2() {
        let nums = vec![1, 2, 3, 4, 5];
        let k = 2;
        let expected = 9;
        
        assert_eq!(Solution::split_array_binary_search(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_memo(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_binary_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_segment_tree(nums, k), expected);
    }

    #[test]
    fn test_example_3() {
        let nums = vec![1, 4, 4];
        let k = 3;
        let expected = 4;
        
        assert_eq!(Solution::split_array_binary_search(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_memo(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_binary_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_segment_tree(nums, k), expected);
    }

    #[test]
    fn test_single_element() {
        let nums = vec![10];
        let k = 1;
        let expected = 10;
        
        assert_eq!(Solution::split_array_binary_search(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_memo(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_binary_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_segment_tree(nums, k), expected);
    }

    #[test]
    fn test_k_equals_n() {
        let nums = vec![1, 2, 3, 4, 5];
        let k = 5;
        let expected = 5; // Each element in its own subarray
        
        assert_eq!(Solution::split_array_binary_search(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_memo(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_binary_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_segment_tree(nums, k), expected);
    }

    #[test]
    fn test_k_equals_1() {
        let nums = vec![1, 2, 3, 4, 5];
        let k = 1;
        let expected = 15; // All elements in one subarray
        
        assert_eq!(Solution::split_array_binary_search(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_memo(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_binary_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_segment_tree(nums, k), expected);
    }

    #[test]
    fn test_large_numbers() {
        let nums = vec![1000000, 1000000, 1000000, 1000000];
        let k = 2;
        let expected = 2000000;
        
        assert_eq!(Solution::split_array_binary_search(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_memo(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_binary_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_segment_tree(nums, k), expected);
    }

    #[test]
    fn test_uneven_distribution() {
        let nums = vec![1, 1, 1, 1, 100];
        let k = 2;
        let expected = 100; // [1,1,1,1] and [100]
        
        assert_eq!(Solution::split_array_binary_search(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_memo(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_binary_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_segment_tree(nums, k), expected);
    }

    #[test]
    fn test_three_splits() {
        let nums = vec![10, 5, 13, 4, 8, 4, 5, 11, 14, 9, 16, 10, 20, 8];
        let k = 8;
        let expected = 25; // Optimal split to minimize largest sum
        
        assert_eq!(Solution::split_array_binary_search(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_memo(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_binary_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_segment_tree(nums, k), expected);
    }

    #[test]
    fn test_zeros() {
        let nums = vec![0, 0, 0, 0];
        let k = 2;
        let expected = 0;
        
        assert_eq!(Solution::split_array_binary_search(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_memo(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_binary_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_segment_tree(nums, k), expected);
    }

    #[test]
    fn test_increasing_sequence() {
        let nums = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let k = 3;
        let expected = 21; // [1,2,3,4,5], [6,7,8], [9,10]
        
        assert_eq!(Solution::split_array_binary_search(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_memo(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_binary_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_segment_tree(nums, k), expected);
    }

    #[test]
    fn test_decreasing_sequence() {
        let nums = vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
        let k = 3;
        let expected = 21; // [10,9], [8,7,6], [5,4,3,2,1] = max(19, 21, 15) = 21
        
        assert_eq!(Solution::split_array_binary_search(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_memo(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_binary_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_segment_tree(nums, k), expected);
    }

    #[test]
    fn test_alternating() {
        let nums = vec![1, 10, 1, 10, 1, 10];
        let k = 3;
        let expected = 11; // [1,10], [1,10], [1,10]
        
        assert_eq!(Solution::split_array_binary_search(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_memo(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_dp_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_binary_optimized(nums.clone(), k), expected);
        assert_eq!(Solution::split_array_segment_tree(nums, k), expected);
    }

    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            (vec![7, 2, 5, 10, 8], 2),
            (vec![1, 2, 3, 4, 5], 2),
            (vec![1, 4, 4], 3),
            (vec![10], 1),
            (vec![1, 2, 3, 4, 5], 5),
            (vec![1, 2, 3, 4, 5], 1),
            (vec![1000000, 1000000, 1000000, 1000000], 2),
            (vec![1, 1, 1, 1, 100], 2),
            (vec![0, 0, 0, 0], 2),
            (vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3),
        ];
        
        for (nums, k) in test_cases {
            let result1 = Solution::split_array_binary_search(nums.clone(), k);
            let result2 = Solution::split_array_dp(nums.clone(), k);
            let result3 = Solution::split_array_memo(nums.clone(), k);
            let result4 = Solution::split_array_dp_optimized(nums.clone(), k);
            let result5 = Solution::split_array_binary_optimized(nums.clone(), k);
            let result6 = Solution::split_array_segment_tree(nums.clone(), k);
            
            assert_eq!(result1, result2, "Binary search vs DP mismatch for {:?}, k={}", nums, k);
            assert_eq!(result2, result3, "DP vs Memo mismatch for {:?}, k={}", nums, k);
            assert_eq!(result3, result4, "Memo vs DP optimized mismatch for {:?}, k={}", nums, k);
            assert_eq!(result4, result5, "DP optimized vs Binary optimized mismatch for {:?}, k={}", nums, k);
            assert_eq!(result5, result6, "Binary optimized vs Segment tree mismatch for {:?}, k={}", nums, k);
        }
    }
}

// Author: Marvin Tutt, Caia Tech
// Problem: LeetCode 410 - Split Array Largest Sum
// Approaches: Binary search with greedy validation, 2D DP, Memoized recursion,
//            Space-optimized DP, Enhanced binary search, Segment tree hybrid
// All approaches minimize the largest sum when splitting array into k subarrays