//! # Problem 268: Missing Number
//!
//! Given an array `nums` containing `n` distinct numbers from the range `[0, n]`,
//! return the only number in that range that is missing from the array.
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::easy::missing_number::Solution;
//!
//! let solution = Solution::new();
//!
//! assert_eq!(solution.missing_number_xor(vec![3, 0, 1]), 2);
//! assert_eq!(solution.missing_number_sum(vec![0, 1]), 2);
//! ```
//!
//! ## Constraints
//!
//! - `n == nums.len()`
//! - `1 <= n <= 10^5`
//! - `0 <= nums[i] <= n`
//! - All the numbers of `nums` are unique.

/// Solution struct following LeetCode conventions
pub struct Solution;

impl Solution {
    /// Create a new instance of `Solution`
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: XOR Accumulation
    ///
    /// XOR all indices and values together. All numbers cancel out except the
    /// missing one, leaving it as the result.
    ///
    /// **Time Complexity:** `O(n)`
    /// **Space Complexity:** `O(1)`
    pub fn missing_number_xor(&self, nums: Vec<i32>) -> i32 {
        let mut xor = 0;
        for (i, &num) in nums.iter().enumerate() {
            xor ^= i as i32 ^ num;
        }
        xor ^ nums.len() as i32
    }

    /// # Approach 2: Sum Formula
    ///
    /// Compute the expected sum of `0..=n` and subtract the actual sum of the
    /// array.
    ///
    /// **Time Complexity:** `O(n)`
    /// **Space Complexity:** `O(1)`
    pub fn missing_number_sum(&self, nums: Vec<i32>) -> i32 {
        let n = nums.len() as i32;
        let expected = n * (n + 1) / 2;
        let actual: i32 = nums.iter().sum();
        expected - actual
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn setup() -> Solution {
        Solution::new()
    }

    #[rstest]
    #[case(vec![3, 0, 1], 2)]
    #[case(vec![0, 1], 2)]
    #[case(vec![9, 6, 4, 2, 3, 5, 7, 0, 1], 8)]
    #[case(vec![0], 1)]
    #[case(vec![1], 0)]
    fn test_missing_number_xor(#[case] nums: Vec<i32>, #[case] expected: i32) {
        let solution = setup();
        assert_eq!(solution.missing_number_xor(nums), expected);
    }

    #[rstest]
    #[case(vec![3, 0, 1], 2)]
    #[case(vec![0, 1], 2)]
    #[case(vec![9, 6, 4, 2, 3, 5, 7, 0, 1], 8)]
    #[case(vec![0], 1)]
    #[case(vec![1], 0)]
    fn test_missing_number_sum(#[case] nums: Vec<i32>, #[case] expected: i32) {
        let solution = setup();
        assert_eq!(solution.missing_number_sum(nums), expected);
    }

    #[rstest]
    #[case(vec![3, 0, 1])]
    #[case(vec![0, 1])]
    #[case(vec![9, 6, 4, 2, 3, 5, 7, 0, 1])]
    #[case(vec![0])]
    #[case(vec![1])]
    fn test_consistency(#[case] nums: Vec<i32>) {
        let solution = setup();
        assert_eq!(
            solution.missing_number_xor(nums.clone()),
            solution.missing_number_sum(nums),
        );
    }
}
