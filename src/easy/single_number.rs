//! # Single Number
//!
//! Given a non-empty array of integers, every element appears twice except for one. Find that single one.
//!
//! We provide two implementations:
//! - [`xor`]: Uses bitwise XOR to cancel out pairs and isolate the unique element.
//! - [`hash_set`]: Tracks seen values in a [`HashSet`](std::collections::HashSet).
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::easy::single_number::Solution;
//!
//! let nums = vec![2, 2, 1];
//! assert_eq!(Solution::default().single_number_xor(nums), 1);
//! ```
//!
//! ```
//! use rust_leetcode::easy::single_number::Solution;
//!
//! let nums = vec![4,1,2,1,2];
//! assert_eq!(Solution::default().single_number_xor(nums), 4);
//! ```
//!
//! The XOR approach runs in `O(n)` time with `O(1)` space, while the hash-set variant uses `O(n)` space.
use std::collections::HashSet;

#[derive(Default)]
pub struct Solution;

impl Solution {
    /// Returns the single number using bitwise XOR.
    pub fn single_number_xor(&self, nums: Vec<i32>) -> i32 {
        nums.into_iter().fold(0, |acc, n| acc ^ n)
    }

    /// Returns the single number using a [`HashSet`] to track occurrences.
    pub fn single_number_hash_set(&self, nums: Vec<i32>) -> i32 {
        let mut set = HashSet::new();
        for n in nums {
            if !set.insert(n) {
                set.remove(&n);
            }
        }
        // Remaining element is the answer.
        *set.iter().next().expect("non-empty input")
    }
}

#[cfg(test)]
mod tests {
    use super::Solution;
    use rstest::rstest;

    #[rstest]
    #[case(vec![2,2,1], 1)]
    #[case(vec![4,1,2,1,2], 4)]
    #[case(vec![1], 1)]
    fn test_xor(#[case] nums: Vec<i32>, #[case] expected: i32) {
        let sol = Solution::default();
        assert_eq!(sol.single_number_xor(nums.clone()), expected);
        assert_eq!(sol.single_number_hash_set(nums), expected);
    }
}
