//! # Problem 217: Contains Duplicate
//!
//! Given an integer array `nums`, return `true` if any value appears at least twice
//! in the array, and return `false` if every element is distinct.
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::easy::contains_duplicate::Solution;
//!
//! let solution = Solution::new();
//!
//! assert!(solution.contains_duplicate_set(vec![1, 2, 3, 1]));
//! assert!(!solution.contains_duplicate_sort(vec![1, 2, 3, 4]));
//! ```
//!
//! ## Constraints
//!
//! - `1 <= nums.len() <= 10^5`
//! - `-10^9 <= nums[i] <= 10^9`
//!
/// Solution struct following LeetCode conventions
pub struct Solution;

impl Solution {
    /// Create a new instance of `Solution`
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Hash Set Tracking
    ///
    /// Inserts elements into a `HashSet` and checks for duplicates as we iterate.
    ///
    /// **Time Complexity:** `O(n)` on average, where `n` is the length of `nums`.
    /// **Space Complexity:** `O(n)` for the set.
    pub fn contains_duplicate_set(&self, nums: Vec<i32>) -> bool {
        use std::collections::HashSet;
        let mut seen = HashSet::with_capacity(nums.len());
        for &num in &nums {
            if !seen.insert(num) {
                return true;
            }
        }
        false
    }

    /// # Approach 2: Sorting
    ///
    /// Sorts the array and then checks adjacent elements for equality.
    ///
    /// **Time Complexity:** `O(n \log n)` due to sorting.
    /// **Space Complexity:** `O(1)` or `O(n)` depending on sort implementation.
    pub fn contains_duplicate_sort(&self, mut nums: Vec<i32>) -> bool {
        nums.sort_unstable();
        nums.windows(2).any(|w| w[0] == w[1])
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
    #[case(vec![1, 2, 3, 1], true)]
    #[case(vec![1, 2, 3, 4], false)]
    #[case(vec![1, 1, 1, 3, 3, 4, 3, 2, 4, 2], true)]
    #[case(vec![0; 1], false)]
    #[case(vec![0, 0], true)]
    fn test_contains_duplicate_set(#[case] nums: Vec<i32>, #[case] expected: bool) {
        let solution = setup();
        assert_eq!(solution.contains_duplicate_set(nums), expected);
    }

    #[rstest]
    #[case(vec![1, 2, 3, 1], true)]
    #[case(vec![1, 2, 3, 4], false)]
    #[case(vec![1, 1, 1, 3, 3, 4, 3, 2, 4, 2], true)]
    #[case(vec![0; 1], false)]
    #[case(vec![0, 0], true)]
    fn test_contains_duplicate_sort(#[case] nums: Vec<i32>, #[case] expected: bool) {
        let solution = setup();
        assert_eq!(solution.contains_duplicate_sort(nums), expected);
    }

    #[test]
    fn test_consistency_between_methods() {
        let solution = setup();
        let cases = vec![
            vec![1, 2, 3, 1],
            vec![1, 2, 3, 4],
            vec![1, 1, 1, 3, 3, 4, 3, 2, 4, 2],
            vec![0; 1],
            vec![0, 0],
        ];
        for nums in cases {
            assert_eq!(
                solution.contains_duplicate_set(nums.clone()),
                solution.contains_duplicate_sort(nums)
            );
        }
    }
}
