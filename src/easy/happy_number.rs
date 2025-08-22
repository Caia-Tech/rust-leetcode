//! # Problem 202: Happy Number
//!
//! Write an algorithm to determine if a number `n` is happy. A happy number is a number defined by
//! the following process:
//!
//! 1. Starting with any positive integer, replace the number by the sum of the squares of its digits.
//! 2. Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle that does not include 1.
//! 3. Those numbers for which this process ends in 1 are happy.
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::easy::happy_number::Solution;
//!
//! let solution = Solution::new();
//!
//! assert!(solution.is_happy_hashset(19));
//! assert!(solution.is_happy_floyd(19));
//! assert!(!solution.is_happy_hashset(2));
//! ```
//!
//! ## Constraints
//!
//! - `1 <= n <= i32::MAX`
//!
/// Solution struct following LeetCode conventions.
pub struct Solution;

impl Solution {
    /// Create a new instance of `Solution`.
    pub fn new() -> Self {
        Solution
    }

    /// Helper to compute the sum of the squares of the digits of `n`.
    fn sum_of_squares(mut n: i32) -> i32 {
        let mut sum = 0;
        while n > 0 {
            let digit = n % 10;
            sum += digit * digit;
            n /= 10;
        }
        sum
    }

    /// # Approach 1: HashSet Cycle Detection
    ///
    /// Continually applies the digit-squared sum and tracks previously seen numbers to
    /// detect loops.
    ///
    /// **Time Complexity:** `O(k)` per iteration, where `k` is the number of digits. In the worst case,
    /// the sequence quickly converges and runs in near-constant time.
    /// **Space Complexity:** `O(m)` for the set of seen numbers.
    pub fn is_happy_hashset(&self, mut n: i32) -> bool {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        while n != 1 && seen.insert(n) {
            n = Self::sum_of_squares(n);
        }
        n == 1
    }

    /// # Approach 2: Floyd's Cycle Detection
    ///
    /// Uses tortoise and hare pointers to detect a cycle without extra space.
    ///
    /// **Time Complexity:** `O(k)` per iteration; converges quickly in practice.
    /// **Space Complexity:** `O(1)`
    pub fn is_happy_floyd(&self, mut n: i32) -> bool {
        let mut slow = n;
        let mut fast = Self::sum_of_squares(n);
        while fast != 1 && slow != fast {
            slow = Self::sum_of_squares(slow);
            fast = Self::sum_of_squares(Self::sum_of_squares(fast));
        }
        fast == 1
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
    #[case(1, true)]
    #[case(2, false)]
    #[case(7, true)]
    #[case(19, true)]
    #[case(20, false)]
    #[case(1111111, true)] // 7 digits of 1 -> sum is 7 -> happy
    fn test_is_happy_hashset(#[case] input: i32, #[case] expected: bool) {
        let solution = setup();
        assert_eq!(solution.is_happy_hashset(input), expected);
    }

    #[rstest]
    #[case(1, true)]
    #[case(2, false)]
    #[case(7, true)]
    #[case(19, true)]
    #[case(20, false)]
    #[case(1111111, true)]
    fn test_is_happy_floyd(#[case] input: i32, #[case] expected: bool) {
        let solution = setup();
        assert_eq!(solution.is_happy_floyd(input), expected);
    }

    #[test]
    fn test_methods_consistency() {
        let solution = setup();
        for n in 1..=1000 {
            assert_eq!(solution.is_happy_hashset(n), solution.is_happy_floyd(n));
        }
    }
}
