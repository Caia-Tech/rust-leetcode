//! # Problem 258: Add Digits
//!
//! Given an integer `num`, repeatedly add all its digits until the result has only one digit, and return it.
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::easy::add_digits::Solution;
//!
//! let solution = Solution::new();
//!
//! assert_eq!(solution.add_digits_iterative(38), 2); // 3 + 8 = 11, 1 + 1 = 2
//! assert_eq!(solution.add_digits_math(38), 2);
//! ```
//!
//! ## Constraints
//!
//! - `0 <= num <= i32::MAX`
//!
/// Solution struct following LeetCode conventions
pub struct Solution;

impl Solution {
    /// Create a new instance of `Solution`
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Iterative Digit Summation
    ///
    /// Repeatedly sums the digits of `num` until a single digit remains.
    ///
    /// **Algorithm**
    /// 1. While `num` has more than one digit, compute the sum of its digits.
    /// 2. Replace `num` with this sum and repeat.
    ///
    /// **Time Complexity:** `O(k)` per iteration where `k` is number of digits. In worst case, `O(log n)` overall.
    /// **Space Complexity:** `O(1)`
    pub fn add_digits_iterative(&self, mut num: i32) -> i32 {
        while num >= 10 {
            let mut sum = 0;
            let mut n = num;
            while n > 0 {
                sum += n % 10;
                n /= 10;
            }
            num = sum;
        }
        num
    }

    /// # Approach 2: Digital Root Formula
    ///
    /// Uses mathematical property of digital roots: the result is `1 + (num - 1) % 9`
    /// for positive `num`, and `0` when `num` is `0`.
    ///
    /// **Time Complexity:** `O(1)`
    /// **Space Complexity:** `O(1)`
    pub fn add_digits_math(&self, num: i32) -> i32 {
        if num == 0 {
            0
        } else {
            1 + (num - 1) % 9
        }
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
    #[case(0, 0)]
    #[case(5, 5)]
    #[case(9, 9)]
    #[case(10, 1)]
    #[case(38, 2)]
    #[case(12345, 6)] // 1+2+3+4+5 = 15 -> 1+5 = 6
    #[case(99999, 9)]
    fn test_add_digits_iterative(#[case] input: i32, #[case] expected: i32) {
        let solution = setup();
        assert_eq!(solution.add_digits_iterative(input), expected);
    }

    #[rstest]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(38, 2)]
    #[case(12345, 6)]
    #[case(99999, 9)]
    fn test_add_digits_math(#[case] input: i32, #[case] expected: i32) {
        let solution = setup();
        assert_eq!(solution.add_digits_math(input), expected);
    }

    #[test]
    fn test_consistency_between_methods() {
        let solution = setup();
        for num in [0, 1, 2, 9, 10, 11, 38, 99, 12345, i32::MAX] {
            assert_eq!(
                solution.add_digits_iterative(num),
                solution.add_digits_math(num)
            );
        }
    }
}
