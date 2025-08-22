//! # Problem 263: Ugly Number
//!
//! An **ugly number** is a positive integer whose prime factors are limited to 2, 3, and 5.
//! Given an integer `n`, return `true` if `n` is an ugly number.
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::easy::ugly_number::Solution;
//!
//! let solution = Solution::new();
//! assert!(solution.is_ugly_iterative(6));   // 6 = 2 * 3
//! assert!(solution.is_ugly_recursive(8));   // 8 = 2^3
//! assert!(!solution.is_ugly_iterative(14)); // 14 has prime factor 7
//! ```
//!
//! ## Constraints
//!
//! - `-2^31 <= n <= 2^31 - 1`
//!
/// Solution struct following LeetCode conventions
#[derive(Default)]
pub struct Solution;

impl Solution {
    /// Create a new instance of `Solution`
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Iterative division
    ///
    /// Continuously divide `n` by 2, 3, and 5 while it is divisible by any of them.
    /// If the final result is 1, then `n` is an ugly number.
    ///
    /// **Time Complexity:** `O(log n)`
    /// **Space Complexity:** `O(1)`
    pub fn is_ugly_iterative(&self, mut n: i32) -> bool {
        if n <= 0 {
            return false;
        }
        for p in [2, 3, 5] {
            while n % p == 0 {
                n /= p;
            }
        }
        n == 1
    }

    /// # Approach 2: Recursive division
    ///
    /// Recursively divides `n` by 2, 3, or 5 when possible.
    /// The recursion bottoms out when `n` becomes 1 or no further division is possible.
    ///
    /// **Time Complexity:** `O(log n)`
    /// **Space Complexity:** `O(log n)` due to recursion depth
    pub fn is_ugly_recursive(&self, n: i32) -> bool {
        if n <= 0 {
            return false;
        }
        match n {
            1 => true,
            _ if n % 2 == 0 => self.is_ugly_recursive(n / 2),
            _ if n % 3 == 0 => self.is_ugly_recursive(n / 3),
            _ if n % 5 == 0 => self.is_ugly_recursive(n / 5),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Solution;
    use rstest::rstest;

    fn setup() -> Solution {
        Solution::new()
    }

    #[rstest]
    #[case(1, true)]
    #[case(6, true)]
    #[case(8, true)]
    #[case(14, false)]
    #[case(0, false)]
    #[case(-6, false)]
    fn test_is_ugly_iterative(#[case] input: i32, #[case] expected: bool) {
        let solution = setup();
        assert_eq!(solution.is_ugly_iterative(input), expected);
    }

    #[rstest]
    #[case(1, true)]
    #[case(6, true)]
    #[case(8, true)]
    #[case(14, false)]
    #[case(0, false)]
    #[case(-6, false)]
    fn test_is_ugly_recursive(#[case] input: i32, #[case] expected: bool) {
        let solution = setup();
        assert_eq!(solution.is_ugly_recursive(input), expected);
    }

    #[test]
    fn test_consistency_between_methods() {
        let solution = setup();
        for n in [-10, -1, 0, 1, 2, 3, 4, 5, 6, 8, 14, 30, 31, i32::MAX] {
            assert_eq!(solution.is_ugly_iterative(n), solution.is_ugly_recursive(n));
        }
    }
}
