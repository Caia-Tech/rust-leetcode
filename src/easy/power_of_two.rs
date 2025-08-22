//! # Problem 231: Power of Two
//!
//! Determine whether an integer `n` is a power of two.
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::easy::power_of_two::Solution;
//!
//! let solution = Solution::new();
//! assert!(solution.is_power_of_two_iterative(16));
//! assert!(solution.is_power_of_two_bitwise(1));
//! assert!(!solution.is_power_of_two_iterative(3));
//! ```
//!
//! ## Constraints
//!
//! - `-2^31 <= n <= 2^31 - 1`
//!
/// Solution struct following LeetCode conventions
pub struct Solution;

impl Solution {
    /// Create a new instance of `Solution`
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Iterative Division
    ///
    /// Repeatedly divide `n` by 2 while it's even. If we reach 1, it's a power of two.
    ///
    /// **Time Complexity:** `O(log n)` where `n` is the value.
    /// **Space Complexity:** `O(1)`
    pub fn is_power_of_two_iterative(&self, mut n: i32) -> bool {
        if n <= 0 {
            return false;
        }
        while n % 2 == 0 {
            n /= 2;
        }
        n == 1
    }

    /// # Approach 2: Bitwise Trick
    ///
    /// A power of two has exactly one bit set. For `n > 0`, `n & (n - 1) == 0`.
    ///
    /// **Time Complexity:** `O(1)`
    /// **Space Complexity:** `O(1)`
    pub fn is_power_of_two_bitwise(&self, n: i32) -> bool {
        n > 0 && (n & (n - 1)) == 0
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
    #[case(2, true)]
    #[case(3, false)]
    #[case(16, true)]
    #[case(218, false)]
    #[case(0, false)]
    #[case(-8, false)]
    fn test_is_power_of_two_iterative(#[case] input: i32, #[case] expected: bool) {
        let solution = setup();
        assert_eq!(solution.is_power_of_two_iterative(input), expected);
    }

    #[rstest]
    #[case(1, true)]
    #[case(2, true)]
    #[case(3, false)]
    #[case(16, true)]
    #[case(218, false)]
    #[case(0, false)]
    #[case(-8, false)]
    fn test_is_power_of_two_bitwise(#[case] input: i32, #[case] expected: bool) {
        let solution = setup();
        assert_eq!(solution.is_power_of_two_bitwise(input), expected);
    }

    #[test]
    fn test_consistency_between_methods() {
        let solution = setup();
        for n in [-16, -1, 0, 1, 2, 3, 4, 8, 16, 31, 64, i32::MAX] {
            assert_eq!(
                solution.is_power_of_two_iterative(n),
                solution.is_power_of_two_bitwise(n)
            );
        }
    }
}
