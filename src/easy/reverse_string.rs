//! # Problem 344: Reverse String
//!
//! Write a function that reverses a string in-place. The input is provided as a
//! mutable vector of characters `s` where each character represents a single
//! UTF-8 scalar value. The goal is to reverse the order of the characters using
//! `O(1)` extra space.
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::easy::reverse_string::Solution;
//!
//! let solution = Solution::new();
//! let mut s = vec!['h', 'e', 'l', 'l', 'o'];
//! solution.reverse_string_two_pointers(&mut s);
//! assert_eq!(s, vec!['o', 'l', 'l', 'e', 'h']);
//!
//! let mut t = vec!['H', 'a', 'n', 'n', 'a', 'h'];
//! solution.reverse_string_builtin(&mut t);
//! assert_eq!(t, vec!['h', 'a', 'n', 'n', 'a', 'H']);
//! ```
//!
//! ## Constraints
//!
//! - `1 <= s.len() <= 10^5`
//! - `s[i]` is any valid UTF-8 character.

/// Solution struct following LeetCode conventions
pub struct Solution;

impl Solution {
    /// Create a new instance of `Solution`
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Two-Pointer Swap
    ///
    /// Uses two indices that start at the ends of the vector and move toward
    /// the center, swapping characters at each step.
    ///
    /// **Time Complexity:** `O(n)`
    /// **Space Complexity:** `O(1)`
    pub fn reverse_string_two_pointers(&self, s: &mut Vec<char>) {
        let mut left = 0;
        let mut right = s.len().saturating_sub(1);
        while left < right {
            s.swap(left, right);
            left += 1;
            right -= 1;
        }
    }

    /// # Approach 2: Using `Vec::reverse`
    ///
    /// Leverages the standard library's [`Vec::reverse`], which performs the
    /// reversal in-place using an optimized implementation.
    ///
    /// **Time Complexity:** `O(n)`
    /// **Space Complexity:** `O(1)`
    pub fn reverse_string_builtin(&self, s: &mut Vec<char>) {
        s.reverse();
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
    #[case(vec!['h', 'e', 'l', 'l', 'o'], vec!['o', 'l', 'l', 'e', 'h'])]
    #[case(vec!['H', 'a', 'n', 'n', 'a', 'h'], vec!['h', 'a', 'n', 'n', 'a', 'H'])]
    #[case(Vec::<char>::new(), Vec::<char>::new())]
    #[case(vec!['a'], vec!['a'])]
    fn test_reverse_string_two_pointers(
        #[case] mut input: Vec<char>,
        #[case] expected: Vec<char>,
    ) {
        let solution = setup();
        solution.reverse_string_two_pointers(&mut input);
        assert_eq!(input, expected);
    }

    #[rstest]
    #[case(vec!['h', 'e', 'l', 'l', 'o'], vec!['o', 'l', 'l', 'e', 'h'])]
    #[case(vec!['H', 'a', 'n', 'n', 'a', 'h'], vec!['h', 'a', 'n', 'n', 'a', 'H'])]
    #[case(Vec::<char>::new(), Vec::<char>::new())]
    #[case(vec!['a'], vec!['a'])]
    fn test_reverse_string_builtin(
        #[case] mut input: Vec<char>,
        #[case] expected: Vec<char>,
    ) {
        let solution = setup();
        solution.reverse_string_builtin(&mut input);
        assert_eq!(input, expected);
    }

    #[test]
    fn test_methods_consistency() {
        let solution = setup();
        let cases = vec![
            vec!['h', 'e', 'l', 'l', 'o'],
            vec!['H', 'a', 'n', 'n', 'a', 'h'],
            vec!['a'],
            Vec::<char>::new(),
        ];
        for mut case in cases {
            let mut alt = case.clone();
            solution.reverse_string_two_pointers(&mut case);
            solution.reverse_string_builtin(&mut alt);
            assert_eq!(case, alt);
        }
    }
}

