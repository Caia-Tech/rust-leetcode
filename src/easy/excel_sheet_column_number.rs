//! # Problem 171: Excel Sheet Column Number
//!
//! Given a column title as it appears in an Excel sheet, return its corresponding column number.
//!
//! Each letter represents a digit in base-26 where `A = 1`, `B = 2`, ..., `Z = 26`.
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::easy::excel_sheet_column_number::Solution;
//!
//! let solution = Solution::new();
//! assert_eq!(solution.title_to_number_iterative("A".to_string()), 1);
//! assert_eq!(solution.title_to_number_iterative("AB".to_string()), 28);
//! assert_eq!(solution.title_to_number_fold("ZY".to_string()), 701);
//! ```
//!
//! ## Constraints
//!
//! - `1 <= column_title.len() <= 7`
//! - `column_title` consists only of uppercase English letters
//!
/// Solution struct following LeetCode conventions
pub struct Solution;

impl Solution {
    /// Create a new instance of `Solution`
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Iterative Base-26 Conversion
    ///
    /// Processes each character from left to right, treating the string as a
    /// number in base-26 where `A` corresponds to 1.
    ///
    /// **Algorithm**
    /// 1. Start with `result = 0`.
    /// 2. For each character `c` in the title:
    ///    - Multiply `result` by 26.
    ///    - Add the value of `c` (`c - 'A' + 1`).
    /// 3. Return `result`.
    ///
    /// **Time Complexity:** `O(n)`
    /// **Space Complexity:** `O(1)`
    pub fn title_to_number_iterative(&self, column_title: String) -> i32 {
        let mut result = 0i32;
        for c in column_title.chars() {
            result = result * 26 + ((c as u8 - b'A' + 1) as i32);
        }
        result
    }

    /// # Approach 2: Iterator `fold`
    ///
    /// Utilizes Rust's iterator `fold` to accumulate the base-26 value.
    ///
    /// **Time Complexity:** `O(n)`
    /// **Space Complexity:** `O(1)`
    pub fn title_to_number_fold(&self, column_title: String) -> i32 {
        column_title
            .bytes()
            .fold(0, |acc, b| acc * 26 + (b - b'A' + 1) as i32)
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
    #[case("A", 1)]
    #[case("AB", 28)]
    #[case("ZY", 701)]
    #[case("FXSHRXW", 2147483647)]
    fn test_title_to_number_iterative(#[case] input: &str, #[case] expected: i32) {
        let solution = setup();
        assert_eq!(solution.title_to_number_iterative(input.to_string()), expected);
    }

    #[rstest]
    #[case("A", 1)]
    #[case("AB", 28)]
    #[case("ZY", 701)]
    #[case("FXSHRXW", 2147483647)]
    fn test_title_to_number_fold(#[case] input: &str, #[case] expected: i32) {
        let solution = setup();
        assert_eq!(solution.title_to_number_fold(input.to_string()), expected);
    }

    #[test]
    fn test_consistency_between_methods() {
        let solution = setup();
        for title in ["A", "Z", "AA", "AZ", "BA", "ZY", "AAA", "FXSHRXW"] {
            assert_eq!(
                solution.title_to_number_iterative(title.to_string()),
                solution.title_to_number_fold(title.to_string())
            );
        }
    }
}

