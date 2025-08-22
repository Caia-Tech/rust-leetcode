//! # Problem 242: Valid Anagram
//!
//! Given two strings `s` and `t`, return `true` if `t` is an anagram of `s`,
//! and `false` otherwise. An *anagram* is a word or phrase formed by
//! rearranging the letters of a different word or phrase, typically using all
//! the original letters exactly once.
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::easy::valid_anagram::Solution;
//!
//! let solution = Solution::new();
//! assert!(solution.is_anagram_sort("anagram".into(), "nagaram".into()));
//! assert!(!solution.is_anagram_count("rat".into(), "car".into()));
//! ```
//!
//! ## Constraints
//!
//! - `1 <= s.len(), t.len() <= 5 * 10^4`
//! - `s` and `t` consist of lowercase English letters
//!
/// Solution struct following LeetCode conventions
pub struct Solution;

impl Solution {
    /// Create a new instance of `Solution`
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Sorting
    ///
    /// Sort both strings and compare them directly.
    ///
    /// **Time Complexity:** `O(n log n)` due to sorting.
    /// **Space Complexity:** `O(n)` to store the sorted character arrays.
    pub fn is_anagram_sort(&self, s: String, t: String) -> bool {
        let mut s_bytes: Vec<u8> = s.into_bytes();
        let mut t_bytes: Vec<u8> = t.into_bytes();
        s_bytes.sort_unstable();
        t_bytes.sort_unstable();
        s_bytes == t_bytes
    }

    /// # Approach 2: Character Counting
    ///
    /// Count occurrences of each character using a fixed-size array
    /// (since inputs are lowercase English letters).
    ///
    /// **Time Complexity:** `O(n)`
    /// **Space Complexity:** `O(1)`
    pub fn is_anagram_count(&self, s: String, t: String) -> bool {
        if s.len() != t.len() {
            return false;
        }
        let mut counts = [0i32; 26];
        for (a, b) in s.bytes().zip(t.bytes()) {
            counts[(a - b'a') as usize] += 1;
            counts[(b - b'a') as usize] -= 1;
        }
        counts.iter().all(|&c| c == 0)
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
    #[case("anagram", "nagaram", true)]
    #[case("rat", "car", false)]
    #[case("", "", true)]
    #[case("a", "ab", false)]
    fn test_is_anagram_sort(#[case] s: &str, #[case] t: &str, #[case] expected: bool) {
        let solution = setup();
        assert_eq!(solution.is_anagram_sort(s.into(), t.into()), expected);
    }

    #[rstest]
    #[case("anagram", "nagaram", true)]
    #[case("rat", "car", false)]
    #[case("", "", true)]
    #[case("a", "ab", false)]
    fn test_is_anagram_count(#[case] s: &str, #[case] t: &str, #[case] expected: bool) {
        let solution = setup();
        assert_eq!(solution.is_anagram_count(s.into(), t.into()), expected);
    }

    #[test]
    fn test_consistency_between_methods() {
        let solution = setup();
        let cases = vec![
            ("anagram", "nagaram"),
            ("rat", "car"),
            ("", ""),
            ("a", "ab"),
        ];
        for (s, t) in cases {
            assert_eq!(
                solution.is_anagram_sort(s.into(), t.into()),
                solution.is_anagram_count(s.into(), t.into())
            );
        }
    }
}

