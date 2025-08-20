//! # Problem 3: Longest Substring Without Repeating Characters
//!
//! Given a string `s`, find the length of the **longest substring** without repeating characters.
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::medium::longest_substring_without_repeating_characters::Solution;
//! 
//! let solution = Solution::new();
//! 
//! // Example 1:
//! assert_eq!(solution.length_of_longest_substring("abcabcbb".to_string()), 3);
//! // Explanation: "abc" has length 3
//! 
//! // Example 2: 
//! assert_eq!(solution.length_of_longest_substring("bbbbb".to_string()), 1);
//! // Explanation: "b" has length 1
//! 
//! // Example 3:
//! assert_eq!(solution.length_of_longest_substring("pwwkew".to_string()), 3);
//! // Explanation: "wke" has length 3
//! ```
//!
//! ## Constraints
//!
//! - 0 <= s.length <= 5 * 10^4
//! - s consists of English letters, digits, symbols and spaces.

use std::collections::{HashMap, HashSet};

/// Solution struct following LeetCode format
pub struct Solution;

impl Solution {
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Sliding Window with HashMap (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Use two pointers (left, right) to maintain a sliding window
    /// 2. Use HashMap to track character positions for efficient lookups
    /// 3. When duplicate found, move left pointer to skip the duplicate
    /// 4. Update maximum length as window expands
    /// 
    /// **Time Complexity:** O(n) - Each character visited at most twice
    /// **Space Complexity:** O(min(m, n)) where m is charset size
    /// 
    /// **Key Insight:** When we find a duplicate character, we don't need to 
    /// incrementally move the left pointer. We can jump directly to the position
    /// after the previous occurrence of the duplicate.
    /// 
    /// **Why this is optimal:**
    /// - Must examine each character → O(n) time minimum
    /// - Space bounded by charset size → optimal for practical inputs
    /// - Single pass with smart pointer movement → no redundant work
    pub fn length_of_longest_substring(&self, s: String) -> i32 {
        let chars: Vec<char> = s.chars().collect();
        let mut char_index: HashMap<char, usize> = HashMap::new();
        let mut left = 0;
        let mut max_length = 0;
        
        for (right, &ch) in chars.iter().enumerate() {
            // If character is repeated and is in current window
            if let Some(&prev_index) = char_index.get(&ch) {
                if prev_index >= left {
                    // Move left pointer to skip the duplicate
                    left = prev_index + 1;
                }
            }
            
            // Update character's latest position
            char_index.insert(ch, right);
            
            // Update maximum length
            max_length = max_length.max(right - left + 1);
        }
        
        max_length as i32
    }

    /// # Approach 2: Sliding Window with HashSet
    /// 
    /// **Algorithm:**
    /// 1. Use HashSet to track characters in current window
    /// 2. Expand right pointer and add characters to set
    /// 3. When duplicate found, shrink from left until duplicate removed
    /// 4. Track maximum window size
    /// 
    /// **Time Complexity:** O(2n) = O(n) - Each character added and removed at most once
    /// **Space Complexity:** O(min(m, n)) - HashSet size bounded by charset
    /// 
    /// **Difference from Approach 1:**
    /// - Uses HashSet instead of HashMap (no position tracking)
    /// - Incrementally shrinks window vs jumping directly
    /// - Slightly more iterations but same time complexity
    /// 
    /// **When to prefer:** When position tracking isn't needed
    pub fn length_of_longest_substring_hashset(&self, s: String) -> i32 {
        let chars: Vec<char> = s.chars().collect();
        let mut char_set: HashSet<char> = HashSet::new();
        let mut left = 0;
        let mut max_length = 0;
        
        for (right, &ch) in chars.iter().enumerate() {
            // Shrink window from left until no duplicate
            while char_set.contains(&ch) {
                char_set.remove(&chars[left]);
                left += 1;
            }
            
            // Add current character and update max length
            char_set.insert(ch);
            max_length = max_length.max(right - left + 1);
        }
        
        max_length as i32
    }

    /// # Approach 3: Brute Force (Inefficient - Educational)
    /// 
    /// **Algorithm:**
    /// 1. Check every possible substring
    /// 2. For each substring, verify it has no duplicate characters
    /// 3. Track the maximum length found
    /// 
    /// **Time Complexity:** O(n³) - O(n²) substrings × O(n) duplicate check
    /// **Space Complexity:** O(min(m, n)) - HashSet for duplicate checking
    /// 
    /// **Why this is inefficient:**
    /// - Recalculates overlapping information
    /// - No early termination optimizations
    /// - Cubic time complexity scales poorly
    /// 
    /// **Educational value:**
    /// - Shows the naive approach most people think of first
    /// - Demonstrates why algorithmic optimization matters
    /// - Baseline for comparing optimized solutions
    pub fn length_of_longest_substring_brute_force(&self, s: String) -> i32 {
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();
        let mut max_length = 0;
        
        // Check every possible substring
        for i in 0..n {
            for j in i..n {
                // Check if substring from i to j has no duplicates
                let mut char_set = HashSet::new();
                let mut has_duplicate = false;
                
                for k in i..=j {
                    if !char_set.insert(chars[k]) {
                        has_duplicate = true;
                        break;
                    }
                }
                
                if !has_duplicate {
                    max_length = max_length.max(j - i + 1);
                }
            }
        }
        
        max_length as i32
    }

    /// # Approach 4: Optimized Brute Force (Still Inefficient)
    /// 
    /// **Algorithm:**
    /// 1. For each starting position, extend as far as possible
    /// 2. Stop extending when duplicate is found
    /// 3. Track maximum length achieved
    /// 
    /// **Time Complexity:** O(n²) - O(n) starting positions × O(n) extension
    /// **Space Complexity:** O(min(m, n)) - HashSet for tracking characters
    /// 
    /// **Improvement over pure brute force:**
    /// - Stops early when duplicate found
    /// - Doesn't check impossible longer substrings
    /// - Still quadratic, but better constants
    /// 
    /// **Why still not optimal:** Doesn't leverage the sliding window insight
    pub fn length_of_longest_substring_optimized_brute_force(&self, s: String) -> i32 {
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();
        let mut max_length = 0;
        
        for i in 0..n {
            let mut char_set = HashSet::new();
            
            // Extend from position i as far as possible
            for j in i..n {
                if char_set.contains(&chars[j]) {
                    break; // Duplicate found, can't extend further
                }
                char_set.insert(chars[j]);
                max_length = max_length.max(j - i + 1);
            }
        }
        
        max_length as i32
    }

    /// # Approach 5: ASCII Array Optimization (Specific Case)
    /// 
    /// **Algorithm:**
    /// 1. Use fixed-size array instead of HashMap for ASCII characters
    /// 2. Track last seen index of each character
    /// 3. Same sliding window logic but with array lookups
    /// 
    /// **Time Complexity:** O(n) - Same as HashMap approach
    /// **Space Complexity:** O(1) - Fixed 128-element array for ASCII
    /// 
    /// **When to use:**
    /// - Input guaranteed to be ASCII characters
    /// - Performance-critical applications
    /// - Memory usage optimization important
    /// 
    /// **Trade-offs:**
    /// - **Pros:** Faster lookups, less memory overhead
    /// - **Cons:** Limited to ASCII, less flexible
    /// 
    /// **Note:** This assumes extended ASCII (128 characters). Adjust size based on constraints.
    pub fn length_of_longest_substring_ascii(&self, s: String) -> i32 {
        let chars: Vec<char> = s.chars().collect();
        let mut char_index = [-1i32; 128]; // ASCII characters
        let mut left = 0;
        let mut max_length = 0;
        
        for (right, &ch) in chars.iter().enumerate() {
            let ascii_val = ch as usize;
            
            // Skip if character is outside ASCII range
            if ascii_val >= 128 {
                continue;
            }
            
            // Check if character is repeated in current window
            if char_index[ascii_val] >= left as i32 {
                left = (char_index[ascii_val] + 1) as usize;
            }
            
            char_index[ascii_val] = right as i32;
            max_length = max_length.max(right - left + 1);
        }
        
        max_length as i32
    }

    /// # Approach 6: Two-Pass Analysis (Academic Interest)
    /// 
    /// **Algorithm:**
    /// 1. First pass: Build frequency map of all characters
    /// 2. Second pass: Use frequency information to guide sliding window
    /// 3. Early termination when max possible length is reached
    /// 
    /// **Time Complexity:** O(n) - Two passes through the string
    /// **Space Complexity:** O(min(m, n)) - Character frequency map
    /// 
    /// **Theoretical optimization:**
    /// - Can skip some work if all characters are unique
    /// - Can terminate early in some cases
    /// 
    /// **Practical reality:** Usually not faster than single-pass sliding window
    /// **Educational value:** Shows that more complex isn't always better
    pub fn length_of_longest_substring_two_pass(&self, s: String) -> i32 {
        let chars: Vec<char> = s.chars().collect();
        
        // Early termination: if all characters are unique
        let mut char_freq: HashMap<char, usize> = HashMap::new();
        for &ch in &chars {
            *char_freq.entry(ch).or_insert(0) += 1;
        }
        
        // If all characters appear only once, entire string is the answer
        if char_freq.values().all(|&freq| freq == 1) {
            return chars.len() as i32;
        }
        
        // Otherwise, use standard sliding window
        self.length_of_longest_substring(s)
    }
}

impl Default for Solution {
    fn default() -> Self {
        Self::new()
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
    #[case("abcabcbb", 3)]  // "abc"
    #[case("bbbbb", 1)]     // "b"  
    #[case("pwwkew", 3)]    // "wke"
    #[case("", 0)]          // Empty string
    #[case(" ", 1)]         // Single space
    #[case("au", 2)]        // Two different characters
    #[case("dvdf", 3)]      // "vdf"
    fn test_basic_cases(#[case] input: &str, #[case] expected: i32) {
        let solution = setup();
        assert_eq!(solution.length_of_longest_substring(input.to_string()), expected);
    }

    #[test]
    fn test_single_characters() {
        let solution = setup();
        
        // Single character
        for ch in ['a', '1', '!', ' '] {
            let s = ch.to_string();
            assert_eq!(solution.length_of_longest_substring(s), 1);
        }
    }

    #[test]
    fn test_all_unique_characters() {
        let solution = setup();
        
        // No duplicates
        let unique_strings = vec![
            "abcdef",
            "123456",
            "!@#$%^",
            "abcdefghijklmnopqrstuvwxyz",
        ];
        
        for s in unique_strings {
            let expected = s.len() as i32;
            assert_eq!(solution.length_of_longest_substring(s.to_string()), expected);
        }
    }

    #[test]
    fn test_all_same_characters() {
        let solution = setup();
        
        // All duplicates
        let long_x_string = "x".repeat(1000);
        let duplicate_strings = vec![
            "aaaa",
            "1111111", 
            "      ",
            &long_x_string,
        ];
        
        for s in duplicate_strings {
            assert_eq!(solution.length_of_longest_substring(s.to_string()), 1);
        }
    }

    #[test]
    fn test_complex_patterns() {
        let solution = setup();
        
        // Complex cases with explanations
        assert_eq!(solution.length_of_longest_substring("abba".to_string()), 2); // "ab" or "ba"
        assert_eq!(solution.length_of_longest_substring("tmmzuxt".to_string()), 5); // "mzuxt"
        assert_eq!(solution.length_of_longest_substring("ohvhjdml".to_string()), 6); // "vhjdml"
        assert_eq!(solution.length_of_longest_substring("wobgrovw".to_string()), 6); // "wbgrov"
    }

    #[test]
    fn test_special_characters() {
        let solution = setup();
        
        // Strings with special characters
        assert_eq!(solution.length_of_longest_substring("a!@#a".to_string()), 4); // "!@#a"
        assert_eq!(solution.length_of_longest_substring("12!34!".to_string()), 5); // "12!34"
        assert_eq!(solution.length_of_longest_substring("   a   ".to_string()), 2); // " a"
    }

    #[test]
    fn test_unicode_characters() {
        let solution = setup();
        
        // Unicode strings
        assert_eq!(solution.length_of_longest_substring("αβγαβ".to_string()), 3); // "αβγ"
        assert_eq!(solution.length_of_longest_substring("🚀🌟🚀".to_string()), 2); // "🚀🌟" or "🌟🚀"
        assert_eq!(solution.length_of_longest_substring("こんにちは".to_string()), 5); // All unique
    }

    #[test]
    fn test_boundary_values() {
        let solution = setup();
        
        // Empty and single character
        assert_eq!(solution.length_of_longest_substring("".to_string()), 0);
        assert_eq!(solution.length_of_longest_substring("x".to_string()), 1);
        
        // Two characters
        assert_eq!(solution.length_of_longest_substring("ab".to_string()), 2);
        assert_eq!(solution.length_of_longest_substring("aa".to_string()), 1);
        
        // Maximum constraint length strings would be tested separately due to size
    }

    #[test]
    fn test_large_inputs() {
        let solution = setup();
        
        // Test with larger inputs
        let mut large_unique = String::new();
        for i in 0..100 {
            large_unique.push(char::from(b'a' + (i % 26) as u8));
        }
        // This creates repeating pattern, so max unique is 26
        assert_eq!(solution.length_of_longest_substring(large_unique), 26);
        
        // Large string with pattern
        let pattern = "abcd".repeat(1000); // "abcdabcd..."
        assert_eq!(solution.length_of_longest_substring(pattern), 4);
    }

    #[test]
    fn test_all_approaches_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            "abcabcbb",
            "bbbbb", 
            "pwwkew",
            "",
            " ",
            "au",
            "dvdf",
            "abba",
            "tmmzuxt",
            "wobgrovw",
            "abcdef",
            "aab",
            "cdd",
            "abcabcbb",
        ];
        
        for case in test_cases {
            let result1 = solution.length_of_longest_substring(case.to_string());
            let result2 = solution.length_of_longest_substring_hashset(case.to_string());
            let result3 = solution.length_of_longest_substring_brute_force(case.to_string());
            let result4 = solution.length_of_longest_substring_optimized_brute_force(case.to_string());
            let result5 = solution.length_of_longest_substring_ascii(case.to_string());
            let result6 = solution.length_of_longest_substring_two_pass(case.to_string());
            
            assert_eq!(result1, result2, "HashSet approach differs for '{}'", case);
            assert_eq!(result1, result3, "Brute force differs for '{}'", case);
            assert_eq!(result1, result4, "Optimized brute force differs for '{}'", case);
            assert_eq!(result1, result5, "ASCII optimization differs for '{}'", case);
            assert_eq!(result1, result6, "Two-pass approach differs for '{}'", case);
        }
    }

    #[test]
    fn test_sliding_window_efficiency() {
        let solution = setup();
        
        // Test cases where sliding window should be more efficient than brute force
        
        // Pattern that would cause brute force to do lots of redundant work
        let inefficient_for_brute_force = "abcdefghijklmnopqrstuvwxyza"; 
        let result = solution.length_of_longest_substring(inefficient_for_brute_force.to_string());
        assert_eq!(result, 26); // 26 unique letters
        
        // Long string with early duplicate
        let early_duplicate = format!("a{}", "bcdefghijklmnopqrstuvwxyz".repeat(10));
        let result = solution.length_of_longest_substring(early_duplicate);
        assert_eq!(result, 26);
    }

    #[test]
    fn test_edge_cases_comprehensive() {
        let solution = setup();
        
        // Edge case: Alternating pattern
        assert_eq!(solution.length_of_longest_substring("abab".to_string()), 2);
        
        // Edge case: Three character cycle
        assert_eq!(solution.length_of_longest_substring("abcabc".to_string()), 3);
        
        // Edge case: Palindromic patterns
        assert_eq!(solution.length_of_longest_substring("abccba".to_string()), 3); // "abc" or "cba"
        
        // Edge case: Single character with different positions
        assert_eq!(solution.length_of_longest_substring("abcxabcxabc".to_string()), 4); // "abcx" or "xcab" etc.
    }

    #[test]
    fn test_performance_patterns() {
        let solution = setup();
        
        // Best case: All unique (O(n) with minimal work)
        let all_unique = "abcdefghijklmnopqrstuvwxyz0123456789";
        let result1 = solution.length_of_longest_substring(all_unique.to_string());
        assert_eq!(result1, all_unique.len() as i32);
        
        // Worst case for naive approaches: Many duplicates early
        let many_early_dups = format!("a{}", "b".repeat(1000));
        let result2 = solution.length_of_longest_substring(many_early_dups);
        assert_eq!(result2, 2); // "ab"
        
        // Sliding window efficiency case: Duplicate at end
        let duplicate_at_end = format!("{}a", "abcdefghijklmnopqrstuvwxyz");
        let result3 = solution.length_of_longest_substring(duplicate_at_end);
        assert_eq!(result3, 26); // The alphabet without the duplicate 'a'
    }
}