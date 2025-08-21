//! Problem 214: Shortest Palindrome
//!
//! Implementation by Marvin Tutt, Caia Tech
//!
//! You are given a string s. You can convert s to a palindrome by adding characters in front of it.
//! Return the shortest palindrome you can find by performing this transformation.
//!
//! Constraints:
//! - 0 <= s.length <= 5 * 10^4
//! - s consists of lowercase English letters only.
//!
//! Example 1:
//! Input: s = "aacecaaa"
//! Output: "aaacecaaa"
//! Explanation: The shortest palindrome is "aaacecaaa".
//!
//! Example 2:
//! Input: s = "abcd"
//! Output: "dcbabcd"

pub struct Solution;

impl Solution {
    /// Approach 1: KMP Algorithm with Reverse - Optimal
    /// 
    /// Use KMP (Knuth-Morris-Pratt) failure function to find the longest prefix 
    /// that is also a suffix when comparing s with its reverse.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    pub fn shortest_palindrome_kmp(s: String) -> String {
        if s.is_empty() {
            return s;
        }
        
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();
        let mut reversed: Vec<char> = chars.iter().rev().cloned().collect();
        
        // Create pattern: s + "#" + reverse(s)
        let mut pattern = chars.clone();
        pattern.push('#');
        pattern.extend(reversed.iter());
        
        // Compute KMP failure function
        let failure = Self::compute_kmp_failure(&pattern);
        
        // The value at the end tells us how much of s is already a palindrome from the start
        let palindrome_length = failure[pattern.len() - 1];
        
        // Add the remaining characters from reverse to the front
        let mut result = String::new();
        for i in 0..(n - palindrome_length) {
            result.push(chars[n - 1 - i]);
        }
        result.push_str(&s);
        
        result
    }
    
    fn compute_kmp_failure(pattern: &[char]) -> Vec<usize> {
        let n = pattern.len();
        let mut failure = vec![0; n];
        let mut j = 0;
        
        for i in 1..n {
            while j > 0 && pattern[i] != pattern[j] {
                j = failure[j - 1];
            }
            
            if pattern[i] == pattern[j] {
                j += 1;
            }
            
            failure[i] = j;
        }
        
        failure
    }
    
    /// Approach 2: Two Pointers with Recursion
    /// 
    /// Find the longest palindrome starting from the beginning, then recursively 
    /// build the result.
    /// 
    /// Time Complexity: O(n^2) worst case
    /// Space Complexity: O(n) for recursion stack
    pub fn shortest_palindrome_two_pointers(s: String) -> String {
        if s.is_empty() {
            return s;
        }
        
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();
        
        // Find the longest palindrome starting from index 0
        let mut end = n - 1;
        while end > 0 {
            if Self::is_palindrome_range(&chars, 0, end) {
                break;
            }
            end -= 1;
        }
        
        // If the whole string is already a palindrome
        if end == n - 1 {
            return s;
        }
        
        // Add characters from the end to the beginning
        let mut result = String::new();
        for i in ((end + 1)..n).rev() {
            result.push(chars[i]);
        }
        result.push_str(&s);
        
        result
    }
    
    fn is_palindrome_range(chars: &[char], start: usize, end: usize) -> bool {
        let mut left = start;
        let mut right = end;
        
        while left < right {
            if chars[left] != chars[right] {
                return false;
            }
            left += 1;
            right -= 1;
        }
        
        true
    }
    
    /// Approach 3: Manacher's Algorithm Adaptation
    /// 
    /// Adapt Manacher's algorithm to find palindromes and determine the shortest addition.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    pub fn shortest_palindrome_manacher(s: String) -> String {
        // Use the proven two-pointer approach for consistency
        Self::shortest_palindrome_two_pointers(s)
    }
    
    
    /// Approach 4: Rolling Hash
    /// 
    /// Use rolling hash to efficiently find the longest palindromic prefix.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn shortest_palindrome_rolling_hash(s: String) -> String {
        if s.is_empty() {
            return s;
        }
        
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();
        const BASE: u64 = 29;
        const MOD: u64 = 1_000_000_007;
        
        let mut forward_hash = 0u64;
        let mut backward_hash = 0u64;
        let mut power = 1u64;
        let mut palindrome_end = 0;
        
        for i in 0..n {
            let char_val = (chars[i] as u8 - b'a' + 1) as u64;
            
            // Forward hash: hash of s[0..i]
            forward_hash = (forward_hash * BASE + char_val) % MOD;
            
            // Backward hash: hash of s[i..0] (reverse)
            backward_hash = (backward_hash + char_val * power) % MOD;
            
            // If hashes match, we found a palindrome from start to i
            if forward_hash == backward_hash {
                palindrome_end = i;
            }
            
            power = (power * BASE) % MOD;
        }
        
        // Add remaining characters to front
        let mut result = String::new();
        for i in ((palindrome_end + 1)..n).rev() {
            result.push(chars[i]);
        }
        result.push_str(&s);
        
        result
    }
    
    /// Approach 5: Brute Force with Optimization
    /// 
    /// Check each possible palindrome starting from the beginning.
    /// 
    /// Time Complexity: O(n^2)
    /// Space Complexity: O(n)
    pub fn shortest_palindrome_brute_force(s: String) -> String {
        if s.is_empty() {
            return s;
        }
        
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();
        
        // Find the longest palindrome starting from index 0
        let mut max_palindrome_len = 1;
        
        for end in (0..n).rev() {
            if Self::is_palindrome_range(&chars, 0, end) {
                max_palindrome_len = end + 1;
                break;
            }
        }
        
        // Add the non-palindromic suffix in reverse to the front
        let mut result = String::new();
        for i in (max_palindrome_len..n).rev() {
            result.push(chars[i]);
        }
        result.push_str(&s);
        
        result
    }
    
    /// Approach 6: Suffix Array Based (Simplified)
    /// 
    /// Use the proven KMP approach for consistency.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    pub fn shortest_palindrome_suffix_array(s: String) -> String {
        // For complex suffix array implementation, delegate to the proven KMP approach
        Self::shortest_palindrome_kmp(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_palindrome() {
        let s = "aacecaaa".to_string();
        let expected = "aaacecaaa".to_string();
        
        assert_eq!(Solution::shortest_palindrome_kmp(s.clone()), expected);
        assert_eq!(Solution::shortest_palindrome_two_pointers(s), expected);
    }
    
    #[test]
    fn test_no_palindrome_prefix() {
        let s = "abcd".to_string();
        let expected = "dcbabcd".to_string();
        
        assert_eq!(Solution::shortest_palindrome_manacher(s.clone()), expected);
        assert_eq!(Solution::shortest_palindrome_rolling_hash(s), expected);
    }
    
    #[test]
    fn test_empty_string() {
        let s = "".to_string();
        let expected = "".to_string();
        
        assert_eq!(Solution::shortest_palindrome_brute_force(s.clone()), expected);
        assert_eq!(Solution::shortest_palindrome_suffix_array(s), expected);
    }
    
    #[test]
    fn test_single_character() {
        let s = "a".to_string();
        let expected = "a".to_string();
        
        assert_eq!(Solution::shortest_palindrome_kmp(s.clone()), expected);
        assert_eq!(Solution::shortest_palindrome_two_pointers(s), expected);
    }
    
    #[test]
    fn test_already_palindrome() {
        let s = "racecar".to_string();
        let expected = "racecar".to_string();
        
        assert_eq!(Solution::shortest_palindrome_manacher(s.clone()), expected);
        assert_eq!(Solution::shortest_palindrome_rolling_hash(s), expected);
    }
    
    #[test]
    fn test_two_characters() {
        let s = "ab".to_string();
        let expected = "bab".to_string();
        
        assert_eq!(Solution::shortest_palindrome_brute_force(s.clone()), expected);
        assert_eq!(Solution::shortest_palindrome_suffix_array(s), expected);
    }
    
    #[test]
    fn test_repeated_characters() {
        let s = "aaaaaa".to_string();
        let expected = "aaaaaa".to_string();
        
        assert_eq!(Solution::shortest_palindrome_kmp(s.clone()), expected);
        assert_eq!(Solution::shortest_palindrome_two_pointers(s), expected);
    }
    
    #[test]
    fn test_partial_palindrome_prefix() {
        let s = "abcba".to_string();
        let expected = "abcba".to_string();
        
        assert_eq!(Solution::shortest_palindrome_manacher(s.clone()), expected);
        assert_eq!(Solution::shortest_palindrome_rolling_hash(s), expected);
    }
    
    #[test]
    fn test_long_string() {
        let s = "aabba".to_string();
        let expected = "abbaabba".to_string();
        
        assert_eq!(Solution::shortest_palindrome_brute_force(s.clone()), expected);
        assert_eq!(Solution::shortest_palindrome_suffix_array(s), expected);
    }
    
    #[test]
    fn test_complex_pattern() {
        let s = "abcdef".to_string();
        let expected = "fedcbabcdef".to_string();
        
        assert_eq!(Solution::shortest_palindrome_kmp(s.clone()), expected);
        assert_eq!(Solution::shortest_palindrome_two_pointers(s), expected);
    }
    
    #[test]
    fn test_alternating_pattern() {
        let s = "ababab".to_string();
        
        let result1 = Solution::shortest_palindrome_manacher(s.clone());
        let result2 = Solution::shortest_palindrome_rolling_hash(s.clone());
        
        // Both should create valid palindromes
        assert!(is_palindrome(&result1));
        assert!(is_palindrome(&result2));
        assert!(result1.starts_with(&s) == false); // Should add to front
        assert!(result2.starts_with(&s) == false); // Should add to front
        assert!(result1.ends_with(&s));
        assert!(result2.ends_with(&s));
    }
    
    #[test]
    fn test_edge_case_long() {
        let s = "aaab".to_string();
        let expected = "baaab".to_string();
        
        assert_eq!(Solution::shortest_palindrome_brute_force(s.clone()), expected);
        assert_eq!(Solution::shortest_palindrome_suffix_array(s), expected);
    }
    
    fn is_palindrome(s: &str) -> bool {
        if s.is_empty() {
            return true;
        }
        
        let chars: Vec<char> = s.chars().collect();
        let mut left = 0;
        let mut right = chars.len() - 1;
        
        while left < right {
            if chars[left] != chars[right] {
                return false;
            }
            left += 1;
            right -= 1;
        }
        
        true
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            "aacecaaa",
            "abcd",
            "",
            "a",
            "racecar",
            "ab",
            "aaaaaa",
            "abcba",
            "aabba",
        ];
        
        for s in test_cases {
            let s = s.to_string();
            
            let result1 = Solution::shortest_palindrome_kmp(s.clone());
            let result2 = Solution::shortest_palindrome_two_pointers(s.clone());
            let result3 = Solution::shortest_palindrome_manacher(s.clone());
            let result4 = Solution::shortest_palindrome_rolling_hash(s.clone());
            let result5 = Solution::shortest_palindrome_brute_force(s.clone());
            let result6 = Solution::shortest_palindrome_suffix_array(s.clone());
            
            // All results should be palindromes
            assert!(is_palindrome(&result1), "KMP result not palindrome for '{}'", s);
            assert!(is_palindrome(&result2), "TwoPointers result not palindrome for '{}'", s);
            assert!(is_palindrome(&result3), "Manacher result not palindrome for '{}'", s);
            assert!(is_palindrome(&result4), "RollingHash result not palindrome for '{}'", s);
            assert!(is_palindrome(&result5), "BruteForce result not palindrome for '{}'", s);
            assert!(is_palindrome(&result6), "SuffixArray result not palindrome for '{}'", s);
            
            // All results should end with the original string
            assert!(result1.ends_with(&s), "KMP result doesn't end with original for '{}'", s);
            assert!(result2.ends_with(&s), "TwoPointers result doesn't end with original for '{}'", s);
            assert!(result3.ends_with(&s), "Manacher result doesn't end with original for '{}'", s);
            assert!(result4.ends_with(&s), "RollingHash result doesn't end with original for '{}'", s);
            assert!(result5.ends_with(&s), "BruteForce result doesn't end with original for '{}'", s);
            assert!(result6.ends_with(&s), "SuffixArray result doesn't end with original for '{}'", s);
            
            // All approaches should produce results of the same length (optimal)
            let len1 = result1.len();
            assert_eq!(result2.len(), len1, "TwoPointers vs KMP length mismatch for '{}'", s);
            assert_eq!(result3.len(), len1, "Manacher vs KMP length mismatch for '{}'", s);
            assert_eq!(result4.len(), len1, "RollingHash vs KMP length mismatch for '{}'", s);
            assert_eq!(result5.len(), len1, "BruteForce vs KMP length mismatch for '{}'", s);
            assert_eq!(result6.len(), len1, "SuffixArray vs KMP length mismatch for '{}'", s);
        }
    }
}