//! # Problem 680: Valid Palindrome II
//!
//! **Difficulty**: Easy
//! **Topics**: Two Pointers, String
//! **Acceptance Rate**: 39.1%

/// Solution struct following LeetCode format
pub struct Solution;

impl Solution {
    /// Create a new solution instance
    pub fn new() -> Self {
        Solution
    }

    /// Main solution approach using two pointers
    /// 
    /// Time Complexity: O(n) - single pass with possible palindrome check
    /// Space Complexity: O(1) - constant space
    pub fn valid_palindrome(&self, s: String) -> bool {
        let chars: Vec<char> = s.chars().collect();
        let mut left = 0;
        let mut right = chars.len().saturating_sub(1);
        
        while left < right {
            if chars[left] != chars[right] {
                // Try removing left character or right character
                return self.is_palindrome_range(&chars, left + 1, right) ||
                       self.is_palindrome_range(&chars, left, right - 1);
            }
            left += 1;
            right -= 1;
        }
        
        true
    }

    /// Helper function to check if substring is palindrome
    fn is_palindrome_range(&self, chars: &[char], mut left: usize, mut right: usize) -> bool {
        while left < right {
            if chars[left] != chars[right] {
                return false;
            }
            left += 1;
            right -= 1;
        }
        true
    }

    /// Alternative solution with explicit character removal simulation
    /// 
    /// Time Complexity: O(n) - linear scan with palindrome verification
    /// Space Complexity: O(n) - for string manipulation in worst case
    pub fn valid_palindrome_alternative(&self, s: String) -> bool {
        // First check if already palindrome
        if self.is_palindrome(&s) {
            return true;
        }
        
        let chars: Vec<char> = s.chars().collect();
        
        // Try removing each character once
        for i in 0..chars.len() {
            let mut modified = chars.clone();
            modified.remove(i);
            let modified_string: String = modified.iter().collect();
            
            if self.is_palindrome(&modified_string) {
                return true;
            }
        }
        
        false
    }

    /// Helper function to check if entire string is palindrome
    fn is_palindrome(&self, s: &str) -> bool {
        let chars: Vec<char> = s.chars().collect();
        let mut left = 0;
        let mut right = chars.len().saturating_sub(1);
        
        while left < right {
            if chars[left] != chars[right] {
                return false;
            }
            left += 1;
            right -= 1;
        }
        
        true
    }

    /// Brute force solution (for comparison)
    /// 
    /// Time Complexity: O(n²) - for each position, check if removal creates palindrome
    /// Space Complexity: O(n) - for string operations
    pub fn valid_palindrome_brute_force(&self, s: String) -> bool {
        // Check if already palindrome
        if s == s.chars().rev().collect::<String>() {
            return true;
        }
        
        // Try removing each character
        for i in 0..s.len() {
            let mut chars: Vec<char> = s.chars().collect();
            chars.remove(i);
            let modified: String = chars.iter().collect();
            
            if modified == modified.chars().rev().collect::<String>() {
                return true;
            }
        }
        
        false
    }

    /// Optimized two-pointer approach with early termination
    /// 
    /// Time Complexity: O(n) - optimal approach
    /// Space Complexity: O(1) - constant space
    pub fn valid_palindrome_optimized(&self, s: String) -> bool {
        let chars: Vec<char> = s.chars().collect();
        self.can_be_palindrome(&chars, 0, chars.len().saturating_sub(1), false)
    }

    /// Recursive helper with deletion tracking
    fn can_be_palindrome(&self, chars: &[char], mut left: usize, mut right: usize, deleted: bool) -> bool {
        while left < right {
            if chars[left] == chars[right] {
                left += 1;
                right -= 1;
            } else if !deleted {
                // Try deleting left or right character
                return self.can_be_palindrome(chars, left + 1, right, true) ||
                       self.can_be_palindrome(chars, left, right - 1, true);
            } else {
                // Already deleted once, can't delete again
                return false;
            }
        }
        true
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

    #[test]
    fn test_basic_cases() {
        let solution = Solution::new();
        
        // Test case 1: "aba" - already palindrome
        assert_eq!(solution.valid_palindrome("aba".to_string()), true);
        
        // Test case 2: "abca" - can remove 'c' to get "aba"
        assert_eq!(solution.valid_palindrome("abca".to_string()), true);
        
        // Test case 3: "abc" - cannot form palindrome by removing one char
        assert_eq!(solution.valid_palindrome("abc".to_string()), false);
        
        // Test case 4: Single character
        assert_eq!(solution.valid_palindrome("a".to_string()), true);
        
        // Test case 5: Empty string
        assert_eq!(solution.valid_palindrome("".to_string()), true);
    }

    #[test]
    fn test_edge_cases() {
        let solution = Solution::new();
        
        // Two characters, same
        assert_eq!(solution.valid_palindrome("aa".to_string()), true);
        
        // Two characters, different
        assert_eq!(solution.valid_palindrome("ab".to_string()), true);
        
        // Three characters, need to remove middle
        assert_eq!(solution.valid_palindrome("abc".to_string()), false);
        
        // Three characters, need to remove first or last
        assert_eq!(solution.valid_palindrome("abb".to_string()), true);
        assert_eq!(solution.valid_palindrome("baa".to_string()), true);
        
        // Longer palindrome with one extra character
        assert_eq!(solution.valid_palindrome("raceacar".to_string()), true);
        
        // Case where we need to remove from the end
        assert_eq!(solution.valid_palindrome("abcdcba".to_string()), true);
    }

    #[test]
    fn test_complex_cases() {
        let solution = Solution::new();
        
        // Multiple possible removals
        assert_eq!(solution.valid_palindrome("racecar".to_string()), true); // already palindrome
        
        // Requires careful choice of which character to remove
        assert_eq!(solution.valid_palindrome("aguokepatgbnvfqmgmlcupuufxoohdfpgjdmysgvhmvffcnqxjjxqncffvmhvgsymdjgpfdhooxfuupuculmgmqfvnbgtapekouga".to_string()), true);
        
        // Long string that can't be made palindrome
        assert_eq!(solution.valid_palindrome("abcdefghijklmnop".to_string()), false);
        
        // Palindrome with repeated patterns
        assert_eq!(solution.valid_palindrome("aabaa".to_string()), true);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = Solution::new();
        
        let test_cases = vec![
            "aba".to_string(),
            "abca".to_string(),
            "abc".to_string(),
            "a".to_string(),
            "".to_string(),
            "raceacar".to_string(),
            "ab".to_string(),
        ];

        for case in test_cases {
            let result1 = solution.valid_palindrome(case.clone());
            let result2 = solution.valid_palindrome_alternative(case.clone());
            let result3 = solution.valid_palindrome_brute_force(case.clone());
            let result4 = solution.valid_palindrome_optimized(case.clone());
            
            assert_eq!(result1, result2, "Main and alternative approaches should match for: {}", case);
            assert_eq!(result1, result3, "Main and brute force approaches should match for: {}", case);
            assert_eq!(result1, result4, "Main and optimized approaches should match for: {}", case);
        }
    }

    #[test]
    fn test_performance_scenarios() {
        let solution = Solution::new();
        
        // Very long palindrome
        let long_palindrome = "a".repeat(1000) + "b" + &"a".repeat(1000);
        assert_eq!(solution.valid_palindrome(long_palindrome), true);
        
        // Very long non-palindrome
        let long_non_palindrome = "abcd".repeat(250);
        assert_eq!(solution.valid_palindrome(long_non_palindrome), false);
        
        // Almost palindrome (needs one removal)
        let almost_palindrome = "racecar".to_string() + "x";
        assert_eq!(solution.valid_palindrome(almost_palindrome), true);
    }
}

#[cfg(test)]
mod benchmarks {
    // Note: Benchmarks will be added to benches/ directory for Criterion integration
}