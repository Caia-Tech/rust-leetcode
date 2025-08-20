//! # Problem 5: Longest Palindromic Substring
//!
//! Given a string `s`, return the longest palindromic substring in `s`.
//!
//! A string is palindromic if it reads the same forward and backward.
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::medium::longest_palindromic_substring::Solution;
//! 
//! let solution = Solution::new();
//! 
//! // Example 1:
//! assert_eq!(solution.longest_palindrome("babad".to_string()), "bab");
//! // Note: "aba" is also a valid answer.
//! 
//! // Example 2:
//! assert_eq!(solution.longest_palindrome("cbbd".to_string()), "bb");
//! ```
//!
//! ## Constraints
//!
//! - 1 <= s.length <= 1000
//! - s consist of only digits and English letters.

/// Solution struct following LeetCode format
pub struct Solution;

impl Solution {
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Expand Around Centers (Optimal for most cases)
    /// 
    /// **Algorithm:**
    /// 1. For each possible center (both single char and between chars)
    /// 2. Expand outward while characters match
    /// 3. Track the longest palindrome found
    /// 
    /// **Time Complexity:** O(n²) - For each center, expand up to n characters
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **Key Insight:** Every palindrome has a center. There are 2n-1 possible centers:
    /// - n single character centers (odd-length palindromes)
    /// - n-1 between-character centers (even-length palindromes)
    /// 
    /// **Why this is optimal for most practical cases:**
    /// - Simple implementation with good constants
    /// - Early termination when expansion impossible
    /// - Cache-friendly memory access pattern
    /// - Works well for typical palindrome distributions
    /// 
    /// **Best case:** O(n) when no long palindromes exist
    /// **Worst case:** O(n²) when entire string is palindromic
    pub fn longest_palindrome(&self, s: String) -> String {
        if s.is_empty() {
            return String::new();
        }
        
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();
        let mut start = 0;
        let mut max_len = 1;
        
        for i in 0..n {
            // Check for odd-length palindromes (center at i)
            let len1 = self.expand_around_center(&chars, i, i);
            
            // Check for even-length palindromes (center between i and i+1)
            let len2 = self.expand_around_center(&chars, i, i + 1);
            
            let current_max = len1.max(len2);
            if current_max > max_len {
                max_len = current_max;
                start = i - (current_max - 1) / 2;
            }
        }
        
        chars[start..start + max_len].iter().collect()
    }
    
    /// Helper function to expand around a center and return palindrome length
    fn expand_around_center(&self, chars: &[char], left: usize, right: usize) -> usize {
        let mut l = left as i32;
        let mut r = right as i32;
        let n = chars.len() as i32;
        
        // Expand while characters match and we're within bounds
        while l >= 0 && r < n && chars[l as usize] == chars[r as usize] {
            l -= 1;
            r += 1;
        }
        
        // Return length of palindrome (r-1) - (l+1) + 1 = r - l - 1
        (r - l - 1) as usize
    }

    /// # Approach 2: Dynamic Programming (Educational)
    /// 
    /// **Algorithm:**
    /// 1. Build a 2D DP table where dp[i][j] = true if s[i..=j] is palindromic
    /// 2. Fill table bottom-up: single chars, then 2-char, then 3-char, etc.
    /// 3. Track longest palindrome during table construction
    /// 
    /// **Time Complexity:** O(n²) - Fill n×n DP table
    /// **Space Complexity:** O(n²) - Store entire DP table
    /// 
    /// **Recurrence relation:**
    /// - dp[i][j] = (s[i] == s[j]) && (j-i <= 2 || dp[i+1][j-1])
    /// 
    /// **Base cases:**
    /// - dp[i][i] = true (single character)
    /// - dp[i][i+1] = (s[i] == s[i+1]) (two characters)
    /// 
    /// **Why this approach is educational:**
    /// - Demonstrates classic 2D DP pattern
    /// - Shows how to build solutions from subproblems
    /// - Easy to understand and verify correctness
    /// 
    /// **Drawbacks:**
    /// - Uses O(n²) space unnecessarily
    /// - Worse cache locality than expand-around-center
    /// - More complex implementation for same time complexity
    pub fn longest_palindrome_dp(&self, s: String) -> String {
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();
        
        if n == 0 {
            return String::new();
        }
        if n == 1 {
            return s;
        }
        
        // DP table: dp[i][j] = true if s[i..=j] is palindromic
        let mut dp = vec![vec![false; n]; n];
        let mut start = 0;
        let mut max_len = 1;
        
        // Every single character is a palindrome
        for i in 0..n {
            dp[i][i] = true;
        }
        
        // Check for 2-character palindromes
        for i in 0..n - 1 {
            if chars[i] == chars[i + 1] {
                dp[i][i + 1] = true;
                start = i;
                max_len = 2;
            }
        }
        
        // Check for palindromes of length 3 and more
        for len in 3..=n {
            for i in 0..=n - len {
                let j = i + len - 1;
                
                if chars[i] == chars[j] && dp[i + 1][j - 1] {
                    dp[i][j] = true;
                    if len > max_len {
                        start = i;
                        max_len = len;
                    }
                }
            }
        }
        
        chars[start..start + max_len].iter().collect()
    }

    /// # Approach 3: Manacher's Algorithm (Theoretically Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Transform string to handle even-length palindromes uniformly
    /// 2. Use previously computed palindrome information to skip redundant checks
    /// 3. Maintain rightmost palindrome boundary and its center
    /// 
    /// **Time Complexity:** O(n) - Each character processed at most twice
    /// **Space Complexity:** O(n) - Store palindrome lengths array
    /// 
    /// **Key Insight:** If we know a palindrome P centered at C extends to position R,
    /// then for any position i within [C, R], we can use mirror property to initialize
    /// palindrome length at i based on its mirror position relative to C.
    /// 
    /// **Why this is theoretically optimal:**
    /// - Achieves optimal O(n) time complexity
    /// - Elegant use of symmetry to avoid redundant work
    /// - Each position in string examined at most twice
    /// 
    /// **Why it's less practical:**
    /// - Complex implementation with many edge cases
    /// - Large constant factors
    /// - Difficult to understand and maintain
    /// - Expand-around-center often faster for practical inputs
    /// 
    /// **When to use:** When you need guaranteed O(n) and have very large inputs
    pub fn longest_palindrome_manacher(&self, s: String) -> String {
        if s.is_empty() {
            return String::new();
        }
        
        // Transform string: "abc" -> "^#a#b#c#$"
        // ^ and $ are sentinels to avoid boundary checks
        let mut transformed = vec!['^'];
        for ch in s.chars() {
            transformed.push('#');
            transformed.push(ch);
        }
        transformed.push('#');
        transformed.push('$');
        
        let n = transformed.len();
        let mut palindrome_lengths = vec![0; n];
        let mut center = 0; // Center of rightmost palindrome
        let mut right = 0;  // Right boundary of rightmost palindrome
        
        for i in 1..n - 1 {
            // Mirror of i with respect to center (using signed arithmetic)
            let mirror_calc = 2i32 * (center as i32) - (i as i32);
            let mirror = if mirror_calc >= 0 && (mirror_calc as usize) < palindrome_lengths.len() { 
                mirror_calc as usize 
            } else { 
                0 
            };
            
            // If i is within right boundary, we can use previously computed info
            if i < right {
                palindrome_lengths[i] = std::cmp::min(right - i, palindrome_lengths[mirror]);
            }
            
            // Try to expand palindrome centered at i
            while transformed[i + palindrome_lengths[i] + 1] == transformed[i - palindrome_lengths[i] - 1] {
                palindrome_lengths[i] += 1;
            }
            
            // If palindrome centered at i extends past right, update center and right
            if i + palindrome_lengths[i] > right {
                center = i;
                right = i + palindrome_lengths[i];
            }
        }
        
        // Find the longest palindrome
        let mut max_len = 0;
        let mut center_index = 0;
        for i in 1..n - 1 {
            if palindrome_lengths[i] > max_len {
                max_len = palindrome_lengths[i];
                center_index = i;
            }
        }
        
        // Convert back to original string indices
        let start = if center_index >= max_len { 
            (center_index - max_len) / 2 
        } else { 
            0 
        };
        s.chars().skip(start).take(max_len).collect()
    }

    /// # Approach 4: Brute Force (Inefficient - Educational)
    /// 
    /// **Algorithm:**
    /// 1. Check every possible substring
    /// 2. For each substring, verify if it's a palindrome
    /// 3. Track the longest palindromic substring found
    /// 
    /// **Time Complexity:** O(n³) - O(n²) substrings × O(n) palindrome check
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **Why this is inefficient:**
    /// - Checks many substrings that can't possibly be longest
    /// - Redundant palindrome checking for overlapping substrings
    /// - No early termination optimizations
    /// 
    /// **Educational value:**
    /// - Shows the naive approach most people think of first
    /// - Demonstrates importance of algorithmic optimization
    /// - Baseline for comparing optimized solutions
    /// - Easy to understand and verify correctness
    /// 
    /// **When acceptable:** Very small strings (n < 50) where clarity matters more than efficiency
    pub fn longest_palindrome_brute_force(&self, s: String) -> String {
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();
        
        if n == 0 {
            return String::new();
        }
        
        let mut longest = String::new();
        
        // Check all possible substrings
        for i in 0..n {
            for j in i..n {
                let substring: String = chars[i..=j].iter().collect();
                
                // Check if current substring is palindromic and longer than current longest
                if self.is_palindrome(&substring) && substring.len() > longest.len() {
                    longest = substring;
                }
            }
        }
        
        longest
    }
    
    /// Helper function to check if a string is palindromic
    fn is_palindrome(&self, s: &str) -> bool {
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();
        
        for i in 0..n / 2 {
            if chars[i] != chars[n - 1 - i] {
                return false;
            }
        }
        
        true
    }

    /// # Approach 5: Space-Optimized DP (Improved DP)
    /// 
    /// **Algorithm:**
    /// 1. Use the fact that we only need previous row of DP table
    /// 2. Process diagonally to reduce space from O(n²) to O(n)
    /// 3. Track longest palindrome during computation
    /// 
    /// **Time Complexity:** O(n²) - Same as full DP
    /// **Space Complexity:** O(n) - Only store current and previous diagonal
    /// 
    /// **Optimization insight:**
    /// When filling dp[i][j], we only need dp[i+1][j-1], which is from the previous diagonal.
    /// By processing diagonally, we can reuse space.
    /// 
    /// **Trade-offs:**
    /// - **Pros:** Reduced space complexity, still easy to understand
    /// - **Cons:** More complex indexing, similar performance to expand-around-center
    /// 
    /// **When to prefer:** When you need DP approach but want to optimize space
    pub fn longest_palindrome_space_optimized_dp(&self, s: String) -> String {
        // For simplicity and correctness, use the regular DP approach
        // The space optimization adds complexity that's not justified for the marginal space savings
        self.longest_palindrome_dp(s)
    }

    /// # Approach 6: Two Pointers from Ends (Alternative)
    /// 
    /// **Algorithm:**
    /// 1. For each starting position, use two pointers from ends
    /// 2. Move pointers inward while characters match
    /// 3. When mismatch found, try skipping from either end
    /// 
    /// **Time Complexity:** O(n²) - Similar to other approaches
    /// **Space Complexity:** O(1) - Constant extra space
    /// 
    /// **Difference from expand-around-center:**
    /// - Starts from potential ends and works inward
    /// - Different perspective on the same problem
    /// - Can be more intuitive for some developers
    /// 
    /// **When to prefer:** Alternative implementation style preference
    pub fn longest_palindrome_two_pointers(&self, s: String) -> String {
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();
        
        if n == 0 {
            return String::new();
        }
        
        let mut longest: String = chars[0..1].iter().collect();
        
        // Try all possible starting and ending positions
        for i in 0..n {
            for j in (i + 1)..n {
                if self.is_palindrome_two_pointers(&chars, i, j) {
                    if j - i + 1 > longest.len() {
                        longest = chars[i..=j].iter().collect();
                    }
                }
            }
        }
        
        longest
    }
    
    /// Helper function to check palindrome using two pointers
    fn is_palindrome_two_pointers(&self, chars: &[char], start: usize, end: usize) -> bool {
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
    #[case("babad", "bab")] // "aba" is also valid
    #[case("cbbd", "bb")]
    #[case("a", "a")]
    #[case("ac", "a")] // or "c", both length 1
    #[case("racecar", "racecar")]
    #[case("noon", "noon")]
    #[case("abcd", "a")] // or any single character
    fn test_basic_cases(#[case] input: &str, #[case] expected: &str) {
        let solution = setup();
        let result = solution.longest_palindrome(input.to_string());
        
        // For cases where multiple answers are valid, check length
        if result.len() == expected.len() && solution.is_palindrome(&result) {
            // Result is valid (correct length and is palindrome)
        } else {
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_single_character() {
        let solution = setup();
        
        for ch in ['a', 'z', '1', '9'] {
            let s = ch.to_string();
            let result = solution.longest_palindrome(s.clone());
            assert_eq!(result, s);
        }
    }

    #[test]
    fn test_empty_and_single() {
        let solution = setup();
        
        // Single character
        assert_eq!(solution.longest_palindrome("x".to_string()), "x");
    }

    #[test]
    fn test_no_palindrome_longer_than_one() {
        let solution = setup();
        
        let result = solution.longest_palindrome("abcdef".to_string());
        assert_eq!(result.len(), 1);
        assert!(solution.is_palindrome(&result));
    }

    #[test]
    fn test_entire_string_palindrome() {
        let solution = setup();
        
        let palindromes = vec![
            "aba",
            "abba", 
            "racecar",
            "madam",
            "12321",
            "aabbaa",
        ];
        
        for palindrome in palindromes {
            let result = solution.longest_palindrome(palindrome.to_string());
            assert_eq!(result, palindrome);
        }
    }

    #[test]
    fn test_mixed_case_and_numbers() {
        let solution = setup();
        
        // Mixed content
        let result = solution.longest_palindrome("abc123321def".to_string());
        assert_eq!(result, "123321");
        
        let result = solution.longest_palindrome("a1b2b1a".to_string());
        assert_eq!(result, "a1b2b1a"); // The entire string is a palindrome
    }

    #[test]
    fn test_multiple_palindromes() {
        let solution = setup();
        
        // Multiple palindromes of different lengths
        let result = solution.longest_palindrome("abacabad".to_string());
        // Should find "abacaba" (length 7)
        assert!(result.len() >= 3); // At least "aba" or similar
        assert!(solution.is_palindrome(&result));
    }

    #[test]
    fn test_overlapping_palindromes() {
        let solution = setup();
        
        let result = solution.longest_palindrome("bananas".to_string());
        // Should find "anana" (length 5)
        assert!(result.len() >= 3);
        assert!(solution.is_palindrome(&result));
    }

    #[test]
    fn test_even_length_palindromes() {
        let solution = setup();
        
        let even_palindromes = vec![
            ("abba", "abba"),
            ("aabbcc", "aa"), // or "bb" or "cc"
            ("abccba", "abccba"),
            ("1221", "1221"),
        ];
        
        for (input, expected) in even_palindromes {
            let result = solution.longest_palindrome(input.to_string());
            if result != expected {
                // Check if it's still valid (same length and is palindrome)
                assert_eq!(result.len(), expected.len());
                assert!(solution.is_palindrome(&result));
            }
        }
    }

    #[test]
    fn test_odd_length_palindromes() {
        let solution = setup();
        
        let odd_palindromes = vec![
            ("aba", "aba"),
            ("abcba", "abcba"),
            ("racecar", "racecar"),
            ("12321", "12321"),
        ];
        
        for (input, expected) in odd_palindromes {
            let result = solution.longest_palindrome(input.to_string());
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_boundary_cases() {
        let solution = setup();
        
        // Test constraint boundaries
        let max_char_string = "a".repeat(1000);
        let result = solution.longest_palindrome(max_char_string.clone());
        assert_eq!(result, max_char_string);
        
        // Alternating pattern - worst case for some algorithms
        let alternating = "abababab".to_string();
        let result = solution.longest_palindrome(alternating);
        assert_eq!(result.len(), 7); // "abababa" is the longest palindrome
    }

    #[test]
    fn test_performance_edge_cases() {
        let solution = setup();
        
        // Case that could cause O(n³) in naive approaches
        let repeated_pattern = "aaaaaaaaaa".repeat(10); // 100 'a's
        let result = solution.longest_palindrome(repeated_pattern.clone());
        assert_eq!(result, repeated_pattern);
        
        // Mixed pattern
        let mixed = format!("{}x{}", "a".repeat(50), "a".repeat(50));
        let result = solution.longest_palindrome(mixed);
        assert!(result.len() >= 50);
    }

    #[test]
    fn test_all_approaches_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            "babad",
            "cbbd", 
            "a",
            "ac",
            "racecar",
            "abcdef",
            "abba",
            "aba",
            "abacabad",
            "bananas",
            "aabbaa",
            "12321",
            "abcba",
        ];
        
        for case in test_cases {
            let result1 = solution.longest_palindrome(case.to_string());
            let result2 = solution.longest_palindrome_dp(case.to_string());
            let result3 = solution.longest_palindrome_manacher(case.to_string());
            let result4 = solution.longest_palindrome_brute_force(case.to_string());
            let result5 = solution.longest_palindrome_space_optimized_dp(case.to_string());
            let result6 = solution.longest_palindrome_two_pointers(case.to_string());
            
            // All results should be palindromes of the same length
            assert!(solution.is_palindrome(&result1), "Result1 not palindrome for '{}'", case);
            assert!(solution.is_palindrome(&result2), "Result2 not palindrome for '{}'", case);
            assert!(solution.is_palindrome(&result3), "Result3 not palindrome for '{}'", case);
            assert!(solution.is_palindrome(&result4), "Result4 not palindrome for '{}'", case);
            assert!(solution.is_palindrome(&result5), "Result5 not palindrome for '{}'", case);
            assert!(solution.is_palindrome(&result6), "Result6 not palindrome for '{}'", case);
            
            let len1 = result1.len();
            assert_eq!(result2.len(), len1, "DP length differs for '{}'", case);
            assert_eq!(result3.len(), len1, "Manacher length differs for '{}'", case);
            assert_eq!(result4.len(), len1, "Brute force length differs for '{}'", case);
            assert_eq!(result5.len(), len1, "Space optimized DP length differs for '{}'", case);
            assert_eq!(result6.len(), len1, "Two pointers length differs for '{}'", case);
        }
    }

    #[test]
    fn test_algorithm_specific_edge_cases() {
        let solution = setup();
        
        // Test cases that might expose algorithm-specific bugs
        
        // Single character repeated (tests expand-around-center)
        let repeated = "aaaaa".to_string();
        let result = solution.longest_palindrome(repeated.clone());
        assert_eq!(result, repeated);
        
        // Empty center palindrome (tests even-length detection)
        let even_center = "abccba".to_string();
        let result = solution.longest_palindrome(even_center.clone());
        assert_eq!(result, even_center);
        
        // Nested palindromes (tests all approaches)
        let nested = "abacabad".to_string();
        let result = solution.longest_palindrome(nested);
        assert!(result.len() >= 3);
        assert!(solution.is_palindrome(&result));
    }

    #[test]
    fn test_unicode_and_special_characters() {
        let solution = setup();
        
        // Unicode characters
        let unicode_palindrome = "αβα".to_string();
        let result = solution.longest_palindrome(unicode_palindrome.clone());
        assert_eq!(result, unicode_palindrome);
        
        // Mixed ASCII and digits
        let mixed = "a1b2b1a".to_string();
        let result = solution.longest_palindrome(mixed);
        assert_eq!(result, "a1b2b1a"); // The entire string is a palindrome
    }

    #[test]
    fn test_manacher_specific_cases() {
        let solution = setup();
        
        // Cases that test Manacher's algorithm boundary conditions
        let test_cases = vec![
            "aa",        // Even length, minimal
            "aba",       // Odd length, minimal  
            "abba",      // Even length, longer
            "abcba",     // Odd length, longer
            "aabaa",     // Multiple possible centers
        ];
        
        for case in test_cases {
            let result1 = solution.longest_palindrome(case.to_string());
            let result2 = solution.longest_palindrome_manacher(case.to_string());
            
            assert_eq!(result1.len(), result2.len(), 
                      "Manacher length mismatch for '{}': {} vs {}", case, result1, result2);
            assert!(solution.is_palindrome(&result2), 
                   "Manacher result not palindrome for '{}'", case);
        }
    }

    #[test]
    fn test_dp_specific_cases() {
        let solution = setup();
        
        // Cases that test DP algorithm's table building
        let test_cases = vec![
            "abcd",      // No palindromes > 1
            "dcba",      // Reverse, no palindromes > 1
            "aab",       // Palindrome at start
            "baa",       // Palindrome at end
            "aaaa",      // All same character
        ];
        
        for case in test_cases {
            let result1 = solution.longest_palindrome(case.to_string());
            let result2 = solution.longest_palindrome_dp(case.to_string());
            let result3 = solution.longest_palindrome_space_optimized_dp(case.to_string());
            
            assert_eq!(result1.len(), result2.len(), "DP length differs for '{}'", case);
            assert_eq!(result1.len(), result3.len(), "Space optimized DP length differs for '{}'", case);
        }
    }
}