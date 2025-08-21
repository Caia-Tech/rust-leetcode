//! Problem 91: Decode Ways
//! 
//! A message containing letters from A-Z can be encoded into numbers using the following mapping:
//! 'A' -> "1", 'B' -> "2", ..., 'Z' -> "26"
//! 
//! To decode an encoded message, all the digits must be grouped then mapped back into letters 
//! using the reverse of the mapping above (there may be multiple ways).
//! 
//! Given a string s containing only digits, return the number of ways to decode it.
//! 
//! Example 1:
//! Input: s = "12"
//! Output: 2
//! Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).
//! 
//! Example 2:
//! Input: s = "226"
//! Output: 3
//! Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
//! 
//! Example 3:
//! Input: s = "06"
//! Output: 0
//! Explanation: "06" cannot be mapped to "F" since "6" is different from "06".

use std::collections::HashMap;

pub struct Solution;

impl Solution {
    /// Approach 1: Dynamic Programming (Bottom-up)
    /// 
    /// dp[i] represents the number of ways to decode s[0..i].
    /// For each position, we can decode as single digit (if valid) or two digits (if valid).
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    pub fn num_decodings_dp(&self, s: String) -> i32 {
        if s.is_empty() || s.starts_with('0') {
            return 0;
        }
        
        let n = s.len();
        let chars: Vec<char> = s.chars().collect();
        let mut dp = vec![0; n + 1];
        
        dp[0] = 1; // Empty string has one way
        dp[1] = if chars[0] != '0' { 1 } else { 0 };
        
        for i in 2..=n {
            // Single digit decode
            if chars[i - 1] != '0' {
                dp[i] += dp[i - 1];
            }
            
            // Two digit decode
            let two_digit = (chars[i - 2] as u8 - b'0') * 10 + (chars[i - 1] as u8 - b'0');
            if two_digit >= 10 && two_digit <= 26 {
                dp[i] += dp[i - 2];
            }
        }
        
        dp[n]
    }
    
    /// Approach 2: Space-Optimized DP
    /// 
    /// Since we only need the previous two values, we can reduce space complexity to O(1).
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn num_decodings_optimized(&self, s: String) -> i32 {
        if s.is_empty() || s.starts_with('0') {
            return 0;
        }
        
        let chars: Vec<char> = s.chars().collect();
        let mut prev2 = 1; // dp[i-2]
        let mut prev1 = if chars[0] != '0' { 1 } else { 0 }; // dp[i-1]
        
        for i in 1..chars.len() {
            let mut current = 0;
            
            // Single digit decode
            if chars[i] != '0' {
                current += prev1;
            }
            
            // Two digit decode
            let two_digit = (chars[i - 1] as u8 - b'0') * 10 + (chars[i] as u8 - b'0');
            if two_digit >= 10 && two_digit <= 26 {
                current += prev2;
            }
            
            prev2 = prev1;
            prev1 = current;
        }
        
        prev1
    }
    
    /// Approach 3: Recursive with Memoization
    /// 
    /// Top-down approach using recursion with memoization.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n) for memoization and recursion stack
    pub fn num_decodings_memo(&self, s: String) -> i32 {
        if s.is_empty() {
            return 0;
        }
        
        let mut memo = HashMap::new();
        self.decode_helper(&s, 0, &mut memo)
    }
    
    fn decode_helper(&self, s: &str, index: usize, memo: &mut HashMap<usize, i32>) -> i32 {
        if index == s.len() {
            return 1;
        }
        
        if let Some(&cached) = memo.get(&index) {
            return cached;
        }
        
        let chars: Vec<char> = s.chars().collect();
        
        // If current character is '0', no valid decoding
        if chars[index] == '0' {
            memo.insert(index, 0);
            return 0;
        }
        
        let mut result = 0;
        
        // Single digit decode
        result += self.decode_helper(s, index + 1, memo);
        
        // Two digit decode
        if index + 1 < s.len() {
            let two_digit = (chars[index] as u8 - b'0') * 10 + (chars[index + 1] as u8 - b'0');
            if two_digit <= 26 {
                result += self.decode_helper(s, index + 2, memo);
            }
        }
        
        memo.insert(index, result);
        result
    }
    
    /// Approach 4: Iterative with State Machine
    /// 
    /// Models the problem as a state machine where each state represents
    /// the validity of the current decoding state.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn num_decodings_state_machine(&self, s: String) -> i32 {
        if s.is_empty() || s.starts_with('0') {
            return 0;
        }
        
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();
        
        // State: (ways_ending_with_single_digit, ways_ending_with_double_digit)
        let mut single = 1;
        let mut double = 0;
        
        for i in 0..n {
            let mut new_single = 0;
            let mut new_double = 0;
            
            let digit = chars[i] as u8 - b'0';
            
            // Can form single digit (1-9)
            if digit >= 1 && digit <= 9 {
                new_single = single + double;
            }
            
            // Can form double digit with previous character
            if i > 0 {
                let prev_digit = chars[i - 1] as u8 - b'0';
                let two_digit = prev_digit * 10 + digit;
                
                if two_digit >= 10 && two_digit <= 26 {
                    new_double = single;
                }
            }
            
            single = new_single;
            double = new_double;
        }
        
        single + double
    }
    
    /// Approach 5: Pattern-based DP
    /// 
    /// Recognizes common patterns in the string to optimize computation.
    /// Handles consecutive zeros and specific digit patterns efficiently.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn num_decodings_pattern(&self, s: String) -> i32 {
        if s.is_empty() || s.starts_with('0') {
            return 0;
        }
        
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();
        
        let mut dp = vec![0; n + 1];
        dp[0] = 1;
        
        for i in 1..=n {
            let current_char = chars[i - 1];
            
            // Handle current character
            match current_char {
                '0' => {
                    // Can only be decoded with previous character
                    if i > 1 {
                        let prev_char = chars[i - 2];
                        if prev_char == '1' || prev_char == '2' {
                            dp[i] = dp[i - 2];
                        } else {
                            return 0; // Invalid sequence like 30, 40, etc.
                        }
                    } else {
                        return 0; // String starts with 0
                    }
                }
                '1'..='9' => {
                    // Can always be decoded as single digit
                    dp[i] = dp[i - 1];
                    
                    // Check if can be decoded with previous character
                    if i > 1 {
                        let prev_char = chars[i - 2];
                        let two_digit = (prev_char as u8 - b'0') * 10 + (current_char as u8 - b'0');
                        
                        if two_digit >= 10 && two_digit <= 26 {
                            dp[i] += dp[i - 2];
                        }
                    }
                }
                _ => return 0, // Invalid character
            }
        }
        
        dp[n]
    }
    
    /// Approach 6: Fibonacci-like Approach
    /// 
    /// Recognizes that valid decoding follows a Fibonacci-like pattern
    /// where each position depends on the previous two positions.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn num_decodings_fibonacci(&self, s: String) -> i32 {
        if s.is_empty() || s.chars().next().unwrap() == '0' {
            return 0;
        }
        
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();
        
        if n == 1 {
            return 1;
        }
        
        // Initialize for first two positions
        let mut prev_prev = 1; // f(0)
        let mut prev = 1;      // f(1)
        
        for i in 2..=n {
            let mut current = 0;
            
            // Check single digit
            if chars[i - 1] != '0' {
                current += prev;
            }
            
            // Check two digits
            let first_digit = chars[i - 2] as u8 - b'0';
            let second_digit = chars[i - 1] as u8 - b'0';
            let two_digit_num = first_digit * 10 + second_digit;
            
            if first_digit != 0 && two_digit_num <= 26 {
                current += prev_prev;
            }
            
            prev_prev = prev;
            prev = current;
        }
        
        prev
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dp() {
        let solution = Solution;
        
        assert_eq!(solution.num_decodings_dp("12".to_string()), 2);
        assert_eq!(solution.num_decodings_dp("226".to_string()), 3);
        assert_eq!(solution.num_decodings_dp("06".to_string()), 0);
        assert_eq!(solution.num_decodings_dp("0".to_string()), 0);
        assert_eq!(solution.num_decodings_dp("10".to_string()), 1);
        assert_eq!(solution.num_decodings_dp("27".to_string()), 1);
        assert_eq!(solution.num_decodings_dp("1".to_string()), 1);
    }
    
    #[test]
    fn test_optimized() {
        let solution = Solution;
        
        assert_eq!(solution.num_decodings_optimized("12".to_string()), 2);
        assert_eq!(solution.num_decodings_optimized("226".to_string()), 3);
        assert_eq!(solution.num_decodings_optimized("06".to_string()), 0);
        assert_eq!(solution.num_decodings_optimized("0".to_string()), 0);
        assert_eq!(solution.num_decodings_optimized("10".to_string()), 1);
        assert_eq!(solution.num_decodings_optimized("27".to_string()), 1);
        assert_eq!(solution.num_decodings_optimized("1".to_string()), 1);
    }
    
    #[test]
    fn test_memo() {
        let solution = Solution;
        
        assert_eq!(solution.num_decodings_memo("12".to_string()), 2);
        assert_eq!(solution.num_decodings_memo("226".to_string()), 3);
        assert_eq!(solution.num_decodings_memo("06".to_string()), 0);
        assert_eq!(solution.num_decodings_memo("0".to_string()), 0);
        assert_eq!(solution.num_decodings_memo("10".to_string()), 1);
        assert_eq!(solution.num_decodings_memo("27".to_string()), 1);
        assert_eq!(solution.num_decodings_memo("1".to_string()), 1);
    }
    
    #[test]
    fn test_state_machine() {
        let solution = Solution;
        
        assert_eq!(solution.num_decodings_state_machine("12".to_string()), 2);
        assert_eq!(solution.num_decodings_state_machine("226".to_string()), 3);
        assert_eq!(solution.num_decodings_state_machine("06".to_string()), 0);
        assert_eq!(solution.num_decodings_state_machine("0".to_string()), 0);
        assert_eq!(solution.num_decodings_state_machine("10".to_string()), 1);
        assert_eq!(solution.num_decodings_state_machine("27".to_string()), 1);
        assert_eq!(solution.num_decodings_state_machine("1".to_string()), 1);
    }
    
    #[test]
    fn test_pattern() {
        let solution = Solution;
        
        assert_eq!(solution.num_decodings_pattern("12".to_string()), 2);
        assert_eq!(solution.num_decodings_pattern("226".to_string()), 3);
        assert_eq!(solution.num_decodings_pattern("06".to_string()), 0);
        assert_eq!(solution.num_decodings_pattern("0".to_string()), 0);
        assert_eq!(solution.num_decodings_pattern("10".to_string()), 1);
        assert_eq!(solution.num_decodings_pattern("27".to_string()), 1);
        assert_eq!(solution.num_decodings_pattern("1".to_string()), 1);
    }
    
    #[test]
    fn test_fibonacci() {
        let solution = Solution;
        
        assert_eq!(solution.num_decodings_fibonacci("12".to_string()), 2);
        assert_eq!(solution.num_decodings_fibonacci("226".to_string()), 3);
        assert_eq!(solution.num_decodings_fibonacci("06".to_string()), 0);
        assert_eq!(solution.num_decodings_fibonacci("0".to_string()), 0);
        assert_eq!(solution.num_decodings_fibonacci("10".to_string()), 1);
        assert_eq!(solution.num_decodings_fibonacci("27".to_string()), 1);
        assert_eq!(solution.num_decodings_fibonacci("1".to_string()), 1);
    }
    
    #[test]
    fn test_edge_cases() {
        let solution = Solution;
        
        // Empty string
        assert_eq!(solution.num_decodings_dp("".to_string()), 0);
        
        // Leading zeros
        assert_eq!(solution.num_decodings_dp("01".to_string()), 0);
        assert_eq!(solution.num_decodings_dp("001".to_string()), 0);
        
        // Multiple zeros
        assert_eq!(solution.num_decodings_dp("100".to_string()), 0);
        assert_eq!(solution.num_decodings_dp("1001".to_string()), 0);
        
        // Valid zeros
        assert_eq!(solution.num_decodings_dp("101".to_string()), 1);
        assert_eq!(solution.num_decodings_dp("1201".to_string()), 1);
        
        // Invalid two-digit numbers
        assert_eq!(solution.num_decodings_dp("30".to_string()), 0);
        assert_eq!(solution.num_decodings_dp("40".to_string()), 0);
        
        // Boundary cases
        assert_eq!(solution.num_decodings_dp("26".to_string()), 2);
        assert_eq!(solution.num_decodings_dp("27".to_string()), 1);
        
        // Long valid sequences
        assert_eq!(solution.num_decodings_dp("1111".to_string()), 5);
        assert_eq!(solution.num_decodings_dp("1212".to_string()), 5);
    }
    
    #[test]
    fn test_complex_cases() {
        let solution = Solution;
        
        // Alternating valid/invalid patterns
        assert_eq!(solution.num_decodings_dp("1203".to_string()), 1);
        assert_eq!(solution.num_decodings_dp("2101".to_string()), 1);
        
        // Multiple consecutive valid two-digit numbers
        assert_eq!(solution.num_decodings_dp("111111".to_string()), 13);
        
        // Mixed patterns
        assert_eq!(solution.num_decodings_dp("2611055971756562".to_string()), 4);
        
        // Edge case with 10 and 20
        assert_eq!(solution.num_decodings_dp("1020".to_string()), 1);
        assert_eq!(solution.num_decodings_dp("10203".to_string()), 1);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let solution = Solution;
        
        let test_cases = vec![
            "12",
            "226",
            "06",
            "0",
            "10",
            "27",
            "1",
            "",
            "01",
            "100",
            "101",
            "120",
            "30",
            "26",
            "1111",
            "1212",
            "1203",
            "2101",
            "1020",
        ];
        
        for case in test_cases {
            let s = case.to_string();
            
            let dp = solution.num_decodings_dp(s.clone());
            let optimized = solution.num_decodings_optimized(s.clone());
            let memo = solution.num_decodings_memo(s.clone());
            let state_machine = solution.num_decodings_state_machine(s.clone());
            let pattern = solution.num_decodings_pattern(s.clone());
            let fibonacci = solution.num_decodings_fibonacci(s.clone());
            
            assert_eq!(dp, optimized, "DP and optimized differ for '{}'", case);
            assert_eq!(dp, memo, "DP and memo differ for '{}'", case);
            assert_eq!(dp, state_machine, "DP and state machine differ for '{}'", case);
            assert_eq!(dp, pattern, "DP and pattern differ for '{}'", case);
            assert_eq!(dp, fibonacci, "DP and fibonacci differ for '{}'", case);
        }
    }
    
    #[test]
    fn test_performance_patterns() {
        let solution = Solution;
        
        // Test with repeated patterns that could cause exponential growth
        let repeated_ones = "1".repeat(20);
        let result = solution.num_decodings_optimized(repeated_ones);
        // This should be the 21st Fibonacci number
        assert!(result > 0);
        
        // Test with pattern that has many valid combinations
        let pattern = "1212121212";
        let dp_result = solution.num_decodings_dp(pattern.to_string());
        let opt_result = solution.num_decodings_optimized(pattern.to_string());
        assert_eq!(dp_result, opt_result);
        assert_eq!(dp_result, 89); // Expected result for this pattern
    }
    
    #[test]
    fn test_special_sequences() {
        let solution = Solution;
        
        // Test sequences with all valid two-digit combinations
        assert_eq!(solution.num_decodings_dp("11".to_string()), 2);
        assert_eq!(solution.num_decodings_dp("22".to_string()), 2);
        assert_eq!(solution.num_decodings_dp("23".to_string()), 2);
        assert_eq!(solution.num_decodings_dp("24".to_string()), 2);
        assert_eq!(solution.num_decodings_dp("25".to_string()), 2);
        assert_eq!(solution.num_decodings_dp("26".to_string()), 2);
        
        // Test boundary of valid two-digit range
        assert_eq!(solution.num_decodings_dp("9".to_string()), 1);
        assert_eq!(solution.num_decodings_dp("99".to_string()), 1);
        
        // Test with zeros in different positions
        assert_eq!(solution.num_decodings_dp("2020".to_string()), 1);
        assert_eq!(solution.num_decodings_dp("1010".to_string()), 1);
    }
}