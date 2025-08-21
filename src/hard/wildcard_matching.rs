//! Problem 44: Wildcard Matching
//!
//! Implementation by Marvin Tutt, Caia Tech
//!
//! Given an input string (s) and a pattern (p), implement wildcard pattern matching 
//! with support for '?' and '*'.
//! - '?' Matches any single character.
//! - '*' Matches any sequence of characters (including the empty sequence).
//!
//! The matching should cover the entire input string (not partial).
//!
//! Constraints:
//! - 0 <= s.length, p.length <= 2000
//! - s contains only lowercase English letters.
//! - p contains only lowercase English letters, '?' or '*'.
//!
//! Example 1:
//! Input: s = "aa", p = "a"
//! Output: false
//! Explanation: "a" does not match the entire string "aa".
//!
//! Example 2:
//! Input: s = "aa", p = "*"
//! Output: true
//! Explanation: '*' matches any sequence.
//!
//! Example 3:
//! Input: s = "cb", p = "?a"
//! Output: false
//! Explanation: '?' matches 'c', but the second letter is 'a', which does not match 'b'.

pub struct Solution;

impl Solution {
    /// Approach 1: Dynamic Programming (2D Table) - Optimal
    /// 
    /// Use 2D DP where dp[i][j] represents if s[0..i] matches p[0..j].
    /// 
    /// Time Complexity: O(m * n)
    /// Space Complexity: O(m * n)
    pub fn is_match_dp_2d(s: String, p: String) -> bool {
        let s_chars: Vec<char> = s.chars().collect();
        let p_chars: Vec<char> = p.chars().collect();
        let m = s_chars.len();
        let n = p_chars.len();
        
        let mut dp = vec![vec![false; n + 1]; m + 1];
        
        // Empty string matches empty pattern
        dp[0][0] = true;
        
        // Handle patterns with '*' that can match empty string
        for j in 1..=n {
            if p_chars[j - 1] == '*' {
                dp[0][j] = dp[0][j - 1];
            }
        }
        
        // Fill DP table
        for i in 1..=m {
            for j in 1..=n {
                match p_chars[j - 1] {
                    '*' => {
                        // '*' can match empty sequence or any character(s)
                        dp[i][j] = dp[i][j - 1] ||  // Match empty sequence
                                   dp[i - 1][j];     // Match one or more characters
                    }
                    '?' => {
                        // '?' matches any single character
                        dp[i][j] = dp[i - 1][j - 1];
                    }
                    c => {
                        // Exact character match
                        dp[i][j] = dp[i - 1][j - 1] && s_chars[i - 1] == c;
                    }
                }
            }
        }
        
        dp[m][n]
    }
    
    /// Approach 2: Space-Optimized Dynamic Programming
    /// 
    /// Optimize space by using only two rows instead of full 2D table.
    /// 
    /// Time Complexity: O(m * n)
    /// Space Complexity: O(n)
    pub fn is_match_dp_optimized(s: String, p: String) -> bool {
        let s_chars: Vec<char> = s.chars().collect();
        let p_chars: Vec<char> = p.chars().collect();
        let m = s_chars.len();
        let n = p_chars.len();
        
        let mut prev = vec![false; n + 1];
        let mut curr = vec![false; n + 1];
        
        // Empty string matches empty pattern
        prev[0] = true;
        
        // Handle patterns with '*' that can match empty string
        for j in 1..=n {
            if p_chars[j - 1] == '*' {
                prev[j] = prev[j - 1];
            }
        }
        
        // Process each character in string
        for i in 1..=m {
            curr[0] = false; // Non-empty string doesn't match empty pattern
            
            for j in 1..=n {
                match p_chars[j - 1] {
                    '*' => {
                        curr[j] = curr[j - 1] || prev[j];
                    }
                    '?' => {
                        curr[j] = prev[j - 1];
                    }
                    c => {
                        curr[j] = prev[j - 1] && s_chars[i - 1] == c;
                    }
                }
            }
            
            std::mem::swap(&mut prev, &mut curr);
        }
        
        prev[n]
    }
    
    /// Approach 3: Recursive with Memoization
    /// 
    /// Use recursion with memoization to solve subproblems.
    /// 
    /// Time Complexity: O(m * n)
    /// Space Complexity: O(m * n)
    pub fn is_match_recursive(s: String, p: String) -> bool {
        let s_chars: Vec<char> = s.chars().collect();
        let p_chars: Vec<char> = p.chars().collect();
        let mut memo = std::collections::HashMap::new();
        
        Self::is_match_helper(&s_chars, &p_chars, 0, 0, &mut memo)
    }
    
    fn is_match_helper(
        s: &[char],
        p: &[char],
        i: usize,
        j: usize,
        memo: &mut std::collections::HashMap<(usize, usize), bool>,
    ) -> bool {
        if let Some(&result) = memo.get(&(i, j)) {
            return result;
        }
        
        let result = if j == p.len() {
            i == s.len()
        } else if i == s.len() {
            // Check if remaining pattern consists only of '*'
            p[j..].iter().all(|&c| c == '*')
        } else {
            match p[j] {
                '*' => {
                    Self::is_match_helper(s, p, i, j + 1, memo) ||  // Match empty
                    Self::is_match_helper(s, p, i + 1, j, memo)     // Match one or more
                }
                '?' => {
                    Self::is_match_helper(s, p, i + 1, j + 1, memo)
                }
                c => {
                    s[i] == c && Self::is_match_helper(s, p, i + 1, j + 1, memo)
                }
            }
        };
        
        memo.insert((i, j), result);
        result
    }
    
    /// Approach 4: Greedy Algorithm with Backtracking
    /// 
    /// Use greedy matching with backtracking for '*' characters.
    /// 
    /// Time Complexity: O(m * n) in worst case, but often much faster
    /// Space Complexity: O(1)
    pub fn is_match_greedy(s: String, p: String) -> bool {
        let s_chars: Vec<char> = s.chars().collect();
        let p_chars: Vec<char> = p.chars().collect();
        
        let mut s_idx = 0;
        let mut p_idx = 0;
        let mut star_idx = None;
        let mut s_tmp = 0;
        
        while s_idx < s_chars.len() {
            // Direct match or '?'
            if p_idx < p_chars.len() && (p_chars[p_idx] == '?' || p_chars[p_idx] == s_chars[s_idx]) {
                s_idx += 1;
                p_idx += 1;
            }
            // Encounter '*'
            else if p_idx < p_chars.len() && p_chars[p_idx] == '*' {
                star_idx = Some(p_idx);
                s_tmp = s_idx;
                p_idx += 1;
            }
            // Mismatch, try backtracking
            else if let Some(star_pos) = star_idx {
                p_idx = star_pos + 1;
                s_tmp += 1;
                s_idx = s_tmp;
            }
            // No backtracking possible
            else {
                return false;
            }
        }
        
        // Skip remaining '*' in pattern
        while p_idx < p_chars.len() && p_chars[p_idx] == '*' {
            p_idx += 1;
        }
        
        p_idx == p_chars.len()
    }
    
    /// Approach 5: Iterative DP with Rolling Array
    /// 
    /// Use iterative DP with rolling array optimization.
    /// 
    /// Time Complexity: O(m * n)
    /// Space Complexity: O(n)
    pub fn is_match_iterative_dp(s: String, p: String) -> bool {
        let s_chars: Vec<char> = s.chars().collect();
        let p_chars: Vec<char> = p.chars().collect();
        let m = s_chars.len();
        let n = p_chars.len();
        
        if n == 0 {
            return m == 0;
        }
        
        let mut dp = vec![false; n + 1];
        dp[0] = true;
        
        // Initialize for patterns that can match empty string
        for j in 1..=n {
            dp[j] = dp[j - 1] && p_chars[j - 1] == '*';
        }
        
        for i in 1..=m {
            let mut new_dp = vec![false; n + 1];
            
            for j in 1..=n {
                match p_chars[j - 1] {
                    '*' => {
                        new_dp[j] = new_dp[j - 1] || dp[j];
                    }
                    '?' => {
                        new_dp[j] = dp[j - 1];
                    }
                    c => {
                        new_dp[j] = dp[j - 1] && s_chars[i - 1] == c;
                    }
                }
            }
            
            dp = new_dp;
        }
        
        dp[n]
    }
    
    /// Approach 6: Pattern Preprocessing with DP
    /// 
    /// Preprocess pattern to remove redundant '*' and then apply DP.
    /// 
    /// Time Complexity: O(m * n)
    /// Space Complexity: O(n)
    pub fn is_match_preprocessed(s: String, p: String) -> bool {
        // Remove consecutive '*' characters
        let mut processed_pattern = String::new();
        let mut prev_char: Option<char> = None;
        
        for ch in p.chars() {
            if ch != '*' || prev_char != Some('*') {
                processed_pattern.push(ch);
                prev_char = Some(ch);
            }
        }
        
        // Use the space-optimized DP approach on preprocessed pattern
        Self::is_match_dp_optimized(s, processed_pattern)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_no_match() {
        assert_eq!(Solution::is_match_dp_2d("aa".to_string(), "a".to_string()), false);
    }
    
    #[test]
    fn test_star_matches_all() {
        assert_eq!(Solution::is_match_dp_optimized("aa".to_string(), "*".to_string()), true);
    }
    
    #[test]
    fn test_question_mark_mismatch() {
        assert_eq!(Solution::is_match_recursive("cb".to_string(), "?a".to_string()), false);
    }
    
    #[test]
    fn test_exact_match() {
        assert_eq!(Solution::is_match_greedy("adceb".to_string(), "*a*b*".to_string()), true);
    }
    
    #[test]
    fn test_no_match_complex() {
        assert_eq!(Solution::is_match_iterative_dp("acdcb".to_string(), "a*c?b".to_string()), false);
    }
    
    #[test]
    fn test_empty_strings() {
        assert_eq!(Solution::is_match_preprocessed("".to_string(), "".to_string()), true);
        assert_eq!(Solution::is_match_dp_2d("".to_string(), "*".to_string()), true);
        assert_eq!(Solution::is_match_dp_optimized("a".to_string(), "".to_string()), false);
    }
    
    #[test]
    fn test_single_characters() {
        assert_eq!(Solution::is_match_recursive("a".to_string(), "?".to_string()), true);
        assert_eq!(Solution::is_match_greedy("a".to_string(), "b".to_string()), false);
    }
    
    #[test]
    fn test_multiple_stars() {
        assert_eq!(Solution::is_match_iterative_dp("ho".to_string(), "**ho".to_string()), true);
        assert_eq!(Solution::is_match_preprocessed("ho".to_string(), "ho**".to_string()), true);
    }
    
    #[test]
    fn test_complex_pattern() {
        assert_eq!(Solution::is_match_dp_2d("abefcdgiescdfimde".to_string(), "ab*cd?i*de".to_string()), true);
    }
    
    #[test]
    fn test_star_at_beginning() {
        assert_eq!(Solution::is_match_dp_optimized("anything".to_string(), "*ing".to_string()), true);
        assert_eq!(Solution::is_match_recursive("anything".to_string(), "*xyz".to_string()), false);
    }
    
    #[test]
    fn test_alternating_pattern() {
        assert_eq!(Solution::is_match_greedy("abcd".to_string(), "a?c?".to_string()), true);
        assert_eq!(Solution::is_match_iterative_dp("abcd".to_string(), "a?x?".to_string()), false);
    }
    
    #[test]
    fn test_long_strings() {
        let s = "abcdefghijklmnopqrstuvwxyz".to_string();
        let p = "*".to_string();
        assert_eq!(Solution::is_match_preprocessed(s, p), true);
        
        let s2 = "aaaaaaaaaa".to_string();
        let p2 = "a*a*a*a*a*a*a*a*a*a*".to_string();
        assert_eq!(Solution::is_match_dp_2d(s2, p2), true);
    }
    
    #[test]
    fn test_edge_case_only_stars() {
        assert_eq!(Solution::is_match_dp_optimized("abc".to_string(), "***".to_string()), true);
        assert_eq!(Solution::is_match_recursive("".to_string(), "***".to_string()), true);
    }
    
    #[test]
    fn test_no_wildcards() {
        assert_eq!(Solution::is_match_greedy("hello".to_string(), "hello".to_string()), true);
        assert_eq!(Solution::is_match_iterative_dp("hello".to_string(), "world".to_string()), false);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            ("aa", "a"),
            ("aa", "*"),
            ("cb", "?a"),
            ("adceb", "*a*b*"),
            ("acdcb", "a*c?b"),
            ("", ""),
            ("", "*"),
            ("a", ""),
            ("a", "?"),
            ("a", "b"),
            ("ho", "**ho"),
            ("ho", "ho**"),
            ("abefcdgiescdfimde", "ab*cd?i*de"),
            ("anything", "*ing"),
            ("anything", "*xyz"),
            ("abcd", "a?c?"),
            ("abcd", "a?x?"),
            ("abc", "***"),
            ("", "***"),
            ("hello", "hello"),
            ("hello", "world"),
        ];
        
        for (s, p) in test_cases {
            let s = s.to_string();
            let p = p.to_string();
            
            let result1 = Solution::is_match_dp_2d(s.clone(), p.clone());
            let result2 = Solution::is_match_dp_optimized(s.clone(), p.clone());
            let result3 = Solution::is_match_recursive(s.clone(), p.clone());
            let result4 = Solution::is_match_greedy(s.clone(), p.clone());
            let result5 = Solution::is_match_iterative_dp(s.clone(), p.clone());
            let result6 = Solution::is_match_preprocessed(s.clone(), p.clone());
            
            assert_eq!(result1, result2, "DP2D vs DPOptimized mismatch for '{}' and '{}'", s, p);
            assert_eq!(result2, result3, "DPOptimized vs Recursive mismatch for '{}' and '{}'", s, p);
            assert_eq!(result3, result4, "Recursive vs Greedy mismatch for '{}' and '{}'", s, p);
            assert_eq!(result4, result5, "Greedy vs IterativeDP mismatch for '{}' and '{}'", s, p);
            assert_eq!(result5, result6, "IterativeDP vs Preprocessed mismatch for '{}' and '{}'", s, p);
        }
    }
}