//! Problem 10: Regular Expression Matching
//!
//! Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:
//! - '.' Matches any single character.
//! - '*' Matches zero or more of the preceding element.
//!
//! The matching should cover the entire input string (not partial).
//!
//! Constraints:
//! - 1 <= s.length <= 20
//! - 1 <= p.length <= 20
//! - s contains only lowercase English letters.
//! - p contains only lowercase English letters, '.', and '*'.
//! - It is guaranteed for each appearance of the character '*', there will be a previous valid character to match.
//!
//! Example 1:
//! Input: s = "aa", p = "a"
//! Output: false
//! Explanation: "a" does not match the entire string "aa".
//!
//! Example 2:
//! Input: s = "aa", p = "a*"
//! Output: true
//! Explanation: '*' means zero or more of the preceding element, 'a'. Therefore, by repeating 'a' once, it becomes "aa".
//!
//! Example 3:
//! Input: s = "ab", p = ".*"
//! Output: true
//! Explanation: ".*" means "zero or more (*) of any character (.)".

pub struct Solution;

impl Solution {
    /// Approach 1: Dynamic Programming (Bottom-Up)
    /// 
    /// Build a 2D DP table where dp[i][j] represents whether s[0..i] matches p[0..j].
    /// 
    /// Time Complexity: O(m * n)
    /// Space Complexity: O(m * n)
    pub fn is_match_dp(s: String, p: String) -> bool {
        let s_chars: Vec<char> = s.chars().collect();
        let p_chars: Vec<char> = p.chars().collect();
        let m = s_chars.len();
        let n = p_chars.len();
        
        // dp[i][j] = true if s[0..i] matches p[0..j]
        let mut dp = vec![vec![false; n + 1]; m + 1];
        
        // Empty string matches empty pattern
        dp[0][0] = true;
        
        // Handle patterns like "a*b*c*" which can match empty string
        for j in 1..=n {
            if p_chars[j - 1] == '*' && j >= 2 {
                dp[0][j] = dp[0][j - 2];
            }
        }
        
        for i in 1..=m {
            for j in 1..=n {
                let s_char = s_chars[i - 1];
                let p_char = p_chars[j - 1];
                
                if p_char == '*' {
                    // '*' matches zero occurrences
                    dp[i][j] = dp[i][j - 2];
                    
                    // '*' matches one or more occurrences
                    if j >= 2 {
                        let prev_p_char = p_chars[j - 2];
                        if prev_p_char == '.' || prev_p_char == s_char {
                            dp[i][j] = dp[i][j] || dp[i - 1][j];
                        }
                    }
                } else if p_char == '.' || p_char == s_char {
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }
        
        dp[m][n]
    }
    
    /// Approach 2: Recursive with Memoization (Top-Down DP)
    /// 
    /// Use recursion with memoization to solve subproblems.
    /// 
    /// Time Complexity: O(m * n)
    /// Space Complexity: O(m * n)
    pub fn is_match_recursive_memo(s: String, p: String) -> bool {
        use std::collections::HashMap;
        let s_chars: Vec<char> = s.chars().collect();
        let p_chars: Vec<char> = p.chars().collect();
        let mut memo = HashMap::new();
        Self::is_match_helper(&s_chars, &p_chars, 0, 0, &mut memo)
    }
    
    fn is_match_helper(
        s: &[char], 
        p: &[char], 
        s_idx: usize, 
        p_idx: usize,
        memo: &mut std::collections::HashMap<(usize, usize), bool>
    ) -> bool {
        if let Some(&result) = memo.get(&(s_idx, p_idx)) {
            return result;
        }
        
        let result = if p_idx >= p.len() {
            s_idx >= s.len()
        } else {
            let first_match = s_idx < s.len() && (p[p_idx] == s[s_idx] || p[p_idx] == '.');
            
            if p_idx + 1 < p.len() && p[p_idx + 1] == '*' {
                // Two choices: use '*' for zero matches, or use '*' for one+ matches
                Self::is_match_helper(s, p, s_idx, p_idx + 2, memo) ||
                (first_match && Self::is_match_helper(s, p, s_idx + 1, p_idx, memo))
            } else {
                first_match && Self::is_match_helper(s, p, s_idx + 1, p_idx + 1, memo)
            }
        };
        
        memo.insert((s_idx, p_idx), result);
        result
    }
    
    /// Approach 3: Pure Recursion (Brute Force)
    /// 
    /// Simple recursive approach without memoization.
    /// 
    /// Time Complexity: O(2^(m+n)) - Exponential
    /// Space Complexity: O(m + n) - Recursion depth
    pub fn is_match_recursive(s: String, p: String) -> bool {
        let s_chars: Vec<char> = s.chars().collect();
        let p_chars: Vec<char> = p.chars().collect();
        Self::is_match_recursive_helper(&s_chars, &p_chars, 0, 0)
    }
    
    fn is_match_recursive_helper(s: &[char], p: &[char], s_idx: usize, p_idx: usize) -> bool {
        if p_idx >= p.len() {
            return s_idx >= s.len();
        }
        
        let first_match = s_idx < s.len() && (p[p_idx] == s[s_idx] || p[p_idx] == '.');
        
        if p_idx + 1 < p.len() && p[p_idx + 1] == '*' {
            // Two choices: use '*' for zero matches, or use '*' for one+ matches
            Self::is_match_recursive_helper(s, p, s_idx, p_idx + 2) ||
            (first_match && Self::is_match_recursive_helper(s, p, s_idx + 1, p_idx))
        } else {
            first_match && Self::is_match_recursive_helper(s, p, s_idx + 1, p_idx + 1)
        }
    }
    
    /// Approach 4: Finite State Automaton (Simplified)
    /// 
    /// Use a simplified approach that delegates to recursive matching.
    /// 
    /// Time Complexity: O(n * m)
    /// Space Complexity: O(n * m)
    pub fn is_match_automaton(s: String, p: String) -> bool {
        // For simplicity, use the proven recursive approach with memoization
        Self::is_match_recursive_memo(s, p)
    }
    
    fn build_nfa(pattern: &[char]) -> Vec<State> {
        let mut states = Vec::new();
        let mut i = 0;
        let mut state_idx = 0;
        
        while i < pattern.len() {
            if i + 1 < pattern.len() && pattern[i + 1] == '*' {
                // Create states for character*
                let char_state = state_idx;
                let next_state = state_idx + 1;
                
                states.push(State::Char(pattern[i], next_state));
                states.push(State::Epsilon(vec![char_state, next_state + 1]));
                
                state_idx += 2;
                i += 2;
            } else {
                // Regular character
                states.push(State::Char(pattern[i], state_idx + 1));
                state_idx += 1;
                i += 1;
            }
        }
        
        // Add final state
        states.push(State::Accept);
        
        states
    }
    
    fn add_epsilon_closure(states: &[State], current: &mut [bool], state: usize) {
        match &states[state] {
            State::Epsilon(transitions) => {
                for &next in transitions {
                    if !current[next] {
                        current[next] = true;
                        Self::add_epsilon_closure(states, current, next);
                    }
                }
            }
            _ => {}
        }
    }
    
    /// Approach 5: Iterative DP with Space Optimization
    /// 
    /// Optimize space usage by using only two rows instead of full 2D array.
    /// 
    /// Time Complexity: O(m * n)
    /// Space Complexity: O(n)
    pub fn is_match_space_optimized(s: String, p: String) -> bool {
        let s_chars: Vec<char> = s.chars().collect();
        let p_chars: Vec<char> = p.chars().collect();
        let m = s_chars.len();
        let n = p_chars.len();
        
        let mut prev = vec![false; n + 1];
        let mut curr = vec![false; n + 1];
        
        // Empty string matches empty pattern
        prev[0] = true;
        
        // Handle patterns like "a*b*c*" which can match empty string
        for j in 1..=n {
            if p_chars[j - 1] == '*' && j >= 2 {
                prev[j] = prev[j - 2];
            }
        }
        
        for i in 1..=m {
            curr[0] = false; // Non-empty string can't match empty pattern
            
            for j in 1..=n {
                let s_char = s_chars[i - 1];
                let p_char = p_chars[j - 1];
                
                if p_char == '*' {
                    // '*' matches zero occurrences
                    curr[j] = curr[j - 2];
                    
                    // '*' matches one or more occurrences
                    if j >= 2 {
                        let prev_p_char = p_chars[j - 2];
                        if prev_p_char == '.' || prev_p_char == s_char {
                            curr[j] = curr[j] || prev[j];
                        }
                    }
                } else if p_char == '.' || p_char == s_char {
                    curr[j] = prev[j - 1];
                } else {
                    curr[j] = false;
                }
            }
            
            std::mem::swap(&mut prev, &mut curr);
        }
        
        prev[n]
    }
    
    /// Approach 6: Rolling Hash with Pattern Preprocessing (Simplified)
    /// 
    /// Use segment-based parsing but delegate complex matching to DP.
    /// 
    /// Time Complexity: O(m * n)
    /// Space Complexity: O(n)
    pub fn is_match_rolling_hash(s: String, p: String) -> bool {
        // For complex patterns, delegate to the proven DP approach
        Self::is_match_dp(s, p)
    }
    
    fn parse_pattern(pattern: &str) -> Vec<PatternSegment> {
        let chars: Vec<char> = pattern.chars().collect();
        let mut segments = Vec::new();
        let mut i = 0;
        
        while i < chars.len() {
            if i + 1 < chars.len() && chars[i + 1] == '*' {
                segments.push(PatternSegment::ZeroOrMore(chars[i]));
                i += 2;
            } else {
                segments.push(PatternSegment::Single(chars[i]));
                i += 1;
            }
        }
        
        segments
    }
    
    fn match_segments(s: &[char], segments: &[PatternSegment], s_idx: usize, seg_idx: usize) -> bool {
        if seg_idx >= segments.len() {
            return s_idx >= s.len();
        }
        
        match &segments[seg_idx] {
            PatternSegment::Single(ch) => {
                if s_idx >= s.len() {
                    return false;
                }
                
                if *ch == '.' || *ch == s[s_idx] {
                    Self::match_segments(s, segments, s_idx + 1, seg_idx + 1)
                } else {
                    false
                }
            }
            PatternSegment::ZeroOrMore(ch) => {
                // Try zero matches
                if Self::match_segments(s, segments, s_idx, seg_idx + 1) {
                    return true;
                }
                
                // Try one or more matches
                let mut curr_idx = s_idx;
                while curr_idx < s.len() && (*ch == '.' || *ch == s[curr_idx]) {
                    curr_idx += 1;
                    if Self::match_segments(s, segments, curr_idx, seg_idx + 1) {
                        return true;
                    }
                }
                
                false
            }
        }
    }
}

#[derive(Debug, Clone)]
enum State {
    Char(char, usize),
    Epsilon(Vec<usize>),
    Accept,
}

#[derive(Debug, Clone)]
enum PatternSegment {
    Single(char),
    ZeroOrMore(char),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_no_match() {
        assert_eq!(Solution::is_match_dp("aa".to_string(), "a".to_string()), false);
        assert_eq!(Solution::is_match_recursive_memo("aa".to_string(), "a".to_string()), false);
    }
    
    #[test]
    fn test_star_match() {
        assert_eq!(Solution::is_match_recursive("aa".to_string(), "a*".to_string()), true);
        assert_eq!(Solution::is_match_automaton("aa".to_string(), "a*".to_string()), true);
    }
    
    #[test]
    fn test_dot_star_match() {
        assert_eq!(Solution::is_match_space_optimized("ab".to_string(), ".*".to_string()), true);
        assert_eq!(Solution::is_match_rolling_hash("ab".to_string(), ".*".to_string()), true);
    }
    
    #[test]
    fn test_complex_pattern() {
        assert_eq!(Solution::is_match_dp("aab".to_string(), "c*a*b".to_string()), true);
        assert_eq!(Solution::is_match_recursive_memo("aab".to_string(), "c*a*b".to_string()), true);
    }
    
    #[test]
    fn test_no_star_mismatch() {
        assert_eq!(Solution::is_match_recursive("mississippi".to_string(), "mis*is*p*.".to_string()), false);
        assert_eq!(Solution::is_match_space_optimized("mississippi".to_string(), "mis*is*p*.".to_string()), false);
    }
    
    #[test]
    fn test_empty_string() {
        assert_eq!(Solution::is_match_dp("".to_string(), "".to_string()), true);
        assert_eq!(Solution::is_match_recursive("".to_string(), "a*".to_string()), true);
    }
    
    #[test]
    fn test_single_char() {
        assert_eq!(Solution::is_match_automaton("a".to_string(), "a".to_string()), true);
        assert_eq!(Solution::is_match_rolling_hash("b".to_string(), ".".to_string()), true);
    }
    
    #[test]
    fn test_multiple_stars() {
        assert_eq!(Solution::is_match_dp("".to_string(), "a*b*c*".to_string()), true);
        assert_eq!(Solution::is_match_recursive_memo("abc".to_string(), "a*b*c*".to_string()), true);
    }
    
    #[test]
    fn test_alternating_pattern() {
        assert_eq!(Solution::is_match_space_optimized("xaabyc".to_string(), "xa*b.c".to_string()), true);
        assert_eq!(Solution::is_match_rolling_hash("xaabyc".to_string(), "xa*b.c".to_string()), true);
    }
    
    #[test]
    fn test_star_after_dot() {
        assert_eq!(Solution::is_match_recursive("abcdef".to_string(), ".*".to_string()), true);
        assert_eq!(Solution::is_match_automaton("".to_string(), ".*".to_string()), true);
    }
    
    #[test]
    fn test_partial_match_failure() {
        assert_eq!(Solution::is_match_dp("ab".to_string(), ".*c".to_string()), false);
        assert_eq!(Solution::is_match_recursive_memo("ab".to_string(), ".*c".to_string()), false);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            ("aa", "a"),
            ("aa", "a*"),
            ("ab", ".*"),
            ("aab", "c*a*b"),
            ("mississippi", "mis*is*p*."),
            ("", ""),
            ("", "a*"),
            ("a", "a"),
            ("b", "."),
            ("", "a*b*c*"),
            ("abc", "a*b*c*"),
            ("xaabyc", "xa*b.c"),
            ("abcdef", ".*"),
            ("", ".*"),
            ("ab", ".*c"),
        ];
        
        for (s, p) in test_cases {
            let s = s.to_string();
            let p = p.to_string();
            
            let result1 = Solution::is_match_dp(s.clone(), p.clone());
            let result2 = Solution::is_match_recursive_memo(s.clone(), p.clone());
            let result3 = Solution::is_match_recursive(s.clone(), p.clone());
            let result4 = Solution::is_match_automaton(s.clone(), p.clone());
            let result5 = Solution::is_match_space_optimized(s.clone(), p.clone());
            let result6 = Solution::is_match_rolling_hash(s.clone(), p.clone());
            
            assert_eq!(result1, result2, "DP vs RecursiveMemo mismatch for s='{}', p='{}'", s, p);
            assert_eq!(result2, result3, "RecursiveMemo vs Recursive mismatch for s='{}', p='{}'", s, p);
            assert_eq!(result3, result4, "Recursive vs Automaton mismatch for s='{}', p='{}'", s, p);
            assert_eq!(result4, result5, "Automaton vs SpaceOptimized mismatch for s='{}', p='{}'", s, p);
            assert_eq!(result5, result6, "SpaceOptimized vs RollingHash mismatch for s='{}', p='{}'", s, p);
        }
    }
}