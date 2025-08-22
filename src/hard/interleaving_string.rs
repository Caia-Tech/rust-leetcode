//! Problem 97: Interleaving String
//!
//! Given strings s1, s2, and s3, find whether s3 is formed by an interleaving of s1 and s2.
//!
//! An interleaving of two strings s and t is a configuration where s and t are divided into 
//! n and m substrings respectively, such that:
//!
//! - s = s1 + s2 + ... + sn
//! - t = t1 + t2 + ... + tm
//! - |n - m| <= 1
//! - The interleaving is s1 + t1 + s2 + t2 + ... or t1 + s1 + t2 + s2 + ...
//!
//! Note: a + b is the concatenation of strings a and b.
//!
//! Constraints:
//! - 0 <= s1.length, s2.length <= 100
//! - 0 <= s3.length <= 200
//! - s1, s2, and s3 consist of lowercase English letters.
//!
//! Example 1:
//! Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
//! Output: true
//! Explanation: One way to obtain s3 is:
//! Split s1 into s1 = "aa" + "bc" + "c", and s2 into s2 = "dbbc" + "a".
//! Interleaving the two splits, we get "aa" + "dbbc" + "bc" + "a" + "c" = "aadbbcbcac".
//! Since s3 can be obtained by interleaving s1 and s2, we return true.
//!
//! Example 2:
//! Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc"
//! Output: false
//! Explanation: Notice how it is impossible to interleave s2 with any other string to get s3.
//!
//! Example 3:
//! Input: s1 = "", s2 = "b", s3 = "b"
//! Output: true

use std::collections::HashMap;

pub struct Solution;

impl Solution {
    /// Approach 1: Dynamic Programming (2D DP)
    /// 
    /// Use a 2D DP table where dp[i][j] represents whether the first i+j characters 
    /// of s3 can be formed by interleaving the first i characters of s1 and 
    /// the first j characters of s2.
    /// 
    /// Time Complexity: O(m * n) where m = s1.len(), n = s2.len()
    /// Space Complexity: O(m * n)
    pub fn is_interleave_dp_2d(s1: String, s2: String, s3: String) -> bool {
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        let s3_chars: Vec<char> = s3.chars().collect();
        
        let m = s1_chars.len();
        let n = s2_chars.len();
        let k = s3_chars.len();
        
        if m + n != k {
            return false;
        }
        
        let mut dp = vec![vec![false; n + 1]; m + 1];
        
        // Base case: empty strings
        dp[0][0] = true;
        
        // Fill first row: using only s2
        for j in 1..=n {
            dp[0][j] = dp[0][j - 1] && s2_chars[j - 1] == s3_chars[j - 1];
        }
        
        // Fill first column: using only s1
        for i in 1..=m {
            dp[i][0] = dp[i - 1][0] && s1_chars[i - 1] == s3_chars[i - 1];
        }
        
        // Fill the DP table
        for i in 1..=m {
            for j in 1..=n {
                let from_s1 = dp[i - 1][j] && s1_chars[i - 1] == s3_chars[i + j - 1];
                let from_s2 = dp[i][j - 1] && s2_chars[j - 1] == s3_chars[i + j - 1];
                dp[i][j] = from_s1 || from_s2;
            }
        }
        
        dp[m][n]
    }
    
    /// Approach 2: Space Optimized DP (1D DP)
    /// 
    /// Since each row only depends on the previous row and current row,
    /// we can optimize space to O(n).
    /// 
    /// Time Complexity: O(m * n)
    /// Space Complexity: O(n)
    pub fn is_interleave_dp_1d(s1: String, s2: String, s3: String) -> bool {
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        let s3_chars: Vec<char> = s3.chars().collect();
        
        let m = s1_chars.len();
        let n = s2_chars.len();
        let k = s3_chars.len();
        
        if m + n != k {
            return false;
        }
        
        let mut dp = vec![false; n + 1];
        
        // Base case
        dp[0] = true;
        
        // Fill first row
        for j in 1..=n {
            dp[j] = dp[j - 1] && s2_chars[j - 1] == s3_chars[j - 1];
        }
        
        // Process each row
        for i in 1..=m {
            // Update dp[0] for current row
            dp[0] = dp[0] && s1_chars[i - 1] == s3_chars[i - 1];
            
            for j in 1..=n {
                let from_s1 = dp[j] && s1_chars[i - 1] == s3_chars[i + j - 1];
                let from_s2 = dp[j - 1] && s2_chars[j - 1] == s3_chars[i + j - 1];
                dp[j] = from_s1 || from_s2;
            }
        }
        
        dp[n]
    }
    
    /// Approach 3: DFS with Memoization
    /// 
    /// Use recursive DFS to try all possible interleavings, with memoization
    /// to avoid recomputing subproblems.
    /// 
    /// Time Complexity: O(m * n)
    /// Space Complexity: O(m * n)
    pub fn is_interleave_dfs_memo(s1: String, s2: String, s3: String) -> bool {
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        let s3_chars: Vec<char> = s3.chars().collect();
        
        if s1_chars.len() + s2_chars.len() != s3_chars.len() {
            return false;
        }
        
        let mut memo = HashMap::new();
        Self::dfs_helper(&s1_chars, &s2_chars, &s3_chars, 0, 0, 0, &mut memo)
    }
    
    fn dfs_helper(
        s1: &[char], 
        s2: &[char], 
        s3: &[char], 
        i: usize, 
        j: usize, 
        k: usize,
        memo: &mut HashMap<(usize, usize), bool>
    ) -> bool {
        if k == s3.len() {
            return i == s1.len() && j == s2.len();
        }
        
        if let Some(&cached) = memo.get(&(i, j)) {
            return cached;
        }
        
        let mut result = false;
        
        // Try taking from s1
        if i < s1.len() && s1[i] == s3[k] {
            if Self::dfs_helper(s1, s2, s3, i + 1, j, k + 1, memo) {
                result = true;
            }
        }
        
        // Try taking from s2
        if !result && j < s2.len() && s2[j] == s3[k] {
            if Self::dfs_helper(s1, s2, s3, i, j + 1, k + 1, memo) {
                result = true;
            }
        }
        
        memo.insert((i, j), result);
        result
    }
    
    /// Approach 4: BFS (Breadth-First Search)
    /// 
    /// Use BFS to explore all possible ways to interleave s1 and s2,
    /// treating each (i, j) state as a node in the graph.
    /// 
    /// Time Complexity: O(m * n)
    /// Space Complexity: O(m * n)
    pub fn is_interleave_bfs(s1: String, s2: String, s3: String) -> bool {
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        let s3_chars: Vec<char> = s3.chars().collect();
        
        let m = s1_chars.len();
        let n = s2_chars.len();
        let k = s3_chars.len();
        
        if m + n != k {
            return false;
        }
        
        let mut queue = std::collections::VecDeque::new();
        let mut visited = vec![vec![false; n + 1]; m + 1];
        
        queue.push_back((0, 0));
        visited[0][0] = true;
        
        while let Some((i, j)) = queue.pop_front() {
            if i == m && j == n {
                return true;
            }
            
            let k_idx = i + j;
            
            // Try taking from s1
            if i < m && s1_chars[i] == s3_chars[k_idx] && !visited[i + 1][j] {
                queue.push_back((i + 1, j));
                visited[i + 1][j] = true;
            }
            
            // Try taking from s2
            if j < n && s2_chars[j] == s3_chars[k_idx] && !visited[i][j + 1] {
                queue.push_back((i, j + 1));
                visited[i][j + 1] = true;
            }
        }
        
        false
    }
    
    /// Approach 5: Iterative DP with Rolling Array
    /// 
    /// Alternative implementation using explicit state management
    /// and rolling array technique for space optimization.
    /// 
    /// Time Complexity: O(m * n)
    /// Space Complexity: O(min(m, n))
    pub fn is_interleave_rolling(s1: String, s2: String, s3: String) -> bool {
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        let s3_chars: Vec<char> = s3.chars().collect();
        
        let mut m = s1_chars.len();
        let mut n = s2_chars.len();
        let k = s3_chars.len();
        
        if m + n != k {
            return false;
        }
        
        // Ensure m <= n for space optimization
        let (s1_ref, s2_ref) = if m > n {
            std::mem::swap(&mut m, &mut n);
            (&s2_chars, &s1_chars)
        } else {
            (&s1_chars, &s2_chars)
        };
        
        let mut dp = vec![false; m + 1];
        
        // Base case
        dp[0] = true;
        
        // Fill first row
        for i in 1..=m {
            dp[i] = dp[i - 1] && s1_ref[i - 1] == s3_chars[i - 1];
        }
        
        // Process each column
        for j in 1..=n {
            // Update dp[0] for current column
            dp[0] = dp[0] && s2_ref[j - 1] == s3_chars[j - 1];
            
            for i in 1..=m {
                let from_s1 = dp[i] && s2_ref[j - 1] == s3_chars[i + j - 1];
                let from_s2 = dp[i - 1] && s1_ref[i - 1] == s3_chars[i + j - 1];
                dp[i] = from_s1 || from_s2;
            }
        }
        
        dp[m]
    }
    
    /// Approach 6: Bottom-Up DP with Character Frequency Check
    /// 
    /// First check if character frequencies match, then use DP.
    /// This can provide early termination for invalid cases.
    /// 
    /// Time Complexity: O(m * n + k)
    /// Space Complexity: O(m * n)
    pub fn is_interleave_freq_check(s1: String, s2: String, s3: String) -> bool {
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        let s3_chars: Vec<char> = s3.chars().collect();
        
        let m = s1_chars.len();
        let n = s2_chars.len();
        let k = s3_chars.len();
        
        if m + n != k {
            return false;
        }
        
        // Check character frequencies
        let mut freq1 = [0; 26];
        let mut freq2 = [0; 26];
        let mut freq3 = [0; 26];
        
        for &c in &s1_chars {
            freq1[(c as u8 - b'a') as usize] += 1;
        }
        for &c in &s2_chars {
            freq2[(c as u8 - b'a') as usize] += 1;
        }
        for &c in &s3_chars {
            freq3[(c as u8 - b'a') as usize] += 1;
        }
        
        for i in 0..26 {
            if freq1[i] + freq2[i] != freq3[i] {
                return false;
            }
        }
        
        // Use DP after frequency check
        let mut dp = vec![vec![false; n + 1]; m + 1];
        dp[0][0] = true;
        
        // Fill first row
        for j in 1..=n {
            dp[0][j] = dp[0][j - 1] && s2_chars[j - 1] == s3_chars[j - 1];
        }
        
        // Fill first column
        for i in 1..=m {
            dp[i][0] = dp[i - 1][0] && s1_chars[i - 1] == s3_chars[i - 1];
        }
        
        // Fill DP table
        for i in 1..=m {
            for j in 1..=n {
                let from_s1 = dp[i - 1][j] && s1_chars[i - 1] == s3_chars[i + j - 1];
                let from_s2 = dp[i][j - 1] && s2_chars[j - 1] == s3_chars[i + j - 1];
                dp[i][j] = from_s1 || from_s2;
            }
        }
        
        dp[m][n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_interleaving() {
        let s1 = "aabcc".to_string();
        let s2 = "dbbca".to_string();
        let s3 = "aadbbcbcac".to_string();
        
        assert!(Solution::is_interleave_dp_2d(s1.clone(), s2.clone(), s3.clone()));
        assert!(Solution::is_interleave_dp_1d(s1, s2, s3));
    }
    
    #[test]
    fn test_impossible_interleaving() {
        let s1 = "aabcc".to_string();
        let s2 = "dbbca".to_string();
        let s3 = "aadbbbaccc".to_string();
        
        assert!(!Solution::is_interleave_dfs_memo(s1.clone(), s2.clone(), s3.clone()));
        assert!(!Solution::is_interleave_bfs(s1, s2, s3));
    }
    
    #[test]
    fn test_empty_string_cases() {
        assert!(Solution::is_interleave_rolling("".to_string(), "b".to_string(), "b".to_string()));
        assert!(Solution::is_interleave_freq_check("a".to_string(), "".to_string(), "a".to_string()));
        assert!(Solution::is_interleave_dp_2d("".to_string(), "".to_string(), "".to_string()));
    }
    
    #[test]
    fn test_single_character() {
        assert!(Solution::is_interleave_dp_1d("a".to_string(), "b".to_string(), "ab".to_string()));
        assert!(Solution::is_interleave_dfs_memo("a".to_string(), "b".to_string(), "ba".to_string()));
        assert!(!Solution::is_interleave_bfs("a".to_string(), "b".to_string(), "aa".to_string()));
    }
    
    #[test]
    fn test_repeated_characters() {
        assert!(Solution::is_interleave_rolling("aaa".to_string(), "aaa".to_string(), "aaaaaa".to_string()));
        assert!(!Solution::is_interleave_freq_check("aaa".to_string(), "aaa".to_string(), "aaraaa".to_string()));
    }
    
    #[test]
    fn test_length_mismatch() {
        assert!(!Solution::is_interleave_dp_2d("ab".to_string(), "cd".to_string(), "abcde".to_string()));
        assert!(!Solution::is_interleave_dp_1d("ab".to_string(), "cd".to_string(), "abc".to_string()));
    }
    
    #[test]
    fn test_identical_strings() {
        assert!(Solution::is_interleave_dfs_memo("abc".to_string(), "abc".to_string(), "aabbcc".to_string()));
        assert!(Solution::is_interleave_bfs("abc".to_string(), "abc".to_string(), "abacbc".to_string()));
    }
    
    #[test]
    fn test_one_empty_string() {
        assert!(Solution::is_interleave_rolling("".to_string(), "abc".to_string(), "abc".to_string()));
        assert!(Solution::is_interleave_freq_check("abc".to_string(), "".to_string(), "abc".to_string()));
        assert!(!Solution::is_interleave_dp_2d("".to_string(), "abc".to_string(), "ab".to_string()));
    }
    
    #[test]
    fn test_complex_interleaving() {
        let s1 = "abcd".to_string();
        let s2 = "efgh".to_string();
        let s3 = "aebfcgdh".to_string();
        
        assert!(Solution::is_interleave_dp_1d(s1.clone(), s2.clone(), s3.clone()));
        assert!(Solution::is_interleave_dfs_memo(s1, s2, s3));
    }
    
    #[test]
    fn test_alternating_pattern() {
        let s1 = "aaaa".to_string();
        let s2 = "bbbb".to_string();
        let s3 = "ababaabb".to_string();
        
        assert!(Solution::is_interleave_bfs(s1.clone(), s2.clone(), s3.clone()));
        assert!(Solution::is_interleave_rolling(s1, s2, s3));
    }
    
    #[test]
    fn test_interleaving_possible() {
        let s1 = "abc".to_string();
        let s2 = "def".to_string();
        let s3 = "abdecf".to_string();
        
        assert!(Solution::is_interleave_freq_check(s1.clone(), s2.clone(), s3.clone()));
        assert!(Solution::is_interleave_dp_2d(s1, s2, s3));
    }
    
    #[test]
    fn test_long_strings() {
        let s1 = "a".repeat(50);
        let s2 = "b".repeat(50);
        let mut s3 = String::new();
        for i in 0..50 {
            s3.push('a');
            s3.push('b');
        }
        
        assert!(Solution::is_interleave_dp_1d(s1.clone(), s2.clone(), s3.clone()));
        assert!(Solution::is_interleave_rolling(s1, s2, s3));
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            ("aabcc", "dbbca", "aadbbcbcac", true),
            ("aabcc", "dbbca", "aadbbbaccc", false),
            ("", "b", "b", true),
            ("a", "b", "ab", true),
            ("a", "b", "ba", true),
            ("a", "b", "aa", false),
            ("abc", "def", "adbecf", true),
            ("abc", "def", "abdecf", true),
            ("", "", "", true),
            ("a", "", "a", true),
            ("", "a", "a", true),
            ("a", "", "b", false),
            ("aaa", "aaa", "aaaaaa", true),
            ("aaa", "aaa", "aaraaa", false),
            ("abcd", "efgh", "aebfcgdh", true),
            ("aaaa", "bbbb", "ababaabb", true),
        ];
        
        for (s1, s2, s3, expected) in test_cases {
            let s1 = s1.to_string();
            let s2 = s2.to_string();
            let s3 = s3.to_string();
            
            let result1 = Solution::is_interleave_dp_2d(s1.clone(), s2.clone(), s3.clone());
            let result2 = Solution::is_interleave_dp_1d(s1.clone(), s2.clone(), s3.clone());
            let result3 = Solution::is_interleave_dfs_memo(s1.clone(), s2.clone(), s3.clone());
            let result4 = Solution::is_interleave_bfs(s1.clone(), s2.clone(), s3.clone());
            let result5 = Solution::is_interleave_rolling(s1.clone(), s2.clone(), s3.clone());
            let result6 = Solution::is_interleave_freq_check(s1.clone(), s2.clone(), s3.clone());
            
            assert_eq!(result1, expected, "DP2D failed for ({}, {}, {})", s1, s2, s3);
            assert_eq!(result2, expected, "DP1D failed for ({}, {}, {})", s1, s2, s3);
            assert_eq!(result3, expected, "DFS_Memo failed for ({}, {}, {})", s1, s2, s3);
            assert_eq!(result4, expected, "BFS failed for ({}, {}, {})", s1, s2, s3);
            assert_eq!(result5, expected, "Rolling failed for ({}, {}, {})", s1, s2, s3);
            assert_eq!(result6, expected, "FreqCheck failed for ({}, {}, {})", s1, s2, s3);
            
            // Ensure all approaches return the same result
            assert_eq!(result1, result2, "DP2D vs DP1D mismatch for ({}, {}, {})", s1, s2, s3);
            assert_eq!(result2, result3, "DP1D vs DFS_Memo mismatch for ({}, {}, {})", s1, s2, s3);
            assert_eq!(result3, result4, "DFS_Memo vs BFS mismatch for ({}, {}, {})", s1, s2, s3);
            assert_eq!(result4, result5, "BFS vs Rolling mismatch for ({}, {}, {})", s1, s2, s3);
            assert_eq!(result5, result6, "Rolling vs FreqCheck mismatch for ({}, {}, {})", s1, s2, s3);
        }
    }
}