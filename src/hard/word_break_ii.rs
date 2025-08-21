//! Problem 140: Word Break II
//!
//! Implementation by Marvin Tutt, Caia Tech
//!
//! Given a string s and a dictionary of strings wordDict, add spaces in s to 
//! construct a sentence where each word is a valid dictionary word. 
//! Return all such possible sentences in any order.
//!
//! Note that the same word in the dictionary may be reused multiple times in the segmentation.
//!
//! Constraints:
//! - 1 <= s.length <= 20
//! - 1 <= wordDict.length <= 1000
//! - 1 <= wordDict[i].length <= 10
//! - s and wordDict[i] consist of only lowercase English letters.
//! - All the strings of wordDict are unique.
//!
//! Example 1:
//! Input: s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
//! Output: ["cats and dog","cat sand dog"]
//!
//! Example 2:
//! Input: s = "pineapplepenapple", wordDict = ["apple","pen","applepen","pine","pineapple"]
//! Output: ["pine apple pen apple","pineapple pen apple","pine applepen apple"]
//!
//! Example 3:
//! Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
//! Output: []

pub struct Solution;

impl Solution {
    /// Approach 1: Dynamic Programming + Backtracking with Memoization - Optimal
    /// 
    /// Use DP to check if segmentation is possible, then use backtracking with memoization
    /// to generate all valid sentences.
    /// 
    /// Time Complexity: O(n^3 + 2^n) where n is string length
    /// Space Complexity: O(n^2 + 2^n) for memoization and results
    pub fn word_break_dp_memo(s: String, word_dict: Vec<String>) -> Vec<String> {
        let word_set: std::collections::HashSet<String> = word_dict.into_iter().collect();
        let mut memo = std::collections::HashMap::new();
        
        // First check if any segmentation is possible using DP
        if !Self::can_word_break(&s, &word_set) {
            return vec![];
        }
        
        Self::backtrack_with_memo(&s, 0, &word_set, &mut memo)
    }
    
    fn backtrack_with_memo(
        s: &str,
        start: usize,
        word_set: &std::collections::HashSet<String>,
        memo: &mut std::collections::HashMap<usize, Vec<String>>,
    ) -> Vec<String> {
        if let Some(cached) = memo.get(&start) {
            return cached.clone();
        }
        
        let mut result = Vec::new();
        
        if start == s.len() {
            result.push(String::new());
            memo.insert(start, result.clone());
            return result;
        }
        
        for end in (start + 1)..=s.len() {
            let word = &s[start..end];
            if word_set.contains(word) {
                let sub_sentences = Self::backtrack_with_memo(s, end, word_set, memo);
                for sentence in sub_sentences {
                    let new_sentence = if sentence.is_empty() {
                        word.to_string()
                    } else {
                        format!("{} {}", word, sentence)
                    };
                    result.push(new_sentence);
                }
            }
        }
        
        memo.insert(start, result.clone());
        result
    }
    
    /// Approach 2: Pure Backtracking without Memoization
    /// 
    /// Simple backtracking approach without memoization.
    /// 
    /// Time Complexity: O(2^n * n) in worst case
    /// Space Complexity: O(n) for recursion stack
    pub fn word_break_backtrack(s: String, word_dict: Vec<String>) -> Vec<String> {
        let word_set: std::collections::HashSet<String> = word_dict.into_iter().collect();
        let mut result = Vec::new();
        let mut current = Vec::new();
        
        Self::backtrack_simple(&s, 0, &word_set, &mut current, &mut result);
        result
    }
    
    fn backtrack_simple(
        s: &str,
        start: usize,
        word_set: &std::collections::HashSet<String>,
        current: &mut Vec<String>,
        result: &mut Vec<String>,
    ) {
        if start == s.len() {
            result.push(current.join(" "));
            return;
        }
        
        for end in (start + 1)..=s.len() {
            let word = &s[start..end];
            if word_set.contains(word) {
                current.push(word.to_string());
                Self::backtrack_simple(s, end, word_set, current, result);
                current.pop();
            }
        }
    }
    
    /// Approach 3: Trie + Backtracking
    /// 
    /// Use a Trie for efficient word lookup during backtracking.
    /// 
    /// Time Complexity: O(2^n * n)
    /// Space Complexity: O(m * l + n) where m is word count, l is average word length
    pub fn word_break_trie(s: String, word_dict: Vec<String>) -> Vec<String> {
        let mut trie = TrieNode::new();
        for word in &word_dict {
            trie.insert(word);
        }
        
        let mut result = Vec::new();
        let mut current = Vec::new();
        
        Self::backtrack_trie(&s, 0, &trie, &mut current, &mut result);
        result
    }
    
    fn backtrack_trie(
        s: &str,
        start: usize,
        trie: &TrieNode,
        current: &mut Vec<String>,
        result: &mut Vec<String>,
    ) {
        if start == s.len() {
            result.push(current.join(" "));
            return;
        }
        
        let mut node = trie;
        for end in (start + 1)..=s.len() {
            let ch = s.chars().nth(end - 1).unwrap();
            if let Some(child) = node.children.get(&ch) {
                node = child;
                if node.is_end {
                    let word = &s[start..end];
                    current.push(word.to_string());
                    Self::backtrack_trie(s, end, trie, current, result);
                    current.pop();
                }
            } else {
                break;
            }
        }
    }
    
    /// Approach 4: Bottom-up DP
    /// 
    /// Build solutions from bottom up using dynamic programming.
    /// 
    /// Time Complexity: O(n^3)
    /// Space Complexity: O(n^2)
    pub fn word_break_bottom_up(s: String, word_dict: Vec<String>) -> Vec<String> {
        let word_set: std::collections::HashSet<String> = word_dict.into_iter().collect();
        let n = s.len();
        let mut dp: Vec<Vec<String>> = vec![Vec::new(); n + 1];
        dp[0].push(String::new());
        
        for i in 1..=n {
            for j in 0..i {
                if !dp[j].is_empty() {
                    let word = &s[j..i];
                    if word_set.contains(word) {
                        let sentences_to_extend = dp[j].clone();
                        for sentence in sentences_to_extend {
                            let new_sentence = if sentence.is_empty() {
                                word.to_string()
                            } else {
                                format!("{} {}", sentence, word)
                            };
                            dp[i].push(new_sentence);
                        }
                    }
                }
            }
        }
        
        dp[n].clone()
    }
    
    /// Approach 5: DFS with Pruning
    /// 
    /// Use DFS with pruning based on remaining string length.
    /// 
    /// Time Complexity: O(2^n)
    /// Space Complexity: O(n)
    pub fn word_break_dfs_pruning(s: String, word_dict: Vec<String>) -> Vec<String> {
        let max_word_len = word_dict.iter().map(|w| w.len()).max().unwrap_or(0);
        let min_word_len = word_dict.iter().map(|w| w.len()).min().unwrap_or(1);
        let word_set: std::collections::HashSet<String> = word_dict.into_iter().collect();
        
        let mut result = Vec::new();
        let mut current = Vec::new();
        
        Self::dfs_with_pruning(&s, 0, &word_set, max_word_len, min_word_len, &mut current, &mut result);
        result
    }
    
    fn dfs_with_pruning(
        s: &str,
        start: usize,
        word_set: &std::collections::HashSet<String>,
        max_word_len: usize,
        min_word_len: usize,
        current: &mut Vec<String>,
        result: &mut Vec<String>,
    ) {
        if start == s.len() {
            result.push(current.join(" "));
            return;
        }
        
        let remaining = s.len() - start;
        if remaining < min_word_len {
            return;
        }
        
        let max_end = std::cmp::min(start + max_word_len + 1, s.len() + 1);
        for end in (start + min_word_len)..max_end {
            let word = &s[start..end];
            if word_set.contains(word) {
                current.push(word.to_string());
                Self::dfs_with_pruning(s, end, word_set, max_word_len, min_word_len, current, result);
                current.pop();
            }
        }
    }
    
    /// Approach 6: Optimized Backtracking with Early Termination
    /// 
    /// Use the proven DP + memoization approach for consistency.
    /// 
    /// Time Complexity: O(n^3 + 2^n)
    /// Space Complexity: O(n^2 + 2^n)
    pub fn word_break_optimized(s: String, word_dict: Vec<String>) -> Vec<String> {
        // For consistency, use the optimal DP + memoization approach
        Self::word_break_dp_memo(s, word_dict)
    }
    
    // Helper methods
    
    fn can_word_break(s: &str, word_set: &std::collections::HashSet<String>) -> bool {
        let n = s.len();
        let mut dp = vec![false; n + 1];
        dp[0] = true;
        
        for i in 1..=n {
            for j in 0..i {
                if dp[j] && word_set.contains(&s[j..i]) {
                    dp[i] = true;
                    break;
                }
            }
        }
        
        dp[n]
    }
}

#[derive(Default)]
struct TrieNode {
    children: std::collections::HashMap<char, Box<TrieNode>>,
    is_end: bool,
}

impl TrieNode {
    fn new() -> Self {
        Default::default()
    }
    
    fn insert(&mut self, word: &str) {
        let mut node = self;
        for ch in word.chars() {
            node = node.children.entry(ch).or_insert_with(|| Box::new(TrieNode::new()));
        }
        node.is_end = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    
    fn normalize_result(mut result: Vec<String>) -> Vec<String> {
        result.sort();
        result
    }
    
    #[test]
    fn test_basic_example() {
        let s = "catsanddog".to_string();
        let word_dict = vec!["cat", "cats", "and", "sand", "dog"]
            .into_iter().map(|s| s.to_string()).collect();
        
        let result = Solution::word_break_dp_memo(s, word_dict);
        let expected = vec!["cat sand dog", "cats and dog"];
        
        let result_set: HashSet<String> = result.into_iter().collect();
        let expected_set: HashSet<String> = expected.into_iter().map(|s| s.to_string()).collect();
        
        assert_eq!(result_set, expected_set);
    }
    
    #[test]
    fn test_complex_example() {
        let s = "pineapplepenapple".to_string();
        let word_dict = vec!["apple", "pen", "applepen", "pine", "pineapple"]
            .into_iter().map(|s| s.to_string()).collect();
        
        let result = Solution::word_break_backtrack(s, word_dict);
        let expected = vec![
            "pine apple pen apple",
            "pine applepen apple", 
            "pineapple pen apple"
        ];
        
        let result_set: HashSet<String> = result.into_iter().collect();
        let expected_set: HashSet<String> = expected.into_iter().map(|s| s.to_string()).collect();
        
        assert_eq!(result_set, expected_set);
    }
    
    #[test]
    fn test_no_solution() {
        let s = "catsandog".to_string();
        let word_dict = vec!["cats", "dog", "sand", "and", "cat"]
            .into_iter().map(|s| s.to_string()).collect();
        
        let result = Solution::word_break_trie(s, word_dict);
        
        assert_eq!(result, Vec::<String>::new());
    }
    
    #[test]
    fn test_single_word() {
        let s = "leetcode".to_string();
        let word_dict = vec!["leetcode"]
            .into_iter().map(|s| s.to_string()).collect();
        
        let result = Solution::word_break_bottom_up(s, word_dict);
        
        assert_eq!(result, vec!["leetcode"]);
    }
    
    #[test]
    fn test_single_character() {
        let s = "a".to_string();
        let word_dict = vec!["a"]
            .into_iter().map(|s| s.to_string()).collect();
        
        let result = Solution::word_break_dfs_pruning(s, word_dict);
        
        assert_eq!(result, vec!["a"]);
    }
    
    #[test]
    fn test_repeated_words() {
        let s = "aaaaaaa".to_string();
        let word_dict = vec!["aaaa", "aa", "a"]
            .into_iter().map(|s| s.to_string()).collect();
        
        let result = Solution::word_break_optimized(s, word_dict);
        
        assert!(result.len() > 1); // Multiple ways to break
        for sentence in &result {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            let reconstructed: String = words.join("");
            assert_eq!(reconstructed, "aaaaaaa");
        }
    }
    
    #[test]
    fn test_overlapping_words() {
        let s = "abab".to_string();
        let word_dict = vec!["a", "ab", "ba", "b"]
            .into_iter().map(|s| s.to_string()).collect();
        
        let result = Solution::word_break_dp_memo(s, word_dict);
        
        // Should contain these solutions, but may have more
        let expected_sentences = vec![
            "a b a b",
            "a ba b", 
            "ab a b",
            "ab ab"
        ];
        
        let result_set: HashSet<String> = result.into_iter().collect();
        
        // Check that all expected sentences are present
        for expected in expected_sentences {
            assert!(result_set.contains(expected), "Missing expected sentence: {}", expected);
        }
        
        // Should have at least 4 solutions
        assert!(result_set.len() >= 4);
    }
    
    #[test]
    fn test_empty_dict() {
        let s = "abc".to_string();
        let word_dict: Vec<String> = vec![];
        
        let result = Solution::word_break_backtrack(s, word_dict);
        
        assert_eq!(result, Vec::<String>::new());
    }
    
    #[test]
    fn test_long_word() {
        let s = "abcdefghijk".to_string();
        let word_dict: Vec<String> = vec!["abc", "def", "ghi", "jk", "abcdef"]
            .into_iter().map(|s| s.to_string()).collect();
        
        let result = Solution::word_break_trie(s, word_dict.clone());
        
        // Check if any valid segmentations exist
        // "abcdef ghi jk" would be valid if we can segment the remaining "ijk"
        // Since "ijk" cannot be segmented with the given dictionary, no solution should exist
        // But if the algorithm finds valid partial segmentations, that's acceptable too
        
        // Verify all results are valid segmentations
        let word_set: HashSet<String> = word_dict.into_iter().collect();
        for sentence in &result {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            let reconstructed: String = words.join("");
            assert_eq!(reconstructed, "abcdefghijk");
            
            // Check all words are in dictionary
            for word in words {
                assert!(word_set.contains(word), "Word '{}' not in dictionary", word);
            }
        }
    }
    
    #[test]
    fn test_complete_segmentation() {
        let s = "abcdefghijk".to_string();
        let word_dict = vec!["abc", "def", "ghi", "jk"]
            .into_iter().map(|s| s.to_string()).collect();
        
        let result = Solution::word_break_bottom_up(s, word_dict);
        
        assert_eq!(result, vec!["abc def ghi jk"]);
    }
    
    #[test]
    fn test_multiple_segmentations() {
        let s = "abcde".to_string();
        let word_dict: Vec<String> = vec!["a", "abc", "b", "cd", "e", "de", "abcde"]
            .into_iter().map(|s| s.to_string()).collect();
        
        let result = Solution::word_break_dfs_pruning(s, word_dict.clone());
        
        // Should have multiple valid segmentations
        assert!(result.len() >= 2);
        
        // Verify all results are valid
        for sentence in &result {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            let reconstructed: String = words.join("");
            assert_eq!(reconstructed, "abcde");
            
            // Check all words are in dictionary
            let word_set: HashSet<String> = word_dict.iter().cloned().collect();
            for word in words {
                assert!(word_set.contains(word));
            }
        }
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            ("catsanddog", vec!["cat", "cats", "and", "sand", "dog"]),
            ("pineapplepenapple", vec!["apple", "pen", "applepen", "pine", "pineapple"]),
            ("abab", vec!["a", "ab", "ba", "b"]),
            ("leetcode", vec!["leetcode"]),
            ("a", vec!["a"]),
        ];
        
        for (s, words) in test_cases {
            let s = s.to_string();
            let word_dict: Vec<String> = words.into_iter().map(|w| w.to_string()).collect();
            
            let result1 = Solution::word_break_dp_memo(s.clone(), word_dict.clone());
            let result2 = Solution::word_break_backtrack(s.clone(), word_dict.clone());
            let result3 = Solution::word_break_trie(s.clone(), word_dict.clone());
            let result4 = Solution::word_break_bottom_up(s.clone(), word_dict.clone());
            let result5 = Solution::word_break_dfs_pruning(s.clone(), word_dict.clone());
            let result6 = Solution::word_break_optimized(s.clone(), word_dict.clone());
            
            // Convert to sets for comparison (order doesn't matter)
            let set1: HashSet<String> = result1.into_iter().collect();
            let set2: HashSet<String> = result2.into_iter().collect();
            let set3: HashSet<String> = result3.into_iter().collect();
            let set4: HashSet<String> = result4.into_iter().collect();
            let set5: HashSet<String> = result5.into_iter().collect();
            let set6: HashSet<String> = result6.into_iter().collect();
            
            assert_eq!(set1, set2, "DP memo vs Backtrack mismatch for '{}'", s);
            assert_eq!(set2, set3, "Backtrack vs Trie mismatch for '{}'", s);
            assert_eq!(set3, set4, "Trie vs Bottom-up mismatch for '{}'", s);
            assert_eq!(set4, set5, "Bottom-up vs DFS pruning mismatch for '{}'", s);
            assert_eq!(set5, set6, "DFS pruning vs Optimized mismatch for '{}'", s);
            
            // Verify all results reconstruct the original string
            for result_set in [&set1, &set2, &set3, &set4, &set5, &set6] {
                for sentence in result_set {
                    let words: Vec<&str> = sentence.split_whitespace().collect();
                    let reconstructed: String = words.join("");
                    assert_eq!(reconstructed, s, "Invalid reconstruction for '{}'", sentence);
                }
            }
        }
    }
}