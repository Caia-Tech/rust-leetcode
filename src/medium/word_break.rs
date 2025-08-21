//! Problem 139: Word Break
//! 
//! Given a string s and a dictionary of strings wordDict, return true if s can be segmented 
//! into a space-separated sequence of one or more dictionary words.
//! 
//! Note that the same word in the dictionary may be reused multiple times in the segmentation.
//! 
//! Example 1:
//! Input: s = "leetcode", wordDict = ["leet","code"]
//! Output: true
//! Explanation: Return true because "leetcode" can be segmented as "leet code".
//! 
//! Example 2:
//! Input: s = "applepenapple", wordDict = ["apple","pen"]
//! Output: true
//! Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
//! Note that you are allowed to reuse a dictionary word.
//! 
//! Example 3:
//! Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
//! Output: false

use std::collections::{HashMap, HashSet, VecDeque};

pub struct Solution;

impl Solution {
    /// Approach 1: Dynamic Programming (Bottom-up)
    /// 
    /// dp[i] represents whether s[0..i] can be segmented using words from dictionary.
    /// For each position i, check all previous positions j where dp[j] is true and s[j..i] is in dict.
    /// 
    /// Time Complexity: O(n² + m) where n = len(s), m = total length of all words
    /// Space Complexity: O(n + m) for dp array and HashSet
    pub fn word_break_dp(&self, s: String, word_dict: Vec<String>) -> bool {
        let n = s.len();
        let word_set: HashSet<String> = word_dict.into_iter().collect();
        let mut dp = vec![false; n + 1];
        dp[0] = true; // Empty string can always be segmented
        
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
    
    /// Approach 2: Memoized Recursion (Top-down DP)
    /// 
    /// Uses recursion with memoization to check if string starting from index can be segmented.
    /// 
    /// Time Complexity: O(n² + m)
    /// Space Complexity: O(n + m) for memoization and recursion stack
    pub fn word_break_memo(&self, s: String, word_dict: Vec<String>) -> bool {
        let word_set: HashSet<String> = word_dict.into_iter().collect();
        let mut memo = HashMap::new();
        self.can_break(&s, 0, &word_set, &mut memo)
    }
    
    fn can_break(&self, s: &str, start: usize, word_set: &HashSet<String>, memo: &mut HashMap<usize, bool>) -> bool {
        if start == s.len() {
            return true;
        }
        
        if let Some(&cached) = memo.get(&start) {
            return cached;
        }
        
        for end in start + 1..=s.len() {
            let substring = &s[start..end];
            if word_set.contains(substring) && self.can_break(s, end, word_set, memo) {
                memo.insert(start, true);
                return true;
            }
        }
        
        memo.insert(start, false);
        false
    }
    
    /// Approach 3: BFS (Breadth-First Search)
    /// 
    /// Treats this as a graph problem where each valid word boundary is a node.
    /// Uses BFS to find if we can reach the end of the string.
    /// 
    /// Time Complexity: O(n² + m)
    /// Space Complexity: O(n + m) for queue and visited set
    pub fn word_break_bfs(&self, s: String, word_dict: Vec<String>) -> bool {
        let n = s.len();
        let word_set: HashSet<String> = word_dict.into_iter().collect();
        let mut visited = vec![false; n + 1];
        let mut queue = VecDeque::new();
        
        queue.push_back(0);
        visited[0] = true;
        
        while let Some(start) = queue.pop_front() {
            if start == n {
                return true;
            }
            
            for end in start + 1..=n {
                if !visited[end] && word_set.contains(&s[start..end]) {
                    visited[end] = true;
                    queue.push_back(end);
                }
            }
        }
        
        false
    }
    
    /// Approach 4: Trie-based Dynamic Programming
    /// 
    /// Uses a Trie data structure to efficiently check word prefixes and reduce
    /// the number of substring checks needed.
    /// 
    /// Time Complexity: O(n² + m) in worst case, but often better in practice
    /// Space Complexity: O(m) for trie + O(n) for dp
    pub fn word_break_trie(&self, s: String, word_dict: Vec<String>) -> bool {
        let mut trie = Trie::new();
        for word in word_dict {
            trie.insert(word);
        }
        
        let n = s.len();
        let chars: Vec<char> = s.chars().collect();
        let mut dp = vec![false; n + 1];
        dp[0] = true;
        
        for i in 1..=n {
            // For each position i, check all possible starting positions j
            for j in 0..i {
                if !dp[j] {
                    continue;
                }
                
                // Check if substring from j to i is a word in trie
                let mut node = &trie.root;
                let mut valid = true;
                
                for k in j..i {
                    if let Some(next_node) = node.children.get(&chars[k]) {
                        node = next_node;
                    } else {
                        valid = false;
                        break;
                    }
                }
                
                if valid && node.is_end {
                    dp[i] = true;
                    break;
                }
            }
        }
        
        dp[n]
    }
    
    /// Approach 5: Optimized DP with Word Length Filtering
    /// 
    /// Optimizes the basic DP approach by only checking substrings that have
    /// the same length as words in the dictionary.
    /// 
    /// Time Complexity: O(n * k + m) where k is the number of unique word lengths
    /// Space Complexity: O(n + m)
    pub fn word_break_optimized(&self, s: String, word_dict: Vec<String>) -> bool {
        let n = s.len();
        let word_set: HashSet<String> = word_dict.iter().cloned().collect();
        
        // Collect unique word lengths for optimization
        let word_lengths: HashSet<usize> = word_dict.iter().map(|w| w.len()).collect();
        
        let mut dp = vec![false; n + 1];
        dp[0] = true;
        
        for i in 1..=n {
            for &length in &word_lengths {
                if length <= i && dp[i - length] && word_set.contains(&s[i - length..i]) {
                    dp[i] = true;
                    break;
                }
            }
        }
        
        dp[n]
    }
    
    /// Approach 6: DFS with Pruning
    /// 
    /// Uses depth-first search with aggressive pruning to reduce search space.
    /// Includes early termination when impossible to complete.
    /// 
    /// Time Complexity: O(n²) with pruning, O(2^n) worst case
    /// Space Complexity: O(n) for recursion stack and memoization
    pub fn word_break_dfs(&self, s: String, word_dict: Vec<String>) -> bool {
        let word_set: HashSet<String> = word_dict.into_iter().collect();
        let max_word_len = word_set.iter().map(|w| w.len()).max().unwrap_or(0);
        let mut memo = HashMap::new();
        
        self.dfs_helper(&s, 0, &word_set, max_word_len, &mut memo)
    }
    
    fn dfs_helper(
        &self,
        s: &str,
        start: usize,
        word_set: &HashSet<String>,
        max_word_len: usize,
        memo: &mut HashMap<usize, bool>
    ) -> bool {
        if start == s.len() {
            return true;
        }
        
        if let Some(&cached) = memo.get(&start) {
            return cached;
        }
        
        // Pruning: if remaining string is shorter than minimum word length, it's impossible
        let remaining = s.len() - start;
        if remaining == 0 {
            memo.insert(start, false);
            return false;
        }
        
        for len in 1..=max_word_len.min(remaining) {
            let end = start + len;
            let substring = &s[start..end];
            
            if word_set.contains(substring) && self.dfs_helper(s, end, word_set, max_word_len, memo) {
                memo.insert(start, true);
                return true;
            }
        }
        
        memo.insert(start, false);
        false
    }
}

struct TrieNode {
    children: HashMap<char, TrieNode>,
    is_end: bool,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            is_end: false,
        }
    }
}

struct Trie {
    root: TrieNode,
}

impl Trie {
    fn new() -> Self {
        Self {
            root: TrieNode::new(),
        }
    }
    
    fn insert(&mut self, word: String) {
        let mut node = &mut self.root;
        for ch in word.chars() {
            node = node.children.entry(ch).or_insert_with(TrieNode::new);
        }
        node.is_end = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dp() {
        let solution = Solution;
        
        assert_eq!(solution.word_break_dp("leetcode".to_string(), vec!["leet".to_string(), "code".to_string()]), true);
        assert_eq!(solution.word_break_dp("applepenapple".to_string(), vec!["apple".to_string(), "pen".to_string()]), true);
        assert_eq!(solution.word_break_dp("catsandog".to_string(), vec!["cats".to_string(), "dog".to_string(), "sand".to_string(), "and".to_string(), "cat".to_string()]), false);
        assert_eq!(solution.word_break_dp("".to_string(), vec!["a".to_string()]), true);
        assert_eq!(solution.word_break_dp("a".to_string(), vec![]), false);
    }
    
    #[test]
    fn test_memo() {
        let solution = Solution;
        
        assert_eq!(solution.word_break_memo("leetcode".to_string(), vec!["leet".to_string(), "code".to_string()]), true);
        assert_eq!(solution.word_break_memo("applepenapple".to_string(), vec!["apple".to_string(), "pen".to_string()]), true);
        assert_eq!(solution.word_break_memo("catsandog".to_string(), vec!["cats".to_string(), "dog".to_string(), "sand".to_string(), "and".to_string(), "cat".to_string()]), false);
        assert_eq!(solution.word_break_memo("".to_string(), vec!["a".to_string()]), true);
        assert_eq!(solution.word_break_memo("a".to_string(), vec![]), false);
    }
    
    #[test]
    fn test_bfs() {
        let solution = Solution;
        
        assert_eq!(solution.word_break_bfs("leetcode".to_string(), vec!["leet".to_string(), "code".to_string()]), true);
        assert_eq!(solution.word_break_bfs("applepenapple".to_string(), vec!["apple".to_string(), "pen".to_string()]), true);
        assert_eq!(solution.word_break_bfs("catsandog".to_string(), vec!["cats".to_string(), "dog".to_string(), "sand".to_string(), "and".to_string(), "cat".to_string()]), false);
        assert_eq!(solution.word_break_bfs("".to_string(), vec!["a".to_string()]), true);
        assert_eq!(solution.word_break_bfs("a".to_string(), vec![]), false);
    }
    
    #[test]
    fn test_trie() {
        let solution = Solution;
        
        assert_eq!(solution.word_break_trie("leetcode".to_string(), vec!["leet".to_string(), "code".to_string()]), true);
        assert_eq!(solution.word_break_trie("applepenapple".to_string(), vec!["apple".to_string(), "pen".to_string()]), true);
        assert_eq!(solution.word_break_trie("catsandog".to_string(), vec!["cats".to_string(), "dog".to_string(), "sand".to_string(), "and".to_string(), "cat".to_string()]), false);
        assert_eq!(solution.word_break_trie("".to_string(), vec!["a".to_string()]), true);
        assert_eq!(solution.word_break_trie("a".to_string(), vec![]), false);
    }
    
    #[test]
    fn test_optimized() {
        let solution = Solution;
        
        assert_eq!(solution.word_break_optimized("leetcode".to_string(), vec!["leet".to_string(), "code".to_string()]), true);
        assert_eq!(solution.word_break_optimized("applepenapple".to_string(), vec!["apple".to_string(), "pen".to_string()]), true);
        assert_eq!(solution.word_break_optimized("catsandog".to_string(), vec!["cats".to_string(), "dog".to_string(), "sand".to_string(), "and".to_string(), "cat".to_string()]), false);
        assert_eq!(solution.word_break_optimized("".to_string(), vec!["a".to_string()]), true);
        assert_eq!(solution.word_break_optimized("a".to_string(), vec![]), false);
    }
    
    #[test]
    fn test_dfs() {
        let solution = Solution;
        
        assert_eq!(solution.word_break_dfs("leetcode".to_string(), vec!["leet".to_string(), "code".to_string()]), true);
        assert_eq!(solution.word_break_dfs("applepenapple".to_string(), vec!["apple".to_string(), "pen".to_string()]), true);
        assert_eq!(solution.word_break_dfs("catsandog".to_string(), vec!["cats".to_string(), "dog".to_string(), "sand".to_string(), "and".to_string(), "cat".to_string()]), false);
        assert_eq!(solution.word_break_dfs("".to_string(), vec!["a".to_string()]), true);
        assert_eq!(solution.word_break_dfs("a".to_string(), vec![]), false);
    }
    
    #[test]
    fn test_edge_cases() {
        let solution = Solution;
        
        // Single character
        assert_eq!(solution.word_break_dp("a".to_string(), vec!["a".to_string()]), true);
        assert_eq!(solution.word_break_dp("b".to_string(), vec!["a".to_string()]), false);
        
        // Long word
        assert_eq!(solution.word_break_dp("aaaaaaa".to_string(), vec!["aaaa".to_string(), "aaa".to_string()]), true);
        
        // Overlapping words
        assert_eq!(solution.word_break_dp("abcd".to_string(), vec!["a".to_string(), "abc".to_string(), "b".to_string(), "cd".to_string()]), true);
        
        // Word reuse
        assert_eq!(solution.word_break_dp("abab".to_string(), vec!["ab".to_string()]), true);
        
        // Partial matches
        assert_eq!(solution.word_break_dp("cars".to_string(), vec!["car".to_string(), "ca".to_string(), "rs".to_string()]), true);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let solution = Solution;
        
        let test_cases = vec![
            ("leetcode".to_string(), vec!["leet".to_string(), "code".to_string()]),
            ("applepenapple".to_string(), vec!["apple".to_string(), "pen".to_string()]),
            ("catsandog".to_string(), vec!["cats".to_string(), "dog".to_string(), "sand".to_string(), "and".to_string(), "cat".to_string()]),
            ("".to_string(), vec!["a".to_string()]),
            ("a".to_string(), vec![]),
            ("a".to_string(), vec!["a".to_string()]),
            ("aaaaaaa".to_string(), vec!["aaaa".to_string(), "aaa".to_string()]),
            ("abcd".to_string(), vec!["a".to_string(), "abc".to_string(), "b".to_string(), "cd".to_string()]),
            ("abab".to_string(), vec!["ab".to_string()]),
            ("cars".to_string(), vec!["car".to_string(), "ca".to_string(), "rs".to_string()]),
        ];
        
        for (s, word_dict) in test_cases {
            let dp = solution.word_break_dp(s.clone(), word_dict.clone());
            let memo = solution.word_break_memo(s.clone(), word_dict.clone());
            let bfs = solution.word_break_bfs(s.clone(), word_dict.clone());
            let trie = solution.word_break_trie(s.clone(), word_dict.clone());
            let optimized = solution.word_break_optimized(s.clone(), word_dict.clone());
            let dfs = solution.word_break_dfs(s.clone(), word_dict.clone());
            
            assert_eq!(dp, memo, "DP and memo differ for s='{}', dict={:?}", s, word_dict);
            assert_eq!(dp, bfs, "DP and BFS differ for s='{}', dict={:?}", s, word_dict);
            assert_eq!(dp, trie, "DP and trie differ for s='{}', dict={:?}", s, word_dict);
            assert_eq!(dp, optimized, "DP and optimized differ for s='{}', dict={:?}", s, word_dict);
            assert_eq!(dp, dfs, "DP and DFS differ for s='{}', dict={:?}", s, word_dict);
        }
    }
    
    #[test]
    fn test_complex_cases() {
        let solution = Solution;
        
        // Case where backtracking is necessary
        assert_eq!(
            solution.word_break_dp(
                "aaaaaaa".to_string(), 
                vec!["aaaa".to_string(), "aa".to_string()]
            ), 
            false
        );
        
        // Case with many possible segmentations
        assert_eq!(
            solution.word_break_dp(
                "abcdef".to_string(),
                vec!["ab".to_string(), "abc".to_string(), "cd".to_string(), "def".to_string(), "abcd".to_string(), "ef".to_string()]
            ),
            true
        );
        
        // Long string with repeated patterns
        let long_string = "cat".repeat(100);
        assert_eq!(
            solution.word_break_optimized(long_string, vec!["cat".to_string()]),
            true
        );
        
        // Impossible case that requires checking many combinations
        assert_eq!(
            solution.word_break_dfs(
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab".to_string(),
                vec!["a".to_string(), "aa".to_string(), "aaa".to_string(), "aaaa".to_string(), "aaaaa".to_string()]
            ),
            false
        );
    }
    
    #[test]
    fn test_trie_functionality() {
        let mut trie = Trie::new();
        trie.insert("hello".to_string());
        trie.insert("world".to_string());
        trie.insert("hell".to_string());
        
        // Test that the trie structure is built correctly
        assert!(trie.root.children.contains_key(&'h'));
        assert!(trie.root.children.contains_key(&'w'));
        
        let h_node = &trie.root.children[&'h'];
        assert!(h_node.children.contains_key(&'e'));
        
        let he_node = &h_node.children[&'e'];
        assert!(he_node.children.contains_key(&'l'));
    }
}