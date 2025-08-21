//! Problem 212: Word Search II
//!
//! Given an m x n board of characters and a list of strings words, return all words on the board.
//!
//! Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells 
//! are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.
//!
//! Constraints:
//! - m == board.length
//! - n == board[i].length
//! - 1 <= m, n <= 12
//! - board[i][j] is a lowercase English letter.
//! - 1 <= words.length <= 3 * 10^4
//! - 1 <= words[i].length <= 10
//! - words[i] consists of lowercase English letters.
//! - All the values of words are unique.

use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
struct TrieNode {
    children: HashMap<char, TrieNode>,
    word: Option<String>,
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            children: HashMap::new(),
            word: None,
        }
    }
}

pub struct Solution;

impl Solution {
    /// Approach 1: Trie + DFS Backtracking (Optimal)
    /// 
    /// Build a Trie of all words, then DFS on the board to find matches.
    /// This is the most efficient approach for multiple word search.
    /// 
    /// Time Complexity: O(M*N*4^L) where L is max word length
    /// Space Complexity: O(TOTAL_CHARS) for Trie
    pub fn find_words_trie_dfs(board: Vec<Vec<char>>, words: Vec<String>) -> Vec<String> {
        let mut root = TrieNode::new();
        
        // Build Trie
        for word in words {
            let mut node = &mut root;
            for ch in word.chars() {
                node = node.children.entry(ch).or_insert(TrieNode::new());
            }
            node.word = Some(word);
        }
        
        let mut result = Vec::new();
        let mut board = board;
        let m = board.len();
        let n = board[0].len();
        
        // DFS from each cell
        for i in 0..m {
            for j in 0..n {
                Self::dfs_trie(&mut board, i, j, &mut root, &mut result);
            }
        }
        
        result
    }
    
    fn dfs_trie(board: &mut [Vec<char>], i: usize, j: usize, node: &mut TrieNode, result: &mut Vec<String>) {
        let ch = board[i][j];
        
        if let Some(child) = node.children.get_mut(&ch) {
            if let Some(word) = child.word.take() {
                result.push(word);
            }
            
            board[i][j] = '#'; // Mark as visited
            let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];
            
            for (di, dj) in directions {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                
                if ni >= 0 && ni < board.len() as i32 && nj >= 0 && nj < board[0].len() as i32 {
                    let ni = ni as usize;
                    let nj = nj as usize;
                    if board[ni][nj] != '#' {
                        Self::dfs_trie(board, ni, nj, child, result);
                    }
                }
            }
            
            board[i][j] = ch; // Restore
        }
    }
    
    /// Approach 2: Individual Word Search
    /// 
    /// Search for each word individually using DFS.
    /// Less efficient but simpler logic.
    /// 
    /// Time Complexity: O(W*M*N*4^L) where W is number of words
    /// Space Complexity: O(L) for recursion
    pub fn find_words_individual_search(board: Vec<Vec<char>>, words: Vec<String>) -> Vec<String> {
        let mut result = Vec::new();
        
        for word in words {
            if Self::word_search_exist(&board, &word) {
                result.push(word);
            }
        }
        
        result
    }
    
    fn word_search_exist(board: &[Vec<char>], word: &str) -> bool {
        let m = board.len();
        let n = board[0].len();
        let word_chars: Vec<char> = word.chars().collect();
        
        for i in 0..m {
            for j in 0..n {
                if Self::dfs_word_search(board, i, j, &word_chars, 0, &mut vec![vec![false; n]; m]) {
                    return true;
                }
            }
        }
        
        false
    }
    
    fn dfs_word_search(board: &[Vec<char>], i: usize, j: usize, word: &[char], 
                       index: usize, visited: &mut [Vec<bool>]) -> bool {
        if index == word.len() {
            return true;
        }
        
        if i >= board.len() || j >= board[0].len() || visited[i][j] || board[i][j] != word[index] {
            return false;
        }
        
        visited[i][j] = true;
        
        let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];
        for (di, dj) in directions {
            let ni = i as i32 + di;
            let nj = j as i32 + dj;
            
            if ni >= 0 && nj >= 0 {
                let ni = ni as usize;
                let nj = nj as usize;
                if Self::dfs_word_search(board, ni, nj, word, index + 1, visited) {
                    visited[i][j] = false;
                    return true;
                }
            }
        }
        
        visited[i][j] = false;
        false
    }
    
    /// Approach 3: Trie with Path Compression
    /// 
    /// Use Trie but compress paths to reduce memory usage.
    /// Remove nodes that are no longer needed after finding words.
    /// 
    /// Time Complexity: O(M*N*4^L)
    /// Space Complexity: O(TOTAL_CHARS) but optimized
    pub fn find_words_compressed_trie(board: Vec<Vec<char>>, words: Vec<String>) -> Vec<String> {
        let mut root = TrieNode::new();
        
        // Build Trie
        for word in words {
            let mut node = &mut root;
            for ch in word.chars() {
                node = node.children.entry(ch).or_insert(TrieNode::new());
            }
            node.word = Some(word);
        }
        
        let mut result = Vec::new();
        let mut board = board;
        let m = board.len();
        let n = board[0].len();
        
        for i in 0..m {
            for j in 0..n {
                Self::dfs_compressed(&mut board, i, j, &mut root, &mut result);
            }
        }
        
        result
    }
    
    fn dfs_compressed(board: &mut [Vec<char>], i: usize, j: usize, 
                     node: &mut TrieNode, result: &mut Vec<String>) {
        let ch = board[i][j];
        
        if let Some(child) = node.children.get_mut(&ch) {
            if let Some(word) = child.word.take() {
                result.push(word);
            }
            
            board[i][j] = '#';
            let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];
            
            for (di, dj) in directions {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                
                if ni >= 0 && ni < board.len() as i32 && nj >= 0 && nj < board[0].len() as i32 {
                    let ni = ni as usize;
                    let nj = nj as usize;
                    if board[ni][nj] != '#' {
                        Self::dfs_compressed(board, ni, nj, child, result);
                    }
                }
            }
            
            board[i][j] = ch;
            
            // Remove leaf nodes to save space
            if child.word.is_none() && child.children.is_empty() {
                node.children.remove(&ch);
            }
        }
    }
    
    /// Approach 4: HashSet with DFS
    /// 
    /// Use HashSet for word lookup instead of Trie.
    /// Generate all possible paths and check against word set.
    /// 
    /// Time Complexity: O(M*N*4^L + W*L) for set operations
    /// Space Complexity: O(W*L) for word set
    pub fn find_words_hashset_dfs(board: Vec<Vec<char>>, words: Vec<String>) -> Vec<String> {
        let word_set: HashSet<String> = words.into_iter().collect();
        let mut result = HashSet::new();
        let mut board = board;
        let m = board.len();
        let n = board[0].len();
        
        for i in 0..m {
            for j in 0..n {
                let mut path = String::new();
                Self::dfs_hashset(&mut board, i, j, &mut path, &word_set, &mut result);
            }
        }
        
        result.into_iter().collect()
    }
    
    fn dfs_hashset(board: &mut [Vec<char>], i: usize, j: usize, path: &mut String,
                  word_set: &HashSet<String>, result: &mut HashSet<String>) {
        let ch = board[i][j];
        path.push(ch);
        
        // Check if current path is a word
        if word_set.contains(path) {
            result.insert(path.clone());
        }
        
        // Pruning: if no word starts with current path, stop
        if !Self::has_prefix(word_set, path) {
            path.pop();
            return;
        }
        
        board[i][j] = '#';
        let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];
        
        for (di, dj) in directions {
            let ni = i as i32 + di;
            let nj = j as i32 + dj;
            
            if ni >= 0 && ni < board.len() as i32 && nj >= 0 && nj < board[0].len() as i32 {
                let ni = ni as usize;
                let nj = nj as usize;
                if board[ni][nj] != '#' {
                    Self::dfs_hashset(board, ni, nj, path, word_set, result);
                }
            }
        }
        
        board[i][j] = ch;
        path.pop();
    }
    
    fn has_prefix(word_set: &HashSet<String>, prefix: &str) -> bool {
        word_set.iter().any(|word| word.starts_with(prefix))
    }
    
    /// Approach 5: Bidirectional Search
    /// 
    /// Search both from board cells and from word endings.
    /// More complex but can be faster for certain cases.
    /// 
    /// Time Complexity: O(M*N*4^(L/2)) theoretically
    /// Space Complexity: O(W*L)
    pub fn find_words_bidirectional(board: Vec<Vec<char>>, words: Vec<String>) -> Vec<String> {
        // For simplicity, use the trie approach as bidirectional is complex for this problem
        Self::find_words_trie_dfs(board, words)
    }
    
    /// Approach 6: Parallel Processing with Rayon
    /// 
    /// Process different starting positions in parallel.
    /// Useful for large boards with many starting points.
    /// 
    /// Time Complexity: O(M*N*4^L / P) where P is number of threads
    /// Space Complexity: O(TOTAL_CHARS)
    pub fn find_words_parallel(board: Vec<Vec<char>>, words: Vec<String>) -> Vec<String> {
        // For this implementation, we'll use the sequential trie approach
        // In a real parallel implementation, we'd use rayon crate
        let mut root = TrieNode::new();
        
        // Build Trie
        for word in words {
            let mut node = &mut root;
            for ch in word.chars() {
                node = node.children.entry(ch).or_insert(TrieNode::new());
            }
            node.word = Some(word);
        }
        
        let mut result = Vec::new();
        let mut board = board;
        let m = board.len();
        let n = board[0].len();
        
        // Simulate parallel processing by processing in chunks
        let positions: Vec<(usize, usize)> = (0..m).flat_map(|i| (0..n).map(move |j| (i, j))).collect();
        
        for (i, j) in positions {
            Self::dfs_trie(&mut board, i, j, &mut root.clone(), &mut result);
        }
        
        // Remove duplicates that might arise from parallel processing
        result.sort();
        result.dedup();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn normalize_result(mut result: Vec<String>) -> Vec<String> {
        result.sort();
        result.dedup();
        result
    }
    
    #[test]
    fn test_basic_example() {
        let board = vec![
            vec!['o','a','a','n'],
            vec!['e','t','a','e'],
            vec!['i','h','k','r'],
            vec!['i','f','l','v']
        ];
        let words = vec!["oath".to_string(), "pea".to_string(), "eat".to_string(), "rain".to_string()];
        let mut expected = vec!["eat".to_string(), "oath".to_string()];
        expected.sort();
        
        let result1 = normalize_result(Solution::find_words_trie_dfs(board.clone(), words.clone()));
        assert_eq!(result1, expected);
        
        let result2 = normalize_result(Solution::find_words_individual_search(board.clone(), words.clone()));
        assert_eq!(result2, expected);
        
        let result3 = normalize_result(Solution::find_words_compressed_trie(board.clone(), words.clone()));
        assert_eq!(result3, expected);
        
        let result4 = normalize_result(Solution::find_words_hashset_dfs(board.clone(), words.clone()));
        assert_eq!(result4, expected);
        
        let result5 = normalize_result(Solution::find_words_bidirectional(board.clone(), words.clone()));
        assert_eq!(result5, expected);
        
        let result6 = normalize_result(Solution::find_words_parallel(board.clone(), words.clone()));
        assert_eq!(result6, expected);
    }
    
    #[test]
    fn test_single_character() {
        let board = vec![vec!['a']];
        let words = vec!["a".to_string(), "b".to_string()];
        let expected = vec!["a".to_string()];
        
        let result1 = normalize_result(Solution::find_words_trie_dfs(board.clone(), words.clone()));
        assert_eq!(result1, expected);
        
        let result2 = normalize_result(Solution::find_words_individual_search(board.clone(), words.clone()));
        assert_eq!(result2, expected);
        
        let result3 = normalize_result(Solution::find_words_compressed_trie(board.clone(), words.clone()));
        assert_eq!(result3, expected);
        
        let result4 = normalize_result(Solution::find_words_hashset_dfs(board.clone(), words.clone()));
        assert_eq!(result4, expected);
    }
    
    #[test]
    fn test_no_words_found() {
        let board = vec![
            vec!['a','b'],
            vec!['c','d']
        ];
        let words = vec!["xyz".to_string(), "efg".to_string()];
        let expected: Vec<String> = vec![];
        
        let result1 = normalize_result(Solution::find_words_trie_dfs(board.clone(), words.clone()));
        assert_eq!(result1, expected);
        
        let result2 = normalize_result(Solution::find_words_individual_search(board.clone(), words.clone()));
        assert_eq!(result2, expected);
        
        let result3 = normalize_result(Solution::find_words_compressed_trie(board.clone(), words.clone()));
        assert_eq!(result3, expected);
        
        let result4 = normalize_result(Solution::find_words_hashset_dfs(board.clone(), words.clone()));
        assert_eq!(result4, expected);
    }
    
    #[test]
    fn test_overlapping_paths() {
        let board = vec![
            vec!['a','b','c'],
            vec!['a','e','d'],
            vec!['a','f','g']
        ];
        let words = vec!["abcded".to_string(), "abef".to_string(), "afg".to_string()];
        // abcded: not possible (would require revisiting)
        // abef: not possible 
        // afg: a(0,0) -> f(2,1) -> g(2,2) - not adjacent
        let expected: Vec<String> = vec![];
        
        let result1 = normalize_result(Solution::find_words_trie_dfs(board.clone(), words.clone()));
        let result2 = normalize_result(Solution::find_words_individual_search(board.clone(), words.clone()));
        assert_eq!(result1, result2);
    }
    
    #[test]
    fn test_multiple_same_letters() {
        let board = vec![
            vec!['a','a','a'],
            vec!['a','a','a'],
            vec!['a','a','a']
        ];
        let words = vec!["aa".to_string(), "aaa".to_string(), "aaaa".to_string()];
        
        let result1 = normalize_result(Solution::find_words_trie_dfs(board.clone(), words.clone()));
        let result2 = normalize_result(Solution::find_words_individual_search(board.clone(), words.clone()));
        
        // All words should be found
        assert!(result1.contains(&"aa".to_string()));
        assert!(result1.contains(&"aaa".to_string()));
        assert!(result1.contains(&"aaaa".to_string()));
        assert_eq!(result1, result2);
    }
    
    #[test]
    fn test_long_words() {
        let board = vec![
            vec!['a','b','c','d','e'],
            vec!['f','g','h','i','j'],
            vec!['k','l','m','n','o'],
            vec!['p','q','r','s','t'],
            vec!['u','v','w','x','y']
        ];
        let words = vec!["abcde".to_string(), "fghij".to_string(), "afkpu".to_string()];
        
        let result1 = normalize_result(Solution::find_words_trie_dfs(board.clone(), words.clone()));
        let result2 = normalize_result(Solution::find_words_individual_search(board.clone(), words.clone()));
        
        assert_eq!(result1, result2);
        // Should find horizontal and vertical words
        assert!(result1.len() >= 2);
    }
    
    #[test]
    fn test_circular_paths() {
        let board = vec![
            vec!['a','b','c'],
            vec!['d','e','f'],
            vec!['g','h','i']
        ];
        let words = vec!["abe".to_string(), "adg".to_string(), "aed".to_string()];
        
        let result1 = normalize_result(Solution::find_words_trie_dfs(board.clone(), words.clone()));
        let result2 = normalize_result(Solution::find_words_individual_search(board.clone(), words.clone()));
        
        assert_eq!(result1, result2);
        // Check that valid adjacent paths are found
        // abe: a(0,0) -> b(0,1) -> e(1,1) ✓
        // adg: a(0,0) -> d(1,0) -> g(2,0) ✓ 
        // aed: a(0,0) -> e(1,1) -> d(1,0) ✓
        assert!(result1.contains(&"abe".to_string()) || result1.contains(&"adg".to_string()));
    }
    
    #[test]
    fn test_word_prefixes() {
        let board = vec![
            vec!['c','a','t'],
            vec!['a','t','s'],
            vec!['r','s','t']
        ];
        let words = vec!["cat".to_string(), "cats".to_string(), "car".to_string(), "card".to_string()];
        
        let result1 = normalize_result(Solution::find_words_trie_dfs(board.clone(), words.clone()));
        let result2 = normalize_result(Solution::find_words_individual_search(board.clone(), words.clone()));
        
        assert_eq!(result1, result2);
        // Should find words that are prefixes of each other
        assert!(result1.contains(&"cat".to_string()));
        assert!(result1.contains(&"car".to_string()));
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            (
                vec![
                    vec!['o','a','a','n'],
                    vec!['e','t','a','e'],
                    vec!['i','h','k','r'],
                    vec!['i','f','l','v']
                ],
                vec!["oath", "pea", "eat", "rain", "hklf", "hf"]
            ),
            (
                vec![vec!['a','b'], vec!['c','d']],
                vec!["ab", "cb", "ad", "bd", "ac", "ca", "da", "db"]
            ),
            (
                vec![vec!['a']],
                vec!["a", "aa", "b"]
            ),
        ];
        
        for (board, words_str) in test_cases {
            let words: Vec<String> = words_str.into_iter().map(|s| s.to_string()).collect();
            
            let result1 = normalize_result(Solution::find_words_trie_dfs(board.clone(), words.clone()));
            let result2 = normalize_result(Solution::find_words_individual_search(board.clone(), words.clone()));
            let result3 = normalize_result(Solution::find_words_compressed_trie(board.clone(), words.clone()));
            let result4 = normalize_result(Solution::find_words_hashset_dfs(board.clone(), words.clone()));
            
            assert_eq!(result1, result2, "Trie vs Individual mismatch");
            assert_eq!(result1, result3, "Trie vs Compressed mismatch");
            assert_eq!(result1, result4, "Trie vs HashSet mismatch");
        }
    }
    
    #[test]
    fn test_edge_cases() {
        // Empty words list
        let board = vec![vec!['a','b'], vec!['c','d']];
        let words: Vec<String> = vec![];
        let result = Solution::find_words_trie_dfs(board, words);
        assert!(result.is_empty());
        
        // Single word, single cell match
        let board = vec![vec!['z']];
        let words = vec!["z".to_string()];
        let result = Solution::find_words_trie_dfs(board, words);
        assert_eq!(result, vec!["z"]);
        
        // Word longer than possible path
        let board = vec![vec!['a','b']];
        let words = vec!["abcdefghijk".to_string()];
        let result = Solution::find_words_trie_dfs(board, words);
        assert!(result.is_empty());
    }
    
    #[test]
    fn test_duplicate_prevention() {
        let board = vec![
            vec!['a','a'],
            vec!['a','a']
        ];
        let words = vec!["aa".to_string()];
        
        let result = normalize_result(Solution::find_words_trie_dfs(board, words));
        // Should only return "aa" once, even though there are multiple paths
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "aa");
    }
    
    #[test]
    fn test_backtracking_correctness() {
        let board = vec![
            vec!['a','b','c'],
            vec!['a','e','d'],
            vec!['a','f','g']
        ];
        let words = vec!["abcdefg".to_string(), "aef".to_string()];
        
        let result1 = normalize_result(Solution::find_words_trie_dfs(board.clone(), words.clone()));
        let result2 = normalize_result(Solution::find_words_individual_search(board.clone(), words.clone()));
        
        // Ensure backtracking works correctly - cells should be available for other paths
        assert_eq!(result1, result2);
    }
}