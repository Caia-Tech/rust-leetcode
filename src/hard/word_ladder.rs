//! Problem 127: Word Ladder
//!
//! A transformation sequence from word beginWord to word endWord using a dictionary wordList 
//! is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:
//!
//! - Every adjacent pair of words differs by exactly one letter.
//! - Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.
//! - sk == endWord
//!
//! Given two words, beginWord and endWord, and a dictionary wordList, return the number of words 
//! in the shortest transformation sequence from beginWord to endWord, or 0 if no such sequence exists.
//!
//! Constraints:
//! - 1 <= beginWord.length <= 10
//! - endWord.length == beginWord.length
//! - 1 <= wordList.length <= 5000
//! - wordList[i].length == beginWord.length
//! - beginWord, endWord, and wordList[i] consist of lowercase English letters.
//! - beginWord != endWord
//! - All the words in wordList are unique.

use std::collections::{HashMap, HashSet, VecDeque};

pub struct Solution;

impl Solution {
    /// Approach 1: BFS (Breadth-First Search)
    /// 
    /// Classic BFS to find shortest path in unweighted graph.
    /// Each word is a node, edges connect words differing by one letter.
    /// 
    /// Time Complexity: O(M^2 * N) where M = word length, N = word list size
    /// Space Complexity: O(M * N)
    pub fn ladder_length_bfs(begin_word: String, end_word: String, word_list: Vec<String>) -> i32 {
        let word_set: HashSet<String> = word_list.into_iter().collect();
        
        if !word_set.contains(&end_word) {
            return 0;
        }
        
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        
        queue.push_back((begin_word.clone(), 1));
        visited.insert(begin_word);
        
        while let Some((word, level)) = queue.pop_front() {
            if word == end_word {
                return level;
            }
            
            for next_word in Self::get_neighbors(&word, &word_set) {
                if !visited.contains(&next_word) {
                    visited.insert(next_word.clone());
                    queue.push_back((next_word, level + 1));
                }
            }
        }
        
        0
    }
    
    fn get_neighbors(word: &str, word_set: &HashSet<String>) -> Vec<String> {
        let mut neighbors = Vec::new();
        let word_chars: Vec<char> = word.chars().collect();
        
        for i in 0..word_chars.len() {
            let original_char = word_chars[i];
            
            for c in 'a'..='z' {
                if c != original_char {
                    let mut new_chars = word_chars.clone();
                    new_chars[i] = c;
                    let new_word: String = new_chars.into_iter().collect();
                    
                    if word_set.contains(&new_word) {
                        neighbors.push(new_word);
                    }
                }
            }
        }
        
        neighbors
    }
    
    /// Approach 2: Bidirectional BFS
    /// 
    /// Search from both ends simultaneously to reduce search space.
    /// More efficient when the path is long.
    /// 
    /// Time Complexity: O(M^2 * N) but potentially faster in practice
    /// Space Complexity: O(M * N)
    pub fn ladder_length_bidirectional(begin_word: String, end_word: String, word_list: Vec<String>) -> i32 {
        let word_set: HashSet<String> = word_list.into_iter().collect();
        
        if !word_set.contains(&end_word) {
            return 0;
        }
        
        let mut begin_set = HashSet::new();
        let mut end_set = HashSet::new();
        let mut visited = HashSet::new();
        
        begin_set.insert(begin_word);
        end_set.insert(end_word);
        
        let mut level = 1;
        
        while !begin_set.is_empty() && !end_set.is_empty() {
            // Always search from the smaller set
            if begin_set.len() > end_set.len() {
                std::mem::swap(&mut begin_set, &mut end_set);
            }
            
            let mut next_set = HashSet::new();
            
            for word in begin_set.drain() {
                for next_word in Self::get_neighbors(&word, &word_set) {
                    if end_set.contains(&next_word) {
                        return level + 1;
                    }
                    
                    if !visited.contains(&next_word) {
                        visited.insert(next_word.clone());
                        next_set.insert(next_word);
                    }
                }
            }
            
            begin_set = next_set;
            level += 1;
        }
        
        0
    }
    
    /// Approach 3: BFS with Pattern Optimization
    /// 
    /// Pre-compute intermediate patterns to speed up neighbor finding.
    /// For each word, create patterns like "*ot", "h*t", "ho*" for "hot".
    /// 
    /// Time Complexity: O(M^2 * N)
    /// Space Complexity: O(M^2 * N)
    pub fn ladder_length_pattern_bfs(begin_word: String, end_word: String, word_list: Vec<String>) -> i32 {
        if !word_list.contains(&end_word) {
            return 0;
        }
        
        // Pre-compute pattern to words mapping
        let mut pattern_map: HashMap<String, Vec<String>> = HashMap::new();
        let word_len = begin_word.len();
        
        let mut all_words = word_list;
        all_words.push(begin_word.clone());
        
        for word in &all_words {
            for i in 0..word_len {
                let pattern = format!("{}*{}", &word[..i], &word[i + 1..]);
                pattern_map.entry(pattern).or_default().push(word.clone());
            }
        }
        
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        
        queue.push_back((begin_word.clone(), 1));
        visited.insert(begin_word);
        
        while let Some((word, level)) = queue.pop_front() {
            if word == end_word {
                return level;
            }
            
            for i in 0..word_len {
                let pattern = format!("{}*{}", &word[..i], &word[i + 1..]);
                
                if let Some(neighbors) = pattern_map.get(&pattern) {
                    for neighbor in neighbors {
                        if !visited.contains(neighbor) {
                            visited.insert(neighbor.clone());
                            queue.push_back((neighbor.clone(), level + 1));
                        }
                    }
                }
            }
        }
        
        0
    }
    
    /// Approach 4: A* Search with Heuristic
    /// 
    /// Use A* with hamming distance as heuristic.
    /// More informed search but requires priority queue.
    /// 
    /// Time Complexity: O(M^2 * N * log N)
    /// Space Complexity: O(M * N)
    pub fn ladder_length_astar(begin_word: String, end_word: String, word_list: Vec<String>) -> i32 {
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;
        
        let word_set: HashSet<String> = word_list.into_iter().collect();
        
        if !word_set.contains(&end_word) {
            return 0;
        }
        
        let mut heap = BinaryHeap::new();
        let mut visited = HashSet::new();
        
        let initial_heuristic = Self::hamming_distance(&begin_word, &end_word);
        heap.push(Reverse((initial_heuristic + 1, begin_word.clone(), 1)));
        
        while let Some(Reverse((_, word, level))) = heap.pop() {
            if word == end_word {
                return level;
            }
            
            if visited.contains(&word) {
                continue;
            }
            visited.insert(word.clone());
            
            for next_word in Self::get_neighbors(&word, &word_set) {
                if !visited.contains(&next_word) {
                    let heuristic = Self::hamming_distance(&next_word, &end_word);
                    let priority = level + 1 + heuristic;
                    heap.push(Reverse((priority, next_word, level + 1)));
                }
            }
        }
        
        0
    }
    
    fn hamming_distance(word1: &str, word2: &str) -> i32 {
        word1.chars()
            .zip(word2.chars())
            .map(|(c1, c2)| if c1 != c2 { 1 } else { 0 })
            .sum()
    }
    
    /// Approach 5: DFS with Memoization
    /// 
    /// Use DFS with memoization to find shortest path.
    /// Less efficient than BFS for shortest path but demonstrates technique.
    /// 
    /// Time Complexity: O(M^2 * N * 2^N) worst case, but pruned by memoization
    /// Space Complexity: O(M * N)
    pub fn ladder_length_dfs_memo(begin_word: String, end_word: String, word_list: Vec<String>) -> i32 {
        let word_set: HashSet<String> = word_list.into_iter().collect();
        
        if !word_set.contains(&end_word) {
            return 0;
        }
        
        let mut memo = HashMap::new();
        let result = Self::dfs_shortest(&begin_word, &end_word, &word_set, &mut HashSet::new(), &mut memo);
        
        if result == i32::MAX {
            0
        } else {
            result
        }
    }
    
    fn dfs_shortest(current: &str, target: &str, word_set: &HashSet<String>, 
                   visited: &mut HashSet<String>, memo: &mut HashMap<String, i32>) -> i32 {
        if current == target {
            return 1;
        }
        
        if visited.contains(current) {
            return i32::MAX;
        }
        
        if let Some(&cached) = memo.get(current) {
            return cached;
        }
        
        visited.insert(current.to_string());
        let mut min_length = i32::MAX;
        
        for neighbor in Self::get_neighbors(current, word_set) {
            let length = Self::dfs_shortest(&neighbor, target, word_set, visited, memo);
            if length != i32::MAX {
                min_length = min_length.min(length + 1);
            }
        }
        
        visited.remove(current);
        memo.insert(current.to_string(), min_length);
        min_length
    }
    
    /// Approach 6: Graph Construction + BFS
    /// 
    /// Explicitly build adjacency graph first, then run BFS.
    /// Separates graph construction from search algorithm.
    /// 
    /// Time Complexity: O(M^2 * N^2) for construction + O(M * N) for BFS
    /// Space Complexity: O(M * N^2)
    pub fn ladder_length_graph_bfs(begin_word: String, end_word: String, word_list: Vec<String>) -> i32 {
        if !word_list.contains(&end_word) {
            return 0;
        }
        
        let mut all_words = word_list;
        if !all_words.contains(&begin_word) {
            all_words.push(begin_word.clone());
        }
        
        // Build adjacency graph
        let mut graph: HashMap<String, Vec<String>> = HashMap::new();
        let n = all_words.len();
        
        for i in 0..n {
            graph.entry(all_words[i].clone()).or_default();
            for j in i + 1..n {
                if Self::is_one_diff(&all_words[i], &all_words[j]) {
                    graph.entry(all_words[i].clone()).or_default().push(all_words[j].clone());
                    graph.entry(all_words[j].clone()).or_default().push(all_words[i].clone());
                }
            }
        }
        
        // BFS on the graph
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        
        queue.push_back((begin_word.clone(), 1));
        visited.insert(begin_word);
        
        while let Some((word, level)) = queue.pop_front() {
            if word == end_word {
                return level;
            }
            
            if let Some(neighbors) = graph.get(&word) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        queue.push_back((neighbor.clone(), level + 1));
                    }
                }
            }
        }
        
        0
    }
    
    fn is_one_diff(word1: &str, word2: &str) -> bool {
        if word1.len() != word2.len() {
            return false;
        }
        
        let mut diff_count = 0;
        for (c1, c2) in word1.chars().zip(word2.chars()) {
            if c1 != c2 {
                diff_count += 1;
                if diff_count > 1 {
                    return false;
                }
            }
        }
        
        diff_count == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_example() {
        let begin_word = "hit".to_string();
        let end_word = "cog".to_string();
        let word_list = vec![
            "hot".to_string(), "dot".to_string(), "dog".to_string(),
            "lot".to_string(), "log".to_string(), "cog".to_string()
        ];
        let expected = 5; // hit -> hot -> dot -> dog -> cog
        
        assert_eq!(Solution::ladder_length_bfs(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_bidirectional(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_pattern_bfs(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_astar(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_dfs_memo(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_graph_bfs(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
    }
    
    #[test]
    fn test_no_path() {
        let begin_word = "hit".to_string();
        let end_word = "cog".to_string();
        let word_list = vec![
            "hot".to_string(), "dot".to_string(), "dog".to_string(),
            "lot".to_string(), "log".to_string()
        ];
        let expected = 0; // No path to "cog"
        
        assert_eq!(Solution::ladder_length_bfs(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_bidirectional(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_pattern_bfs(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_astar(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_dfs_memo(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_graph_bfs(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
    }
    
    #[test]
    fn test_single_step() {
        let begin_word = "a".to_string();
        let end_word = "c".to_string();
        let word_list = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let expected = 2; // a -> c (direct transformation)
        
        assert_eq!(Solution::ladder_length_bfs(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_bidirectional(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_pattern_bfs(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_astar(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_dfs_memo(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_graph_bfs(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
    }
    
    
    #[test]
    fn test_longer_words() {
        let begin_word = "word".to_string();
        let end_word = "work".to_string();
        let word_list = vec![
            "word".to_string(), "wort".to_string(), "wart".to_string(),
            "work".to_string(), "fork".to_string()
        ];
        let expected = 2; // word -> work (direct transformation, only 'd' -> 'k')
        
        assert_eq!(Solution::ladder_length_bfs(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_bidirectional(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_pattern_bfs(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_astar(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_dfs_memo(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_graph_bfs(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
    }
    
    #[test]
    fn test_multiple_paths() {
        let begin_word = "red".to_string();
        let end_word = "tax".to_string();
        let word_list = vec![
            "red".to_string(), "ted".to_string(), "tad".to_string(), "tax".to_string(),
            "rad".to_string(), "rex".to_string(), "tex".to_string()
        ];
        let expected = 4; // red -> ted -> tad -> tax
        
        assert_eq!(Solution::ladder_length_bfs(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_bidirectional(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_pattern_bfs(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_astar(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_dfs_memo(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_graph_bfs(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
    }
    
    #[test]
    fn test_isolated_words() {
        let begin_word = "abc".to_string();
        let end_word = "xyz".to_string();
        let word_list = vec![
            "abc".to_string(), "def".to_string(), "ghi".to_string(),
            "jkl".to_string(), "mno".to_string(), "xyz".to_string()
        ];
        let expected = 0; // No path possible
        
        assert_eq!(Solution::ladder_length_bfs(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_bidirectional(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_pattern_bfs(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_astar(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_dfs_memo(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_graph_bfs(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
    }
    
    #[test]
    fn test_single_char_diff() {
        let begin_word = "cat".to_string();
        let end_word = "car".to_string();
        let word_list = vec!["cat".to_string(), "car".to_string()];
        let expected = 2; // cat -> car
        
        assert_eq!(Solution::ladder_length_bfs(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_bidirectional(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_pattern_bfs(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_astar(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_dfs_memo(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
        assert_eq!(Solution::ladder_length_graph_bfs(begin_word.clone(), end_word.clone(), word_list.clone()), expected);
    }
    
    #[test]
    fn test_consistency() {
        let test_cases = vec![
            ("hit", "cog", vec!["hot", "dot", "dog", "lot", "log", "cog"]),
            ("a", "c", vec!["a", "b", "c"]),
            ("hot", "dog", vec!["hot", "dog", "dot"]),
            ("red", "tax", vec!["red", "ted", "tad", "tax", "rad"]),
        ];
        
        for (begin, end, words) in test_cases {
            let begin_word = begin.to_string();
            let end_word = end.to_string();
            let word_list: Vec<String> = words.into_iter().map(|s| s.to_string()).collect();
            
            let result1 = Solution::ladder_length_bfs(begin_word.clone(), end_word.clone(), word_list.clone());
            let result2 = Solution::ladder_length_bidirectional(begin_word.clone(), end_word.clone(), word_list.clone());
            let result3 = Solution::ladder_length_pattern_bfs(begin_word.clone(), end_word.clone(), word_list.clone());
            let result4 = Solution::ladder_length_astar(begin_word.clone(), end_word.clone(), word_list.clone());
            let result5 = Solution::ladder_length_dfs_memo(begin_word.clone(), end_word.clone(), word_list.clone());
            let result6 = Solution::ladder_length_graph_bfs(begin_word.clone(), end_word.clone(), word_list.clone());
            
            assert_eq!(result1, result2, "BFS vs Bidirectional mismatch for {}->{}", begin, end);
            assert_eq!(result1, result3, "BFS vs Pattern mismatch for {}->{}", begin, end);
            assert_eq!(result1, result4, "BFS vs A* mismatch for {}->{}", begin, end);
            assert_eq!(result1, result5, "BFS vs DFS memo mismatch for {}->{}", begin, end);
            assert_eq!(result1, result6, "BFS vs Graph mismatch for {}->{}", begin, end);
        }
    }
    
    #[test]
    fn test_edge_cases() {
        // Empty word list
        assert_eq!(Solution::ladder_length_bfs("a".to_string(), "b".to_string(), vec![]), 0);
        
        // End word not in list
        assert_eq!(Solution::ladder_length_bfs("a".to_string(), "z".to_string(), vec!["b".to_string()]), 0);
        
        // Begin word same as end word but not in list
        assert_eq!(Solution::ladder_length_bfs("same".to_string(), "same".to_string(), vec!["diff".to_string()]), 0);
    }
    
    #[test]
    fn test_hamming_distance() {
        assert_eq!(Solution::hamming_distance("abc", "abc"), 0);
        assert_eq!(Solution::hamming_distance("abc", "abd"), 1);
        assert_eq!(Solution::hamming_distance("abc", "xyz"), 3);
        assert_eq!(Solution::hamming_distance("hot", "dot"), 1);
        assert_eq!(Solution::hamming_distance("hit", "cog"), 3);
    }
    
    #[test]
    fn test_is_one_diff() {
        assert!(Solution::is_one_diff("abc", "abd"));
        assert!(Solution::is_one_diff("hot", "dot"));
        assert!(!Solution::is_one_diff("abc", "xyz"));
        assert!(!Solution::is_one_diff("abc", "abc"));
        assert!(!Solution::is_one_diff("a", "ab"));
    }
}