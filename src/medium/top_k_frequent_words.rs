//! # Problem 692: Top K Frequent Words
//!
//! **Difficulty**: Medium
//! **Topics**: Hash Table, String, Trie, Sorting, Heap (Priority Queue), Bucket Sort, Counting
//! **Acceptance Rate**: 54.8%

use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;

/// Solution struct following LeetCode format
pub struct Solution;

#[derive(Debug, Eq, PartialEq)]
struct WordFreq {
    word: String,
    freq: i32,
}

impl Ord for WordFreq {
    fn cmp(&self, other: &Self) -> Ordering {
        // First compare by frequency (descending for max heap)
        match self.freq.cmp(&other.freq) {
            Ordering::Equal => {
                // If frequencies are equal, compare lexicographically (ascending)
                other.word.cmp(&self.word)
            }
            other => other,
        }
    }
}

impl PartialOrd for WordFreq {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Solution {
    /// Create a new solution instance
    pub fn new() -> Self {
        Solution
    }

    /// Main solution approach using heap
    /// 
    /// Time Complexity: O(n log k) where n is number of words, k is the result size
    /// Space Complexity: O(n) for the frequency map
    pub fn top_k_frequent(&self, words: Vec<String>, k: i32) -> Vec<String> {
        // Count frequency of each word
        let mut freq_map = HashMap::new();
        for word in words {
            *freq_map.entry(word).or_insert(0) += 1;
        }
        
        // Use a min-heap to keep track of top k elements
        let mut heap = BinaryHeap::new();
        
        for (word, freq) in freq_map {
            heap.push(WordFreq {
                word: word.clone(),
                freq,
            });
        }
        
        // Extract top k elements
        let mut result = Vec::new();
        for _ in 0..k {
            if let Some(word_freq) = heap.pop() {
                result.push(word_freq.word);
            }
        }
        
        result
    }

    /// Alternative solution using sorting
    /// 
    /// Time Complexity: O(n log n) where n is number of unique words
    /// Space Complexity: O(n) for frequency map and result vector
    pub fn top_k_frequent_sort(&self, words: Vec<String>, k: i32) -> Vec<String> {
        // Count frequency of each word
        let mut freq_map = HashMap::new();
        for word in words {
            *freq_map.entry(word).or_insert(0) += 1;
        }
        
        // Convert to vector and sort
        let mut word_freq_pairs: Vec<(String, i32)> = freq_map.into_iter().collect();
        
        // Sort by frequency (descending) then by lexicographic order (ascending)
        word_freq_pairs.sort_by(|a, b| {
            match b.1.cmp(&a.1) {
                Ordering::Equal => a.0.cmp(&b.0),
                other => other,
            }
        });
        
        // Take top k elements
        word_freq_pairs
            .into_iter()
            .take(k as usize)
            .map(|(word, _)| word)
            .collect()
    }

    /// Bucket sort approach for better performance when k is small
    /// 
    /// Time Complexity: O(n) where n is total number of words
    /// Space Complexity: O(n) for frequency map and buckets
    pub fn top_k_frequent_bucket_sort(&self, words: Vec<String>, k: i32) -> Vec<String> {
        // Count frequency of each word
        let mut freq_map = HashMap::new();
        for word in &words {
            *freq_map.entry(word.clone()).or_insert(0) += 1;
        }
        
        let n = words.len();
        // Create buckets for each possible frequency
        let mut buckets: Vec<Vec<String>> = vec![Vec::new(); n + 1];
        
        // Place words in buckets based on frequency
        for (word, freq) in freq_map {
            buckets[freq as usize].push(word);
        }
        
        // Sort words within each bucket lexicographically
        for bucket in &mut buckets {
            bucket.sort();
        }
        
        // Collect top k words from highest frequency buckets
        let mut result = Vec::new();
        for bucket in buckets.iter().rev() {
            for word in bucket {
                if result.len() < k as usize {
                    result.push(word.clone());
                } else {
                    break;
                }
            }
            if result.len() >= k as usize {
                break;
            }
        }
        
        result
    }

    /// Trie-based solution for lexicographic ordering optimization
    /// 
    /// Time Complexity: O(n * m + k log k) where m is average word length
    /// Space Complexity: O(n * m) for the trie structure
    pub fn top_k_frequent_trie(&self, words: Vec<String>, k: i32) -> Vec<String> {
        // Count frequencies
        let mut freq_map = HashMap::new();
        for word in words {
            *freq_map.entry(word).or_insert(0) += 1;
        }
        
        // Group words by frequency
        let mut freq_groups: HashMap<i32, Vec<String>> = HashMap::new();
        for (word, freq) in freq_map {
            freq_groups.entry(freq).or_insert_with(Vec::new).push(word);
        }
        
        // Sort words within each frequency group
        for words_list in freq_groups.values_mut() {
            words_list.sort();
        }
        
        // Collect results from highest frequency to lowest
        let mut frequencies: Vec<i32> = freq_groups.keys().cloned().collect();
        frequencies.sort_by(|a, b| b.cmp(a));
        
        let mut result = Vec::new();
        for freq in frequencies {
            if let Some(words_list) = freq_groups.get(&freq) {
                for word in words_list {
                    if result.len() < k as usize {
                        result.push(word.clone());
                    } else {
                        return result;
                    }
                }
            }
        }
        
        result
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
        
        // Test case 1: ["i","love","leetcode","i","love","coding"], k = 2
        let words1 = vec![
            "i".to_string(), "love".to_string(), "leetcode".to_string(),
            "i".to_string(), "love".to_string(), "coding".to_string()
        ];
        assert_eq!(solution.top_k_frequent(words1, 2), vec!["i", "love"]);
        
        // Test case 2: ["the","day","is","sunny","the","the","the","sunny","is","is"], k = 4
        let words2 = vec![
            "the".to_string(), "day".to_string(), "is".to_string(), "sunny".to_string(),
            "the".to_string(), "the".to_string(), "the".to_string(), "sunny".to_string(),
            "is".to_string(), "is".to_string()
        ];
        assert_eq!(solution.top_k_frequent(words2, 4), vec!["the", "is", "sunny", "day"]);
    }

    #[test]
    fn test_edge_cases() {
        let solution = Solution::new();
        
        // Single word
        let words = vec!["word".to_string()];
        assert_eq!(solution.top_k_frequent(words, 1), vec!["word"]);
        
        // All words have same frequency - should be lexicographically ordered
        let words = vec!["b".to_string(), "a".to_string(), "c".to_string()];
        assert_eq!(solution.top_k_frequent(words, 2), vec!["a", "b"]);
        
        // k equals number of unique words
        let words = vec!["a".to_string(), "b".to_string()];
        assert_eq!(solution.top_k_frequent(words, 2), vec!["a", "b"]);
    }

    #[test]
    fn test_lexicographic_ordering() {
        let solution = Solution::new();
        
        // Test lexicographic ordering when frequencies are equal
        let words = vec![
            "apple".to_string(), "banana".to_string(), "apple".to_string(), "banana".to_string(),
            "cherry".to_string(), "date".to_string()
        ];
        // apple: 2, banana: 2, cherry: 1, date: 1
        // Expected: ["apple", "banana"] (frequency 2, lexicographic order)
        let result = solution.top_k_frequent(words, 2);
        assert_eq!(result, vec!["apple", "banana"]);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = Solution::new();
        
        let test_cases = vec![
            (vec!["i".to_string(), "love".to_string(), "leetcode".to_string(),
                  "i".to_string(), "love".to_string(), "coding".to_string()], 2),
            (vec!["the".to_string(), "day".to_string(), "is".to_string(), "sunny".to_string(),
                  "the".to_string(), "the".to_string(), "the".to_string(), "sunny".to_string(),
                  "is".to_string(), "is".to_string()], 4),
            (vec!["a".to_string(), "aa".to_string(), "aaa".to_string()], 3),
        ];

        for (words, k) in test_cases {
            let result1 = solution.top_k_frequent(words.clone(), k);
            let result2 = solution.top_k_frequent_sort(words.clone(), k);
            let result3 = solution.top_k_frequent_bucket_sort(words.clone(), k);
            let result4 = solution.top_k_frequent_trie(words.clone(), k);
            
            assert_eq!(result1, result2, "Heap and sort approaches should match");
            assert_eq!(result1, result3, "Heap and bucket sort approaches should match");
            assert_eq!(result1, result4, "Heap and trie approaches should match");
        }
    }

    #[test]
    fn test_performance_scenarios() {
        let solution = Solution::new();
        
        // Large number of words with few unique values
        let mut words = Vec::new();
        for _ in 0..1000 {
            words.push("frequent".to_string());
        }
        for _ in 0..10 {
            words.push("rare".to_string());
        }
        words.push("unique".to_string());
        
        let result = solution.top_k_frequent(words, 2);
        assert_eq!(result[0], "frequent");
        assert_eq!(result[1], "rare");
        
        // Many unique words with same frequency
        let words: Vec<String> = (0..100).map(|i| format!("word{}", i)).collect();
        let result = solution.top_k_frequent(words, 5);
        assert_eq!(result.len(), 5);
        // Should be lexicographically ordered
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }
}

#[cfg(test)]
mod benchmarks {
    // Note: Benchmarks will be added to benches/ directory for Criterion integration
}