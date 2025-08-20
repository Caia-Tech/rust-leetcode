//! # Problem 14: Longest Common Prefix
//!
//! Write a function to find the longest common prefix string amongst an array of strings.
//! If there is no common prefix, return an empty string `""`.
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::easy::longest_common_prefix::Solution;
//! 
//! let solution = Solution::new();
//! 
//! // Example 1:
//! let strs = vec!["flower".to_string(), "flow".to_string(), "flight".to_string()];
//! assert_eq!(solution.longest_common_prefix(strs), "fl".to_string());
//! 
//! // Example 2: 
//! let strs = vec!["dog".to_string(), "racecar".to_string(), "car".to_string()];
//! assert_eq!(solution.longest_common_prefix(strs), "".to_string());
//! ```
//!
//! ## Constraints
//!
//! - 1 <= strs.length <= 200
//! - 0 <= strs[i].length <= 200
//! - strs[i] consists of only lowercase English letters.

/// Solution struct following LeetCode format
pub struct Solution;

impl Solution {
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Horizontal Scanning (Optimal for most cases)
    /// 
    /// **Algorithm:**
    /// 1. Start with the first string as the initial prefix
    /// 2. For each subsequent string, find the common prefix
    /// 3. Update the prefix to be the shorter common part
    /// 4. If prefix becomes empty, return immediately
    /// 
    /// **Time Complexity:** O(S) where S is the sum of all characters in all strings
    /// **Space Complexity:** O(1) - Only using constant extra space (excluding output)
    /// 
    /// **Key Insight:** We can stop early if the prefix becomes empty, as no further
    /// strings can extend an empty prefix.
    /// 
    /// **Best Case:** O(n*m) where n is number of strings and m is length of shortest string
    /// **Worst Case:** O(S) when we need to examine all characters
    pub fn longest_common_prefix(&self, strs: Vec<String>) -> String {
        if strs.is_empty() {
            return String::new();
        }
        
        let mut prefix = strs[0].clone();
        
        for i in 1..strs.len() {
            // Find common prefix between current prefix and strs[i]
            let mut j = 0;
            let chars1: Vec<char> = prefix.chars().collect();
            let chars2: Vec<char> = strs[i].chars().collect();
            
            while j < chars1.len() && j < chars2.len() && chars1[j] == chars2[j] {
                j += 1;
            }
            
            prefix = chars1[..j].iter().collect();
            
            // Early termination: if prefix is empty, no point continuing
            if prefix.is_empty() {
                return prefix;
            }
        }
        
        prefix
    }

    /// # Approach 2: Vertical Scanning
    /// 
    /// **Algorithm:**
    /// 1. Compare characters column by column
    /// 2. For each character position, check if all strings have the same character
    /// 3. Stop when we find a mismatch or reach end of any string
    /// 
    /// **Time Complexity:** O(S) - Same as horizontal, but with different access pattern
    /// **Space Complexity:** O(1) - Constant extra space
    /// 
    /// **Advantage:** Better cache locality, stops as soon as a mismatch is found
    /// **Use Case:** Good when strings are long but prefix is expected to be short
    pub fn longest_common_prefix_vertical(&self, strs: Vec<String>) -> String {
        if strs.is_empty() {
            return String::new();
        }
        
        let first_str = &strs[0];
        
        for (i, ch) in first_str.char_indices() {
            // Check if all other strings have the same character at position i
            for s in &strs[1..] {
                if i >= s.len() || s.chars().nth(i) != Some(ch) {
                    return first_str[..i].to_string();
                }
            }
        }
        
        first_str.clone()
    }

    /// # Approach 3: Divide and Conquer
    /// 
    /// **Algorithm:**
    /// 1. Divide the array into two halves
    /// 2. Recursively find LCP of left half and right half
    /// 3. Merge by finding LCP of the two results
    /// 
    /// **Time Complexity:** O(S) - Each character is examined exactly once
    /// **Space Complexity:** O(m * log n) - Recursion depth * string length
    /// 
    /// **Use Case:** Educational; shows divide-and-conquer thinking
    /// **Performance:** Generally slower due to string allocation overhead
    pub fn longest_common_prefix_divide_conquer(&self, strs: Vec<String>) -> String {
        if strs.is_empty() {
            return String::new();
        }
        
        self.lcp_helper(&strs, 0, strs.len() - 1)
    }
    
    fn lcp_helper(&self, strs: &[String], left: usize, right: usize) -> String {
        if left == right {
            return strs[left].clone();
        }
        
        let mid = left + (right - left) / 2;
        let lcp_left = self.lcp_helper(strs, left, mid);
        let lcp_right = self.lcp_helper(strs, mid + 1, right);
        
        self.common_prefix(&lcp_left, &lcp_right)
    }
    
    fn common_prefix(&self, str1: &str, str2: &str) -> String {
        let chars1: Vec<char> = str1.chars().collect();
        let chars2: Vec<char> = str2.chars().collect();
        let mut i = 0;
        
        while i < chars1.len() && i < chars2.len() && chars1[i] == chars2[i] {
            i += 1;
        }
        
        chars1[..i].iter().collect()
    }

    /// # Approach 4: Trie-based (Most Complex)
    /// 
    /// **Algorithm:**
    /// 1. Build a trie with all strings
    /// 2. Traverse from root until we encounter a node with != 1 child
    /// 3. The path traversed is the longest common prefix
    /// 
    /// **Time Complexity:** O(S) for building trie + O(m) for traversal
    /// **Space Complexity:** O(S) for the trie structure
    /// 
    /// **Use Case:** When you need to perform multiple LCP queries on the same dataset
    /// **Overhead:** Significant space and setup cost for single queries
    pub fn longest_common_prefix_trie(&self, strs: Vec<String>) -> String {
        if strs.is_empty() {
            return String::new();
        }
        
        let mut trie = TrieNode::new();
        
        // Build trie
        for s in &strs {
            trie.insert(s);
        }
        
        // Find LCP by traversing until we hit a node with != 1 child or end of string
        let mut current = &trie;
        let mut prefix = String::new();
        
        while current.children.len() == 1 && !current.is_end {
            let (&ch, child) = current.children.iter().next().unwrap();
            prefix.push(ch);
            current = child;
        }
        
        prefix
    }

    /// # Approach 5: Binary Search on Answer Length
    /// 
    /// **Algorithm:**
    /// 1. Binary search on the length of the common prefix (0 to min_length)
    /// 2. For each candidate length, check if all strings share that prefix
    /// 3. Find the maximum valid length
    /// 
    /// **Time Complexity:** O(S * log m) where m is length of shortest string
    /// **Space Complexity:** O(1) - Constant extra space
    /// 
    /// **Use Case:** Academic interest; generally slower than simpler approaches
    pub fn longest_common_prefix_binary_search(&self, strs: Vec<String>) -> String {
        if strs.is_empty() {
            return String::new();
        }
        
        let min_length = strs.iter().map(|s| s.len()).min().unwrap_or(0);
        let mut left = 0;
        let mut right = min_length;
        
        while left < right {
            let mid = left + (right - left + 1) / 2;
            
            if self.is_common_prefix(&strs, mid) {
                left = mid;
            } else {
                right = mid - 1;
            }
        }
        
        strs[0][..left].to_string()
    }
    
    fn is_common_prefix(&self, strs: &[String], length: usize) -> bool {
        let prefix = &strs[0][..length];
        strs.iter().all(|s| s.starts_with(prefix))
    }
}

/// Simple Trie implementation for the trie-based approach
#[derive(Default)]
struct TrieNode {
    children: std::collections::HashMap<char, TrieNode>,
    is_end: bool,
}

impl TrieNode {
    fn new() -> Self {
        Self::default()
    }
    
    fn insert(&mut self, word: &str) {
        let mut current = self;
        for ch in word.chars() {
            current = current.children.entry(ch).or_insert_with(TrieNode::new);
        }
        current.is_end = true;
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
    #[case(vec!["flower", "flow", "flight"], "fl")]
    #[case(vec!["dog", "racecar", "car"], "")]
    #[case(vec!["interspecies", "interstellar", "interstate"], "inters")]
    #[case(vec!["throne", "throne"], "throne")]
    #[case(vec![""], "")]
    fn test_basic_cases(#[case] input: Vec<&str>, #[case] expected: &str) {
        let solution = setup();
        let strs: Vec<String> = input.iter().map(|s| s.to_string()).collect();
        assert_eq!(solution.longest_common_prefix(strs), expected.to_string());
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // Single string
        let single = vec!["hello".to_string()];
        assert_eq!(solution.longest_common_prefix(single), "hello");
        
        // Empty string in array
        let with_empty = vec!["hello".to_string(), "".to_string(), "help".to_string()];
        assert_eq!(solution.longest_common_prefix(with_empty), "");
        
        // All same strings
        let all_same = vec!["test".to_string(), "test".to_string(), "test".to_string()];
        assert_eq!(solution.longest_common_prefix(all_same), "test");
        
        // No common prefix
        let no_common = vec!["abc".to_string(), "def".to_string(), "ghi".to_string()];
        assert_eq!(solution.longest_common_prefix(no_common), "");
    }

    #[test]
    fn test_varying_lengths() {
        let solution = setup();
        
        // Shortest string determines max possible prefix
        let varying = vec![
            "application".to_string(),
            "app".to_string(),
            "apple".to_string(),
            "apply".to_string()
        ];
        assert_eq!(solution.longest_common_prefix(varying), "app");
        
        // Long common prefix
        let long_prefix = vec![
            "abcdefghijklmnop".to_string(),
            "abcdefghijklmnopqrs".to_string(),
            "abcdefghijklmnopq".to_string()
        ];
        assert_eq!(solution.longest_common_prefix(long_prefix), "abcdefghijklmnop");
    }

    #[test]
    fn test_special_characters_and_case_sensitivity() {
        let solution = setup();
        
        // Case sensitivity
        let case_sensitive = vec!["Apple".to_string(), "apple".to_string()];
        assert_eq!(solution.longest_common_prefix(case_sensitive), "");
        
        // Single character differences
        let single_diff = vec!["aa".to_string(), "ab".to_string()];
        assert_eq!(solution.longest_common_prefix(single_diff), "a");
    }

    #[test]
    fn test_all_approaches_consistency() {
        let solution = setup();
        let test_cases = vec![
            vec!["flower", "flow", "flight"],
            vec!["dog", "racecar", "car"],
            vec!["interspecies", "interstellar", "interstate"],
            vec!["throne", "throne"],
            vec![""],
            vec!["a"],
            vec!["abc", "def"],
            vec!["application", "app", "apple", "apply"]
        ];
        
        for case in test_cases {
            let strs: Vec<String> = case.iter().map(|s| s.to_string()).collect();
            
            let result1 = solution.longest_common_prefix(strs.clone());
            let result2 = solution.longest_common_prefix_vertical(strs.clone());
            let result3 = solution.longest_common_prefix_divide_conquer(strs.clone());
            let result4 = solution.longest_common_prefix_trie(strs.clone());
            let result5 = solution.longest_common_prefix_binary_search(strs.clone());
            
            assert_eq!(result1, result2, "Vertical approach differs for {:?}", case);
            assert_eq!(result1, result3, "Divide-conquer approach differs for {:?}", case);
            assert_eq!(result1, result4, "Trie approach differs for {:?}", case);
            assert_eq!(result1, result5, "Binary search approach differs for {:?}", case);
        }
    }

    #[test]
    fn test_performance_edge_cases() {
        let solution = setup();
        
        // Many strings with short common prefix
        let many_strings: Vec<String> = (0..100)
            .map(|i| format!("prefix{}", i))
            .collect();
        assert_eq!(solution.longest_common_prefix(many_strings), "prefix");
        
        // Long strings with no common prefix
        let long_no_prefix = vec![
            "a".repeat(100),
            "b".repeat(100),
        ];
        assert_eq!(solution.longest_common_prefix(long_no_prefix), "");
        
        // Long strings with long common prefix
        let long_with_prefix = vec![
            format!("{}a", "common".repeat(20)),
            format!("{}b", "common".repeat(20)),
            format!("{}c", "common".repeat(20)),
        ];
        assert_eq!(solution.longest_common_prefix(long_with_prefix), "common".repeat(20));
    }

    #[test]
    fn test_algorithmic_properties() {
        let solution = setup();
        
        // Test that order doesn't matter (commutative property)
        let strs1 = vec!["abc".to_string(), "ab".to_string(), "abcd".to_string()];
        let strs2 = vec!["ab".to_string(), "abcd".to_string(), "abc".to_string()];
        let strs3 = vec!["abcd".to_string(), "abc".to_string(), "ab".to_string()];
        
        let result1 = solution.longest_common_prefix(strs1);
        let result2 = solution.longest_common_prefix(strs2);
        let result3 = solution.longest_common_prefix(strs3);
        
        assert_eq!(result1, result2);
        assert_eq!(result2, result3);
        assert_eq!(result1, "ab");
    }
}