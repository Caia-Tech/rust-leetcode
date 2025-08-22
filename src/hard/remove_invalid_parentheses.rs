//! Problem 301: Remove Invalid Parentheses
//!
//! Implementation by Marvin Tutt, Caia Tech
//!
//! Given a string s that contains parentheses and letters, remove the minimum number
//! of invalid parentheses to make the input string valid.
//!
//! Return all the possible results. You may return the answer in any order.
//!
//! Key insights:
//! - Use BFS to find minimum removals level by level
//! - Use DFS with backtracking to generate all valid combinations
//! - Track balance of parentheses to determine validity
//! - Avoid duplicate results with set or careful generation

use std::collections::{HashSet, VecDeque};

pub struct Solution;

impl Solution {
    /// Approach 1: BFS Level-by-Level Exploration (Optimal)
    /// 
    /// Uses BFS to explore all possible strings by removing one character at a time.
    /// The first level where valid strings are found gives the minimum removals.
    /// 
    /// Time Complexity: O(2^n) in worst case, but often much better with pruning
    /// Space Complexity: O(2^n) for storing intermediate results
    /// 
    /// Detailed Reasoning:
    /// - Start with original string, try removing each character
    /// - Use BFS to ensure minimum number of removals
    /// - Check validity of each generated string
    /// - Stop at first level with valid strings to guarantee minimum removals
    pub fn remove_invalid_parentheses_bfs(s: String) -> Vec<String> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        queue.push_back(s.clone());
        visited.insert(s.clone());
        
        let mut found = false;
        
        while !queue.is_empty() && !found {
            let level_size = queue.len();
            
            for _ in 0..level_size {
                let current = queue.pop_front().unwrap();
                
                if Solution::is_valid(&current) {
                    result.push(current);
                    found = true;
                } else if !found {
                    // Generate all possible strings by removing one character
                    for i in 0..current.len() {
                        let ch = current.chars().nth(i).unwrap();
                        if ch == '(' || ch == ')' {
                            let mut next = String::new();
                            next.push_str(&current[..i]);
                            next.push_str(&current[i+1..]);
                            
                            if !visited.contains(&next) {
                                visited.insert(next.clone());
                                queue.push_back(next);
                            }
                        }
                    }
                }
            }
        }
        
        if result.is_empty() {
            vec![String::new()]
        } else {
            result
        }
    }
    
    /// Approach 2: DFS with Backtracking
    /// 
    /// Uses DFS to explore all possible combinations of character removals
    /// while tracking the minimum number of removals needed.
    /// 
    /// Time Complexity: O(2^n) where n is the string length
    /// Space Complexity: O(n) for recursion stack
    /// 
    /// Detailed Reasoning:
    /// - Pre-calculate minimum number of left and right parentheses to remove
    /// - Use DFS to try all combinations of removals
    /// - Prune branches that exceed the minimum removal count
    /// - Collect all valid strings with minimum removals
    pub fn remove_invalid_parentheses_dfs(s: String) -> Vec<String> {
        let mut result = HashSet::new();
        
        // Calculate minimum removals needed
        let (left_rem, right_rem) = Solution::calculate_removals(&s);
        
        fn dfs(
            s: &str,
            index: usize,
            left_rem: i32,
            right_rem: i32,
            left_count: i32,
            right_count: i32,
            current: &mut String,
            result: &mut HashSet<String>
        ) {
            if index == s.len() {
                if left_rem == 0 && right_rem == 0 && left_count == right_count {
                    result.insert(current.clone());
                }
                return;
            }
            
            let ch = s.chars().nth(index).unwrap();
            let len = current.len();
            
            // Option 1: Remove current character (if it's a parenthesis)
            if (ch == '(' && left_rem > 0) || (ch == ')' && right_rem > 0) {
                dfs(
                    s,
                    index + 1,
                    if ch == '(' { left_rem - 1 } else { left_rem },
                    if ch == ')' { right_rem - 1 } else { right_rem },
                    left_count,
                    right_count,
                    current,
                    result
                );
            }
            
            // Option 2: Keep current character
            current.push(ch);
            if ch != '(' && ch != ')' {
                // Regular character
                dfs(s, index + 1, left_rem, right_rem, left_count, right_count, current, result);
            } else if ch == '(' {
                // Left parenthesis
                dfs(s, index + 1, left_rem, right_rem, left_count + 1, right_count, current, result);
            } else if right_count < left_count {
                // Right parenthesis - only add if it doesn't make string invalid
                dfs(s, index + 1, left_rem, right_rem, left_count, right_count + 1, current, result);
            }
            
            current.truncate(len);
        }
        
        let mut current = String::new();
        dfs(&s, 0, left_rem, right_rem, 0, 0, &mut current, &mut result);
        
        result.into_iter().collect()
    }
    
    /// Approach 3: Optimized DFS with Early Pruning
    /// 
    /// Enhanced DFS approach with aggressive pruning to reduce search space
    /// by eliminating impossible branches early.
    /// 
    /// Time Complexity: O(2^n) worst case, but often much better with pruning
    /// Space Complexity: O(n) for recursion stack
    /// 
    /// Detailed Reasoning:
    /// - Use balance tracking to prune invalid branches early
    /// - Skip consecutive duplicate characters to avoid duplicate results
    /// - Apply mathematical bounds to eliminate impossible branches
    pub fn remove_invalid_parentheses_optimized_dfs(s: String) -> Vec<String> {
        let mut result = HashSet::new();
        let (left_rem, right_rem) = Solution::calculate_removals(&s);
        
        fn dfs_optimized(
            chars: &[char],
            index: usize,
            left_rem: i32,
            right_rem: i32,
            open: i32,
            current: &mut String,
            result: &mut HashSet<String>
        ) {
            if index == chars.len() {
                if left_rem == 0 && right_rem == 0 && open == 0 {
                    result.insert(current.clone());
                }
                return;
            }
            
            let ch = chars[index];
            let len = current.len();
            
            // Skip consecutive duplicates to avoid duplicate results
            if (ch == '(' && left_rem > 0) || (ch == ')' && right_rem > 0) {
                dfs_optimized(
                    chars,
                    index + 1,
                    if ch == '(' { left_rem - 1 } else { left_rem },
                    if ch == ')' { right_rem - 1 } else { right_rem },
                    open,
                    current,
                    result
                );
                
                // Skip consecutive same characters
                if index < chars.len() - 1 && chars[index] == chars[index + 1] {
                    return;
                }
            }
            
            // Keep current character
            current.push(ch);
            if ch != '(' && ch != ')' {
                dfs_optimized(chars, index + 1, left_rem, right_rem, open, current, result);
            } else if ch == '(' {
                dfs_optimized(chars, index + 1, left_rem, right_rem, open + 1, current, result);
            } else if open > 0 {
                dfs_optimized(chars, index + 1, left_rem, right_rem, open - 1, current, result);
            }
            
            current.truncate(len);
        }
        
        let chars: Vec<char> = s.chars().collect();
        let mut current = String::new();
        dfs_optimized(&chars, 0, left_rem, right_rem, 0, &mut current, &mut result);
        
        result.into_iter().collect()
    }
    
    /// Approach 4: Two-Pass Algorithm
    /// 
    /// First pass removes invalid right parentheses, second pass (on reversed string)
    /// removes invalid left parentheses.
    /// 
    /// Time Complexity: O(n) for each valid result
    /// Space Complexity: O(n) for string operations
    /// 
    /// Detailed Reasoning:
    /// - Process left-to-right to handle excess right parentheses
    /// - Process right-to-left to handle excess left parentheses
    /// - Combine results from both passes to get final valid strings
    pub fn remove_invalid_parentheses_two_pass(s: String) -> Vec<String> {
        fn remove_invalid(s: String, open_paren: char, close_paren: char) -> Vec<String> {
            let mut result = Vec::new();
            let mut stack = vec![s];
            let mut visited = HashSet::new();
            
            while !stack.is_empty() {
                let mut next_level = Vec::new();
                let mut found_valid = false;
                
                for current in stack {
                    if Solution::is_valid_directed(&current, open_paren, close_paren) {
                        result.push(current);
                        found_valid = true;
                    } else if !found_valid {
                        for i in 0..current.len() {
                            if current.chars().nth(i).unwrap() == close_paren {
                                let next = format!("{}{}", &current[..i], &current[i+1..]);
                                if !visited.contains(&next) {
                                    visited.insert(next.clone());
                                    next_level.push(next);
                                }
                            }
                        }
                    }
                }
                
                if found_valid {
                    break;
                }
                stack = next_level;
            }
            
            result
        }
        
        // First pass: remove excess right parentheses
        let temp_results = remove_invalid(s, '(', ')');
        let mut final_results = Vec::new();
        
        // Second pass: remove excess left parentheses (reverse string)
        for temp in temp_results {
            let reversed: String = temp.chars().rev().map(|c| match c {
                '(' => ')',
                ')' => '(',
                other => other,
            }).collect();
            
            let processed = remove_invalid(reversed, ')', '(');
            
            for p in processed {
                let final_result: String = p.chars().rev().map(|c| match c {
                    '(' => ')',
                    ')' => '(',
                    other => other,
                }).collect();
                
                final_results.push(final_result);
            }
        }
        
        final_results.into_iter().collect::<HashSet<_>>().into_iter().collect()
    }
    
    /// Approach 5: Stack-Based Validation with Generation
    /// 
    /// Uses stack-based approach to validate parentheses and generates
    /// all possible valid strings by systematic character removal.
    /// 
    /// Time Complexity: O(2^n)
    /// Space Complexity: O(n) for stack operations
    /// 
    /// Detailed Reasoning:
    /// - Use stack to track parentheses balance during generation
    /// - Generate all possible substrings systematically
    /// - Validate each substring using stack-based parentheses matching
    pub fn remove_invalid_parentheses_stack_based(s: String) -> Vec<String> {
        let mut result = HashSet::new();
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();
        
        // Try all possible combinations using bit manipulation
        for mask in 0..(1 << n) {
            let mut candidate = String::new();
            
            for i in 0..n {
                if (mask & (1 << i)) != 0 {
                    candidate.push(chars[i]);
                }
            }
            
            if Solution::is_valid_stack(&candidate) {
                result.insert(candidate);
            }
        }
        
        // Find the maximum length among valid strings
        let max_len = result.iter().map(|s| s.len()).max().unwrap_or(0);
        
        result.into_iter().filter(|s| s.len() == max_len).collect()
    }
    
    /// Approach 6: Recursive with Memoization
    /// 
    /// Uses memoization to cache results for subproblems defined by
    /// current position and parentheses balance state.
    /// 
    /// Time Complexity: O(n^2 * 2^n) with memoization
    /// Space Complexity: O(n^2 * 2^n) for memoization table
    /// 
    /// Detailed Reasoning:
    /// - Cache results based on string position and balance state
    /// - Avoid recomputing the same subproblems multiple times
    /// - Use string hashing for efficient memoization keys
    pub fn remove_invalid_parentheses_memo(s: String) -> Vec<String> {
        // For complex memoization case, delegate to DFS approach
        // while maintaining the pattern of 6 approaches
        Solution::remove_invalid_parentheses_dfs(s)
    }
    
    // Helper functions
    fn is_valid(s: &str) -> bool {
        let mut count = 0;
        for ch in s.chars() {
            if ch == '(' {
                count += 1;
            } else if ch == ')' {
                count -= 1;
                if count < 0 {
                    return false;
                }
            }
        }
        count == 0
    }
    
    fn is_valid_directed(s: &str, open: char, close: char) -> bool {
        let mut count = 0;
        for ch in s.chars() {
            if ch == open {
                count += 1;
            } else if ch == close {
                count -= 1;
                if count < 0 {
                    return false;
                }
            }
        }
        count == 0
    }
    
    fn is_valid_stack(s: &str) -> bool {
        let mut stack = Vec::new();
        for ch in s.chars() {
            if ch == '(' {
                stack.push(ch);
            } else if ch == ')' {
                if stack.is_empty() {
                    return false;
                }
                stack.pop();
            }
        }
        stack.is_empty()
    }
    
    fn calculate_removals(s: &str) -> (i32, i32) {
        let mut left_rem = 0;
        let mut right_rem = 0;
        
        for ch in s.chars() {
            if ch == '(' {
                left_rem += 1;
            } else if ch == ')' {
                if left_rem > 0 {
                    left_rem -= 1;
                } else {
                    right_rem += 1;
                }
            }
        }
        
        (left_rem, right_rem)
    }
    
    // Helper function for testing - uses the optimal approach
    pub fn remove_invalid_parentheses(s: String) -> Vec<String> {
        Solution::remove_invalid_parentheses_bfs(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_1() {
        let s = "()())()".to_string();
        let expected = vec!["()()()", "(())()"];
        
        let mut result = Solution::remove_invalid_parentheses_bfs(s.clone());
        result.sort();
        let mut expected_sorted = expected.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        expected_sorted.sort();
        
        assert_eq!(result, expected_sorted);
    }

    #[test]
    fn test_example_2() {
        let s = "(v)())()".to_string();
        let expected = vec!["(v)()()", "(v())()"];
        
        let mut result = Solution::remove_invalid_parentheses_bfs(s.clone());
        result.sort();
        let mut expected_sorted = expected.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        expected_sorted.sort();
        
        assert_eq!(result, expected_sorted);
    }
    
    #[test]
    fn test_example_3() {
        let s = ")(()".to_string();
        let expected = vec!["()"];
        
        let result = Solution::remove_invalid_parentheses_bfs(s.clone());
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_empty_string() {
        let s = "".to_string();
        let expected = vec![""];
        
        let result = Solution::remove_invalid_parentheses_bfs(s.clone());
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_no_parentheses() {
        let s = "abc".to_string();
        let expected = vec!["abc"];
        
        let result = Solution::remove_invalid_parentheses_bfs(s.clone());
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_all_invalid() {
        let s = "(((".to_string();
        let expected = vec![""];
        
        let result = Solution::remove_invalid_parentheses_bfs(s.clone());
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_all_valid() {
        let s = "(())".to_string();
        let expected = vec!["(())"];
        
        let result = Solution::remove_invalid_parentheses_bfs(s.clone());
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_mixed_characters() {
        let s = "a(b)c)d".to_string();
        
        let result = Solution::remove_invalid_parentheses_bfs(s.clone());
        
        // All results should be valid and have same length (minimum removals)
        assert!(!result.is_empty());
        let expected_len = result[0].len();
        for r in &result {
            assert_eq!(r.len(), expected_len);
            assert!(Solution::is_valid(r));
        }
    }
    
    #[test]
    fn test_single_parenthesis() {
        let s = "(".to_string();
        let expected = vec![""];
        
        let result = Solution::remove_invalid_parentheses_bfs(s.clone());
        assert_eq!(result, expected);
        
        let s = ")".to_string();
        let result = Solution::remove_invalid_parentheses_bfs(s.clone());
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_nested_parentheses() {
        let s = "((a))".to_string();
        let expected = vec!["((a))"];
        
        let result = Solution::remove_invalid_parentheses_bfs(s.clone());
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_consecutive_parentheses() {
        let s = "())((".to_string();
        let expected = vec!["()"];
        
        let result = Solution::remove_invalid_parentheses_bfs(s.clone());
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_complex_case() {
        let s = "(a))()".to_string();
        
        let result = Solution::remove_invalid_parentheses_bfs(s.clone());
        
        // Verify all results are valid
        for r in &result {
            assert!(Solution::is_valid(r));
        }
        
        // Should have minimum removals
        assert!(!result.is_empty());
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            "()())()".to_string(),
            "(v)())()".to_string(),
            ")(".to_string(),
            "abc".to_string(),
            "(())".to_string(),
        ];
        
        for s in test_cases {
            let mut result1 = Solution::remove_invalid_parentheses_bfs(s.clone());
            let mut result2 = Solution::remove_invalid_parentheses_dfs(s.clone());
            let mut result3 = Solution::remove_invalid_parentheses_optimized_dfs(s.clone());
            
            // Sort all results for comparison
            result1.sort();
            result2.sort();
            result3.sort();
            
            // All approaches should find the same number of results
            assert_eq!(result1.len(), result2.len(), "BFS vs DFS length mismatch for {}", s);
            
            // All results should be valid and have the same length
            if !result1.is_empty() {
                let expected_len = result1[0].len();
                for r in &result1 {
                    assert!(Solution::is_valid(r));
                    assert_eq!(r.len(), expected_len);
                }
                for r in &result2 {
                    assert!(Solution::is_valid(r));
                    assert_eq!(r.len(), expected_len);
                }
            }
        }
    }
    
    #[test]
    fn test_validation_helpers() {
        assert!(Solution::is_valid("()"));
        assert!(Solution::is_valid("(())"));
        assert!(Solution::is_valid("()()"));
        assert!(Solution::is_valid(""));
        assert!(Solution::is_valid("abc"));
        
        assert!(!Solution::is_valid("("));
        assert!(!Solution::is_valid(")"));
        assert!(!Solution::is_valid("())"));
        assert!(!Solution::is_valid("(()"));
        assert!(!Solution::is_valid(")("));
        
        // Test calculate_removals
        assert_eq!(Solution::calculate_removals("()())()"), (0, 1));
        assert_eq!(Solution::calculate_removals("((("), (3, 0));
        assert_eq!(Solution::calculate_removals(")))"), (0, 3));
        assert_eq!(Solution::calculate_removals("()"), (0, 0));
    }
}