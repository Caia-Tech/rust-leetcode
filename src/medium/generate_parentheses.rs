//! Problem 22: Generate Parentheses
//! 
//! Given n pairs of parentheses, write a function to generate all combinations
//! of well-formed parentheses.
//! 
//! Example 1:
//! Input: n = 3
//! Output: ["((()))","(()())","(())()","()(())","()()()"]
//! 
//! Example 2:
//! Input: n = 1
//! Output: ["()"]

use std::collections::VecDeque;

pub struct Solution;

impl Solution {
    /// Approach 1: Classic Backtracking with Valid Tracking
    /// 
    /// Uses backtracking to build valid parentheses by tracking open and close counts.
    /// Only adds '(' if we haven't used all n opening parentheses.
    /// Only adds ')' if it won't exceed the number of '(' already placed.
    /// 
    /// Time Complexity: O(4^n / √n) - Catalan number bounds
    /// Space Complexity: O(4^n / √n) for result + O(n) for recursion stack
    pub fn generate_parenthesis_backtrack(&self, n: i32) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = String::new();
        self.backtrack(n as usize, 0, 0, &mut current, &mut result);
        result
    }
    
    fn backtrack(&self, n: usize, open: usize, close: usize, current: &mut String, result: &mut Vec<String>) {
        // Base case: we've used all n pairs
        if current.len() == 2 * n {
            result.push(current.clone());
            return;
        }
        
        // Add '(' if we haven't used all n opening parentheses
        if open < n {
            current.push('(');
            self.backtrack(n, open + 1, close, current, result);
            current.pop();
        }
        
        // Add ')' if it doesn't exceed the number of '(' already placed
        if close < open {
            current.push(')');
            self.backtrack(n, open, close + 1, current, result);
            current.pop();
        }
    }
    
    /// Approach 2: BFS with State Tracking
    /// 
    /// Uses breadth-first search to generate all valid combinations level by level.
    /// Each state contains the current string, open count, and close count.
    /// 
    /// Time Complexity: O(4^n / √n)
    /// Space Complexity: O(4^n / √n) for queue and result
    pub fn generate_parenthesis_bfs(&self, n: i32) -> Vec<String> {
        let n = n as usize;
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        
        // State: (current_string, open_count, close_count)
        queue.push_back((String::new(), 0, 0));
        
        while let Some((current, open, close)) = queue.pop_front() {
            if current.len() == 2 * n {
                result.push(current);
                continue;
            }
            
            // Add '(' if possible
            if open < n {
                let mut next = current.clone();
                next.push('(');
                queue.push_back((next, open + 1, close));
            }
            
            // Add ')' if possible
            if close < open {
                let mut next = current.clone();
                next.push(')');
                queue.push_back((next, open, close + 1));
            }
        }
        
        result
    }
    
    /// Approach 3: Dynamic Programming with Memoization
    /// 
    /// Builds solutions incrementally using previously computed results.
    /// For n pairs, we can form combinations by choosing where to place
    /// the first complete pair and recursively solving subproblems.
    /// 
    /// Time Complexity: O(4^n / √n)
    /// Space Complexity: O(4^n / √n) for memoization table
    pub fn generate_parenthesis_dp(&self, n: i32) -> Vec<String> {
        let mut memo = vec![Vec::new(); (n + 1) as usize];
        self.dp_helper(n as usize, &mut memo)
    }
    
    fn dp_helper(&self, n: usize, memo: &mut Vec<Vec<String>>) -> Vec<String> {
        if !memo[n].is_empty() {
            return memo[n].clone();
        }
        
        if n == 0 {
            memo[0] = vec![String::new()];
            return memo[0].clone();
        }
        
        let mut result = Vec::new();
        
        // For each way to split n pairs
        for i in 0..n {
            let left = self.dp_helper(i, memo);
            let right = self.dp_helper(n - 1 - i, memo);
            
            for l in &left {
                for r in &right {
                    result.push(format!("({}){}", l, r));
                }
            }
        }
        
        memo[n] = result.clone();
        result
    }
    
    /// Approach 4: Iterative with Stack Simulation
    /// 
    /// Simulates the backtracking process using an explicit stack.
    /// Each stack entry contains the current state and next action to take.
    /// 
    /// Time Complexity: O(4^n / √n)
    /// Space Complexity: O(4^n / √n) for stack and result
    pub fn generate_parenthesis_iterative(&self, n: i32) -> Vec<String> {
        let n = n as usize;
        let mut result = Vec::new();
        let mut stack = Vec::new();
        
        // State: (current_string, open_count, close_count)
        stack.push((String::new(), 0, 0));
        
        while let Some((current, open, close)) = stack.pop() {
            if current.len() == 2 * n {
                result.push(current);
                continue;
            }
            
            // Try adding ')' first (will be processed after '(' due to stack order)
            if close < open {
                let mut next = current.clone();
                next.push(')');
                stack.push((next, open, close + 1));
            }
            
            // Try adding '('
            if open < n {
                let mut next = current.clone();
                next.push('(');
                stack.push((next, open + 1, close));
            }
        }
        
        result
    }
    
    /// Approach 5: Closure-based Generation
    /// 
    /// Generates parentheses by thinking of valid combinations as
    /// placing pairs of parentheses in different "closure" positions.
    /// Each valid string can be decomposed as (A)B where A and B are valid strings.
    /// 
    /// Time Complexity: O(4^n / √n)
    /// Space Complexity: O(4^n / √n) for result + O(n) for recursion
    pub fn generate_parenthesis_closure(&self, n: i32) -> Vec<String> {
        if n == 0 {
            return vec![String::new()];
        }
        
        let mut result = Vec::new();
        
        // For each way to place the first pair of parentheses
        for i in 0..n {
            // Generate all valid strings with i pairs inside the first pair
            let inside = self.generate_parenthesis_closure(i);
            // Generate all valid strings with (n-1-i) pairs after the first pair
            let outside = self.generate_parenthesis_closure(n - 1 - i);
            
            for inner in &inside {
                for outer in &outside {
                    result.push(format!("({}){}", inner, outer));
                }
            }
        }
        
        result
    }
    
    /// Approach 6: Binary Generation with Validation
    /// 
    /// Generates all possible binary strings of length 2n where 0='(' and 1=')'.
    /// Validates each string to check if it forms valid parentheses.
    /// Less efficient but demonstrates a different approach.
    /// 
    /// Time Complexity: O(2^(2n) * n) - generates all 2^(2n) strings and validates each
    /// Space Complexity: O(4^n / √n) for valid results + O(n) for validation
    pub fn generate_parenthesis_binary(&self, n: i32) -> Vec<String> {
        let n = n as usize;
        let mut result = Vec::new();
        let total_length = 2 * n;
        
        // Generate all possible combinations
        for i in 0..(1 << total_length) {
            let mut candidate = String::new();
            let mut open_count = 0;
            
            for j in 0..total_length {
                if (i >> j) & 1 == 0 {
                    candidate.push('(');
                    open_count += 1;
                } else {
                    candidate.push(')');
                }
            }
            
            // Check if this candidate has exactly n '(' and n ')'
            if open_count == n && self.is_valid_parentheses(&candidate) {
                result.push(candidate);
            }
        }
        
        result
    }
    
    fn is_valid_parentheses(&self, s: &str) -> bool {
        let mut balance = 0;
        for c in s.chars() {
            if c == '(' {
                balance += 1;
            } else {
                balance -= 1;
                if balance < 0 {
                    return false;
                }
            }
        }
        balance == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn sort_strings(mut vec: Vec<String>) -> Vec<String> {
        vec.sort();
        vec
    }
    
    #[test]
    fn test_backtrack() {
        let solution = Solution;
        
        assert_eq!(sort_strings(solution.generate_parenthesis_backtrack(1)), vec!["()"]);
        assert_eq!(sort_strings(solution.generate_parenthesis_backtrack(2)), 
                   vec!["(())", "()()"]);
        assert_eq!(sort_strings(solution.generate_parenthesis_backtrack(3)), 
                   vec!["((()))", "(()())", "(())()", "()(())", "()()()"]);
        assert_eq!(solution.generate_parenthesis_backtrack(0), vec![""]);
    }
    
    #[test]
    fn test_bfs() {
        let solution = Solution;
        
        assert_eq!(sort_strings(solution.generate_parenthesis_bfs(1)), vec!["()"]);
        assert_eq!(sort_strings(solution.generate_parenthesis_bfs(2)), 
                   vec!["(())", "()()"]);
        assert_eq!(sort_strings(solution.generate_parenthesis_bfs(3)), 
                   vec!["((()))", "(()())", "(())()", "()(())", "()()()"]);
        assert_eq!(solution.generate_parenthesis_bfs(0), vec![""]);
    }
    
    #[test]
    fn test_dp() {
        let solution = Solution;
        
        assert_eq!(sort_strings(solution.generate_parenthesis_dp(1)), vec!["()"]);
        assert_eq!(sort_strings(solution.generate_parenthesis_dp(2)), 
                   vec!["(())", "()()"]);
        assert_eq!(sort_strings(solution.generate_parenthesis_dp(3)), 
                   vec!["((()))", "(()())", "(())()", "()(())", "()()()"]);
        assert_eq!(solution.generate_parenthesis_dp(0), vec![""]);
    }
    
    #[test]
    fn test_iterative() {
        let solution = Solution;
        
        assert_eq!(sort_strings(solution.generate_parenthesis_iterative(1)), vec!["()"]);
        assert_eq!(sort_strings(solution.generate_parenthesis_iterative(2)), 
                   vec!["(())", "()()"]);
        assert_eq!(sort_strings(solution.generate_parenthesis_iterative(3)), 
                   vec!["((()))", "(()())", "(())()", "()(())", "()()()"]);
        assert_eq!(solution.generate_parenthesis_iterative(0), vec![""]);
    }
    
    #[test]
    fn test_closure() {
        let solution = Solution;
        
        assert_eq!(sort_strings(solution.generate_parenthesis_closure(1)), vec!["()"]);
        assert_eq!(sort_strings(solution.generate_parenthesis_closure(2)), 
                   vec!["(())", "()()"]);
        assert_eq!(sort_strings(solution.generate_parenthesis_closure(3)), 
                   vec!["((()))", "(()())", "(())()", "()(())", "()()()"]);
        assert_eq!(solution.generate_parenthesis_closure(0), vec![""]);
    }
    
    #[test]
    fn test_binary() {
        let solution = Solution;
        
        assert_eq!(sort_strings(solution.generate_parenthesis_binary(1)), vec!["()"]);
        assert_eq!(sort_strings(solution.generate_parenthesis_binary(2)), 
                   vec!["(())", "()()"]);
        assert_eq!(sort_strings(solution.generate_parenthesis_binary(3)), 
                   vec!["((()))", "(()())", "(())()", "()(())", "()()()"]);
        assert_eq!(solution.generate_parenthesis_binary(0), vec![""]);
    }
    
    #[test]
    fn test_edge_cases() {
        let solution = Solution;
        
        // Test n = 4 for more complex case
        let result_4 = solution.generate_parenthesis_backtrack(4);
        assert_eq!(result_4.len(), 14); // Catalan number C(4) = 14
        
        // Verify all results have correct length
        for s in &result_4 {
            assert_eq!(s.len(), 8);
            assert_eq!(s.chars().filter(|&c| c == '(').count(), 4);
            assert_eq!(s.chars().filter(|&c| c == ')').count(), 4);
        }
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let solution = Solution;
        
        for n in 0..=4 {
            let backtrack = sort_strings(solution.generate_parenthesis_backtrack(n));
            let bfs = sort_strings(solution.generate_parenthesis_bfs(n));
            let dp = sort_strings(solution.generate_parenthesis_dp(n));
            let iterative = sort_strings(solution.generate_parenthesis_iterative(n));
            let closure = sort_strings(solution.generate_parenthesis_closure(n));
            let binary = sort_strings(solution.generate_parenthesis_binary(n));
            
            assert_eq!(backtrack, bfs, "Backtrack and BFS differ for n={}", n);
            assert_eq!(backtrack, dp, "Backtrack and DP differ for n={}", n);
            assert_eq!(backtrack, iterative, "Backtrack and Iterative differ for n={}", n);
            assert_eq!(backtrack, closure, "Backtrack and Closure differ for n={}", n);
            assert_eq!(backtrack, binary, "Backtrack and Binary differ for n={}", n);
        }
    }
    
    #[test]
    fn test_catalan_numbers() {
        let solution = Solution;
        
        // Test that the number of results matches Catalan numbers
        let catalan = [1, 1, 2, 5, 14, 42]; // C(0) through C(5)
        
        for (n, &expected_count) in catalan.iter().enumerate() {
            let result = solution.generate_parenthesis_backtrack(n as i32);
            assert_eq!(result.len(), expected_count, 
                      "Wrong count for n={}, expected {}, got {}", 
                      n, expected_count, result.len());
        }
    }
    
    #[test]
    fn test_all_valid() {
        let solution = Solution;
        
        for n in 0..=4 {
            let results = solution.generate_parenthesis_backtrack(n);
            
            for result in results {
                assert!(solution.is_valid_parentheses(&result), 
                       "Invalid parentheses: {}", result);
                assert_eq!(result.chars().filter(|&c| c == '(').count(), n as usize);
                assert_eq!(result.chars().filter(|&c| c == ')').count(), n as usize);
            }
        }
    }
}