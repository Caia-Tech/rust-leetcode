//! # Problem 39: Combination Sum
//!
//! Given an array of distinct integers `candidates` and a target integer `target`, return a list 
//! of all unique combinations of `candidates` where the chosen numbers sum to `target`. You may 
//! return the combinations in any order.
//!
//! The same number may be chosen from `candidates` an unlimited number of times. Two combinations 
//! are unique if the frequency of at least one of the chosen numbers is different.
//!
//! ## Examples
//!
//! ```text
//! Input: candidates = [2,3,6,7], target = 7
//! Output: [[2,2,3],[7]]
//! Explanation:
//! 2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
//! 7 is a candidate, and 7 = 7.
//! These are the only two combinations.
//! ```
//!
//! ```text
//! Input: candidates = [2,3,5], target = 8
//! Output: [[2,2,2,2],[2,3,3],[3,5]]
//! ```
//!
//! ```text
//! Input: candidates = [2], target = 1
//! Output: []
//! ```
//!
//! ## Constraints
//!
//! * 1 <= candidates.length <= 30
//! * 2 <= candidates[i] <= 40
//! * All elements of candidates are distinct
//! * 1 <= target <= 40

/// Solution for Combination Sum problem
pub struct Solution;

impl Solution {
    /// Creates a new instance of Solution
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Backtracking with Pruning (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Sort candidates for early termination
    /// 2. Use DFS to explore all valid combinations
    /// 3. At each step, decide to include current candidate or not
    /// 4. Allow reusing same candidate multiple times
    /// 5. Prune when remaining target becomes negative
    /// 
    /// **Time Complexity:** O(N^(T/M)) where N=candidates, T=target, M=minimal candidate
    /// **Space Complexity:** O(T/M) - Recursion depth and combination storage
    /// 
    /// **Key Insights:**
    /// - This is an unbounded knapsack variant
    /// - We can reuse elements unlimited times
    /// - Need to avoid duplicates by maintaining order
    /// - Sorting enables early pruning
    /// 
    /// **Why backtracking works:**
    /// - Systematically explores all possibilities
    /// - Builds combinations incrementally
    /// - Backtracks when path becomes invalid
    /// - Naturally handles unlimited reuse
    /// 
    /// **Pruning strategies:**
    /// - Stop when target becomes negative
    /// - Skip candidates larger than remaining target
    /// - Maintain order to avoid duplicate combinations
    /// 
    /// **Visualization:**
    /// ```text
    /// candidates = [2,3,6,7], target = 7
    ///              7
    ///            /   \
    ///       +2: 5     skip 2
    ///          / \      \
    ///    +2: 3   skip    ...
    ///       / \
    /// +2: 1   +3: 0 ✓ [2,2,3]
    ///    ✗
    /// ```
    pub fn combination_sum(&self, candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        let mut candidates = candidates;
        candidates.sort_unstable(); // Enable early pruning
        
        let mut result = Vec::new();
        let mut current = Vec::new();
        
        self.backtrack(&candidates, target, 0, &mut current, &mut result);
        result
    }
    
    fn backtrack(&self, candidates: &[i32], target: i32, start: usize, 
                 current: &mut Vec<i32>, result: &mut Vec<Vec<i32>>) {
        if target == 0 {
            result.push(current.clone());
            return;
        }
        
        for i in start..candidates.len() {
            let candidate = candidates[i];
            
            // Early pruning: if current candidate > target, 
            // all remaining will also be > target (sorted array)
            if candidate > target {
                break;
            }
            
            current.push(candidate);
            // Allow reusing same element: pass i (not i+1)
            self.backtrack(candidates, target - candidate, i, current, result);
            current.pop();
        }
    }

    /// # Approach 2: Iterative with Stack
    /// 
    /// **Algorithm:**
    /// 1. Use explicit stack instead of recursion
    /// 2. Each stack entry contains: (index, target, path)
    /// 3. Process stack until empty
    /// 4. Add valid combinations to result
    /// 
    /// **Time Complexity:** O(N^(T/M)) - Same as recursive
    /// **Space Complexity:** O(T/M) - Stack depth instead of recursion
    /// 
    /// **Advantages:**
    /// - Avoids stack overflow for deep recursion
    /// - More control over memory usage
    /// - Can implement custom stack management
    /// 
    /// **When to use:** Very deep recursion expected
    pub fn combination_sum_iterative(&self, candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        let mut candidates = candidates;
        candidates.sort_unstable();
        
        let mut result = Vec::new();
        let mut stack = Vec::new();
        
        // (start_index, remaining_target, current_path)
        stack.push((0, target, Vec::new()));
        
        while let Some((start, remaining, path)) = stack.pop() {
            if remaining == 0 {
                result.push(path);
                continue;
            }
            
            for i in start..candidates.len() {
                let candidate = candidates[i];
                
                if candidate > remaining {
                    break; // Pruning
                }
                
                let mut new_path = path.clone();
                new_path.push(candidate);
                stack.push((i, remaining - candidate, new_path));
            }
        }
        
        result
    }

    /// # Approach 3: DP with Memoization
    /// 
    /// **Algorithm:**
    /// 1. Define dp(target) = all combinations that sum to target
    /// 2. For each target, try each candidate
    /// 3. Combine results from subproblems
    /// 4. Memoize to avoid redundant computation
    /// 
    /// **Time Complexity:** O(target * N * result_size)
    /// **Space Complexity:** O(target * result_size) - Memoization table
    /// 
    /// **Challenge with memoization:**
    /// - Hard to memoize list of combinations efficiently
    /// - Memory usage can be very high
    /// - May not provide significant speedup
    /// 
    /// **When effective:** When many overlapping subproblems exist
    pub fn combination_sum_dp(&self, candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        let mut dp: Vec<Vec<Vec<i32>>> = vec![Vec::new(); (target + 1) as usize];
        dp[0].push(Vec::new()); // Base case: empty combination for target 0
        
        for t in 1..=target {
            let mut new_combinations = Vec::new();
            
            for &candidate in &candidates {
                if candidate <= t {
                    let prev_target = (t - candidate) as usize;
                    for combo in &dp[prev_target] {
                        let mut new_combo = combo.clone();
                        new_combo.push(candidate);
                        new_combo.sort_unstable(); // Maintain order for uniqueness
                        
                        // Check for duplicates
                        if !new_combinations.contains(&new_combo) && !dp[t as usize].contains(&new_combo) {
                            new_combinations.push(new_combo);
                        }
                    }
                }
            }
            
            dp[t as usize].extend(new_combinations);
        }
        
        dp[target as usize].clone()
    }

    /// # Approach 4: Backtracking with Frequency Map
    /// 
    /// **Algorithm:**
    /// 1. Use frequency map to track how many times each candidate is used
    /// 2. Generate combinations by iterating through frequencies
    /// 3. Convert frequency maps back to lists
    /// 
    /// **Time Complexity:** O(N^(T/M)) - Similar to basic backtracking
    /// **Space Complexity:** O(N) - Frequency map size
    /// 
    /// **Different representation:**
    /// - Track counts instead of building lists
    /// - Can be more memory efficient for some cases
    /// - Easier to check constraints on usage counts
    /// 
    /// **When useful:** When you need to limit usage counts per candidate
    pub fn combination_sum_frequency(&self, candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        let mut result = Vec::new();
        let mut freq = vec![0; candidates.len()];
        
        self.backtrack_freq(&candidates, target, 0, &mut freq, &mut result);
        result
    }
    
    fn backtrack_freq(&self, candidates: &[i32], target: i32, idx: usize, 
                      freq: &mut Vec<usize>, result: &mut Vec<Vec<i32>>) {
        if target == 0 {
            let mut combination = Vec::new();
            for (i, &count) in freq.iter().enumerate() {
                for _ in 0..count {
                    combination.push(candidates[i]);
                }
            }
            result.push(combination);
            return;
        }
        
        if idx >= candidates.len() || target < 0 {
            return;
        }
        
        // Try using current candidate 0, 1, 2, ... times
        let max_use = target / candidates[idx];
        for use_count in 0..=max_use {
            freq[idx] = use_count as usize;
            self.backtrack_freq(candidates, target - candidates[idx] * use_count, 
                              idx + 1, freq, result);
        }
        freq[idx] = 0; // Reset for backtracking
    }

    /// # Approach 5: BFS Level-by-Level
    /// 
    /// **Algorithm:**
    /// 1. Use BFS to explore combinations level by level
    /// 2. Each level adds one more number to combinations
    /// 3. Queue stores: (current_sum, path, start_index)
    /// 4. Process until all valid combinations found
    /// 
    /// **Time Complexity:** O(N^(T/M)) - Explores same space as DFS
    /// **Space Complexity:** O(N^(T/M)) - Queue can be large
    /// 
    /// **Characteristics:**
    /// - Finds combinations in order of length
    /// - Good for finding shortest combinations first
    /// - Higher memory usage than DFS
    /// 
    /// **When to use:** When you want combinations ordered by length
    pub fn combination_sum_bfs(&self, candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        use std::collections::VecDeque;
        
        let mut candidates = candidates;
        candidates.sort_unstable();
        
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        
        // (current_sum, path, start_index)
        queue.push_back((0, Vec::new(), 0));
        
        while let Some((current_sum, path, start)) = queue.pop_front() {
            if current_sum == target {
                result.push(path);
                continue;
            }
            
            for i in start..candidates.len() {
                let candidate = candidates[i];
                let new_sum = current_sum + candidate;
                
                if new_sum > target {
                    break; // Pruning
                }
                
                let mut new_path = path.clone();
                new_path.push(candidate);
                queue.push_back((new_sum, new_path, i));
            }
        }
        
        result
    }

    /// # Approach 6: Recursive with Early Optimization
    /// 
    /// **Algorithm:**
    /// 1. Enhanced backtracking with multiple optimizations
    /// 2. Skip candidates that are too large
    /// 3. Use heuristics to reorder exploration
    /// 4. Early termination when remaining candidates can't help
    /// 
    /// **Time Complexity:** O(N^(T/M)) - Same worst case, better average
    /// **Space Complexity:** O(T/M) - Recursion depth
    /// 
    /// **Optimizations:**
    /// - Sort candidates by efficiency (target/candidate ratio)
    /// - Skip candidates larger than remaining target
    /// - Estimate lower bound for early termination
    /// 
    /// **When effective:** Large search spaces with sparse solutions
    pub fn combination_sum_optimized(&self, candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        let mut candidates = candidates;
        
        // Sort by efficiency: prefer candidates that divide target evenly
        candidates.sort_by_key(|&x| {
            let quotient = target / x;
            let remainder = target % x;
            // Prefer smaller remainder, then smaller quotient
            (remainder, quotient)
        });
        
        let mut result = Vec::new();
        let mut current = Vec::new();
        
        self.backtrack_optimized(&candidates, target, 0, &mut current, &mut result);
        result
    }
    
    fn backtrack_optimized(&self, candidates: &[i32], target: i32, start: usize,
                          current: &mut Vec<i32>, result: &mut Vec<Vec<i32>>) {
        if target == 0 {
            result.push(current.clone());
            return;
        }
        
        for i in start..candidates.len() {
            let candidate = candidates[i];
            
            if candidate > target {
                continue; // Skip instead of break (not necessarily sorted by value)
            }
            
            current.push(candidate);
            self.backtrack_optimized(candidates, target - candidate, i, current, result);
            current.pop();
        }
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

    fn setup() -> Solution {
        Solution::new()
    }

    fn sort_result(mut result: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        for combo in &mut result {
            combo.sort_unstable();
        }
        result.sort();
        result
    }

    #[test]
    fn test_basic_examples() {
        let solution = setup();
        
        // Example 1: [2,3,6,7], target=7 → [[2,2,3],[7]]
        let result = solution.combination_sum(vec![2, 3, 6, 7], 7);
        let expected = vec![vec![2, 2, 3], vec![7]];
        assert_eq!(sort_result(result), sort_result(expected));
        
        // Example 2: [2,3,5], target=8 → [[2,2,2,2],[2,3,3],[3,5]]
        let result = solution.combination_sum(vec![2, 3, 5], 8);
        let expected = vec![vec![2, 2, 2, 2], vec![2, 3, 3], vec![3, 5]];
        assert_eq!(sort_result(result), sort_result(expected));
        
        // Example 3: [2], target=1 → []
        let result = solution.combination_sum(vec![2], 1);
        assert_eq!(result, Vec::<Vec<i32>>::new());
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // Single candidate equals target
        let result = solution.combination_sum(vec![5], 5);
        assert_eq!(sort_result(result), vec![vec![5]]);
        
        // Single candidate, multiple needed
        let result = solution.combination_sum(vec![3], 9);
        assert_eq!(sort_result(result), vec![vec![3, 3, 3]]);
        
        // Multiple candidates, multiple solutions
        let result = solution.combination_sum(vec![2, 3, 5], 5);
        let expected = vec![vec![2, 3], vec![5]];
        assert_eq!(sort_result(result), sort_result(expected));
        
        // No solution possible
        let result = solution.combination_sum(vec![3, 5], 1);
        assert_eq!(result, Vec::<Vec<i32>>::new());
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            (vec![2, 3, 6, 7], 7),
            (vec![2, 3, 5], 8),
            (vec![2, 7, 6, 3, 5, 1], 9),
            (vec![1, 2], 3),
            (vec![1], 1),
            (vec![1], 2),
        ];
        
        for (candidates, target) in test_cases {
            let result1 = sort_result(solution.combination_sum(candidates.clone(), target));
            let result2 = sort_result(solution.combination_sum_iterative(candidates.clone(), target));
            let result3 = sort_result(solution.combination_sum_dp(candidates.clone(), target));
            let result4 = sort_result(solution.combination_sum_frequency(candidates.clone(), target));
            let result5 = sort_result(solution.combination_sum_bfs(candidates.clone(), target));
            let result6 = sort_result(solution.combination_sum_optimized(candidates.clone(), target));
            
            assert_eq!(result1, result2, "Mismatch for {:?}, target={}", candidates, target);
            assert_eq!(result2, result3, "Mismatch for {:?}, target={}", candidates, target);
            assert_eq!(result3, result4, "Mismatch for {:?}, target={}", candidates, target);
            assert_eq!(result4, result5, "Mismatch for {:?}, target={}", candidates, target);
            assert_eq!(result5, result6, "Mismatch for {:?}, target={}", candidates, target);
        }
    }

    #[test]
    fn test_unlimited_reuse() {
        let solution = setup();
        
        // Can reuse same number multiple times
        let result = solution.combination_sum(vec![2], 6);
        assert_eq!(sort_result(result), vec![vec![2, 2, 2]]);
        
        // Mix of reuse and different numbers
        let result = solution.combination_sum(vec![2, 3], 7);
        let expected = vec![vec![2, 2, 3]];
        assert_eq!(sort_result(result), sort_result(expected));
    }

    #[test]
    fn test_no_duplicates() {
        let solution = setup();
        
        // Should not have duplicate combinations
        let result = solution.combination_sum(vec![2, 3, 5], 8);
        let mut sorted_result = sort_result(result);
        sorted_result.dedup();
        
        let expected = vec![vec![2, 2, 2, 2], vec![2, 3, 3], vec![3, 5]];
        assert_eq!(sorted_result, sort_result(expected));
    }

    #[test]
    fn test_single_element() {
        let solution = setup();
        
        // Single element array
        let result = solution.combination_sum(vec![7], 14);
        assert_eq!(sort_result(result), vec![vec![7, 7]]);
        
        // Single element, impossible
        let result = solution.combination_sum(vec![5], 7);
        assert_eq!(result, Vec::<Vec<i32>>::new());
    }

    #[test]
    fn test_large_candidates() {
        let solution = setup();
        
        // Large candidate values
        let result = solution.combination_sum(vec![10, 20, 30], 40);
        let expected = vec![vec![10, 10, 10, 10], vec![10, 10, 20], vec![10, 30], vec![20, 20]];
        assert_eq!(sort_result(result), sort_result(expected));
        
        // Mix of large and small
        let result = solution.combination_sum(vec![1, 10], 11);
        let expected = vec![vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], vec![1, 10]];
        assert_eq!(sort_result(result), sort_result(expected));
    }

    #[test]
    fn test_minimum_target() {
        let solution = setup();
        
        // Target is smallest candidate
        let result = solution.combination_sum(vec![2, 3, 5], 2);
        assert_eq!(sort_result(result), vec![vec![2]]);
        
        // Target smaller than all candidates
        let result = solution.combination_sum(vec![3, 4, 5], 2);
        assert_eq!(result, Vec::<Vec<i32>>::new());
    }

    #[test]
    fn test_multiple_solutions() {
        let solution = setup();
        
        // Many different combinations possible
        let result = solution.combination_sum(vec![1, 2, 3], 4);
        let expected = vec![
            vec![1, 1, 1, 1],
            vec![1, 1, 2],
            vec![1, 3],
            vec![2, 2]
        ];
        assert_eq!(sort_result(result), sort_result(expected));
    }

    #[test]
    fn test_order_independence() {
        let solution = setup();
        
        // Different input order should give same results
        let result1 = sort_result(solution.combination_sum(vec![2, 3, 5], 8));
        let result2 = sort_result(solution.combination_sum(vec![5, 2, 3], 8));
        let result3 = sort_result(solution.combination_sum(vec![3, 5, 2], 8));
        
        assert_eq!(result1, result2);
        assert_eq!(result2, result3);
    }

    #[test]
    fn test_constraint_boundaries() {
        let solution = setup();
        
        // Minimum constraints
        let result = solution.combination_sum(vec![2], 2);
        assert_eq!(sort_result(result), vec![vec![2]]);
        
        // Test with value 40 (max candidate)
        let result = solution.combination_sum(vec![40], 40);
        assert_eq!(sort_result(result), vec![vec![40]]);
        
        // Test with target 40 (max target)
        let result = solution.combination_sum(vec![10, 20], 40);
        let expected = vec![vec![10, 10, 10, 10], vec![10, 10, 20], vec![20, 20]];
        assert_eq!(sort_result(result), sort_result(expected));
    }

    #[test]
    fn test_performance_characteristics() {
        let solution = setup();
        
        // Should handle reasonable sized inputs efficiently
        let result = solution.combination_sum(vec![2, 3, 5, 7], 10);
        assert!(!result.is_empty());
        assert!(result.len() > 1); // Multiple solutions exist
        
        // All solutions should sum to target
        for combination in result {
            let sum: i32 = combination.iter().sum();
            assert_eq!(sum, 10);
        }
    }
}