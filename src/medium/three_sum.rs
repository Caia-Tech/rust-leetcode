//! # Problem 15: 3Sum
//!
//! Given an integer array `nums`, return all the triplets `[nums[i], nums[j], nums[k]]`
//! such that `i != j`, `i != k`, and `j != k`, and `nums[i] + nums[j] + nums[k] == 0`.
//!
//! Notice that the solution set must not contain duplicate triplets.
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::medium::three_sum::Solution;
//! 
//! let solution = Solution::new();
//! 
//! // Example 1:
//! let nums = vec![-1, 0, 1, 2, -1, -4];
//! let mut result = solution.three_sum(nums);
//! result.sort();
//! let mut expected = vec![vec![-1, -1, 2], vec![-1, 0, 1]];
//! expected.sort();
//! assert_eq!(result, expected);
//! 
//! // Example 2:
//! assert_eq!(solution.three_sum(vec![0, 1, 1]), Vec::<Vec<i32>>::new());
//! 
//! // Example 3:
//! assert_eq!(solution.three_sum(vec![0, 0, 0]), vec![vec![0, 0, 0]]);
//! ```
//!
//! ## Constraints
//!
//! - 3 <= nums.length <= 3000
//! - -10^5 <= nums[i] <= 10^5

use std::collections::HashSet;

/// Solution struct following LeetCode format
pub struct Solution;

impl Solution {
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Two Pointers (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Sort the array to enable two-pointer technique
    /// 2. For each element as the first number, use two pointers for remaining two
    /// 3. Skip duplicates to avoid duplicate triplets
    /// 4. Move pointers based on sum comparison with target (0)
    /// 
    /// **Time Complexity:** O(n²) - O(n log n) sort + O(n) × O(n) two-pointer loops
    /// **Space Complexity:** O(1) - Not counting output array, only constant extra space
    /// 
    /// **Key Insight:** After sorting, for each first element, the remaining problem
    /// becomes finding two elements that sum to `-first_element`. This is the classic
    /// two-pointer technique on sorted arrays.
    /// 
    /// **Why this is optimal:**
    /// - Avoids nested loops and redundant hash operations
    /// - Duplicate handling is built into the algorithm structure
    /// - Single pass for each first element
    /// - No additional space for deduplication
    /// 
    /// **Duplicate handling:**
    /// - Skip duplicate first elements
    /// - Skip duplicate second elements after finding valid triplet
    /// - Skip duplicate third elements after finding valid triplet
    pub fn three_sum(&self, nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut nums = nums;
        nums.sort();
        let mut result = Vec::new();
        let n = nums.len();
        
        for i in 0..n - 2 {
            // Skip duplicates for first element
            if i > 0 && nums[i] == nums[i - 1] {
                continue;
            }
            
            let mut left = i + 1;
            let mut right = n - 1;
            let target = -nums[i];
            
            while left < right {
                let sum = nums[left] + nums[right];
                
                if sum == target {
                    result.push(vec![nums[i], nums[left], nums[right]]);
                    
                    // Skip duplicates for second element
                    while left < right && nums[left] == nums[left + 1] {
                        left += 1;
                    }
                    // Skip duplicates for third element
                    while left < right && nums[right] == nums[right - 1] {
                        right -= 1;
                    }
                    
                    left += 1;
                    right -= 1;
                } else if sum < target {
                    left += 1;
                } else {
                    right -= 1;
                }
            }
        }
        
        result
    }

    /// # Approach 2: Hash Set Based (Less Optimal)
    /// 
    /// **Algorithm:**
    /// 1. For each pair (i, j), use hash set to find third element
    /// 2. Use HashSet to store seen elements for current iteration
    /// 3. Use HashSet to deduplicate final results
    /// 
    /// **Time Complexity:** O(n²) - Two nested loops with O(1) hash operations
    /// **Space Complexity:** O(n) - Hash set for seen elements + deduplication
    /// 
    /// **Key Insight:** For each pair of elements, the third element is determined.
    /// We can use a hash set to check if this third element exists.
    /// 
    /// **Why less optimal than two-pointers:**
    /// - Additional space for hash operations
    /// - More complex duplicate handling
    /// - Hash operations have constants overhead
    /// - Requires sorting for deduplication
    /// 
    /// **When to prefer:** When the input cannot be modified (needs to stay unsorted)
    pub fn three_sum_hashset(&self, nums: Vec<i32>) -> Vec<Vec<i32>> {
        let n = nums.len();
        let mut result_set: HashSet<Vec<i32>> = HashSet::new();
        
        for i in 0..n - 2 {
            let mut seen = HashSet::new();
            
            for j in i + 1..n {
                let complement = -(nums[i] + nums[j]);
                
                if seen.contains(&complement) {
                    let mut triplet = vec![nums[i], nums[j], complement];
                    triplet.sort(); // Sort to ensure consistent ordering for deduplication
                    result_set.insert(triplet);
                }
                
                seen.insert(nums[j]);
            }
        }
        
        result_set.into_iter().collect()
    }

    /// # Approach 3: Brute Force with Deduplication (Inefficient)
    /// 
    /// **Algorithm:**
    /// 1. Check all possible triplets with three nested loops
    /// 2. Use HashSet to avoid duplicate triplets
    /// 3. Sort each triplet for consistent deduplication
    /// 
    /// **Time Complexity:** O(n³) - Three nested loops
    /// **Space Complexity:** O(n) - HashSet for result deduplication
    /// 
    /// **Why this is inefficient:**
    /// - Checks many impossible combinations
    /// - No early termination optimizations
    /// - Cubic time complexity scales poorly
    /// - Redundant duplicate checking
    /// 
    /// **Educational value:**
    /// - Shows the naive approach
    /// - Demonstrates why algorithmic optimization matters
    /// - Baseline for performance comparison
    /// - Easy to understand and verify correctness
    pub fn three_sum_brute_force(&self, nums: Vec<i32>) -> Vec<Vec<i32>> {
        let n = nums.len();
        let mut result_set: HashSet<Vec<i32>> = HashSet::new();
        
        for i in 0..n {
            for j in i + 1..n {
                for k in j + 1..n {
                    if nums[i] + nums[j] + nums[k] == 0 {
                        let mut triplet = vec![nums[i], nums[j], nums[k]];
                        triplet.sort(); // Sort for consistent deduplication
                        result_set.insert(triplet);
                    }
                }
            }
        }
        
        result_set.into_iter().collect()
    }

    /// # Approach 4: Recursive Approach (Educational)
    /// 
    /// **Algorithm:**
    /// 1. Use recursion to generate all combinations of 3 elements
    /// 2. Check if each combination sums to zero
    /// 3. Use HashSet for result deduplication
    /// 
    /// **Time Complexity:** O(n³) - All combinations of 3 elements from n
    /// **Space Complexity:** O(n) - Recursion stack + result deduplication
    /// 
    /// **Educational insight:** Shows how combinatorial problems can be
    /// approached recursively, though not optimal for this specific problem.
    /// 
    /// **When to consider:** When the problem extends to k-sum for variable k,
    /// recursive approaches become more natural to generalize.
    pub fn three_sum_recursive(&self, nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut result_set: HashSet<Vec<i32>> = HashSet::new();
        let mut current = Vec::new();
        
        self.find_triplets(&nums, 0, &mut current, &mut result_set);
        result_set.into_iter().collect()
    }
    
    fn find_triplets(
        &self,
        nums: &[i32], 
        start: usize, 
        current: &mut Vec<i32>, 
        result_set: &mut HashSet<Vec<i32>>
    ) {
        if current.len() == 3 {
            if current.iter().sum::<i32>() == 0 {
                let mut triplet = current.clone();
                triplet.sort();
                result_set.insert(triplet);
            }
            return;
        }
        
        for i in start..nums.len() {
            current.push(nums[i]);
            self.find_triplets(nums, i + 1, current, result_set);
            current.pop();
        }
    }

    /// # Approach 5: Early Termination Optimization
    /// 
    /// **Algorithm:**
    /// 1. Same as two-pointer approach but with additional optimizations
    /// 2. Skip iterations when minimum possible sum > 0
    /// 3. Skip iterations when maximum possible sum < 0
    /// 4. Early break when current element is positive (after sorting)
    /// 
    /// **Time Complexity:** O(n²) - Same as basic two-pointer but with better constants
    /// **Space Complexity:** O(1) - Same as basic two-pointer
    /// 
    /// **Optimizations:**
    /// - If `nums[i] > 0`, all remaining triplets will be positive (early break)
    /// - If `nums[i] + nums[i+1] + nums[i+2] > 0`, skip current i (too small)
    /// - If `nums[i] + nums[n-2] + nums[n-1] < 0`, skip current i (too large)
    /// 
    /// **When to prefer:** When you need maximum performance on large inputs
    /// or when input has specific patterns that benefit from early termination.
    pub fn three_sum_optimized(&self, nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut nums = nums;
        nums.sort();
        let mut result = Vec::new();
        let n = nums.len();
        
        for i in 0..n - 2 {
            // Early termination: if first element is positive, all sums will be positive
            if nums[i] > 0 {
                break;
            }
            
            // Skip duplicates for first element
            if i > 0 && nums[i] == nums[i - 1] {
                continue;
            }
            
            // Skip if minimum possible sum is too large
            if nums[i] + nums[i + 1] + nums[i + 2] > 0 {
                break;
            }
            
            // Skip if maximum possible sum is too small
            if nums[i] + nums[n - 2] + nums[n - 1] < 0 {
                continue;
            }
            
            let mut left = i + 1;
            let mut right = n - 1;
            let target = -nums[i];
            
            while left < right {
                let sum = nums[left] + nums[right];
                
                if sum == target {
                    result.push(vec![nums[i], nums[left], nums[right]]);
                    
                    // Skip duplicates for second element
                    while left < right && nums[left] == nums[left + 1] {
                        left += 1;
                    }
                    // Skip duplicates for third element
                    while left < right && nums[right] == nums[right - 1] {
                        right -= 1;
                    }
                    
                    left += 1;
                    right -= 1;
                } else if sum < target {
                    left += 1;
                } else {
                    right -= 1;
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
    use rstest::rstest;

    fn setup() -> Solution {
        Solution::new()
    }

    fn sort_result(mut result: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        for triplet in &mut result {
            triplet.sort();
        }
        result.sort();
        result
    }

    #[rstest]
    #[case(vec![-1, 0, 1, 2, -1, -4], vec![vec![-1, -1, 2], vec![-1, 0, 1]])]
    #[case(vec![0, 1, 1], vec![])]
    #[case(vec![0, 0, 0], vec![vec![0, 0, 0]])]
    #[case(vec![-2, 0, 1, 1, 2], vec![vec![-2, 0, 2], vec![-2, 1, 1]])]
    #[case(vec![3, 0, -2, -1, 1, 2], vec![vec![-2, -1, 3], vec![-2, 0, 2], vec![-1, 0, 1]])]
    fn test_basic_cases(#[case] nums: Vec<i32>, #[case] expected: Vec<Vec<i32>>) {
        let solution = setup();
        let result = sort_result(solution.three_sum(nums));
        let expected = sort_result(expected);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_minimum_length() {
        let solution = setup();
        
        // Exactly 3 elements
        assert_eq!(sort_result(solution.three_sum(vec![0, 0, 0])), vec![vec![0, 0, 0]]);
        assert_eq!(sort_result(solution.three_sum(vec![1, 2, 3])), Vec::<Vec<i32>>::new());
        assert_eq!(sort_result(solution.three_sum(vec![-3, 1, 2])), vec![vec![-3, 1, 2]]);
    }

    #[test]
    fn test_no_solution() {
        let solution = setup();
        
        let no_solution_cases = vec![
            vec![1, 2, 3],           // All positive
            vec![-1, -2, -3],        // All negative  
            vec![1, 1, 1],           // All same positive
            vec![-1, -1, -1],        // All same negative
            vec![1, 2, 4],           // Positive, no sum to 0
        ];
        
        for case in no_solution_cases {
            let result = solution.three_sum(case.clone());
            assert!(result.is_empty(), "Expected empty result for {:?}", case);
        }
    }

    #[test]
    fn test_duplicate_handling() {
        let solution = setup();
        
        // Multiple duplicates
        let result = sort_result(solution.three_sum(vec![-1, -1, -1, 0, 1, 1, 2]));
        let expected = sort_result(vec![vec![-1, -1, 2], vec![-1, 0, 1]]);
        assert_eq!(result, expected);
        
        // All zeros with different counts
        assert_eq!(sort_result(solution.three_sum(vec![0, 0, 0, 0])), vec![vec![0, 0, 0]]);
        assert_eq!(sort_result(solution.three_sum(vec![0, 0, 0, 0, 0])), vec![vec![0, 0, 0]]);
    }

    #[test]
    fn test_mixed_signs() {
        let solution = setup();
        
        // Mixed positive and negative with zeros
        let result = sort_result(solution.three_sum(vec![-2, 0, 0, 2, 2]));
        let expected = sort_result(vec![vec![-2, 0, 2]]);
        assert_eq!(result, expected);
        
        // Multiple valid combinations
        let result = sort_result(solution.three_sum(vec![-4, -1, -1, 0, 1, 2]));
        let expected = sort_result(vec![vec![-1, -1, 2], vec![-1, 0, 1]]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_boundary_values() {
        let solution = setup();
        
        // Test constraint boundaries: -10^5 <= nums[i] <= 10^5
        let large_values = vec![-100000, 0, 100000];
        let result = sort_result(solution.three_sum(large_values.clone()));
        let expected = sort_result(vec![vec![-100000, 0, 100000]]);
        assert_eq!(result, expected);
        
        // Mix of boundary and normal values
        let mixed = vec![-100000, -1, 1, 100000, 0];
        let result = sort_result(solution.three_sum(mixed));
        let expected = sort_result(vec![vec![-100000, 0, 100000], vec![-1, 0, 1]]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_large_input() {
        let solution = setup();
        
        // Test near maximum constraint: up to 3000 elements
        let mut large_input = vec![-1000; 500];
        large_input.extend(vec![0; 1000]);
        large_input.extend(vec![1000; 500]);
        large_input.extend(vec![500; 500]);
        large_input.extend(vec![-500; 500]);
        
        let result = solution.three_sum(large_input);
        
        // Should find multiple valid combinations
        assert!(!result.is_empty());
        
        // Verify all results are valid
        for triplet in result {
            assert_eq!(triplet.len(), 3);
            assert_eq!(triplet.iter().sum::<i32>(), 0);
        }
    }

    #[test]
    fn test_performance_patterns() {
        let solution = setup();
        
        // Pattern that could cause issues with naive algorithms
        
        // Many zeros (should only return one triplet)
        let many_zeros = vec![0; 1000];
        let result = solution.three_sum(many_zeros);
        assert_eq!(result, vec![vec![0, 0, 0]]);
        
        // Arithmetic sequence
        let arithmetic: Vec<i32> = (-50..=50).collect(); // [-50, -49, ..., 49, 50]
        let result = solution.three_sum(arithmetic);
        assert!(!result.is_empty());
        
        // Verify all results are valid
        for triplet in result {
            assert_eq!(triplet.iter().sum::<i32>(), 0);
        }
    }

    #[test]
    fn test_all_approaches_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            vec![-1, 0, 1, 2, -1, -4],
            vec![0, 1, 1],
            vec![0, 0, 0],
            vec![-2, 0, 1, 1, 2],
            vec![3, 0, -2, -1, 1, 2],
            vec![-4, -1, -1, 0, 1, 2],
            vec![1, 2, 3], // No solution
            vec![-1, -1, -1, 0, 1, 1, 2],
            vec![0, 0, 0, 0],
        ];
        
        for case in test_cases {
            let result1 = sort_result(solution.three_sum(case.clone()));
            let result2 = sort_result(solution.three_sum_hashset(case.clone()));
            let result3 = sort_result(solution.three_sum_brute_force(case.clone()));
            let result4 = sort_result(solution.three_sum_recursive(case.clone()));
            let result5 = sort_result(solution.three_sum_optimized(case.clone()));
            
            assert_eq!(result1, result2, "HashSet approach differs for {:?}", case);
            assert_eq!(result1, result3, "Brute force differs for {:?}", case);
            assert_eq!(result1, result4, "Recursive approach differs for {:?}", case);
            assert_eq!(result1, result5, "Optimized approach differs for {:?}", case);
        }
    }

    #[test]
    fn test_edge_cases_comprehensive() {
        let solution = setup();
        
        // Single valid triplet among many elements
        let mut sparse = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        sparse.extend(vec![-1, -2]);
        let result = sort_result(solution.three_sum(sparse));
        // Note: Need to check what triplets actually exist in the sparse array
        assert!(!result.is_empty());
        
        // Verify all results sum to 0
        for triplet in result {
            assert_eq!(triplet.iter().sum::<i32>(), 0);
        }
    }

    #[test]
    fn test_sorted_vs_unsorted_input() {
        let solution = setup();
        
        // Test that result is same regardless of input order
        let base = vec![-1, 0, 1, 2, -1, -4];
        let mut shuffled = base.clone();
        shuffled.reverse(); // Different order
        
        let result1 = sort_result(solution.three_sum(base));
        let result2 = sort_result(solution.three_sum(shuffled));
        
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_two_pointer_specific_cases() {
        let solution = setup();
        
        // Cases that test two-pointer logic thoroughly
        
        // Case where left and right pointers need to move multiple times
        let result = sort_result(solution.three_sum(vec![-3, -2, -1, 0, 1, 2, 3]));
        
        // Verify all actual results are valid
        for triplet in &result {
            assert_eq!(triplet.iter().sum::<i32>(), 0);
            assert_eq!(triplet.len(), 3);
        }
        
        assert!(!result.is_empty());
    }

    #[test]
    fn test_optimization_scenarios() {
        let solution = setup();
        
        // Test cases where optimizations should provide benefits
        
        // All positive (should terminate early in optimized version)
        let all_positive = vec![1, 2, 3, 4, 5];
        assert!(solution.three_sum_optimized(all_positive).is_empty());
        
        // Mostly negative with few positives
        let mostly_negative = vec![-10, -9, -8, -7, -6, 1, 2];
        let result = solution.three_sum_optimized(mostly_negative);
        
        // Verify results
        for triplet in result {
            assert_eq!(triplet.iter().sum::<i32>(), 0);
        }
    }

    #[test] 
    fn test_mathematical_edge_cases() {
        let solution = setup();
        
        // Perfect arithmetic progressions
        let arithmetic = vec![-6, -3, 0, 3, 6];
        let result = sort_result(solution.three_sum(arithmetic));
        let expected = sort_result(vec![vec![-6, 0, 6], vec![-3, 0, 3]]);
        assert_eq!(result, expected);
        
        // Geometric-like patterns  
        let geometric = vec![-8, -4, -2, 2, 4, 8];
        let result = sort_result(solution.three_sum(geometric));
        
        // Verify results are mathematically correct
        for triplet in result {
            assert_eq!(triplet.iter().sum::<i32>(), 0);
            assert_eq!(triplet.len(), 3);
        }
    }
}