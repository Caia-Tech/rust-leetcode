//! Problem 78: Subsets
//!
//! Given an integer array nums of unique elements, return all possible subsets (the power set).
//! The solution set must not contain duplicate subsets. Return the solution in any order.
//!
//! Constraints:
//! - 1 <= nums.length <= 10
//! - -10 <= nums[i] <= 10
//! - All the numbers of nums are unique.

pub struct Solution;

impl Solution {
    /// Approach 1: Recursive Backtracking
    /// 
    /// Classic backtracking approach where we make a choice to include or exclude
    /// each element. This explores all 2^n possibilities.
    /// 
    /// Time Complexity: O(2^n * n) - Generate 2^n subsets, each takes O(n) to copy
    /// Space Complexity: O(n) - Recursion depth
    pub fn subsets_backtracking(nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut result = Vec::new();
        let mut current = Vec::new();
        Self::backtrack(&nums, 0, &mut current, &mut result);
        result
    }
    
    fn backtrack(nums: &[i32], start: usize, current: &mut Vec<i32>, result: &mut Vec<Vec<i32>>) {
        // Add current subset to result
        result.push(current.clone());
        
        // Try adding each remaining element
        for i in start..nums.len() {
            current.push(nums[i]);
            Self::backtrack(nums, i + 1, current, result);
            current.pop();
        }
    }
    
    /// Approach 2: Iterative with Bit Manipulation
    /// 
    /// Use binary representation where each bit indicates whether to include
    /// the element at that position. For n elements, iterate from 0 to 2^n - 1.
    /// 
    /// Time Complexity: O(2^n * n)
    /// Space Complexity: O(1) - No recursion
    pub fn subsets_bit_manipulation(nums: Vec<i32>) -> Vec<Vec<i32>> {
        let n = nums.len();
        let mut result = Vec::new();
        let total_subsets = 1 << n; // 2^n
        
        for mask in 0..total_subsets {
            let mut subset = Vec::new();
            for i in 0..n {
                if (mask & (1 << i)) != 0 {
                    subset.push(nums[i]);
                }
            }
            result.push(subset);
        }
        
        result
    }
    
    /// Approach 3: Iterative Building
    /// 
    /// Start with empty set, then for each number, add it to all existing subsets
    /// to create new subsets. This builds up the power set iteratively.
    /// 
    /// Time Complexity: O(2^n * n)
    /// Space Complexity: O(1) - No recursion
    pub fn subsets_iterative(nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut result = vec![vec![]];
        
        for num in nums {
            let mut new_subsets = Vec::new();
            for subset in &result {
                let mut new_subset = subset.clone();
                new_subset.push(num);
                new_subsets.push(new_subset);
            }
            result.extend(new_subsets);
        }
        
        result
    }
    
    /// Approach 4: Binary Choice Tree DFS
    /// 
    /// Treat the problem as a binary tree where at each level we decide
    /// whether to include the current element or not.
    /// 
    /// Time Complexity: O(2^n * n)
    /// Space Complexity: O(n) - Recursion depth
    pub fn subsets_binary_tree(nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut result = Vec::new();
        Self::dfs_binary(&nums, 0, vec![], &mut result);
        result
    }
    
    fn dfs_binary(nums: &[i32], index: usize, current: Vec<i32>, result: &mut Vec<Vec<i32>>) {
        if index == nums.len() {
            result.push(current);
            return;
        }
        
        // Don't include nums[index]
        Self::dfs_binary(nums, index + 1, current.clone(), result);
        
        // Include nums[index]
        let mut with_current = current;
        with_current.push(nums[index]);
        Self::dfs_binary(nums, index + 1, with_current, result);
    }
    
    /// Approach 5: Queue-Based BFS
    /// 
    /// Use BFS to explore all possibilities level by level.
    /// Each level represents decisions about including/excluding an element.
    /// 
    /// Time Complexity: O(2^n * n)
    /// Space Complexity: O(2^n) - Queue can hold all subsets
    pub fn subsets_bfs(nums: Vec<i32>) -> Vec<Vec<i32>> {
        use std::collections::VecDeque;
        
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back((vec![], 0));
        
        while let Some((subset, start)) = queue.pop_front() {
            result.push(subset.clone());
            
            for i in start..nums.len() {
                let mut new_subset = subset.clone();
                new_subset.push(nums[i]);
                queue.push_back((new_subset, i + 1));
            }
        }
        
        result
    }
    
    /// Approach 6: Functional Style with Fold
    /// 
    /// Use functional programming with fold to build subsets.
    /// More idiomatic Rust approach using iterators and combinators.
    /// 
    /// Time Complexity: O(2^n * n)
    /// Space Complexity: O(1) - No explicit recursion
    pub fn subsets_functional(nums: Vec<i32>) -> Vec<Vec<i32>> {
        nums.into_iter()
            .fold(vec![vec![]], |mut acc, num| {
                let new_subsets: Vec<Vec<i32>> = acc
                    .iter()
                    .map(|subset| {
                        let mut new_subset = subset.clone();
                        new_subset.push(num);
                        new_subset
                    })
                    .collect();
                acc.extend(new_subsets);
                acc
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    
    fn normalize_result(mut result: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        for subset in &mut result {
            subset.sort();
        }
        result.sort();
        result
    }
    
    fn verify_subsets(nums: &[i32], result: &[Vec<i32>]) -> bool {
        let n = nums.len();
        let expected_count = 1 << n; // 2^n
        
        if result.len() != expected_count {
            return false;
        }
        
        // Check for duplicates
        let set: HashSet<Vec<i32>> = result.iter().cloned().collect();
        if set.len() != result.len() {
            return false;
        }
        
        // Verify all subsets are valid
        for subset in result {
            for &num in subset {
                if !nums.contains(&num) {
                    return false;
                }
            }
        }
        
        true
    }
    
    #[test]
    fn test_basic_example() {
        let nums = vec![1, 2, 3];
        let expected = vec![
            vec![],
            vec![1],
            vec![2],
            vec![3],
            vec![1, 2],
            vec![1, 3],
            vec![2, 3],
            vec![1, 2, 3],
        ];
        
        let result1 = Solution::subsets_backtracking(nums.clone());
        assert!(verify_subsets(&nums, &result1));
        assert_eq!(normalize_result(result1), normalize_result(expected.clone()));
        
        let result2 = Solution::subsets_bit_manipulation(nums.clone());
        assert!(verify_subsets(&nums, &result2));
        
        let result3 = Solution::subsets_iterative(nums.clone());
        assert!(verify_subsets(&nums, &result3));
        
        let result4 = Solution::subsets_binary_tree(nums.clone());
        assert!(verify_subsets(&nums, &result4));
        
        let result5 = Solution::subsets_bfs(nums.clone());
        assert!(verify_subsets(&nums, &result5));
        
        let result6 = Solution::subsets_functional(nums.clone());
        assert!(verify_subsets(&nums, &result6));
    }
    
    #[test]
    fn test_single_element() {
        let nums = vec![0];
        let expected = vec![vec![], vec![0]];
        
        let result1 = Solution::subsets_backtracking(nums.clone());
        assert_eq!(normalize_result(result1), normalize_result(expected.clone()));
        
        let result2 = Solution::subsets_bit_manipulation(nums.clone());
        assert_eq!(normalize_result(result2), normalize_result(expected.clone()));
        
        let result3 = Solution::subsets_iterative(nums.clone());
        assert_eq!(normalize_result(result3), normalize_result(expected.clone()));
        
        let result4 = Solution::subsets_binary_tree(nums.clone());
        assert_eq!(normalize_result(result4), normalize_result(expected.clone()));
        
        let result5 = Solution::subsets_bfs(nums.clone());
        assert_eq!(normalize_result(result5), normalize_result(expected.clone()));
        
        let result6 = Solution::subsets_functional(nums.clone());
        assert_eq!(normalize_result(result6), normalize_result(expected.clone()));
    }
    
    #[test]
    fn test_two_elements() {
        let nums = vec![1, 2];
        
        let result1 = Solution::subsets_backtracking(nums.clone());
        assert!(verify_subsets(&nums, &result1));
        assert_eq!(result1.len(), 4);
        
        let result2 = Solution::subsets_bit_manipulation(nums.clone());
        assert!(verify_subsets(&nums, &result2));
        assert_eq!(result2.len(), 4);
        
        let result3 = Solution::subsets_iterative(nums.clone());
        assert!(verify_subsets(&nums, &result3));
        assert_eq!(result3.len(), 4);
        
        let result4 = Solution::subsets_binary_tree(nums.clone());
        assert!(verify_subsets(&nums, &result4));
        assert_eq!(result4.len(), 4);
        
        let result5 = Solution::subsets_bfs(nums.clone());
        assert!(verify_subsets(&nums, &result5));
        assert_eq!(result5.len(), 4);
        
        let result6 = Solution::subsets_functional(nums.clone());
        assert!(verify_subsets(&nums, &result6));
        assert_eq!(result6.len(), 4);
    }
    
    #[test]
    fn test_negative_numbers() {
        let nums = vec![-3, -1, 0, 2];
        
        let result1 = Solution::subsets_backtracking(nums.clone());
        assert!(verify_subsets(&nums, &result1));
        assert_eq!(result1.len(), 16);
        
        let result2 = Solution::subsets_bit_manipulation(nums.clone());
        assert!(verify_subsets(&nums, &result2));
        assert_eq!(result2.len(), 16);
        
        let result3 = Solution::subsets_iterative(nums.clone());
        assert!(verify_subsets(&nums, &result3));
        assert_eq!(result3.len(), 16);
        
        let result4 = Solution::subsets_binary_tree(nums.clone());
        assert!(verify_subsets(&nums, &result4));
        assert_eq!(result4.len(), 16);
        
        let result5 = Solution::subsets_bfs(nums.clone());
        assert!(verify_subsets(&nums, &result5));
        assert_eq!(result5.len(), 16);
        
        let result6 = Solution::subsets_functional(nums.clone());
        assert!(verify_subsets(&nums, &result6));
        assert_eq!(result6.len(), 16);
    }
    
    #[test]
    fn test_maximum_size() {
        let nums = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        
        let result1 = Solution::subsets_backtracking(nums.clone());
        assert!(verify_subsets(&nums, &result1));
        assert_eq!(result1.len(), 1024); // 2^10
        
        let result2 = Solution::subsets_bit_manipulation(nums.clone());
        assert!(verify_subsets(&nums, &result2));
        assert_eq!(result2.len(), 1024);
        
        let result3 = Solution::subsets_iterative(nums.clone());
        assert!(verify_subsets(&nums, &result3));
        assert_eq!(result3.len(), 1024);
        
        let result4 = Solution::subsets_binary_tree(nums.clone());
        assert!(verify_subsets(&nums, &result4));
        assert_eq!(result4.len(), 1024);
        
        let result5 = Solution::subsets_bfs(nums.clone());
        assert!(verify_subsets(&nums, &result5));
        assert_eq!(result5.len(), 1024);
        
        let result6 = Solution::subsets_functional(nums.clone());
        assert!(verify_subsets(&nums, &result6));
        assert_eq!(result6.len(), 1024);
    }
    
    #[test]
    fn test_contains_empty_set() {
        let test_cases = vec![
            vec![1],
            vec![1, 2],
            vec![1, 2, 3],
            vec![-1, 0, 1],
        ];
        
        for nums in test_cases {
            let result1 = Solution::subsets_backtracking(nums.clone());
            assert!(result1.iter().any(|s| s.is_empty()));
            
            let result2 = Solution::subsets_bit_manipulation(nums.clone());
            assert!(result2.iter().any(|s| s.is_empty()));
            
            let result3 = Solution::subsets_iterative(nums.clone());
            assert!(result3.iter().any(|s| s.is_empty()));
            
            let result4 = Solution::subsets_binary_tree(nums.clone());
            assert!(result4.iter().any(|s| s.is_empty()));
            
            let result5 = Solution::subsets_bfs(nums.clone());
            assert!(result5.iter().any(|s| s.is_empty()));
            
            let result6 = Solution::subsets_functional(nums.clone());
            assert!(result6.iter().any(|s| s.is_empty()));
        }
    }
    
    #[test]
    fn test_contains_full_set() {
        let test_cases = vec![
            vec![1],
            vec![1, 2],
            vec![1, 2, 3],
            vec![-1, 0, 1],
        ];
        
        for nums in test_cases {
            let mut sorted_nums = nums.clone();
            sorted_nums.sort();
            
            let result1 = Solution::subsets_backtracking(nums.clone());
            assert!(result1.iter().any(|s| {
                let mut sorted = s.clone();
                sorted.sort();
                sorted == sorted_nums
            }));
            
            let result2 = Solution::subsets_bit_manipulation(nums.clone());
            assert!(result2.iter().any(|s| {
                let mut sorted = s.clone();
                sorted.sort();
                sorted == sorted_nums
            }));
            
            let result3 = Solution::subsets_iterative(nums.clone());
            assert!(result3.iter().any(|s| {
                let mut sorted = s.clone();
                sorted.sort();
                sorted == sorted_nums
            }));
            
            let result4 = Solution::subsets_binary_tree(nums.clone());
            assert!(result4.iter().any(|s| {
                let mut sorted = s.clone();
                sorted.sort();
                sorted == sorted_nums
            }));
            
            let result5 = Solution::subsets_bfs(nums.clone());
            assert!(result5.iter().any(|s| {
                let mut sorted = s.clone();
                sorted.sort();
                sorted == sorted_nums
            }));
            
            let result6 = Solution::subsets_functional(nums.clone());
            assert!(result6.iter().any(|s| {
                let mut sorted = s.clone();
                sorted.sort();
                sorted == sorted_nums
            }));
        }
    }
    
    #[test]
    fn test_subset_sizes() {
        let nums = vec![1, 2, 3, 4];
        let n = nums.len();
        
        let result = Solution::subsets_backtracking(nums.clone());
        
        // Count subsets of each size
        let mut size_counts = vec![0; n + 1];
        for subset in &result {
            size_counts[subset.len()] += 1;
        }
        
        // Verify binomial coefficients
        // C(n, k) = n! / (k! * (n-k)!)
        for k in 0..=n {
            let expected = binomial_coefficient(n, k);
            assert_eq!(size_counts[k], expected);
        }
    }
    
    fn binomial_coefficient(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }
        
        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            vec![1, 2],
            vec![0],
            vec![-2, -1, 0, 1, 2],
            vec![5, 4, 3, 2, 1],
        ];
        
        for nums in test_cases {
            let result1 = normalize_result(Solution::subsets_backtracking(nums.clone()));
            let result2 = normalize_result(Solution::subsets_bit_manipulation(nums.clone()));
            let result3 = normalize_result(Solution::subsets_iterative(nums.clone()));
            let result4 = normalize_result(Solution::subsets_binary_tree(nums.clone()));
            let result5 = normalize_result(Solution::subsets_bfs(nums.clone()));
            let result6 = normalize_result(Solution::subsets_functional(nums.clone()));
            
            // All approaches should produce the same result (when normalized)
            assert_eq!(result1.len(), result2.len());
            assert_eq!(result2.len(), result3.len());
            assert_eq!(result3.len(), result4.len());
            assert_eq!(result4.len(), result5.len());
            assert_eq!(result5.len(), result6.len());
            
            // Convert to sets for comparison
            let set1: HashSet<Vec<i32>> = result1.into_iter().collect();
            let set2: HashSet<Vec<i32>> = result2.into_iter().collect();
            let set3: HashSet<Vec<i32>> = result3.into_iter().collect();
            let set4: HashSet<Vec<i32>> = result4.into_iter().collect();
            let set5: HashSet<Vec<i32>> = result5.into_iter().collect();
            let set6: HashSet<Vec<i32>> = result6.into_iter().collect();
            
            assert_eq!(set1, set2);
            assert_eq!(set2, set3);
            assert_eq!(set3, set4);
            assert_eq!(set4, set5);
            assert_eq!(set5, set6);
        }
    }
    
    #[test]
    fn test_sequential_numbers() {
        let nums = vec![1, 2, 3, 4, 5];
        
        let result = Solution::subsets_backtracking(nums.clone());
        assert!(verify_subsets(&nums, &result));
        assert_eq!(result.len(), 32); // 2^5
        
        // Check that all single element subsets exist
        for &num in &nums {
            assert!(result.iter().any(|s| s == &vec![num]));
        }
        
        // Check that all pairs exist
        for i in 0..nums.len() {
            for j in i + 1..nums.len() {
                let pair = vec![nums[i], nums[j]];
                assert!(result.iter().any(|s| {
                    let mut sorted = s.clone();
                    sorted.sort();
                    sorted == pair
                }));
            }
        }
    }
    
    #[test]
    fn test_boundary_values() {
        let nums = vec![-10, -5, 0, 5, 10];
        
        let result1 = Solution::subsets_backtracking(nums.clone());
        assert!(verify_subsets(&nums, &result1));
        assert_eq!(result1.len(), 32);
        
        let result2 = Solution::subsets_bit_manipulation(nums.clone());
        assert!(verify_subsets(&nums, &result2));
        
        let result3 = Solution::subsets_iterative(nums.clone());
        assert!(verify_subsets(&nums, &result3));
        
        let result4 = Solution::subsets_binary_tree(nums.clone());
        assert!(verify_subsets(&nums, &result4));
        
        let result5 = Solution::subsets_bfs(nums.clone());
        assert!(verify_subsets(&nums, &result5));
        
        let result6 = Solution::subsets_functional(nums.clone());
        assert!(verify_subsets(&nums, &result6));
    }
}