//! Problem 90: Subsets II
//!
//! Given an integer array nums that may contain duplicates, return all possible subsets (the power set).
//! The solution set must not contain duplicate subsets. Return the solution in any order.
//!
//! Constraints:
//! - 1 <= nums.length <= 10
//! - -10 <= nums[i] <= 10

pub struct Solution;

impl Solution {
    /// Approach 1: Backtracking with Sorting
    /// 
    /// Sort the array first, then use backtracking while skipping duplicates
    /// at the same recursion level.
    /// 
    /// Time Complexity: O(2^n * n) - Generate unique subsets, each takes O(n) to copy
    /// Space Complexity: O(n) - Recursion depth
    pub fn subsets_with_dup_backtracking(mut nums: Vec<i32>) -> Vec<Vec<i32>> {
        nums.sort_unstable();
        let mut result = Vec::new();
        let mut current = Vec::new();
        Self::backtrack(&nums, 0, &mut current, &mut result);
        result
    }
    
    fn backtrack(nums: &[i32], start: usize, current: &mut Vec<i32>, result: &mut Vec<Vec<i32>>) {
        result.push(current.clone());
        
        for i in start..nums.len() {
            // Skip duplicates at the same recursion level
            if i > start && nums[i] == nums[i - 1] {
                continue;
            }
            current.push(nums[i]);
            Self::backtrack(nums, i + 1, current, result);
            current.pop();
        }
    }
    
    /// Approach 2: Iterative with Duplicate Counting
    /// 
    /// Count duplicates and iteratively build subsets by adding different
    /// counts of each unique element to existing subsets.
    /// 
    /// Time Complexity: O(2^n * n)
    /// Space Complexity: O(1) - No recursion
    pub fn subsets_with_dup_iterative(mut nums: Vec<i32>) -> Vec<Vec<i32>> {
        nums.sort_unstable();
        let mut result = vec![vec![]];
        
        let mut i = 0;
        while i < nums.len() {
            let count = {
                let mut j = i;
                while j < nums.len() && nums[j] == nums[i] {
                    j += 1;
                }
                j - i
            };
            
            let prev_size = result.len();
            for idx in 0..prev_size {
                let mut subset = result[idx].clone();
                for _ in 0..count {
                    subset.push(nums[i]);
                    result.push(subset.clone());
                }
            }
            
            i += count;
        }
        
        result
    }
    
    /// Approach 3: Bit Manipulation with Set
    /// 
    /// Use bit manipulation to generate all possible subsets, then use a HashSet
    /// to remove duplicates.
    /// 
    /// Time Complexity: O(2^n * n log n) - Sorting each subset for comparison
    /// Space Complexity: O(2^n) - Set storage
    pub fn subsets_with_dup_bit_set(mut nums: Vec<i32>) -> Vec<Vec<i32>> {
        use std::collections::HashSet;
        
        nums.sort_unstable();
        let n = nums.len();
        let mut result_set = HashSet::new();
        let total_subsets = 1 << n;
        
        for mask in 0..total_subsets {
            let mut subset = Vec::new();
            for i in 0..n {
                if (mask & (1 << i)) != 0 {
                    subset.push(nums[i]);
                }
            }
            result_set.insert(subset);
        }
        
        result_set.into_iter().collect()
    }
    
    /// Approach 4: Cascading with Smart Duplicate Handling
    /// 
    /// Start with empty subset, then for each number, add it to subsets.
    /// For duplicates, only add to subsets created in the last iteration.
    /// 
    /// Time Complexity: O(2^n * n)
    /// Space Complexity: O(1) - No recursion
    pub fn subsets_with_dup_cascading(mut nums: Vec<i32>) -> Vec<Vec<i32>> {
        nums.sort_unstable();
        let mut result = vec![vec![]];
        let mut start_idx = 0;
        
        for i in 0..nums.len() {
            let start = if i > 0 && nums[i] == nums[i - 1] {
                start_idx
            } else {
                0
            };
            
            let end = result.len();
            start_idx = end;
            
            for j in start..end {
                let mut subset = result[j].clone();
                subset.push(nums[i]);
                result.push(subset);
            }
        }
        
        result
    }
    
    /// Approach 5: DFS with Pruning
    /// 
    /// Use DFS to explore all paths in the decision tree, pruning duplicate
    /// branches by checking if we've seen the same element at this level.
    /// 
    /// Time Complexity: O(2^n * n)
    /// Space Complexity: O(n) - Recursion depth
    pub fn subsets_with_dup_dfs(mut nums: Vec<i32>) -> Vec<Vec<i32>> {
        nums.sort_unstable();
        let mut result = Vec::new();
        Self::dfs(&nums, 0, vec![], &mut result);
        result
    }
    
    fn dfs(nums: &[i32], start: usize, current: Vec<i32>, result: &mut Vec<Vec<i32>>) {
        result.push(current.clone());
        
        let mut prev = None;
        for i in start..nums.len() {
            if prev == Some(nums[i]) {
                continue;
            }
            
            let mut next = current.clone();
            next.push(nums[i]);
            Self::dfs(nums, i + 1, next, result);
            prev = Some(nums[i]);
        }
    }
    
    /// Approach 6: Frequency Map Based
    /// 
    /// Count frequency of each element and generate subsets by choosing
    /// different counts (0 to frequency) for each unique element.
    /// 
    /// Time Complexity: O(2^n * n)
    /// Space Complexity: O(n) - Frequency map
    pub fn subsets_with_dup_frequency(nums: Vec<i32>) -> Vec<Vec<i32>> {
        use std::collections::HashMap;
        
        let mut freq_map = HashMap::new();
        for num in nums {
            *freq_map.entry(num).or_insert(0) += 1;
        }
        
        let unique_nums: Vec<i32> = freq_map.keys().copied().collect();
        let mut result = Vec::new();
        let mut current = Vec::new();
        
        Self::generate_by_frequency(&unique_nums, &freq_map, 0, &mut current, &mut result);
        result
    }
    
    fn generate_by_frequency(
        unique_nums: &[i32],
        freq_map: &std::collections::HashMap<i32, usize>,
        index: usize,
        current: &mut Vec<i32>,
        result: &mut Vec<Vec<i32>>
    ) {
        if index == unique_nums.len() {
            result.push(current.clone());
            return;
        }
        
        let num = unique_nums[index];
        let freq = freq_map[&num];
        
        // Choose 0 to freq occurrences of current number
        for count in 0..=freq {
            for _ in 0..count {
                current.push(num);
            }
            Self::generate_by_frequency(unique_nums, freq_map, index + 1, current, result);
            for _ in 0..count {
                current.pop();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    
    fn normalize_and_verify(mut result: Vec<Vec<i32>>, expected: Vec<Vec<i32>>) -> bool {
        // Sort each subset
        for subset in &mut result {
            subset.sort_unstable();
        }
        
        // Sort the result
        result.sort_unstable();
        
        let mut expected = expected;
        for subset in &mut expected {
            subset.sort_unstable();
        }
        expected.sort_unstable();
        
        result == expected
    }
    
    fn verify_no_duplicates(result: &[Vec<i32>]) -> bool {
        let mut normalized = result.to_vec();
        for subset in &mut normalized {
            subset.sort_unstable();
        }
        
        let set: HashSet<Vec<i32>> = normalized.iter().cloned().collect();
        set.len() == result.len()
    }
    
    #[test]
    fn test_basic_with_duplicates() {
        let nums = vec![1, 2, 2];
        let expected = vec![
            vec![],
            vec![1],
            vec![1, 2],
            vec![1, 2, 2],
            vec![2],
            vec![2, 2],
        ];
        
        let result1 = Solution::subsets_with_dup_backtracking(nums.clone());
        assert!(verify_no_duplicates(&result1));
        assert!(normalize_and_verify(result1, expected.clone()));
        
        let result2 = Solution::subsets_with_dup_iterative(nums.clone());
        assert!(verify_no_duplicates(&result2));
        assert!(normalize_and_verify(result2, expected.clone()));
        
        let result3 = Solution::subsets_with_dup_bit_set(nums.clone());
        assert!(verify_no_duplicates(&result3));
        assert!(normalize_and_verify(result3, expected.clone()));
        
        let result4 = Solution::subsets_with_dup_cascading(nums.clone());
        assert!(verify_no_duplicates(&result4));
        assert!(normalize_and_verify(result4, expected.clone()));
        
        let result5 = Solution::subsets_with_dup_dfs(nums.clone());
        assert!(verify_no_duplicates(&result5));
        assert!(normalize_and_verify(result5, expected.clone()));
        
        let result6 = Solution::subsets_with_dup_frequency(nums.clone());
        assert!(verify_no_duplicates(&result6));
        assert!(normalize_and_verify(result6, expected.clone()));
    }
    
    #[test]
    fn test_no_duplicates() {
        let nums = vec![1, 2, 3];
        let expected_count = 8; // 2^3
        
        let result1 = Solution::subsets_with_dup_backtracking(nums.clone());
        assert_eq!(result1.len(), expected_count);
        assert!(verify_no_duplicates(&result1));
        
        let result2 = Solution::subsets_with_dup_iterative(nums.clone());
        assert_eq!(result2.len(), expected_count);
        assert!(verify_no_duplicates(&result2));
        
        let result3 = Solution::subsets_with_dup_bit_set(nums.clone());
        assert_eq!(result3.len(), expected_count);
        assert!(verify_no_duplicates(&result3));
        
        let result4 = Solution::subsets_with_dup_cascading(nums.clone());
        assert_eq!(result4.len(), expected_count);
        assert!(verify_no_duplicates(&result4));
        
        let result5 = Solution::subsets_with_dup_dfs(nums.clone());
        assert_eq!(result5.len(), expected_count);
        assert!(verify_no_duplicates(&result5));
        
        let result6 = Solution::subsets_with_dup_frequency(nums.clone());
        assert_eq!(result6.len(), expected_count);
        assert!(verify_no_duplicates(&result6));
    }
    
    #[test]
    fn test_all_duplicates() {
        let nums = vec![2, 2, 2];
        let expected = vec![
            vec![],
            vec![2],
            vec![2, 2],
            vec![2, 2, 2],
        ];
        
        let result1 = Solution::subsets_with_dup_backtracking(nums.clone());
        assert!(verify_no_duplicates(&result1));
        assert!(normalize_and_verify(result1, expected.clone()));
        
        let result2 = Solution::subsets_with_dup_iterative(nums.clone());
        assert!(verify_no_duplicates(&result2));
        assert!(normalize_and_verify(result2, expected.clone()));
        
        let result3 = Solution::subsets_with_dup_bit_set(nums.clone());
        assert!(verify_no_duplicates(&result3));
        assert!(normalize_and_verify(result3, expected.clone()));
        
        let result4 = Solution::subsets_with_dup_cascading(nums.clone());
        assert!(verify_no_duplicates(&result4));
        assert!(normalize_and_verify(result4, expected.clone()));
        
        let result5 = Solution::subsets_with_dup_dfs(nums.clone());
        assert!(verify_no_duplicates(&result5));
        assert!(normalize_and_verify(result5, expected.clone()));
        
        let result6 = Solution::subsets_with_dup_frequency(nums.clone());
        assert!(verify_no_duplicates(&result6));
        assert!(normalize_and_verify(result6, expected.clone()));
    }
    
    #[test]
    fn test_single_element() {
        let nums = vec![1];
        let expected = vec![vec![], vec![1]];
        
        let result1 = Solution::subsets_with_dup_backtracking(nums.clone());
        assert!(normalize_and_verify(result1, expected.clone()));
        
        let result2 = Solution::subsets_with_dup_iterative(nums.clone());
        assert!(normalize_and_verify(result2, expected.clone()));
        
        let result3 = Solution::subsets_with_dup_bit_set(nums.clone());
        assert!(normalize_and_verify(result3, expected.clone()));
        
        let result4 = Solution::subsets_with_dup_cascading(nums.clone());
        assert!(normalize_and_verify(result4, expected.clone()));
        
        let result5 = Solution::subsets_with_dup_dfs(nums.clone());
        assert!(normalize_and_verify(result5, expected.clone()));
        
        let result6 = Solution::subsets_with_dup_frequency(nums.clone());
        assert!(normalize_and_verify(result6, expected.clone()));
    }
    
    #[test]
    fn test_mixed_duplicates() {
        let nums = vec![4, 4, 4, 1, 4];
        let expected = vec![
            vec![],
            vec![1],
            vec![1, 4],
            vec![1, 4, 4],
            vec![1, 4, 4, 4],
            vec![1, 4, 4, 4, 4],
            vec![4],
            vec![4, 4],
            vec![4, 4, 4],
            vec![4, 4, 4, 4],
        ];
        
        let result1 = Solution::subsets_with_dup_backtracking(nums.clone());
        assert!(verify_no_duplicates(&result1));
        assert!(normalize_and_verify(result1, expected.clone()));
        
        let result2 = Solution::subsets_with_dup_iterative(nums.clone());
        assert!(verify_no_duplicates(&result2));
        assert!(normalize_and_verify(result2, expected.clone()));
        
        let result3 = Solution::subsets_with_dup_bit_set(nums.clone());
        assert!(verify_no_duplicates(&result3));
        assert!(normalize_and_verify(result3, expected.clone()));
        
        let result4 = Solution::subsets_with_dup_cascading(nums.clone());
        assert!(verify_no_duplicates(&result4));
        assert!(normalize_and_verify(result4, expected.clone()));
        
        let result5 = Solution::subsets_with_dup_dfs(nums.clone());
        assert!(verify_no_duplicates(&result5));
        assert!(normalize_and_verify(result5, expected.clone()));
        
        let result6 = Solution::subsets_with_dup_frequency(nums.clone());
        assert!(verify_no_duplicates(&result6));
        assert!(normalize_and_verify(result6, expected.clone()));
    }
    
    #[test]
    fn test_negative_numbers_with_duplicates() {
        let nums = vec![-1, -1, 0, 1, 1];
        
        let result1 = Solution::subsets_with_dup_backtracking(nums.clone());
        assert!(verify_no_duplicates(&result1));
        
        let result2 = Solution::subsets_with_dup_iterative(nums.clone());
        assert!(verify_no_duplicates(&result2));
        
        let result3 = Solution::subsets_with_dup_bit_set(nums.clone());
        assert!(verify_no_duplicates(&result3));
        
        let result4 = Solution::subsets_with_dup_cascading(nums.clone());
        assert!(verify_no_duplicates(&result4));
        
        let result5 = Solution::subsets_with_dup_dfs(nums.clone());
        assert!(verify_no_duplicates(&result5));
        
        let result6 = Solution::subsets_with_dup_frequency(nums.clone());
        assert!(verify_no_duplicates(&result6));
        
        // All methods should produce the same number of unique subsets
        assert_eq!(result1.len(), result2.len());
        assert_eq!(result2.len(), result3.len());
        assert_eq!(result3.len(), result4.len());
        assert_eq!(result4.len(), result5.len());
        assert_eq!(result5.len(), result6.len());
    }
    
    #[test]
    fn test_large_input_all_same() {
        let nums = vec![1; 10];
        let expected_count = 11; // Can choose 0 to 10 ones
        
        let result1 = Solution::subsets_with_dup_backtracking(nums.clone());
        assert_eq!(result1.len(), expected_count);
        assert!(verify_no_duplicates(&result1));
        
        let result2 = Solution::subsets_with_dup_iterative(nums.clone());
        assert_eq!(result2.len(), expected_count);
        assert!(verify_no_duplicates(&result2));
        
        let result3 = Solution::subsets_with_dup_bit_set(nums.clone());
        assert_eq!(result3.len(), expected_count);
        assert!(verify_no_duplicates(&result3));
        
        let result4 = Solution::subsets_with_dup_cascading(nums.clone());
        assert_eq!(result4.len(), expected_count);
        assert!(verify_no_duplicates(&result4));
        
        let result5 = Solution::subsets_with_dup_dfs(nums.clone());
        assert_eq!(result5.len(), expected_count);
        assert!(verify_no_duplicates(&result5));
        
        let result6 = Solution::subsets_with_dup_frequency(nums.clone());
        assert_eq!(result6.len(), expected_count);
        assert!(verify_no_duplicates(&result6));
    }
    
    #[test]
    fn test_two_groups_of_duplicates() {
        let nums = vec![1, 1, 2, 2];
        let expected = vec![
            vec![],
            vec![1],
            vec![1, 1],
            vec![1, 1, 2],
            vec![1, 1, 2, 2],
            vec![1, 2],
            vec![1, 2, 2],
            vec![2],
            vec![2, 2],
        ];
        
        let result1 = Solution::subsets_with_dup_backtracking(nums.clone());
        assert!(verify_no_duplicates(&result1));
        assert!(normalize_and_verify(result1, expected.clone()));
        
        let result2 = Solution::subsets_with_dup_iterative(nums.clone());
        assert!(verify_no_duplicates(&result2));
        assert!(normalize_and_verify(result2, expected.clone()));
        
        let result3 = Solution::subsets_with_dup_bit_set(nums.clone());
        assert!(verify_no_duplicates(&result3));
        assert!(normalize_and_verify(result3, expected.clone()));
        
        let result4 = Solution::subsets_with_dup_cascading(nums.clone());
        assert!(verify_no_duplicates(&result4));
        assert!(normalize_and_verify(result4, expected.clone()));
        
        let result5 = Solution::subsets_with_dup_dfs(nums.clone());
        assert!(verify_no_duplicates(&result5));
        assert!(normalize_and_verify(result5, expected.clone()));
        
        let result6 = Solution::subsets_with_dup_frequency(nums.clone());
        assert!(verify_no_duplicates(&result6));
        assert!(normalize_and_verify(result6, expected.clone()));
    }
    
    #[test]
    fn test_empty_set_included() {
        let test_cases = vec![
            vec![1, 2, 2],
            vec![0],
            vec![1, 1, 1],
            vec![-1, 0, 1, 1],
        ];
        
        for nums in test_cases {
            let result1 = Solution::subsets_with_dup_backtracking(nums.clone());
            assert!(result1.iter().any(|s| s.is_empty()));
            
            let result2 = Solution::subsets_with_dup_iterative(nums.clone());
            assert!(result2.iter().any(|s| s.is_empty()));
            
            let result3 = Solution::subsets_with_dup_bit_set(nums.clone());
            assert!(result3.iter().any(|s| s.is_empty()));
            
            let result4 = Solution::subsets_with_dup_cascading(nums.clone());
            assert!(result4.iter().any(|s| s.is_empty()));
            
            let result5 = Solution::subsets_with_dup_dfs(nums.clone());
            assert!(result5.iter().any(|s| s.is_empty()));
            
            let result6 = Solution::subsets_with_dup_frequency(nums.clone());
            assert!(result6.iter().any(|s| s.is_empty()));
        }
    }
    
    #[test]
    fn test_consistency_across_methods() {
        let test_cases = vec![
            vec![1, 2, 2],
            vec![0, 0, 0],
            vec![1, 2, 3, 3],
            vec![-1, -1, 0, 1],
        ];
        
        for nums in test_cases {
            let mut result1 = Solution::subsets_with_dup_backtracking(nums.clone());
            let mut result2 = Solution::subsets_with_dup_iterative(nums.clone());
            let mut result3 = Solution::subsets_with_dup_bit_set(nums.clone());
            let mut result4 = Solution::subsets_with_dup_cascading(nums.clone());
            let mut result5 = Solution::subsets_with_dup_dfs(nums.clone());
            let mut result6 = Solution::subsets_with_dup_frequency(nums.clone());
            
            // Normalize all results
            for subset in &mut result1 { subset.sort_unstable(); }
            for subset in &mut result2 { subset.sort_unstable(); }
            for subset in &mut result3 { subset.sort_unstable(); }
            for subset in &mut result4 { subset.sort_unstable(); }
            for subset in &mut result5 { subset.sort_unstable(); }
            for subset in &mut result6 { subset.sort_unstable(); }
            
            result1.sort_unstable();
            result2.sort_unstable();
            result3.sort_unstable();
            result4.sort_unstable();
            result5.sort_unstable();
            result6.sort_unstable();
            
            assert_eq!(result1, result2);
            assert_eq!(result2, result3);
            assert_eq!(result3, result4);
            assert_eq!(result4, result5);
            assert_eq!(result5, result6);
        }
    }
    
    #[test]
    fn test_three_different_groups() {
        let nums = vec![1, 2, 2, 3, 3, 3];
        
        let result1 = Solution::subsets_with_dup_backtracking(nums.clone());
        assert!(verify_no_duplicates(&result1));
        
        let result2 = Solution::subsets_with_dup_iterative(nums.clone());
        assert!(verify_no_duplicates(&result2));
        
        let result3 = Solution::subsets_with_dup_bit_set(nums.clone());
        assert!(verify_no_duplicates(&result3));
        
        let result4 = Solution::subsets_with_dup_cascading(nums.clone());
        assert!(verify_no_duplicates(&result4));
        
        let result5 = Solution::subsets_with_dup_dfs(nums.clone());
        assert!(verify_no_duplicates(&result5));
        
        let result6 = Solution::subsets_with_dup_frequency(nums.clone());
        assert!(verify_no_duplicates(&result6));
        
        // Expected: 1 * 3 * 4 = 12 subsets
        // (0 or 1 of "1") * (0, 1, or 2 of "2") * (0, 1, 2, or 3 of "3")
        assert_eq!(result1.len(), 2 * 3 * 4);
        assert_eq!(result2.len(), 2 * 3 * 4);
        assert_eq!(result3.len(), 2 * 3 * 4);
        assert_eq!(result4.len(), 2 * 3 * 4);
        assert_eq!(result5.len(), 2 * 3 * 4);
        assert_eq!(result6.len(), 2 * 3 * 4);
    }
    
    #[test]
    fn test_unsorted_input() {
        let nums = vec![2, 1, 2, 1];
        let expected = vec![
            vec![],
            vec![1],
            vec![1, 1],
            vec![1, 1, 2],
            vec![1, 1, 2, 2],
            vec![1, 2],
            vec![1, 2, 2],
            vec![2],
            vec![2, 2],
        ];
        
        let result1 = Solution::subsets_with_dup_backtracking(nums.clone());
        assert!(verify_no_duplicates(&result1));
        assert!(normalize_and_verify(result1, expected.clone()));
        
        let result2 = Solution::subsets_with_dup_iterative(nums.clone());
        assert!(verify_no_duplicates(&result2));
        assert!(normalize_and_verify(result2, expected.clone()));
        
        let result3 = Solution::subsets_with_dup_bit_set(nums.clone());
        assert!(verify_no_duplicates(&result3));
        assert!(normalize_and_verify(result3, expected.clone()));
        
        let result4 = Solution::subsets_with_dup_cascading(nums.clone());
        assert!(verify_no_duplicates(&result4));
        assert!(normalize_and_verify(result4, expected.clone()));
        
        let result5 = Solution::subsets_with_dup_dfs(nums.clone());
        assert!(verify_no_duplicates(&result5));
        assert!(normalize_and_verify(result5, expected.clone()));
        
        let result6 = Solution::subsets_with_dup_frequency(nums.clone());
        assert!(verify_no_duplicates(&result6));
        assert!(normalize_and_verify(result6, expected.clone()));
    }
}