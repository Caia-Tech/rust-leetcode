//! # 46. Permutations
//!
//! Given an array `nums` of distinct integers, return all the possible permutations. You can return the answer in any order.
//!
//! **Example 1:**
//! ```
//! Input: nums = [1,2,3]
//! Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
//! ```
//!
//! **Example 2:**
//! ```
//! Input: nums = [0,1]
//! Output: [[0,1],[1,0]]
//! ```
//!
//! **Example 3:**
//! ```
//! Input: nums = [1]
//! Output: [[1]]
//! ```
//!
//! **Constraints:**
//! - 1 <= nums.length <= 6
//! - -10 <= nums[i] <= 10
//! - All the integers of nums are unique.

/// Solution for Permutations - 6 different approaches
pub struct Solution;

impl Solution {
    /// Approach 1: Classic Backtracking with Used Array
    /// 
    /// Use backtracking to generate all permutations by tracking which elements are used.
    /// At each position, try every unused element.
    ///
    /// Time Complexity: O(N! * N) where N is array length
    /// Space Complexity: O(N!) for storing all permutations + O(N) for recursion
    pub fn permute_backtrack(&self, nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut result = Vec::new();
        let mut current = Vec::new();
        let mut used = vec![false; nums.len()];
        
        self.backtrack(&nums, &mut current, &mut used, &mut result);
        result
    }
    
    fn backtrack(&self, nums: &[i32], current: &mut Vec<i32>, used: &mut [bool], result: &mut Vec<Vec<i32>>) {
        if current.len() == nums.len() {
            result.push(current.clone());
            return;
        }
        
        for i in 0..nums.len() {
            if !used[i] {
                used[i] = true;
                current.push(nums[i]);
                self.backtrack(nums, current, used, result);
                current.pop();
                used[i] = false;
            }
        }
    }
    
    /// Approach 2: Backtracking with In-Place Swapping
    /// 
    /// Instead of tracking used elements, swap elements to front and recurse.
    /// This avoids the need for a separate used array.
    ///
    /// Time Complexity: O(N! * N)
    /// Space Complexity: O(N!) for storing results + O(N) for recursion
    pub fn permute_swap(&self, nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut result = Vec::new();
        let mut nums = nums;
        self.backtrack_swap(&mut nums, 0, &mut result);
        result
    }
    
    fn backtrack_swap(&self, nums: &mut Vec<i32>, start: usize, result: &mut Vec<Vec<i32>>) {
        if start == nums.len() {
            result.push(nums.clone());
            return;
        }
        
        for i in start..nums.len() {
            nums.swap(start, i);
            self.backtrack_swap(nums, start + 1, result);
            nums.swap(start, i); // backtrack
        }
    }
    
    /// Approach 3: Iterative Approach
    /// 
    /// Build permutations iteratively by inserting each new element
    /// into all possible positions of existing permutations.
    ///
    /// Time Complexity: O(N! * N)
    /// Space Complexity: O(N!) for storing results
    pub fn permute_iterative(&self, nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut result = vec![Vec::new()];
        
        for num in nums {
            let mut new_perms = Vec::new();
            
            for perm in result {
                for i in 0..=perm.len() {
                    let mut new_perm = perm.clone();
                    new_perm.insert(i, num);
                    new_perms.push(new_perm);
                }
            }
            
            result = new_perms;
        }
        
        result
    }
    
    /// Approach 4: Using Heap's Algorithm
    /// 
    /// Generate permutations using Heap's algorithm for minimal number of swaps.
    /// More efficient in terms of swap operations.
    ///
    /// Time Complexity: O(N!)
    /// Space Complexity: O(N!) for storing results + O(N) for recursion
    pub fn permute_heaps(&self, nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut result = Vec::new();
        let mut nums = nums;
        let n = nums.len();
        self.heaps_algorithm(&mut nums, n, &mut result);
        result
    }
    
    fn heaps_algorithm(&self, nums: &mut Vec<i32>, k: usize, result: &mut Vec<Vec<i32>>) {
        if k == 1 {
            result.push(nums.clone());
            return;
        }
        
        for i in 0..k {
            self.heaps_algorithm(nums, k - 1, result);
            
            if k % 2 == 1 {
                nums.swap(0, k - 1);
            } else {
                nums.swap(i, k - 1);
            }
        }
    }
    
    /// Approach 5: Using Standard Library Permutations (Educational)
    /// 
    /// Demonstrate how to use Rust's built-in permutation capabilities.
    /// Note: This approach is mainly educational since LeetCode expects custom implementation.
    ///
    /// Time Complexity: O(N!)
    /// Space Complexity: O(N!)
    pub fn permute_builtin(&self, nums: Vec<i32>) -> Vec<Vec<i32>> {
        use std::collections::HashSet;
        
        // Generate all permutations using recursive approach similar to backtracking
        let mut result = Vec::new();
        let mut used = HashSet::new();
        let mut current = Vec::new();
        
        self.generate_with_set(&nums, &mut used, &mut current, &mut result);
        result
    }
    
    fn generate_with_set(&self, nums: &[i32], used: &mut std::collections::HashSet<usize>, 
                        current: &mut Vec<i32>, result: &mut Vec<Vec<i32>>) {
        if current.len() == nums.len() {
            result.push(current.clone());
            return;
        }
        
        for i in 0..nums.len() {
            if !used.contains(&i) {
                used.insert(i);
                current.push(nums[i]);
                self.generate_with_set(nums, used, current, result);
                current.pop();
                used.remove(&i);
            }
        }
    }
    
    /// Approach 6: Lexicographic Generation
    /// 
    /// Generate permutations in lexicographic order by finding the next permutation iteratively.
    /// This approach generates permutations one by one without recursion.
    ///
    /// Time Complexity: O(N! * N)
    /// Space Complexity: O(N!) for storing results
    pub fn permute_lexicographic(&self, nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut result = Vec::new();
        let mut nums = nums;
        nums.sort(); // Start with lexicographically smallest permutation
        
        loop {
            result.push(nums.clone());
            if !self.next_permutation(&mut nums) {
                break;
            }
        }
        
        result
    }
    
    /// Generate the next lexicographic permutation
    /// Returns false if no next permutation exists
    fn next_permutation(&self, nums: &mut Vec<i32>) -> bool {
        let len = nums.len();
        if len <= 1 {
            return false;
        }
        
        // Find the largest index i such that nums[i] < nums[i+1]
        let mut i = len - 2;
        while i > 0 && nums[i] >= nums[i + 1] {
            i -= 1;
        }
        
        if i == 0 && nums[i] >= nums[i + 1] {
            return false; // No next permutation
        }
        
        // Find the largest index j such that nums[i] < nums[j]
        let mut j = len - 1;
        while nums[j] <= nums[i] {
            j -= 1;
        }
        
        // Swap nums[i] and nums[j]
        nums.swap(i, j);
        
        // Reverse the suffix starting at index i+1
        nums[i + 1..].reverse();
        
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sort_result(mut result: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        result.sort();
        result
    }

    #[test]
    fn test_basic_cases() {
        let solution = Solution;
        
        // Example 1: [1,2,3]
        let expected = vec![
            vec![1,2,3], vec![1,3,2], vec![2,1,3], 
            vec![2,3,1], vec![3,1,2], vec![3,2,1]
        ];
        assert_eq!(sort_result(solution.permute_backtrack(vec![1,2,3])), sort_result(expected.clone()));
        assert_eq!(sort_result(solution.permute_swap(vec![1,2,3])), sort_result(expected.clone()));
        assert_eq!(sort_result(solution.permute_iterative(vec![1,2,3])), sort_result(expected.clone()));
        assert_eq!(sort_result(solution.permute_heaps(vec![1,2,3])), sort_result(expected.clone()));
        assert_eq!(sort_result(solution.permute_builtin(vec![1,2,3])), sort_result(expected.clone()));
        assert_eq!(sort_result(solution.permute_lexicographic(vec![1,2,3])), sort_result(expected.clone()));
        
        // Example 2: [0,1]
        let expected2 = vec![vec![0,1], vec![1,0]];
        assert_eq!(sort_result(solution.permute_backtrack(vec![0,1])), sort_result(expected2.clone()));
        assert_eq!(sort_result(solution.permute_swap(vec![0,1])), sort_result(expected2.clone()));
        assert_eq!(sort_result(solution.permute_iterative(vec![0,1])), sort_result(expected2.clone()));
        assert_eq!(sort_result(solution.permute_heaps(vec![0,1])), sort_result(expected2.clone()));
        assert_eq!(sort_result(solution.permute_builtin(vec![0,1])), sort_result(expected2.clone()));
        assert_eq!(sort_result(solution.permute_lexicographic(vec![0,1])), sort_result(expected2.clone()));
        
        // Example 3: [1]
        let expected3 = vec![vec![1]];
        assert_eq!(solution.permute_backtrack(vec![1]), expected3);
        assert_eq!(solution.permute_swap(vec![1]), expected3);
        assert_eq!(solution.permute_iterative(vec![1]), expected3);
        assert_eq!(solution.permute_heaps(vec![1]), expected3);
        assert_eq!(solution.permute_builtin(vec![1]), expected3);
        assert_eq!(solution.permute_lexicographic(vec![1]), expected3);
    }

    #[test]
    fn test_edge_cases() {
        let solution = Solution;
        
        // Single negative number
        let expected_neg = vec![vec![-1]];
        assert_eq!(solution.permute_backtrack(vec![-1]), expected_neg);
        assert_eq!(solution.permute_swap(vec![-1]), expected_neg);
        assert_eq!(solution.permute_iterative(vec![-1]), expected_neg);
        assert_eq!(solution.permute_heaps(vec![-1]), expected_neg);
        assert_eq!(solution.permute_builtin(vec![-1]), expected_neg);
        assert_eq!(solution.permute_lexicographic(vec![-1]), expected_neg);
    }

    #[test]
    fn test_larger_arrays() {
        let solution = Solution;
        
        // Four elements
        let nums = vec![1,2,3,4];
        let result = solution.permute_backtrack(nums.clone());
        assert_eq!(result.len(), 24); // 4! = 24
        
        // All approaches should give same results
        let result1 = sort_result(solution.permute_backtrack(nums.clone()));
        let result2 = sort_result(solution.permute_swap(nums.clone()));
        let result3 = sort_result(solution.permute_iterative(nums.clone()));
        let result4 = sort_result(solution.permute_heaps(nums.clone()));
        let result5 = sort_result(solution.permute_builtin(nums.clone()));
        let result6 = sort_result(solution.permute_lexicographic(nums.clone()));
        
        assert_eq!(result1, result2);
        assert_eq!(result2, result3);
        assert_eq!(result3, result4);
        assert_eq!(result4, result5);
        assert_eq!(result5, result6);
    }

    #[test]
    fn test_negative_numbers() {
        let solution = Solution;
        
        // Mix of positive and negative
        let expected = vec![
            vec![-1,1], vec![1,-1]
        ];
        assert_eq!(sort_result(solution.permute_backtrack(vec![-1,1])), sort_result(expected.clone()));
        assert_eq!(sort_result(solution.permute_swap(vec![-1,1])), sort_result(expected.clone()));
        assert_eq!(sort_result(solution.permute_iterative(vec![-1,1])), sort_result(expected.clone()));
        assert_eq!(sort_result(solution.permute_heaps(vec![-1,1])), sort_result(expected.clone()));
        assert_eq!(sort_result(solution.permute_builtin(vec![-1,1])), sort_result(expected.clone()));
        assert_eq!(sort_result(solution.permute_lexicographic(vec![-1,1])), sort_result(expected.clone()));
    }

    #[test]
    fn test_zeros() {
        let solution = Solution;
        
        // Include zero
        let expected = vec![
            vec![0,2], vec![2,0]
        ];
        assert_eq!(sort_result(solution.permute_backtrack(vec![0,2])), sort_result(expected.clone()));
        assert_eq!(sort_result(solution.permute_swap(vec![0,2])), sort_result(expected.clone()));
        assert_eq!(sort_result(solution.permute_iterative(vec![0,2])), sort_result(expected.clone()));
        assert_eq!(sort_result(solution.permute_heaps(vec![0,2])), sort_result(expected.clone()));
        assert_eq!(sort_result(solution.permute_builtin(vec![0,2])), sort_result(expected.clone()));
        assert_eq!(sort_result(solution.permute_lexicographic(vec![0,2])), sort_result(expected.clone()));
    }

    #[test]
    fn test_three_elements_different_orders() {
        let solution = Solution;
        
        // Different input orders should give same permutation sets
        let nums1 = vec![1,2,3];
        let nums2 = vec![3,1,2]; 
        let nums3 = vec![2,3,1];
        
        let result1 = sort_result(solution.permute_backtrack(nums1));
        let result2 = sort_result(solution.permute_backtrack(nums2));
        let result3 = sort_result(solution.permute_backtrack(nums3));
        
        assert_eq!(result1, result2);
        assert_eq!(result2, result3);
        assert_eq!(result1.len(), 6); // 3! = 6
    }

    #[test]
    fn test_boundary_values() {
        let solution = Solution;
        
        // Maximum negative value in constraints
        let result = solution.permute_backtrack(vec![-10]);
        assert_eq!(result, vec![vec![-10]]);
        
        // Maximum positive value in constraints
        let result = solution.permute_backtrack(vec![10]);
        assert_eq!(result, vec![vec![10]]);
        
        // Mix of boundary values
        let expected = vec![vec![-10,10], vec![10,-10]];
        assert_eq!(sort_result(solution.permute_backtrack(vec![-10,10])), sort_result(expected));
    }

    #[test]
    fn test_factorial_counts() {
        let solution = Solution;
        
        // Verify factorial property
        let test_cases = vec![
            (vec![1], 1),           // 1! = 1
            (vec![1,2], 2),         // 2! = 2
            (vec![1,2,3], 6),       // 3! = 6
            (vec![1,2,3,4], 24),    // 4! = 24
            (vec![1,2,3,4,5], 120), // 5! = 120
        ];
        
        for (nums, expected_count) in test_cases {
            let result = solution.permute_backtrack(nums);
            assert_eq!(result.len(), expected_count);
        }
    }

    #[test]
    fn test_all_unique_permutations() {
        let solution = Solution;
        
        // Ensure all permutations are unique
        let nums = vec![1,2,3];
        let result = solution.permute_backtrack(nums);
        
        let mut sorted_result = result.clone();
        sorted_result.sort();
        sorted_result.dedup();
        
        assert_eq!(result.len(), sorted_result.len());
        
        // Each permutation should be different
        for i in 0..result.len() {
            for j in i+1..result.len() {
                assert_ne!(result[i], result[j]);
            }
        }
    }

    #[test]
    fn test_all_elements_used() {
        let solution = Solution;
        
        // Each permutation should use all original elements exactly once
        let nums = vec![1,2,3,4];
        let result = solution.permute_backtrack(nums.clone());
        
        for perm in result {
            assert_eq!(perm.len(), nums.len());
            let mut sorted_perm = perm.clone();
            sorted_perm.sort();
            let mut sorted_nums = nums.clone();
            sorted_nums.sort();
            assert_eq!(sorted_perm, sorted_nums);
        }
    }

    #[test]
    fn test_approach_performance() {
        let solution = Solution;
        
        // All approaches should handle reasonable input sizes
        let nums = vec![1,2,3,4,5];
        
        let result1 = solution.permute_backtrack(nums.clone());
        let result2 = solution.permute_swap(nums.clone());
        let result3 = solution.permute_iterative(nums.clone());
        let result4 = solution.permute_heaps(nums.clone());
        let result5 = solution.permute_builtin(nums.clone());
        let result6 = solution.permute_lexicographic(nums.clone());
        
        assert_eq!(result1.len(), 120);
        assert_eq!(result2.len(), 120);
        assert_eq!(result3.len(), 120);
        assert_eq!(result4.len(), 120);
        assert_eq!(result5.len(), 120);
        assert_eq!(result6.len(), 120);
    }
}