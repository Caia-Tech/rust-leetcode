//! # Problem 1: Two Sum
//!
//! Given an array of integers `nums` and an integer `target`, return indices of the two numbers 
//! such that they add up to `target`. You may assume that each input would have exactly one solution, 
//! and you may not use the same element twice. You can return the answer in any order.
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::easy::two_sum::Solution;
//! 
//! let solution = Solution::new();
//! 
//! // Example 1:
//! let nums = vec![2, 7, 11, 15];
//! let target = 9;
//! let result = solution.two_sum(nums, target);
//! assert_eq!(result, vec![0, 1]); // Because nums[0] + nums[1] = 2 + 7 = 9
//! ```
//!
//! ## Constraints
//!
//! - 2 <= nums.length <= 10^4
//! - -10^9 <= nums[i] <= 10^9
//! - -10^9 <= target <= 10^9
//! - Only one valid answer exists.

use std::collections::HashMap;

/// Solution struct following LeetCode format
pub struct Solution;

impl Solution {
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Brute Force
    /// 
    /// **Algorithm:**
    /// 1. Use two nested loops to check all pairs
    /// 2. For each pair (i, j) where i < j, check if nums[i] + nums[j] == target
    /// 3. Return [i, j] when found
    /// 
    /// **Time Complexity:** O(n²) - We examine every possible pair
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **Use Case:** Good for understanding, but inefficient for large inputs
    pub fn two_sum_brute_force(&self, nums: Vec<i32>, target: i32) -> Vec<i32> {
        let n = nums.len();
        
        // Check every possible pair
        for i in 0..n {
            for j in (i + 1)..n {
                if nums[i] + nums[j] == target {
                    return vec![i as i32, j as i32];
                }
            }
        }
        
        // This should never happen given the problem constraints
        vec![]
    }

    /// # Approach 2: Hash Map (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Create a HashMap to store value -> index mappings
    /// 2. For each number, calculate complement = target - current_number
    /// 3. Check if complement exists in HashMap
    /// 4. If exists, return [complement_index, current_index]
    /// 5. If not exists, store current number and index in HashMap
    /// 
    /// **Time Complexity:** O(n) - Single pass through the array
    /// **Space Complexity:** O(n) - HashMap can store up to n elements
    /// 
    /// **Key Insight:** Instead of looking for nums[j] such that nums[i] + nums[j] = target,
    /// we look for target - nums[i] (the complement) in our hash map.
    pub fn two_sum(&self, nums: Vec<i32>, target: i32) -> Vec<i32> {
        let mut map: HashMap<i32, usize> = HashMap::new();
        
        for (i, &num) in nums.iter().enumerate() {
            let complement = target - num;
            
            // Check if complement exists in map
            if let Some(&complement_index) = map.get(&complement) {
                return vec![complement_index as i32, i as i32];
            }
            
            // Store current number and its index
            map.insert(num, i);
        }
        
        // This should never happen given the problem constraints
        vec![]
    }

    /// # Approach 3: Two-Pass Hash Map
    /// 
    /// **Algorithm:**
    /// 1. First pass: build HashMap with all value -> index mappings
    /// 2. Second pass: for each element, look for its complement
    /// 3. Ensure we don't use the same element twice (i != complement_index)
    /// 
    /// **Time Complexity:** O(n) - Two passes through the array
    /// **Space Complexity:** O(n) - HashMap stores all elements
    /// 
    /// **Trade-off:** Slightly more space usage but conceptually simpler
    pub fn two_sum_two_pass(&self, nums: Vec<i32>, target: i32) -> Vec<i32> {
        let mut map: HashMap<i32, usize> = HashMap::new();
        
        // First pass: build the hash map
        for (i, &num) in nums.iter().enumerate() {
            map.insert(num, i);
        }
        
        // Second pass: find complement
        for (i, &num) in nums.iter().enumerate() {
            let complement = target - num;
            if let Some(&j) = map.get(&complement) {
                if i != j {  // Don't use same element twice
                    return vec![i as i32, j as i32];
                }
            }
        }
        
        vec![]
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
    #[case(vec![2, 7, 11, 15], 9, vec![0, 1])]
    #[case(vec![3, 2, 4], 6, vec![1, 2])]
    #[case(vec![3, 3], 6, vec![0, 1])]
    #[case(vec![-1, -2, -3, -4, -5], -8, vec![2, 4])]
    #[case(vec![0, 4, 3, 0], 0, vec![0, 3])]
    fn test_two_sum_optimal(#[case] nums: Vec<i32>, #[case] target: i32, #[case] expected: Vec<i32>) {
        let solution = setup();
        let result = solution.two_sum(nums, target);
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(vec![2, 7, 11, 15], 9, vec![0, 1])]
    #[case(vec![3, 2, 4], 6, vec![1, 2])]
    #[case(vec![3, 3], 6, vec![0, 1])]
    fn test_two_sum_brute_force(#[case] nums: Vec<i32>, #[case] target: i32, #[case] expected: Vec<i32>) {
        let solution = setup();
        let result = solution.two_sum_brute_force(nums, target);
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(vec![2, 7, 11, 15], 9, vec![0, 1])]
    #[case(vec![3, 2, 4], 6, vec![1, 2])]
    #[case(vec![3, 3], 6, vec![0, 1])]
    fn test_two_sum_two_pass(#[case] nums: Vec<i32>, #[case] target: i32, #[case] expected: Vec<i32>) {
        let solution = setup();
        let result = solution.two_sum_two_pass(nums, target);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // Minimum size array
        assert_eq!(solution.two_sum(vec![1, 2], 3), vec![0, 1]);
        
        // Negative numbers
        assert_eq!(solution.two_sum(vec![-3, 4, 3, 90], 0), vec![0, 2]);
        
        // Large numbers within constraints
        let large_nums = vec![1000000000, -999999999, 1];
        assert_eq!(solution.two_sum(large_nums, 1000000001), vec![0, 2]);
    }

    #[test]
    fn test_all_approaches_consistency() {
        let solution = setup();
        let test_cases = vec![
            (vec![2, 7, 11, 15], 9),
            (vec![3, 2, 4], 6),
            (vec![3, 3], 6),
            (vec![-1, -2, -3, -4, -5], -8),
        ];

        for (nums, target) in test_cases {
            let result1 = solution.two_sum_brute_force(nums.clone(), target);
            let result2 = solution.two_sum(nums.clone(), target);
            let result3 = solution.two_sum_two_pass(nums.clone(), target);
            
            // All approaches should find valid solutions (may be different order)
            assert_eq!(nums[result1[0] as usize] + nums[result1[1] as usize], target);
            assert_eq!(nums[result2[0] as usize] + nums[result2[1] as usize], target);
            assert_eq!(nums[result3[0] as usize] + nums[result3[1] as usize], target);
        }
    }
}