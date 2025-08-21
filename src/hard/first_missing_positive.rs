//! Problem 41: First Missing Positive
//!
//! Given an unsorted integer array nums, return the smallest missing positive integer.
//! You must implement an algorithm that runs in O(n) time and uses O(1) auxiliary space.
//!
//! Constraints:
//! - 1 <= nums.length <= 10^5
//! - -2^31 <= nums[i] <= 2^31 - 1
//!
//! Example 1:
//! Input: nums = [1,2,0]
//! Output: 3
//! Explanation: The numbers in the range [1,2] are all in the array.
//!
//! Example 2:
//! Input: nums = [3,4,-1,1]
//! Output: 2
//! Explanation: 1 is in the array but 2 is missing.
//!
//! Example 3:
//! Input: nums = [7,8,9,11,12]
//! Output: 1
//! Explanation: The smallest positive integer 1 is missing.

use std::collections::HashSet;

pub struct Solution;

impl Solution {
    /// Approach 1: Cyclic Sort (Optimal)
    /// 
    /// Place each positive number i at position i-1. After placement,
    /// the first position with wrong number gives the answer.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn first_missing_positive_cyclic_sort(nums: Vec<i32>) -> i32 {
        let mut nums = nums;
        let n = nums.len();
        
        let mut i = 0;
        while i < n {
            // If nums[i] is in range [1,n] and not in correct position
            let val = nums[i] as usize;
            if val >= 1 && val <= n && nums[val - 1] != nums[i] {
                // Swap to place nums[i] at correct position
                nums.swap(i, val - 1);
                // Don't increment i, check the swapped element
            } else {
                i += 1;
            }
        }
        
        // Find first missing positive
        for i in 0..n {
            if nums[i] != (i + 1) as i32 {
                return (i + 1) as i32;
            }
        }
        
        // All positions 1 to n are filled correctly
        (n + 1) as i32
    }
    
    /// Approach 2: Mark and Sign (In-Place)
    /// 
    /// Use array indices as hash keys and sign as existence indicator.
    /// Mark presence of number x by making nums[x-1] negative.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn first_missing_positive_mark_sign(nums: Vec<i32>) -> i32 {
        let mut nums = nums;
        let n = nums.len();
        
        // Step 1: Handle numbers <= 0 and > n
        // Replace with n+1 (out of range positive number)
        for i in 0..n {
            if nums[i] <= 0 || nums[i] > n as i32 {
                nums[i] = (n + 1) as i32;
            }
        }
        
        // Step 2: Mark presence using sign
        for i in 0..n {
            let val = nums[i].abs() as usize;
            if val <= n {
                nums[val - 1] = -nums[val - 1].abs();
            }
        }
        
        // Step 3: Find first positive number
        for i in 0..n {
            if nums[i] > 0 {
                return (i + 1) as i32;
            }
        }
        
        (n + 1) as i32
    }
    
    /// Approach 3: HashSet (Extra Space)
    /// 
    /// Store all positive numbers in set, then check 1,2,3...
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    pub fn first_missing_positive_hashset(nums: Vec<i32>) -> i32 {
        let positive_nums: HashSet<i32> = nums.into_iter()
            .filter(|&x| x > 0)
            .collect();
        
        for i in 1..=positive_nums.len() as i32 + 1 {
            if !positive_nums.contains(&i) {
                return i;
            }
        }
        
        1
    }
    
    /// Approach 4: Sort First
    /// 
    /// Sort the array first, then find the first missing positive.
    /// 
    /// Time Complexity: O(n log n)
    /// Space Complexity: O(1) - if sorting is in-place
    pub fn first_missing_positive_sort_first(nums: Vec<i32>) -> i32 {
        let mut nums = nums;
        nums.sort_unstable();
        
        let mut expected = 1;
        for num in nums {
            if num == expected {
                expected += 1;
            } else if num > expected {
                break;
            }
            // Skip duplicates and negatives
        }
        
        expected
    }
    
    /// Approach 5: Two-Pass Indexing
    /// 
    /// First pass: move all numbers to their "correct" positions if possible.
    /// Second pass: find the first incorrect position.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn first_missing_positive_two_pass(nums: Vec<i32>) -> i32 {
        let mut nums = nums;
        let n = nums.len();
        
        // Pass 1: Move positive numbers in range [1,n] to correct positions
        for i in 0..n {
            while nums[i] > 0 && nums[i] <= n as i32 && 
                  nums[nums[i] as usize - 1] != nums[i] {
                let target_idx = nums[i] as usize - 1;
                let temp = nums[target_idx];
                nums[target_idx] = nums[i];
                nums[i] = temp;
            }
        }
        
        // Pass 2: Find first missing positive
        for i in 0..n {
            if nums[i] != (i + 1) as i32 {
                return (i + 1) as i32;
            }
        }
        
        (n + 1) as i32
    }
    
    /// Approach 6: Segregation + Marking
    /// 
    /// First segregate positive numbers, then mark their presence.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn first_missing_positive_segregation(nums: Vec<i32>) -> i32 {
        let mut nums = nums;
        let n = nums.len();
        
        // Step 1: Segregate positive numbers to the left
        let mut j = 0;
        for i in 0..n {
            if nums[i] > 0 {
                nums.swap(i, j);
                j += 1;
            }
        }
        
        // Now positive numbers are in [0..j)
        if j == 0 {
            return 1; // No positive numbers
        }
        
        // Step 2: Mark presence in the positive subarray
        for i in 0..n {
            let val = nums[i].abs();
            if val > 0 && val <= j as i32 {
                let idx = val as usize - 1;
                if nums[idx] > 0 {
                    nums[idx] = -nums[idx];
                }
            }
        }
        
        // Step 3: Find first unmarked position
        for i in 0..j {
            if nums[i] > 0 {
                return (i + 1) as i32;
            }
        }
        
        (j + 1) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_case() {
        assert_eq!(Solution::first_missing_positive_cyclic_sort(vec![1, 2, 0]), 3);
        assert_eq!(Solution::first_missing_positive_mark_sign(vec![1, 2, 0]), 3);
    }
    
    #[test]
    fn test_missing_in_middle() {
        assert_eq!(Solution::first_missing_positive_hashset(vec![3, 4, -1, 1]), 2);
        assert_eq!(Solution::first_missing_positive_sort_first(vec![3, 4, -1, 1]), 2);
    }
    
    #[test]
    fn test_missing_first() {
        assert_eq!(Solution::first_missing_positive_two_pass(vec![7, 8, 9, 11, 12]), 1);
        assert_eq!(Solution::first_missing_positive_segregation(vec![7, 8, 9, 11, 12]), 1);
    }
    
    #[test]
    fn test_single_element() {
        assert_eq!(Solution::first_missing_positive_cyclic_sort(vec![1]), 2);
        assert_eq!(Solution::first_missing_positive_mark_sign(vec![2]), 1);
        assert_eq!(Solution::first_missing_positive_hashset(vec![0]), 1);
    }
    
    #[test]
    fn test_consecutive_numbers() {
        assert_eq!(Solution::first_missing_positive_sort_first(vec![1, 2, 3, 4, 5]), 6);
        assert_eq!(Solution::first_missing_positive_two_pass(vec![5, 4, 3, 2, 1]), 6);
    }
    
    #[test]
    fn test_duplicates() {
        assert_eq!(Solution::first_missing_positive_segregation(vec![1, 1, 1, 1]), 2);
        assert_eq!(Solution::first_missing_positive_cyclic_sort(vec![1, 2, 2, 3]), 4);
    }
    
    #[test]
    fn test_negative_numbers() {
        assert_eq!(Solution::first_missing_positive_mark_sign(vec![-1, -2, -3]), 1);
        assert_eq!(Solution::first_missing_positive_hashset(vec![-10, -5, 0]), 1);
    }
    
    #[test]
    fn test_large_numbers() {
        assert_eq!(Solution::first_missing_positive_two_pass(vec![1000, 1001, 1002]), 1);
        assert_eq!(Solution::first_missing_positive_segregation(vec![2147483647, 1]), 2);
    }
    
    #[test]
    fn test_mixed_numbers() {
        assert_eq!(Solution::first_missing_positive_cyclic_sort(vec![0, 2, 2, 1, 1]), 3);
        assert_eq!(Solution::first_missing_positive_sort_first(vec![-1, 4, 2, 1, 9, 10]), 3);
    }
    
    #[test]
    fn test_gap_at_start() {
        assert_eq!(Solution::first_missing_positive_mark_sign(vec![2, 3, 4]), 1);
        assert_eq!(Solution::first_missing_positive_hashset(vec![5, 6, 7, 8]), 1);
    }
    
    #[test]
    fn test_unsorted_with_gaps() {
        assert_eq!(Solution::first_missing_positive_two_pass(vec![3, 1, 6, 4, 5, 8]), 2);
        assert_eq!(Solution::first_missing_positive_segregation(vec![1, 3, 5, 7, 9]), 2);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            vec![1, 2, 0],
            vec![3, 4, -1, 1],
            vec![7, 8, 9, 11, 12],
            vec![1],
            vec![2],
            vec![0],
            vec![1, 2, 3, 4, 5],
            vec![5, 4, 3, 2, 1],
            vec![1, 1, 1, 1],
            vec![1, 2, 2, 3],
            vec![-1, -2, -3],
            vec![-10, -5, 0],
            vec![1000, 1001, 1002],
            vec![2147483647, 1],
            vec![0, 2, 2, 1, 1],
            vec![-1, 4, 2, 1, 9, 10],
            vec![2, 3, 4],
            vec![5, 6, 7, 8],
            vec![3, 1, 6, 4, 5, 8],
            vec![1, 3, 5, 7, 9],
        ];
        
        for nums in test_cases {
            let result1 = Solution::first_missing_positive_cyclic_sort(nums.clone());
            let result2 = Solution::first_missing_positive_mark_sign(nums.clone());
            let result3 = Solution::first_missing_positive_hashset(nums.clone());
            let result4 = Solution::first_missing_positive_sort_first(nums.clone());
            let result5 = Solution::first_missing_positive_two_pass(nums.clone());
            let result6 = Solution::first_missing_positive_segregation(nums.clone());
            
            assert_eq!(result1, result2, "Cyclic vs Mark mismatch for {:?}", nums);
            assert_eq!(result2, result3, "Mark vs HashSet mismatch for {:?}", nums);
            assert_eq!(result3, result4, "HashSet vs Sort mismatch for {:?}", nums);
            assert_eq!(result4, result5, "Sort vs Two-pass mismatch for {:?}", nums);
            assert_eq!(result5, result6, "Two-pass vs Segregation mismatch for {:?}", nums);
        }
    }
}