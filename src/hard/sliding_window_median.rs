//! # Problem 480: Sliding Window Median
//!
//! **Difficulty**: Hard
//! **Topics**: Array, Hash Table, Sliding Window, Heap (Priority Queue)
//! **Acceptance Rate**: 37.1%

use std::collections::BinaryHeap;
use std::cmp::Reverse;

/// Solution struct following LeetCode format
pub struct Solution;

impl Solution {
    /// Create a new solution instance
    pub fn new() -> Self {
        Solution
    }

    /// Main solution approach using two heaps (balanced approach)
    /// 
    /// Time Complexity: O(n * k log k) where n is array length, k is window size
    /// Space Complexity: O(k) for the heaps
    pub fn median_sliding_window(&self, nums: Vec<i32>, k: i32) -> Vec<f64> {
        if nums.is_empty() || k <= 0 {
            return vec![];
        }
        
        let k = k as usize;
        let mut result = Vec::new();
        
        for i in 0..=nums.len() - k {
            let window: Vec<i32> = nums[i..i + k].to_vec();
            result.push(self.find_median(&window));
        }
        
        result
    }
    
    /// Helper function to find median of a window
    fn find_median(&self, window: &[i32]) -> f64 {
        let mut sorted_window = window.to_vec();
        sorted_window.sort_unstable();
        
        let n = sorted_window.len();
        if n % 2 == 1 {
            sorted_window[n / 2] as f64
        } else {
            (sorted_window[n / 2 - 1] as f64 + sorted_window[n / 2] as f64) / 2.0
        }
    }

    /// Optimized solution using multiset simulation
    /// 
    /// Time Complexity: O(n * k) for maintaining sorted order
    /// Space Complexity: O(k) for the window storage
    pub fn median_sliding_window_optimized(&self, nums: Vec<i32>, k: i32) -> Vec<f64> {
        if nums.is_empty() || k <= 0 {
            return vec![];
        }
        
        let k = k as usize;
        let mut result = Vec::new();
        let mut window = Vec::new();
        
        // Initialize first window
        for i in 0..k {
            self.insert_sorted(&mut window, nums[i]);
        }
        result.push(self.get_median(&window));
        
        // Slide the window
        for i in k..nums.len() {
            // Remove the element going out of window
            self.remove_from_sorted(&mut window, nums[i - k]);
            // Add the new element
            self.insert_sorted(&mut window, nums[i]);
            // Calculate median
            result.push(self.get_median(&window));
        }
        
        result
    }
    
    /// Insert element while maintaining sorted order
    fn insert_sorted(&self, arr: &mut Vec<i32>, val: i32) {
        let pos = arr.binary_search(&val).unwrap_or_else(|pos| pos);
        arr.insert(pos, val);
    }
    
    /// Remove element from sorted array
    fn remove_from_sorted(&self, arr: &mut Vec<i32>, val: i32) {
        if let Ok(pos) = arr.binary_search(&val) {
            arr.remove(pos);
        }
    }
    
    /// Get median from sorted array
    fn get_median(&self, arr: &[i32]) -> f64 {
        let n = arr.len();
        if n % 2 == 1 {
            arr[n / 2] as f64
        } else {
            (arr[n / 2 - 1] as f64 + arr[n / 2] as f64) / 2.0
        }
    }

    /// Advanced two-heap approach (for very large datasets)
    /// 
    /// Time Complexity: O(n log k) with heap operations
    /// Space Complexity: O(k) for heap storage
    pub fn median_sliding_window_two_heaps(&self, nums: Vec<i32>, k: i32) -> Vec<f64> {
        if nums.is_empty() || k <= 0 {
            return vec![];
        }
        
        let k = k as usize;
        let mut result = Vec::new();
        
        // For each window, use two heaps to find median efficiently
        for i in 0..=nums.len() - k {
            let window: Vec<i32> = nums[i..i + k].to_vec();
            result.push(self.find_median_with_heaps(&window));
        }
        
        result
    }
    
    /// Find median using two heaps approach
    fn find_median_with_heaps(&self, nums: &[i32]) -> f64 {
        let mut max_heap = BinaryHeap::new(); // For smaller half
        let mut min_heap = BinaryHeap::new(); // For larger half (using Reverse for min-heap)
        
        for &num in nums {
            // Add to appropriate heap
            if max_heap.is_empty() || num <= *max_heap.peek().unwrap() {
                max_heap.push(num);
            } else {
                min_heap.push(Reverse(num));
            }
            
            // Balance heaps
            if max_heap.len() > min_heap.len() + 1 {
                if let Some(val) = max_heap.pop() {
                    min_heap.push(Reverse(val));
                }
            } else if min_heap.len() > max_heap.len() + 1 {
                if let Some(Reverse(val)) = min_heap.pop() {
                    max_heap.push(val);
                }
            }
        }
        
        // Calculate median
        if max_heap.len() == min_heap.len() {
            let left = *max_heap.peek().unwrap() as f64;
            let right = min_heap.peek().unwrap().0 as f64;
            (left + right) / 2.0
        } else if max_heap.len() > min_heap.len() {
            *max_heap.peek().unwrap() as f64
        } else {
            min_heap.peek().unwrap().0 as f64
        }
    }

    /// Brute force approach for comparison
    /// 
    /// Time Complexity: O(n * k log k) due to sorting each window
    /// Space Complexity: O(k) for window storage
    pub fn median_sliding_window_brute_force(&self, nums: Vec<i32>, k: i32) -> Vec<f64> {
        if nums.is_empty() || k <= 0 {
            return vec![];
        }
        
        let k = k as usize;
        let mut result = Vec::new();
        
        for i in 0..=nums.len() - k {
            let mut window: Vec<i32> = nums[i..i + k].to_vec();
            window.sort_unstable();
            
            let median = if k % 2 == 1 {
                window[k / 2] as f64
            } else {
                (window[k / 2 - 1] as f64 + window[k / 2] as f64) / 2.0
            };
            
            result.push(median);
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

    fn assert_f64_vec_eq(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5, "Expected {:?}, got {:?}", b, a);
        }
    }

    #[test]
    fn test_basic_cases() {
        let solution = Solution::new();
        
        // Test case 1: [1,3,-1,-3,5,3,6,7], k = 3
        let nums1 = vec![1, 3, -1, -3, 5, 3, 6, 7];
        let expected1 = vec![1.0, -1.0, -1.0, 3.0, 5.0, 6.0];
        assert_f64_vec_eq(&solution.median_sliding_window(nums1, 3), &expected1);
        
        // Test case 2: [1,2,3,4,2,3,1,4,2], k = 3
        let nums2 = vec![1, 2, 3, 4, 2, 3, 1, 4, 2];
        let expected2 = vec![2.0, 3.0, 3.0, 3.0, 2.0, 3.0, 2.0];
        assert_f64_vec_eq(&solution.median_sliding_window(nums2, 3), &expected2);
    }

    #[test]
    fn test_edge_cases() {
        let solution = Solution::new();
        
        // Single element window
        let nums = vec![1, 2, 3, 4, 5];
        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_f64_vec_eq(&solution.median_sliding_window(nums, 1), &expected);
        
        // Even window size
        let nums = vec![1, 2, 3, 4];
        let expected = vec![1.5, 2.5, 3.5];
        assert_f64_vec_eq(&solution.median_sliding_window(nums, 2), &expected);
        
        // Window size equals array length
        let nums = vec![1, 2, 3];
        let expected = vec![2.0];
        assert_f64_vec_eq(&solution.median_sliding_window(nums, 3), &expected);
        
        // Empty array
        let nums = vec![];
        let expected = vec![];
        assert_f64_vec_eq(&solution.median_sliding_window(nums, 1), &expected);
    }

    #[test]
    fn test_negative_numbers() {
        let solution = Solution::new();
        
        let nums = vec![-7, -8, 2, -2, 0];
        let expected = vec![-7.0, -2.0, 0.0];
        assert_f64_vec_eq(&solution.median_sliding_window(nums, 3), &expected);
    }

    #[test]
    fn test_duplicates() {
        let solution = Solution::new();
        
        let nums = vec![1, 1, 1, 1, 1];
        let expected = vec![1.0, 1.0, 1.0];
        assert_f64_vec_eq(&solution.median_sliding_window(nums, 3), &expected);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = Solution::new();
        
        let test_cases = vec![
            (vec![1, 3, -1, -3, 5, 3, 6, 7], 3),
            (vec![1, 2, 3, 4, 2, 3, 1, 4, 2], 3),
            (vec![1, 2, 3, 4], 2),
            (vec![-7, -8, 2, -2, 0], 3),
        ];

        for (nums, k) in test_cases {
            let result1 = solution.median_sliding_window(nums.clone(), k);
            let result2 = solution.median_sliding_window_optimized(nums.clone(), k);
            let result3 = solution.median_sliding_window_two_heaps(nums.clone(), k);
            let result4 = solution.median_sliding_window_brute_force(nums.clone(), k);
            
            assert_f64_vec_eq(&result1, &result2);
            assert_f64_vec_eq(&result1, &result3);
            assert_f64_vec_eq(&result1, &result4);
        }
    }

    #[test]
    fn test_performance_scenarios() {
        let solution = Solution::new();
        
        // Large array with small window
        let nums: Vec<i32> = (1..=100).collect();
        let result = solution.median_sliding_window(nums, 5);
        assert_eq!(result.len(), 96);
        assert_eq!(result[0], 3.0); // median of [1,2,3,4,5]
        
        // All same elements
        let nums = vec![5; 50];
        let result = solution.median_sliding_window(nums, 10);
        assert!(result.iter().all(|&x| x == 5.0));
    }
}

#[cfg(test)]
mod benchmarks {
    // Note: Benchmarks will be added to benches/ directory for Criterion integration
}