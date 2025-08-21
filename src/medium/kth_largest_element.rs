//! Problem 215: Kth Largest Element in an Array
//! 
//! Given an integer array nums and an integer k, return the kth largest element in the array.
//! Note that it is the kth largest element in the sorted order, not the kth distinct element.
//! You must solve it in O(n) time complexity.
//! 
//! Example 1:
//! Input: nums = [3,2,1,5,6,4], k = 2
//! Output: 5
//! 
//! Example 2:
//! Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
//! Output: 4

use std::collections::BinaryHeap;
use std::cmp::Reverse;
use rand::Rng;

pub struct Solution;

impl Solution {
    /// Approach 1: Min Heap of Size K
    /// 
    /// Maintains a min heap of size k. The root of the heap will be the kth largest element.
    /// This approach is optimal when k is small compared to n.
    /// 
    /// Time Complexity: O(n log k)
    /// Space Complexity: O(k)
    pub fn find_kth_largest_heap(&self, nums: Vec<i32>, k: i32) -> i32 {
        let k = k as usize;
        let mut min_heap = BinaryHeap::new();
        
        for num in nums {
            min_heap.push(Reverse(num));
            if min_heap.len() > k {
                min_heap.pop();
            }
        }
        
        min_heap.peek().unwrap().0
    }
    
    /// Approach 2: Quickselect Algorithm
    /// 
    /// Uses the partition logic from quicksort to find the kth largest element.
    /// Average case is O(n), worst case is O(n²) but can be made O(n) with median-of-medians.
    /// 
    /// Time Complexity: O(n) average, O(n²) worst case
    /// Space Complexity: O(1) iterative version
    pub fn find_kth_largest_quickselect(&self, mut nums: Vec<i32>, k: i32) -> i32 {
        let k = k as usize;
        let target_index = nums.len() - k; // Convert to 0-indexed from end
        
        let mut left = 0;
        let mut right = nums.len() - 1;
        
        loop {
            let pivot_index = self.partition(&mut nums, left, right);
            
            if pivot_index == target_index {
                return nums[pivot_index];
            } else if pivot_index < target_index {
                left = pivot_index + 1;
            } else {
                right = pivot_index - 1;
            }
        }
    }
    
    fn partition(&self, nums: &mut Vec<i32>, left: usize, right: usize) -> usize {
        // Use random pivot for better average performance
        let pivot_idx = left + (right - left) / 2;
        nums.swap(pivot_idx, right);
        
        let pivot = nums[right];
        let mut i = left;
        
        for j in left..right {
            if nums[j] <= pivot {
                nums.swap(i, j);
                i += 1;
            }
        }
        
        nums.swap(i, right);
        i
    }
    
    /// Approach 3: Counting Sort (for bounded range)
    /// 
    /// When the range of values is known and reasonable, counting sort can achieve O(n + range).
    /// This approach assumes values are in a reasonable range.
    /// 
    /// Time Complexity: O(n + range)
    /// Space Complexity: O(range)
    pub fn find_kth_largest_counting(&self, nums: Vec<i32>, k: i32) -> i32 {
        let min_val = *nums.iter().min().unwrap();
        let max_val = *nums.iter().max().unwrap();
        let range = (max_val - min_val + 1) as usize;
        
        // If range is too large, fall back to sorting
        if range > 100000 {
            let mut sorted_nums = nums;
            sorted_nums.sort_unstable();
            return sorted_nums[sorted_nums.len() - k as usize];
        }
        
        let mut count = vec![0; range];
        
        for num in nums {
            count[(num - min_val) as usize] += 1;
        }
        
        let mut remaining = k;
        for i in (0..range).rev() {
            remaining -= count[i];
            if remaining <= 0 {
                return min_val + i as i32;
            }
        }
        
        unreachable!()
    }
    
    /// Approach 4: Max Heap with All Elements
    /// 
    /// Builds a max heap with all elements and pops k-1 times to get the kth largest.
    /// Simple but not optimal for large arrays with small k.
    /// 
    /// Time Complexity: O(n + k log n)
    /// Space Complexity: O(n)
    pub fn find_kth_largest_max_heap(&self, nums: Vec<i32>, k: i32) -> i32 {
        let mut max_heap = BinaryHeap::from(nums);
        
        for _ in 0..k - 1 {
            max_heap.pop();
        }
        
        *max_heap.peek().unwrap()
    }
    
    /// Approach 5: Randomized Quickselect
    /// 
    /// Improved quickselect with randomized pivot selection to ensure O(n) average performance.
    /// The randomization makes worst-case O(n²) very unlikely.
    /// 
    /// Time Complexity: O(n) average, O(n²) worst case (very unlikely)
    /// Space Complexity: O(1)
    pub fn find_kth_largest_randomized(&self, mut nums: Vec<i32>, k: i32) -> i32 {
        
        let k = k as usize;
        let target_index = nums.len() - k;
        
        let mut left = 0;
        let mut right = nums.len() - 1;
        let mut rng = rand::thread_rng();
        
        loop {
            let pivot_index = self.randomized_partition(&mut nums, left, right, &mut rng);
            
            if pivot_index == target_index {
                return nums[pivot_index];
            } else if pivot_index < target_index {
                left = pivot_index + 1;
            } else {
                right = pivot_index - 1;
            }
        }
    }
    
    fn randomized_partition(&self, nums: &mut Vec<i32>, left: usize, right: usize, rng: &mut impl Rng) -> usize {
        let random_idx = rng.gen_range(left..=right);
        nums.swap(random_idx, right);
        
        let pivot = nums[right];
        let mut i = left;
        
        for j in left..right {
            if nums[j] <= pivot {
                nums.swap(i, j);
                i += 1;
            }
        }
        
        nums.swap(i, right);
        i
    }
    
    /// Approach 6: Bucket Sort Approach
    /// 
    /// Distributes elements into buckets and finds the kth largest by processing buckets.
    /// Useful when we want to avoid worst-case scenarios of quickselect.
    /// 
    /// Time Complexity: O(n) average
    /// Space Complexity: O(n)
    pub fn find_kth_largest_bucket(&self, nums: Vec<i32>, k: i32) -> i32 {
        let min_val = *nums.iter().min().unwrap();
        let max_val = *nums.iter().max().unwrap();
        
        if min_val == max_val {
            return min_val;
        }
        
        let bucket_count = (nums.len() / 2).max(1);
        let bucket_size = ((max_val - min_val) as f64 / bucket_count as f64).ceil() as i32;
        
        let mut buckets: Vec<Vec<i32>> = vec![Vec::new(); bucket_count];
        
        for num in nums {
            let bucket_idx = ((num - min_val) / bucket_size).min(bucket_count as i32 - 1) as usize;
            buckets[bucket_idx].push(num);
        }
        
        // Sort each bucket
        for bucket in &mut buckets {
            bucket.sort_unstable();
        }
        
        let mut remaining = k;
        for bucket in buckets.iter().rev() {
            if remaining <= bucket.len() as i32 {
                return bucket[bucket.len() - remaining as usize];
            }
            remaining -= bucket.len() as i32;
        }
        
        unreachable!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_heap() {
        let solution = Solution;
        
        assert_eq!(solution.find_kth_largest_heap(vec![3,2,1,5,6,4], 2), 5);
        assert_eq!(solution.find_kth_largest_heap(vec![3,2,3,1,2,4,5,5,6], 4), 4);
        assert_eq!(solution.find_kth_largest_heap(vec![1], 1), 1);
        assert_eq!(solution.find_kth_largest_heap(vec![1,2], 1), 2);
        assert_eq!(solution.find_kth_largest_heap(vec![1,2], 2), 1);
    }
    
    #[test]
    fn test_quickselect() {
        let solution = Solution;
        
        assert_eq!(solution.find_kth_largest_quickselect(vec![3,2,1,5,6,4], 2), 5);
        assert_eq!(solution.find_kth_largest_quickselect(vec![3,2,3,1,2,4,5,5,6], 4), 4);
        assert_eq!(solution.find_kth_largest_quickselect(vec![1], 1), 1);
        assert_eq!(solution.find_kth_largest_quickselect(vec![1,2], 1), 2);
        assert_eq!(solution.find_kth_largest_quickselect(vec![1,2], 2), 1);
    }
    
    #[test]
    fn test_counting() {
        let solution = Solution;
        
        assert_eq!(solution.find_kth_largest_counting(vec![3,2,1,5,6,4], 2), 5);
        assert_eq!(solution.find_kth_largest_counting(vec![3,2,3,1,2,4,5,5,6], 4), 4);
        assert_eq!(solution.find_kth_largest_counting(vec![1], 1), 1);
        assert_eq!(solution.find_kth_largest_counting(vec![1,2], 1), 2);
        assert_eq!(solution.find_kth_largest_counting(vec![1,2], 2), 1);
    }
    
    #[test]
    fn test_max_heap() {
        let solution = Solution;
        
        assert_eq!(solution.find_kth_largest_max_heap(vec![3,2,1,5,6,4], 2), 5);
        assert_eq!(solution.find_kth_largest_max_heap(vec![3,2,3,1,2,4,5,5,6], 4), 4);
        assert_eq!(solution.find_kth_largest_max_heap(vec![1], 1), 1);
        assert_eq!(solution.find_kth_largest_max_heap(vec![1,2], 1), 2);
        assert_eq!(solution.find_kth_largest_max_heap(vec![1,2], 2), 1);
    }
    
    #[test]
    fn test_randomized() {
        let solution = Solution;
        
        assert_eq!(solution.find_kth_largest_randomized(vec![3,2,1,5,6,4], 2), 5);
        assert_eq!(solution.find_kth_largest_randomized(vec![3,2,3,1,2,4,5,5,6], 4), 4);
        assert_eq!(solution.find_kth_largest_randomized(vec![1], 1), 1);
        assert_eq!(solution.find_kth_largest_randomized(vec![1,2], 1), 2);
        assert_eq!(solution.find_kth_largest_randomized(vec![1,2], 2), 1);
    }
    
    #[test]
    fn test_bucket() {
        let solution = Solution;
        
        assert_eq!(solution.find_kth_largest_bucket(vec![3,2,1,5,6,4], 2), 5);
        assert_eq!(solution.find_kth_largest_bucket(vec![3,2,3,1,2,4,5,5,6], 4), 4);
        assert_eq!(solution.find_kth_largest_bucket(vec![1], 1), 1);
        assert_eq!(solution.find_kth_largest_bucket(vec![1,2], 1), 2);
        assert_eq!(solution.find_kth_largest_bucket(vec![1,2], 2), 1);
    }
    
    #[test]
    fn test_edge_cases() {
        let solution = Solution;
        
        // All same elements
        assert_eq!(solution.find_kth_largest_heap(vec![1,1,1,1], 2), 1);
        assert_eq!(solution.find_kth_largest_quickselect(vec![1,1,1,1], 2), 1);
        
        // k equals array length (smallest element)
        assert_eq!(solution.find_kth_largest_heap(vec![3,2,1,5,6,4], 6), 1);
        assert_eq!(solution.find_kth_largest_quickselect(vec![3,2,1,5,6,4], 6), 1);
        
        // k equals 1 (largest element)
        assert_eq!(solution.find_kth_largest_heap(vec![3,2,1,5,6,4], 1), 6);
        assert_eq!(solution.find_kth_largest_quickselect(vec![3,2,1,5,6,4], 1), 6);
        
        // Negative numbers
        assert_eq!(solution.find_kth_largest_heap(vec![-1,-3,2,0,-8,4], 2), 2);
        assert_eq!(solution.find_kth_largest_quickselect(vec![-1,-3,2,0,-8,4], 2), 2);
        
        // Large array with duplicates
        let large_array = vec![1; 1000];
        assert_eq!(solution.find_kth_largest_heap(large_array.clone(), 500), 1);
        assert_eq!(solution.find_kth_largest_quickselect(large_array, 500), 1);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let solution = Solution;
        
        let test_cases = vec![
            (vec![3,2,1,5,6,4], 2),
            (vec![3,2,3,1,2,4,5,5,6], 4),
            (vec![1], 1),
            (vec![1,2], 1),
            (vec![1,2], 2),
            (vec![1,1,1,1], 2),
            (vec![3,2,1,5,6,4], 6),
            (vec![3,2,1,5,6,4], 1),
            (vec![-1,-3,2,0,-8,4], 2),
            (vec![7,10,4,3,20,15], 3),
            (vec![1,2,3,4,5,6,7,8,9,10], 5),
        ];
        
        for (nums, k) in test_cases {
            let heap = solution.find_kth_largest_heap(nums.clone(), k);
            let quickselect = solution.find_kth_largest_quickselect(nums.clone(), k);
            let counting = solution.find_kth_largest_counting(nums.clone(), k);
            let max_heap = solution.find_kth_largest_max_heap(nums.clone(), k);
            let randomized = solution.find_kth_largest_randomized(nums.clone(), k);
            let bucket = solution.find_kth_largest_bucket(nums.clone(), k);
            
            assert_eq!(heap, quickselect, "Heap and quickselect differ for {:?}, k={}", nums, k);
            assert_eq!(heap, counting, "Heap and counting differ for {:?}, k={}", nums, k);
            assert_eq!(heap, max_heap, "Heap and max_heap differ for {:?}, k={}", nums, k);
            assert_eq!(heap, randomized, "Heap and randomized differ for {:?}, k={}", nums, k);
            assert_eq!(heap, bucket, "Heap and bucket differ for {:?}, k={}", nums, k);
        }
    }
    
    #[test]
    fn test_performance_characteristics() {
        let solution = Solution;
        
        // Test with sorted array (worst case for basic quickselect)
        let sorted_desc = (1..=100).rev().collect::<Vec<i32>>();
        let sorted_asc = (1..=100).collect::<Vec<i32>>();
        
        assert_eq!(solution.find_kth_largest_heap(sorted_desc.clone(), 25), 76);
        assert_eq!(solution.find_kth_largest_quickselect(sorted_desc.clone(), 25), 76);
        assert_eq!(solution.find_kth_largest_randomized(sorted_desc, 25), 76);
        
        assert_eq!(solution.find_kth_largest_heap(sorted_asc.clone(), 25), 76);
        assert_eq!(solution.find_kth_largest_quickselect(sorted_asc.clone(), 25), 76);
        assert_eq!(solution.find_kth_largest_randomized(sorted_asc, 25), 76);
        
        // Test with many duplicates
        let mut many_dups = vec![5; 50];
        many_dups.extend(vec![10; 30]);
        many_dups.extend(vec![1; 20]);
        
        assert_eq!(solution.find_kth_largest_heap(many_dups.clone(), 20), 10);
        assert_eq!(solution.find_kth_largest_quickselect(many_dups.clone(), 20), 10);
        assert_eq!(solution.find_kth_largest_bucket(many_dups, 20), 10);
    }
    
    #[test]
    fn test_large_range_counting() {
        let solution = Solution;
        
        // Test case where counting sort falls back to regular sorting
        let large_range = vec![1, 1000000, 2, 999999, 3];
        assert_eq!(solution.find_kth_largest_counting(large_range, 3), 3);
        
        // Test case where counting sort works efficiently
        let small_range = vec![10, 15, 12, 18, 11, 14, 13, 16, 17, 19];
        assert_eq!(solution.find_kth_largest_counting(small_range, 5), 15);
    }
    
    #[test]
    fn test_bucket_sort_edge_cases() {
        let solution = Solution;
        
        // Test with all same values
        assert_eq!(solution.find_kth_largest_bucket(vec![5, 5, 5, 5], 2), 5);
        
        // Test with two distinct values
        assert_eq!(solution.find_kth_largest_bucket(vec![1, 2, 1, 2, 1], 3), 1);
        
        // Test with wide range
        assert_eq!(solution.find_kth_largest_bucket(vec![1, 100, 50, 75, 25], 2), 75);
    }
    
    #[test]
    fn test_randomized_stability() {
        let solution = Solution;
        
        // Run randomized version multiple times to ensure stability
        let nums = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        let k = 6;
        
        let expected = solution.find_kth_largest_heap(nums.clone(), k);
        
        for _ in 0..10 {
            assert_eq!(solution.find_kth_largest_randomized(nums.clone(), k), expected);
        }
    }
}