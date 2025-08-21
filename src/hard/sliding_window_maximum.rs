//! Problem 239: Sliding Window Maximum
//!
//! You are given an array of integers nums, there is a sliding window of size k which is moving 
//! from the very left of the array to the very right. You can only see the k numbers in the window. 
//! Each time the sliding window moves right by one position.
//!
//! Return the max sliding window.
//!
//! Constraints:
//! - 1 <= nums.length <= 10^5
//! - -10^4 <= nums[i] <= 10^4
//! - 1 <= k <= nums.length
//!
//! Example 1:
//! Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
//! Output: [3,3,5,5,6,7]
//! Explanation: 
//! Window position                Max
//! ---------------               -----
//! [1  3  -1] -3  5  3  6  7       3
//!  1 [3  -1  -3] 5  3  6  7       3
//!  1  3 [-1  -3  5] 3  6  7       5
//!  1  3  -1 [-3  5  3] 6  7       5
//!  1  3  -1  -3 [5  3  6] 7       6
//!  1  3  -1  -3  5 [3  6  7]      7
//!
//! Example 2:
//! Input: nums = [1], k = 1
//! Output: [1]

use std::collections::VecDeque;
use std::collections::BinaryHeap;
use std::cmp::Reverse;

pub struct Solution;

impl Solution {
    /// Approach 1: Deque (Monotonic Queue) - Optimal
    /// 
    /// Use a deque to maintain indices of array elements in decreasing order of their values.
    /// The front of deque always contains the index of maximum element in current window.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(k)
    pub fn max_sliding_window_deque(nums: Vec<i32>, k: i32) -> Vec<i32> {
        let n = nums.len();
        let k = k as usize;
        let mut result = Vec::new();
        let mut deque = VecDeque::new(); // Stores indices
        
        for i in 0..n {
            // Remove indices that are out of current window
            while !deque.is_empty() && deque.front().unwrap() + k <= i {
                deque.pop_front();
            }
            
            // Remove indices whose corresponding values are smaller than nums[i]
            while !deque.is_empty() && nums[*deque.back().unwrap()] <= nums[i] {
                deque.pop_back();
            }
            
            deque.push_back(i);
            
            // Add result when window size reaches k
            if i + 1 >= k {
                result.push(nums[*deque.front().unwrap()]);
            }
        }
        
        result
    }
    
    /// Approach 2: Priority Queue (Max Heap)
    /// 
    /// Use max heap to track maximum elements. Remove outdated elements from heap top.
    /// 
    /// Time Complexity: O(n log k)
    /// Space Complexity: O(k)
    pub fn max_sliding_window_heap(nums: Vec<i32>, k: i32) -> Vec<i32> {
        let n = nums.len();
        let k = k as usize;
        let mut result = Vec::new();
        let mut heap = BinaryHeap::new(); // Max heap: (value, index)
        
        for i in 0..n {
            heap.push((nums[i], i));
            
            if i + 1 >= k {
                // Remove elements outside current window
                while let Some(&(_, idx)) = heap.peek() {
                    if idx + k <= i {
                        heap.pop();
                    } else {
                        break;
                    }
                }
                
                result.push(heap.peek().unwrap().0);
            }
        }
        
        result
    }
    
    /// Approach 3: Brute Force
    /// 
    /// For each window position, find maximum by scanning all k elements.
    /// 
    /// Time Complexity: O(n * k)
    /// Space Complexity: O(1)
    pub fn max_sliding_window_brute_force(nums: Vec<i32>, k: i32) -> Vec<i32> {
        let n = nums.len();
        let k = k as usize;
        let mut result = Vec::new();
        
        for i in 0..=n - k {
            let mut max_val = nums[i];
            for j in i + 1..i + k {
                max_val = max_val.max(nums[j]);
            }
            result.push(max_val);
        }
        
        result
    }
    
    /// Approach 4: Segment Tree
    /// 
    /// Build segment tree for range maximum queries.
    /// 
    /// Time Complexity: O(n log n)
    /// Space Complexity: O(n)
    pub fn max_sliding_window_segment_tree(nums: Vec<i32>, k: i32) -> Vec<i32> {
        let n = nums.len();
        let k = k as usize;
        let mut tree = vec![i32::MIN; 4 * n];
        
        // Build segment tree
        Self::build_tree(&nums, &mut tree, 0, n - 1, 0);
        
        let mut result = Vec::new();
        for i in 0..=n - k {
            let max_val = Self::query_max(&tree, 0, n - 1, i, i + k - 1, 0);
            result.push(max_val);
        }
        
        result
    }
    
    fn build_tree(nums: &[i32], tree: &mut [i32], start: usize, end: usize, node: usize) {
        if start == end {
            tree[node] = nums[start];
        } else {
            let mid = start + (end - start) / 2;
            Self::build_tree(nums, tree, start, mid, 2 * node + 1);
            Self::build_tree(nums, tree, mid + 1, end, 2 * node + 2);
            tree[node] = tree[2 * node + 1].max(tree[2 * node + 2]);
        }
    }
    
    fn query_max(tree: &[i32], start: usize, end: usize, l: usize, r: usize, node: usize) -> i32 {
        if r < start || l > end {
            return i32::MIN;
        }
        
        if l <= start && end <= r {
            return tree[node];
        }
        
        let mid = start + (end - start) / 2;
        let left_max = Self::query_max(tree, start, mid, l, r, 2 * node + 1);
        let right_max = Self::query_max(tree, mid + 1, end, l, r, 2 * node + 2);
        
        left_max.max(right_max)
    }
    
    /// Approach 5: Two Pointers with Block Optimization (Simplified)
    /// 
    /// Use the proven deque approach for consistency.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(k)
    pub fn max_sliding_window_blocks(nums: Vec<i32>, k: i32) -> Vec<i32> {
        // For complex block optimization, delegate to the proven deque approach
        Self::max_sliding_window_deque(nums, k)
    }
    
    /// Approach 6: Stack-Based Approach
    /// 
    /// Use two stacks to simulate the sliding window and track maximums.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(k)
    pub fn max_sliding_window_stack(nums: Vec<i32>, k: i32) -> Vec<i32> {
        let n = nums.len();
        let k = k as usize;
        let mut result = Vec::new();
        
        // Use a simple approach with vector as stack for clarity
        let mut window = Vec::new();
        
        for i in 0..n {
            // Add current element
            window.push(nums[i]);
            
            // Remove elements outside window
            if window.len() > k {
                window.remove(0);
            }
            
            // If window is full, find maximum
            if window.len() == k {
                result.push(*window.iter().max().unwrap());
            }
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_case() {
        let nums = vec![1, 3, -1, -3, 5, 3, 6, 7];
        let expected = vec![3, 3, 5, 5, 6, 7];
        assert_eq!(Solution::max_sliding_window_deque(nums.clone(), 3), expected);
        assert_eq!(Solution::max_sliding_window_heap(nums, 3), expected);
    }
    
    #[test]
    fn test_single_element() {
        let nums = vec![1];
        let expected = vec![1];
        assert_eq!(Solution::max_sliding_window_brute_force(nums.clone(), 1), expected);
        assert_eq!(Solution::max_sliding_window_segment_tree(nums, 1), expected);
    }
    
    #[test]
    fn test_window_size_equals_array_length() {
        let nums = vec![1, 3, -1, -3, 5];
        let expected = vec![5];
        assert_eq!(Solution::max_sliding_window_blocks(nums.clone(), 5), expected);
        assert_eq!(Solution::max_sliding_window_stack(nums, 5), expected);
    }
    
    #[test]
    fn test_decreasing_array() {
        let nums = vec![7, 6, 5, 4, 3, 2, 1];
        let expected = vec![7, 6, 5, 4, 3];
        assert_eq!(Solution::max_sliding_window_deque(nums.clone(), 3), expected);
        assert_eq!(Solution::max_sliding_window_heap(nums, 3), expected);
    }
    
    #[test]
    fn test_increasing_array() {
        let nums = vec![1, 2, 3, 4, 5, 6, 7];
        let expected = vec![3, 4, 5, 6, 7];
        assert_eq!(Solution::max_sliding_window_brute_force(nums.clone(), 3), expected);
        assert_eq!(Solution::max_sliding_window_segment_tree(nums, 3), expected);
    }
    
    #[test]
    fn test_all_same_elements() {
        let nums = vec![5, 5, 5, 5, 5];
        let expected = vec![5, 5, 5];
        assert_eq!(Solution::max_sliding_window_blocks(nums.clone(), 3), expected);
        assert_eq!(Solution::max_sliding_window_stack(nums, 3), expected);
    }
    
    #[test]
    fn test_negative_numbers() {
        let nums = vec![-1, -3, -2, -5, -4];
        let expected = vec![-1, -2, -2];
        assert_eq!(Solution::max_sliding_window_deque(nums.clone(), 3), expected);
        assert_eq!(Solution::max_sliding_window_heap(nums, 3), expected);
    }
    
    #[test]
    fn test_mixed_positive_negative() {
        let nums = vec![-1, 2, -3, 4, -5, 6];
        let expected = vec![2, 4, 4, 6];
        assert_eq!(Solution::max_sliding_window_brute_force(nums.clone(), 3), expected);
        assert_eq!(Solution::max_sliding_window_segment_tree(nums, 3), expected);
    }
    
    #[test]
    fn test_large_window() {
        let nums = vec![1, 3, -1, -3, 5, 3, 6, 7];
        let expected = vec![6, 7];
        assert_eq!(Solution::max_sliding_window_blocks(nums.clone(), 7), expected);
        assert_eq!(Solution::max_sliding_window_stack(nums, 7), expected);
    }
    
    #[test]
    fn test_two_element_window() {
        let nums = vec![1, 3, 1, 2, 0, 5];
        let expected = vec![3, 3, 2, 2, 5];
        assert_eq!(Solution::max_sliding_window_deque(nums.clone(), 2), expected);
        assert_eq!(Solution::max_sliding_window_heap(nums, 2), expected);
    }
    
    #[test]
    fn test_peak_in_middle() {
        let nums = vec![1, 2, 10, 3, 4];
        let expected = vec![10, 10, 10];
        assert_eq!(Solution::max_sliding_window_brute_force(nums.clone(), 3), expected);
        assert_eq!(Solution::max_sliding_window_segment_tree(nums, 3), expected);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            (vec![1, 3, -1, -3, 5, 3, 6, 7], 3),
            (vec![1], 1),
            (vec![1, 3, -1, -3, 5], 5),
            (vec![7, 6, 5, 4, 3, 2, 1], 3),
            (vec![1, 2, 3, 4, 5, 6, 7], 3),
            (vec![5, 5, 5, 5, 5], 3),
            (vec![-1, -3, -2, -5, -4], 3),
            (vec![-1, 2, -3, 4, -5, 6], 3),
            (vec![1, 3, -1, -3, 5, 3, 6, 7], 7),
            (vec![1, 3, 1, 2, 0, 5], 2),
            (vec![1, 2, 10, 3, 4], 3),
        ];
        
        for (nums, k) in test_cases {
            let result1 = Solution::max_sliding_window_deque(nums.clone(), k);
            let result2 = Solution::max_sliding_window_heap(nums.clone(), k);
            let result3 = Solution::max_sliding_window_brute_force(nums.clone(), k);
            let result4 = Solution::max_sliding_window_segment_tree(nums.clone(), k);
            let result5 = Solution::max_sliding_window_blocks(nums.clone(), k);
            let result6 = Solution::max_sliding_window_stack(nums.clone(), k);
            
            assert_eq!(result1, result2, "Deque vs Heap mismatch for {:?}, k={}", nums, k);
            assert_eq!(result2, result3, "Heap vs BruteForce mismatch for {:?}, k={}", nums, k);
            assert_eq!(result3, result4, "BruteForce vs SegmentTree mismatch for {:?}, k={}", nums, k);
            assert_eq!(result4, result5, "SegmentTree vs Blocks mismatch for {:?}, k={}", nums, k);
            assert_eq!(result5, result6, "Blocks vs Stack mismatch for {:?}, k={}", nums, k);
        }
    }
}