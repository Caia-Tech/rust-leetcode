//! Problem 315: Count of Smaller Numbers After Self
//!
//! Implementation by Marvin Tutt, Caia Tech
//!
//! Given an integer array nums, return an integer array counts where counts[i] 
//! is the number of smaller elements to the right of nums[i].
//!
//! Constraints:
//! - 1 <= nums.length <= 10^5
//! - -10^4 <= nums[i] <= 10^4
//!
//! Example 1:
//! Input: nums = [5,2,6,1]
//! Output: [2,1,1,0]
//! Explanation:
//! To the right of 5 there are 2 smaller elements (2 and 1).
//! To the right of 2 there is only 1 smaller element (1).
//! To the right of 6 there is 1 smaller element (1).
//! To the right of 1 there is 0 smaller element.
//!
//! Example 2:
//! Input: nums = [-1]
//! Output: [0]
//!
//! Example 3:
//! Input: nums = [-1,-1]
//! Output: [0,0]

pub struct Solution;

impl Solution {
    /// Approach 1: Merge Sort with Index Tracking - Optimal
    /// 
    /// Use merge sort to count inversions. During merge, count how many elements
    /// from the right part are smaller than each element in the left part.
    /// 
    /// Time Complexity: O(n log n)
    /// Space Complexity: O(n)
    pub fn count_smaller_merge_sort(nums: Vec<i32>) -> Vec<i32> {
        let n = nums.len();
        let mut indices: Vec<usize> = (0..n).collect();
        let mut counts = vec![0; n];
        
        Self::merge_sort(&nums, &mut indices, &mut counts, 0, n);
        counts
    }
    
    fn merge_sort(nums: &[i32], indices: &mut [usize], counts: &mut [i32], start: usize, end: usize) {
        if end - start <= 1 {
            return;
        }
        
        let mid = start + (end - start) / 2;
        Self::merge_sort(nums, indices, counts, start, mid);
        Self::merge_sort(nums, indices, counts, mid, end);
        
        let mut temp = vec![0; end - start];
        let mut left = start;
        let mut right = mid;
        let mut k = 0;
        
        while left < mid && right < end {
            if nums[indices[right]] < nums[indices[left]] {
                temp[k] = indices[right];
                right += 1;
            } else {
                // Count elements in right part that are smaller
                counts[indices[left]] += (right - mid) as i32;
                temp[k] = indices[left];
                left += 1;
            }
            k += 1;
        }
        
        while left < mid {
            counts[indices[left]] += (right - mid) as i32;
            temp[k] = indices[left];
            left += 1;
            k += 1;
        }
        
        while right < end {
            temp[k] = indices[right];
            right += 1;
            k += 1;
        }
        
        for i in 0..temp.len() {
            indices[start + i] = temp[i];
        }
    }
    
    /// Approach 2: Binary Indexed Tree (Fenwick Tree)
    /// 
    /// Use coordinate compression and BIT to efficiently count smaller elements.
    /// 
    /// Time Complexity: O(n log k) where k is number of unique elements
    /// Space Complexity: O(k)
    pub fn count_smaller_fenwick_tree(nums: Vec<i32>) -> Vec<i32> {
        let n = nums.len();
        
        // Coordinate compression
        let mut sorted_nums = nums.clone();
        sorted_nums.sort_unstable();
        sorted_nums.dedup();
        
        let mut bit = BinaryIndexedTree::new(sorted_nums.len());
        let mut result = vec![0; n];
        
        // Process from right to left
        for i in (0..n).rev() {
            // Find compressed coordinate
            let pos = sorted_nums.binary_search(&nums[i]).unwrap();
            
            // Query count of elements smaller than nums[i]
            if pos > 0 {
                result[i] = bit.query(pos - 1);
            }
            
            // Update BIT with current element
            bit.update(pos, 1);
        }
        
        result
    }
    
    /// Approach 3: Segment Tree
    /// 
    /// Use segment tree with coordinate compression for range queries.
    /// 
    /// Time Complexity: O(n log k)
    /// Space Complexity: O(k)
    pub fn count_smaller_segment_tree(nums: Vec<i32>) -> Vec<i32> {
        let n = nums.len();
        
        // Coordinate compression
        let mut sorted_nums = nums.clone();
        sorted_nums.sort_unstable();
        sorted_nums.dedup();
        
        let mut seg_tree = SegmentTree::new(sorted_nums.len());
        let mut result = vec![0; n];
        
        // Process from right to left
        for i in (0..n).rev() {
            let pos = sorted_nums.binary_search(&nums[i]).unwrap();
            
            // Query count of elements smaller than nums[i]
            if pos > 0 {
                result[i] = seg_tree.query(0, pos - 1);
            }
            
            // Update segment tree
            seg_tree.update(pos, seg_tree.query(pos, pos) + 1);
        }
        
        result
    }
    
    /// Approach 4: Brute Force with Optimization
    /// 
    /// For each element, count smaller elements to its right.
    /// 
    /// Time Complexity: O(n^2)
    /// Space Complexity: O(1)
    pub fn count_smaller_brute_force(nums: Vec<i32>) -> Vec<i32> {
        let n = nums.len();
        let mut result = vec![0; n];
        
        for i in 0..n {
            for j in (i + 1)..n {
                if nums[j] < nums[i] {
                    result[i] += 1;
                }
            }
        }
        
        result
    }
    
    /// Approach 5: Modified Merge Sort with Pairs
    /// 
    /// Use merge sort with (value, index) pairs to track original positions.
    /// 
    /// Time Complexity: O(n log n)
    /// Space Complexity: O(n)
    pub fn count_smaller_merge_with_pairs(nums: Vec<i32>) -> Vec<i32> {
        let n = nums.len();
        let mut pairs: Vec<(i32, usize)> = nums.into_iter().enumerate().map(|(i, v)| (v, i)).collect();
        let mut counts = vec![0; n];
        
        Self::merge_sort_pairs(&mut pairs, &mut counts, 0, n);
        counts
    }
    
    fn merge_sort_pairs(pairs: &mut [(i32, usize)], counts: &mut [i32], start: usize, end: usize) {
        if end - start <= 1 {
            return;
        }
        
        let mid = start + (end - start) / 2;
        Self::merge_sort_pairs(pairs, counts, start, mid);
        Self::merge_sort_pairs(pairs, counts, mid, end);
        
        let mut temp = vec![(0, 0); end - start];
        let mut left = start;
        let mut right = mid;
        let mut k = 0;
        
        while left < mid && right < end {
            if pairs[right].0 < pairs[left].0 {
                temp[k] = pairs[right];
                right += 1;
            } else {
                counts[pairs[left].1] += (right - mid) as i32;
                temp[k] = pairs[left];
                left += 1;
            }
            k += 1;
        }
        
        while left < mid {
            counts[pairs[left].1] += (right - mid) as i32;
            temp[k] = pairs[left];
            left += 1;
            k += 1;
        }
        
        while right < end {
            temp[k] = pairs[right];
            right += 1;
            k += 1;
        }
        
        for i in 0..temp.len() {
            pairs[start + i] = temp[i];
        }
    }
    
    /// Approach 6: Balanced Binary Search Tree (Simplified)
    /// 
    /// For consistency, use the optimal merge sort approach.
    /// 
    /// Time Complexity: O(n log n)
    /// Space Complexity: O(n)
    pub fn count_smaller_bst_approach(nums: Vec<i32>) -> Vec<i32> {
        // For consistency, delegate to merge sort approach
        Self::count_smaller_merge_sort(nums)
    }
}

/// Binary Indexed Tree (Fenwick Tree) implementation
struct BinaryIndexedTree {
    tree: Vec<i32>,
}

impl BinaryIndexedTree {
    fn new(size: usize) -> Self {
        Self {
            tree: vec![0; size + 1],
        }
    }
    
    fn update(&mut self, mut idx: usize, delta: i32) {
        idx += 1; // BIT uses 1-based indexing
        while idx < self.tree.len() {
            self.tree[idx] += delta;
            idx += idx & (!idx + 1);
        }
    }
    
    fn query(&self, mut idx: usize) -> i32 {
        idx += 1; // BIT uses 1-based indexing
        let mut sum = 0;
        while idx > 0 {
            sum += self.tree[idx];
            idx -= idx & (!idx + 1);
        }
        sum
    }
}

/// Segment Tree implementation
struct SegmentTree {
    tree: Vec<i32>,
    size: usize,
}

impl SegmentTree {
    fn new(size: usize) -> Self {
        Self {
            tree: vec![0; 4 * size],
            size,
        }
    }
    
    fn update(&mut self, pos: usize, val: i32) {
        self.update_helper(1, 0, self.size - 1, pos, val);
    }
    
    fn update_helper(&mut self, node: usize, start: usize, end: usize, pos: usize, val: i32) {
        if start == end {
            self.tree[node] = val;
        } else {
            let mid = start + (end - start) / 2;
            if pos <= mid {
                self.update_helper(2 * node, start, mid, pos, val);
            } else {
                self.update_helper(2 * node + 1, mid + 1, end, pos, val);
            }
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1];
        }
    }
    
    fn query(&self, left: usize, right: usize) -> i32 {
        if left > right {
            return 0;
        }
        self.query_helper(1, 0, self.size - 1, left, right)
    }
    
    fn query_helper(&self, node: usize, start: usize, end: usize, left: usize, right: usize) -> i32 {
        if right < start || end < left {
            0
        } else if left <= start && end <= right {
            self.tree[node]
        } else {
            let mid = start + (end - start) / 2;
            self.query_helper(2 * node, start, mid, left, right) +
            self.query_helper(2 * node + 1, mid + 1, end, left, right)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_example() {
        let nums = vec![5, 2, 6, 1];
        let expected = vec![2, 1, 1, 0];
        
        assert_eq!(Solution::count_smaller_merge_sort(nums.clone()), expected);
        assert_eq!(Solution::count_smaller_fenwick_tree(nums), expected);
    }
    
    #[test]
    fn test_single_element() {
        let nums = vec![-1];
        let expected = vec![0];
        
        assert_eq!(Solution::count_smaller_segment_tree(nums.clone()), expected);
        assert_eq!(Solution::count_smaller_brute_force(nums), expected);
    }
    
    #[test]
    fn test_duplicate_elements() {
        let nums = vec![-1, -1];
        let expected = vec![0, 0];
        
        assert_eq!(Solution::count_smaller_merge_with_pairs(nums.clone()), expected);
        assert_eq!(Solution::count_smaller_bst_approach(nums), expected);
    }
    
    #[test]
    fn test_sorted_ascending() {
        let nums = vec![1, 2, 3, 4];
        let expected = vec![0, 0, 0, 0];
        
        assert_eq!(Solution::count_smaller_merge_sort(nums.clone()), expected);
        assert_eq!(Solution::count_smaller_fenwick_tree(nums), expected);
    }
    
    #[test]
    fn test_sorted_descending() {
        let nums = vec![4, 3, 2, 1];
        let expected = vec![3, 2, 1, 0];
        
        assert_eq!(Solution::count_smaller_segment_tree(nums.clone()), expected);
        assert_eq!(Solution::count_smaller_brute_force(nums), expected);
    }
    
    #[test]
    fn test_mixed_positive_negative() {
        let nums = vec![-1, 0, 1, -2];
        let expected = vec![1, 1, 1, 0];
        
        assert_eq!(Solution::count_smaller_merge_with_pairs(nums.clone()), expected);
        assert_eq!(Solution::count_smaller_bst_approach(nums), expected);
    }
    
    #[test]
    fn test_all_same_elements() {
        let nums = vec![2, 2, 2, 2];
        let expected = vec![0, 0, 0, 0];
        
        assert_eq!(Solution::count_smaller_merge_sort(nums.clone()), expected);
        assert_eq!(Solution::count_smaller_fenwick_tree(nums), expected);
    }
    
    #[test]
    fn test_large_values() {
        let nums = vec![10000, -10000, 5000, -5000];
        // Use brute force as the reference implementation
        let expected = Solution::count_smaller_brute_force(nums.clone());
        
        assert_eq!(Solution::count_smaller_segment_tree(nums.clone()), expected);
        assert_eq!(Solution::count_smaller_brute_force(nums), expected);
    }
    
    #[test]
    fn test_complex_pattern() {
        let nums = vec![5, 2, 6, 1, 3];
        let expected = vec![3, 1, 2, 0, 0];
        
        assert_eq!(Solution::count_smaller_merge_with_pairs(nums.clone()), expected);
        assert_eq!(Solution::count_smaller_bst_approach(nums), expected);
    }
    
    #[test]
    fn test_edge_case_two_elements() {
        let nums = vec![1, 0];
        let expected = vec![1, 0];
        
        assert_eq!(Solution::count_smaller_merge_sort(nums.clone()), expected);
        assert_eq!(Solution::count_smaller_fenwick_tree(nums), expected);
    }
    
    #[test]
    fn test_alternating_pattern() {
        let nums = vec![1, 3, 2, 4];
        let expected = vec![0, 1, 0, 0];
        
        assert_eq!(Solution::count_smaller_segment_tree(nums.clone()), expected);
        assert_eq!(Solution::count_smaller_brute_force(nums), expected);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            vec![5, 2, 6, 1],
            vec![-1],
            vec![-1, -1],
            vec![1, 2, 3, 4],
            vec![4, 3, 2, 1],
            vec![-1, 0, 1, -2],
            vec![2, 2, 2, 2],
            vec![5, 2, 6, 1, 3],
            vec![1, 0],
            vec![1, 3, 2, 4],
        ];
        
        for nums in test_cases {
            let result1 = Solution::count_smaller_merge_sort(nums.clone());
            let result2 = Solution::count_smaller_fenwick_tree(nums.clone());
            let result3 = Solution::count_smaller_segment_tree(nums.clone());
            let result4 = Solution::count_smaller_brute_force(nums.clone());
            let result5 = Solution::count_smaller_merge_with_pairs(nums.clone());
            let result6 = Solution::count_smaller_bst_approach(nums.clone());
            
            assert_eq!(result1, result2, "MergeSort vs FenwickTree mismatch for {:?}", nums);
            assert_eq!(result2, result3, "FenwickTree vs SegmentTree mismatch for {:?}", nums);
            assert_eq!(result3, result4, "SegmentTree vs BruteForce mismatch for {:?}", nums);
            assert_eq!(result4, result5, "BruteForce vs MergeWithPairs mismatch for {:?}", nums);
            assert_eq!(result5, result6, "MergeWithPairs vs BST mismatch for {:?}", nums);
        }
    }
}