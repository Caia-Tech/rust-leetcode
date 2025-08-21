//! Problem 300: Longest Increasing Subsequence
//! 
//! Given an integer array nums, return the length of the longest strictly increasing subsequence.
//! 
//! Example 1:
//! Input: nums = [10,9,2,5,3,7,101,18]
//! Output: 4
//! Explanation: The longest increasing subsequence is [2,3,7,18], therefore the length is 4.
//! 
//! Example 2:
//! Input: nums = [0,1,0,3,2,3]
//! Output: 4
//! 
//! Example 3:
//! Input: nums = [7,7,7,7,7,7,7]
//! Output: 1

use std::collections::HashMap;

pub struct Solution;

impl Solution {
    /// Approach 1: Dynamic Programming (Classic)
    /// 
    /// For each position i, dp[i] represents the length of the longest increasing subsequence
    /// ending at index i. For each i, check all previous elements j < i, and if nums[j] < nums[i],
    /// then dp[i] = max(dp[i], dp[j] + 1).
    /// 
    /// Time Complexity: O(n²)
    /// Space Complexity: O(n)
    pub fn length_of_lis_dp(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let n = nums.len();
        let mut dp = vec![1; n]; // dp[i] = length of LIS ending at i
        
        for i in 1..n {
            for j in 0..i {
                if nums[j] < nums[i] {
                    dp[i] = dp[i].max(dp[j] + 1);
                }
            }
        }
        
        *dp.iter().max().unwrap()
    }
    
    /// Approach 2: Binary Search with Patience Sorting
    /// 
    /// Maintains an array `tails` where tails[i] is the smallest ending element
    /// of all increasing subsequences of length i+1. For each element, use binary
    /// search to find the position to replace or extend.
    /// 
    /// Time Complexity: O(n log n)
    /// Space Complexity: O(n)
    pub fn length_of_lis_binary_search(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let mut tails = Vec::new();
        
        for num in nums {
            // Binary search for the position to insert/replace
            let pos = tails.binary_search(&num).unwrap_or_else(|x| x);
            
            if pos == tails.len() {
                tails.push(num);
            } else {
                tails[pos] = num;
            }
        }
        
        tails.len() as i32
    }
    
    /// Approach 3: Recursive with Memoization
    /// 
    /// Uses top-down dynamic programming approach. For each position,
    /// recursively find the maximum LIS starting from that position.
    /// 
    /// Time Complexity: O(n²)
    /// Space Complexity: O(n²) for memoization
    pub fn length_of_lis_memo(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let mut memo = HashMap::new();
        let mut max_len = 1;
        
        for i in 0..nums.len() {
            max_len = max_len.max(self.lis_from_index(&nums, i, &mut memo));
        }
        
        max_len
    }
    
    fn lis_from_index(&self, nums: &[i32], start: usize, memo: &mut HashMap<usize, i32>) -> i32 {
        if let Some(&cached) = memo.get(&start) {
            return cached;
        }
        
        let mut max_len = 1;
        
        for i in start + 1..nums.len() {
            if nums[i] > nums[start] {
                max_len = max_len.max(1 + self.lis_from_index(nums, i, memo));
            }
        }
        
        memo.insert(start, max_len);
        max_len
    }
    
    /// Approach 4: Segment Tree Based
    /// 
    /// Uses a segment tree to efficiently query and update the maximum LIS length
    /// for ranges of values. This approach handles coordinate compression.
    /// 
    /// Time Complexity: O(n log n)
    /// Space Complexity: O(n)
    pub fn length_of_lis_segment_tree(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        // Coordinate compression
        let mut sorted_nums = nums.clone();
        sorted_nums.sort_unstable();
        sorted_nums.dedup();
        
        let coord_map: HashMap<i32, usize> = sorted_nums
            .iter()
            .enumerate()
            .map(|(i, &x)| (x, i))
            .collect();
        
        let mut seg_tree = SegmentTree::new(sorted_nums.len());
        let mut result = 0;
        
        for num in nums {
            let idx = coord_map[&num];
            // Query maximum LIS length for all values < num
            let max_prev = if idx > 0 { seg_tree.query(0, idx - 1) } else { 0 };
            let current_len = max_prev + 1;
            
            // Update the segment tree
            seg_tree.update(idx, current_len);
            result = result.max(current_len);
        }
        
        result
    }
    
    /// Approach 5: Fenwick Tree (Binary Indexed Tree)
    /// 
    /// Similar to segment tree but using a Fenwick tree for range maximum queries.
    /// More space-efficient than segment tree.
    /// 
    /// Time Complexity: O(n log n)
    /// Space Complexity: O(n)
    pub fn length_of_lis_fenwick(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        // Coordinate compression
        let mut sorted_nums = nums.clone();
        sorted_nums.sort_unstable();
        sorted_nums.dedup();
        
        let coord_map: HashMap<i32, usize> = sorted_nums
            .iter()
            .enumerate()
            .map(|(i, &x)| (x, i + 1)) // 1-indexed for Fenwick tree
            .collect();
        
        let mut fenwick = FenwickTree::new(sorted_nums.len());
        let mut result = 0;
        
        for num in nums {
            let idx = coord_map[&num];
            // Query maximum LIS length for all values < num
            let max_prev = if idx > 1 { fenwick.query(idx - 1) } else { 0 };
            let current_len = max_prev + 1;
            
            // Update the Fenwick tree
            fenwick.update(idx, current_len);
            result = result.max(current_len);
        }
        
        result
    }
    
    /// Approach 6: Stack-based with Binary Search
    /// 
    /// Maintains a stack of increasing elements and uses binary search
    /// to efficiently maintain the "patience sorting" invariant.
    /// 
    /// Time Complexity: O(n log n)
    /// Space Complexity: O(n)
    pub fn length_of_lis_stack(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let mut stacks = Vec::new();
        
        for num in nums {
            // Find the leftmost stack where we can place this number
            let mut left = 0;
            let mut right = stacks.len();
            
            while left < right {
                let mid = left + (right - left) / 2;
                if stacks[mid] < num {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            
            // If we need a new stack
            if left == stacks.len() {
                stacks.push(num);
            } else {
                stacks[left] = num; // Replace with smaller number
            }
        }
        
        stacks.len() as i32
    }
}

struct SegmentTree {
    tree: Vec<i32>,
    n: usize,
}

impl SegmentTree {
    fn new(size: usize) -> Self {
        Self {
            tree: vec![0; 4 * size],
            n: size,
        }
    }
    
    fn update(&mut self, idx: usize, val: i32) {
        self.update_helper(0, 0, self.n - 1, idx, val);
    }
    
    fn update_helper(&mut self, node: usize, start: usize, end: usize, idx: usize, val: i32) {
        if start == end {
            self.tree[node] = val;
        } else {
            let mid = (start + end) / 2;
            if idx <= mid {
                self.update_helper(2 * node + 1, start, mid, idx, val);
            } else {
                self.update_helper(2 * node + 2, mid + 1, end, idx, val);
            }
            self.tree[node] = self.tree[2 * node + 1].max(self.tree[2 * node + 2]);
        }
    }
    
    fn query(&self, left: usize, right: usize) -> i32 {
        if left > right || right >= self.n {
            return 0;
        }
        self.query_helper(0, 0, self.n - 1, left, right)
    }
    
    fn query_helper(&self, node: usize, start: usize, end: usize, left: usize, right: usize) -> i32 {
        if right < start || end < left {
            return 0;
        }
        if left <= start && end <= right {
            return self.tree[node];
        }
        
        let mid = (start + end) / 2;
        let left_max = self.query_helper(2 * node + 1, start, mid, left, right);
        let right_max = self.query_helper(2 * node + 2, mid + 1, end, left, right);
        left_max.max(right_max)
    }
}

struct FenwickTree {
    tree: Vec<i32>,
}

impl FenwickTree {
    fn new(size: usize) -> Self {
        Self {
            tree: vec![0; size + 1],
        }
    }
    
    fn update(&mut self, mut idx: usize, val: i32) {
        while idx < self.tree.len() {
            self.tree[idx] = self.tree[idx].max(val);
            idx += idx & (!idx + 1); // Add lowest set bit
        }
    }
    
    fn query(&self, mut idx: usize) -> i32 {
        let mut result = 0;
        while idx > 0 {
            result = result.max(self.tree[idx]);
            idx -= idx & (!idx + 1); // Remove lowest set bit
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dp() {
        let solution = Solution;
        
        assert_eq!(solution.length_of_lis_dp(vec![10,9,2,5,3,7,101,18]), 4);
        assert_eq!(solution.length_of_lis_dp(vec![0,1,0,3,2,3]), 4);
        assert_eq!(solution.length_of_lis_dp(vec![7,7,7,7,7,7,7]), 1);
        assert_eq!(solution.length_of_lis_dp(vec![]), 0);
        assert_eq!(solution.length_of_lis_dp(vec![1]), 1);
        assert_eq!(solution.length_of_lis_dp(vec![1,3,6,7,9,4,10,5,6]), 6);
    }
    
    #[test]
    fn test_binary_search() {
        let solution = Solution;
        
        assert_eq!(solution.length_of_lis_binary_search(vec![10,9,2,5,3,7,101,18]), 4);
        assert_eq!(solution.length_of_lis_binary_search(vec![0,1,0,3,2,3]), 4);
        assert_eq!(solution.length_of_lis_binary_search(vec![7,7,7,7,7,7,7]), 1);
        assert_eq!(solution.length_of_lis_binary_search(vec![]), 0);
        assert_eq!(solution.length_of_lis_binary_search(vec![1]), 1);
        assert_eq!(solution.length_of_lis_binary_search(vec![1,3,6,7,9,4,10,5,6]), 6);
    }
    
    #[test]
    fn test_memo() {
        let solution = Solution;
        
        assert_eq!(solution.length_of_lis_memo(vec![10,9,2,5,3,7,101,18]), 4);
        assert_eq!(solution.length_of_lis_memo(vec![0,1,0,3,2,3]), 4);
        assert_eq!(solution.length_of_lis_memo(vec![7,7,7,7,7,7,7]), 1);
        assert_eq!(solution.length_of_lis_memo(vec![]), 0);
        assert_eq!(solution.length_of_lis_memo(vec![1]), 1);
        assert_eq!(solution.length_of_lis_memo(vec![1,3,6,7,9,4,10,5,6]), 6);
    }
    
    #[test]
    fn test_segment_tree() {
        let solution = Solution;
        
        assert_eq!(solution.length_of_lis_segment_tree(vec![10,9,2,5,3,7,101,18]), 4);
        assert_eq!(solution.length_of_lis_segment_tree(vec![0,1,0,3,2,3]), 4);
        assert_eq!(solution.length_of_lis_segment_tree(vec![7,7,7,7,7,7,7]), 1);
        assert_eq!(solution.length_of_lis_segment_tree(vec![]), 0);
        assert_eq!(solution.length_of_lis_segment_tree(vec![1]), 1);
        assert_eq!(solution.length_of_lis_segment_tree(vec![1,3,6,7,9,4,10,5,6]), 6);
    }
    
    #[test]
    fn test_fenwick() {
        let solution = Solution;
        
        assert_eq!(solution.length_of_lis_fenwick(vec![10,9,2,5,3,7,101,18]), 4);
        assert_eq!(solution.length_of_lis_fenwick(vec![0,1,0,3,2,3]), 4);
        assert_eq!(solution.length_of_lis_fenwick(vec![7,7,7,7,7,7,7]), 1);
        assert_eq!(solution.length_of_lis_fenwick(vec![]), 0);
        assert_eq!(solution.length_of_lis_fenwick(vec![1]), 1);
        assert_eq!(solution.length_of_lis_fenwick(vec![1,3,6,7,9,4,10,5,6]), 6);
    }
    
    #[test]
    fn test_stack() {
        let solution = Solution;
        
        assert_eq!(solution.length_of_lis_stack(vec![10,9,2,5,3,7,101,18]), 4);
        assert_eq!(solution.length_of_lis_stack(vec![0,1,0,3,2,3]), 4);
        assert_eq!(solution.length_of_lis_stack(vec![7,7,7,7,7,7,7]), 1);
        assert_eq!(solution.length_of_lis_stack(vec![]), 0);
        assert_eq!(solution.length_of_lis_stack(vec![1]), 1);
        assert_eq!(solution.length_of_lis_stack(vec![1,3,6,7,9,4,10,5,6]), 6);
    }
    
    #[test]
    fn test_edge_cases() {
        let solution = Solution;
        
        // Strictly decreasing
        assert_eq!(solution.length_of_lis_dp(vec![5,4,3,2,1]), 1);
        assert_eq!(solution.length_of_lis_binary_search(vec![5,4,3,2,1]), 1);
        
        // Already sorted
        assert_eq!(solution.length_of_lis_dp(vec![1,2,3,4,5]), 5);
        assert_eq!(solution.length_of_lis_binary_search(vec![1,2,3,4,5]), 5);
        
        // Two elements
        assert_eq!(solution.length_of_lis_dp(vec![1,2]), 2);
        assert_eq!(solution.length_of_lis_dp(vec![2,1]), 1);
        
        // Large values
        assert_eq!(solution.length_of_lis_binary_search(vec![1000,999,998,1001,1002]), 3);
        
        // Negative numbers
        assert_eq!(solution.length_of_lis_binary_search(vec![-5,-1,-3,0,2]), 4);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let solution = Solution;
        
        let test_cases = vec![
            vec![10,9,2,5,3,7,101,18],
            vec![0,1,0,3,2,3],
            vec![7,7,7,7,7,7,7],
            vec![],
            vec![1],
            vec![1,3,6,7,9,4,10,5,6],
            vec![5,4,3,2,1],
            vec![1,2,3,4,5],
            vec![2,1],
            vec![1000,999,998,1001,1002],
            vec![-5,-1,-3,0,2],
        ];
        
        for nums in test_cases {
            let dp = solution.length_of_lis_dp(nums.clone());
            let binary_search = solution.length_of_lis_binary_search(nums.clone());
            let memo = solution.length_of_lis_memo(nums.clone());
            let segment_tree = solution.length_of_lis_segment_tree(nums.clone());
            let fenwick = solution.length_of_lis_fenwick(nums.clone());
            let stack = solution.length_of_lis_stack(nums.clone());
            
            assert_eq!(dp, binary_search, "DP and binary search differ for {:?}", nums);
            assert_eq!(dp, memo, "DP and memo differ for {:?}", nums);
            assert_eq!(dp, segment_tree, "DP and segment tree differ for {:?}", nums);
            assert_eq!(dp, fenwick, "DP and fenwick differ for {:?}", nums);
            assert_eq!(dp, stack, "DP and stack differ for {:?}", nums);
        }
    }
    
    #[test]
    fn test_performance_comparison() {
        let solution = Solution;
        
        // Test with a larger array to verify O(n log n) vs O(n²) performance
        let nums: Vec<i32> = (0..100).rev().collect(); // Decreasing sequence
        
        let dp_result = solution.length_of_lis_dp(nums.clone());
        let binary_search_result = solution.length_of_lis_binary_search(nums.clone());
        
        assert_eq!(dp_result, 1);
        assert_eq!(binary_search_result, 1);
        
        // Test with increasing sequence
        let nums: Vec<i32> = (0..100).collect();
        
        let dp_result = solution.length_of_lis_dp(nums.clone());
        let binary_search_result = solution.length_of_lis_binary_search(nums);
        
        assert_eq!(dp_result, 100);
        assert_eq!(binary_search_result, 100);
    }
    
    #[test]
    fn test_complex_patterns() {
        let solution = Solution;
        
        // Alternating pattern
        let nums = vec![1, 3, 2, 4, 3, 5, 4, 6];
        assert_eq!(solution.length_of_lis_binary_search(nums), 5); // [1,2,3,4,6] or similar
        
        // Multiple local maxima
        let nums = vec![1, 5, 2, 6, 3, 7, 4, 8];
        assert_eq!(solution.length_of_lis_binary_search(nums), 5); // [1,2,3,4,8]
        
        // Plateau pattern
        let nums = vec![1, 2, 2, 3, 3, 4, 4, 5];
        assert_eq!(solution.length_of_lis_binary_search(nums), 5); // [1,2,3,4,5]
    }
}