//! Problem 84: Largest Rectangle in Histogram
//!
//! Given an array of integers heights representing the histogram's bar height where the width
//! of each bar is 1, return the area of the largest rectangle in the histogram.
//!
//! Constraints:
//! - 1 <= heights.length <= 10^5
//! - 0 <= heights[i] <= 10^4
//!
//! Example 1:
//! Input: heights = [2,1,5,6,2,3]
//! Output: 10
//! Explanation: The largest rectangle is from index 2 to 4 with height 2, area = 2 * 5 = 10
//!
//! Example 2:
//! Input: heights = [2,4]
//! Output: 4

pub struct Solution;

impl Solution {
    /// Approach 1: Stack-Based (Optimal)
    /// 
    /// Use stack to track indices of increasing heights. When we find a smaller height,
    /// pop from stack and calculate area using the popped height as the minimum.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    pub fn largest_rectangle_area_stack(heights: Vec<i32>) -> i32 {
        let mut stack = Vec::new();
        let mut max_area = 0;
        let n = heights.len();
        
        for i in 0..=n {
            // Use 0 as sentinel at the end to pop all remaining elements
            let current_height = if i == n { 0 } else { heights[i] };
            
            while let Some(&top_idx) = stack.last() {
                if heights[top_idx] <= current_height {
                    break;
                }
                
                stack.pop();
                let height = heights[top_idx];
                let width = if stack.is_empty() {
                    i
                } else {
                    i - stack.last().unwrap() - 1
                };
                
                max_area = max_area.max(height * width as i32);
            }
            
            stack.push(i);
        }
        
        max_area
    }
    
    /// Approach 2: Divide and Conquer
    /// 
    /// Find minimum height, use it to divide array into subarrays.
    /// Recursively find max area in each subarray and compare with current.
    /// 
    /// Time Complexity: O(n log n) average, O(n²) worst case
    /// Space Complexity: O(log n) recursion depth
    pub fn largest_rectangle_area_divide_conquer(heights: Vec<i32>) -> i32 {
        Self::divide_conquer_helper(&heights, 0, heights.len())
    }
    
    fn divide_conquer_helper(heights: &[i32], start: usize, end: usize) -> i32 {
        if start >= end {
            return 0;
        }
        
        if start + 1 == end {
            return heights[start];
        }
        
        // Find minimum height and its index
        let mut min_height = heights[start];
        let mut min_idx = start;
        
        for i in start..end {
            if heights[i] < min_height {
                min_height = heights[i];
                min_idx = i;
            }
        }
        
        // Area using minimum height as the height
        let area_with_min = min_height * (end - start) as i32;
        
        // Recursively find max area in left and right subarrays
        let left_area = Self::divide_conquer_helper(heights, start, min_idx);
        let right_area = Self::divide_conquer_helper(heights, min_idx + 1, end);
        
        area_with_min.max(left_area).max(right_area)
    }
    
    /// Approach 3: Brute Force with Optimization
    /// 
    /// For each position, expand left and right to find maximum width.
    /// Use the current height as the rectangle height.
    /// 
    /// Time Complexity: O(n²)
    /// Space Complexity: O(1)
    pub fn largest_rectangle_area_brute_force(heights: Vec<i32>) -> i32 {
        let mut max_area = 0;
        let n = heights.len();
        
        for i in 0..n {
            let mut min_height = heights[i];
            
            for j in i..n {
                min_height = min_height.min(heights[j]);
                let width = (j - i + 1) as i32;
                max_area = max_area.max(min_height * width);
            }
        }
        
        max_area
    }
    
    /// Approach 4: Two Pointers with Height Tracking
    /// 
    /// Use two pointers approach, similar to container with most water,
    /// but adapted for histogram problem by tracking minimum heights.
    /// 
    /// Time Complexity: O(n²)
    /// Space Complexity: O(1)
    pub fn largest_rectangle_area_two_pointers(heights: Vec<i32>) -> i32 {
        let mut max_area = 0;
        let n = heights.len();
        
        for i in 0..n {
            let mut left = i;
            let mut right = i;
            let height = heights[i];
            
            // Expand left while height is sufficient
            while left > 0 && heights[left - 1] >= height {
                left -= 1;
            }
            
            // Expand right while height is sufficient
            while right < n - 1 && heights[right + 1] >= height {
                right += 1;
            }
            
            let width = (right - left + 1) as i32;
            max_area = max_area.max(height * width);
        }
        
        max_area
    }
    
    /// Approach 5: Segment Tree
    /// 
    /// Build segment tree for range minimum queries, then use divide and conquer
    /// with efficient minimum finding.
    /// 
    /// Time Complexity: O(n log n)
    /// Space Complexity: O(n)
    pub fn largest_rectangle_area_segment_tree(heights: Vec<i32>) -> i32 {
        let n = heights.len();
        if n == 0 {
            return 0;
        }
        
        let mut tree = vec![0; 4 * n];
        Self::build_tree(&heights, &mut tree, 0, n - 1, 0);
        Self::segment_tree_helper(&heights, &tree, 0, n - 1, 0)
    }
    
    fn build_tree(heights: &[i32], tree: &mut [usize], start: usize, end: usize, node: usize) {
        if start == end {
            tree[node] = start;
        } else {
            let mid = start + (end - start) / 2;
            Self::build_tree(heights, tree, start, mid, 2 * node + 1);
            Self::build_tree(heights, tree, mid + 1, end, 2 * node + 2);
            
            let left_min = tree[2 * node + 1];
            let right_min = tree[2 * node + 2];
            
            tree[node] = if heights[left_min] <= heights[right_min] {
                left_min
            } else {
                right_min
            };
        }
    }
    
    fn query_min(heights: &[i32], tree: &[usize], start: usize, end: usize, 
                 l: usize, r: usize, node: usize) -> usize {
        if r < start || l > end {
            return usize::MAX;
        }
        
        if l <= start && end <= r {
            return tree[node];
        }
        
        let mid = start + (end - start) / 2;
        let left_min = Self::query_min(heights, tree, start, mid, l, r, 2 * node + 1);
        let right_min = Self::query_min(heights, tree, mid + 1, end, l, r, 2 * node + 2);
        
        if left_min == usize::MAX {
            right_min
        } else if right_min == usize::MAX {
            left_min
        } else if heights[left_min] <= heights[right_min] {
            left_min
        } else {
            right_min
        }
    }
    
    fn segment_tree_helper(heights: &[i32], tree: &[usize], start: usize, end: usize, node: usize) -> i32 {
        if start > end {
            return 0;
        }
        
        let min_idx = Self::query_min(heights, tree, 0, heights.len() - 1, start, end, 0);
        let area_with_min = heights[min_idx] * (end - start + 1) as i32;
        
        let left_area = if min_idx > start {
            Self::segment_tree_helper(heights, tree, start, min_idx - 1, node)
        } else {
            0
        };
        
        let right_area = if min_idx < end {
            Self::segment_tree_helper(heights, tree, min_idx + 1, end, node)
        } else {
            0
        };
        
        area_with_min.max(left_area).max(right_area)
    }
    
    /// Approach 6: Dynamic Programming with Memoization
    /// 
    /// Use memoization to cache results of subproblems in divide and conquer.
    /// 
    /// Time Complexity: O(n²)
    /// Space Complexity: O(n²)
    pub fn largest_rectangle_area_dp(heights: Vec<i32>) -> i32 {
        use std::collections::HashMap;
        let mut memo = HashMap::new();
        Self::dp_helper(&heights, 0, heights.len(), &mut memo)
    }
    
    fn dp_helper(heights: &[i32], start: usize, end: usize, 
                memo: &mut std::collections::HashMap<(usize, usize), i32>) -> i32 {
        if start >= end {
            return 0;
        }
        
        if let Some(&cached) = memo.get(&(start, end)) {
            return cached;
        }
        
        if start + 1 == end {
            let result = heights[start];
            memo.insert((start, end), result);
            return result;
        }
        
        // Find minimum height and its index
        let mut min_height = heights[start];
        let mut min_idx = start;
        
        for i in start..end {
            if heights[i] < min_height {
                min_height = heights[i];
                min_idx = i;
            }
        }
        
        // Area using minimum height as the height
        let area_with_min = min_height * (end - start) as i32;
        
        // Recursively find max area in left and right subarrays
        let left_area = Self::dp_helper(heights, start, min_idx, memo);
        let right_area = Self::dp_helper(heights, min_idx + 1, end, memo);
        
        let result = area_with_min.max(left_area).max(right_area);
        memo.insert((start, end), result);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_case() {
        assert_eq!(Solution::largest_rectangle_area_stack(vec![2, 1, 5, 6, 2, 3]), 10);
        assert_eq!(Solution::largest_rectangle_area_divide_conquer(vec![2, 1, 5, 6, 2, 3]), 10);
    }
    
    #[test]
    fn test_simple_case() {
        assert_eq!(Solution::largest_rectangle_area_brute_force(vec![2, 4]), 4);
        assert_eq!(Solution::largest_rectangle_area_two_pointers(vec![2, 4]), 4);
    }
    
    #[test]
    fn test_single_element() {
        assert_eq!(Solution::largest_rectangle_area_segment_tree(vec![5]), 5);
        assert_eq!(Solution::largest_rectangle_area_dp(vec![5]), 5);
    }
    
    #[test]
    fn test_increasing_heights() {
        assert_eq!(Solution::largest_rectangle_area_stack(vec![1, 2, 3, 4, 5]), 9);
        assert_eq!(Solution::largest_rectangle_area_divide_conquer(vec![1, 2, 3, 4, 5]), 9);
    }
    
    #[test]
    fn test_decreasing_heights() {
        assert_eq!(Solution::largest_rectangle_area_brute_force(vec![5, 4, 3, 2, 1]), 9);
        assert_eq!(Solution::largest_rectangle_area_two_pointers(vec![5, 4, 3, 2, 1]), 9);
    }
    
    #[test]
    fn test_all_same_height() {
        assert_eq!(Solution::largest_rectangle_area_segment_tree(vec![3, 3, 3, 3]), 12);
        assert_eq!(Solution::largest_rectangle_area_dp(vec![3, 3, 3, 3]), 12);
    }
    
    #[test]
    fn test_with_zeros() {
        assert_eq!(Solution::largest_rectangle_area_stack(vec![2, 0, 2]), 2);
        assert_eq!(Solution::largest_rectangle_area_divide_conquer(vec![2, 0, 2]), 2);
    }
    
    #[test]
    fn test_peak_in_middle() {
        assert_eq!(Solution::largest_rectangle_area_brute_force(vec![1, 8, 6, 2, 5, 4, 8, 3, 7]), 16);
        assert_eq!(Solution::largest_rectangle_area_two_pointers(vec![1, 8, 6, 2, 5, 4, 8, 3, 7]), 16);
    }
    
    #[test]
    fn test_alternating_pattern() {
        assert_eq!(Solution::largest_rectangle_area_segment_tree(vec![1, 3, 1, 3, 1]), 5);
        assert_eq!(Solution::largest_rectangle_area_dp(vec![1, 3, 1, 3, 1]), 5);
    }
    
    #[test]
    fn test_large_rectangle() {
        assert_eq!(Solution::largest_rectangle_area_stack(vec![6, 7, 5, 2, 4, 5, 9, 3]), 16);
        assert_eq!(Solution::largest_rectangle_area_divide_conquer(vec![6, 7, 5, 2, 4, 5, 9, 3]), 16);
    }
    
    #[test]
    fn test_mountain_shape() {
        assert_eq!(Solution::largest_rectangle_area_brute_force(vec![1, 2, 3, 4, 3, 2, 1]), 10);
        assert_eq!(Solution::largest_rectangle_area_two_pointers(vec![1, 2, 3, 4, 3, 2, 1]), 10);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            vec![2, 1, 5, 6, 2, 3],
            vec![2, 4],
            vec![5],
            vec![1, 2, 3, 4, 5],
            vec![5, 4, 3, 2, 1],
            vec![3, 3, 3, 3],
            vec![2, 0, 2],
            vec![1, 8, 6, 2, 5, 4, 8, 3, 7],
            vec![1, 3, 1, 3, 1],
            vec![6, 7, 5, 2, 4, 5, 9, 3],
            vec![1, 2, 3, 4, 3, 2, 1],
            vec![0, 9],
            vec![9, 0],
            vec![4, 2, 0, 3, 2, 5],
        ];
        
        for heights in test_cases {
            let result1 = Solution::largest_rectangle_area_stack(heights.clone());
            let result2 = Solution::largest_rectangle_area_divide_conquer(heights.clone());
            let result3 = Solution::largest_rectangle_area_brute_force(heights.clone());
            let result4 = Solution::largest_rectangle_area_two_pointers(heights.clone());
            let result5 = Solution::largest_rectangle_area_segment_tree(heights.clone());
            let result6 = Solution::largest_rectangle_area_dp(heights.clone());
            
            assert_eq!(result1, result2, "Stack vs Divide&Conquer mismatch for {:?}: {} vs {}", heights, result1, result2);
            assert_eq!(result2, result3, "Divide&Conquer vs BruteForce mismatch for {:?}: {} vs {}", heights, result2, result3);
            assert_eq!(result3, result4, "BruteForce vs TwoPointers mismatch for {:?}: {} vs {}", heights, result3, result4);
            assert_eq!(result4, result5, "TwoPointers vs SegmentTree mismatch for {:?}: {} vs {}", heights, result4, result5);
            assert_eq!(result5, result6, "SegmentTree vs DP mismatch for {:?}: {} vs {}", heights, result5, result6);
        }
    }
}