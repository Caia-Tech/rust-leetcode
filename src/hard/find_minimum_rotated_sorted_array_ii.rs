//! Problem 154: Find Minimum in Rotated Sorted Array II
//!
//! Implementation by Marvin Tutt, Caia Tech
//!
//! Suppose an array of length n sorted in ascending order is rotated between 1 and n times.
//! For example, the array nums = [0,1,4,4,5,6,7] might become:
//! - [4,5,6,7,0,1,4] if it was rotated 4 times.
//! - [0,1,4,4,5,6,7] if it was rotated 7 times.
//!
//! Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array
//! [a[n-1], a[0], a[1], a[2], ..., a[n-2]].
//!
//! Given the sorted rotated array nums that may contain duplicates, return the minimum element of this array.
//!
//! You must decrease the overall operation count as much as possible.
//!
//! The key challenge is handling duplicates which can make nums[left] == nums[mid] == nums[right],
//! making it impossible to determine which half contains the minimum. In such cases, we must
//! reduce the search space incrementally.
//!
//! Example 1:
//! Input: nums = [1,3,5]
//! Output: 1
//!
//! Example 2:
//! Input: nums = [2,2,2,0,1]
//! Output: 0
//!
//! Example 3:
//! Input: nums = [10,1,10,10,10]
//! Output: 1
//!
//! Constraints:
//! - n == nums.length
//! - 1 <= n <= 5000
//! - -5000 <= nums[i] <= 5000
//! - nums is sorted and rotated between 1 and n times.
//!
//! Follow up: This problem is similar to Find Minimum in Rotated Sorted Array, 
//! but nums may contain duplicates. Would this affect the run-time complexity? How and why?

pub struct Solution;

impl Solution {
    /// Approach 1: Binary Search with Duplicate Handling - Optimal (Average Case)
    /// 
    /// Use binary search but handle the case where nums[left] == nums[mid] == nums[right]
    /// by incrementally reducing the search space. When we can't determine which half
    /// contains the minimum due to duplicates, we eliminate one element at a time.
    /// 
    /// Time Complexity: O(log n) average case, O(n) worst case (all duplicates)
    /// Space Complexity: O(1)
    /// 
    /// Detailed Reasoning:
    /// - If nums[mid] < nums[right]: minimum is in left half (including mid)
    /// - If nums[mid] > nums[right]: minimum is in right half (excluding mid)
    /// - If nums[mid] == nums[right]: can't determine, reduce right boundary
    /// - The third case degrades to O(n) when array is mostly duplicates
    pub fn find_min_binary_search(nums: Vec<i32>) -> i32 {
        let mut left = 0;
        let mut right = nums.len() - 1;
        
        while left < right {
            let mid = left + (right - left) / 2;
            
            if nums[mid] < nums[right] {
                // Minimum is in left half (including mid)
                right = mid;
            } else if nums[mid] > nums[right] {
                // Minimum is in right half (excluding mid)
                left = mid + 1;
            } else {
                // nums[mid] == nums[right], can't determine which half
                // Reduce search space by eliminating rightmost duplicate
                right -= 1;
            }
        }
        
        nums[left]
    }
    
    /// Approach 2: Enhanced Binary Search with Left Boundary Check
    /// 
    /// Also consider the relationship with the left boundary to potentially
    /// make better decisions when encountering duplicates.
    /// 
    /// Time Complexity: O(log n) average case, O(n) worst case
    /// Space Complexity: O(1)
    /// 
    /// Detailed Reasoning:
    /// - Check both left and right boundaries against mid
    /// - When nums[left] == nums[mid] == nums[right], eliminate both ends
    /// - This can potentially reduce the search space faster in some cases
    pub fn find_min_enhanced_binary_search(nums: Vec<i32>) -> i32 {
        let mut left = 0;
        let mut right = nums.len() - 1;
        
        while left < right {
            let mid = left + (right - left) / 2;
            
            if nums[mid] < nums[right] {
                right = mid;
            } else if nums[mid] > nums[right] {
                left = mid + 1;
            } else if nums[left] == nums[mid] {
                // When left == mid == right, we can't determine direction
                // Eliminate both boundaries to reduce duplicates
                left += 1;
                right -= 1;
            } else {
                // nums[left] < nums[mid] == nums[right]
                // Minimum could be in either half, but we know it's not at right
                right -= 1;
            }
        }
        
        nums[left]
    }
    
    /// Approach 3: Linear Search (Fallback for Worst Case)
    /// 
    /// When the array is heavily duplicated, linear search might be more
    /// straightforward and has the same worst-case complexity.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    /// 
    /// Detailed Reasoning:
    /// - Simple linear scan to find the minimum
    /// - Guaranteed O(n) but no best-case optimization
    /// - Useful for comparison and as a fallback strategy
    pub fn find_min_linear_search(nums: Vec<i32>) -> i32 {
        *nums.iter().min().unwrap()
    }
    
    /// Approach 4: Three-Way Binary Search
    /// 
    /// Divide the array into three parts and recursively search the parts
    /// that could contain the minimum.
    /// 
    /// Time Complexity: O(log n) average case, O(n) worst case
    /// Space Complexity: O(log n) due to recursion
    /// 
    /// Detailed Reasoning:
    /// - Split array into three roughly equal parts
    /// - Compare the boundary values to determine which parts to search
    /// - Can potentially eliminate more elements in each iteration
    pub fn find_min_three_way_search(nums: Vec<i32>) -> i32 {
        Self::three_way_helper(&nums, 0, nums.len() - 1)
    }
    
    fn three_way_helper(nums: &[i32], left: usize, right: usize) -> i32 {
        // Base cases
        if left >= right {
            return nums[left];
        }
        
        // For very small ranges, use simpler approach
        if right - left <= 3 {
            let mut min_val = nums[left];
            for i in left + 1..=right {
                min_val = min_val.min(nums[i]);
            }
            return min_val;
        }
        
        // Three-way partitioning based on binary search principles
        let mid = left + (right - left) / 2;
        
        if nums[mid] < nums[right] {
            // Right half is sorted, minimum is in left half (including mid)
            Self::three_way_helper(nums, left, mid)
        } else if nums[mid] > nums[right] {
            // Left half contains larger elements, minimum is in right half
            Self::three_way_helper(nums, mid + 1, right)
        } else {
            // nums[mid] == nums[right], can't determine which half
            // Search both sides and take minimum
            let left_min = Self::three_way_helper(nums, left, mid);
            let right_min = Self::three_way_helper(nums, mid + 1, right);
            left_min.min(right_min)
        }
    }
    
    /// Approach 5: Modified Binary Search with Pattern Recognition
    /// 
    /// Attempt to recognize rotation patterns to make better decisions
    /// when encountering duplicates.
    /// 
    /// Time Complexity: O(log n) average case, O(n) worst case
    /// Space Complexity: O(1)
    /// 
    /// Detailed Reasoning:
    /// - Look for patterns like increasing/decreasing sequences
    /// - Use additional comparisons to break ties when possible
    /// - Still degrades to O(n) in worst case but may perform better on average
    pub fn find_min_pattern_recognition(nums: Vec<i32>) -> i32 {
        let n = nums.len();
        let mut left = 0;
        let mut right = n - 1;
        
        while left < right {
            let mid = left + (right - left) / 2;
            
            if nums[mid] < nums[right] {
                right = mid;
            } else if nums[mid] > nums[right] {
                left = mid + 1;
            } else {
                // nums[mid] == nums[right]
                // Try to find a pattern to make a better decision
                
                // Check if we can find a non-duplicate value nearby
                let mut found_pattern = false;
                
                // Look ahead from mid
                if mid + 1 < right && nums[mid + 1] != nums[mid] {
                    if nums[mid + 1] < nums[mid] {
                        left = mid + 1;
                        found_pattern = true;
                    } else if nums[mid + 1] > nums[right] {
                        left = mid + 1;
                        found_pattern = true;
                    }
                }
                
                // Look behind from mid
                if !found_pattern && mid > left && nums[mid - 1] != nums[mid] {
                    if nums[mid - 1] > nums[mid] {
                        right = mid;
                        found_pattern = true;
                    }
                }
                
                // If no pattern found, fall back to eliminating one boundary
                if !found_pattern {
                    right -= 1;
                }
            }
        }
        
        nums[left]
    }
    
    /// Approach 6: Hybrid Approach - Choose Best Strategy Based on Input
    /// 
    /// Analyze the input characteristics and choose the most appropriate algorithm.
    /// 
    /// Time Complexity: O(log n) average case, O(n) worst case
    /// Space Complexity: O(1) or O(log n) depending on chosen algorithm
    /// 
    /// Detailed Reasoning:
    /// - For small arrays: use linear search
    /// - For arrays with few duplicates: use standard binary search
    /// - For heavily duplicated arrays: use enhanced binary search
    /// - Makes the best choice based on input characteristics
    pub fn find_min_hybrid(nums: Vec<i32>) -> i32 {
        let n = nums.len();
        
        // For very small arrays, linear search is faster
        if n <= 10 {
            return Self::find_min_linear_search(nums);
        }
        
        // Count unique elements to determine strategy
        let mut unique_count = 1;
        for i in 1..n.min(20) { // Sample first 20 elements
            if nums[i] != nums[i - 1] {
                unique_count += 1;
            }
        }
        
        let duplicate_ratio = 1.0 - (unique_count as f64 / 20.0_f64.min(n as f64));
        
        // Choose strategy based on duplicate ratio
        if duplicate_ratio > 0.8 {
            // Heavily duplicated - use enhanced binary search
            Self::find_min_enhanced_binary_search(nums)
        } else if duplicate_ratio > 0.5 {
            // Moderately duplicated - use pattern recognition
            Self::find_min_pattern_recognition(nums)
        } else {
            // Lightly duplicated - use standard binary search
            Self::find_min_binary_search(nums)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_example_1() {
        let nums = vec![1, 3, 5];
        assert_eq!(Solution::find_min_binary_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_enhanced_binary_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_linear_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_three_way_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_pattern_recognition(nums.clone()), 1);
        assert_eq!(Solution::find_min_hybrid(nums), 1);
    }
    
    #[test]
    fn test_example_2() {
        let nums = vec![2, 2, 2, 0, 1];
        assert_eq!(Solution::find_min_binary_search(nums.clone()), 0);
        assert_eq!(Solution::find_min_enhanced_binary_search(nums.clone()), 0);
        assert_eq!(Solution::find_min_linear_search(nums.clone()), 0);
        assert_eq!(Solution::find_min_three_way_search(nums.clone()), 0);
        assert_eq!(Solution::find_min_pattern_recognition(nums.clone()), 0);
        assert_eq!(Solution::find_min_hybrid(nums), 0);
    }
    
    #[test]
    fn test_example_3() {
        let nums = vec![10, 1, 10, 10, 10];
        assert_eq!(Solution::find_min_binary_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_enhanced_binary_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_linear_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_three_way_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_pattern_recognition(nums.clone()), 1);
        assert_eq!(Solution::find_min_hybrid(nums), 1);
    }
    
    #[test]
    fn test_single_element() {
        let nums = vec![5];
        assert_eq!(Solution::find_min_binary_search(nums.clone()), 5);
        assert_eq!(Solution::find_min_enhanced_binary_search(nums.clone()), 5);
        assert_eq!(Solution::find_min_linear_search(nums.clone()), 5);
        assert_eq!(Solution::find_min_three_way_search(nums.clone()), 5);
        assert_eq!(Solution::find_min_pattern_recognition(nums.clone()), 5);
        assert_eq!(Solution::find_min_hybrid(nums), 5);
    }
    
    #[test]
    fn test_two_elements() {
        let nums = vec![2, 1];
        assert_eq!(Solution::find_min_binary_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_enhanced_binary_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_linear_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_three_way_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_pattern_recognition(nums.clone()), 1);
        assert_eq!(Solution::find_min_hybrid(nums), 1);
    }
    
    #[test]
    fn test_no_rotation() {
        let nums = vec![1, 2, 3, 4, 5];
        assert_eq!(Solution::find_min_binary_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_enhanced_binary_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_linear_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_three_way_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_pattern_recognition(nums.clone()), 1);
        assert_eq!(Solution::find_min_hybrid(nums), 1);
    }
    
    #[test]
    fn test_all_same() {
        let nums = vec![3, 3, 3, 3, 3];
        assert_eq!(Solution::find_min_binary_search(nums.clone()), 3);
        assert_eq!(Solution::find_min_enhanced_binary_search(nums.clone()), 3);
        assert_eq!(Solution::find_min_linear_search(nums.clone()), 3);
        assert_eq!(Solution::find_min_three_way_search(nums.clone()), 3);
        assert_eq!(Solution::find_min_pattern_recognition(nums.clone()), 3);
        assert_eq!(Solution::find_min_hybrid(nums), 3);
    }
    
    #[test]
    fn test_rotated_once() {
        let nums = vec![7, 1, 2, 3, 4, 5, 6];
        assert_eq!(Solution::find_min_binary_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_enhanced_binary_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_linear_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_three_way_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_pattern_recognition(nums.clone()), 1);
        assert_eq!(Solution::find_min_hybrid(nums), 1);
    }
    
    #[test]
    fn test_rotated_multiple_times() {
        let nums = vec![4, 5, 6, 7, 0, 1, 2];
        assert_eq!(Solution::find_min_binary_search(nums.clone()), 0);
        assert_eq!(Solution::find_min_enhanced_binary_search(nums.clone()), 0);
        assert_eq!(Solution::find_min_linear_search(nums.clone()), 0);
        assert_eq!(Solution::find_min_three_way_search(nums.clone()), 0);
        assert_eq!(Solution::find_min_pattern_recognition(nums.clone()), 0);
        assert_eq!(Solution::find_min_hybrid(nums), 0);
    }
    
    #[test]
    fn test_duplicates_at_boundaries() {
        let nums = vec![3, 1, 3, 3, 3];
        assert_eq!(Solution::find_min_binary_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_enhanced_binary_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_linear_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_three_way_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_pattern_recognition(nums.clone()), 1);
        assert_eq!(Solution::find_min_hybrid(nums), 1);
    }
    
    #[test]
    fn test_minimum_at_start() {
        let nums = vec![0, 1, 1, 2, 2, 3];
        assert_eq!(Solution::find_min_binary_search(nums.clone()), 0);
        assert_eq!(Solution::find_min_enhanced_binary_search(nums.clone()), 0);
        assert_eq!(Solution::find_min_linear_search(nums.clone()), 0);
        assert_eq!(Solution::find_min_three_way_search(nums.clone()), 0);
        assert_eq!(Solution::find_min_pattern_recognition(nums.clone()), 0);
        assert_eq!(Solution::find_min_hybrid(nums), 0);
    }
    
    #[test]
    fn test_minimum_at_end() {
        let nums = vec![2, 3, 4, 5, 1];
        assert_eq!(Solution::find_min_binary_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_enhanced_binary_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_linear_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_three_way_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_pattern_recognition(nums.clone()), 1);
        assert_eq!(Solution::find_min_hybrid(nums), 1);
    }
    
    #[test]
    fn test_negative_numbers() {
        let nums = vec![-1, -2, -3, -4, -5];
        assert_eq!(Solution::find_min_binary_search(nums.clone()), -5);
        assert_eq!(Solution::find_min_enhanced_binary_search(nums.clone()), -5);
        assert_eq!(Solution::find_min_linear_search(nums.clone()), -5);
        assert_eq!(Solution::find_min_three_way_search(nums.clone()), -5);
        assert_eq!(Solution::find_min_pattern_recognition(nums.clone()), -5);
        assert_eq!(Solution::find_min_hybrid(nums), -5);
    }
    
    #[test]
    fn test_mixed_positive_negative() {
        let nums = vec![3, 4, 5, -2, -1, 0, 1, 2];
        assert_eq!(Solution::find_min_binary_search(nums.clone()), -2);
        assert_eq!(Solution::find_min_enhanced_binary_search(nums.clone()), -2);
        assert_eq!(Solution::find_min_linear_search(nums.clone()), -2);
        assert_eq!(Solution::find_min_three_way_search(nums.clone()), -2);
        assert_eq!(Solution::find_min_pattern_recognition(nums.clone()), -2);
        assert_eq!(Solution::find_min_hybrid(nums), -2);
    }
    
    #[test]
    fn test_large_duplicated_array() {
        let nums = vec![1, 1, 1, 1, 0, 1, 1, 1, 1];
        assert_eq!(Solution::find_min_binary_search(nums.clone()), 0);
        assert_eq!(Solution::find_min_enhanced_binary_search(nums.clone()), 0);
        assert_eq!(Solution::find_min_linear_search(nums.clone()), 0);
        assert_eq!(Solution::find_min_three_way_search(nums.clone()), 0);
        assert_eq!(Solution::find_min_pattern_recognition(nums.clone()), 0);
        assert_eq!(Solution::find_min_hybrid(nums), 0);
    }
    
    #[test]
    fn test_worst_case_scenario() {
        // Array where binary search degrades to O(n)
        let nums = vec![2, 2, 2, 2, 2, 2, 2, 1, 2];
        assert_eq!(Solution::find_min_binary_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_enhanced_binary_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_linear_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_three_way_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_pattern_recognition(nums.clone()), 1);
        assert_eq!(Solution::find_min_hybrid(nums), 1);
    }
    
    #[test]
    fn test_alternating_pattern() {
        let nums = vec![1, 3, 1, 3, 1];
        assert_eq!(Solution::find_min_binary_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_enhanced_binary_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_linear_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_three_way_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_pattern_recognition(nums.clone()), 1);
        assert_eq!(Solution::find_min_hybrid(nums), 1);
    }
    
    #[test]
    fn test_boundary_values() {
        // Test with constraint boundary values - properly rotated array
        let nums = vec![5000, -5000, -4999];
        assert_eq!(Solution::find_min_binary_search(nums.clone()), -5000);
        assert_eq!(Solution::find_min_enhanced_binary_search(nums.clone()), -5000);
        assert_eq!(Solution::find_min_linear_search(nums.clone()), -5000);
        assert_eq!(Solution::find_min_three_way_search(nums.clone()), -5000);
        assert_eq!(Solution::find_min_pattern_recognition(nums.clone()), -5000);
        assert_eq!(Solution::find_min_hybrid(nums), -5000);
    }
    
    #[test]
    fn test_complex_rotation() {
        let nums = vec![10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(Solution::find_min_binary_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_enhanced_binary_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_linear_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_three_way_search(nums.clone()), 1);
        assert_eq!(Solution::find_min_pattern_recognition(nums.clone()), 1);
        assert_eq!(Solution::find_min_hybrid(nums), 1);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            vec![1, 3, 5],
            vec![2, 2, 2, 0, 1],
            vec![10, 1, 10, 10, 10],
            vec![5],
            vec![2, 1],
            vec![1, 2, 3, 4, 5],
            vec![3, 3, 3, 3, 3],
            vec![7, 1, 2, 3, 4, 5, 6],
            vec![4, 5, 6, 7, 0, 1, 2],
            vec![3, 1, 3, 3, 3],
            vec![0, 1, 1, 2, 2, 3],
            vec![2, 3, 4, 5, 1],
            vec![-1, -2, -3, -4, -5],
            vec![3, 4, 5, -2, -1, 0, 1, 2],
            vec![1, 1, 1, 1, 0, 1, 1, 1, 1],
            vec![2, 2, 2, 2, 2, 2, 2, 1, 2],
            vec![1, 3, 1, 3, 1],
            vec![5000, -5000, -4999],
        ];
        
        for nums in test_cases {
            let result1 = Solution::find_min_binary_search(nums.clone());
            let result2 = Solution::find_min_enhanced_binary_search(nums.clone());
            let result3 = Solution::find_min_linear_search(nums.clone());
            let result4 = Solution::find_min_three_way_search(nums.clone());
            let result5 = Solution::find_min_pattern_recognition(nums.clone());
            let result6 = Solution::find_min_hybrid(nums.clone());
            
            assert_eq!(result1, result2, "Binary vs Enhanced mismatch for {:?}", nums);
            assert_eq!(result2, result3, "Enhanced vs Linear mismatch for {:?}", nums);
            assert_eq!(result3, result4, "Linear vs Three-way mismatch for {:?}", nums);
            assert_eq!(result4, result5, "Three-way vs Pattern mismatch for {:?}", nums);
            assert_eq!(result5, result6, "Pattern vs Hybrid mismatch for {:?}", nums);
        }
    }
    
    #[test]
    fn test_performance_characteristics() {
        // Test cases designed to showcase different algorithm strengths
        
        // Small array - should prefer linear
        let small = vec![3, 1, 2];
        assert_eq!(Solution::find_min_hybrid(small), 1);
        
        // Lightly duplicated - should prefer binary search
        let light_dup = vec![1, 2, 3, 4, 5, 6, 0];
        assert_eq!(Solution::find_min_hybrid(light_dup), 0);
        
        // Heavily duplicated - should prefer enhanced binary search
        let heavy_dup = vec![2; 50];
        assert_eq!(Solution::find_min_hybrid(heavy_dup), 2);
        
        // Mixed case
        let mixed = vec![5, 5, 5, 1, 2, 3, 4];
        assert_eq!(Solution::find_min_hybrid(mixed), 1);
    }
}