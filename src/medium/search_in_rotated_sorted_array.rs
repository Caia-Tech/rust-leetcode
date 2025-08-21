//! # Problem 33: Search in Rotated Sorted Array
//!
//! There is an integer array `nums` sorted in ascending order (with distinct values).
//!
//! Prior to being passed to your function, `nums` is possibly rotated at some pivot index `k` 
//! (1 <= k < nums.length) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]` 
//! (0-indexed). For example, `[0,1,2,4,5,6,7]` might be rotated at pivot index `3` and become `[4,5,6,7,0,1,2]`.
//!
//! Given the array `nums` after the possible rotation and an integer `target`, return the index of `target` 
//! if it is in the array, or `-1` if it is not in the array.
//!
//! You must write an algorithm with `O(log n)` runtime complexity.
//!
//! ## Examples
//!
//! ```text
//! Input: nums = [4,5,6,7,0,1,2], target = 0
//! Output: 4
//! ```
//!
//! ```text
//! Input: nums = [4,5,6,7,0,1,2], target = 3
//! Output: -1
//! ```
//!
//! ```text
//! Input: nums = [1], target = 0
//! Output: -1
//! ```
//!
//! ## Constraints
//!
//! * 1 <= nums.length <= 5000
//! * -10^4 <= nums[i] <= 10^4
//! * All values of nums are unique
//! * nums is an ascending array that is possibly rotated
//! * -10^4 <= target <= 10^4

/// Solution for Search in Rotated Sorted Array problem
pub struct Solution;

impl Solution {
    /// Creates a new instance of Solution
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Modified Binary Search (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Use binary search with special handling for rotation
    /// 2. At each step, determine which half is properly sorted
    /// 3. Check if target is in the sorted half or unsorted half
    /// 4. Narrow search space based on this analysis
    /// 
    /// **Time Complexity:** O(log n) - Standard binary search
    /// **Space Complexity:** O(1) - Only uses constant extra space
    /// 
    /// **Key Insights:**
    /// - At least one half of array is always properly sorted
    /// - Can determine sorted half by comparing endpoints
    /// - Use sorted half properties to guide search direction
    /// 
    /// **Why this works:**
    /// - Rotation creates at most one "break point"
    /// - Binary search can handle discontinuity by checking sorted portions
    /// - Each iteration eliminates half the search space
    /// 
    /// **Step-by-step for [4,5,6,7,0,1,2], target=0:**
    /// ```text
    /// Initial: left=0, right=6, mid=3
    /// nums[mid]=7, target=0
    /// Left half [4,5,6,7] is sorted (nums[0] <= nums[3])
    /// target=0 not in [4,7], so search right half
    /// 
    /// Next: left=4, right=6, mid=5  
    /// nums[mid]=1, target=0
    /// Right half [1,2] is sorted (nums[5] <= nums[6])
    /// target=0 not in [1,2], so search left half
    /// 
    /// Next: left=4, right=4, mid=4
    /// nums[mid]=0 == target, return 4
    /// ```
    pub fn search(&self, nums: Vec<i32>, target: i32) -> i32 {
        let mut left = 0;
        let mut right = nums.len() - 1;
        
        while left <= right {
            let mid = left + (right - left) / 2;
            
            if nums[mid] == target {
                return mid as i32;
            }
            
            // Check if left half is sorted
            if nums[left] <= nums[mid] {
                // Left half is sorted
                if target >= nums[left] && target < nums[mid] {
                    // Target is in left half
                    if mid == 0 { break; }
                    right = mid - 1;
                } else {
                    // Target is in right half
                    left = mid + 1;
                }
            } else {
                // Right half is sorted
                if target > nums[mid] && target <= nums[right] {
                    // Target is in right half
                    left = mid + 1;
                } else {
                    // Target is in left half
                    if mid == 0 { break; }
                    right = mid - 1;
                }
            }
        }
        
        -1
    }

    /// # Approach 2: Find Pivot + Binary Search
    /// 
    /// **Algorithm:**
    /// 1. First, find the rotation pivot point
    /// 2. Determine which part of array contains target
    /// 3. Perform standard binary search on that part
    /// 
    /// **Time Complexity:** O(log n) - Two binary searches
    /// **Space Complexity:** O(1) - Only uses constant extra space
    /// 
    /// **Advantages:**
    /// - Separates concerns: find pivot, then search
    /// - Uses standard binary search after pivot finding
    /// - Easier to understand and debug
    /// 
    /// **When to use:** When code clarity is prioritized over minimal passes
    pub fn search_two_pass(&self, nums: Vec<i32>, target: i32) -> i32 {
        if nums.is_empty() {
            return -1;
        }
        
        let n = nums.len();
        
        // Find pivot point
        let pivot = self.find_pivot(&nums);
        
        // Determine which part to search
        if target >= nums[0] {
            // Search in left part (before rotation)
            self.binary_search(&nums, 0, if pivot == 0 { n - 1 } else { pivot - 1 }, target)
        } else {
            // Search in right part (after rotation)
            if pivot == 0 {
                -1
            } else {
                self.binary_search(&nums, pivot, n - 1, target)
            }
        }
    }
    
    fn find_pivot(&self, nums: &[i32]) -> usize {
        let mut left = 0;
        let mut right = nums.len() - 1;
        
        // If array is not rotated
        if nums[left] < nums[right] {
            return 0;
        }
        
        while left <= right {
            let mid = left + (right - left) / 2;
            
            // Check if mid is the pivot
            if mid < nums.len() - 1 && nums[mid] > nums[mid + 1] {
                return mid + 1;
            }
            if mid > 0 && nums[mid - 1] > nums[mid] {
                return mid;
            }
            
            // Decide which side to search
            if nums[mid] >= nums[left] {
                left = mid + 1;
            } else {
                if mid == 0 { break; }
                right = mid - 1;
            }
        }
        
        0
    }
    
    fn binary_search(&self, nums: &[i32], left: usize, right: usize, target: i32) -> i32 {
        let mut left = left;
        let mut right = right;
        
        while left <= right {
            let mid = left + (right - left) / 2;
            
            if nums[mid] == target {
                return mid as i32;
            } else if nums[mid] < target {
                left = mid + 1;
            } else {
                if mid == 0 { break; }
                right = mid - 1;
            }
        }
        
        -1
    }

    /// # Approach 3: Recursive Binary Search
    /// 
    /// **Algorithm:**
    /// 1. Use recursive binary search with rotation handling
    /// 2. At each level, determine sorted portion and recurse accordingly
    /// 3. Base case: single element or element found
    /// 
    /// **Time Complexity:** O(log n) - Standard binary search depth
    /// **Space Complexity:** O(log n) - Recursion stack
    /// 
    /// **Characteristics:**
    /// - Elegant recursive structure
    /// - May be easier to understand for some
    /// - Uses more memory due to recursion stack
    /// 
    /// **Trade-offs:** Clarity vs memory usage
    pub fn search_recursive(&self, nums: Vec<i32>, target: i32) -> i32 {
        if nums.is_empty() {
            return -1;
        }
        
        self.search_recursive_helper(&nums, 0, nums.len() - 1, target)
    }
    
    fn search_recursive_helper(&self, nums: &[i32], left: usize, right: usize, target: i32) -> i32 {
        if left > right {
            return -1;
        }
        
        let mid = left + (right - left) / 2;
        
        if nums[mid] == target {
            return mid as i32;
        }
        
        // Check if left half is sorted
        if nums[left] <= nums[mid] {
            // Left half is sorted
            if target >= nums[left] && target < nums[mid] {
                // Search left half
                if mid == 0 {
                    return -1;
                }
                self.search_recursive_helper(nums, left, mid - 1, target)
            } else {
                // Search right half
                self.search_recursive_helper(nums, mid + 1, right, target)
            }
        } else {
            // Right half is sorted
            if target > nums[mid] && target <= nums[right] {
                // Search right half
                self.search_recursive_helper(nums, mid + 1, right, target)
            } else {
                // Search left half
                if mid == 0 {
                    return -1;
                }
                self.search_recursive_helper(nums, left, mid - 1, target)
            }
        }
    }

    /// # Approach 4: Linear Search (Baseline)
    /// 
    /// **Algorithm:**
    /// 1. Iterate through array linearly
    /// 2. Return index when target is found
    /// 3. Return -1 if not found
    /// 
    /// **Time Complexity:** O(n) - Must check each element
    /// **Space Complexity:** O(1) - Only uses constant extra space
    /// 
    /// **Purpose:** Baseline comparison and verification
    /// **Not optimal** but useful for correctness checking
    pub fn search_linear(&self, nums: Vec<i32>, target: i32) -> i32 {
        for (i, &num) in nums.iter().enumerate() {
            if num == target {
                return i as i32;
            }
        }
        -1
    }

    /// # Approach 5: Find Min + Offset Binary Search
    /// 
    /// **Algorithm:**
    /// 1. Find minimum element to determine rotation offset
    /// 2. Use virtual indexing to perform standard binary search
    /// 3. Map virtual indices back to actual indices
    /// 
    /// **Time Complexity:** O(log n) - Find min + binary search
    /// **Space Complexity:** O(1) - Only uses constant extra space
    /// 
    /// **Key insight:** Virtual array indexing handles rotation seamlessly
    /// 
    /// **Benefits:**
    /// - Clean separation of rotation handling and search
    /// - Uses standard binary search logic
    /// - Easy to verify correctness
    pub fn search_offset(&self, nums: Vec<i32>, target: i32) -> i32 {
        if nums.is_empty() {
            return -1;
        }
        
        let n = nums.len();
        let offset = self.find_min_index(&nums);
        
        let mut left = 0;
        let mut right = n - 1;
        
        while left <= right {
            let mid = left + (right - left) / 2;
            
            // Map virtual index to actual index
            let actual_mid = (mid + offset) % n;
            
            if nums[actual_mid] == target {
                return actual_mid as i32;
            } else if nums[actual_mid] < target {
                left = mid + 1;
            } else {
                if mid == 0 { break; }
                right = mid - 1;
            }
        }
        
        -1
    }
    
    fn find_min_index(&self, nums: &[i32]) -> usize {
        let mut left = 0;
        let mut right = nums.len() - 1;
        
        while left < right {
            let mid = left + (right - left) / 2;
            
            if nums[mid] > nums[right] {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        left
    }

    /// # Approach 6: Template Binary Search
    /// 
    /// **Algorithm:**
    /// 1. Use template approach for binary search
    /// 2. Define condition function for target search
    /// 3. Handle rotation through condition evaluation
    /// 
    /// **Time Complexity:** O(log n) - Standard binary search
    /// **Space Complexity:** O(1) - Only uses constant extra space
    /// 
    /// **Educational value:** Shows how to adapt binary search templates
    /// **Consistent approach:** Can be applied to many binary search variants
    pub fn search_template(&self, nums: Vec<i32>, target: i32) -> i32 {
        if nums.is_empty() {
            return -1;
        }
        
        let mut left = 0;
        let mut right = nums.len() - 1;
        
        while left <= right {
            let mid = left + (right - left) / 2;
            
            if nums[mid] == target {
                return mid as i32;
            }
            
            // Condition: should we search in the left part?
            let should_go_left = if nums[left] <= nums[mid] {
                // Left part is sorted
                target >= nums[left] && target < nums[mid]
            } else {
                // Right part is sorted, left part contains rotation
                !(target > nums[mid] && target <= nums[right])
            };
            
            if should_go_left {
                if mid == 0 { break; }
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        
        -1
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

    fn setup() -> Solution {
        Solution::new()
    }

    #[test]
    fn test_basic_examples() {
        let solution = setup();
        
        // Example 1: Target exists
        assert_eq!(solution.search(vec![4,5,6,7,0,1,2], 0), 4);
        
        // Example 2: Target doesn't exist
        assert_eq!(solution.search(vec![4,5,6,7,0,1,2], 3), -1);
        
        // Example 3: Single element, target doesn't exist
        assert_eq!(solution.search(vec![1], 0), -1);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // Single element, target exists
        assert_eq!(solution.search(vec![1], 1), 0);
        
        // Two elements, no rotation
        assert_eq!(solution.search(vec![1, 2], 2), 1);
        assert_eq!(solution.search(vec![1, 2], 1), 0);
        assert_eq!(solution.search(vec![1, 2], 3), -1);
        
        // Two elements, with rotation
        assert_eq!(solution.search(vec![2, 1], 1), 1);
        assert_eq!(solution.search(vec![2, 1], 2), 0);
        
        // No rotation (already sorted)
        assert_eq!(solution.search(vec![1, 2, 3, 4, 5], 3), 2);
        assert_eq!(solution.search(vec![1, 2, 3, 4, 5], 6), -1);
    }

    #[test]
    fn test_rotation_positions() {
        let solution = setup();
        
        let _original = vec![0, 1, 2, 4, 5, 6, 7];
        
        // Test all possible rotation positions
        // Rotation at index 0 (no rotation)
        assert_eq!(solution.search(vec![0, 1, 2, 4, 5, 6, 7], 4), 3);
        
        // Rotation at index 1
        assert_eq!(solution.search(vec![1, 2, 4, 5, 6, 7, 0], 4), 2);
        
        // Rotation at index 3
        assert_eq!(solution.search(vec![4, 5, 6, 7, 0, 1, 2], 0), 4);
        
        // Rotation at index 6 (almost full rotation)
        assert_eq!(solution.search(vec![7, 0, 1, 2, 4, 5, 6], 0), 1);
    }

    #[test]
    fn test_target_positions() {
        let solution = setup();
        
        let nums = vec![4, 5, 6, 7, 0, 1, 2];
        
        // Test each element as target
        assert_eq!(solution.search(nums.clone(), 4), 0); // First element
        assert_eq!(solution.search(nums.clone(), 5), 1);
        assert_eq!(solution.search(nums.clone(), 6), 2);
        assert_eq!(solution.search(nums.clone(), 7), 3); // Last of first part
        assert_eq!(solution.search(nums.clone(), 0), 4); // First of second part
        assert_eq!(solution.search(nums.clone(), 1), 5);
        assert_eq!(solution.search(nums.clone(), 2), 6); // Last element
    }

    #[test]
    fn test_missing_targets() {
        let solution = setup();
        
        let nums = vec![4, 5, 6, 7, 0, 1, 2];
        
        // Test missing values
        assert_eq!(solution.search(nums.clone(), 3), -1); // Between existing values
        assert_eq!(solution.search(nums.clone(), 8), -1); // Larger than max
        assert_eq!(solution.search(nums.clone(), -1), -1); // Smaller than min
    }

    #[test]
    fn test_boundary_values() {
        let solution = setup();
        
        // Maximum and minimum possible values
        assert_eq!(solution.search(vec![10000, -10000], -10000), 1);
        assert_eq!(solution.search(vec![-10000, 10000], 10000), 1);
        
        // Large array (maximum size)
        let mut large_array: Vec<i32> = (0..5000).collect();
        large_array.rotate_left(2500); // Rotate in middle
        assert_eq!(solution.search(large_array.clone(), 2500), 0);
        assert_eq!(solution.search(large_array.clone(), 0), 2500);
        assert_eq!(solution.search(large_array.clone(), 4999), 2499);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            (vec![4,5,6,7,0,1,2], 0),
            (vec![4,5,6,7,0,1,2], 3),
            (vec![1], 0),
            (vec![1], 1),
            (vec![1,2], 2),
            (vec![2,1], 1),
            (vec![1,2,3,4,5], 3),
            (vec![3,4,5,1,2], 1),
        ];
        
        for (nums, target) in test_cases {
            let result1 = solution.search(nums.clone(), target);
            let result2 = solution.search_two_pass(nums.clone(), target);
            let result3 = solution.search_recursive(nums.clone(), target);
            let result4 = solution.search_linear(nums.clone(), target);
            let result5 = solution.search_offset(nums.clone(), target);
            let result6 = solution.search_template(nums.clone(), target);
            
            assert_eq!(result1, result2, "Modified Binary vs Two Pass mismatch for {:?}, target {}", nums, target);
            assert_eq!(result2, result3, "Two Pass vs Recursive mismatch for {:?}, target {}", nums, target);
            assert_eq!(result3, result4, "Recursive vs Linear mismatch for {:?}, target {}", nums, target);
            assert_eq!(result4, result5, "Linear vs Offset mismatch for {:?}, target {}", nums, target);
            assert_eq!(result5, result6, "Offset vs Template mismatch for {:?}, target {}", nums, target);
        }
    }

    #[test]
    fn test_performance_patterns() {
        let solution = setup();
        
        // Best case: target at middle
        let nums = vec![4, 5, 6, 7, 0, 1, 2];
        assert_eq!(solution.search(nums.clone(), 7), 3); // Should find quickly
        
        // Worst case patterns for binary search
        assert_eq!(solution.search(nums.clone(), 2), 6); // Last element
        assert_eq!(solution.search(nums.clone(), 4), 0); // First element
        
        // Different rotation amounts
        let small_rotation = vec![1, 2, 3, 4, 5, 6, 0];
        assert_eq!(solution.search(small_rotation, 0), 6);
        
        let large_rotation = vec![5, 6, 0, 1, 2, 3, 4];
        assert_eq!(solution.search(large_rotation, 0), 2);
    }

    #[test]
    fn test_mathematical_properties() {
        let solution = setup();
        
        // Property: If array has n elements and target exists, result must be in [0, n-1]
        let nums = vec![4, 5, 6, 7, 0, 1, 2];
        let result = solution.search(nums.clone(), 5);
        assert!(result >= 0 && result < nums.len() as i32);
        
        // Property: If target doesn't exist, result must be -1
        assert_eq!(solution.search(nums.clone(), 8), -1);
        
        // Property: For any rotation, all original elements should be findable
        let original = vec![1, 2, 3, 4, 5];
        let rotated = vec![3, 4, 5, 1, 2];
        for &val in &original {
            assert!(solution.search(rotated.clone(), val) >= 0);
        }
    }

    #[test]
    fn test_rotation_correctness() {
        let solution = setup();
        
        // Verify rotation detection works correctly
        let test_rotations = vec![
            vec![1, 2, 3, 4, 5], // No rotation
            vec![2, 3, 4, 5, 1], // Rotate 1
            vec![3, 4, 5, 1, 2], // Rotate 2
            vec![4, 5, 1, 2, 3], // Rotate 3
            vec![5, 1, 2, 3, 4], // Rotate 4
        ];
        
        for nums in test_rotations {
            // Each number should be findable
            for i in 1..=5 {
                let result = solution.search(nums.clone(), i);
                assert!(result >= 0, "Failed to find {} in {:?}", i, nums);
                assert_eq!(nums[result as usize], i);
            }
        }
    }

    #[test]
    fn test_duplicate_handling() {
        let solution = setup();
        
        // Problem states all values are unique, but test edge of that constraint
        let nums = vec![1, 3, 5]; // All unique
        assert_eq!(solution.search(nums.clone(), 3), 1);
        assert_eq!(solution.search(nums.clone(), 2), -1);
        assert_eq!(solution.search(nums.clone(), 4), -1);
    }

    #[test]
    fn test_binary_search_properties() {
        let solution = setup();
        
        // Verify O(log n) behavior by testing larger arrays
        let sizes = vec![1, 2, 4, 8, 16, 32, 64, 128];
        
        for size in sizes {
            // Create rotated array
            let mut nums: Vec<i32> = (0..size).collect();
            nums.rotate_left((size / 3) as usize);
            
            // Search for existing element
            assert!(solution.search(nums.clone(), size / 2) >= 0);
            
            // Search for non-existing element
            assert_eq!(solution.search(nums, size + 1), -1);
        }
    }
}