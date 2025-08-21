//! # Problem 153: Find Minimum in Rotated Sorted Array
//!
//! Suppose an array of length `n` sorted in ascending order is rotated between `1` and `n` times.
//! For example, the array `nums = [0,1,2,4,5,6,7]` might become:
//!
//! * `[4,5,6,7,0,1,2]` if it was rotated 4 times.
//! * `[0,1,2,4,5,6,7]` if it was rotated 7 times.
//!
//! Notice that rotating an array `[a[0], a[1], a[2], ..., a[n-1]]` 1 time results in the array `[a[n-1], a[0], a[1], a[2], ..., a[n-2]]`.
//!
//! Given the sorted rotated array `nums` of unique elements, return the minimum element of this array.
//!
//! You must write an algorithm that runs in `O(log n)` time.
//!
//! ## Examples
//!
//! ```text
//! Input: nums = [3,4,5,1,2]
//! Output: 1
//! Explanation: The original array was [1,2,3,4,5] rotated 3 times.
//! ```
//!
//! ```text
//! Input: nums = [4,5,6,7,0,1,2]
//! Output: 0
//! Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.
//! ```
//!
//! ```text
//! Input: nums = [11,13,15,17]
//! Output: 11
//! Explanation: The original array was [11,13,15,17] and it was rotated 4 times (or 0 times).
//! ```
//!
//! ## Constraints
//!
//! * n == nums.length
//! * 1 <= n <= 5000
//! * -5000 <= nums[i] <= 5000
//! * All the integers of nums are unique
//! * nums is sorted and rotated between 1 and n times

/// Solution for Find Minimum in Rotated Sorted Array problem
pub struct Solution;

impl Solution {
    /// Creates a new instance of Solution
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Binary Search (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Use binary search to find the rotation point
    /// 2. Compare middle element with rightmost element
    /// 3. If mid > right, minimum is in right half
    /// 4. If mid < right, minimum is in left half (including mid)
    /// 5. Continue until search space is narrowed to one element
    /// 
    /// **Time Complexity:** O(log n) - Binary search halves space each time
    /// **Space Complexity:** O(1) - Only uses constant extra space
    /// 
    /// **Key Insights:**
    /// - Rotation creates exactly one "break point" where order decreases
    /// - Minimum element is always at the rotation point
    /// - Comparing with rightmost element disambiguates search direction
    /// 
    /// **Why this works:**
    /// - If nums[mid] > nums[right], the array looks like [large..., mid, ..., small, right]
    ///   so minimum must be in right half
    /// - If nums[mid] < nums[right], the array looks like [small, ..., mid, ..., large, right]
    ///   so minimum could be mid or in left half
    /// 
    /// **Step-by-step for [4,5,6,7,0,1,2]:**
    /// ```text
    /// Initial: left=0, right=6, mid=3
    /// nums[mid]=7, nums[right]=2, 7 > 2
    /// Minimum in right half, left = mid + 1 = 4
    /// 
    /// Next: left=4, right=6, mid=5  
    /// nums[mid]=1, nums[right]=2, 1 < 2
    /// Minimum in left half (including mid), right = mid = 5
    /// 
    /// Next: left=4, right=5, mid=4
    /// nums[mid]=0, nums[right]=1, 0 < 1
    /// Minimum in left half (including mid), right = mid = 4
    /// 
    /// Final: left=4, right=4, return nums[4]=0
    /// ```
    pub fn find_min(&self, nums: Vec<i32>) -> i32 {
        let mut left = 0;
        let mut right = nums.len() - 1;
        
        while left < right {
            let mid = left + (right - left) / 2;
            
            if nums[mid] > nums[right] {
                // Minimum is in right half
                left = mid + 1;
            } else {
                // Minimum is in left half (including mid)
                right = mid;
            }
        }
        
        nums[left]
    }

    /// # Approach 2: Find Pivot Point
    /// 
    /// **Algorithm:**
    /// 1. Find the exact pivot point where rotation occurred
    /// 2. Pivot is where nums[i] > nums[i+1]
    /// 3. Return nums[pivot+1] as minimum
    /// 4. Handle case where array is not rotated
    /// 
    /// **Time Complexity:** O(log n) - Binary search for pivot
    /// **Space Complexity:** O(1) - Only uses constant extra space
    /// 
    /// **Characteristics:**
    /// - More explicit about finding rotation point
    /// - Separates pivot finding from minimum extraction
    /// - Easier to understand conceptually
    /// 
    /// **When to use:** When you need to know the exact rotation index
    pub fn find_min_pivot(&self, nums: Vec<i32>) -> i32 {
        if nums.len() == 1 {
            return nums[0];
        }
        
        // Check if array is rotated
        if nums[0] < nums[nums.len() - 1] {
            return nums[0];
        }
        
        let mut left = 0;
        let mut right = nums.len() - 1;
        
        while left <= right {
            let mid = left + (right - left) / 2;
            
            // Check if mid is the pivot
            if mid < nums.len() - 1 && nums[mid] > nums[mid + 1] {
                return nums[mid + 1];
            }
            if mid > 0 && nums[mid - 1] > nums[mid] {
                return nums[mid];
            }
            
            // Decide which side to search
            if nums[mid] > nums[0] {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        nums[0]
    }

    /// # Approach 3: Recursive Binary Search
    /// 
    /// **Algorithm:**
    /// 1. Use recursive approach for binary search
    /// 2. Base case: single element
    /// 3. Recursive case: search appropriate half
    /// 
    /// **Time Complexity:** O(log n) - Same as iterative but with recursion
    /// **Space Complexity:** O(log n) - Recursion stack
    /// 
    /// **Trade-offs:**
    /// - More elegant recursive structure
    /// - Uses additional stack space
    /// - May be clearer for some developers
    pub fn find_min_recursive(&self, nums: Vec<i32>) -> i32 {
        self.find_min_helper(&nums, 0, nums.len() - 1)
    }
    
    fn find_min_helper(&self, nums: &[i32], left: usize, right: usize) -> i32 {
        if left == right {
            return nums[left];
        }
        
        let mid = left + (right - left) / 2;
        
        if nums[mid] > nums[right] {
            // Minimum is in right half
            self.find_min_helper(nums, mid + 1, right)
        } else {
            // Minimum is in left half (including mid)
            self.find_min_helper(nums, left, mid)
        }
    }

    /// # Approach 4: Linear Search (Baseline)
    /// 
    /// **Algorithm:**
    /// 1. Iterate through array to find minimum
    /// 2. Keep track of smallest element seen
    /// 3. Return minimum after full scan
    /// 
    /// **Time Complexity:** O(n) - Must check every element
    /// **Space Complexity:** O(1) - Only uses constant extra space
    /// 
    /// **Purpose:** Baseline for correctness verification
    /// **Not optimal** but useful for testing
    pub fn find_min_linear(&self, nums: Vec<i32>) -> i32 {
        *nums.iter().min().unwrap()
    }

    /// # Approach 5: Modified Binary Search with Early Termination
    /// 
    /// **Algorithm:**
    /// 1. Check if array is already sorted (not rotated)
    /// 2. Use binary search with early termination conditions
    /// 3. Handle special cases for small arrays
    /// 
    /// **Time Complexity:** O(log n) - Binary search with optimizations
    /// **Space Complexity:** O(1) - Only uses constant extra space
    /// 
    /// **Optimizations:**
    /// - Early return for non-rotated arrays
    /// - Special handling for small arrays
    /// - Reduced number of iterations in best cases
    pub fn find_min_optimized(&self, nums: Vec<i32>) -> i32 {
        let n = nums.len();
        
        // Handle edge cases
        if n == 1 {
            return nums[0];
        }
        if n == 2 {
            return nums[0].min(nums[1]);
        }
        
        // Check if array is not rotated
        if nums[0] < nums[n - 1] {
            return nums[0];
        }
        
        let mut left = 0;
        let mut right = n - 1;
        
        while left < right {
            // Early termination: if search space is small enough
            if right - left <= 2 {
                return nums[left].min(nums[right]).min(nums[left + 1]);
            }
            
            let mid = left + (right - left) / 2;
            
            if nums[mid] > nums[right] {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        nums[left]
    }

    /// # Approach 6: Ternary Search
    /// 
    /// **Algorithm:**
    /// 1. Divide search space into three parts instead of two
    /// 2. Use two midpoints: mid1 and mid2
    /// 3. Eliminate one-third of search space each iteration
    /// 4. Continue until minimum is found
    /// 
    /// **Time Complexity:** O(log₃ n) - Ternary search
    /// **Space Complexity:** O(1) - Only uses constant extra space
    /// 
    /// **Characteristics:**
    /// - Theoretically faster convergence than binary search
    /// - More complex logic and comparisons per iteration
    /// - Often not practically faster due to overhead
    /// 
    /// **Educational value:** Shows alternative divide-and-conquer approach
    pub fn find_min_ternary(&self, nums: Vec<i32>) -> i32 {
        let mut left = 0;
        let mut right = nums.len() - 1;
        
        while left < right {
            if right - left <= 2 {
                // Handle small search space directly
                let mut min_val = nums[left];
                for i in left..=right {
                    min_val = min_val.min(nums[i]);
                }
                return min_val;
            }
            
            let mid1 = left + (right - left) / 3;
            let mid2 = right - (right - left) / 3;
            
            // Determine which third contains the minimum
            if nums[mid1] > nums[right] {
                // Minimum is in right two-thirds
                left = mid1 + 1;
            } else if nums[mid2] > nums[right] {
                // Minimum is in middle third or right third
                left = mid1;
                right = mid2;
            } else {
                // Minimum is in left two-thirds
                right = mid2;
            }
        }
        
        nums[left]
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
        
        // Example 1: Rotated array
        assert_eq!(solution.find_min(vec![3,4,5,1,2]), 1);
        
        // Example 2: Another rotation
        assert_eq!(solution.find_min(vec![4,5,6,7,0,1,2]), 0);
        
        // Example 3: No rotation (or full rotation)
        assert_eq!(solution.find_min(vec![11,13,15,17]), 11);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // Single element
        assert_eq!(solution.find_min(vec![1]), 1);
        
        // Two elements - no rotation
        assert_eq!(solution.find_min(vec![1, 2]), 1);
        
        // Two elements - with rotation
        assert_eq!(solution.find_min(vec![2, 1]), 1);
        
        // Three elements - various rotations
        assert_eq!(solution.find_min(vec![1, 2, 3]), 1); // No rotation
        assert_eq!(solution.find_min(vec![2, 3, 1]), 1); // Rotate 1
        assert_eq!(solution.find_min(vec![3, 1, 2]), 1); // Rotate 2
    }

    #[test]
    fn test_rotation_positions() {
        let solution = setup();
        
        // Test all possible rotations of [1,2,3,4,5]
        assert_eq!(solution.find_min(vec![1, 2, 3, 4, 5]), 1); // No rotation
        assert_eq!(solution.find_min(vec![2, 3, 4, 5, 1]), 1); // Rotate 1
        assert_eq!(solution.find_min(vec![3, 4, 5, 1, 2]), 1); // Rotate 2
        assert_eq!(solution.find_min(vec![4, 5, 1, 2, 3]), 1); // Rotate 3
        assert_eq!(solution.find_min(vec![5, 1, 2, 3, 4]), 1); // Rotate 4
    }

    #[test]
    fn test_negative_numbers() {
        let solution = setup();
        
        // Array with negative numbers
        assert_eq!(solution.find_min(vec![-1, 0, 3, 5, 9, 12]), -1);
        assert_eq!(solution.find_min(vec![3, 5, 9, 12, -1, 0]), -1);
        assert_eq!(solution.find_min(vec![0, 3, 5, 9, 12, -1]), -1);
        
        // All negative numbers
        assert_eq!(solution.find_min(vec![-5, -3, -1]), -5);
        assert_eq!(solution.find_min(vec![-3, -1, -5]), -5);
        assert_eq!(solution.find_min(vec![-1, -5, -3]), -5);
    }

    #[test]
    fn test_large_arrays() {
        let solution = setup();
        
        // Larger arrays with different rotation points
        let mut arr1: Vec<i32> = (1..=10).collect();
        assert_eq!(solution.find_min(arr1.clone()), 1);
        
        // Rotate by 3
        arr1.rotate_left(3);
        assert_eq!(solution.find_min(arr1), 1);
        
        // Rotate by 7
        let mut arr2: Vec<i32> = (1..=10).collect();
        arr2.rotate_left(7);
        assert_eq!(solution.find_min(arr2), 1);
    }

    #[test]
    fn test_boundary_values() {
        let solution = setup();
        
        // Maximum and minimum constraints
        assert_eq!(solution.find_min(vec![5000, -5000]), -5000);
        assert_eq!(solution.find_min(vec![-5000, 5000]), -5000);
        
        // Large array (maximum size)
        let mut large_array: Vec<i32> = (-2500..2500).collect();
        large_array.rotate_left(1000);
        assert_eq!(solution.find_min(large_array), -2500);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            vec![3,4,5,1,2],
            vec![4,5,6,7,0,1,2],
            vec![11,13,15,17],
            vec![1],
            vec![2,1],
            vec![1,2,3],
            vec![2,3,1],
            vec![3,1,2],
            vec![-1,0,3,5,9,12],
            vec![3,5,9,12,-1,0],
        ];
        
        for nums in test_cases {
            let result1 = solution.find_min(nums.clone());
            let result2 = solution.find_min_pivot(nums.clone());
            let result3 = solution.find_min_recursive(nums.clone());
            let result4 = solution.find_min_linear(nums.clone());
            let result5 = solution.find_min_optimized(nums.clone());
            let result6 = solution.find_min_ternary(nums.clone());
            
            assert_eq!(result1, result2, "Binary vs Pivot mismatch for {:?}", nums);
            assert_eq!(result2, result3, "Pivot vs Recursive mismatch for {:?}", nums);
            assert_eq!(result3, result4, "Recursive vs Linear mismatch for {:?}", nums);
            assert_eq!(result4, result5, "Linear vs Optimized mismatch for {:?}", nums);
            assert_eq!(result5, result6, "Optimized vs Ternary mismatch for {:?}", nums);
        }
    }

    #[test]
    fn test_performance_patterns() {
        let solution = setup();
        
        // Best case: minimum at middle
        assert_eq!(solution.find_min(vec![3, 4, 1, 2]), 1);
        
        // Worst case: minimum at end
        assert_eq!(solution.find_min(vec![2, 3, 4, 5, 1]), 1);
        
        // No rotation case
        assert_eq!(solution.find_min(vec![1, 2, 3, 4, 5]), 1);
        
        // Nearly full rotation
        assert_eq!(solution.find_min(vec![2, 1]), 1);
    }

    #[test]
    fn test_mathematical_properties() {
        let solution = setup();
        
        // Property: Result must be smallest element in array
        let nums = vec![4, 5, 6, 7, 0, 1, 2];
        let min_result = solution.find_min(nums.clone());
        let actual_min = *nums.iter().min().unwrap();
        assert_eq!(min_result, actual_min);
        
        // Property: For any rotation of sorted array, minimum stays same
        let original = vec![1, 2, 3, 4, 5];
        let expected_min = 1;
        
        for rotation in 0..original.len() {
            let mut rotated = original.clone();
            rotated.rotate_left(rotation);
            assert_eq!(solution.find_min(rotated), expected_min);
        }
    }

    #[test]
    fn test_rotation_detection() {
        let solution = setup();
        
        // Test different rotation patterns
        let base = vec![10, 20, 30, 40, 50];
        
        for i in 0..base.len() {
            let mut rotated = base.clone();
            rotated.rotate_left(i);
            
            let result = solution.find_min(rotated.clone());
            assert_eq!(result, 10, "Failed for rotation {} of {:?}", i, rotated);
        }
    }

    #[test]
    fn test_monotonicity() {
        let solution = setup();
        
        // Verify that in rotated array, there's exactly one position
        // where nums[i] > nums[i+1] (the rotation point)
        let nums = vec![4, 5, 6, 7, 0, 1, 2];
        let min_val = solution.find_min(nums.clone());
        
        // Find where minimum occurs
        let min_pos = nums.iter().position(|&x| x == min_val).unwrap();
        
        // Verify monotonicity property
        for i in 0..nums.len() - 1 {
            if i + 1 == min_pos {
                // This should be the only decreasing position
                assert!(nums[i] > nums[i + 1]);
            } else {
                // All other positions should be non-decreasing or wrap around
                if i + 1 < nums.len() && i != nums.len() - 1 {
                    // Allow for wrap-around case
                    if min_pos == 0 {
                        assert!(nums[i] <= nums[i + 1]);
                    }
                }
            }
        }
    }

    #[test]
    fn test_binary_search_efficiency() {
        let solution = setup();
        
        // Test that algorithm works efficiently for various sizes
        let sizes = vec![1, 2, 3, 5, 8, 13, 21, 34, 55, 89];
        
        for size in sizes {
            let mut nums: Vec<i32> = (0..size).collect();
            nums.rotate_left((size / 3) as usize);
            
            let result = solution.find_min(nums.clone());
            assert_eq!(result, 0, "Failed for array of size {}", size);
        }
    }

    #[test]
    fn test_special_sequences() {
        let solution = setup();
        
        // Arithmetic progression
        assert_eq!(solution.find_min(vec![2, 4, 6, 8, 10]), 2);
        assert_eq!(solution.find_min(vec![6, 8, 10, 2, 4]), 2);
        
        // Powers of 2
        assert_eq!(solution.find_min(vec![1, 2, 4, 8, 16]), 1);
        assert_eq!(solution.find_min(vec![4, 8, 16, 1, 2]), 1);
        
        // Large gaps
        assert_eq!(solution.find_min(vec![100, 1000, 10000, 1, 10]), 1);
    }

    #[test]
    fn test_identical_elements_constraint() {
        let solution = setup();
        
        // Problem states all elements are unique
        // Test arrays with maximum uniqueness
        let mut unique_array: Vec<i32> = (-10..10).collect();
        unique_array.rotate_left(5);
        assert_eq!(solution.find_min(unique_array), -10);
    }
}