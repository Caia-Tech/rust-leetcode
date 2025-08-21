//! Problem 75: Sort Colors
//! 
//! Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects
//! of the same color are adjacent, with the colors in the order red, white, and blue.
//! 
//! We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.
//! 
//! You must solve this problem without using the library's sort function.
//! 
//! Example 1:
//! Input: nums = [2,0,2,1,1,0]
//! Output: [0,0,1,1,2,2]
//! 
//! Example 2:
//! Input: nums = [2,0,1]
//! Output: [0,1,2]

pub struct Solution;

impl Solution {
    /// Approach 1: Dutch National Flag Algorithm (Three Pointers)
    /// 
    /// The classic and most efficient solution using three pointers:
    /// - left: boundary for 0s (everything before left is 0)
    /// - right: boundary for 2s (everything after right is 2)
    /// - current: current element being processed
    /// 
    /// Time Complexity: O(n) - single pass
    /// Space Complexity: O(1) - in-place sorting
    pub fn sort_colors_dutch_flag(&self, nums: &mut Vec<i32>) {
        let mut left = 0;
        let mut right = nums.len() as i32 - 1;
        let mut current = 0;
        
        while current <= right && right >= 0 {
            match nums[current as usize] {
                0 => {
                    // Swap with left boundary and advance both pointers
                    nums.swap(current as usize, left as usize);
                    left += 1;
                    current += 1;
                }
                1 => {
                    // White is in correct position, just advance current
                    current += 1;
                }
                2 => {
                    // Swap with right boundary, advance right boundary
                    // Don't advance current as we need to check the swapped element
                    nums.swap(current as usize, right as usize);
                    right -= 1;
                }
                _ => unreachable!(),
            }
        }
    }
    
    /// Approach 2: Counting Sort
    /// 
    /// Count occurrences of each color, then reconstruct the array.
    /// This is the most straightforward approach.
    /// 
    /// Time Complexity: O(n) - two passes through the array
    /// Space Complexity: O(1) - only using constant extra space for counters
    pub fn sort_colors_counting(&self, nums: &mut Vec<i32>) {
        let mut count = [0; 3]; // count[0] = reds, count[1] = whites, count[2] = blues
        
        // Count each color
        for &num in nums.iter() {
            count[num as usize] += 1;
        }
        
        // Reconstruct array
        let mut index = 0;
        for color in 0..3 {
            for _ in 0..count[color] {
                nums[index] = color as i32;
                index += 1;
            }
        }
    }
    
    /// Approach 3: Two-Pass Approach
    /// 
    /// First pass to partition 0s to the left, second pass to partition 2s to the right.
    /// 
    /// Time Complexity: O(n) - two passes
    /// Space Complexity: O(1) - in-place
    pub fn sort_colors_two_pass(&self, nums: &mut Vec<i32>) {
        if nums.is_empty() {
            return;
        }
        
        let n = nums.len();
        
        // First pass: move all 0s to the front
        let mut boundary = 0;
        for i in 0..n {
            if nums[i] == 0 {
                nums.swap(i, boundary);
                boundary += 1;
            }
        }
        
        // Second pass: move all 2s to the back (start from boundary to avoid affecting 0s)
        let mut boundary = n - 1;
        for i in (0..n).rev() {
            if nums[i] == 2 && i <= boundary {
                nums.swap(i, boundary);
                if boundary > 0 {
                    boundary -= 1;
                } else {
                    break;
                }
            }
        }
    }
    
    /// Approach 4: Partition-based (Similar to QuickSort)
    /// 
    /// Uses the partitioning idea from QuickSort to separate the array.
    /// First partition around 1 (0s go left, 1s and 2s go right),
    /// then partition the right part around 2.
    /// 
    /// Time Complexity: O(n) - each element is moved at most twice
    /// Space Complexity: O(1) - in-place
    pub fn sort_colors_partition(&self, nums: &mut Vec<i32>) {
        if nums.is_empty() {
            return;
        }
        
        // Partition around 1: 0s go to left, 1s and 2s go to right
        let boundary_1 = self.partition(nums, 0, nums.len() - 1, 1);
        
        // Partition the right part around 2: 1s stay, 2s go to right
        if boundary_1 < nums.len() - 1 {
            self.partition(nums, boundary_1, nums.len() - 1, 2);
        }
    }
    
    fn partition(&self, nums: &mut Vec<i32>, start: usize, end: usize, pivot: i32) -> usize {
        let mut i = start;
        for j in start..=end {
            if nums[j] < pivot {
                nums.swap(i, j);
                i += 1;
            }
        }
        i
    }
    
    /// Approach 5: Iterative Three-Way Partitioning
    /// 
    /// Uses three-way partitioning in a single pass (similar to Dutch flag but with different implementation).
    /// 
    /// Time Complexity: O(n) - single pass
    /// Space Complexity: O(1) - iterative approach
    pub fn sort_colors_three_way_quicksort(&self, nums: &mut Vec<i32>) {
        if nums.len() <= 1 {
            return;
        }
        
        let pivot = 1; // Use 1 as pivot for this problem
        let mut equal_start = 0;
        let mut equal_end = 0;
        let mut greater_start = nums.len();
        
        while equal_end < greater_start {
            if nums[equal_end] < pivot {
                nums.swap(equal_start, equal_end);
                equal_start += 1;
                equal_end += 1;
            } else if nums[equal_end] == pivot {
                equal_end += 1;
            } else { // nums[equal_end] > pivot
                greater_start -= 1;
                nums.swap(equal_end, greater_start);
                // Don't increment equal_end since we need to check the swapped element
            }
        }
    }
    
    /// Approach 6: Bubble Sort Optimization
    /// 
    /// A modified bubble sort that's optimized for this specific case.
    /// Since we only have 3 values, we can optimize the bubbling process.
    /// 
    /// Time Complexity: O(n²) worst case, but often much better for this problem
    /// Space Complexity: O(1) - in-place
    pub fn sort_colors_bubble_optimized(&self, nums: &mut Vec<i32>) {
        let n = nums.len();
        if n <= 1 {
            return;
        }
        
        // Modified bubble sort: bubble 2s to the right first, then 1s
        for target in [2, 1].iter() {
            for i in 0..n {
                for j in 0..n - 1 - i {
                    if nums[j] == *target && nums[j + 1] < *target {
                        nums.swap(j, j + 1);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn test_sort_function<F>(sort_fn: F)
    where
        F: Fn(&Solution, &mut Vec<i32>),
    {
        let solution = Solution;
        
        // Test case 1: [2,0,2,1,1,0] -> [0,0,1,1,2,2]
        let mut nums1 = vec![2, 0, 2, 1, 1, 0];
        sort_fn(&solution, &mut nums1);
        assert_eq!(nums1, vec![0, 0, 1, 1, 2, 2]);
        
        // Test case 2: [2,0,1] -> [0,1,2]
        let mut nums2 = vec![2, 0, 1];
        sort_fn(&solution, &mut nums2);
        assert_eq!(nums2, vec![0, 1, 2]);
        
        // Test case 3: Empty array
        let mut nums3 = vec![];
        sort_fn(&solution, &mut nums3);
        assert_eq!(nums3, vec![]);
        
        // Test case 4: Single element
        let mut nums4 = vec![1];
        sort_fn(&solution, &mut nums4);
        assert_eq!(nums4, vec![1]);
        
        // Test case 5: All same color
        let mut nums5 = vec![2, 2, 2, 2];
        sort_fn(&solution, &mut nums5);
        assert_eq!(nums5, vec![2, 2, 2, 2]);
        
        // Test case 6: Already sorted
        let mut nums6 = vec![0, 0, 1, 1, 2, 2];
        sort_fn(&solution, &mut nums6);
        assert_eq!(nums6, vec![0, 0, 1, 1, 2, 2]);
        
        // Test case 7: Reverse sorted
        let mut nums7 = vec![2, 2, 1, 1, 0, 0];
        sort_fn(&solution, &mut nums7);
        assert_eq!(nums7, vec![0, 0, 1, 1, 2, 2]);
    }
    
    #[test]
    fn test_dutch_flag() {
        test_sort_function(|solution, nums| solution.sort_colors_dutch_flag(nums));
    }
    
    #[test]
    fn test_counting() {
        test_sort_function(|solution, nums| solution.sort_colors_counting(nums));
    }
    
    #[test]
    fn test_two_pass() {
        test_sort_function(|solution, nums| solution.sort_colors_two_pass(nums));
    }
    
    #[test]
    fn test_partition() {
        test_sort_function(|solution, nums| solution.sort_colors_partition(nums));
    }
    
    #[test]
    fn test_three_way_quicksort() {
        test_sort_function(|solution, nums| solution.sort_colors_three_way_quicksort(nums));
    }
    
    #[test]
    fn test_bubble_optimized() {
        test_sort_function(|solution, nums| solution.sort_colors_bubble_optimized(nums));
    }
    
    #[test]
    fn test_edge_cases() {
        let solution = Solution;
        
        // Test with two elements
        let mut nums = vec![1, 0];
        solution.sort_colors_dutch_flag(&mut nums);
        assert_eq!(nums, vec![0, 1]);
        
        let mut nums = vec![2, 1];
        solution.sort_colors_dutch_flag(&mut nums);
        assert_eq!(nums, vec![1, 2]);
        
        // Test with all three colors in different orders
        let mut nums = vec![0, 1, 2];
        solution.sort_colors_dutch_flag(&mut nums);
        assert_eq!(nums, vec![0, 1, 2]);
        
        let mut nums = vec![2, 1, 0];
        solution.sort_colors_dutch_flag(&mut nums);
        assert_eq!(nums, vec![0, 1, 2]);
        
        // Test with repeated pattern
        let mut nums = vec![0, 1, 2, 0, 1, 2];
        solution.sort_colors_dutch_flag(&mut nums);
        assert_eq!(nums, vec![0, 0, 1, 1, 2, 2]);
    }
    
    #[test]
    fn test_large_array() {
        let solution = Solution;
        
        // Create a larger test array
        let mut nums = vec![
            2, 0, 1, 2, 1, 0, 2, 1, 0, 1, 2, 0, 1, 2, 0, 1,
            0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0
        ];
        
        let expected_0_count = nums.iter().filter(|&&x| x == 0).count();
        let expected_1_count = nums.iter().filter(|&&x| x == 1).count();
        let expected_2_count = nums.iter().filter(|&&x| x == 2).count();
        
        solution.sort_colors_dutch_flag(&mut nums);
        
        // Verify the array is sorted
        let mut is_sorted = true;
        for i in 1..nums.len() {
            if nums[i] < nums[i - 1] {
                is_sorted = false;
                break;
            }
        }
        assert!(is_sorted);
        
        // Verify counts are preserved
        let actual_0_count = nums.iter().filter(|&&x| x == 0).count();
        let actual_1_count = nums.iter().filter(|&&x| x == 1).count();
        let actual_2_count = nums.iter().filter(|&&x| x == 2).count();
        
        assert_eq!(actual_0_count, expected_0_count);
        assert_eq!(actual_1_count, expected_1_count);
        assert_eq!(actual_2_count, expected_2_count);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            vec![2, 0, 2, 1, 1, 0],
            vec![2, 0, 1],
            vec![],
            vec![1],
            vec![2, 2, 2, 2],
            vec![0, 0, 1, 1, 2, 2],
            vec![2, 2, 1, 1, 0, 0],
            vec![1, 0],
            vec![2, 1, 0],
            vec![0, 1, 2, 0, 1, 2],
        ];
        
        let solution = Solution;
        
        for original in test_cases {
            let mut dutch_flag = original.clone();
            let mut counting = original.clone();
            let mut two_pass = original.clone();
            let mut partition = original.clone();
            let mut quicksort = original.clone();
            let mut bubble = original.clone();
            
            solution.sort_colors_dutch_flag(&mut dutch_flag);
            solution.sort_colors_counting(&mut counting);
            solution.sort_colors_two_pass(&mut two_pass);
            solution.sort_colors_partition(&mut partition);
            solution.sort_colors_three_way_quicksort(&mut quicksort);
            solution.sort_colors_bubble_optimized(&mut bubble);
            
            assert_eq!(dutch_flag, counting, "Dutch flag and counting differ for {:?}", original);
            assert_eq!(dutch_flag, two_pass, "Dutch flag and two-pass differ for {:?}", original);
            assert_eq!(dutch_flag, partition, "Dutch flag and partition differ for {:?}", original);
            assert_eq!(dutch_flag, quicksort, "Dutch flag and quicksort differ for {:?}", original);
            assert_eq!(dutch_flag, bubble, "Dutch flag and bubble differ for {:?}", original);
        }
    }
    
    #[test]
    fn test_stability_properties() {
        let solution = Solution;
        
        // Test that the algorithms correctly handle all three colors
        for &approach in &["dutch_flag", "counting", "two_pass", "partition", "quicksort", "bubble"] {
            let mut nums = vec![2, 0, 2, 1, 1, 0, 2, 1, 0];
            
            match approach {
                "dutch_flag" => solution.sort_colors_dutch_flag(&mut nums),
                "counting" => solution.sort_colors_counting(&mut nums),
                "two_pass" => solution.sort_colors_two_pass(&mut nums),
                "partition" => solution.sort_colors_partition(&mut nums),
                "quicksort" => solution.sort_colors_three_way_quicksort(&mut nums),
                "bubble" => solution.sort_colors_bubble_optimized(&mut nums),
                _ => unreachable!(),
            }
            
            // Verify proper ordering
            let mut prev = -1;
            for &num in &nums {
                assert!(num >= prev, "Array not properly sorted by {}: {:?}", approach, nums);
                assert!(num >= 0 && num <= 2, "Invalid color value");
                prev = num;
            }
        }
    }
}