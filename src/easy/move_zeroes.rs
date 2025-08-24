//! # Problem 283: Move Zeroes
//!
//! **Difficulty**: Easy
//! **Topics**: Array, Two Pointers
//! **Acceptance Rate**: 61.8%

/// Solution struct following LeetCode format
pub struct Solution;

impl Solution {
    /// Create a new solution instance
    pub fn new() -> Self {
        Solution
    }

    /// Main solution approach using two pointers (in-place)
    /// 
    /// Time Complexity: O(n) - single pass through array
    /// Space Complexity: O(1) - constant extra space
    pub fn move_zeroes(&self, nums: &mut Vec<i32>) {
        let mut write_index = 0;
        
        // Move all non-zero elements to the front
        for read_index in 0..nums.len() {
            if nums[read_index] != 0 {
                nums[write_index] = nums[read_index];
                write_index += 1;
            }
        }
        
        // Fill remaining positions with zeros
        for i in write_index..nums.len() {
            nums[i] = 0;
        }
    }

    /// Alternative approach with swapping (maintains relative order)
    /// 
    /// Time Complexity: O(n) - single pass through array
    /// Space Complexity: O(1) - constant extra space
    pub fn move_zeroes_swap(&self, nums: &mut Vec<i32>) {
        let mut left = 0;
        
        for right in 0..nums.len() {
            if nums[right] != 0 {
                nums.swap(left, right);
                left += 1;
            }
        }
    }

    /// Optimized approach that minimizes writes
    /// 
    /// Time Complexity: O(n) - single pass
    /// Space Complexity: O(1) - constant space
    pub fn move_zeroes_optimized(&self, nums: &mut Vec<i32>) {
        let mut insert_pos = 0;
        
        // First pass: move non-zero elements
        for i in 0..nums.len() {
            if nums[i] != 0 {
                if i != insert_pos {
                    nums[insert_pos] = nums[i];
                }
                insert_pos += 1;
            }
        }
        
        // Second pass: fill with zeros (only if necessary)
        while insert_pos < nums.len() {
            nums[insert_pos] = 0;
            insert_pos += 1;
        }
    }

    /// Stable partition approach (preserves relative order)
    /// 
    /// Time Complexity: O(n) - single traversal
    /// Space Complexity: O(1) - in-place modification
    pub fn move_zeroes_stable_partition(&self, nums: &mut Vec<i32>) {
        let mut boundary = 0;
        
        // Partition: non-zeros on left, zeros on right
        for current in 0..nums.len() {
            if nums[current] != 0 {
                if current != boundary {
                    // Shift elements to make room
                    let temp = nums[current];
                    for i in (boundary..current).rev() {
                        nums[i + 1] = nums[i];
                    }
                    nums[boundary] = temp;
                }
                boundary += 1;
            }
        }
    }

    /// Brute force approach for comparison (creates new array)
    /// 
    /// Time Complexity: O(n) - two passes
    /// Space Complexity: O(n) - temporary array
    pub fn move_zeroes_brute_force(&self, nums: &mut Vec<i32>) {
        let mut non_zeros = Vec::new();
        let mut zero_count = 0;
        
        // Collect non-zero elements and count zeros
        for &num in nums.iter() {
            if num != 0 {
                non_zeros.push(num);
            } else {
                zero_count += 1;
            }
        }
        
        // Reconstruct the array
        nums.clear();
        nums.extend(non_zeros);
        nums.extend(vec![0; zero_count]);
    }

    /// Advanced approach using quicksort partition logic
    /// 
    /// Time Complexity: O(n) - single pass
    /// Space Complexity: O(1) - in-place
    pub fn move_zeroes_partition(&self, nums: &mut Vec<i32>) {
        let mut i = 0;
        let mut j = 0;
        
        while j < nums.len() {
            if nums[j] != 0 {
                if i != j {
                    nums.swap(i, j);
                }
                i += 1;
            }
            j += 1;
        }
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

    #[test]
    fn test_basic_cases() {
        let solution = Solution::new();
        
        // Test case 1: [0,1,0,3,12]
        let mut nums1 = vec![0, 1, 0, 3, 12];
        solution.move_zeroes(&mut nums1);
        assert_eq!(nums1, vec![1, 3, 12, 0, 0]);
        
        // Test case 2: [0]
        let mut nums2 = vec![0];
        solution.move_zeroes(&mut nums2);
        assert_eq!(nums2, vec![0]);
        
        // Test case 3: [1]
        let mut nums3 = vec![1];
        solution.move_zeroes(&mut nums3);
        assert_eq!(nums3, vec![1]);
        
        // Test case 4: [1,2,3] (no zeros)
        let mut nums4 = vec![1, 2, 3];
        solution.move_zeroes(&mut nums4);
        assert_eq!(nums4, vec![1, 2, 3]);
    }

    #[test]
    fn test_edge_cases() {
        let solution = Solution::new();
        
        // Empty array
        let mut nums: Vec<i32> = vec![];
        solution.move_zeroes(&mut nums);
        assert_eq!(nums, Vec::<i32>::new());
        
        // All zeros
        let mut nums = vec![0, 0, 0];
        solution.move_zeroes(&mut nums);
        assert_eq!(nums, vec![0, 0, 0]);
        
        // No zeros
        let mut nums = vec![1, 2, 3, 4, 5];
        solution.move_zeroes(&mut nums);
        assert_eq!(nums, vec![1, 2, 3, 4, 5]);
        
        // Single zero at beginning
        let mut nums = vec![0, 1, 2];
        solution.move_zeroes(&mut nums);
        assert_eq!(nums, vec![1, 2, 0]);
        
        // Single zero at end
        let mut nums = vec![1, 2, 0];
        solution.move_zeroes(&mut nums);
        assert_eq!(nums, vec![1, 2, 0]);
    }

    #[test]
    fn test_multiple_zeros() {
        let solution = Solution::new();
        
        // Multiple consecutive zeros
        let mut nums = vec![0, 0, 1, 0, 0, 2, 3];
        solution.move_zeroes(&mut nums);
        assert_eq!(nums, vec![1, 2, 3, 0, 0, 0, 0]);
        
        // Alternating zeros and non-zeros
        let mut nums = vec![1, 0, 2, 0, 3, 0, 4];
        solution.move_zeroes(&mut nums);
        assert_eq!(nums, vec![1, 2, 3, 4, 0, 0, 0]);
    }

    #[test]
    fn test_order_preservation() {
        let solution = Solution::new();
        
        // Verify that relative order is preserved
        let mut nums = vec![2, 1, 0, 4, 3, 0, 5];
        solution.move_zeroes(&mut nums);
        assert_eq!(nums, vec![2, 1, 4, 3, 5, 0, 0]);
        
        // The non-zero elements should maintain their relative order
        let non_zeros: Vec<i32> = nums.iter().filter(|&&x| x != 0).cloned().collect();
        assert_eq!(non_zeros, vec![2, 1, 4, 3, 5]);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = Solution::new();
        
        let test_cases = vec![
            vec![0, 1, 0, 3, 12],
            vec![0],
            vec![1],
            vec![0, 0, 1],
            vec![1, 0, 0],
            vec![1, 2, 3, 4, 5],
            vec![0, 0, 0, 0, 0],
            vec![1, 0, 2, 0, 3, 0, 4],
        ];

        for original in test_cases {
            let mut nums1 = original.clone();
            let mut nums2 = original.clone();
            let mut nums3 = original.clone();
            let mut nums4 = original.clone();
            let mut nums5 = original.clone();
            
            solution.move_zeroes(&mut nums1);
            solution.move_zeroes_swap(&mut nums2);
            solution.move_zeroes_optimized(&mut nums3);
            solution.move_zeroes_brute_force(&mut nums4);
            solution.move_zeroes_partition(&mut nums5);
            
            assert_eq!(nums1, nums2, "Main and swap approaches should match");
            assert_eq!(nums1, nums3, "Main and optimized approaches should match");
            assert_eq!(nums1, nums4, "Main and brute force approaches should match");
            assert_eq!(nums1, nums5, "Main and partition approaches should match");
        }
    }

    #[test]
    fn test_performance_scenarios() {
        let solution = Solution::new();
        
        // Large array with many zeros
        let mut nums: Vec<i32> = (0..1000).map(|i| if i % 3 == 0 { 0 } else { i }).collect();
        let original_non_zeros: Vec<i32> = nums.iter().filter(|&&x| x != 0).cloned().collect();
        
        solution.move_zeroes(&mut nums);
        
        // Verify all non-zeros are at the beginning and maintain order
        let result_non_zeros: Vec<i32> = nums.iter()
            .take_while(|&&x| x != 0)
            .cloned()
            .collect();
        
        assert_eq!(result_non_zeros, original_non_zeros);
        
        // Verify all remaining elements are zeros
        let remaining: Vec<i32> = nums.iter()
            .skip(result_non_zeros.len())
            .cloned()
            .collect();
        
        assert!(remaining.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_negative_numbers() {
        let solution = Solution::new();
        
        let mut nums = vec![-1, 0, -2, 0, 3];
        solution.move_zeroes(&mut nums);
        assert_eq!(nums, vec![-1, -2, 3, 0, 0]);
    }

    #[test]
    fn test_stability() {
        let solution = Solution::new();
        
        // Test with duplicate non-zero values to ensure stability
        let mut nums = vec![1, 0, 1, 0, 1];
        solution.move_zeroes(&mut nums);
        assert_eq!(nums, vec![1, 1, 1, 0, 0]);
        
        let mut nums = vec![2, 0, 2, 0, 2, 1];
        solution.move_zeroes(&mut nums);
        assert_eq!(nums, vec![2, 2, 2, 1, 0, 0]);
    }
}

#[cfg(test)]
mod benchmarks {
    // Note: Benchmarks will be added to benches/ directory for Criterion integration
}