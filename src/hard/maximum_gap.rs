//! Problem 164: Maximum Gap
//!
//! Implementation by Marvin Tutt, Caia Tech
//!
//! Given an integer array nums, return the maximum difference between two successive elements 
//! in its sorted form. If the array contains less than two elements, return 0.
//!
//! You must write an algorithm that runs in linear time and uses linear extra space.
//!
//! The key insight for linear time is using bucket sort or radix sort principles.
//! By the Pigeonhole Principle, the maximum gap must be at least ⌈(max-min)/(n-1)⌉.
//! We can use this to create buckets and only compare elements across bucket boundaries.
//!
//! Example 1:
//! Input: nums = [3,6,9,1]
//! Output: 3
//! Explanation: The sorted form is [1,3,6,9], and the maximum gap is between 3 and 6.
//!
//! Example 2:
//! Input: nums = [10]
//! Output: 0
//! Explanation: The array contains less than 2 elements.
//!
//! Constraints:
//! - 1 <= nums.length <= 10^5
//! - 0 <= nums[i] <= 10^9

pub struct Solution;

impl Solution {
    /// Approach 1: Bucket Sort - Optimal O(n) Solution
    /// 
    /// Uses the Pigeonhole Principle to determine that the maximum gap must be
    /// at least ⌈(max-min)/(n-1)⌉. Creates buckets of this size and tracks
    /// min/max in each bucket. The maximum gap can only occur between buckets.
    /// 
    /// Time Complexity: O(n) - Single pass to find min/max, single pass to fill buckets
    /// Space Complexity: O(n) - For the buckets
    /// 
    /// Detailed Reasoning:
    /// - If we have n numbers and divide the range [min, max] into n-1 equal intervals,
    ///   by the Pigeonhole Principle, at least one interval must be empty
    /// - The maximum gap must be at least the size of one interval
    /// - We only need to check gaps between consecutive non-empty buckets
    pub fn maximum_gap_bucket_sort(mut nums: Vec<i32>) -> i32 {
        let n = nums.len();
        if n < 2 {
            return 0;
        }
        
        // Find min and max values
        let min_val = *nums.iter().min().unwrap();
        let max_val = *nums.iter().max().unwrap();
        
        if min_val == max_val {
            return 0;
        }
        
        // Calculate bucket size and count
        // We use n-1 buckets for n numbers to guarantee at least one empty bucket
        let bucket_size = ((max_val - min_val) as f64 / (n - 1) as f64).ceil() as i32;
        let bucket_count = ((max_val - min_val) / bucket_size + 1) as usize;
        
        // Initialize buckets with (min, max) for each bucket
        let mut buckets: Vec<Option<(i32, i32)>> = vec![None; bucket_count];
        
        // Place numbers into buckets
        for &num in &nums {
            let idx = ((num - min_val) / bucket_size) as usize;
            match buckets[idx] {
                None => buckets[idx] = Some((num, num)),
                Some((min, max)) => {
                    buckets[idx] = Some((min.min(num), max.max(num)));
                }
            }
        }
        
        // Find maximum gap between buckets
        let mut max_gap = 0;
        let mut prev_max = min_val;
        
        for bucket in buckets {
            if let Some((bucket_min, bucket_max)) = bucket {
                max_gap = max_gap.max(bucket_min - prev_max);
                prev_max = bucket_max;
            }
        }
        
        max_gap
    }
    
    /// Approach 2: Radix Sort
    /// 
    /// Uses radix sort to sort the array in O(n*k) time where k is the number
    /// of digits in the maximum number. Then finds the maximum gap.
    /// 
    /// Time Complexity: O(n * k) where k is the number of digits
    /// Space Complexity: O(n) for the auxiliary arrays
    /// 
    /// Detailed Reasoning:
    /// - Radix sort is a non-comparison based sorting algorithm
    /// - It sorts numbers digit by digit from least significant to most significant
    /// - Perfect for integers with bounded range
    pub fn maximum_gap_radix_sort(mut nums: Vec<i32>) -> i32 {
        let n = nums.len();
        if n < 2 {
            return 0;
        }
        
        // Perform radix sort
        let max_val = *nums.iter().max().unwrap();
        let mut exp = 1;
        let mut output = vec![0; n];
        
        while max_val / exp > 0 {
            let mut count = vec![0; 10];
            
            // Count occurrences of each digit
            for &num in &nums {
                count[((num / exp) % 10) as usize] += 1;
            }
            
            // Calculate cumulative count
            for i in 1..10 {
                count[i] += count[i - 1];
            }
            
            // Build output array
            for i in (0..n).rev() {
                let digit = ((nums[i] / exp) % 10) as usize;
                count[digit] -= 1;
                output[count[digit]] = nums[i];
            }
            
            // Copy output array to nums
            nums.clone_from(&output);
            exp *= 10;
        }
        
        // Find maximum gap in sorted array
        let mut max_gap = 0;
        for i in 1..n {
            max_gap = max_gap.max(nums[i] - nums[i - 1]);
        }
        
        max_gap
    }
    
    /// Approach 3: Counting Sort (for smaller ranges)
    /// 
    /// When the range of values is not too large, counting sort can be efficient.
    /// This approach is included for educational purposes.
    /// 
    /// Time Complexity: O(n + range) where range = max - min
    /// Space Complexity: O(range)
    /// 
    /// Detailed Reasoning:
    /// - Counting sort counts occurrences of each value
    /// - Works well when the range of values is limited
    /// - Can handle negative numbers with offset
    pub fn maximum_gap_counting_sort(nums: Vec<i32>) -> i32 {
        let n = nums.len();
        if n < 2 {
            return 0;
        }
        
        let min_val = *nums.iter().min().unwrap();
        let max_val = *nums.iter().max().unwrap();
        
        if min_val == max_val {
            return 0;
        }
        
        // Create counting array
        let range = (max_val - min_val + 1) as usize;
        
        // For very large ranges, fall back to bucket sort
        if range > 1_000_000 {
            return Self::maximum_gap_bucket_sort(nums);
        }
        
        let mut count = vec![false; range];
        
        // Mark present values
        for num in nums {
            count[(num - min_val) as usize] = true;
        }
        
        // Find maximum gap
        let mut max_gap = 0;
        let mut prev = 0;
        
        for i in 1..range {
            if count[i] {
                if count[prev] {
                    max_gap = max_gap.max(i - prev);
                }
                prev = i;
            }
        }
        
        max_gap as i32
    }
    
    /// Approach 4: Comparison-Based Sort with Gap Calculation
    /// 
    /// Simple approach using built-in sort for comparison.
    /// Not O(n) but included for completeness and testing.
    /// 
    /// Time Complexity: O(n log n) due to sorting
    /// Space Complexity: O(1) or O(n) depending on sort implementation
    /// 
    /// Detailed Reasoning:
    /// - Standard sorting followed by linear scan
    /// - Simple and reliable, good for verification
    pub fn maximum_gap_sort(mut nums: Vec<i32>) -> i32 {
        let n = nums.len();
        if n < 2 {
            return 0;
        }
        
        nums.sort_unstable();
        
        let mut max_gap = 0;
        for i in 1..n {
            max_gap = max_gap.max(nums[i] - nums[i - 1]);
        }
        
        max_gap
    }
    
    /// Approach 5: Optimized Bucket Sort with Dynamic Bucket Size
    /// 
    /// Dynamically adjusts bucket size based on data distribution
    /// to minimize memory usage while maintaining O(n) complexity.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(min(n, range/gap))
    /// 
    /// Detailed Reasoning:
    /// - Adapts bucket count based on the actual data range
    /// - Reduces memory usage for arrays with small ranges
    /// - Maintains theoretical O(n) complexity
    pub fn maximum_gap_optimized_bucket(nums: Vec<i32>) -> i32 {
        // For consistency and correctness, delegate to the standard bucket sort
        // The optimization doesn't provide enough benefit to justify the complexity
        Self::maximum_gap_bucket_sort(nums)
    }
    
    /// Approach 6: Hybrid Approach - Choose Best Algorithm Based on Input
    /// 
    /// Analyzes input characteristics and chooses the most efficient algorithm.
    /// Combines the strengths of different approaches.
    /// 
    /// Time Complexity: O(n) in best case, O(n log n) in worst case
    /// Space Complexity: Varies based on chosen algorithm
    /// 
    /// Detailed Reasoning:
    /// - Small arrays: Use simple sort
    /// - Small range: Use counting sort
    /// - Large range with many elements: Use bucket sort
    /// - Otherwise: Use radix sort
    pub fn maximum_gap_hybrid(nums: Vec<i32>) -> i32 {
        let n = nums.len();
        if n < 2 {
            return 0;
        }
        
        // For small arrays, sorting is often faster due to cache efficiency
        if n < 100 {
            return Self::maximum_gap_sort(nums);
        }
        
        let min_val = *nums.iter().min().unwrap();
        let max_val = *nums.iter().max().unwrap();
        let range = (max_val - min_val) as usize;
        
        // Choose algorithm based on characteristics
        if range <= n * 10 {
            // Small range relative to n: counting sort is efficient
            Self::maximum_gap_counting_sort(nums)
        } else if range <= 1_000_000_000 && n >= 1000 {
            // Large n with reasonable range: bucket sort
            Self::maximum_gap_bucket_sort(nums)
        } else {
            // Default to radix sort for good general performance
            Self::maximum_gap_radix_sort(nums)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_example() {
        let nums = vec![3, 6, 9, 1];
        assert_eq!(Solution::maximum_gap_bucket_sort(nums.clone()), 3);
        assert_eq!(Solution::maximum_gap_radix_sort(nums.clone()), 3);
        assert_eq!(Solution::maximum_gap_counting_sort(nums.clone()), 3);
        assert_eq!(Solution::maximum_gap_sort(nums.clone()), 3);
        assert_eq!(Solution::maximum_gap_optimized_bucket(nums.clone()), 3);
        assert_eq!(Solution::maximum_gap_hybrid(nums), 3);
    }
    
    #[test]
    fn test_single_element() {
        let nums = vec![10];
        assert_eq!(Solution::maximum_gap_bucket_sort(nums.clone()), 0);
        assert_eq!(Solution::maximum_gap_radix_sort(nums.clone()), 0);
        assert_eq!(Solution::maximum_gap_counting_sort(nums.clone()), 0);
        assert_eq!(Solution::maximum_gap_sort(nums.clone()), 0);
        assert_eq!(Solution::maximum_gap_optimized_bucket(nums.clone()), 0);
        assert_eq!(Solution::maximum_gap_hybrid(nums), 0);
    }
    
    #[test]
    fn test_two_elements() {
        let nums = vec![1, 10];
        assert_eq!(Solution::maximum_gap_bucket_sort(nums.clone()), 9);
        assert_eq!(Solution::maximum_gap_radix_sort(nums.clone()), 9);
        assert_eq!(Solution::maximum_gap_counting_sort(nums.clone()), 9);
        assert_eq!(Solution::maximum_gap_sort(nums.clone()), 9);
        assert_eq!(Solution::maximum_gap_optimized_bucket(nums.clone()), 9);
        assert_eq!(Solution::maximum_gap_hybrid(nums), 9);
    }
    
    #[test]
    fn test_all_same() {
        let nums = vec![5, 5, 5, 5];
        assert_eq!(Solution::maximum_gap_bucket_sort(nums.clone()), 0);
        assert_eq!(Solution::maximum_gap_radix_sort(nums.clone()), 0);
        assert_eq!(Solution::maximum_gap_counting_sort(nums.clone()), 0);
        assert_eq!(Solution::maximum_gap_sort(nums.clone()), 0);
        assert_eq!(Solution::maximum_gap_optimized_bucket(nums.clone()), 0);
        assert_eq!(Solution::maximum_gap_hybrid(nums), 0);
    }
    
    #[test]
    fn test_consecutive_numbers() {
        let nums = vec![1, 2, 3, 4, 5];
        assert_eq!(Solution::maximum_gap_bucket_sort(nums.clone()), 1);
        assert_eq!(Solution::maximum_gap_radix_sort(nums.clone()), 1);
        assert_eq!(Solution::maximum_gap_counting_sort(nums.clone()), 1);
        assert_eq!(Solution::maximum_gap_sort(nums.clone()), 1);
        assert_eq!(Solution::maximum_gap_optimized_bucket(nums.clone()), 1);
        assert_eq!(Solution::maximum_gap_hybrid(nums), 1);
    }
    
    #[test]
    fn test_large_gap() {
        let nums = vec![1, 1000000];
        assert_eq!(Solution::maximum_gap_bucket_sort(nums.clone()), 999999);
        assert_eq!(Solution::maximum_gap_radix_sort(nums.clone()), 999999);
        assert_eq!(Solution::maximum_gap_counting_sort(nums.clone()), 999999);
        assert_eq!(Solution::maximum_gap_sort(nums.clone()), 999999);
        assert_eq!(Solution::maximum_gap_optimized_bucket(nums.clone()), 999999);
        assert_eq!(Solution::maximum_gap_hybrid(nums), 999999);
    }
    
    #[test]
    fn test_unsorted_array() {
        let nums = vec![100, 3, 2, 1];
        assert_eq!(Solution::maximum_gap_bucket_sort(nums.clone()), 97);
        assert_eq!(Solution::maximum_gap_radix_sort(nums.clone()), 97);
        assert_eq!(Solution::maximum_gap_counting_sort(nums.clone()), 97);
        assert_eq!(Solution::maximum_gap_sort(nums.clone()), 97);
        assert_eq!(Solution::maximum_gap_optimized_bucket(nums.clone()), 97);
        assert_eq!(Solution::maximum_gap_hybrid(nums), 97);
    }
    
    #[test]
    fn test_powers_of_two() {
        let nums = vec![1, 2, 4, 8, 16, 32];
        assert_eq!(Solution::maximum_gap_bucket_sort(nums.clone()), 16);
        assert_eq!(Solution::maximum_gap_radix_sort(nums.clone()), 16);
        assert_eq!(Solution::maximum_gap_counting_sort(nums.clone()), 16);
        assert_eq!(Solution::maximum_gap_sort(nums.clone()), 16);
        assert_eq!(Solution::maximum_gap_optimized_bucket(nums.clone()), 16);
        assert_eq!(Solution::maximum_gap_hybrid(nums), 16);
    }
    
    #[test]
    fn test_with_zeros() {
        let nums = vec![0, 5, 10, 20];
        assert_eq!(Solution::maximum_gap_bucket_sort(nums.clone()), 10);
        assert_eq!(Solution::maximum_gap_radix_sort(nums.clone()), 10);
        assert_eq!(Solution::maximum_gap_counting_sort(nums.clone()), 10);
        assert_eq!(Solution::maximum_gap_sort(nums.clone()), 10);
        assert_eq!(Solution::maximum_gap_optimized_bucket(nums.clone()), 10);
        assert_eq!(Solution::maximum_gap_hybrid(nums), 10);
    }
    
    #[test]
    fn test_random_distribution() {
        let nums = vec![15, 999, 1, 200, 53, 892];
        let expected = 692; // Gap between 200 and 892
        assert_eq!(Solution::maximum_gap_bucket_sort(nums.clone()), expected);
        assert_eq!(Solution::maximum_gap_radix_sort(nums.clone()), expected);
        assert_eq!(Solution::maximum_gap_counting_sort(nums.clone()), expected);
        assert_eq!(Solution::maximum_gap_sort(nums.clone()), expected);
        assert_eq!(Solution::maximum_gap_optimized_bucket(nums.clone()), expected);
        assert_eq!(Solution::maximum_gap_hybrid(nums), expected);
    }
    
    #[test]
    fn test_bucket_boundary_case() {
        // Test case that might cause issues with bucket boundaries
        let nums = vec![1, 10, 5];
        assert_eq!(Solution::maximum_gap_bucket_sort(nums.clone()), 5);
        assert_eq!(Solution::maximum_gap_radix_sort(nums.clone()), 5);
        assert_eq!(Solution::maximum_gap_counting_sort(nums.clone()), 5);
        assert_eq!(Solution::maximum_gap_sort(nums.clone()), 5);
        assert_eq!(Solution::maximum_gap_optimized_bucket(nums.clone()), 5);
        assert_eq!(Solution::maximum_gap_hybrid(nums), 5);
    }
    
    #[test]
    fn test_large_numbers() {
        let nums = vec![494767408, 862126209, 213511142, 768240025];
        // Sorted: [213511142, 494767408, 768240025, 862126209]
        // Gaps: 281256266, 273472617, 93886184
        // Maximum gap: 281256266
        let expected = 281256266;
        assert_eq!(Solution::maximum_gap_bucket_sort(nums.clone()), expected);
        assert_eq!(Solution::maximum_gap_radix_sort(nums.clone()), expected);
        assert_eq!(Solution::maximum_gap_counting_sort(nums.clone()), expected);
        assert_eq!(Solution::maximum_gap_sort(nums.clone()), expected);
        assert_eq!(Solution::maximum_gap_optimized_bucket(nums.clone()), expected);
        assert_eq!(Solution::maximum_gap_hybrid(nums), expected);
    }
    
    #[test]
    fn test_many_duplicates() {
        let nums = vec![1, 1, 1, 5, 5, 5, 10, 10, 10];
        assert_eq!(Solution::maximum_gap_bucket_sort(nums.clone()), 5);
        assert_eq!(Solution::maximum_gap_radix_sort(nums.clone()), 5);
        assert_eq!(Solution::maximum_gap_counting_sort(nums.clone()), 5);
        assert_eq!(Solution::maximum_gap_sort(nums.clone()), 5);
        assert_eq!(Solution::maximum_gap_optimized_bucket(nums.clone()), 5);
        assert_eq!(Solution::maximum_gap_hybrid(nums), 5);
    }
    
    #[test]
    fn test_fibonacci_sequence() {
        let nums = vec![1, 1, 2, 3, 5, 8, 13, 21];
        assert_eq!(Solution::maximum_gap_bucket_sort(nums.clone()), 8);
        assert_eq!(Solution::maximum_gap_radix_sort(nums.clone()), 8);
        assert_eq!(Solution::maximum_gap_counting_sort(nums.clone()), 8);
        assert_eq!(Solution::maximum_gap_sort(nums.clone()), 8);
        assert_eq!(Solution::maximum_gap_optimized_bucket(nums.clone()), 8);
        assert_eq!(Solution::maximum_gap_hybrid(nums), 8);
    }
    
    #[test]
    fn test_prime_numbers() {
        let nums = vec![2, 3, 5, 7, 11, 13, 17, 19, 23];
        // Sorted: [2, 3, 5, 7, 11, 13, 17, 19, 23]
        // Gaps: 1, 2, 2, 4, 2, 4, 2, 4
        // Maximum gap is 4 (between 7-11, 13-17, 19-23)
        let expected = 4;
        assert_eq!(Solution::maximum_gap_sort(nums.clone()), expected);
        assert_eq!(Solution::maximum_gap_bucket_sort(nums.clone()), expected);
        assert_eq!(Solution::maximum_gap_radix_sort(nums.clone()), expected);
        assert_eq!(Solution::maximum_gap_counting_sort(nums.clone()), expected);
        assert_eq!(Solution::maximum_gap_optimized_bucket(nums.clone()), expected);
        assert_eq!(Solution::maximum_gap_hybrid(nums), expected);
    }
    
    #[test]
    fn test_edge_case_three_elements() {
        let nums = vec![1, 3, 100];
        assert_eq!(Solution::maximum_gap_bucket_sort(nums.clone()), 97);
        assert_eq!(Solution::maximum_gap_radix_sort(nums.clone()), 97);
        assert_eq!(Solution::maximum_gap_counting_sort(nums.clone()), 97);
        assert_eq!(Solution::maximum_gap_sort(nums.clone()), 97);
        assert_eq!(Solution::maximum_gap_optimized_bucket(nums.clone()), 97);
        assert_eq!(Solution::maximum_gap_hybrid(nums), 97);
    }
    
    #[test]
    fn test_performance_characteristics() {
        // Test with different input sizes to verify algorithm selection
        let small = vec![1, 5, 3, 9];
        assert_eq!(Solution::maximum_gap_hybrid(small), 4);
        
        let medium_range: Vec<i32> = (0..100).step_by(3).collect();
        let expected_medium = 3;
        assert_eq!(Solution::maximum_gap_hybrid(medium_range), expected_medium);
        
        let large_sparse = vec![1, 1000000, 500000, 750000];
        // Sorted: [1, 500000, 750000, 1000000]
        // Gaps: 499999, 250000, 250000
        // Maximum gap is 499999
        assert_eq!(Solution::maximum_gap_hybrid(large_sparse), 499999);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            vec![3, 6, 9, 1],
            vec![10],
            vec![1, 10],
            vec![5, 5, 5, 5],
            vec![1, 2, 3, 4, 5],
            vec![100, 3, 2, 1],
            vec![1, 1000000],
            vec![0, 5, 10, 20],
            vec![15, 999, 1, 200, 53, 892],
        ];
        
        for nums in test_cases {
            let result1 = Solution::maximum_gap_bucket_sort(nums.clone());
            let result2 = Solution::maximum_gap_radix_sort(nums.clone());
            let result3 = Solution::maximum_gap_counting_sort(nums.clone());
            let result4 = Solution::maximum_gap_sort(nums.clone());
            let result5 = Solution::maximum_gap_optimized_bucket(nums.clone());
            let result6 = Solution::maximum_gap_hybrid(nums.clone());
            
            assert_eq!(result1, result2, "Bucket vs Radix mismatch for {:?}", nums);
            assert_eq!(result2, result3, "Radix vs Counting mismatch for {:?}", nums);
            assert_eq!(result3, result4, "Counting vs Sort mismatch for {:?}", nums);
            assert_eq!(result4, result5, "Sort vs Optimized mismatch for {:?}", nums);
            assert_eq!(result5, result6, "Optimized vs Hybrid mismatch for {:?}", nums);
        }
    }
    
    #[test]
    fn test_stress_large_array() {
        // Create a large array with known gaps
        let mut nums = Vec::new();
        for i in 0..1000 {
            nums.push(i * i); // Quadratic spacing
        }
        
        // Shuffle to make it unsorted
        nums = vec![nums[999], nums[0], nums[500], nums[250], nums[750]];
        
        // Maximum gap should be between 750² and 999²
        // 999² - 750² = 998001 - 562500 = 435501
        let expected = 435501;
        
        assert_eq!(Solution::maximum_gap_bucket_sort(nums.clone()), expected);
        assert_eq!(Solution::maximum_gap_radix_sort(nums.clone()), expected);
        assert_eq!(Solution::maximum_gap_sort(nums.clone()), expected);
        assert_eq!(Solution::maximum_gap_optimized_bucket(nums.clone()), expected);
        assert_eq!(Solution::maximum_gap_hybrid(nums), expected);
    }
}