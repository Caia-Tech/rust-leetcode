//! # Problem 4: Median of Two Sorted Arrays
//!
//! Given two sorted arrays `nums1` and `nums2` of size `m` and `n` respectively,
//! return **the median** of the two sorted arrays.
//!
//! The overall run time complexity should be `O(log (m+n))`.
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::hard::median_of_two_sorted_arrays::Solution;
//! 
//! let solution = Solution::new();
//! 
//! // Example 1:
//! let nums1 = vec![1, 3];
//! let nums2 = vec![2];
//! assert_eq!(solution.find_median_sorted_arrays(nums1, nums2), 2.0);
//! // Explanation: merged array = [1,2,3] and median is 2.
//! 
//! // Example 2:
//! let nums1 = vec![1, 2];
//! let nums2 = vec![3, 4];
//! assert_eq!(solution.find_median_sorted_arrays(nums1, nums2), 2.5);
//! // Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.
//! ```
//!
//! ## Constraints
//!
//! - nums1.length == m
//! - nums2.length == n
//! - 0 <= m <= 1000
//! - 0 <= n <= 1000
//! - 1 <= m + n <= 2000
//! - -10^6 <= nums1[i], nums2[i] <= 10^6

/// Solution struct following LeetCode format
pub struct Solution;

impl Solution {
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Binary Search on Partitions (Optimal - O(log(min(m,n))))
    /// 
    /// **Algorithm:**
    /// 1. Ensure nums1 is the smaller array (optimize for smaller search space)
    /// 2. Binary search on partition of smaller array
    /// 3. For each partition of nums1, calculate corresponding partition of nums2
    /// 4. Check if partition is valid (left elements ≤ right elements)
    /// 5. Adjust search bounds based on partition validity
    /// 
    /// **Time Complexity:** O(log(min(m, n))) - Binary search on smaller array
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **Key Insight:** We need to partition both arrays such that:
    /// - Left partition has exactly (m+n+1)/2 elements  
    /// - max(left_partition) ≤ min(right_partition)
    /// 
    /// **Why this is optimal:**
    /// - Achieves required O(log(m+n)) complexity
    /// - More specifically O(log(min(m,n))) which is even better
    /// - Avoids merging arrays (saves space and time)
    /// - Uses mathematical insight about median properties
    /// 
    /// **Partition validity conditions:**
    /// - maxLeftX ≤ minRightY (left of X ≤ right of Y)
    /// - maxLeftY ≤ minRightX (left of Y ≤ right of X)
    pub fn find_median_sorted_arrays(&self, nums1: Vec<i32>, nums2: Vec<i32>) -> f64 {
        // Ensure nums1 is smaller for optimization
        if nums1.len() > nums2.len() {
            return self.find_median_sorted_arrays(nums2, nums1);
        }
        
        let m = nums1.len();
        let n = nums2.len();
        let total_left = (m + n + 1) / 2; // Elements in left partition
        
        let mut left = 0;
        let mut right = m;
        
        while left <= right {
            // Partition nums1 at cut1, nums2 at cut2
            let cut1 = (left + right) / 2;
            let cut2 = total_left - cut1;
            
            // Elements at partition boundaries
            let max_left1 = if cut1 == 0 { i32::MIN } else { nums1[cut1 - 1] };
            let min_right1 = if cut1 == m { i32::MAX } else { nums1[cut1] };
            
            let max_left2 = if cut2 == 0 { i32::MIN } else { nums2[cut2 - 1] };
            let min_right2 = if cut2 == n { i32::MAX } else { nums2[cut2] };
            
            // Check if partition is valid
            if max_left1 <= min_right2 && max_left2 <= min_right1 {
                // Found correct partition
                if (m + n) % 2 == 0 {
                    // Even total length: median is average of two middle elements
                    let max_left = max_left1.max(max_left2) as f64;
                    let min_right = min_right1.min(min_right2) as f64;
                    return (max_left + min_right) / 2.0;
                } else {
                    // Odd total length: median is max of left partition
                    return max_left1.max(max_left2) as f64;
                }
            } else if max_left1 > min_right2 {
                // cut1 is too large, search left half
                right = cut1 - 1;
            } else {
                // cut1 is too small, search right half  
                left = cut1 + 1;
            }
        }
        
        unreachable!("Invalid input: arrays are not properly sorted")
    }

    /// # Approach 2: Merge Two Arrays (Simple but not optimal)
    /// 
    /// **Algorithm:**
    /// 1. Merge two sorted arrays using two pointers
    /// 2. Stop when we reach the middle position(s)
    /// 3. Calculate median based on total length parity
    /// 
    /// **Time Complexity:** O(m + n) - Must traverse until middle
    /// **Space Complexity:** O(1) - No extra array storage needed
    /// 
    /// **Why not optimal:**
    /// - Doesn't meet O(log(m+n)) requirement
    /// - Still processes unnecessary elements after median
    /// 
    /// **When this is acceptable:**
    /// - Small arrays where constant factor matters more than big-O
    /// - Implementation simplicity is prioritized
    /// - Educational purposes to show the straightforward approach
    pub fn find_median_sorted_arrays_merge(&self, nums1: Vec<i32>, nums2: Vec<i32>) -> f64 {
        let m = nums1.len();
        let n = nums2.len();
        let total = m + n;
        let is_even = total % 2 == 0;
        
        let mut i = 0; // Index for nums1
        let mut j = 0; // Index for nums2
        let mut current = 0;
        let mut prev = 0;
        
        // We only need to iterate until we find the median position(s)
        let target = total / 2;
        
        for _ in 0..=target {
            prev = current;
            
            // Choose next element from the smaller of current heads
            if i < m && (j >= n || nums1[i] <= nums2[j]) {
                current = nums1[i];
                i += 1;
            } else {
                current = nums2[j];
                j += 1;
            }
        }
        
        if is_even {
            (prev + current) as f64 / 2.0
        } else {
            current as f64
        }
    }

    /// # Approach 3: Brute Force with Full Merge (Inefficient - Educational)
    /// 
    /// **Algorithm:**
    /// 1. Fully merge both arrays into a single sorted array
    /// 2. Find median from merged array
    /// 
    /// **Time Complexity:** O(m + n) - Full merge required
    /// **Space Complexity:** O(m + n) - Stores entire merged array
    /// 
    /// **Why this is inefficient:**
    /// - **Violates O(log(m+n)) requirement**
    /// - **Uses unnecessary space** for elements beyond median
    /// - **No early termination** optimization
    /// 
    /// **Educational value:**
    /// - Shows the most intuitive approach
    /// - Demonstrates why constraints matter
    /// - Baseline for comparing optimized solutions
    /// - Easy to understand and verify correctness
    pub fn find_median_sorted_arrays_brute_force(&self, nums1: Vec<i32>, nums2: Vec<i32>) -> f64 {
        let mut merged = Vec::with_capacity(nums1.len() + nums2.len());
        let mut i = 0;
        let mut j = 0;
        
        // Merge arrays completely
        while i < nums1.len() && j < nums2.len() {
            if nums1[i] <= nums2[j] {
                merged.push(nums1[i]);
                i += 1;
            } else {
                merged.push(nums2[j]);
                j += 1;
            }
        }
        
        // Add remaining elements
        merged.extend_from_slice(&nums1[i..]);
        merged.extend_from_slice(&nums2[j..]);
        
        // Find median
        let len = merged.len();
        if len % 2 == 0 {
            (merged[len / 2 - 1] + merged[len / 2]) as f64 / 2.0
        } else {
            merged[len / 2] as f64
        }
    }

    /// # Approach 4: Kth Element Algorithm (Alternative O(log(m+n)))
    /// 
    /// **Algorithm:**
    /// 1. Transform median problem into "find kth smallest element"
    /// 2. Use binary search to find kth element in merged conceptual array
    /// 3. For even length, find both k/2 and k/2+1 elements
    /// 
    /// **Time Complexity:** O(log(m + n)) - Binary search on element ranks
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **Key insight:** Finding median = finding kth element where k = (m+n)/2
    /// 
    /// **Difference from Approach 1:**
    /// - Searches on element values rather than partition indices
    /// - More general solution (can find any kth element)
    /// - Same complexity but different implementation approach
    /// 
    /// **When to prefer:**
    /// - When you need to solve more general "kth element" problems
    /// - Alternative perspective on binary search approach
    pub fn find_median_sorted_arrays_kth(&self, nums1: Vec<i32>, nums2: Vec<i32>) -> f64 {
        let total = nums1.len() + nums2.len();
        
        if total % 2 == 1 {
            // Odd length: find middle element
            self.find_kth_element(&nums1, &nums2, total / 2 + 1) as f64
        } else {
            // Even length: average of two middle elements
            let left = self.find_kth_element(&nums1, &nums2, total / 2);
            let right = self.find_kth_element(&nums1, &nums2, total / 2 + 1);
            (left + right) as f64 / 2.0
        }
    }
    
    /// Helper function to find the kth smallest element in two sorted arrays
    fn find_kth_element(&self, nums1: &[i32], nums2: &[i32], k: usize) -> i32 {
        // Ensure nums1 is the smaller array
        if nums1.len() > nums2.len() {
            return self.find_kth_element(nums2, nums1, k);
        }
        
        let m = nums1.len();
        let n = nums2.len();
        
        if m == 0 {
            return nums2[k - 1];
        }
        
        if k == 1 {
            return nums1[0].min(nums2[0]);
        }
        
        // Binary search bounds
        let i = (k / 2).min(m);
        let j = k - i;
        
        if j > n {
            // Not enough elements in nums2, must take more from nums1
            return self.find_kth_element(&nums1[k - n..], nums2, k - (k - n));
        }
        
        if nums1[i - 1] < nums2[j - 1] {
            // Discard first i elements of nums1
            self.find_kth_element(&nums1[i..], nums2, k - i)
        } else {
            // Discard first j elements of nums2
            self.find_kth_element(nums1, &nums2[j..], k - j)
        }
    }

    /// # Approach 5: Recursive Binary Search (Alternative Implementation)
    /// 
    /// **Algorithm:**
    /// 1. Recursively binary search on the shorter array
    /// 2. At each step, determine if partition is correct
    /// 3. Recurse on appropriate half based on partition validity
    /// 
    /// **Time Complexity:** O(log(min(m, n))) - Recursive binary search
    /// **Space Complexity:** O(log(min(m, n))) - Recursion stack depth
    /// 
    /// **Difference from Approach 1:**
    /// - Recursive implementation vs iterative
    /// - Same algorithm, different style
    /// - Uses call stack instead of explicit loop
    /// 
    /// **Trade-offs:**
    /// - **Pros:** More mathematical/functional style, cleaner logic flow
    /// - **Cons:** Uses call stack space, potential stack overflow on deep recursion
    /// 
    /// **When to prefer:** When recursive style fits better with codebase patterns
    pub fn find_median_sorted_arrays_recursive(&self, nums1: Vec<i32>, nums2: Vec<i32>) -> f64 {
        if nums1.len() > nums2.len() {
            return self.find_median_sorted_arrays_recursive(nums2, nums1);
        }
        
        self.find_median_recursive(&nums1, &nums2, 0, nums1.len())
    }
    
    fn find_median_recursive(&self, nums1: &[i32], nums2: &[i32], left: usize, right: usize) -> f64 {
        let m = nums1.len();
        let n = nums2.len();
        
        if left > right {
            return self.find_median_recursive(nums2, nums1, 0, n);
        }
        
        let cut1 = (left + right) / 2;
        let cut2 = (m + n + 1) / 2 - cut1;
        
        let max_left1 = if cut1 == 0 { i32::MIN } else { nums1[cut1 - 1] };
        let min_right1 = if cut1 == m { i32::MAX } else { nums1[cut1] };
        
        let max_left2 = if cut2 == 0 { i32::MIN } else { nums2[cut2 - 1] };
        let min_right2 = if cut2 == n { i32::MAX } else { nums2[cut2] };
        
        if max_left1 <= min_right2 && max_left2 <= min_right1 {
            if (m + n) % 2 == 0 {
                let max_left = max_left1.max(max_left2) as f64;
                let min_right = min_right1.min(min_right2) as f64;
                (max_left + min_right) / 2.0
            } else {
                max_left1.max(max_left2) as f64
            }
        } else if max_left1 > min_right2 {
            self.find_median_recursive(nums1, nums2, left, cut1 - 1)
        } else {
            self.find_median_recursive(nums1, nums2, cut1 + 1, right)
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
    use rstest::rstest;

    fn setup() -> Solution {
        Solution::new()
    }

    #[rstest]
    #[case(vec![1, 3], vec![2], 2.0)]         // [1,2,3] -> 2
    #[case(vec![1, 2], vec![3, 4], 2.5)]      // [1,2,3,4] -> 2.5
    #[case(vec![0, 0], vec![0, 0], 0.0)]      // [0,0,0,0] -> 0
    #[case(vec![], vec![1], 1.0)]             // [1] -> 1
    #[case(vec![2], vec![], 2.0)]             // [2] -> 2
    #[case(vec![1], vec![2, 3], 2.0)]         // [1,2,3] -> 2
    #[case(vec![1, 2, 3], vec![4, 5, 6], 3.5)] // [1,2,3,4,5,6] -> 3.5
    fn test_basic_cases(
        #[case] nums1: Vec<i32>,
        #[case] nums2: Vec<i32>, 
        #[case] expected: f64
    ) {
        let solution = setup();
        let result = solution.find_median_sorted_arrays(nums1, nums2);
        assert!((result - expected).abs() < 1e-9, "Expected {}, got {}", expected, result);
    }

    #[test]
    fn test_single_elements() {
        let solution = setup();
        
        // Both single elements
        assert_eq!(solution.find_median_sorted_arrays(vec![1], vec![2]), 1.5);
        assert_eq!(solution.find_median_sorted_arrays(vec![2], vec![1]), 1.5);
        assert_eq!(solution.find_median_sorted_arrays(vec![5], vec![5]), 5.0);
    }

    #[test]
    fn test_one_empty_array() {
        let solution = setup();
        
        // One array empty
        assert_eq!(solution.find_median_sorted_arrays(vec![], vec![1, 2, 3, 4, 5]), 3.0);
        assert_eq!(solution.find_median_sorted_arrays(vec![1, 2, 3, 4], vec![]), 2.5);
        assert_eq!(solution.find_median_sorted_arrays(vec![], vec![2]), 2.0);
        assert_eq!(solution.find_median_sorted_arrays(vec![3], vec![]), 3.0);
    }

    #[test]
    fn test_different_lengths() {
        let solution = setup();
        
        // Significantly different lengths
        assert_eq!(solution.find_median_sorted_arrays(vec![1], vec![2, 3, 4, 5, 6, 7]), 4.0);
        assert_eq!(solution.find_median_sorted_arrays(vec![1, 2, 3, 4, 5], vec![6]), 3.5);
        
        // More complex cases
        let result = solution.find_median_sorted_arrays(vec![1, 3], vec![2, 4, 5, 6, 7, 8]);
        assert!((result - 4.5).abs() < 1e-9); // [1,2,3,4,5,6,7,8] -> 4.5
    }

    #[test]
    fn test_negative_numbers() {
        let solution = setup();
        
        assert_eq!(solution.find_median_sorted_arrays(vec![-5, -3, -1], vec![-2, 0, 2]), -1.5);
        assert_eq!(solution.find_median_sorted_arrays(vec![-10, -5], vec![-7, -3, 1]), -5.0);
        assert_eq!(solution.find_median_sorted_arrays(vec![-100], vec![-200, -150]), -150.0);
    }

    #[test]
    fn test_duplicate_elements() {
        let solution = setup();
        
        // Arrays with duplicates
        assert_eq!(solution.find_median_sorted_arrays(vec![1, 1, 1], vec![1, 1, 1]), 1.0);
        assert_eq!(solution.find_median_sorted_arrays(vec![1, 2, 2], vec![2, 3, 3]), 2.0);
        assert_eq!(solution.find_median_sorted_arrays(vec![1, 1, 3, 3], vec![1, 1, 3, 3]), 2.0);
    }

    #[test]
    fn test_extreme_values() {
        let solution = setup();
        
        // Test constraint boundaries: -10^6 <= nums[i] <= 10^6
        let large = 1_000_000;
        let small = -1_000_000;
        
        assert_eq!(solution.find_median_sorted_arrays(vec![small], vec![large]), 0.0);
        assert_eq!(solution.find_median_sorted_arrays(vec![small, small], vec![large, large]), 0.0);
        assert_eq!(solution.find_median_sorted_arrays(vec![small, 0, large], vec![]), 0.0);
    }

    #[test]
    fn test_edge_cases_comprehensive() {
        let solution = setup();
        
        // Edge case: One element overlaps with range
        assert_eq!(solution.find_median_sorted_arrays(vec![1, 2], vec![1, 2]), 1.5);
        
        // Edge case: All elements same
        assert_eq!(solution.find_median_sorted_arrays(vec![5, 5, 5], vec![5, 5]), 5.0);
        
        // Edge case: Interleaved perfectly
        let result = solution.find_median_sorted_arrays(vec![1, 3, 5, 7], vec![2, 4, 6, 8]);
        assert!((result - 4.5).abs() < 1e-9);
        
        // Edge case: No overlap
        assert_eq!(solution.find_median_sorted_arrays(vec![1, 2, 3], vec![7, 8, 9]), 5.0);
    }

    #[test]
    fn test_large_arrays() {
        let solution = setup();
        
        // Test with larger arrays (within constraints: 1000 elements max)
        let nums1: Vec<i32> = (0..500).map(|i| i * 2).collect();     // [0, 2, 4, 6, ...]
        let nums2: Vec<i32> = (0..500).map(|i| i * 2 + 1).collect(); // [1, 3, 5, 7, ...]
        
        // Combined: [0, 1, 2, 3, 4, 5, ...] - perfect sequence
        // Median of [0..1000) is 499.5
        let result = solution.find_median_sorted_arrays(nums1, nums2);
        assert!((result - 499.5).abs() < 1e-9);
    }

    #[test]
    fn test_precision_edge_cases() {
        let solution = setup();
        
        // Cases that test floating point precision
        assert_eq!(solution.find_median_sorted_arrays(vec![1, 3], vec![2, 4]), 2.5);
        assert_eq!(solution.find_median_sorted_arrays(vec![1, 3, 5], vec![2, 4, 6]), 3.5);
        
        // Odd number of total elements
        assert_eq!(solution.find_median_sorted_arrays(vec![1, 2, 3], vec![4, 5]), 3.0);
        assert_eq!(solution.find_median_sorted_arrays(vec![1], vec![2, 3, 4, 5]), 3.0);
    }

    #[test]
    fn test_all_approaches_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            (vec![1, 3], vec![2]),
            (vec![1, 2], vec![3, 4]),
            (vec![], vec![1]),
            (vec![2], vec![]),
            (vec![0, 0], vec![0, 0]),
            (vec![1, 1, 3, 3], vec![1, 1, 3, 3]),
            (vec![-5, -3, -1], vec![-2, 0, 2]),
            (vec![1], vec![2, 3, 4, 5]),
            (vec![1, 2, 3, 4], vec![5, 6, 7, 8]),
        ];
        
        for (nums1, nums2) in test_cases {
            let result1 = solution.find_median_sorted_arrays(nums1.clone(), nums2.clone());
            let result2 = solution.find_median_sorted_arrays_merge(nums1.clone(), nums2.clone());
            let result3 = solution.find_median_sorted_arrays_brute_force(nums1.clone(), nums2.clone());
            let result4 = solution.find_median_sorted_arrays_kth(nums1.clone(), nums2.clone());
            let result5 = solution.find_median_sorted_arrays_recursive(nums1.clone(), nums2.clone());
            
            assert!((result1 - result2).abs() < 1e-9, 
                   "Merge approach differs: {} vs {} for {:?}, {:?}", result1, result2, nums1, nums2);
            assert!((result1 - result3).abs() < 1e-9, 
                   "Brute force differs: {} vs {} for {:?}, {:?}", result1, result3, nums1, nums2);
            assert!((result1 - result4).abs() < 1e-9, 
                   "Kth element differs: {} vs {} for {:?}, {:?}", result1, result4, nums1, nums2);
            assert!((result1 - result5).abs() < 1e-9, 
                   "Recursive differs: {} vs {} for {:?}, {:?}", result1, result5, nums1, nums2);
        }
    }

    #[test]
    fn test_algorithm_complexity_scenarios() {
        let solution = setup();
        
        // Scenario 1: Binary search should excel (large arrays, small difference)
        let large1: Vec<i32> = (0..500).collect();
        let large2: Vec<i32> = (500..1000).collect();
        let result1 = solution.find_median_sorted_arrays(large1, large2);
        assert!((result1 - 499.5).abs() < 1e-9);
        
        // Scenario 2: One much smaller than other (optimal binary search benefits)
        let small = vec![1];
        let large: Vec<i32> = (2..1001).collect();
        let result2 = solution.find_median_sorted_arrays(small, large);
        assert!((result2 - 500.5).abs() < 1e-9); // Combined [1,2,3,...,1000] -> (500+501)/2 = 500.5
        
        // Scenario 3: Highly interleaved (tests partition logic thoroughly)
        let evens: Vec<i32> = (0..100).map(|i| i * 2).collect();
        let odds: Vec<i32> = (0..100).map(|i| i * 2 + 1).collect();
        let result3 = solution.find_median_sorted_arrays(evens, odds);
        assert!((result3 - 99.5).abs() < 1e-9); // Middle of [0, 1, 2, ..., 199]
    }

    #[test]
    fn test_boundary_partition_cases() {
        let solution = setup();
        
        // Case where optimal partition is at boundary of one array
        assert_eq!(solution.find_median_sorted_arrays(vec![1, 2, 3], vec![4]), 2.5);
        assert_eq!(solution.find_median_sorted_arrays(vec![4], vec![1, 2, 3]), 2.5);
        
        // Case where partition cuts through middle of both arrays
        assert_eq!(solution.find_median_sorted_arrays(vec![1, 3, 5], vec![2, 4, 6]), 3.5);
        
        // Case where one array is completely on one side
        assert_eq!(solution.find_median_sorted_arrays(vec![1, 2], vec![3, 4, 5, 6]), 3.5);
    }

    #[test] 
    fn test_mathematical_edge_cases() {
        let solution = setup();
        
        // Test cases that stress the mathematical logic
        
        // Perfect split case
        assert_eq!(solution.find_median_sorted_arrays(vec![1, 2], vec![3, 4]), 2.5);
        
        // Uneven split with even total
        assert_eq!(solution.find_median_sorted_arrays(vec![1], vec![2, 3, 4]), 2.5);
        
        // Uneven split with odd total
        assert_eq!(solution.find_median_sorted_arrays(vec![1, 2], vec![3, 4, 5]), 3.0);
        
        // All elements from one array in left partition
        assert_eq!(solution.find_median_sorted_arrays(vec![1, 2, 3], vec![10, 11, 12]), 6.5);
    }

    #[test]
    fn test_stress_scenarios() {
        let solution = setup();
        
        // Maximum size arrays (1000 each)
        let max1: Vec<i32> = (0..1000).collect();
        let max2: Vec<i32> = (1000..2000).collect(); 
        let result = solution.find_median_sorted_arrays(max1, max2);
        assert!((result - 999.5).abs() < 1e-9);
        
        // Minimum size (both arrays have 1 element)
        assert_eq!(solution.find_median_sorted_arrays(vec![1], vec![2]), 1.5);
        
        // Edge of constraint values
        let min_vals = vec![-1_000_000, -999_999];
        let max_vals = vec![999_999, 1_000_000];
        let result = solution.find_median_sorted_arrays(min_vals, max_vals);
        assert!((result - 0.0).abs() < 1e-9); // [-1000000, -999999, 999999, 1000000] -> (-999999+999999)/2 = 0
    }
}