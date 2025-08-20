//! # Problem 53: Maximum Subarray
//!
//! Given an integer array `nums`, find the contiguous subarray (containing at least one number)
//! which has the largest sum and return its sum.
//!
//! A subarray is a contiguous part of an array.
//!
//! ## Examples
//!
//! ```text
//! Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
//! Output: 6
//! Explanation: [4,-1,2,1] has the largest sum = 6.
//! ```
//!
//! ```text
//! Input: nums = [1]
//! Output: 1
//! ```
//!
//! ```text
//! Input: nums = [5,4,-1,7,8]
//! Output: 23
//! ```
//!
//! ## Constraints
//!
//! * 1 <= nums.length <= 10^5
//! * -10^4 <= nums[i] <= 10^4
//!
//! ## Follow up
//!
//! If you have figured out the O(n) solution, try coding another solution using 
//! the divide and conquer approach, which is more subtle.

/// Solution for Maximum Subarray problem
pub struct Solution;

impl Solution {
    /// Creates a new instance of Solution
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Kadane's Algorithm (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Keep track of maximum sum ending at current position
    /// 2. For each element, decide: extend current subarray or start new one
    /// 3. Update global maximum at each step
    /// 
    /// **Time Complexity:** O(n) - Single pass through array
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **Key Insight:** At each position, we have two choices:
    /// - Extend the existing subarray to include current element
    /// - Start a new subarray from current element
    /// We choose whichever gives larger sum
    /// 
    /// **Why this works:**
    /// - If previous sum is negative, starting fresh is better
    /// - If previous sum is positive, extending is beneficial
    /// - We track the best sum seen so far
    /// 
    /// **Visualization:**
    /// ```text
    /// nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    /// max_ending_here: -2 → 1 → -2 → 4 → 3 → 5 → 6 → 1 → 5
    /// max_so_far:      -2 → 1 → 1 → 4 → 4 → 5 → 6 → 6 → 6
    /// ```
    pub fn max_sub_array(&self, nums: Vec<i32>) -> i32 {
        let mut max_ending_here = nums[0];
        let mut max_so_far = nums[0];
        
        for &num in nums.iter().skip(1) {
            // Decide: extend current subarray or start new one
            max_ending_here = max_ending_here.max(0) + num;
            // Update global maximum
            max_so_far = max_so_far.max(max_ending_here);
        }
        
        max_so_far
    }

    /// # Approach 2: Dynamic Programming (Alternative)
    /// 
    /// **Algorithm:**
    /// 1. Define dp[i] = maximum sum of subarray ending at index i
    /// 2. dp[i] = max(nums[i], dp[i-1] + nums[i])
    /// 3. Result is maximum value in dp array
    /// 
    /// **Time Complexity:** O(n) - Process each element once
    /// **Space Complexity:** O(n) - Store dp array
    /// 
    /// **DP State Definition:**
    /// - dp[i] represents the maximum sum of any subarray that ends at index i
    /// - This must include nums[i] (constraint of ending at i)
    /// 
    /// **Recurrence Relation:**
    /// - Either start new subarray at i: dp[i] = nums[i]
    /// - Or extend previous subarray: dp[i] = dp[i-1] + nums[i]
    /// - Take maximum of both choices
    /// 
    /// **Why use this approach:**
    /// - More explicit about DP state
    /// - Easier to understand for DP beginners
    /// - Can be modified to track actual subarray indices
    pub fn max_sub_array_dp(&self, nums: Vec<i32>) -> i32 {
        let n = nums.len();
        let mut dp = vec![0; n];
        dp[0] = nums[0];
        let mut max_sum = nums[0];
        
        for i in 1..n {
            // Either extend previous subarray or start new one
            dp[i] = nums[i].max(dp[i - 1] + nums[i]);
            max_sum = max_sum.max(dp[i]);
        }
        
        max_sum
    }

    /// # Approach 3: Divide and Conquer (Follow-up)
    /// 
    /// **Algorithm:**
    /// 1. Divide array into two halves
    /// 2. Recursively find maximum subarray in left half
    /// 3. Recursively find maximum subarray in right half
    /// 4. Find maximum subarray that crosses the middle
    /// 5. Return maximum of all three
    /// 
    /// **Time Complexity:** O(n log n) - T(n) = 2T(n/2) + O(n)
    /// **Space Complexity:** O(log n) - Recursion stack depth
    /// 
    /// **Key Insight:** Maximum subarray is either:
    /// - Entirely in left half
    /// - Entirely in right half
    /// - Crosses the middle point
    /// 
    /// **Why this approach is interesting:**
    /// - Demonstrates divide and conquer paradigm
    /// - Similar to merge sort structure
    /// - Can be parallelized (left and right computed independently)
    /// 
    /// **Finding crossing subarray:**
    /// - Start from middle, expand left to find max left sum
    /// - Start from middle+1, expand right to find max right sum
    /// - Crossing sum = left sum + right sum
    pub fn max_sub_array_divide_conquer(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        self.divide_conquer_helper(&nums, 0, nums.len() - 1)
    }
    
    fn divide_conquer_helper(&self, nums: &[i32], left: usize, right: usize) -> i32 {
        if left == right {
            return nums[left];
        }
        
        let mid = left + (right - left) / 2;
        
        // Find max subarray sum in left half
        let left_sum = self.divide_conquer_helper(nums, left, mid);
        
        // Find max subarray sum in right half
        let right_sum = self.divide_conquer_helper(nums, mid + 1, right);
        
        // Find max subarray sum that crosses the middle
        let cross_sum = self.max_crossing_sum(nums, left, mid, right);
        
        // Return maximum of all three
        left_sum.max(right_sum).max(cross_sum)
    }
    
    fn max_crossing_sum(&self, nums: &[i32], left: usize, mid: usize, right: usize) -> i32 {
        // Find max sum for left part (including mid)
        let mut left_sum = i32::MIN;
        let mut sum = 0;
        for i in (left..=mid).rev() {
            sum += nums[i];
            left_sum = left_sum.max(sum);
        }
        
        // Find max sum for right part (starting from mid+1)
        let mut right_sum = i32::MIN;
        sum = 0;
        for i in (mid + 1)..=right {
            sum += nums[i];
            right_sum = right_sum.max(sum);
        }
        
        // Return sum of both parts
        left_sum + right_sum
    }

    /// # Approach 4: Brute Force (Inefficient - Educational)
    /// 
    /// **Algorithm:**
    /// 1. Try all possible subarrays
    /// 2. Calculate sum for each subarray
    /// 3. Keep track of maximum sum
    /// 
    /// **Time Complexity:** O(n²) - Check all pairs of indices
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **Why this approach is inefficient:**
    /// - Recalculates sums repeatedly
    /// - No reuse of previous computations
    /// - Doesn't leverage problem structure
    /// 
    /// **When to use:** Only for very small arrays or verification
    pub fn max_sub_array_brute_force(&self, nums: Vec<i32>) -> i32 {
        let mut max_sum = i32::MIN;
        
        for i in 0..nums.len() {
            let mut current_sum = 0;
            for j in i..nums.len() {
                current_sum += nums[j];
                max_sum = max_sum.max(current_sum);
            }
        }
        
        max_sum
    }

    /// # Approach 5: Prefix Sum (Alternative)
    /// 
    /// **Algorithm:**
    /// 1. Compute prefix sums
    /// 2. For each position, find minimum prefix sum before it
    /// 3. Maximum subarray sum ending at i = prefix[i] - min_prefix_before_i
    /// 
    /// **Time Complexity:** O(n) - Two passes through array
    /// **Space Complexity:** O(1) - Can be done with running values
    /// 
    /// **Key Insight:** 
    /// - Subarray sum[i..j] = prefix[j] - prefix[i-1]
    /// - To maximize this, minimize prefix[i-1]
    /// 
    /// **Why this works:**
    /// - Transform problem to finding maximum difference
    /// - Similar to "best time to buy and sell stock" problem
    pub fn max_sub_array_prefix_sum(&self, nums: Vec<i32>) -> i32 {
        let mut max_sum = nums[0];
        let mut current_sum = 0;
        let mut min_prefix = 0;
        
        for &num in &nums {
            current_sum += num;
            // Maximum subarray ending here = current_sum - minimum prefix before
            max_sum = max_sum.max(current_sum - min_prefix);
            // Update minimum prefix sum seen so far
            min_prefix = min_prefix.min(current_sum);
        }
        
        max_sum
    }

    /// # Approach 6: Modified Kadane's (Tracks Indices)
    /// 
    /// **Algorithm:**
    /// Same as Kadane's but also tracks start and end indices of maximum subarray
    /// 
    /// **Time Complexity:** O(n) - Single pass
    /// **Space Complexity:** O(1) - Only tracking indices
    /// 
    /// **Additional Feature:** Returns the actual subarray that has maximum sum
    /// 
    /// **Practical Use:** When you need to know which elements form the maximum subarray
    pub fn max_sub_array_with_indices(&self, nums: Vec<i32>) -> (i32, usize, usize) {
        let mut max_ending_here = nums[0];
        let mut max_so_far = nums[0];
        
        let mut start = 0;
        let mut end = 0;
        let mut temp_start = 0;
        
        for i in 1..nums.len() {
            // If starting fresh is better, update temp_start
            if nums[i] > max_ending_here + nums[i] {
                max_ending_here = nums[i];
                temp_start = i;
            } else {
                max_ending_here += nums[i];
            }
            
            // If we found a new maximum, update indices
            if max_ending_here > max_so_far {
                max_so_far = max_ending_here;
                start = temp_start;
                end = i;
            }
        }
        
        (max_so_far, start, end)
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
    fn test_basic_cases() {
        let solution = setup();
        
        // Example from problem description
        assert_eq!(solution.max_sub_array(vec![-2,1,-3,4,-1,2,1,-5,4]), 6);
        
        // Single element
        assert_eq!(solution.max_sub_array(vec![1]), 1);
        
        // All positive
        assert_eq!(solution.max_sub_array(vec![5,4,-1,7,8]), 23);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // All negative numbers - should return least negative
        assert_eq!(solution.max_sub_array(vec![-5, -2, -8, -1, -4]), -1);
        
        // Single negative number
        assert_eq!(solution.max_sub_array(vec![-1]), -1);
        
        // Alternating positive and negative
        assert_eq!(solution.max_sub_array(vec![1, -1, 1, -1, 1]), 1);
        
        // Large positive at the end
        assert_eq!(solution.max_sub_array(vec![-2, -3, -1, 100]), 100);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            vec![-2,1,-3,4,-1,2,1,-5,4],
            vec![1],
            vec![5,4,-1,7,8],
            vec![-5, -2, -8, -1, -4],
            vec![1, -1, 1, -1, 1],
            vec![-2, -3, -1, 100],
            vec![0, -2, -3, 0, -1],
        ];
        
        for nums in test_cases {
            let result1 = solution.max_sub_array(nums.clone());
            let result2 = solution.max_sub_array_dp(nums.clone());
            let result3 = solution.max_sub_array_divide_conquer(nums.clone());
            let result4 = solution.max_sub_array_brute_force(nums.clone());
            let result5 = solution.max_sub_array_prefix_sum(nums.clone());
            let (result6, _, _) = solution.max_sub_array_with_indices(nums.clone());
            
            assert_eq!(result1, result2, "Kadane vs DP mismatch for {:?}", nums);
            assert_eq!(result2, result3, "DP vs Divide & Conquer mismatch for {:?}", nums);
            assert_eq!(result3, result4, "Divide & Conquer vs Brute Force mismatch for {:?}", nums);
            assert_eq!(result4, result5, "Brute Force vs Prefix Sum mismatch for {:?}", nums);
            assert_eq!(result5, result6, "Prefix Sum vs Kadane with indices mismatch for {:?}", nums);
        }
    }

    #[test]
    fn test_with_zeros() {
        let solution = setup();
        
        // Array with zeros
        assert_eq!(solution.max_sub_array(vec![0, -2, 0, -1, 0]), 0);
        
        // Zeros at boundaries
        assert_eq!(solution.max_sub_array(vec![0, 1, 2, 3, 0]), 6);
        
        // All zeros
        assert_eq!(solution.max_sub_array(vec![0, 0, 0, 0]), 0);
    }

    #[test]
    fn test_indices_tracking() {
        let solution = setup();
        
        // Test that indices are correct
        let nums = vec![-2,1,-3,4,-1,2,1,-5,4];
        let (sum, start, end) = solution.max_sub_array_with_indices(nums.clone());
        
        assert_eq!(sum, 6);
        assert_eq!(start, 3);
        assert_eq!(end, 6);
        
        // Verify the subarray sum
        let subarray_sum: i32 = nums[start..=end].iter().sum();
        assert_eq!(subarray_sum, sum);
    }

    #[test]
    fn test_mathematical_properties() {
        let solution = setup();
        
        // Sum of entire array when all positive
        let all_positive = vec![1, 2, 3, 4, 5];
        let total: i32 = all_positive.iter().sum();
        assert_eq!(solution.max_sub_array(all_positive), total);
        
        // Maximum element when all negative
        let all_negative = vec![-5, -3, -7, -2, -9];
        let max_element = *all_negative.iter().max().unwrap();
        assert_eq!(solution.max_sub_array(all_negative), max_element);
        
        // Subarray sum should not exceed total sum for non-negative arrays
        let non_negative = vec![0, 1, 2, 0, 3, 0];
        let total_sum: i32 = non_negative.iter().sum();
        let max_sub = solution.max_sub_array(non_negative);
        assert!(max_sub <= total_sum);
    }

    #[test]
    fn test_large_values() {
        let solution = setup();
        
        // Large positive and negative values
        assert_eq!(solution.max_sub_array(vec![10000, -9999, 10000]), 10001);
        
        // Single large value dominates
        assert_eq!(solution.max_sub_array(vec![-1, -2, 10000, -3, -4]), 10000);
    }

    #[test]
    fn test_patterns() {
        let solution = setup();
        
        // Mountain pattern (peak in middle) - sum is 1+2+3+2+1 = 9
        assert_eq!(solution.max_sub_array(vec![1, 2, 3, 2, 1]), 9);
        
        // Valley pattern (dip in middle)
        assert_eq!(solution.max_sub_array(vec![3, -1, 3]), 5);
        
        // Sawtooth pattern
        assert_eq!(solution.max_sub_array(vec![5, -2, 5, -2, 5]), 11);
        
        // Ascending pattern - last 3 elements: 1 + 3 + 5 = 9
        assert_eq!(solution.max_sub_array(vec![-5, -3, -1, 1, 3, 5]), 9);
    }

    #[test]
    fn test_complex_scenarios() {
        let solution = setup();
        
        // Multiple equal maximum subarrays (returns one)
        assert_eq!(solution.max_sub_array(vec![1, 2, -3, 2, 1, -3, 3]), 3);
        
        // Entire array is the maximum subarray
        assert_eq!(solution.max_sub_array(vec![1, 2, 1, 2, 1]), 7);
        
        // Maximum subarray at the beginning
        assert_eq!(solution.max_sub_array(vec![5, 4, -100, 1, 2]), 9);
        
        // Maximum subarray at the end
        assert_eq!(solution.max_sub_array(vec![1, 2, -100, 4, 5]), 9);
        
        // Maximum subarray in the middle
        assert_eq!(solution.max_sub_array(vec![-1, 2, 3, 4, -1]), 9);
    }

    #[test]
    fn test_performance_characteristics() {
        let solution = setup();
        
        // Large array with known pattern
        let mut large_array = vec![-1; 1000];
        large_array[500] = 1000; // Single large positive value
        
        let result = solution.max_sub_array(large_array);
        assert_eq!(result, 1000);
        
        // Ensure all approaches handle large inputs
        let large_input: Vec<i32> = (0..1000).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let kadane_result = solution.max_sub_array(large_input.clone());
        let dp_result = solution.max_sub_array_dp(large_input.clone());
        assert_eq!(kadane_result, dp_result);
    }
}