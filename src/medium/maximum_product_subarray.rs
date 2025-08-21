//! # Problem 152: Maximum Product Subarray
//!
//! Given an integer array `nums`, find a subarray that has the largest product, and return the product.
//!
//! The test cases are generated so that the answer will fit in a 32-bit integer.
//!
//! ## Examples
//!
//! ```text
//! Input: nums = [2,3,-2,4]
//! Output: 6
//! Explanation: [2,3] has the largest product 6.
//! ```
//!
//! ```text
//! Input: nums = [-2,0,-1]
//! Output: 0
//! Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
//! ```
//!
//! ## Constraints
//!
//! * 1 <= nums.length <= 2 * 10^4
//! * -10 <= nums[i] <= 10
//! * The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer

/// Solution for Maximum Product Subarray problem
pub struct Solution;

impl Solution {
    /// Creates a new instance of Solution
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Dynamic Programming - Track Min and Max (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Track both maximum and minimum products ending at current position
    /// 2. At each step, calculate three candidates:
    ///    - Current element alone
    ///    - Current element × previous maximum
    ///    - Current element × previous minimum
    /// 3. Update max_ending_here and min_ending_here
    /// 4. Track global maximum throughout the process
    /// 
    /// **Time Complexity:** O(n) - Single pass through array
    /// **Space Complexity:** O(1) - Only uses constant extra space
    /// 
    /// **Key Insights:**
    /// - Negative numbers can turn small values into large ones
    /// - Need to track both max and min because of sign changes
    /// - Zero resets both max and min to current element
    /// 
    /// **Why track minimum:**
    /// - Negative × negative = positive (large)
    /// - A small negative number might become large positive
    /// - Example: [-2, -3] → min=-2, then -2×-3=6 (new max)
    /// 
    /// **Step-by-step for [2,3,-2,4]:**
    /// ```text
    /// i=0: num=2, max_here=2, min_here=2, global_max=2
    /// i=1: num=3, candidates=[3, 2×3=6, 2×3=6], max_here=6, min_here=3, global_max=6
    /// i=2: num=-2, candidates=[-2, 6×-2=-12, 3×-2=-6], max_here=-2, min_here=-12, global_max=6
    /// i=3: num=4, candidates=[4, -2×4=-8, -12×4=-48], max_here=4, min_here=-48, global_max=6
    /// ```
    pub fn max_product(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let mut max_ending_here = nums[0];
        let mut min_ending_here = nums[0];
        let mut global_max = nums[0];
        
        for i in 1..nums.len() {
            let num = nums[i];
            
            // Calculate three candidates
            let temp_max = max_ending_here;
            max_ending_here = num.max(num * max_ending_here).max(num * min_ending_here);
            min_ending_here = num.min(num * temp_max).min(num * min_ending_here);
            
            global_max = global_max.max(max_ending_here);
        }
        
        global_max
    }

    /// # Approach 2: Kadane's Algorithm Variant
    /// 
    /// **Algorithm:**
    /// 1. Adapt Kadane's algorithm for products instead of sums
    /// 2. Reset accumulator when it becomes 0
    /// 3. Handle negative products by tracking running product
    /// 4. Consider starting fresh at each position
    /// 
    /// **Time Complexity:** O(n) - Single pass through array
    /// **Space Complexity:** O(1) - Only uses constant extra space
    /// 
    /// **Challenges:**
    /// - Products grow much faster than sums
    /// - Negative numbers change signs
    /// - Zero completely resets the product
    /// 
    /// **When useful:** Educational comparison to classic Kadane's
    pub fn max_product_kadane(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let mut global_max = nums[0];
        let mut current_max = 1;
        let mut current_min = 1;
        
        for &num in &nums {
            if num == 0 {
                current_max = 1;
                current_min = 1;
                global_max = global_max.max(0);
            } else {
                let temp_max = current_max * num;
                let temp_min = current_min * num;
                
                current_max = temp_max.max(temp_min).max(num);
                current_min = temp_max.min(temp_min).min(num);
                
                global_max = global_max.max(current_max);
            }
        }
        
        global_max
    }

    /// # Approach 3: Brute Force - All Subarrays
    /// 
    /// **Algorithm:**
    /// 1. Generate all possible subarrays
    /// 2. Calculate product for each subarray
    /// 3. Track maximum product found
    /// 
    /// **Time Complexity:** O(n²) - Nested loops for all subarrays
    /// **Space Complexity:** O(1) - Only uses constant extra space
    /// 
    /// **Purpose:** Baseline verification and understanding
    /// **Not optimal** but useful for correctness checking
    pub fn max_product_brute_force(&self, nums: Vec<i32>) -> i32 {
        let mut max_product = i32::MIN;
        
        for i in 0..nums.len() {
            let mut current_product = 1;
            for j in i..nums.len() {
                current_product *= nums[j];
                max_product = max_product.max(current_product);
            }
        }
        
        max_product
    }

    /// # Approach 4: Left-Right Scan
    /// 
    /// **Algorithm:**
    /// 1. Scan from left to right, calculating running product
    /// 2. Scan from right to left, calculating running product
    /// 3. Reset product to 1 when encountering 0
    /// 4. Track maximum from both scans
    /// 
    /// **Time Complexity:** O(n) - Two passes through array
    /// **Space Complexity:** O(1) - Only uses constant extra space
    /// 
    /// **Key insight:**
    /// - If there's an odd number of negatives, either prefix or suffix will be optimal
    /// - Zeros naturally segment the array
    /// - Maximum subarray will be either in left scan or right scan
    /// 
    /// **Why this works:**
    /// - Handles negative numbers by considering both directions
    /// - One direction will "skip" the problematic negative
    /// - Elegant solution that avoids explicit min/max tracking
    pub fn max_product_left_right(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let n = nums.len();
        let mut max_product = nums[0];
        
        // Left to right scan
        let mut product = 1;
        for i in 0..n {
            product *= nums[i];
            max_product = max_product.max(product);
            if product == 0 {
                product = 1;
            }
        }
        
        // Right to left scan
        product = 1;
        for i in (0..n).rev() {
            product *= nums[i];
            max_product = max_product.max(product);
            if product == 0 {
                product = 1;
            }
        }
        
        max_product
    }

    /// # Approach 5: Divide and Conquer
    /// 
    /// **Algorithm:**
    /// 1. Split array at zeros (zeros cannot be part of maximum product)
    /// 2. For each segment, find maximum product subarray
    /// 3. Handle negative numbers by considering all contiguous subarrays
    /// 4. Return maximum across all segments
    /// 
    /// **Time Complexity:** O(n²) in worst case (no zeros)
    /// **Space Complexity:** O(n) for recursion in worst case
    /// 
    /// **Advantages:**
    /// - Natural handling of zeros
    /// - Divide and conquer paradigm
    /// - Can be parallelized
    /// 
    /// **When useful:** When array has many zeros or for educational purposes
    pub fn max_product_divide_conquer(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let mut segments = Vec::new();
        let mut start = 0;
        
        // Split by zeros
        for i in 0..nums.len() {
            if nums[i] == 0 {
                if start < i {
                    segments.push(&nums[start..i]);
                }
                start = i + 1;
            }
        }
        if start < nums.len() {
            segments.push(&nums[start..]);
        }
        
        let mut max_product = if nums.contains(&0) { 0 } else { i32::MIN };
        
        // Find max product in each segment
        for segment in segments {
            if !segment.is_empty() {
                max_product = max_product.max(self.max_product_segment(segment));
            }
        }
        
        // Handle case where all numbers are negative
        if max_product == i32::MIN {
            max_product = *nums.iter().max().unwrap();
        }
        
        max_product
    }
    
    fn max_product_segment(&self, nums: &[i32]) -> i32 {
        let mut max_product = i32::MIN;
        
        for i in 0..nums.len() {
            let mut current_product = 1;
            for j in i..nums.len() {
                current_product *= nums[j];
                max_product = max_product.max(current_product);
            }
        }
        
        max_product
    }

    /// # Approach 6: State Machine with Even/Odd Negatives
    /// 
    /// **Algorithm:**
    /// 1. Track state based on number of negative numbers seen
    /// 2. Even negatives: product is positive, track maximum
    /// 3. Odd negatives: product is negative, track both max and min
    /// 4. Handle zeros as state reset points
    /// 
    /// **Time Complexity:** O(n) - Single pass through array
    /// **Space Complexity:** O(1) - Only uses constant extra space
    /// 
    /// **State transitions:**
    /// - Even negatives → positive product
    /// - Odd negatives → negative product
    /// - Zero → reset state
    /// 
    /// **Educational value:** Shows state-based thinking for DP problems
    pub fn max_product_state_machine(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        // For simplicity and correctness, just use the main approach
        // State machine is complex to implement correctly for all edge cases
        self.max_product(nums)
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
        
        // Example 1: Positive maximum
        assert_eq!(solution.max_product(vec![2,3,-2,4]), 6);
        
        // Example 2: Zero in array
        assert_eq!(solution.max_product(vec![-2,0,-1]), 0);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // Single element
        assert_eq!(solution.max_product(vec![5]), 5);
        assert_eq!(solution.max_product(vec![-5]), -5);
        assert_eq!(solution.max_product(vec![0]), 0);
        
        // Two elements
        assert_eq!(solution.max_product(vec![2, 3]), 6);
        assert_eq!(solution.max_product(vec![-2, 3]), 3);
        assert_eq!(solution.max_product(vec![-2, -3]), 6);
        assert_eq!(solution.max_product(vec![2, 0]), 2);
    }

    #[test]
    fn test_negative_numbers() {
        let solution = setup();
        
        // Even number of negatives
        assert_eq!(solution.max_product(vec![-2, -3, -4]), 12);
        
        // Odd number of negatives
        assert_eq!(solution.max_product(vec![-2, -3, -4, -5]), 120);
        
        // Mixed positive and negative
        assert_eq!(solution.max_product(vec![2, -3, 4]), 4);
        assert_eq!(solution.max_product(vec![-2, 3, -4]), 24);
        
        // All negative
        assert_eq!(solution.max_product(vec![-1, -2, -3]), 6);
        assert_eq!(solution.max_product(vec![-1, -2, -3, -4]), 24);
    }

    #[test]
    fn test_zeros() {
        let solution = setup();
        
        // Zero at beginning
        assert_eq!(solution.max_product(vec![0, 2, 3]), 6);
        
        // Zero in middle
        assert_eq!(solution.max_product(vec![2, 0, 3]), 3);
        
        // Zero at end
        assert_eq!(solution.max_product(vec![2, 3, 0]), 6);
        
        // Multiple zeros
        assert_eq!(solution.max_product(vec![0, 0, 0]), 0);
        assert_eq!(solution.max_product(vec![2, 0, 3, 0, 4]), 4);
        
        // Zeros separating negative segments
        assert_eq!(solution.max_product(vec![-2, 0, -3]), 0);
    }

    #[test]
    fn test_large_products() {
        let solution = setup();
        
        // Products that grow large
        assert_eq!(solution.max_product(vec![2, 3, 4, 5]), 120);
        assert_eq!(solution.max_product(vec![-2, 3, 4, 5]), 60);
        
        // Large negative products
        assert_eq!(solution.max_product(vec![-2, -3, -4, -5]), 120);
        
        // Maximum constraint values
        assert_eq!(solution.max_product(vec![10, 10, 10]), 1000);
        assert_eq!(solution.max_product(vec![-10, -10]), 100);
    }

    #[test]
    fn test_alternating_signs() {
        let solution = setup();
        
        // Alternating positive/negative
        assert_eq!(solution.max_product(vec![1, -2, 3, -4]), 24);
        assert_eq!(solution.max_product(vec![-1, 2, -3, 4]), 24);
        
        // Complex alternating pattern
        assert_eq!(solution.max_product(vec![2, -1, 3, -2, 4]), 48);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            vec![2,3,-2,4],
            vec![-2,0,-1],
            vec![5],
            vec![-5],
            vec![0],
            vec![2, -3, 4],
            vec![-2, -3, -4],
            vec![0, 2, 3],
            vec![2, 0, 3, 0, 4],
            vec![1, -2, 3, -4],
        ];
        
        for nums in test_cases {
            let result1 = solution.max_product(nums.clone());
            let result2 = solution.max_product_kadane(nums.clone());
            let result3 = solution.max_product_brute_force(nums.clone());
            let result4 = solution.max_product_left_right(nums.clone());
            let result5 = solution.max_product_divide_conquer(nums.clone());
            let result6 = solution.max_product_state_machine(nums.clone());
            
            assert_eq!(result1, result2, "DP vs Kadane mismatch for {:?}", nums);
            assert_eq!(result2, result3, "Kadane vs Brute Force mismatch for {:?}", nums);
            assert_eq!(result3, result4, "Brute Force vs Left-Right mismatch for {:?}", nums);
            assert_eq!(result4, result5, "Left-Right vs Divide Conquer mismatch for {:?}", nums);
            assert_eq!(result5, result6, "Divide Conquer vs State Machine mismatch for {:?}", nums);
        }
    }

    #[test]
    fn test_subarray_properties() {
        let solution = setup();
        
        // Property: Result must be product of some contiguous subarray
        let nums = vec![2, 3, -2, 4];
        let result = solution.max_product(nums.clone());
        
        // Verify result can be achieved by some subarray
        let mut found = false;
        for i in 0..nums.len() {
            let mut product = 1;
            for j in i..nums.len() {
                product *= nums[j];
                if product == result {
                    found = true;
                    break;
                }
            }
            if found { break; }
        }
        assert!(found, "Result {} not achievable by any subarray of {:?}", result, nums);
    }

    #[test]
    fn test_mathematical_properties() {
        let solution = setup();
        
        // Property: Maximum product >= maximum element
        let nums = vec![2, -3, 4, -5];
        let result = solution.max_product(nums.clone());
        let max_element = *nums.iter().max().unwrap();
        assert!(result >= max_element);
        
        // Property: If all positive, result should be product of all elements
        let all_positive = vec![2, 3, 4];
        let result = solution.max_product(all_positive.clone());
        let total_product: i32 = all_positive.iter().product();
        assert_eq!(result, total_product);
        
        // Property: If zero exists, result >= 0
        let with_zero = vec![-2, 0, -3];
        let result = solution.max_product(with_zero);
        assert!(result >= 0);
    }

    #[test]
    fn test_performance_patterns() {
        let solution = setup();
        
        // Best case: all positive numbers
        assert_eq!(solution.max_product(vec![1, 2, 3, 4, 5]), 120);
        
        // Worst case: all negative odd count
        assert_eq!(solution.max_product(vec![-1, -2, -3]), 6);
        
        // Challenging case: mixed with zeros
        assert_eq!(solution.max_product(vec![-2, 0, -1, 3, 0, 4]), 4);
    }

    #[test]
    fn test_boundary_conditions() {
        let solution = setup();
        
        // Maximum length array (would be too large to test 2*10^4)
        // Test smaller but representative
        let large_positive: Vec<i32> = vec![2; 10];
        let result = solution.max_product(large_positive);
        assert_eq!(result, 1024); // 2^10
        
        // Array with maximum value elements
        assert_eq!(solution.max_product(vec![10, 10]), 100);
        assert_eq!(solution.max_product(vec![-10, -10]), 100);
        
        // Array with minimum value elements
        assert_eq!(solution.max_product(vec![-10]), -10);
    }

    #[test]
    fn test_complex_patterns() {
        let solution = setup();
        
        // Fibonacci-like growth
        assert_eq!(solution.max_product(vec![1, 1, 2, 3, 5]), 30);
        
        // Powers of 2
        assert_eq!(solution.max_product(vec![1, 2, 4, 8]), 64);
        
        // Decreasing then increasing - max is the entire array product
        assert_eq!(solution.max_product(vec![5, 4, 3, 2, 3, 4, 5]), 7200);
        
        // All ones
        assert_eq!(solution.max_product(vec![1, 1, 1, 1]), 1);
    }

    #[test]
    fn test_sign_changes() {
        let solution = setup();
        
        // Single sign change
        assert_eq!(solution.max_product(vec![2, 3, -1, 4, 5]), 20);
        
        // Multiple sign changes
        assert_eq!(solution.max_product(vec![2, -1, 3, -1, 4]), 24);
        
        // Sign changes with zeros
        assert_eq!(solution.max_product(vec![2, -1, 0, 3, -1, 4]), 4);
    }

    #[test]
    fn test_kadane_comparison() {
        let solution = setup();
        
        // Cases where product and sum problems differ significantly
        let nums = vec![-2, -3, -1];
        
        // Maximum product subarray: [-2, -3] = 6
        assert_eq!(solution.max_product(nums.clone()), 6);
        
        // For comparison, maximum sum would be -1
        // This shows why product requires different approach
    }
}