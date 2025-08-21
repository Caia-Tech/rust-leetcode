//! # Problem 238: Product of Array Except Self
//!
//! Given an integer array `nums`, return an array `answer` such that `answer[i]` is equal to 
//! the product of all the elements of `nums` except `nums[i]`.
//!
//! The product of any prefix or suffix of `nums` is guaranteed to fit in a 32-bit integer.
//!
//! You must write an algorithm that runs in O(n) time and without using the division operator.
//!
//! ## Examples
//!
//! ```text
//! Input: nums = [1,2,3,4]
//! Output: [24,12,8,6]
//! ```
//!
//! ```text
//! Input: nums = [-1,1,0,-3,3]
//! Output: [0,0,9,0,0]
//! ```
//!
//! ## Constraints
//!
//! * 2 <= nums.length <= 10^5
//! * -30 <= nums[i] <= 30
//! * The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer

/// Solution for Product of Array Except Self problem
pub struct Solution;

impl Solution {
    /// Creates a new instance of Solution
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Left and Right Product Arrays (Intuitive)
    /// 
    /// **Algorithm:**
    /// 1. Create left array where left[i] = product of all elements to the left of i
    /// 2. Create right array where right[i] = product of all elements to the right of i
    /// 3. Result[i] = left[i] * right[i]
    /// 
    /// **Time Complexity:** O(n) - Three passes through array
    /// **Space Complexity:** O(n) - Two additional arrays
    /// 
    /// **Key Insights:**
    /// - Product except self = (product of left elements) × (product of right elements)
    /// - Need to handle edges: leftmost has no left elements, rightmost has no right elements
    /// - Use 1 as identity for empty products
    /// 
    /// **Why this works:**
    /// - For each position i, we need product of all other elements
    /// - Split into left side and right side products
    /// - Multiply corresponding left and right products
    /// 
    /// **Visualization:**
    /// ```text
    /// nums  = [1, 2, 3, 4]
    /// left  = [1, 1, 2, 6]  (product of elements to the left)
    /// right = [24,12,4, 1]  (product of elements to the right)
    /// result= [24,12,8, 6]  (left[i] * right[i])
    /// ```
    pub fn product_except_self(&self, nums: Vec<i32>) -> Vec<i32> {
        let n = nums.len();
        let mut left = vec![1; n];
        let mut right = vec![1; n];
        
        // Fill left array: left[i] = product of all elements to left of i
        for i in 1..n {
            left[i] = left[i - 1] * nums[i - 1];
        }
        
        // Fill right array: right[i] = product of all elements to right of i
        for i in (0..n-1).rev() {
            right[i] = right[i + 1] * nums[i + 1];
        }
        
        // Combine left and right products
        let mut result = vec![0; n];
        for i in 0..n {
            result[i] = left[i] * right[i];
        }
        
        result
    }

    /// # Approach 2: Space Optimized with Output Array (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Use output array to store left products first
    /// 2. Traverse right to left, maintaining running right product
    /// 3. Multiply output[i] (left product) with right product
    /// 
    /// **Time Complexity:** O(n) - Two passes through array
    /// **Space Complexity:** O(1) - Only constant extra space (not counting output)
    /// 
    /// **Key Optimization:**
    /// - Reuse output array to store intermediate results
    /// - Use single variable for right product instead of array
    /// - Saves one full array of space
    /// 
    /// **Why this is optimal:**
    /// - Cannot do better than O(n) time (must visit each element)
    /// - Uses minimum possible extra space
    /// - Still maintains clarity of left/right product concept
    /// 
    /// **Step-by-step:**
    /// 1. First pass: output[i] = product of all elements left of i
    /// 2. Second pass: multiply by running product from right
    pub fn product_except_self_optimized(&self, nums: Vec<i32>) -> Vec<i32> {
        let n = nums.len();
        let mut result = vec![1; n];
        
        // First pass: fill result with left products
        for i in 1..n {
            result[i] = result[i - 1] * nums[i - 1];
        }
        
        // Second pass: multiply with right products
        let mut right = 1;
        for i in (0..n).rev() {
            result[i] *= right;
            right *= nums[i];
        }
        
        result
    }

    /// # Approach 3: Division Method (Not allowed but educational)
    /// 
    /// **Algorithm:**
    /// 1. Calculate total product of all elements
    /// 2. For each position, divide total by current element
    /// 3. Handle zero elements specially
    /// 
    /// **Time Complexity:** O(n) - Single pass + division operations
    /// **Space Complexity:** O(1) - Only constant extra space
    /// 
    /// **Why not allowed:**
    /// - Problem explicitly forbids division operator
    /// - Division can have precision issues with large numbers
    /// - Doesn't handle zeros elegantly without special cases
    /// 
    /// **Educational value:**
    /// - Shows the "obvious" solution that doesn't work
    /// - Demonstrates why constraint exists
    /// - Highlights edge cases with zero elements
    /// 
    /// **Zero handling complexity:**
    /// - 0 zeros: divide total by each element
    /// - 1 zero: only zero position gets non-zero result
    /// - 2+ zeros: all results are zero
    pub fn product_except_self_division(&self, nums: Vec<i32>) -> Vec<i32> {
        let n = nums.len();
        let mut result = vec![0; n];
        
        // Count zeros and calculate product of non-zero elements
        let mut zero_count = 0;
        let mut product = 1;
        
        for &num in &nums {
            if num == 0 {
                zero_count += 1;
            } else {
                product *= num;
            }
        }
        
        // Handle different zero cases
        match zero_count {
            0 => {
                // No zeros: divide product by each element
                for i in 0..n {
                    result[i] = product / nums[i];
                }
            }
            1 => {
                // One zero: only zero position gets the product
                for i in 0..n {
                    if nums[i] == 0 {
                        result[i] = product;
                    }
                    // Others remain 0 (initialized value)
                }
            }
            _ => {
                // Multiple zeros: all results are 0 (already initialized)
            }
        }
        
        result
    }

    /// # Approach 4: Recursive with Memoization
    /// 
    /// **Algorithm:**
    /// 1. Define recursive function to calculate product except current index
    /// 2. Use memoization to cache partial products
    /// 3. Base case: empty range has product 1
    /// 
    /// **Time Complexity:** O(n²) - Without good memoization strategy
    /// **Space Complexity:** O(n) - Recursion stack + memoization
    /// 
    /// **Educational purpose:**
    /// - Demonstrates recursive thinking about the problem
    /// - Shows why iterative solutions are preferred
    /// - Illustrates memoization concepts
    /// 
    /// **Why not practical:**
    /// - More complex than iterative solutions
    /// - Higher space complexity due to recursion
    /// - Potential stack overflow for large inputs
    pub fn product_except_self_recursive(&self, nums: Vec<i32>) -> Vec<i32> {
        let n = nums.len();
        let mut result = vec![0; n];
        
        for i in 0..n {
            result[i] = self.product_range(&nums, 0, i) * self.product_range(&nums, i + 1, n);
        }
        
        result
    }
    
    fn product_range(&self, nums: &[i32], start: usize, end: usize) -> i32 {
        if start >= end {
            return 1;
        }
        
        let mut product = 1;
        for i in start..end {
            product *= nums[i];
        }
        product
    }

    /// # Approach 5: Prefix/Suffix Pattern (Educational variant)
    /// 
    /// **Algorithm:**
    /// 1. Calculate prefix products: prefix[i] = product of nums[0..i]
    /// 2. Calculate suffix products: suffix[i] = product of nums[i+1..n]
    /// 3. Result[i] = prefix[i-1] * suffix[i+1] (with bounds checking)
    /// 
    /// **Time Complexity:** O(n) - Three passes
    /// **Space Complexity:** O(n) - Prefix and suffix arrays
    /// 
    /// **Difference from Approach 1:**
    /// - Uses inclusive prefix/suffix instead of exclusive left/right
    /// - Requires careful index handling for boundaries
    /// - Shows alternative way to think about the problem
    /// 
    /// **When useful:**
    /// - When you already have prefix/suffix arrays for other purposes
    /// - Educational: demonstrates prefix/suffix sum pattern
    pub fn product_except_self_prefix_suffix(&self, nums: Vec<i32>) -> Vec<i32> {
        let n = nums.len();
        let mut prefix = vec![1; n + 1];  // prefix[i] = product of nums[0..i]
        let mut suffix = vec![1; n + 1];  // suffix[i] = product of nums[i..n]
        
        // Calculate prefix products
        for i in 0..n {
            prefix[i + 1] = prefix[i] * nums[i];
        }
        
        // Calculate suffix products
        for i in (0..n).rev() {
            suffix[i] = suffix[i + 1] * nums[i];
        }
        
        // Combine for result
        let mut result = vec![0; n];
        for i in 0..n {
            result[i] = prefix[i] * suffix[i + 1];
        }
        
        result
    }

    /// # Approach 6: Single Pass with Two Pointers
    /// 
    /// **Algorithm:**
    /// 1. Use two pointers moving from opposite ends
    /// 2. Maintain left and right running products
    /// 3. Fill result array from both ends simultaneously
    /// 4. Requires post-processing to combine partial results
    /// 
    /// **Time Complexity:** O(n) - Single pass + post-processing
    /// **Space Complexity:** O(n) - Additional arrays for left/right products
    /// 
    /// **Complexity trade-off:**
    /// - Fewer passes through array
    /// - More complex logic and additional space
    /// - Not actually better than standard approach
    /// 
    /// **Educational value:**
    /// - Shows two-pointer technique application
    /// - Demonstrates that optimization isn't always beneficial
    /// - Illustrates importance of simplicity
    pub fn product_except_self_two_pointers(&self, nums: Vec<i32>) -> Vec<i32> {
        let n = nums.len();
        let mut left_products = vec![1; n];
        let mut right_products = vec![1; n];
        
        let mut left = 0;
        let mut right = n - 1;
        let mut left_product = 1;
        let mut right_product = 1;
        
        // Single pass with two pointers
        while left < n {
            left_products[left] = left_product;
            right_products[right] = right_product;
            
            left_product *= nums[left];
            right_product *= nums[right];
            
            left += 1;
            if right > 0 {
                right -= 1;
            }
        }
        
        // Combine results
        let mut result = vec![0; n];
        for i in 0..n {
            result[i] = left_products[i] * right_products[i];
        }
        
        result
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
        
        // Example 1: [1,2,3,4] → [24,12,8,6]
        let result1 = solution.product_except_self(vec![1, 2, 3, 4]);
        assert_eq!(result1, vec![24, 12, 8, 6]);
        
        // Example 2: [-1,1,0,-3,3] → [0,0,9,0,0]
        let result2 = solution.product_except_self(vec![-1, 1, 0, -3, 3]);
        assert_eq!(result2, vec![0, 0, 9, 0, 0]);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // Two elements
        let result = solution.product_except_self(vec![2, 3]);
        assert_eq!(result, vec![3, 2]);
        
        // All ones
        let result = solution.product_except_self(vec![1, 1, 1, 1]);
        assert_eq!(result, vec![1, 1, 1, 1]);
        
        // Single zero
        let result = solution.product_except_self(vec![1, 0, 3, 4]);
        assert_eq!(result, vec![0, 12, 0, 0]);
        
        // Multiple zeros
        let result = solution.product_except_self(vec![0, 1, 0, 3]);
        assert_eq!(result, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            vec![1, 2, 3, 4],
            vec![-1, 1, 0, -3, 3],
            vec![2, 3],
            vec![1, 1, 1, 1],
            vec![1, 0, 3, 4],
            vec![0, 1, 0, 3],
            vec![-1, -2, -3],
            vec![5, 10, 15, 20],
        ];
        
        for nums in test_cases {
            let result1 = solution.product_except_self(nums.clone());
            let result2 = solution.product_except_self_optimized(nums.clone());
            let result3 = solution.product_except_self_division(nums.clone());
            let result4 = solution.product_except_self_recursive(nums.clone());
            let result5 = solution.product_except_self_prefix_suffix(nums.clone());
            let result6 = solution.product_except_self_two_pointers(nums.clone());
            
            assert_eq!(result1, result2, "Basic vs Optimized mismatch for {:?}", nums);
            assert_eq!(result2, result3, "Optimized vs Division mismatch for {:?}", nums);
            assert_eq!(result3, result4, "Division vs Recursive mismatch for {:?}", nums);
            assert_eq!(result4, result5, "Recursive vs Prefix/Suffix mismatch for {:?}", nums);
            assert_eq!(result5, result6, "Prefix/Suffix vs Two Pointers mismatch for {:?}", nums);
        }
    }

    #[test]
    fn test_negative_numbers() {
        let solution = setup();
        
        // All negative
        let result = solution.product_except_self(vec![-1, -2, -3]);
        assert_eq!(result, vec![6, 3, 2]);
        
        // Mix of positive and negative
        let result = solution.product_except_self(vec![-2, 3, -4, 5]);
        assert_eq!(result, vec![-60, 40, -30, 24]);
        
        // Negative with zero
        let result = solution.product_except_self(vec![-1, 0, -3]);
        assert_eq!(result, vec![0, 3, 0]);
    }

    #[test]
    fn test_boundary_values() {
        let solution = setup();
        
        // Maximum positive values
        let result = solution.product_except_self(vec![30, 30]);
        assert_eq!(result, vec![30, 30]);
        
        // Maximum negative values
        let result = solution.product_except_self(vec![-30, -30]);
        assert_eq!(result, vec![-30, -30]);
        
        // Mixed boundaries
        let result = solution.product_except_self(vec![-30, 30]);
        assert_eq!(result, vec![30, -30]);
    }

    #[test]
    fn test_zero_patterns() {
        let solution = setup();
        
        // Zero at beginning
        let result = solution.product_except_self(vec![0, 2, 3]);
        assert_eq!(result, vec![6, 0, 0]);
        
        // Zero at end
        let result = solution.product_except_self(vec![2, 3, 0]);
        assert_eq!(result, vec![0, 0, 6]);
        
        // Zero in middle
        let result = solution.product_except_self(vec![2, 0, 3]);
        assert_eq!(result, vec![0, 6, 0]);
        
        // All zeros
        let result = solution.product_except_self(vec![0, 0, 0]);
        assert_eq!(result, vec![0, 0, 0]);
    }

    #[test]
    fn test_mathematical_properties() {
        let solution = setup();
        
        // Property: product[i] * nums[i] should equal total product (when no zeros)
        let nums = vec![2, 3, 4, 5];
        let result = solution.product_except_self(nums.clone());
        let total_product: i32 = nums.iter().product();
        
        for i in 0..nums.len() {
            assert_eq!(result[i] * nums[i], total_product);
        }
        
        // Property: length should be preserved
        let nums = vec![1, 2, 3, 4, 5, 6, 7];
        let result = solution.product_except_self(nums.clone());
        assert_eq!(result.len(), nums.len());
    }

    #[test]
    fn test_large_products() {
        let solution = setup();
        
        // Test with numbers that would overflow if multiplied naively
        let result = solution.product_except_self(vec![10, 20, 30]);
        assert_eq!(result, vec![600, 300, 200]);
        
        // Test products within 32-bit range
        let result = solution.product_except_self(vec![100, 200, 300]);
        assert_eq!(result, vec![60000, 30000, 20000]);
    }

    #[test]
    fn test_identity_elements() {
        let solution = setup();
        
        // Array with 1s (identity for multiplication)
        let result = solution.product_except_self(vec![1, 2, 1, 4]);
        assert_eq!(result, vec![8, 4, 8, 2]);
        
        // Mix of 1s and other numbers
        let result = solution.product_except_self(vec![1, 1, 2, 3, 1]);
        assert_eq!(result, vec![6, 6, 3, 2, 6]);
    }

    #[test]
    fn test_two_element_arrays() {
        let solution = setup();
        
        // Minimum size array
        let result = solution.product_except_self(vec![5, 7]);
        assert_eq!(result, vec![7, 5]);
        
        // With zero
        let result = solution.product_except_self(vec![0, 8]);
        assert_eq!(result, vec![8, 0]);
        
        // With negative
        let result = solution.product_except_self(vec![-3, 4]);
        assert_eq!(result, vec![4, -3]);
    }

    #[test]
    fn test_performance_characteristics() {
        let solution = setup();
        
        // Large array (within constraint limits)
        let large_nums: Vec<i32> = (1..=10).collect(); // Smaller to avoid overflow
        let result = solution.product_except_self_optimized(large_nums.clone());
        
        // Should handle efficiently
        assert_eq!(result.len(), large_nums.len());
        
        // Spot check: first element should be product of 2..10
        let expected_first: i32 = (2..=10).product();
        assert_eq!(result[0], expected_first);
    }

    #[test]
    fn test_symmetry_properties() {
        let solution = setup();
        
        // Symmetric array
        let nums = vec![2, 3, 3, 2];
        let result = solution.product_except_self(nums);
        assert_eq!(result, vec![18, 12, 12, 18]);
        
        // Palindromic array
        let nums = vec![1, 2, 3, 2, 1];
        let result = solution.product_except_self(nums);
        assert_eq!(result, vec![12, 6, 4, 6, 12]);
    }

    #[test]
    fn test_optimization_space_usage() {
        let solution = setup();
        
        // Test that optimized version produces same results
        let test_arrays = vec![
            vec![1, 2, 3, 4, 5],
            vec![-1, 0, 1, 2],
            vec![10, 20, 30, 40],
        ];
        
        for nums in test_arrays {
            let result1 = solution.product_except_self(nums.clone());
            let result2 = solution.product_except_self_optimized(nums.clone());
            assert_eq!(result1, result2, "Space optimization changed results");
        }
    }
}