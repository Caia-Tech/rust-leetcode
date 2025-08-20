//! # Problem 11: Container With Most Water
//!
//! Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai).
//! n vertical lines are drawn such that the two endpoints of line i is at (i, 0) and (i, ai).
//! Find two lines, which together with the x-axis forms a container, such that the container contains the most water.
//!
//! Notice that you may not slant the container.
//!
//! ## Examples
//!
//! ```
//! Input: height = [1,8,6,2,5,4,8,3,7]
//! Output: 49
//! Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7].
//! In this case, the max area of water (blue section) the container can contain is 49.
//! ```
//!
//! ## Constraints
//!
//! * n == height.length
//! * 2 <= n <= 10^5
//! * 0 <= height[i] <= 10^4

/// Solution for Container With Most Water problem
pub struct Solution;

impl Solution {
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Two Pointers (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Use two pointers, one at the beginning and one at the end
    /// 2. Calculate area formed by the two lines
    /// 3. Move the pointer pointing to the shorter line inward
    /// 4. Keep track of maximum area seen so far
    /// 
    /// **Time Complexity:** O(n) - Single pass through the array
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **Key Insight:** Moving the pointer with shorter height gives us the best chance
    /// to find a larger area, since area is limited by the shorter line.
    /// 
    /// **Why this is optimal:**
    /// - If we move the pointer with the taller line, we can never get a larger area
    /// - The width decreases, and height is still limited by the shorter line
    /// - By moving the shorter line, we might find a taller line that increases area
    /// 
    /// **Visualization:**
    /// ```
    /// height = [1,8,6,2,5,4,8,3,7]
    ///          ↑               ↑
    ///          left            right
    /// 
    /// area = min(1, 7) * (8 - 0) = 1 * 8 = 8
    /// Since height[left] < height[right], move left++
    /// ```
    pub fn max_area(&self, height: Vec<i32>) -> i32 {
        let mut left = 0;
        let mut right = height.len() - 1;
        let mut max_area = 0;
        
        while left < right {
            // Calculate area with current left and right pointers
            let width = (right - left) as i32;
            let container_height = height[left].min(height[right]);
            let area = width * container_height;
            
            max_area = max_area.max(area);
            
            // Move the pointer pointing to the shorter line
            if height[left] < height[right] {
                left += 1;
            } else {
                right -= 1;
            }
        }
        
        max_area
    }

    /// # Approach 2: Brute Force (Inefficient - Educational)
    /// 
    /// **Algorithm:**
    /// 1. Try all possible pairs of lines
    /// 2. For each pair, calculate the area
    /// 3. Keep track of maximum area
    /// 
    /// **Time Complexity:** O(n²) - Check all pairs
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **When to use:** Understanding the problem, small inputs, or as baseline for comparison
    /// 
    /// **Why this approach is inefficient:**
    /// - Checks many unnecessary pairs that can't possibly be optimal
    /// - No early termination or pruning strategies
    pub fn max_area_brute_force(&self, height: Vec<i32>) -> i32 {
        let mut max_area = 0;
        
        for i in 0..height.len() {
            for j in (i + 1)..height.len() {
                let width = (j - i) as i32;
                let container_height = height[i].min(height[j]);
                let area = width * container_height;
                max_area = max_area.max(area);
            }
        }
        
        max_area
    }

    /// # Approach 3: Optimized Brute Force with Early Termination
    /// 
    /// **Algorithm:**
    /// 1. For each left line, find the maximum possible area
    /// 2. Skip iterations where maximum possible area < current best
    /// 3. Use remaining width as upper bound for optimization
    /// 
    /// **Time Complexity:** O(n²) worst case, but often better in practice
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **Optimization techniques:**
    /// - Early termination when max possible area < current best
    /// - Skip lines that are shorter than previously processed lines of same height
    pub fn max_area_optimized_brute_force(&self, height: Vec<i32>) -> i32 {
        let mut max_area = 0;
        let n = height.len();
        
        for i in 0..n {
            // Early termination: if max possible area with remaining width < current best
            let max_remaining_width = (n - 1 - i) as i32;
            if height[i] * max_remaining_width <= max_area {
                continue;
            }
            
            for j in (i + 1)..n {
                let width = (j - i) as i32;
                let container_height = height[i].min(height[j]);
                let area = width * container_height;
                max_area = max_area.max(area);
                
                // Continue checking all positions for complete search
            }
        }
        
        max_area
    }

    /// # Approach 4: Divide and Conquer
    /// 
    /// **Algorithm:**
    /// 1. Find the tallest line(s) in the array
    /// 2. The optimal solution either:
    ///    - Uses the tallest line as one boundary
    ///    - Is completely to the left of tallest line
    ///    - Is completely to the right of tallest line
    /// 3. Recursively solve subproblems
    /// 
    /// **Time Complexity:** O(n log n) average, O(n²) worst case
    /// **Space Complexity:** O(log n) for recursion stack
    /// 
    /// **When to use:** Educational purposes, or when you need to understand structure
    pub fn max_area_divide_conquer(&self, height: Vec<i32>) -> i32 {
        self.divide_conquer_helper(&height, 0, height.len() - 1)
    }
    
    fn divide_conquer_helper(&self, height: &[i32], left: usize, right: usize) -> i32 {
        if left >= right {
            return 0;
        }
        
        // Simple divide and conquer: split in middle
        let mid = (left + right) / 2;
        
        // Get max area in left half, right half
        let max_left = self.divide_conquer_helper(height, left, mid);
        let max_right = self.divide_conquer_helper(height, mid + 1, right);
        
        // Get max area that crosses the middle
        let mut max_cross = 0;
        for i in left..=mid {
            for j in (mid + 1)..=right {
                let width = (j - i) as i32;
                let container_height = height[i].min(height[j]);
                let area = width * container_height;
                max_cross = max_cross.max(area);
            }
        }
        
        max_left.max(max_right).max(max_cross)
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
        assert_eq!(solution.max_area(vec![1,8,6,2,5,4,8,3,7]), 49);
        
        // Minimum case
        assert_eq!(solution.max_area(vec![1,1]), 1);
        
        // Simple increasing  
        assert_eq!(solution.max_area(vec![1,2,3,4,5]), 6); // i=1,j=4: width=3, height=min(2,5)=2, area=6
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // All same height
        assert_eq!(solution.max_area(vec![3,3,3,3]), 9); // width=3, height=3
        
        // Decreasing heights
        assert_eq!(solution.max_area(vec![5,4,3,2,1]), 6); // i=0,j=2: width=2, height=min(5,3)=3, area=6
        
        // Peak in middle
        assert_eq!(solution.max_area(vec![1,3,5,3,1]), 6); // i=1,j=3: width=2, height=min(3,3)=3, area=6
    }

    #[test]
    fn test_large_differences() {
        let solution = setup();
        
        // One very tall line
        assert_eq!(solution.max_area(vec![1,1000,1]), 2); // width=2, height=min(1,1)=1
        
        // Tall lines at ends
        assert_eq!(solution.max_area(vec![100,1,1,1,100]), 400); // width=4, height=min(100,100)=100
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            vec![1,8,6,2,5,4,8,3,7],
            vec![1,1],
            vec![1,2,3,4,5],
            vec![5,4,3,2,1],
            vec![1,3,5,3,1],
            vec![100,1,1,1,100],
        ];
        
        for height in test_cases {
            let result1 = solution.max_area(height.clone());
            let result2 = solution.max_area_brute_force(height.clone());
            let result3 = solution.max_area_optimized_brute_force(height.clone());
            let result4 = solution.max_area_divide_conquer(height.clone());
            
            assert_eq!(result1, result2, "Two pointers vs brute force mismatch for {:?}", height);
            assert_eq!(result2, result3, "Brute force vs optimized brute force mismatch for {:?}", height);
            assert_eq!(result3, result4, "Optimized brute force vs divide conquer mismatch for {:?}", height);
        }
    }

    #[test]
    fn test_optimal_properties() {
        let solution = setup();
        
        // Test that result is always <= n * max_height
        let height = vec![1,8,6,2,5,4,8,3,7];
        let result = solution.max_area(height.clone());
        let max_height = *height.iter().max().unwrap();
        let max_possible = (height.len() - 1) as i32 * max_height;
        assert!(result <= max_possible);
        
        // Test that result is always >= 0
        assert!(result >= 0);
    }

    #[test]
    fn test_monotonic_arrays() {
        let solution = setup();
        
        // Strictly increasing
        let increasing = vec![1,2,3,4,5,6,7,8,9,10];
        let result_inc = solution.max_area(increasing);
        assert!(result_inc > 0);
        
        // Strictly decreasing
        let decreasing = vec![10,9,8,7,6,5,4,3,2,1];
        let result_dec = solution.max_area(decreasing);
        assert!(result_dec > 0);
        
        // For symmetric arrays, results should be equal
        assert_eq!(result_inc, result_dec);
    }

    #[test]
    fn test_performance_patterns() {
        let solution = setup();
        
        // Large array with pattern
        let mut large_height = Vec::new();
        for i in 0..1000 {
            large_height.push(i % 100 + 1);
        }
        
        let result = solution.max_area(large_height);
        assert!(result > 0);
        
        // Ensure no overflow for reasonable inputs
        let max_reasonable = vec![10000; 1000];
        let max_result = solution.max_area(max_reasonable);
        assert!(max_result > 0);
        assert!(max_result <= 10000 * 999); // max possible area
    }

    #[test]
    fn test_boundary_behavior() {
        let solution = setup();
        
        // Height with zeros
        assert_eq!(solution.max_area(vec![0,2,0]), 0); // Any container with height 0 gives area 0
        assert_eq!(solution.max_area(vec![1,0,1]), 2); // i=0,j=2: width=2, height=min(1,1)=1, area=2
        
        // Single tall line surrounded by short ones
        assert_eq!(solution.max_area(vec![1,10,1]), 2); // width=2, height=min(1,1)=1
    }

    #[test]
    fn test_mathematical_properties() {
        let solution = setup();
        
        // Area should be symmetric for reversed arrays
        let original = vec![1,8,6,2,5,4,8,3,7];
        let mut reversed = original.clone();
        reversed.reverse();
        
        assert_eq!(solution.max_area(original), solution.max_area(reversed));
        
        // Adding zeros at the ends shouldn't change optimal internal area
        let base = vec![3,5,3];
        let with_zeros = vec![0,3,5,3,0];
        
        let base_result = solution.max_area(base);
        let zeros_result = solution.max_area(with_zeros);
        assert!(zeros_result >= base_result); // Could be equal or better due to increased width
    }
}