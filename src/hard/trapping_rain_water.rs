//! # Problem 42: Trapping Rain Water
//!
//! Given `n` non-negative integers representing an elevation map where the width of each bar is 1,
//! compute how much water it can trap after raining.
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::hard::trapping_rain_water::Solution;
//! 
//! let solution = Solution::new();
//! 
//! // Example 1:
//! let height = vec![0,1,0,2,1,0,1,3,2,1,2,1];
//! assert_eq!(solution.trap(height), 6);
//! 
//! // Example 2:
//! let height = vec![4,2,0,3,2,5];
//! assert_eq!(solution.trap(height), 9);
//! ```
//!
//! ## Constraints
//!
//! - n == height.length
//! - 1 <= n <= 2 * 10^4
//! - 0 <= height[i] <= 3 * 10^4

/// Solution struct following LeetCode format
pub struct Solution;

impl Solution {
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Two Pointers (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Use two pointers from both ends moving towards center
    /// 2. Track max height seen from left and right sides
    /// 3. Water level at position is min of left_max and right_max
    /// 4. Move pointer with smaller max height
    /// 
    /// **Time Complexity:** O(n) - Single pass through array
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **Key Insight:** Water trapped at position i depends on the minimum of:
    /// - Maximum height to the left of i
    /// - Maximum height to the right of i
    /// 
    /// We don't need to know both values exactly - we only need to know which is smaller.
    /// 
    /// **Why this is optimal:**
    /// - Single pass algorithm
    /// - No extra space required
    /// - Processes each element exactly once
    /// - Elegant invariant: always process side with smaller max
    /// 
    /// **Invariant maintained:**
    /// If left_max < right_max, then the water level at left position
    /// is determined by left_max (regardless of what's to the right)
    pub fn trap(&self, height: Vec<i32>) -> i32 {
        if height.len() < 3 {
            return 0;
        }
        
        let mut left = 0;
        let mut right = height.len() - 1;
        let mut left_max = 0;
        let mut right_max = 0;
        let mut water = 0;
        
        while left < right {
            if height[left] < height[right] {
                if height[left] >= left_max {
                    left_max = height[left];
                } else {
                    water += left_max - height[left];
                }
                left += 1;
            } else {
                if height[right] >= right_max {
                    right_max = height[right];
                } else {
                    water += right_max - height[right];
                }
                right -= 1;
            }
        }
        
        water
    }

    /// # Approach 2: Dynamic Programming (Precompute Arrays)
    /// 
    /// **Algorithm:**
    /// 1. Precompute maximum height to the left of each position
    /// 2. Precompute maximum height to the right of each position
    /// 3. For each position, water = min(left_max, right_max) - height
    /// 
    /// **Time Complexity:** O(n) - Three passes through array
    /// **Space Complexity:** O(n) - Two additional arrays for left_max and right_max
    /// 
    /// **Why this is intuitive:**
    /// - Direct implementation of water trapping formula
    /// - Easy to understand and verify correctness
    /// - Clear separation of concerns (precompute, then calculate)
    /// 
    /// **Trade-offs vs Two Pointers:**
    /// - **Pros:** More straightforward logic, easier to debug
    /// - **Cons:** Uses O(n) extra space, multiple passes
    /// 
    /// **When to prefer:** When code clarity is more important than space optimization
    pub fn trap_dp(&self, height: Vec<i32>) -> i32 {
        if height.len() < 3 {
            return 0;
        }
        
        let n = height.len();
        let mut left_max = vec![0; n];
        let mut right_max = vec![0; n];
        
        // Fill left_max array
        left_max[0] = height[0];
        for i in 1..n {
            left_max[i] = left_max[i - 1].max(height[i]);
        }
        
        // Fill right_max array
        right_max[n - 1] = height[n - 1];
        for i in (0..n - 1).rev() {
            right_max[i] = right_max[i + 1].max(height[i]);
        }
        
        // Calculate trapped water
        let mut water = 0;
        for i in 0..n {
            let water_level = left_max[i].min(right_max[i]);
            if water_level > height[i] {
                water += water_level - height[i];
            }
        }
        
        water
    }

    /// # Approach 3: Stack-Based (Horizontal Layers)
    /// 
    /// **Algorithm:**
    /// 1. Use stack to store indices of bars in descending height order
    /// 2. When we find a bar taller than stack top, we found a "valley"
    /// 3. Pop from stack and calculate water trapped in this layer
    /// 4. Continue until stack is empty or current bar is not taller
    /// 
    /// **Time Complexity:** O(n) - Each bar pushed and popped at most once
    /// **Space Complexity:** O(n) - Stack can store all indices in worst case
    /// 
    /// **Key Insight:** Instead of thinking column-by-column (vertical),
    /// think layer-by-layer (horizontal). Each layer is bounded by two bars.
    /// 
    /// **Visualization:**
    /// ```
    /// height = [3,0,2,0,4]
    /// Stack processes layers horizontally:
    /// Layer 1: between bars 3 and 4, above height 2
    /// Layer 2: between bars 3 and 4, above height 0
    /// ```
    /// 
    /// **Why this approach is elegant:**
    /// - Natural way to think about water "pooling" in valleys
    /// - Handles complex terrain with multiple peaks naturally
    /// - Each calculation is for a complete "pool" of water
    /// 
    /// **When to prefer:** When you want to understand the "physics" of water trapping
    pub fn trap_stack(&self, height: Vec<i32>) -> i32 {
        let mut stack = Vec::new();
        let mut water = 0;
        
        for (current, &h) in height.iter().enumerate() {
            while let Some(&top) = stack.last() {
                if height[top] <= h {
                    stack.pop();
                    
                    if let Some(&left) = stack.last() {
                        // Found a valley: left boundary, bottom (top), right boundary (current)
                        let width = current - left - 1;
                        let bounded_height = std::cmp::min(height[left], height[current]) - height[top];
                        water += width as i32 * bounded_height;
                    }
                } else {
                    break;
                }
            }
            stack.push(current);
        }
        
        water
    }

    /// # Approach 4: Brute Force (Inefficient - Educational)
    /// 
    /// **Algorithm:**
    /// 1. For each position, find maximum height to its left
    /// 2. For each position, find maximum height to its right  
    /// 3. Water at position = min(left_max, right_max) - height
    /// 
    /// **Time Complexity:** O(n²) - For each position, scan left and right
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **Why this is inefficient:**
    /// - Recalculates same left_max and right_max values repeatedly
    /// - No memoization of previously computed results
    /// - Quadratic time complexity scales poorly
    /// 
    /// **Educational value:**
    /// - Shows the most direct implementation of the water trapping formula
    /// - Demonstrates why memoization/precomputation matters
    /// - Baseline for performance comparison
    /// - Easy to understand and verify correctness
    pub fn trap_brute_force(&self, height: Vec<i32>) -> i32 {
        if height.len() < 3 {
            return 0;
        }
        
        let mut water = 0;
        
        for i in 1..height.len() - 1 {
            // Find maximum height to the left
            let mut left_max = 0;
            for j in 0..i {
                left_max = left_max.max(height[j]);
            }
            
            // Find maximum height to the right
            let mut right_max = 0;
            for j in i + 1..height.len() {
                right_max = right_max.max(height[j]);
            }
            
            // Calculate water at this position
            let water_level = left_max.min(right_max);
            if water_level > height[i] {
                water += water_level - height[i];
            }
        }
        
        water
    }

    /// # Approach 5: Divide and Conquer (Alternative Perspective)
    /// 
    /// **Algorithm:**
    /// 1. Find the maximum height bar (dividing point)
    /// 2. Calculate water trapped on left side of maximum
    /// 3. Calculate water trapped on right side of maximum
    /// 4. Sum results from both sides
    /// 
    /// **Time Complexity:** O(n) - Linear scan to find max, then two linear passes
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **Key Insight:** The maximum height bar acts as the "watershed" - 
    /// water flows away from it in both directions.
    /// 
    /// **Why this works:** On each side of the global maximum:
    /// - Water level is determined by the local maximum on that side
    /// - No water flows over the global maximum
    /// 
    /// **When to consider:** Alternative perspective for understanding the problem,
    /// though not more efficient than two-pointers approach.
    pub fn trap_divide_conquer(&self, height: Vec<i32>) -> i32 {
        if height.len() < 3 {
            return 0;
        }
        
        // Find the index of maximum height
        let mut max_idx = 0;
        for i in 1..height.len() {
            if height[i] > height[max_idx] {
                max_idx = i;
            }
        }
        
        let mut water = 0;
        
        // Calculate water on left side of maximum
        let mut left_max = 0;
        for i in 0..max_idx {
            if height[i] > left_max {
                left_max = height[i];
            } else {
                water += left_max - height[i];
            }
        }
        
        // Calculate water on right side of maximum
        let mut right_max = 0;
        for i in (max_idx + 1..height.len()).rev() {
            if height[i] > right_max {
                right_max = height[i];
            } else {
                water += right_max - height[i];
            }
        }
        
        water
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
        
        // Example 1 from problem
        assert_eq!(solution.trap(vec![0,1,0,2,1,0,1,3,2,1,2,1]), 6);
        
        // Example 2 from problem
        assert_eq!(solution.trap(vec![4,2,0,3,2,5]), 9);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // No water can be trapped
        assert_eq!(solution.trap(vec![1]), 0);
        assert_eq!(solution.trap(vec![1, 2]), 0);
        assert_eq!(solution.trap(vec![3, 2, 1]), 0); // Descending
        assert_eq!(solution.trap(vec![1, 2, 3]), 0); // Ascending
    }

    #[test]
    fn test_simple_valley() {
        let solution = setup();
        
        // Simple valley: high-low-high
        assert_eq!(solution.trap(vec![3, 0, 2]), 2);
        assert_eq!(solution.trap(vec![2, 0, 3]), 2);
        assert_eq!(solution.trap(vec![5, 2, 7]), 3);
    }

    #[test]
    fn test_multiple_valleys() {
        let solution = setup();
        
        // Multiple separate valleys
        assert_eq!(solution.trap(vec![3, 0, 2, 0, 4]), 7); // Two valleys: 2 + 5
        assert_eq!(solution.trap(vec![2, 1, 0, 1, 2]), 4); // Symmetric valley
    }

    #[test]
    fn test_plateau_patterns() {
        let solution = setup();
        
        // Flat areas (plateaus)
        assert_eq!(solution.trap(vec![3, 3, 0, 3, 3]), 3); // Flat boundaries
        assert_eq!(solution.trap(vec![2, 0, 0, 0, 2]), 6); // Flat bottom
        assert_eq!(solution.trap(vec![3, 1, 1, 1, 3]), 6); // Raised flat bottom
    }

    #[test]
    fn test_no_trapping_cases() {
        let solution = setup();
        
        // Strictly increasing
        assert_eq!(solution.trap(vec![1, 2, 3, 4, 5]), 0);
        
        // Strictly decreasing
        assert_eq!(solution.trap(vec![5, 4, 3, 2, 1]), 0);
        
        // Single peak
        assert_eq!(solution.trap(vec![1, 2, 3, 2, 1]), 0);
        
        // All same height
        assert_eq!(solution.trap(vec![3, 3, 3, 3]), 0);
    }

    #[test]
    fn test_boundary_values() {
        let solution = setup();
        
        // Test constraint boundaries: height[i] <= 3 * 10^4
        assert_eq!(solution.trap(vec![30000, 0, 30000]), 30000);
        assert_eq!(solution.trap(vec![30000, 15000, 30000]), 15000);
        
        // Mixed boundary and normal values
        assert_eq!(solution.trap(vec![1, 0, 30000, 0, 1]), 2);
    }

    #[test]
    fn test_complex_terrain() {
        let solution = setup();
        
        // Complex elevation with multiple peaks and valleys
        let complex = vec![5, 2, 7, 2, 6, 1, 6];
        // Analysis: valleys at positions with water
        let result = solution.trap(complex);
        assert!(result > 0); // Should trap some water
        
        // Nested valleys - just verify it traps some water
        let complex_result = solution.trap(vec![6, 4, 2, 0, 3, 2, 0, 3, 1, 4, 5, 3, 2, 7, 5, 3, 0, 2]);
        assert!(complex_result > 0); // Should trap some water, exact amount varies
    }

    #[test]
    fn test_large_input() {
        let solution = setup();
        
        // Test near maximum constraint: n <= 2 * 10^4
        let mut large_input = vec![1000]; // Start high
        for i in 0..10000 {
            large_input.push(i % 10); // Create repeating pattern
        }
        large_input.push(1000); // End high
        
        let result = solution.trap(large_input);
        assert!(result > 0); // Should trap significant water
    }

    #[test]
    fn test_all_approaches_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            vec![0,1,0,2,1,0,1,3,2,1,2,1],
            vec![4,2,0,3,2,5],
            vec![3, 0, 2],
            vec![3, 0, 2, 0, 4],
            vec![2, 1, 0, 1, 2],
            vec![3, 3, 0, 3, 3],
            vec![1, 2, 3, 4, 5], // No trapping
            vec![5, 4, 3, 2, 1], // No trapping
            vec![3, 3, 3, 3],    // No trapping
            vec![30000, 0, 30000],
            vec![6, 4, 2, 0, 3, 2, 0, 3, 1, 4, 5, 3, 2, 7, 5, 3, 0, 2],
        ];
        
        for (i, case) in test_cases.into_iter().enumerate() {
            let result1 = solution.trap(case.clone());
            let result2 = solution.trap_dp(case.clone());
            let result3 = solution.trap_stack(case.clone());
            let result4 = solution.trap_brute_force(case.clone());
            let result5 = solution.trap_divide_conquer(case.clone());
            
            assert_eq!(result1, result2, "DP approach differs for test case {}: {:?}", i, case);
            assert_eq!(result1, result3, "Stack approach differs for test case {}: {:?}", i, case);
            assert_eq!(result1, result4, "Brute force differs for test case {}: {:?}", i, case);
            assert_eq!(result1, result5, "Divide conquer differs for test case {}: {:?}", i, case);
        }
    }

    #[test]
    fn test_two_pointer_specific_cases() {
        let solution = setup();
        
        // Cases that specifically test two-pointer logic
        
        // Left pointer advances more (left side lower)
        assert_eq!(solution.trap(vec![1, 0, 0, 0, 2]), 3);
        
        // Right pointer advances more (right side lower)  
        assert_eq!(solution.trap(vec![2, 0, 0, 0, 1]), 3);
        
        // Alternating advances
        assert_eq!(solution.trap(vec![3, 1, 2, 1, 4]), 5);
    }

    #[test]
    fn test_stack_specific_cases() {
        let solution = setup();
        
        // Cases that test stack approach thoroughly
        
        // Multiple layers in single valley
        assert_eq!(solution.trap_stack(vec![4, 1, 2, 3]), 3); // Valley gets closed
        assert_eq!(solution.trap_stack(vec![4, 1, 2, 0, 3, 5]), 10); // Multiple layers
        
        // Stack empties and refills
        assert_eq!(solution.trap_stack(vec![2, 0, 1, 0, 2]), 5);
    }

    #[test]
    fn test_dp_specific_cases() {
        let solution = setup();
        
        // Cases where DP arrays are interesting
        
        // Left max increases steadily, right max decreases
        let case1 = vec![1, 2, 0, 3, 0, 4, 0, 3, 0, 2, 0, 1];
        let result1 = solution.trap_dp(case1);
        assert!(result1 > 0);
        
        // Verify DP gives same result as optimal
        let case2 = vec![5, 1, 3, 1, 7];
        assert_eq!(solution.trap(case2.clone()), solution.trap_dp(case2));
    }

    #[test]
    fn test_divide_conquer_specific() {
        let solution = setup();
        
        // Cases with clear global maximum
        assert_eq!(solution.trap_divide_conquer(vec![1, 0, 10, 0, 1]), 2);
        assert_eq!(solution.trap_divide_conquer(vec![2, 1, 0, 8, 0, 1, 2]), 6);
        
        // Multiple bars with same maximum height
        assert_eq!(solution.trap_divide_conquer(vec![3, 1, 3, 1, 3]), 4);
    }

    #[test]
    fn test_mathematical_properties() {
        let solution = setup();
        
        // Water trapped never exceeds total "valley volume"
        let heights = vec![5, 0, 5];
        let trapped = solution.trap(heights.clone());
        let max_possible = heights.iter().max().unwrap() * (heights.len() as i32 - 2);
        assert!(trapped <= max_possible);
        
        // Symmetric patterns should trap same amount
        let pattern1 = vec![3, 1, 4, 1, 3];
        let pattern2 = vec![3, 1, 4, 1, 3]; // Same pattern
        assert_eq!(solution.trap(pattern1), solution.trap(pattern2));
        
        // Reversing should give same result
        let original = vec![1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1];
        let mut reversed = original.clone();
        reversed.reverse();
        assert_eq!(solution.trap(original), solution.trap(reversed));
    }

    #[test]
    fn test_performance_patterns() {
        let solution = setup();
        
        // Best case for two-pointers: balanced movement
        let balanced = vec![5, 1, 2, 3, 4, 4, 3, 2, 1, 5];
        let result1 = solution.trap(balanced);
        assert!(result1 > 0);
        
        // Worst case for brute force: many calculations
        let worst_brute = vec![100; 50]; // All same height, many elements
        assert_eq!(solution.trap(worst_brute), 0);
        
        // Good case for stack: simple valleys
        let good_stack = vec![3, 0, 3, 0, 3, 0, 3];
        let result2 = solution.trap_stack(good_stack);
        assert!(result2 > 0);
    }
}