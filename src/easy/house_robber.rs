//! # Problem 198: House Robber
//!
//! You are a professional robber planning to rob houses along a street. Each house has a certain amount 
//! of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses 
//! have security systems connected and it will automatically contact the police if two adjacent houses 
//! were broken into on the same night.
//!
//! Given an integer array `nums` representing the amount of money of each house, return the maximum amount 
//! of money you can rob tonight without alerting the police.
//!
//! ## Examples
//!
//! ```text
//! Input: nums = [1,2,3,1]
//! Output: 4
//! Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
//! Total amount you can rob = 1 + 3 = 4.
//! ```
//!
//! ```text
//! Input: nums = [2,7,9,3,1]
//! Output: 12
//! Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
//! Total amount you can rob = 2 + 9 + 1 = 12.
//! ```
//!
//! ## Constraints
//!
//! * 1 <= nums.length <= 100
//! * 0 <= nums[i] <= 400

/// Solution for House Robber problem
pub struct Solution;

impl Solution {
    /// Creates a new instance of Solution
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Dynamic Programming - Space Optimized (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Track two states: max money if we rob current house vs. if we don't
    /// 2. For each house, decide whether to rob it or skip it
    /// 3. If rob current: prev_no_rob + current_money
    /// 4. If skip current: max(prev_rob, prev_no_rob)
    /// 5. Update states and continue
    /// 
    /// **Time Complexity:** O(n) - Single pass through houses
    /// **Space Complexity:** O(1) - Only uses constant extra space
    /// 
    /// **Key Insights:**
    /// - Each house decision depends only on previous two choices
    /// - Don't need to store entire DP array, just previous states
    /// - Can think of it as "rob" vs "don't rob" at each step
    /// 
    /// **Why this works:**
    /// - If we rob house i, we can't rob house i-1, so we take best from i-2
    /// - If we don't rob house i, we take the best from i-1
    /// - This gives us optimal substructure for DP
    /// 
    /// **Step-by-step for [2,7,9,3,1]:**
    /// ```text
    /// i=0: house=2, rob=2, no_rob=0
    /// i=1: house=7, rob=0+7=7, no_rob=max(2,0)=2
    /// i=2: house=9, rob=2+9=11, no_rob=max(7,2)=7
    /// i=3: house=3, rob=7+3=10, no_rob=max(11,7)=11
    /// i=4: house=1, rob=11+1=12, no_rob=max(10,11)=11
    /// Result: max(12,11)=12
    /// ```
    pub fn rob(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let mut rob = 0;        // Max money if we rob the current house
        let mut no_rob = 0;     // Max money if we don't rob the current house
        
        for money in nums {
            let new_rob = no_rob + money;  // Rob current house
            let new_no_rob = rob.max(no_rob);  // Don't rob current house
            
            rob = new_rob;
            no_rob = new_no_rob;
        }
        
        rob.max(no_rob)
    }

    /// # Approach 2: Classic DP with Array
    /// 
    /// **Algorithm:**
    /// 1. Create DP array where dp[i] = max money robbing houses 0..i
    /// 2. For each house i, dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    /// 3. Return dp[n-1]
    /// 
    /// **Time Complexity:** O(n) - Fill DP array once
    /// **Space Complexity:** O(n) - DP array storage
    /// 
    /// **Advantages:**
    /// - Clear visualization of subproblem solutions
    /// - Easy to understand and implement
    /// - Can trace back optimal solution if needed
    /// 
    /// **When to use:** When you need to reconstruct the solution path
    pub fn rob_dp_array(&self, nums: Vec<i32>) -> i32 {
        let n = nums.len();
        if n == 0 { return 0; }
        if n == 1 { return nums[0]; }
        
        let mut dp = vec![0; n];
        dp[0] = nums[0];
        dp[1] = nums[0].max(nums[1]);
        
        for i in 2..n {
            dp[i] = dp[i-1].max(dp[i-2] + nums[i]);
        }
        
        dp[n-1]
    }

    /// # Approach 3: Recursion with Memoization
    /// 
    /// **Algorithm:**
    /// 1. Define recursive function: rob_from(index) = max money from index onwards
    /// 2. Two choices at each house: rob it or skip it
    /// 3. Use memoization to avoid recomputing subproblems
    /// 4. Base cases: index >= length returns 0
    /// 
    /// **Time Complexity:** O(n) - Each subproblem computed once
    /// **Space Complexity:** O(n) - Memoization table + recursion stack
    /// 
    /// **Recurrence relation:**
    /// rob_from(i) = max(nums[i] + rob_from(i+2), rob_from(i+1))
    /// 
    /// **Educational value:** Shows top-down DP approach
    pub fn rob_memoized(&self, nums: Vec<i32>) -> i32 {
        let mut memo = vec![-1; nums.len()];
        self.rob_from_memo(&nums, 0, &mut memo)
    }
    
    fn rob_from_memo(&self, nums: &[i32], index: usize, memo: &mut Vec<i32>) -> i32 {
        if index >= nums.len() {
            return 0;
        }
        
        if memo[index] != -1 {
            return memo[index];
        }
        
        // Two choices: rob current house or skip it
        let rob_current = nums[index] + self.rob_from_memo(nums, index + 2, memo);
        let skip_current = self.rob_from_memo(nums, index + 1, memo);
        
        memo[index] = rob_current.max(skip_current);
        memo[index]
    }

    /// # Approach 4: Pure Recursion (Exponential - For Educational Purpose)
    /// 
    /// **Algorithm:**
    /// 1. At each house, recursively try both options
    /// 2. Rob current house + solve for houses starting at i+2
    /// 3. Skip current house + solve for houses starting at i+1
    /// 4. Return maximum of both options
    /// 
    /// **Time Complexity:** O(2^n) - Exponential due to overlapping subproblems
    /// **Space Complexity:** O(n) - Recursion stack depth
    /// 
    /// **Purpose:** Shows naive approach and why memoization is needed
    /// **Not practical** for large inputs due to exponential time
    pub fn rob_recursive(&self, nums: Vec<i32>) -> i32 {
        self.rob_from_recursive(&nums, 0)
    }
    
    fn rob_from_recursive(&self, nums: &[i32], index: usize) -> i32 {
        if index >= nums.len() {
            return 0;
        }
        
        // Two choices: rob current house or skip it
        let rob_current = nums[index] + self.rob_from_recursive(nums, index + 2);
        let skip_current = self.rob_from_recursive(nums, index + 1);
        
        rob_current.max(skip_current)
    }

    /// # Approach 5: Even/Odd Index Analysis
    /// 
    /// **Algorithm:**
    /// 1. Separately calculate max sum of even-indexed houses
    /// 2. Separately calculate max sum of odd-indexed houses
    /// 3. Return maximum of both sums
    /// 
    /// **Time Complexity:** O(n) - Single pass through array
    /// **Space Complexity:** O(1) - Only uses constant extra space
    /// 
    /// **Limitation:** This approach is incorrect for the general case
    /// **Why it fails:** Optimal solution might not be all evens or all odds
    /// **Example:** [2,1,1,9] - optimal is [2,9]=11, not evens [2,1]=3 or odds [1,9]=10
    /// 
    /// **Educational value:** Shows why greedy approaches often fail
    pub fn rob_even_odd_incorrect(&self, nums: Vec<i32>) -> i32 {
        let mut even_sum = 0;
        let mut odd_sum = 0;
        
        for (i, &money) in nums.iter().enumerate() {
            if i % 2 == 0 {
                even_sum += money;
            } else {
                odd_sum += money;
            }
        }
        
        even_sum.max(odd_sum)
    }

    /// # Approach 6: Finite State Machine
    /// 
    /// **Algorithm:**
    /// 1. Model as state machine with two states: "just robbed" and "didn't rob"
    /// 2. Track maximum money in each state
    /// 3. Transition between states based on current decision
    /// 4. Return maximum money from either final state
    /// 
    /// **Time Complexity:** O(n) - Single pass through houses
    /// **Space Complexity:** O(1) - Only uses constant extra space
    /// 
    /// **States:**
    /// - robbed_prev: Maximum money if we robbed the previous house
    /// - not_robbed_prev: Maximum money if we didn't rob the previous house
    /// 
    /// **Transitions:**
    /// - To rob current: must come from not_robbed_prev state
    /// - To not rob current: can come from either state (take max)
    pub fn rob_state_machine(&self, nums: Vec<i32>) -> i32 {
        let mut robbed_prev = 0;      // Max money ending with robbing previous house
        let mut not_robbed_prev = 0;  // Max money ending without robbing previous house
        
        for money in nums {
            let robbed_curr = not_robbed_prev + money;
            let not_robbed_curr = robbed_prev.max(not_robbed_prev);
            
            robbed_prev = robbed_curr;
            not_robbed_prev = not_robbed_curr;
        }
        
        robbed_prev.max(not_robbed_prev)
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
        
        // Example 1: Small array
        assert_eq!(solution.rob(vec![1,2,3,1]), 4);
        
        // Example 2: Longer array
        assert_eq!(solution.rob(vec![2,7,9,3,1]), 12);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // Single house
        assert_eq!(solution.rob(vec![5]), 5);
        
        // Two houses
        assert_eq!(solution.rob(vec![1, 2]), 2);
        assert_eq!(solution.rob(vec![2, 1]), 2);
        
        // Three houses
        assert_eq!(solution.rob(vec![1, 2, 3]), 4);  // Rob houses 0 and 2
        assert_eq!(solution.rob(vec![5, 1, 2]), 7);  // Rob houses 0 and 2
        assert_eq!(solution.rob(vec![1, 5, 2]), 5);  // Rob house 1 only
    }

    #[test]
    fn test_all_zeros() {
        let solution = setup();
        
        // All houses have no money
        assert_eq!(solution.rob(vec![0, 0, 0, 0]), 0);
        
        // Mix of zeros and values
        assert_eq!(solution.rob(vec![0, 1, 0, 3, 0]), 4);
        assert_eq!(solution.rob(vec![5, 0, 0, 1]), 6);
    }

    #[test]
    fn test_increasing_sequence() {
        let solution = setup();
        
        // Strictly increasing
        assert_eq!(solution.rob(vec![1, 2, 3, 4, 5]), 9);  // Rob 1,3,5
        
        // Non-strictly increasing
        assert_eq!(solution.rob(vec![1, 2, 2, 4, 5]), 8);  // Rob 1,2,5 or 2,4
    }

    #[test]
    fn test_decreasing_sequence() {
        let solution = setup();
        
        // Strictly decreasing
        assert_eq!(solution.rob(vec![5, 4, 3, 2, 1]), 9);  // Rob 5,3,1
        
        // Large first value
        assert_eq!(solution.rob(vec![10, 1, 1, 1]), 11);  // Rob houses 0 and 2
    }

    #[test]
    fn test_alternating_pattern() {
        let solution = setup();
        
        // High-low-high pattern
        assert_eq!(solution.rob(vec![5, 1, 3, 1, 4]), 12);  // Rob 5,3,4
        
        // Low-high-low pattern
        assert_eq!(solution.rob(vec![1, 5, 1, 5, 1]), 10);  // Rob both 5s
    }

    #[test]
    fn test_boundary_values() {
        let solution = setup();
        
        // Maximum values within constraints
        assert_eq!(solution.rob(vec![400]), 400);
        assert_eq!(solution.rob(vec![400, 400]), 400);
        assert_eq!(solution.rob(vec![400, 400, 400]), 800);
        
        // Maximum length array (100 houses)
        let max_houses = vec![1; 100];
        assert_eq!(solution.rob(max_houses), 50);  // Rob every other house
        
        let alternating = (0..100).map(|i| if i % 2 == 0 { 2 } else { 1 }).collect();
        assert_eq!(solution.rob(alternating), 100);  // Rob all even indices (50 houses * 2)
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            vec![1,2,3,1],
            vec![2,7,9,3,1],
            vec![5],
            vec![1, 2],
            vec![2, 1],
            vec![1, 2, 3],
            vec![0, 0, 0, 0],
            vec![5, 1, 3, 1, 4],
            vec![400, 400, 400],
        ];
        
        for nums in test_cases {
            let result1 = solution.rob(nums.clone());
            let result2 = solution.rob_dp_array(nums.clone());
            let result3 = solution.rob_memoized(nums.clone());
            let result4 = solution.rob_recursive(nums.clone());
            let result5 = solution.rob_state_machine(nums.clone());
            
            assert_eq!(result1, result2, "Space Optimized vs DP Array mismatch for {:?}", nums);
            assert_eq!(result2, result3, "DP Array vs Memoized mismatch for {:?}", nums);
            assert_eq!(result3, result4, "Memoized vs Recursive mismatch for {:?}", nums);
            assert_eq!(result4, result5, "Recursive vs State Machine mismatch for {:?}", nums);
            
            // Note: We don't test even_odd_incorrect as it's intentionally wrong
        }
    }

    #[test]
    fn test_optimal_substructure() {
        let solution = setup();
        
        // Verify that optimal solution has optimal substructure
        let nums = vec![2, 7, 9, 3, 1];
        let result = solution.rob(nums.clone());
        
        // The optimal solution should be better than any greedy approach
        let sum_evens = nums.iter().step_by(2).sum::<i32>();  // 2 + 9 + 1 = 12
        let sum_odds = nums[1..].iter().step_by(2).sum::<i32>();  // 7 + 3 = 10
        
        assert!(result >= sum_evens.max(sum_odds));
        assert_eq!(result, 12);  // Should equal the better of the two
    }

    #[test]
    fn test_mathematical_properties() {
        let solution = setup();
        
        // Property: Result should be >= maximum single house value
        let nums = vec![1, 3, 2, 5, 1];
        let result = solution.rob(nums.clone());
        let max_single = *nums.iter().max().unwrap();
        assert!(result >= max_single);
        
        // Property: Result should be <= sum of all houses
        let total_sum: i32 = nums.iter().sum();
        assert!(result <= total_sum);
        
        // Property: For two houses, result should be max of the two
        assert_eq!(solution.rob(vec![3, 8]), 8);
        assert_eq!(solution.rob(vec![8, 3]), 8);
    }

    #[test]
    fn test_performance_patterns() {
        let solution = setup();
        
        // Best case: all even indices have large values
        let best_case = vec![100, 1, 100, 1, 100];
        assert_eq!(solution.rob(best_case), 300);
        
        // Challenging case: need to skip profitable adjacent houses
        assert_eq!(solution.rob(vec![5, 5, 10, 100, 10, 5]), 110);
        
        // Dense optimal case
        assert_eq!(solution.rob(vec![2, 1, 1, 9, 1, 1, 2]), 13);  // Rob 2, 9, 2
    }

    #[test]
    fn test_dynamic_programming_property() {
        let solution = setup();
        
        // Test that DP builds optimal solution incrementally
        let nums = vec![2, 1, 1, 9];
        
        // Manually trace DP evolution
        // dp[0] = 2
        // dp[1] = max(2, 1) = 2
        // dp[2] = max(2, 2+1) = 3
        // dp[3] = max(3, 2+9) = 11
        
        assert_eq!(solution.rob(nums), 11);
    }

    #[test]
    fn test_greedy_failure_cases() {
        let solution = setup();
        
        // Case where even/odd greedy fails
        let nums = vec![2, 1, 1, 9];
        let result = solution.rob(nums.clone());
        let even_sum = 2 + 1; // indices 0, 2
        let odd_sum = 1 + 9;  // indices 1, 3
        let greedy_result = even_sum.max(odd_sum);
        
        assert!(result >= greedy_result);
        assert_eq!(result, 11);  // Should be 2 + 9, not 1 + 9
        
        // The greedy approach would give 10, but optimal is 11
        assert_eq!(solution.rob_even_odd_incorrect(nums), 10);
        assert!(result > solution.rob_even_odd_incorrect(vec![2, 1, 1, 9]));
    }

    #[test]
    fn test_recursion_correctness() {
        let solution = setup();
        
        // Test with small arrays to ensure recursion works
        // (avoiding large arrays due to exponential complexity)
        let small_tests = vec![
            vec![1],
            vec![1, 2],
            vec![1, 2, 3],
            vec![2, 1, 1, 9],
            vec![5, 1, 3, 1, 4],
        ];
        
        for nums in small_tests {
            let dp_result = solution.rob(nums.clone());
            let recursive_result = solution.rob_recursive(nums.clone());
            assert_eq!(dp_result, recursive_result, "DP vs Recursive mismatch for {:?}", nums);
        }
    }

    #[test]
    fn test_state_transitions() {
        let solution = setup();
        
        // Test that state machine correctly models the problem
        let nums = vec![2, 7, 9, 3, 1];
        
        // State machine should produce same result as DP
        let dp_result = solution.rob(nums.clone());
        let state_result = solution.rob_state_machine(nums);
        
        assert_eq!(dp_result, state_result);
    }
}