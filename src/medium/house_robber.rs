//! # Problem 198: House Robber
//!
//! You are a professional robber planning to rob houses along a street. Each house has a certain 
//! amount of money stashed, the only constraint stopping you from robbing each of them is that 
//! adjacent houses have security systems connected and it will automatically contact the police 
//! if two adjacent houses were broken into on the same night.
//!
//! Given an integer array `nums` representing the amount of money of each house, return the 
//! maximum amount of money you can rob tonight without alerting the police.
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

    /// # Approach 1: Dynamic Programming with Space Optimization (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. At each house, decide: rob it or skip it
    /// 2. If rob current: take current + max from 2 houses back
    /// 3. If skip current: take max from previous house
    /// 4. Track only last two values for space optimization
    /// 
    /// **Time Complexity:** O(n) - Single pass through houses
    /// **Space Complexity:** O(1) - Only two variables needed
    /// 
    /// **Key Insight:** 
    /// - At house i, we have two choices:
    ///   1. Rob it: nums[i] + dp[i-2] (can't rob i-1)
    ///   2. Skip it: dp[i-1] (take best up to i-1)
    /// - Take maximum of these two choices
    /// 
    /// **Why this works:**
    /// - We can't rob adjacent houses
    /// - So if we rob house i, we must skip i-1
    /// - This creates optimal substructure for DP
    /// 
    /// **Visualization:**
    /// ```text
    /// Houses: [2, 7, 9, 3, 1]
    /// i=0: rob=2, skip=0 → max=2
    /// i=1: rob=7+0=7, skip=2 → max=7
    /// i=2: rob=9+2=11, skip=7 → max=11
    /// i=3: rob=3+7=10, skip=11 → max=11
    /// i=4: rob=1+11=12, skip=11 → max=12
    /// ```
    pub fn rob(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        if nums.len() == 1 {
            return nums[0];
        }
        
        let mut prev2 = 0;  // Max money up to i-2
        let mut prev1 = nums[0];  // Max money up to i-1
        
        for i in 1..nums.len() {
            let current = (nums[i] + prev2).max(prev1);
            prev2 = prev1;
            prev1 = current;
        }
        
        prev1
    }

    /// # Approach 2: Dynamic Programming with Array
    /// 
    /// **Algorithm:**
    /// 1. Create dp array where dp[i] = max money up to house i
    /// 2. Base: dp[0] = nums[0], dp[1] = max(nums[0], nums[1])
    /// 3. Recurrence: dp[i] = max(nums[i] + dp[i-2], dp[i-1])
    /// 4. Return dp[n-1]
    /// 
    /// **Time Complexity:** O(n) - Fill dp array once
    /// **Space Complexity:** O(n) - Store entire dp array
    /// 
    /// **DP State Definition:**
    /// - dp[i] = maximum money that can be robbed from houses 0 to i
    /// 
    /// **Recurrence Relation:**
    /// - dp[i] = max(nums[i] + dp[i-2], dp[i-1])
    /// - Either rob house i and add to best from i-2, or skip i
    /// 
    /// **When to use:** When you need to track all intermediate values
    pub fn rob_dp_array(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        if nums.len() == 1 {
            return nums[0];
        }
        
        let n = nums.len();
        let mut dp = vec![0; n];
        dp[0] = nums[0];
        dp[1] = nums[0].max(nums[1]);
        
        for i in 2..n {
            dp[i] = (nums[i] + dp[i - 2]).max(dp[i - 1]);
        }
        
        dp[n - 1]
    }

    /// # Approach 3: Recursive with Memoization (Top-Down DP)
    /// 
    /// **Algorithm:**
    /// 1. Define recursive function: rob(i) = max money from houses 0 to i
    /// 2. Base cases: i < 0 returns 0, i = 0 returns nums[0]
    /// 3. Recurrence: rob(i) = max(nums[i] + rob(i-2), rob(i-1))
    /// 4. Use memoization to avoid redundant calculations
    /// 
    /// **Time Complexity:** O(n) - Each subproblem solved once
    /// **Space Complexity:** O(n) - Memoization table + recursion stack
    /// 
    /// **Why memoization is essential:**
    /// - Without it, we'd solve same subproblems multiple times
    /// - Time complexity would be exponential O(2^n)
    /// - Memoization ensures linear time
    /// 
    /// **When to use:** When problem naturally fits recursive thinking
    pub fn rob_memo(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let mut memo = vec![-1; nums.len()];
        self.rob_memo_helper(&nums, nums.len() as i32 - 1, &mut memo)
    }
    
    fn rob_memo_helper(&self, nums: &[i32], i: i32, memo: &mut Vec<i32>) -> i32 {
        if i < 0 {
            return 0;
        }
        
        let idx = i as usize;
        if memo[idx] != -1 {
            return memo[idx];
        }
        
        let rob_current = nums[idx] + self.rob_memo_helper(nums, i - 2, memo);
        let skip_current = self.rob_memo_helper(nums, i - 1, memo);
        
        memo[idx] = rob_current.max(skip_current);
        memo[idx]
    }

    /// # Approach 4: Iterative with Even/Odd Pattern
    /// 
    /// **Algorithm:**
    /// 1. Track max for even and odd positioned houses separately
    /// 2. Update based on whether current index is even or odd
    /// 3. Return maximum of both trackers
    /// 
    /// **Time Complexity:** O(n) - Single pass
    /// **Space Complexity:** O(1) - Only two variables
    /// 
    /// **Key Insight:**
    /// - Track maximum considering even/odd patterns
    /// - Helps visualize the alternating selection pattern
    /// 
    /// **When to use:** Alternative way to think about the problem
    pub fn rob_even_odd(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let mut include = 0;  // Max money if we include current house
        let mut exclude = 0;  // Max money if we exclude current house
        
        for num in nums {
            let temp = include;
            include = exclude + num;
            exclude = exclude.max(temp);
        }
        
        include.max(exclude)
    }

    /// # Approach 5: Modified Kadane's Algorithm Style
    /// 
    /// **Algorithm:**
    /// 1. Similar to max subarray, but with non-adjacent constraint
    /// 2. Track max ending at current position with constraint
    /// 3. Decision at each step: take current or previous best
    /// 
    /// **Time Complexity:** O(n) - Single pass
    /// **Space Complexity:** O(1) - Constant extra space
    /// 
    /// **Connection to Kadane's:**
    /// - Both problems involve making optimal local decisions
    /// - Difference: adjacency constraint vs continuity constraint
    /// - Shows how DP patterns can be adapted
    /// 
    /// **Mathematical formulation:**
    /// - Let f(i) = max money robbing houses ending at or before i
    /// - f(i) = max(f(i-1), nums[i] + f(i-2))
    pub fn rob_kadane_style(&self, nums: Vec<i32>) -> i32 {
        nums.iter().fold((0, 0), |(prev2, prev1), &num| {
            (prev1, prev1.max(prev2 + num))
        }).1
    }

    /// # Approach 6: Bottom-Up with State Machine
    /// 
    /// **Algorithm:**
    /// 1. Model as state machine with two states: robbed/not-robbed
    /// 2. Transition: can rob if previous wasn't robbed
    /// 3. Track best value for each state at each position
    /// 
    /// **Time Complexity:** O(n) - Process each house once
    /// **Space Complexity:** O(1) - Only track current states
    /// 
    /// **State Machine Design:**
    /// ```text
    /// States: ROBBED, NOT_ROBBED
    /// Transitions:
    /// - ROBBED → NOT_ROBBED (must skip next)
    /// - NOT_ROBBED → ROBBED or NOT_ROBBED
    /// ```
    /// 
    /// **Why state machines are useful:**
    /// - Clear representation of constraints
    /// - Easy to extend for variations
    /// - Natural for problems with states/transitions
    pub fn rob_state_machine(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let mut robbed = nums[0];     // Max if we robbed current
        let mut not_robbed = 0;       // Max if we didn't rob current
        
        for i in 1..nums.len() {
            let new_robbed = not_robbed + nums[i];
            let new_not_robbed = robbed.max(not_robbed);
            
            robbed = new_robbed;
            not_robbed = new_not_robbed;
        }
        
        robbed.max(not_robbed)
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
        
        // Example 1: [1,2,3,1] → 4
        assert_eq!(solution.rob(vec![1, 2, 3, 1]), 4);
        
        // Example 2: [2,7,9,3,1] → 12
        assert_eq!(solution.rob(vec![2, 7, 9, 3, 1]), 12);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // Single house
        assert_eq!(solution.rob(vec![5]), 5);
        
        // Two houses
        assert_eq!(solution.rob(vec![1, 2]), 2);
        assert_eq!(solution.rob(vec![2, 1]), 2);
        
        // All same values
        assert_eq!(solution.rob(vec![5, 5, 5, 5]), 10);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            vec![1, 2, 3, 1],
            vec![2, 7, 9, 3, 1],
            vec![2, 1, 1, 2],
            vec![5, 1, 3, 9, 4],
            vec![1, 3, 1, 3, 100],
            vec![10, 1, 1, 10, 1],
        ];
        
        for nums in test_cases {
            let result1 = solution.rob(nums.clone());
            let result2 = solution.rob_dp_array(nums.clone());
            let result3 = solution.rob_memo(nums.clone());
            let result4 = solution.rob_even_odd(nums.clone());
            let result5 = solution.rob_kadane_style(nums.clone());
            let result6 = solution.rob_state_machine(nums.clone());
            
            assert_eq!(result1, result2, "Mismatch for {:?}", nums);
            assert_eq!(result2, result3, "Mismatch for {:?}", nums);
            assert_eq!(result3, result4, "Mismatch for {:?}", nums);
            assert_eq!(result4, result5, "Mismatch for {:?}", nums);
            assert_eq!(result5, result6, "Mismatch for {:?}", nums);
        }
    }

    #[test]
    fn test_alternating_pattern() {
        let solution = setup();
        
        // Perfect alternating: rob every other house
        assert_eq!(solution.rob(vec![1, 10, 1, 10, 1]), 20);
        assert_eq!(solution.rob(vec![10, 1, 10, 1, 10]), 30);
        
        // Not always alternating is optimal
        assert_eq!(solution.rob(vec![1, 100, 1, 1, 100]), 200);
    }

    #[test]
    fn test_increasing_values() {
        let solution = setup();
        
        // Strictly increasing
        assert_eq!(solution.rob(vec![1, 2, 3, 4, 5]), 9); // 1+3+5
        
        // Large increase
        assert_eq!(solution.rob(vec![1, 2, 4, 8, 16]), 21); // 1+4+16
    }

    #[test]
    fn test_decreasing_values() {
        let solution = setup();
        
        // Strictly decreasing
        assert_eq!(solution.rob(vec![5, 4, 3, 2, 1]), 9); // 5+3+1
        
        // Large decrease
        assert_eq!(solution.rob(vec![100, 50, 25, 12, 6]), 131); // 100+25+6
    }

    #[test]
    fn test_zero_values() {
        let solution = setup();
        
        // With zeros
        assert_eq!(solution.rob(vec![0, 0, 0, 0, 0]), 0);
        assert_eq!(solution.rob(vec![1, 0, 1, 0, 1]), 3);
        assert_eq!(solution.rob(vec![0, 2, 0, 4, 0]), 6);
    }

    #[test]
    fn test_large_values() {
        let solution = setup();
        
        // Max constraint value
        assert_eq!(solution.rob(vec![400]), 400);
        assert_eq!(solution.rob(vec![400, 400]), 400);
        assert_eq!(solution.rob(vec![400, 400, 400]), 800);
        
        // Mix of large and small
        assert_eq!(solution.rob(vec![1, 400, 1, 400, 1]), 800);
    }

    #[test]
    fn test_optimal_selection() {
        let solution = setup();
        
        // Case where skipping multiple houses is optimal
        assert_eq!(solution.rob(vec![2, 1, 1, 2, 100]), 103); // 2+1+100
        
        // Case where greedy doesn't work
        assert_eq!(solution.rob(vec![5, 1, 2, 10]), 15); // 5+10, not 5+2
    }

    #[test]
    fn test_mathematical_properties() {
        let solution = setup();
        
        // Property: result >= max(nums)
        let nums = vec![3, 8, 4, 7, 2];
        let result = solution.rob(nums.clone());
        let max_val = *nums.iter().max().unwrap();
        assert!(result >= max_val);
        
        // Property: result <= sum of alternating elements
        let alt_sum1 = nums.iter().step_by(2).sum::<i32>();
        let alt_sum2 = nums.iter().skip(1).step_by(2).sum::<i32>();
        assert!(result >= alt_sum1.min(alt_sum2));
    }

    #[test]
    fn test_longer_sequences() {
        let solution = setup();
        
        // Longer sequence
        let nums = vec![2, 3, 2, 3, 2, 3, 2, 3, 2, 3];
        assert_eq!(solution.rob(nums), 15); // All 3s
        
        // Random-like pattern
        let nums = vec![7, 1, 1, 5, 8, 2, 1, 7, 9, 1];
        assert_eq!(solution.rob(nums), 26); // optimal selection
    }

    #[test]
    fn test_boundary_conditions() {
        let solution = setup();
        
        // Min length (1)
        assert_eq!(solution.rob(vec![10]), 10);
        
        // Max values at boundaries
        assert_eq!(solution.rob(vec![400, 1, 1, 400]), 800);
        
        // Optimal includes first and last
        assert_eq!(solution.rob(vec![10, 1, 1, 1, 10]), 21);
    }
}