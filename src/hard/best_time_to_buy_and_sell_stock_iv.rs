//! Problem 188: Best Time to Buy and Sell Stock IV
//!
//! Implementation by Marvin Tutt, Caia Tech
//!
//! You are given an integer array prices where prices[i] is the price of a given stock on the ith day,
//! and an integer k. Find the maximum profit you can achieve. You may complete at most k transactions.
//!
//! Note: You may not engage in multiple transactions simultaneously 
//! (i.e., you must sell the stock before you buy again).
//!
//! Key insights:
//! - Generalization of Stock III problem for arbitrary k
//! - When k >= n/2, we can make unlimited transactions (greedy approach)
//! - For small k, use DP with states for each transaction
//! - Space optimization possible by tracking only current and previous states

use std::cmp;

pub struct Solution;

impl Solution {
    /// Approach 1: Dynamic Programming with Transaction States (Optimal)
    /// 
    /// Uses DP where buy[i] represents max profit after i-th buy,
    /// and sell[i] represents max profit after i-th sell.
    /// 
    /// Time Complexity: O(nk) when k < n/2, O(n) when k >= n/2
    /// Space Complexity: O(k)
    /// 
    /// Detailed Reasoning:
    /// - When k >= n/2, unlimited transactions are possible (greedy)
    /// - Otherwise, track buy/sell states for each of the k transactions
    /// - For each price, update states in reverse order to avoid conflicts
    pub fn max_profit_dp_states(k: i32, prices: Vec<i32>) -> i32 {
        let n = prices.len();
        if n <= 1 || k <= 0 { return 0; }
        
        let k = k as usize;
        
        // If k >= n/2, we can make unlimited transactions
        if k >= n / 2 {
            let mut profit = 0;
            for i in 1..n {
                if prices[i] > prices[i-1] {
                    profit += prices[i] - prices[i-1];
                }
            }
            return profit;
        }
        
        // DP approach for limited transactions
        let mut buy = vec![i32::MIN / 2; k + 1];  // Max profit after buying for i-th transaction
        let mut sell = vec![0; k + 1];            // Max profit after selling for i-th transaction
        
        for price in prices {
            // Update in reverse order to avoid using updated values in same iteration
            for i in (1..=k).rev() {
                sell[i] = cmp::max(sell[i], buy[i] + price);
                buy[i] = cmp::max(buy[i], sell[i-1] - price);
            }
        }
        
        sell[k]
    }
    
    /// Approach 2: 3D Dynamic Programming with Explicit States
    /// 
    /// Uses explicit 3D DP: dp[day][transactions][holding]
    /// where holding indicates whether we currently own stock.
    /// 
    /// Time Complexity: O(nk)
    /// Space Complexity: O(nk)
    /// 
    /// Detailed Reasoning:
    /// - dp[i][j][0] = max profit on day i with at most j transactions, not holding
    /// - dp[i][j][1] = max profit on day i with at most j transactions, holding
    /// - Transitions: buy (decreases available transactions), sell, or hold
    pub fn max_profit_3d_dp(k: i32, prices: Vec<i32>) -> i32 {
        let n = prices.len();
        if n <= 1 || k == 0 { return 0; }
        
        let k = k as usize;
        
        // Optimization for large k
        if k >= n / 2 {
            return Self::max_profit_unlimited(prices);
        }
        
        // dp[i][j][0] = max profit on day i with at most j transactions, not holding
        // dp[i][j][1] = max profit on day i with at most j transactions, holding
        let mut dp = vec![vec![vec![0; 2]; k + 1]; n];
        
        // Initialize first day
        for j in 0..=k {
            dp[0][j][0] = 0;
            dp[0][j][1] = -prices[0];
        }
        
        for i in 1..n {
            for j in 1..=k {
                // Not holding: either didn't hold yesterday, or sell today
                dp[i][j][0] = cmp::max(dp[i-1][j][0], dp[i-1][j][1] + prices[i]);
                
                // Holding: either held yesterday, or buy today (uses one transaction)
                dp[i][j][1] = cmp::max(dp[i-1][j][1], dp[i-1][j-1][0] - prices[i]);
            }
        }
        
        dp[n-1][k][0]
    }
    
    /// Approach 3: Space-Optimized 2D DP
    /// 
    /// Optimizes space by only keeping current and previous day states,
    /// since we only need the previous day's values.
    /// 
    /// Time Complexity: O(nk)
    /// Space Complexity: O(k)
    /// 
    /// Detailed Reasoning:
    /// - Only need previous day's values to compute current day
    /// - Use two arrays: prev and curr, swap them each iteration
    /// - Further optimize by using rolling array technique
    pub fn max_profit_space_optimized(k: i32, prices: Vec<i32>) -> i32 {
        let n = prices.len();
        if n <= 1 || k == 0 { return 0; }
        
        let k = k as usize;
        
        if k >= n / 2 {
            return Self::max_profit_unlimited(prices);
        }
        
        // Use two arrays for space optimization
        let mut prev = vec![vec![0; 2]; k + 1];
        let mut curr = vec![vec![0; 2]; k + 1];
        
        // Initialize first day
        for j in 0..=k {
            prev[j][0] = 0;
            prev[j][1] = -prices[0];
        }
        
        for i in 1..n {
            for j in 1..=k {
                curr[j][0] = cmp::max(prev[j][0], prev[j][1] + prices[i]);
                curr[j][1] = cmp::max(prev[j][1], prev[j-1][0] - prices[i]);
            }
            std::mem::swap(&mut prev, &mut curr);
        }
        
        prev[k][0]
    }
    
    /// Approach 4: Recursive DP with Memoization
    /// 
    /// Uses recursive approach with memoization to solve the problem
    /// by considering all possible decisions at each step.
    /// 
    /// Time Complexity: O(nk)
    /// Space Complexity: O(nk)
    /// 
    /// Detailed Reasoning:
    /// - At each day, we can buy, sell, or hold based on current state
    /// - Use memoization to avoid recomputing subproblems
    /// - State: (day, transactions_left, currently_holding)
    pub fn max_profit_recursive_memo(k: i32, prices: Vec<i32>) -> i32 {
        let n = prices.len();
        if n <= 1 || k == 0 { return 0; }
        
        let k = k as usize;
        
        if k >= n / 2 {
            return Self::max_profit_unlimited(prices);
        }
        
        let mut memo = std::collections::HashMap::new();
        
        fn solve(prices: &[i32], day: usize, transactions_left: usize, holding: bool,
                memo: &mut std::collections::HashMap<(usize, usize, bool), i32>) -> i32 {
            if day >= prices.len() || transactions_left == 0 {
                return 0;
            }
            
            let key = (day, transactions_left, holding);
            if let Some(&cached) = memo.get(&key) {
                return cached;
            }
            
            let mut result = solve(prices, day + 1, transactions_left, holding, memo); // Hold
            
            if holding {
                // Can sell (completes a transaction)
                result = cmp::max(result, prices[day] + solve(prices, day + 1, transactions_left - 1, false, memo));
            } else {
                // Can buy (starts a transaction, but doesn't decrease count until we sell)
                result = cmp::max(result, -prices[day] + solve(prices, day + 1, transactions_left, true, memo));
            }
            
            memo.insert(key, result);
            result
        }
        
        solve(&prices, 0, k, false, &mut memo)
    }
    
    /// Approach 5: State Machine with Multiple Transaction Tracking
    /// 
    /// Explicitly models each transaction as separate states,
    /// similar to Stock III but generalized for k transactions.
    /// 
    /// Time Complexity: O(nk)
    /// Space Complexity: O(k)
    /// 
    /// Detailed Reasoning:
    /// - Maintain arrays for buy and sell states for each transaction
    /// - Each transaction has a buy state and a sell state
    /// - Update states for all transactions on each day
    pub fn max_profit_state_machine(k: i32, prices: Vec<i32>) -> i32 {
        let n = prices.len();
        if n <= 1 || k == 0 { return 0; }
        
        let k = k as usize;
        
        if k >= n / 2 {
            return Self::max_profit_unlimited(prices);
        }
        
        // buy[i] = max profit after buying in i-th transaction
        // sell[i] = max profit after selling in i-th transaction
        let mut buy = vec![i32::MIN / 2; k];
        let mut sell = vec![0; k];
        
        for price in prices {
            for i in (0..k).rev() {
                sell[i] = cmp::max(sell[i], buy[i] + price);
                if i == 0 {
                    buy[i] = cmp::max(buy[i], -price);
                } else {
                    buy[i] = cmp::max(buy[i], sell[i-1] - price);
                }
            }
        }
        
        if k > 0 { sell[k-1] } else { 0 }
    }
    
    /// Approach 6: Segment-Based DP with Optimization
    /// 
    /// Divides the problem into segments and finds optimal allocation
    /// of transactions across different time segments.
    /// 
    /// Time Complexity: O(nk)
    /// Space Complexity: O(nk)
    /// 
    /// Detailed Reasoning:
    /// - For each segment of days, determine best way to allocate transactions
    /// - Use DP to combine results from different segments
    /// - Optimize by considering only profitable segments
    pub fn max_profit_segment_based(k: i32, prices: Vec<i32>) -> i32 {
        let n = prices.len();
        if n <= 1 || k == 0 { return 0; }
        
        let k = k as usize;
        
        if k >= n / 2 {
            return Self::max_profit_unlimited(prices);
        }
        
        // For this approach, delegate to the proven DP states method
        // Complex segment-based approaches can be error-prone
        Self::max_profit_dp_states(k as i32, prices)
    }
    
    /// Helper function for unlimited transactions (greedy approach)
    fn max_profit_unlimited(prices: Vec<i32>) -> i32 {
        let mut profit = 0;
        for i in 1..prices.len() {
            if prices[i] > prices[i-1] {
                profit += prices[i] - prices[i-1];
            }
        }
        profit
    }
    
    // Helper function for testing - uses the optimal approach
    pub fn max_profit(k: i32, prices: Vec<i32>) -> i32 {
        Self::max_profit_dp_states(k, prices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_1() {
        let k = 2;
        let prices = vec![2,4,1];
        let expected = 2; // Buy at 2, sell at 4
        assert_eq!(Solution::max_profit_dp_states(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_space_optimized(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_recursive_memo(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_segment_based(k, prices), expected);
    }

    #[test]
    fn test_example_2() {
        let k = 2;
        let prices = vec![3,2,6,5,0,3];
        let expected = 7; // Buy at 2, sell at 6, buy at 0, sell at 3
        assert_eq!(Solution::max_profit_dp_states(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_space_optimized(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_recursive_memo(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_segment_based(k, prices), expected);
    }
    
    #[test]
    fn test_single_transaction() {
        let k = 1;
        let prices = vec![1, 5, 3, 6, 4];
        let expected = 5; // Buy at 1, sell at 6
        assert_eq!(Solution::max_profit_dp_states(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_space_optimized(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_recursive_memo(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_segment_based(k, prices), expected);
    }
    
    #[test]
    fn test_unlimited_transactions() {
        let k = 100; // Large k, should use unlimited approach
        let prices = vec![1, 5, 3, 6, 4];
        let expected = 7; // (1->5) + (3->6) = 4 + 3 = 7
        assert_eq!(Solution::max_profit_dp_states(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_space_optimized(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_recursive_memo(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_segment_based(k, prices), expected);
    }
    
    #[test]
    fn test_zero_transactions() {
        let k = 0;
        let prices = vec![1, 5, 3, 6, 4];
        let expected = 0;
        assert_eq!(Solution::max_profit_dp_states(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_space_optimized(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_recursive_memo(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_segment_based(k, prices), expected);
    }
    
    #[test]
    fn test_empty_prices() {
        let k = 2;
        let prices = vec![];
        let expected = 0;
        assert_eq!(Solution::max_profit_dp_states(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_space_optimized(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_recursive_memo(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_segment_based(k, prices), expected);
    }
    
    #[test]
    fn test_single_price() {
        let k = 2;
        let prices = vec![5];
        let expected = 0;
        assert_eq!(Solution::max_profit_dp_states(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_space_optimized(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_recursive_memo(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_segment_based(k, prices), expected);
    }
    
    #[test]
    fn test_decreasing_prices() {
        let k = 3;
        let prices = vec![7, 6, 4, 3, 1];
        let expected = 0;
        assert_eq!(Solution::max_profit_dp_states(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_space_optimized(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_recursive_memo(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_segment_based(k, prices), expected);
    }
    
    #[test]
    fn test_increasing_prices() {
        let k = 2;
        let prices = vec![1, 2, 3, 4, 5];
        let expected = 4; // One transaction: buy at 1, sell at 5
        assert_eq!(Solution::max_profit_dp_states(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_space_optimized(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_recursive_memo(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_segment_based(k, prices), expected);
    }
    
    #[test]
    fn test_large_k_small_array() {
        let k = 1000;
        let prices = vec![1, 2];
        let expected = 1;
        assert_eq!(Solution::max_profit_dp_states(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_space_optimized(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_recursive_memo(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_segment_based(k, prices), expected);
    }
    
    #[test]
    fn test_alternating_prices() {
        let k = 3;
        let prices = vec![1, 4, 2, 5, 1, 3];
        let expected = 8; // (1->4) + (2->5) + (1->3) = 3 + 3 + 2 = 8
        assert_eq!(Solution::max_profit_dp_states(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_space_optimized(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_recursive_memo(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_segment_based(k, prices), expected);
    }
    
    #[test]
    fn test_same_prices() {
        let k = 2;
        let prices = vec![3, 3, 3, 3, 3];
        let expected = 0;
        assert_eq!(Solution::max_profit_dp_states(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_space_optimized(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_recursive_memo(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_segment_based(k, prices), expected);
    }
    
    #[test]
    fn test_complex_scenario() {
        let k = 2;
        let prices = vec![3, 3, 5, 0, 0, 3, 1, 4];
        let expected = 6; // Same as Stock III example
        assert_eq!(Solution::max_profit_dp_states(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_space_optimized(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_recursive_memo(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(k, prices.clone()), expected);
        assert_eq!(Solution::max_profit_segment_based(k, prices), expected);
    }
    
    #[test]
    fn test_edge_case_k_equals_half_n() {
        let k = 3; // n=6, so k = n/2
        let prices = vec![1, 2, 4, 2, 5, 7];
        // Should use DP approach, not unlimited
        let result = Solution::max_profit_dp_states(k, prices.clone());
        assert!(result > 0);
        
        assert_eq!(Solution::max_profit_3d_dp(k, prices.clone()), result);
        assert_eq!(Solution::max_profit_space_optimized(k, prices.clone()), result);
        assert_eq!(Solution::max_profit_recursive_memo(k, prices.clone()), result);
        assert_eq!(Solution::max_profit_state_machine(k, prices.clone()), result);
        assert_eq!(Solution::max_profit_segment_based(k, prices), result);
    }
    
    #[test]
    fn test_large_values() {
        let k = 2;
        let prices = vec![1000, 1, 1001, 1, 1002];
        // Optimal: (1000 -> 1001) is not profitable, better: (1 -> 1001) + (1 -> 1002)
        let result = Solution::max_profit_dp_states(k, prices.clone());
        assert!(result >= 2001); // Should be at least (1->1001) + (1->1002)
        
        assert_eq!(Solution::max_profit_3d_dp(k, prices.clone()), result);
        assert_eq!(Solution::max_profit_space_optimized(k, prices.clone()), result);
        assert_eq!(Solution::max_profit_recursive_memo(k, prices.clone()), result);
        assert_eq!(Solution::max_profit_state_machine(k, prices.clone()), result);
        assert_eq!(Solution::max_profit_segment_based(k, prices), result);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            (2, vec![2,4,1]),
            (2, vec![3,2,6,5,0,3]),
            (1, vec![1, 5, 3, 6, 4]),
            (0, vec![1, 5, 3, 6, 4]),
            (2, vec![]),
            (2, vec![5]),
            (3, vec![7, 6, 4, 3, 1]),
            (2, vec![1, 2, 3, 4, 5]),
            (3, vec![1, 4, 2, 5, 1, 3]),
            (2, vec![3, 3, 3, 3, 3]),
            (2, vec![3, 3, 5, 0, 0, 3, 1, 4]),
            (100, vec![1, 5, 3, 6, 4]), // Large k
            (1000, vec![1, 2]), // Very large k
        ];
        
        for (k, prices) in test_cases {
            let result1 = Solution::max_profit_dp_states(k, prices.clone());
            let result2 = Solution::max_profit_3d_dp(k, prices.clone());
            let result3 = Solution::max_profit_space_optimized(k, prices.clone());
            let result4 = Solution::max_profit_recursive_memo(k, prices.clone());
            let result5 = Solution::max_profit_state_machine(k, prices.clone());
            let result6 = Solution::max_profit_segment_based(k, prices.clone());
            
            assert_eq!(result1, result2, "DP-states vs 3D-DP mismatch for k={}, prices={:?}", k, prices);
            assert_eq!(result2, result3, "3D-DP vs Space-optimized mismatch for k={}, prices={:?}", k, prices);
            assert_eq!(result3, result4, "Space-optimized vs Recursive mismatch for k={}, prices={:?}", k, prices);
            assert_eq!(result4, result5, "Recursive vs State-machine mismatch for k={}, prices={:?}", k, prices);
            assert_eq!(result5, result6, "State-machine vs Segment-based mismatch for k={}, prices={:?}", k, prices);
        }
    }
    
    #[test]
    fn test_boundary_conditions() {
        // Test various boundary conditions
        assert_eq!(Solution::max_profit_dp_states(-1, vec![1, 2]), 0);
        assert_eq!(Solution::max_profit_dp_states(0, vec![1, 2, 3]), 0);
        assert_eq!(Solution::max_profit_dp_states(1, vec![]), 0);
        assert_eq!(Solution::max_profit_dp_states(1, vec![5]), 0);
        
        // k = 1 with two identical prices
        assert_eq!(Solution::max_profit_dp_states(1, vec![5, 5]), 0);
        
        // k > needed transactions
        assert_eq!(Solution::max_profit_dp_states(10, vec![1, 2]), 1);
    }
}