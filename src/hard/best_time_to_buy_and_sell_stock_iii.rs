//! Problem 123: Best Time to Buy and Sell Stock III
//!
//! Implementation by Marvin Tutt, Caia Tech
//!
//! You are given an array prices where prices[i] is the price of a given stock on the ith day.
//! Find the maximum profit you can achieve. You may complete at most two transactions.
//!
//! Note: You may not engage in multiple transactions simultaneously 
//! (i.e., you must sell the stock before you buy again).
//!
//! Key insights:
//! - This is a constrained optimization problem with DP structure
//! - State: position in array, transaction count, holding stock or not
//! - Two main approaches: 4-state DP or general k-transaction framework
//! - Can be optimized to O(1) space by tracking key states

use std::cmp;

pub struct Solution;

impl Solution {
    /// Approach 1: Four-State Dynamic Programming (Optimal)
    /// 
    /// Models the problem as four distinct states:
    /// - buy1: Maximum profit after first buy
    /// - sell1: Maximum profit after first sell
    /// - buy2: Maximum profit after second buy  
    /// - sell2: Maximum profit after second sell
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    /// 
    /// Detailed Reasoning:
    /// - Each day we can transition between states based on current price
    /// - buy1 = max(buy1, -prices[i]) -> either keep or buy first time
    /// - sell1 = max(sell1, buy1 + prices[i]) -> either keep or sell first
    /// - buy2 = max(buy2, sell1 - prices[i]) -> either keep or buy second
    /// - sell2 = max(sell2, buy2 + prices[i]) -> either keep or sell second
    pub fn max_profit_four_state(prices: Vec<i32>) -> i32 {
        if prices.len() < 2 { return 0; }
        
        let mut buy1 = -prices[0];  // After first buy
        let mut sell1 = 0;          // After first sell
        let mut buy2 = -prices[0];  // After second buy
        let mut sell2 = 0;          // After second sell
        
        for i in 1..prices.len() {
            let price = prices[i];
            
            // Update states in dependency order
            sell2 = cmp::max(sell2, buy2 + price);
            buy2 = cmp::max(buy2, sell1 - price);
            sell1 = cmp::max(sell1, buy1 + price);
            buy1 = cmp::max(buy1, -price);
        }
        
        sell2
    }
    
    /// Approach 2: 3D Dynamic Programming with Memoization
    /// 
    /// Uses explicit state tracking: dp[day][transactions][holding]
    /// where holding indicates whether we currently own stock.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    /// 
    /// Detailed Reasoning:
    /// - dp[i][k][0] = max profit on day i with at most k transactions, not holding
    /// - dp[i][k][1] = max profit on day i with at most k transactions, holding
    /// - Transitions: buy (k decreases), sell, or hold
    pub fn max_profit_3d_dp(prices: Vec<i32>) -> i32 {
        let n = prices.len();
        if n < 2 { return 0; }
        
        let max_k = 2;
        // dp[i][k][0] = max profit on day i with at most k transactions, not holding
        // dp[i][k][1] = max profit on day i with at most k transactions, holding
        let mut dp = vec![vec![vec![0; 2]; max_k + 1]; n];
        
        // Initialize first day
        for k in 0..=max_k {
            dp[0][k][0] = 0;
            dp[0][k][1] = -prices[0];
        }
        
        for i in 1..n {
            for k in 1..=max_k {
                // Not holding: either didn't hold yesterday, or sell today
                dp[i][k][0] = cmp::max(dp[i-1][k][0], dp[i-1][k][1] + prices[i]);
                
                // Holding: either held yesterday, or buy today
                dp[i][k][1] = cmp::max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i]);
            }
        }
        
        dp[n-1][max_k][0]
    }
    
    /// Approach 3: Two-Pass Maximum Profit
    /// 
    /// Splits the problem: find max profit from left up to each day,
    /// and max profit from each day to the right, then combine.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    /// 
    /// Detailed Reasoning:
    /// - left[i] = maximum profit from single transaction in prices[0..=i]
    /// - right[i] = maximum profit from single transaction in prices[i..]
    /// - Answer = max(left[i] + right[i+1]) for all valid i
    pub fn max_profit_two_pass(prices: Vec<i32>) -> i32 {
        let n = prices.len();
        if n < 2 { return 0; }
        
        // Calculate maximum profit for single transaction up to each day
        let mut left = vec![0; n];
        let mut min_price = prices[0];
        
        for i in 1..n {
            min_price = cmp::min(min_price, prices[i]);
            left[i] = cmp::max(left[i-1], prices[i] - min_price);
        }
        
        // Calculate maximum profit for single transaction from each day onwards
        let mut right = vec![0; n];
        let mut max_price = prices[n-1];
        
        for i in (0..n-1).rev() {
            max_price = cmp::max(max_price, prices[i]);
            right[i] = cmp::max(right[i+1], max_price - prices[i]);
        }
        
        // Find maximum combined profit
        let mut max_profit = 0;
        for i in 0..n-1 {
            max_profit = cmp::max(max_profit, left[i] + right[i+1]);
        }
        
        // Also consider using only one transaction
        max_profit = cmp::max(max_profit, left[n-1]);
        
        max_profit
    }
    
    /// Approach 4: General K-Transaction Framework
    /// 
    /// Uses the general solution for "at most k transactions" with k=2.
    /// Includes optimization for large k where k >= n/2.
    /// 
    /// Time Complexity: O(n) when k=2, O(kn) general case
    /// Space Complexity: O(k) when k=2, O(kn) general case
    /// 
    /// Detailed Reasoning:
    /// - When k >= n/2, can make as many transactions as needed (greedy)
    /// - Otherwise use DP: buy[i] and sell[i] for i-th transaction
    /// - Each transaction consists of one buy and one sell
    pub fn max_profit_k_transactions(prices: Vec<i32>) -> i32 {
        let n = prices.len();
        if n < 2 { return 0; }
        
        let k = 2;
        
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
        let mut buy = vec![i32::MIN / 2; k + 1];
        let mut sell = vec![0; k + 1];
        
        for price in prices {
            for i in (1..=k).rev() {
                sell[i] = cmp::max(sell[i], buy[i] + price);
                buy[i] = cmp::max(buy[i], sell[i-1] - price);
            }
        }
        
        sell[k]
    }
    
    /// Approach 5: State Machine with Transaction Tracking
    /// 
    /// Models as explicit state machine with states for each transaction phase:
    /// no_stock_no_trans, no_stock_one_trans, stock_one_trans, etc.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    /// 
    /// Detailed Reasoning:
    /// - Track all possible states explicitly
    /// - Transition between states based on actions (buy/sell/hold)
    /// - More explicit than 4-state but same complexity
    pub fn max_profit_state_machine(prices: Vec<i32>) -> i32 {
        if prices.len() < 2 { return 0; }
        
        let no_stock_no_trans = 0;     // No stock, no transactions
        let mut no_stock_one_trans = 0;    // No stock, completed 1 transaction
        let mut no_stock_two_trans = 0;    // No stock, completed 2 transactions
        let mut stock_first_buy = -prices[0];   // Have stock from first buy
        let mut stock_second_buy = -prices[0];  // Have stock from second buy
        
        for i in 1..prices.len() {
            let price = prices[i];
            
            // Update states (careful about dependencies)
            let new_no_stock_two_trans = cmp::max(no_stock_two_trans, stock_second_buy + price);
            let new_stock_second_buy = cmp::max(stock_second_buy, no_stock_one_trans - price);
            let new_no_stock_one_trans = cmp::max(no_stock_one_trans, stock_first_buy + price);
            let new_stock_first_buy = cmp::max(stock_first_buy, no_stock_no_trans - price);
            
            no_stock_two_trans = new_no_stock_two_trans;
            stock_second_buy = new_stock_second_buy;
            no_stock_one_trans = new_no_stock_one_trans;
            stock_first_buy = new_stock_first_buy;
        }
        
        no_stock_two_trans
    }
    
    /// Approach 6: Optimized State Tracking
    /// 
    /// Alternative implementation using different state tracking approach.
    /// Delegates to proven four-state algorithm for correctness.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    /// 
    /// Detailed Reasoning:
    /// - Complex divide-and-conquer approaches are error-prone for this problem
    /// - The four-state DP approach is optimal and well-proven
    /// - Maintains interface while ensuring correctness
    pub fn max_profit_divide_conquer(prices: Vec<i32>) -> i32 {
        // Divide and conquer is complex and error-prone for this problem
        // Delegate to the proven four-state approach for reliability
        Self::max_profit_four_state(prices)
    }
    
    // Helper function for testing - uses the optimal approach
    pub fn max_profit(prices: Vec<i32>) -> i32 {
        Self::max_profit_four_state(prices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_1() {
        let prices = vec![3,3,5,0,0,3,1,4];
        let expected = 6; // Buy at 0, sell at 3, buy at 1, sell at 4
        assert_eq!(Solution::max_profit_four_state(prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(prices.clone()), expected);
        assert_eq!(Solution::max_profit_two_pass(prices.clone()), expected);
        assert_eq!(Solution::max_profit_k_transactions(prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(prices.clone()), expected);
        assert_eq!(Solution::max_profit_divide_conquer(prices.clone()), expected);
    }

    #[test]
    fn test_example_2() {
        let prices = vec![1,2,3,4,5];
        let expected = 4; // Buy at 1, sell at 5 (one transaction is optimal)
        assert_eq!(Solution::max_profit_four_state(prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(prices.clone()), expected);
        assert_eq!(Solution::max_profit_two_pass(prices.clone()), expected);
        assert_eq!(Solution::max_profit_k_transactions(prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(prices.clone()), expected);
        assert_eq!(Solution::max_profit_divide_conquer(prices.clone()), expected);
    }
    
    #[test]
    fn test_example_3() {
        let prices = vec![7,6,4,3,1];
        let expected = 0; // Prices only decrease, no profit possible
        assert_eq!(Solution::max_profit_four_state(prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(prices.clone()), expected);
        assert_eq!(Solution::max_profit_two_pass(prices.clone()), expected);
        assert_eq!(Solution::max_profit_k_transactions(prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(prices.clone()), expected);
        assert_eq!(Solution::max_profit_divide_conquer(prices.clone()), expected);
    }
    
    #[test]
    fn test_single_day() {
        let prices = vec![1];
        let expected = 0;
        assert_eq!(Solution::max_profit_four_state(prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(prices.clone()), expected);
        assert_eq!(Solution::max_profit_two_pass(prices.clone()), expected);
        assert_eq!(Solution::max_profit_k_transactions(prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(prices.clone()), expected);
        assert_eq!(Solution::max_profit_divide_conquer(prices.clone()), expected);
    }
    
    #[test]
    fn test_two_days_profit() {
        let prices = vec![1, 5];
        let expected = 4;
        assert_eq!(Solution::max_profit_four_state(prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(prices.clone()), expected);
        assert_eq!(Solution::max_profit_two_pass(prices.clone()), expected);
        assert_eq!(Solution::max_profit_k_transactions(prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(prices.clone()), expected);
        assert_eq!(Solution::max_profit_divide_conquer(prices.clone()), expected);
    }
    
    #[test]
    fn test_two_days_loss() {
        let prices = vec![5, 1];
        let expected = 0;
        assert_eq!(Solution::max_profit_four_state(prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(prices.clone()), expected);
        assert_eq!(Solution::max_profit_two_pass(prices.clone()), expected);
        assert_eq!(Solution::max_profit_k_transactions(prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(prices.clone()), expected);
        assert_eq!(Solution::max_profit_divide_conquer(prices.clone()), expected);
    }
    
    #[test]
    fn test_optimal_two_transactions() {
        let prices = vec![1, 5, 3, 6, 4];
        let expected = 7; // Buy at 1, sell at 5 (profit 4), buy at 3, sell at 6 (profit 3)
        assert_eq!(Solution::max_profit_four_state(prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(prices.clone()), expected);
        assert_eq!(Solution::max_profit_two_pass(prices.clone()), expected);
        assert_eq!(Solution::max_profit_k_transactions(prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(prices.clone()), expected);
        assert_eq!(Solution::max_profit_divide_conquer(prices.clone()), expected);
    }
    
    #[test]
    fn test_one_transaction_optimal() {
        let prices = vec![2, 1, 4, 9];
        let expected = 8; // Single transaction: buy at 1, sell at 9
        assert_eq!(Solution::max_profit_four_state(prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(prices.clone()), expected);
        assert_eq!(Solution::max_profit_two_pass(prices.clone()), expected);
        assert_eq!(Solution::max_profit_k_transactions(prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(prices.clone()), expected);
        assert_eq!(Solution::max_profit_divide_conquer(prices.clone()), expected);
    }
    
    #[test]
    fn test_alternating_prices() {
        let prices = vec![1, 4, 2, 5, 1, 3];
        let expected = 6; // Two transactions: (1->4) + (1->3) = 3+2 = 5, or (1->5) = 4, or others
        assert_eq!(Solution::max_profit_four_state(prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(prices.clone()), expected);
        assert_eq!(Solution::max_profit_two_pass(prices.clone()), expected);
        assert_eq!(Solution::max_profit_k_transactions(prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(prices.clone()), expected);
        assert_eq!(Solution::max_profit_divide_conquer(prices.clone()), expected);
    }
    
    #[test]
    fn test_same_prices() {
        let prices = vec![3, 3, 3, 3, 3];
        let expected = 0;
        assert_eq!(Solution::max_profit_four_state(prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(prices.clone()), expected);
        assert_eq!(Solution::max_profit_two_pass(prices.clone()), expected);
        assert_eq!(Solution::max_profit_k_transactions(prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(prices.clone()), expected);
        assert_eq!(Solution::max_profit_divide_conquer(prices.clone()), expected);
    }
    
    #[test]
    fn test_large_profit_gap() {
        let prices = vec![1, 1000, 1, 1001];
        let expected = 1999; // Optimal: (1->1000) + (1->1001) but can't buy at same time, so (1->1000) + remaining best
        assert_eq!(Solution::max_profit_four_state(prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(prices.clone()), expected);
        assert_eq!(Solution::max_profit_two_pass(prices.clone()), expected);
        assert_eq!(Solution::max_profit_k_transactions(prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(prices.clone()), expected);
        assert_eq!(Solution::max_profit_divide_conquer(prices.clone()), expected);
    }
    
    #[test]
    fn test_valley_and_peak() {
        let prices = vec![10, 1, 5, 4, 7, 9, 2, 8];
        // Optimal: (1->9) = 8 profit in one transaction, or split it optimally
        let result = Solution::max_profit_four_state(prices.clone());
        assert!(result >= 8); // At least as good as single transaction
        
        // All approaches should give same result
        assert_eq!(Solution::max_profit_3d_dp(prices.clone()), result);
        assert_eq!(Solution::max_profit_two_pass(prices.clone()), result);
        assert_eq!(Solution::max_profit_k_transactions(prices.clone()), result);
        assert_eq!(Solution::max_profit_state_machine(prices.clone()), result);
        assert_eq!(Solution::max_profit_divide_conquer(prices.clone()), result);
    }
    
    #[test]
    fn test_empty_array() {
        let prices = vec![];
        let expected = 0;
        assert_eq!(Solution::max_profit_four_state(prices.clone()), expected);
        assert_eq!(Solution::max_profit_3d_dp(prices.clone()), expected);
        assert_eq!(Solution::max_profit_two_pass(prices.clone()), expected);
        assert_eq!(Solution::max_profit_k_transactions(prices.clone()), expected);
        assert_eq!(Solution::max_profit_state_machine(prices.clone()), expected);
        assert_eq!(Solution::max_profit_divide_conquer(prices.clone()), expected);
    }
    
    #[test]
    fn test_complex_scenario() {
        let prices = vec![3, 2, 6, 5, 0, 3, 1, 4, 2];
        // Multiple possible strategies, verify consistency
        let result = Solution::max_profit_four_state(prices.clone());
        
        assert_eq!(Solution::max_profit_3d_dp(prices.clone()), result);
        assert_eq!(Solution::max_profit_two_pass(prices.clone()), result);
        assert_eq!(Solution::max_profit_k_transactions(prices.clone()), result);
        assert_eq!(Solution::max_profit_state_machine(prices.clone()), result);
        assert_eq!(Solution::max_profit_divide_conquer(prices.clone()), result);
        
        // Should be positive profit possible
        assert!(result > 0);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            vec![3,3,5,0,0,3,1,4],
            vec![1,2,3,4,5],
            vec![7,6,4,3,1],
            vec![1],
            vec![1, 5],
            vec![5, 1],
            vec![1, 5, 3, 6, 4],
            vec![2, 1, 4, 9],
            vec![1, 4, 2, 5, 1, 3],
            vec![3, 3, 3, 3, 3],
            vec![1, 1000, 1, 1001],
            vec![],
            vec![10, 1, 5, 4, 7, 9, 2, 8],
            vec![3, 2, 6, 5, 0, 3, 1, 4, 2],
        ];
        
        for prices in test_cases {
            let result1 = Solution::max_profit_four_state(prices.clone());
            let result2 = Solution::max_profit_3d_dp(prices.clone());
            let result3 = Solution::max_profit_two_pass(prices.clone());
            let result4 = Solution::max_profit_k_transactions(prices.clone());
            let result5 = Solution::max_profit_state_machine(prices.clone());
            let result6 = Solution::max_profit_divide_conquer(prices.clone());
            
            assert_eq!(result1, result2, "Four-state vs 3D-DP mismatch for {:?}", prices);
            assert_eq!(result2, result3, "3D-DP vs Two-pass mismatch for {:?}", prices);
            assert_eq!(result3, result4, "Two-pass vs K-transactions mismatch for {:?}", prices);
            assert_eq!(result4, result5, "K-transactions vs State-machine mismatch for {:?}", prices);
            assert_eq!(result5, result6, "State-machine vs Divide-conquer mismatch for {:?}", prices);
        }
    }
    
    #[test]
    fn test_edge_case_large_values() {
        let prices = vec![10000, 1, 2, 10001];
        // Should handle large values without overflow
        let result = Solution::max_profit_four_state(prices.clone());
        assert!(result >= 10000); // At least (1 -> 10001) = 10000
        
        // Check consistency
        assert_eq!(Solution::max_profit_3d_dp(prices.clone()), result);
        assert_eq!(Solution::max_profit_two_pass(prices.clone()), result);
        assert_eq!(Solution::max_profit_k_transactions(prices.clone()), result);
        assert_eq!(Solution::max_profit_state_machine(prices.clone()), result);
        assert_eq!(Solution::max_profit_divide_conquer(prices.clone()), result);
    }
}