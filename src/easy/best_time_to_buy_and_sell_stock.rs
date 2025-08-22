//! # Problem 121: Best Time to Buy and Sell Stock
//!
//! You are given an array prices where prices[i] is the price of a given stock on the ith day.
//!
//! You want to maximize your profit by choosing a single day to buy one stock and choosing 
//! a different day in the future to sell that stock.
//!
//! Return the maximum profit you can achieve from this transaction. If you cannot achieve 
//! any profit, return 0.
//!
//! ## Examples
//!
//! ```text
//! Input: prices = [7,1,5,3,6,4]
//! Output: 5
//! Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
//! Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
//! ```
//!
//! ```text
//! Input: prices = [7,6,4,3,2,1]
//! Output: 0
//! Explanation: In this case, no transactions are done and the max profit = 0.
//! ```
//!
//! ## Constraints
//!
//! * 1 <= prices.length <= 10^5
//! * 0 <= prices[i] <= 10^4

/// Solution for Best Time to Buy and Sell Stock problem
pub struct Solution;

impl Solution {
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Single Pass (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Keep track of minimum price seen so far
    /// 2. For each price, calculate profit if selling today
    /// 3. Update maximum profit if current profit is better
    /// 4. Update minimum price if current price is lower
    /// 
    /// **Time Complexity:** O(n) - Single pass through array
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **Key Insight:** We need to buy before we sell, so for any selling day,
    /// we want to have bought on the day with minimum price before that day.
    /// 
    /// **Why this is optimal:**
    /// - Only one pass needed through the array
    /// - Constant space usage
    /// - Natural logic that follows the problem constraints
    /// 
    /// **Visualization:**
    /// ```text
    /// prices = [7,1,5,3,6,4]
    /// min_price: 7→1→1→1→1→1
    /// profit:    0→0→4→4→5→5
    /// ```
    pub fn max_profit(&self, prices: Vec<i32>) -> i32 {
        if prices.is_empty() {
            return 0;
        }
        
        let mut min_price = prices[0];
        let mut max_profit = 0;
        
        for price in prices.iter().skip(1) {
            // Calculate profit if we sell today
            let profit = price - min_price;
            max_profit = max_profit.max(profit);
            
            // Update minimum price if current price is lower
            min_price = min_price.min(*price);
        }
        
        max_profit
    }

    /// # Approach 2: Brute Force (Inefficient - Educational)
    /// 
    /// **Algorithm:**
    /// 1. Try all possible buy-sell combinations
    /// 2. For each buy day, try all possible sell days after it
    /// 3. Keep track of maximum profit
    /// 
    /// **Time Complexity:** O(n²) - Check all pairs
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **When to use:** Understanding the problem, very small inputs, or verification
    /// 
    /// **Why this approach is inefficient:**
    /// - Checks many unnecessary combinations
    /// - No optimization to avoid redundant calculations
    pub fn max_profit_brute_force(&self, prices: Vec<i32>) -> i32 {
        let mut max_profit = 0;
        
        for i in 0..prices.len() {
            for j in (i + 1)..prices.len() {
                let profit = prices[j] - prices[i];
                max_profit = max_profit.max(profit);
            }
        }
        
        max_profit
    }

    /// # Approach 3: Dynamic Programming (Alternative)
    /// 
    /// **Algorithm:**
    /// 1. For each day, calculate max profit if we sell on that day
    /// 2. This equals current price minus minimum price up to that day
    /// 3. Keep track of running minimum and maximum profit
    /// 
    /// **Time Complexity:** O(n) - Single pass through array
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **When to use:** When you want to think in terms of DP states
    /// 
    /// **DP State Definition:**
    /// - dp[i] = maximum profit achievable if we sell on day i
    /// - dp[i] = prices[i] - min(prices[0..i])
    pub fn max_profit_dp(&self, prices: Vec<i32>) -> i32 {
        if prices.is_empty() {
            return 0;
        }
        
        let mut min_price_so_far = prices[0];
        let mut max_profit = 0;
        
        for price in prices.iter() {
            // DP recurrence: max profit if selling today
            let profit_today = price - min_price_so_far;
            max_profit = max_profit.max(profit_today);
            
            // Update state for next iteration
            min_price_so_far = min_price_so_far.min(*price);
        }
        
        max_profit
    }

    /// # Approach 4: Sliding Window Maximum
    /// 
    /// **Algorithm:**
    /// 1. Use two pointers: buy (left) and sell (right)
    /// 2. Expand window by moving sell pointer
    /// 3. If price drops below buy price, move buy pointer to current position
    /// 4. Track maximum profit in current window
    /// 
    /// **Time Complexity:** O(n) - Each element visited at most twice
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **When to use:** When you want to think in terms of sliding window
    pub fn max_profit_sliding_window(&self, prices: Vec<i32>) -> i32 {
        if prices.len() < 2 {
            return 0;
        }
        
        let mut buy = 0;
        let mut max_profit = 0;
        
        for sell in 1..prices.len() {
            // If current price is lower than buy price, move buy pointer
            if prices[sell] < prices[buy] {
                buy = sell;
            } else {
                // Calculate profit with current window
                let profit = prices[sell] - prices[buy];
                max_profit = max_profit.max(profit);
            }
        }
        
        max_profit
    }

    /// # Approach 5: Kadane's Algorithm Adaptation
    /// 
    /// **Algorithm:**
    /// 1. Transform problem: instead of prices, consider daily price differences
    /// 2. Find maximum sum subarray in the difference array
    /// 3. This corresponds to maximum profit from buy-sell transaction
    /// 
    /// **Time Complexity:** O(n) - Single pass through array
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **Key Insight:** Maximum profit = maximum sum of consecutive differences
    /// where differences[i] = prices[i+1] - prices[i]
    /// 
    /// **Why this works:** 
    /// - Buying on day i and selling on day j gives profit = prices[j] - prices[i]
    /// - This equals sum of daily differences from day i to day j-1
    /// - Maximum subarray sum finds the best consecutive period
    pub fn max_profit_kadane(&self, prices: Vec<i32>) -> i32 {
        if prices.len() < 2 {
            return 0;
        }
        
        let mut max_ending_here = 0;
        let mut max_so_far = 0;
        
        for i in 1..prices.len() {
            let diff = prices[i] - prices[i - 1];
            max_ending_here = (max_ending_here + diff).max(0);
            max_so_far = max_so_far.max(max_ending_here);
        }
        
        max_so_far
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
        assert_eq!(solution.max_profit(vec![7,1,5,3,6,4]), 5);
        
        // No profit possible
        assert_eq!(solution.max_profit(vec![7,6,4,3,2,1]), 0);
        
        // Simple profit
        assert_eq!(solution.max_profit(vec![1,2]), 1);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // Single day
        assert_eq!(solution.max_profit(vec![1]), 0);
        
        // All same price
        assert_eq!(solution.max_profit(vec![3,3,3,3]), 0);
        
        // Strictly increasing
        assert_eq!(solution.max_profit(vec![1,2,3,4,5]), 4);
        
        // Strictly decreasing
        assert_eq!(solution.max_profit(vec![5,4,3,2,1]), 0);
    }

    #[test]
    fn test_optimal_timing() {
        let solution = setup();
        
        // Best buy at beginning, sell at end
        assert_eq!(solution.max_profit(vec![1,10,2,9]), 9);
        
        // Best buy in middle
        assert_eq!(solution.max_profit(vec![5,1,10,3]), 9);
        
        // Multiple local maxima
        assert_eq!(solution.max_profit(vec![3,1,4,1,5,9,2,6]), 8); // Buy at 1, sell at 9
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            vec![7,1,5,3,6,4],
            vec![7,6,4,3,2,1],
            vec![1,2,3,4,5],
            vec![5,4,3,2,1],
            vec![1],
            vec![3,3,3,3],
            vec![1,10,2,9],
            vec![3,1,4,1,5,9,2,6],
        ];
        
        for prices in test_cases {
            let result1 = solution.max_profit(prices.clone());
            let result2 = solution.max_profit_brute_force(prices.clone());
            let result3 = solution.max_profit_dp(prices.clone());
            let result4 = solution.max_profit_sliding_window(prices.clone());
            let result5 = solution.max_profit_kadane(prices.clone());
            
            assert_eq!(result1, result2, "Single pass vs brute force mismatch for {:?}", prices);
            assert_eq!(result2, result3, "Brute force vs DP mismatch for {:?}", prices);
            assert_eq!(result3, result4, "DP vs sliding window mismatch for {:?}", prices);
            assert_eq!(result4, result5, "Sliding window vs Kadane mismatch for {:?}", prices);
        }
    }

    #[test]
    fn test_boundary_values() {
        let solution = setup();
        
        // Maximum values
        assert_eq!(solution.max_profit(vec![0, 10000]), 10000);
        
        // Zero values
        assert_eq!(solution.max_profit(vec![0, 0, 0]), 0);
        
        // Large fluctuations
        assert_eq!(solution.max_profit(vec![10000, 1, 10000, 1]), 9999);
    }

    #[test]
    fn test_profit_properties() {
        let solution = setup();
        
        // Profit should never be negative
        let prices = vec![10, 5, 2, 1];
        let profit = solution.max_profit(prices);
        assert!(profit >= 0);
        
        // Profit should not exceed max_price - min_price
        let prices2 = vec![3, 1, 8, 2, 9, 4];
        let profit2 = solution.max_profit(prices2.clone());
        let max_price = *prices2.iter().max().unwrap();
        let min_price = *prices2.iter().min().unwrap();
        assert!(profit2 <= max_price - min_price);
        assert_eq!(profit2, 8); // Buy at 1, sell at 9
    }

    #[test]
    fn test_complex_patterns() {
        let solution = setup();
        
        // V-shaped (crash then recovery)
        assert_eq!(solution.max_profit(vec![10, 5, 1, 8]), 7);
        
        // Mountain shaped
        assert_eq!(solution.max_profit(vec![1, 5, 10, 5, 1]), 9);
        
        // Multiple peaks
        assert_eq!(solution.max_profit(vec![1, 8, 2, 9, 3, 7]), 8); // Buy at 1, sell at 9
        
        // Gradual rise with dips
        assert_eq!(solution.max_profit(vec![1, 3, 2, 5, 4, 8, 6, 10]), 9); // Buy at 1, sell at 10
    }

    #[test]
    fn test_large_arrays() {
        let solution = setup();
        
        // Large array with known pattern
        let mut prices = vec![1000];
        for i in 1..1000 {
            prices.push(i);
        }
        let profit = solution.max_profit(prices);
        assert_eq!(profit, 999 - 1); // Buy at 1, sell at 999
        
        // Large decreasing array
        let decreasing: Vec<i32> = (1..1000).rev().collect();
        assert_eq!(solution.max_profit(decreasing), 0);
    }

    #[test]
    fn test_mathematical_properties() {
        let solution = setup();
        
        // Adding constant to all prices shouldn't change profit
        let original = vec![1, 5, 3, 8, 2];
        let shifted: Vec<i32> = original.iter().map(|x| x + 100).collect();
        
        assert_eq!(solution.max_profit(original), solution.max_profit(shifted));
        
        // Scaling all prices by same factor should scale profit
        let base = vec![1, 2, 1, 3];
        let scaled: Vec<i32> = base.iter().map(|x| x * 2).collect();
        
        let base_profit = solution.max_profit(base);
        let scaled_profit = solution.max_profit(scaled);
        assert_eq!(scaled_profit, base_profit * 2);
    }
}