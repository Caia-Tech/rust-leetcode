//! # Problem 322: Coin Change
//!
//! You are given an integer array `coins` representing coins of different denominations and an 
//! integer `amount` representing a total amount of money.
//!
//! Return the fewest number of coins that you need to make up that amount. If that amount of 
//! money cannot be made up by any combination of the coins, return -1.
//!
//! You may assume that you have an infinite number of each kind of coin.
//!
//! ## Examples
//!
//! ```text
//! Input: coins = [1,2,5], amount = 11
//! Output: 3
//! Explanation: 11 = 5 + 5 + 1
//! ```
//!
//! ```text
//! Input: coins = [2], amount = 3
//! Output: -1
//! Explanation: Cannot make amount 3 with coins of 2
//! ```
//!
//! ```text
//! Input: coins = [1], amount = 0
//! Output: 0
//! ```
//!
//! ## Constraints
//!
//! * 1 <= coins.length <= 12
//! * 1 <= coins[i] <= 2^31 - 1
//! * 0 <= amount <= 10^4

use std::collections::{HashMap, VecDeque};

/// Solution for Coin Change problem
pub struct Solution;

impl Solution {
    /// Creates a new instance of Solution
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Dynamic Programming Bottom-Up (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Create dp array where dp[i] = min coins for amount i
    /// 2. Initialize dp[0] = 0 (zero coins for zero amount)
    /// 3. For each amount from 1 to target:
    ///    - Try each coin denomination
    ///    - If coin <= amount: dp[amount] = min(dp[amount], dp[amount-coin] + 1)
    /// 4. Return dp[amount] or -1 if impossible
    /// 
    /// **Time Complexity:** O(amount * coins.len()) - Nested loops
    /// **Space Complexity:** O(amount) - DP array
    /// 
    /// **Key Insight:**
    /// - This is an unbounded knapsack problem
    /// - We can use each coin unlimited times
    /// - Build solution from smaller amounts to larger
    /// 
    /// **Why this works:**
    /// - Optimal substructure: min coins for amount n uses optimal solution for n-coin
    /// - No greedy choice: largest coin first doesn't always work
    /// - Must try all possibilities systematically
    /// 
    /// **Visualization:**
    /// ```text
    /// coins = [1,2,5], amount = 11
    /// dp[0] = 0
    /// dp[1] = 1 (1 coin of 1)
    /// dp[2] = 1 (1 coin of 2)
    /// dp[3] = 2 (1+2)
    /// dp[4] = 2 (2+2)
    /// dp[5] = 1 (1 coin of 5)
    /// ...
    /// dp[11] = 3 (5+5+1)
    /// ```
    pub fn coin_change(&self, coins: Vec<i32>, amount: i32) -> i32 {
        if amount == 0 {
            return 0;
        }
        
        let amount = amount as usize;
        let mut dp = vec![i32::MAX; amount + 1];
        dp[0] = 0;
        
        for i in 1..=amount {
            for &coin in &coins {
                let coin = coin as usize;
                if coin <= i && dp[i - coin] != i32::MAX {
                    dp[i] = dp[i].min(dp[i - coin] + 1);
                }
            }
        }
        
        if dp[amount] == i32::MAX {
            -1
        } else {
            dp[amount]
        }
    }

    /// # Approach 2: BFS (Breadth-First Search)
    /// 
    /// **Algorithm:**
    /// 1. Start with amount, try to reach 0
    /// 2. Each level represents using one more coin
    /// 3. For each amount in current level, subtract each coin
    /// 4. First time we reach 0 is minimum coins
    /// 5. Use visited set to avoid redundant work
    /// 
    /// **Time Complexity:** O(amount * coins.len()) - Each amount visited once
    /// **Space Complexity:** O(amount) - Queue and visited set
    /// 
    /// **Why BFS works here:**
    /// - We want minimum number of steps (coins)
    /// - BFS explores level by level
    /// - First solution found is guaranteed to be optimal
    /// 
    /// **Advantages:**
    /// - Can terminate early when solution found
    /// - Intuitive for "shortest path" problems
    /// - Natural for finding minimum steps
    pub fn coin_change_bfs(&self, coins: Vec<i32>, amount: i32) -> i32 {
        if amount == 0 {
            return 0;
        }
        
        let mut queue = VecDeque::new();
        let mut visited = vec![false; (amount + 1) as usize];
        
        queue.push_back((amount, 0));
        visited[amount as usize] = true;
        
        while let Some((curr_amount, num_coins)) = queue.pop_front() {
            for &coin in &coins {
                let next_amount = curr_amount - coin;
                
                if next_amount == 0 {
                    return num_coins + 1;
                }
                
                if next_amount > 0 && !visited[next_amount as usize] {
                    visited[next_amount as usize] = true;
                    queue.push_back((next_amount, num_coins + 1));
                }
            }
        }
        
        -1
    }

    /// # Approach 3: DFS with Memoization (Top-Down DP)
    /// 
    /// **Algorithm:**
    /// 1. Define recursive function: minCoins(amount)
    /// 2. Base case: amount = 0 returns 0
    /// 3. Try each coin, recurse on amount - coin
    /// 4. Take minimum across all valid choices
    /// 5. Memoize to avoid redundant calculations
    /// 
    /// **Time Complexity:** O(amount * coins.len()) - Each subproblem solved once
    /// **Space Complexity:** O(amount) - Memoization table + recursion stack
    /// 
    /// **Trade-offs:**
    /// - More intuitive than bottom-up for some
    /// - Can have stack overflow for large amounts
    /// - May not compute all subproblems (good for sparse solutions)
    /// 
    /// **When to use:** When you think recursively about the problem
    pub fn coin_change_dfs_memo(&self, coins: Vec<i32>, amount: i32) -> i32 {
        if amount == 0 {
            return 0;
        }
        
        let mut memo = HashMap::new();
        let result = self.dfs_helper(&coins, amount, &mut memo);
        
        if result == i32::MAX {
            -1
        } else {
            result
        }
    }
    
    fn dfs_helper(&self, coins: &[i32], amount: i32, memo: &mut HashMap<i32, i32>) -> i32 {
        if amount == 0 {
            return 0;
        }
        if amount < 0 {
            return i32::MAX;
        }
        
        if let Some(&cached) = memo.get(&amount) {
            return cached;
        }
        
        let mut min_coins = i32::MAX;
        for &coin in coins {
            let sub_result = self.dfs_helper(coins, amount - coin, memo);
            if sub_result != i32::MAX {
                min_coins = min_coins.min(sub_result + 1);
            }
        }
        
        memo.insert(amount, min_coins);
        min_coins
    }

    /// # Approach 4: DP with Coin Optimization
    /// 
    /// **Algorithm:**
    /// 1. Sort coins in descending order for potential early termination
    /// 2. Use DP but check if we can improve current solution
    /// 3. Skip coins that are too large for current amount
    /// 
    /// **Time Complexity:** O(amount * coins.len()) - Same as basic DP
    /// **Space Complexity:** O(amount) - DP array
    /// 
    /// **Optimizations:**
    /// - Sort coins for better cache locality
    /// - Early termination when coin > amount
    /// - Can prune search space in some cases
    /// 
    /// **Note:** Sorting doesn't change asymptotic complexity but can improve constants
    pub fn coin_change_optimized(&self, mut coins: Vec<i32>, amount: i32) -> i32 {
        if amount == 0 {
            return 0;
        }
        
        // Sort coins for potential optimization
        coins.sort_unstable();
        
        let amount = amount as usize;
        let mut dp = vec![i32::MAX; amount + 1];
        dp[0] = 0;
        
        for i in 1..=amount {
            for &coin in &coins {
                let coin = coin as usize;
                if coin > i {
                    break;  // All remaining coins are too large
                }
                if dp[i - coin] != i32::MAX {
                    dp[i] = dp[i].min(dp[i - coin] + 1);
                }
            }
        }
        
        if dp[amount] == i32::MAX {
            -1
        } else {
            dp[amount]
        }
    }

    /// # Approach 5: Complete Knapsack Variant
    /// 
    /// **Algorithm:**
    /// 1. Treat as unbounded knapsack with weight = value = coin
    /// 2. Minimize items (coins) instead of maximizing value
    /// 3. For each coin, update all amounts it can contribute to
    /// 
    /// **Time Complexity:** O(amount * coins.len())
    /// **Space Complexity:** O(amount)
    /// 
    /// **Connection to Knapsack:**
    /// - Unbounded: can use each coin multiple times
    /// - Weight = coin value, capacity = amount
    /// - Minimize items vs maximize value
    /// 
    /// **Different iteration order:**
    /// - Iterate coins first, then amounts
    /// - Updates all amounts a coin can affect
    /// - Sometimes better cache performance
    pub fn coin_change_knapsack(&self, coins: Vec<i32>, amount: i32) -> i32 {
        if amount == 0 {
            return 0;
        }
        
        let amount = amount as usize;
        let mut dp = vec![i32::MAX; amount + 1];
        dp[0] = 0;
        
        // Iterate coins first (different from approach 1)
        for coin in coins {
            let coin = coin as usize;
            for i in coin..=amount {
                if dp[i - coin] != i32::MAX {
                    dp[i] = dp[i].min(dp[i - coin] + 1);
                }
            }
        }
        
        if dp[amount] == i32::MAX {
            -1
        } else {
            dp[amount]
        }
    }

    /// # Approach 6: Branch and Bound DFS
    /// 
    /// **Algorithm:**
    /// 1. Use DFS but with pruning based on current best
    /// 2. Sort coins descending to try larger coins first
    /// 3. Prune branches that can't improve current best
    /// 4. Track minimum coins found so far
    /// 
    /// **Time Complexity:** O(amount^coins.len()) worst case, often much better
    /// **Space Complexity:** O(coins.len()) - Recursion stack
    /// 
    /// **Pruning strategies:**
    /// - Stop if current path uses more coins than best so far
    /// - Estimate lower bound and prune if can't improve
    /// - Try larger coins first for faster reduction
    /// 
    /// **When effective:** When solution uses few large coins
    pub fn coin_change_branch_bound(&self, mut coins: Vec<i32>, amount: i32) -> i32 {
        if amount == 0 {
            return 0;
        }
        
        // Sort descending to try larger coins first
        coins.sort_unstable_by(|a, b| b.cmp(a));
        
        let mut min_coins = i32::MAX;
        self.branch_bound_helper(&coins, amount, 0, 0, &mut min_coins);
        
        if min_coins == i32::MAX {
            -1
        } else {
            min_coins
        }
    }
    
    fn branch_bound_helper(&self, coins: &[i32], amount: i32, idx: usize, 
                           count: i32, min_coins: &mut i32) {
        if amount == 0 {
            *min_coins = (*min_coins).min(count);
            return;
        }
        
        if idx >= coins.len() || amount < 0 || count >= *min_coins {
            return;  // Pruning
        }
        
        // Try using current coin multiple times
        let max_use = amount / coins[idx];
        for use_count in (0..=max_use).rev() {
            if count + use_count >= *min_coins {
                break;  // Prune
            }
            self.branch_bound_helper(coins, amount - coins[idx] * use_count, 
                                   idx + 1, count + use_count, min_coins);
        }
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
        
        // Example 1: [1,2,5], amount=11 → 3
        assert_eq!(solution.coin_change(vec![1, 2, 5], 11), 3);
        
        // Example 2: [2], amount=3 → -1
        assert_eq!(solution.coin_change(vec![2], 3), -1);
        
        // Example 3: [1], amount=0 → 0
        assert_eq!(solution.coin_change(vec![1], 0), 0);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // Single coin equals amount
        assert_eq!(solution.coin_change(vec![5], 5), 1);
        
        // Single coin, multiple needed
        assert_eq!(solution.coin_change(vec![3], 9), 3);
        
        // Amount is 1
        assert_eq!(solution.coin_change(vec![1, 2, 5], 1), 1);
        
        // Large coin values
        assert_eq!(solution.coin_change(vec![1000, 2000], 3000), 2);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            (vec![1, 2, 5], 11),
            (vec![2, 5, 10], 15),
            (vec![1, 3, 4], 6),
            (vec![1, 5, 6, 9], 11),
            (vec![186, 419, 83, 408], 6249),
        ];
        
        for (coins, amount) in test_cases {
            let result1 = solution.coin_change(coins.clone(), amount);
            let result2 = solution.coin_change_bfs(coins.clone(), amount);
            let result3 = solution.coin_change_dfs_memo(coins.clone(), amount);
            let result4 = solution.coin_change_optimized(coins.clone(), amount);
            let result5 = solution.coin_change_knapsack(coins.clone(), amount);
            let result6 = solution.coin_change_branch_bound(coins.clone(), amount);
            
            assert_eq!(result1, result2, "Mismatch for {:?}, amount={}", coins, amount);
            assert_eq!(result2, result3, "Mismatch for {:?}, amount={}", coins, amount);
            assert_eq!(result3, result4, "Mismatch for {:?}, amount={}", coins, amount);
            assert_eq!(result4, result5, "Mismatch for {:?}, amount={}", coins, amount);
            assert_eq!(result5, result6, "Mismatch for {:?}, amount={}", coins, amount);
        }
    }

    #[test]
    fn test_impossible_cases() {
        let solution = setup();
        
        // No way to make odd amount with only even coins
        assert_eq!(solution.coin_change(vec![2, 4, 6], 7), -1);
        
        // Amount less than smallest coin
        assert_eq!(solution.coin_change(vec![5, 10], 3), -1);
        
        // Prime amount with non-factor coins
        assert_eq!(solution.coin_change(vec![3, 5], 1), -1);
    }

    #[test]
    fn test_greedy_doesnt_work() {
        let solution = setup();
        
        // Greedy would choose 6+1+1+1=4 coins, optimal is 4+4+1=3
        assert_eq!(solution.coin_change(vec![1, 4, 6], 9), 3);
        
        // Greedy would fail, optimal exists
        assert_eq!(solution.coin_change(vec![1, 3, 4], 6), 2);
    }

    #[test]
    fn test_single_coin_type() {
        let solution = setup();
        
        // Divisible
        assert_eq!(solution.coin_change(vec![5], 25), 5);
        
        // Not divisible
        assert_eq!(solution.coin_change(vec![5], 27), -1);
        
        // Coin value 1 always works
        assert_eq!(solution.coin_change(vec![1], 100), 100);
    }

    #[test]
    fn test_all_coins_same() {
        let solution = setup();
        
        // All same value
        assert_eq!(solution.coin_change(vec![5, 5, 5], 15), 3);
        
        // Redundant coins don't affect result
        assert_eq!(solution.coin_change(vec![1, 2, 2, 5, 5], 11), 3);
    }

    #[test]
    fn test_large_amounts() {
        let solution = setup();
        
        // Large amount, small coins
        assert_eq!(solution.coin_change(vec![1], 100), 100);
        assert_eq!(solution.coin_change(vec![1, 5, 10, 25], 100), 4);
        
        // Maximum constraint
        assert_eq!(solution.coin_change(vec![1, 2, 5], 10000), 2000);
    }

    #[test]
    fn test_coin_combinations() {
        let solution = setup();
        
        // US coins
        assert_eq!(solution.coin_change(vec![1, 5, 10, 25], 41), 4); // 25+10+5+1
        
        // Euro coins (in cents)
        assert_eq!(solution.coin_change(vec![1, 2, 5, 10, 20, 50], 93), 5); // 50+20+20+2+1
        
        // Powers of 2
        assert_eq!(solution.coin_change(vec![1, 2, 4, 8, 16], 31), 5); // 16+8+4+2+1
    }

    #[test]
    fn test_minimum_coins_property() {
        let solution = setup();
        
        // Result should be <= amount (using coin of 1)
        let coins = vec![1, 5, 10];
        for amount in 1..=20 {
            let result = solution.coin_change(coins.clone(), amount);
            assert!(result <= amount && result > 0);
        }
    }

    #[test]
    fn test_specific_patterns() {
        let solution = setup();
        
        // Fibonacci-like coins
        assert_eq!(solution.coin_change(vec![1, 2, 3, 5, 8], 13), 2); // 8+5
        
        // Prime coins
        assert_eq!(solution.coin_change(vec![2, 3, 5, 7], 17), 3); // 7+5+5
        
        // Consecutive integers
        assert_eq!(solution.coin_change(vec![1, 2, 3, 4, 5], 11), 3); // 5+5+1
    }

    #[test]
    fn test_zero_amount() {
        let solution = setup();
        
        // Zero amount always needs 0 coins
        assert_eq!(solution.coin_change(vec![1, 2, 5], 0), 0);
        assert_eq!(solution.coin_change(vec![100, 200], 0), 0);
        assert_eq!(solution.coin_change(vec![7], 0), 0);
    }
}