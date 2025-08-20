//! # Problem 70: Climbing Stairs
//!
//! You are climbing a staircase. It takes `n` steps to reach the top.
//!
//! Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
//!
//! ## Examples
//!
//! ```text
//! Input: n = 2
//! Output: 2
//! Explanation: There are two ways to climb to the top.
//! 1. 1 step + 1 step
//! 2. 2 steps
//! ```
//!
//! ```text
//! Input: n = 3
//! Output: 3
//! Explanation: There are three ways to climb to the top.
//! 1. 1 step + 1 step + 1 step
//! 2. 1 step + 2 steps
//! 3. 2 steps + 1 step
//! ```
//!
//! ## Constraints
//!
//! * 1 <= n <= 45

/// Solution for Climbing Stairs problem
pub struct Solution;

impl Solution {
    /// Creates a new instance of Solution
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Dynamic Programming with Space Optimization (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Realize this is a Fibonacci sequence problem
    /// 2. To reach step n, we can come from step n-1 or n-2
    /// 3. So ways(n) = ways(n-1) + ways(n-2)
    /// 4. Use two variables to track last two values
    /// 
    /// **Time Complexity:** O(n) - Single pass through n steps
    /// **Space Complexity:** O(1) - Only using two variables
    /// 
    /// **Key Insight:** This is essentially the Fibonacci sequence!
    /// - To reach step n, we either:
    ///   - Take 1 step from position n-1, OR
    ///   - Take 2 steps from position n-2
    /// - Total ways = ways to reach n-1 + ways to reach n-2
    /// 
    /// **Why this is optimal:**
    /// - Linear time is optimal (must compute all values)
    /// - Constant space using only two variables
    /// - No redundant calculations
    /// 
    /// **Visualization:**
    /// ```text
    /// n=1: 1 way [1]
    /// n=2: 2 ways [1,1] or [2]  
    /// n=3: 3 ways [1,1,1], [1,2], [2,1]
    /// n=4: 5 ways [1,1,1,1], [1,1,2], [1,2,1], [2,1,1], [2,2]
    /// Pattern: 1, 2, 3, 5, 8, 13... (Fibonacci!)
    /// ```
    pub fn climb_stairs(&self, n: i32) -> i32 {
        if n <= 2 {
            return n;
        }
        
        let mut prev2 = 1; // ways to reach step n-2
        let mut prev1 = 2; // ways to reach step n-1
        
        for _ in 3..=n {
            let current = prev1 + prev2;
            prev2 = prev1;
            prev1 = current;
        }
        
        prev1
    }

    /// # Approach 2: Dynamic Programming with Array
    /// 
    /// **Algorithm:**
    /// 1. Create dp array where dp[i] = ways to reach step i
    /// 2. Base cases: dp[0]=1, dp[1]=1  
    /// 3. For each step i: dp[i] = dp[i-1] + dp[i-2]
    /// 4. Return dp[n]
    /// 
    /// **Time Complexity:** O(n) - Fill array once
    /// **Space Complexity:** O(n) - Store entire dp array
    /// 
    /// **DP State Definition:**
    /// - dp[i] = number of distinct ways to reach step i
    /// 
    /// **Recurrence Relation:**
    /// - dp[i] = dp[i-1] + dp[i-2] for i >= 2
    /// - Base: dp[0] = 1, dp[1] = 1
    /// 
    /// **When to use:** When you need to track all intermediate values
    pub fn climb_stairs_dp_array(&self, n: i32) -> i32 {
        if n <= 2 {
            return n;
        }
        
        let n = n as usize;
        let mut dp = vec![0; n + 1];
        dp[1] = 1;
        dp[2] = 2;
        
        for i in 3..=n {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        
        dp[n]
    }

    /// # Approach 3: Recursive with Memoization
    /// 
    /// **Algorithm:**
    /// 1. Define recursive function: ways(n) = ways(n-1) + ways(n-2)
    /// 2. Use memoization to avoid redundant calculations
    /// 3. Base cases: ways(1) = 1, ways(2) = 2
    /// 
    /// **Time Complexity:** O(n) - Each subproblem solved once
    /// **Space Complexity:** O(n) - Memoization table + recursion stack
    /// 
    /// **Why memoization is crucial:**
    /// - Without it, time complexity would be O(2^n)
    /// - Many subproblems are solved multiple times
    /// - Memoization ensures each subproblem is solved only once
    /// 
    /// **When to use:** When problem naturally fits recursive thinking
    pub fn climb_stairs_memo(&self, n: i32) -> i32 {
        let mut memo = vec![-1; (n + 1) as usize];
        self.climb_stairs_memo_helper(n, &mut memo)
    }
    
    fn climb_stairs_memo_helper(&self, n: i32, memo: &mut Vec<i32>) -> i32 {
        if n <= 2 {
            return n;
        }
        
        if memo[n as usize] != -1 {
            return memo[n as usize];
        }
        
        let result = self.climb_stairs_memo_helper(n - 1, memo) + 
                    self.climb_stairs_memo_helper(n - 2, memo);
        memo[n as usize] = result;
        result
    }

    /// # Approach 4: Matrix Exponentiation (Advanced)
    /// 
    /// **Algorithm:**
    /// 1. Express Fibonacci recurrence as matrix multiplication
    /// 2. [F(n+1), F(n)] = [F(n), F(n-1)] * [[1,1],[1,0]]
    /// 3. Use fast matrix exponentiation
    /// 
    /// **Time Complexity:** O(log n) - Matrix exponentiation
    /// **Space Complexity:** O(1) - Only store 2x2 matrices
    /// 
    /// **Mathematical Foundation:**
    /// ```text
    /// [F(n+1)]   [1 1] ^ n   [F(1)]
    /// [F(n)  ] = [1 0]     * [F(0)]
    /// ```
    /// 
    /// **Why this approach is interesting:**
    /// - Achieves logarithmic time complexity
    /// - Demonstrates advanced mathematical optimization
    /// - Useful for very large n (though not needed for n <= 45)
    pub fn climb_stairs_matrix(&self, n: i32) -> i32 {
        if n <= 2 {
            return n;
        }
        
        let mut result = [[1, 1], [1, 0]];
        let _base = [[1, 1], [1, 0]];
        let mut power = n - 2;
        
        // Fast matrix exponentiation
        let mut answer = [[1, 0], [0, 1]]; // Identity matrix
        
        while power > 0 {
            if power % 2 == 1 {
                answer = self.matrix_multiply(&answer, &result);
            }
            result = self.matrix_multiply(&result, &result);
            power /= 2;
        }
        
        // Result is in answer[0][0] * 2 + answer[0][1] * 1
        answer[0][0].saturating_mul(2).saturating_add(answer[0][1])
    }
    
    fn matrix_multiply(&self, a: &[[i32; 2]; 2], b: &[[i32; 2]; 2]) -> [[i32; 2]; 2] {
        [
            [
                a[0][0].saturating_mul(b[0][0]).saturating_add(a[0][1].saturating_mul(b[1][0])),
                a[0][0].saturating_mul(b[0][1]).saturating_add(a[0][1].saturating_mul(b[1][1])),
            ],
            [
                a[1][0].saturating_mul(b[0][0]).saturating_add(a[1][1].saturating_mul(b[1][0])),
                a[1][0].saturating_mul(b[0][1]).saturating_add(a[1][1].saturating_mul(b[1][1])),
            ],
        ]
    }

    /// # Approach 5: Binet's Formula (Mathematical)
    /// 
    /// **Algorithm:**
    /// 1. Use closed-form formula for Fibonacci numbers
    /// 2. F(n) = (φ^n - ψ^n) / √5
    /// 3. Where φ = (1 + √5) / 2 and ψ = (1 - √5) / 2
    /// 
    /// **Time Complexity:** O(1) - Direct calculation
    /// **Space Complexity:** O(1) - Only store constants
    /// 
    /// **Mathematical Basis:**
    /// - Golden ratio formula for Fibonacci sequence
    /// - Result needs adjustment since our sequence starts differently
    /// 
    /// **Limitations:**
    /// - Floating point precision issues for large n
    /// - Not exact for very large values
    /// 
    /// **When to use:** Theoretical interest or when approximate answer is acceptable
    pub fn climb_stairs_binet(&self, n: i32) -> i32 {
        if n <= 2 {
            return n;
        }
        
        let sqrt5 = (5.0_f64).sqrt();
        let phi = (1.0 + sqrt5) / 2.0;
        let psi = (1.0 - sqrt5) / 2.0;
        
        // Adjust for our sequence (starts with 1, 2 instead of 0, 1)
        let n = n + 1;
        let result = (phi.powi(n) - psi.powi(n)) / sqrt5;
        
        result.round() as i32
    }

    /// # Approach 6: Recursive (Inefficient - Educational)
    /// 
    /// **Algorithm:**
    /// 1. Base cases: n=1 returns 1, n=2 returns 2
    /// 2. Recursive case: ways(n) = ways(n-1) + ways(n-2)
    /// 
    /// **Time Complexity:** O(2^n) - Exponential due to redundant calculations
    /// **Space Complexity:** O(n) - Recursion stack depth
    /// 
    /// **Why this is inefficient:**
    /// - Solves same subproblems multiple times
    /// - Tree of recursive calls grows exponentially
    /// - ways(5) calls ways(4) and ways(3), ways(4) also calls ways(3)
    /// 
    /// **When to use:** Never in practice, only for understanding the problem
    pub fn climb_stairs_recursive(&self, n: i32) -> i32 {
        if n <= 2 {
            return n;
        }
        
        self.climb_stairs_recursive(n - 1) + self.climb_stairs_recursive(n - 2)
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
        
        // Examples from problem
        assert_eq!(solution.climb_stairs(2), 2);
        assert_eq!(solution.climb_stairs(3), 3);
        
        // Single step
        assert_eq!(solution.climb_stairs(1), 1);
        
        // Four steps
        assert_eq!(solution.climb_stairs(4), 5);
    }

    #[test]
    fn test_fibonacci_pattern() {
        let solution = setup();
        
        // Verify Fibonacci sequence: 1, 2, 3, 5, 8, 13, 21, 34...
        assert_eq!(solution.climb_stairs(1), 1);
        assert_eq!(solution.climb_stairs(2), 2);
        assert_eq!(solution.climb_stairs(3), 3);
        assert_eq!(solution.climb_stairs(4), 5);
        assert_eq!(solution.climb_stairs(5), 8);
        assert_eq!(solution.climb_stairs(6), 13);
        assert_eq!(solution.climb_stairs(7), 21);
        assert_eq!(solution.climb_stairs(8), 34);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        for n in 1..=20 {
            let result1 = solution.climb_stairs(n);
            let result2 = solution.climb_stairs_dp_array(n);
            let result3 = solution.climb_stairs_memo(n);
            let result4 = solution.climb_stairs_matrix(n);
            let result5 = solution.climb_stairs_binet(n);
            // Skip recursive for n > 10 due to exponential time
            
            assert_eq!(result1, result2, "Optimized vs DP array mismatch for n={}", n);
            assert_eq!(result2, result3, "DP array vs Memoization mismatch for n={}", n);
            assert_eq!(result3, result4, "Memoization vs Matrix mismatch for n={}", n);
            assert_eq!(result4, result5, "Matrix vs Binet mismatch for n={}", n);
            
            if n <= 10 {
                let result6 = solution.climb_stairs_recursive(n);
                assert_eq!(result1, result6, "Optimized vs Recursive mismatch for n={}", n);
            }
        }
    }

    #[test]
    fn test_larger_values() {
        let solution = setup();
        
        // Test with larger values (up to constraint limit)
        assert_eq!(solution.climb_stairs(10), 89);
        assert_eq!(solution.climb_stairs(15), 987);
        assert_eq!(solution.climb_stairs(20), 10946);
        assert_eq!(solution.climb_stairs(35), 14930352);
        assert_eq!(solution.climb_stairs(45), 1836311903);
    }

    #[test]
    fn test_mathematical_properties() {
        let solution = setup();
        
        // Property: ways(n) = ways(n-1) + ways(n-2)
        for n in 3..=20 {
            let ways_n = solution.climb_stairs(n);
            let ways_n_minus_1 = solution.climb_stairs(n - 1);
            let ways_n_minus_2 = solution.climb_stairs(n - 2);
            
            assert_eq!(ways_n, ways_n_minus_1 + ways_n_minus_2,
                      "Fibonacci property failed for n={}", n);
        }
        
        // Property: ways(n) >= n for n >= 3
        for n in 3..=20 {
            assert!(solution.climb_stairs(n) >= n,
                   "Ways should be at least n for n={}", n);
        }
    }

    #[test]
    fn test_boundary_values() {
        let solution = setup();
        
        // Minimum value
        assert_eq!(solution.climb_stairs(1), 1);
        
        // Maximum value per constraint
        assert_eq!(solution.climb_stairs(45), 1836311903);
    }

    #[test]
    fn test_growth_rate() {
        let solution = setup();
        
        // Verify exponential growth pattern
        // Fibonacci ratio approaches golden ratio (~1.618)
        for n in 5..=10 {
            let current = solution.climb_stairs(n) as f64;
            let prev = solution.climb_stairs(n - 1) as f64;
            let ratio = current / prev;
            // Ratio should be between 1.5 and 1.7 (approaching golden ratio)
            assert!(ratio > 1.5 && ratio < 1.7, 
                   "Growth rate ratio {} out of bounds at n={}", ratio, n);
        }
    }

    #[test]
    fn test_specific_scenarios() {
        let solution = setup();
        
        // Verify specific cases with manual calculation
        // n=4: [1,1,1,1], [1,1,2], [1,2,1], [2,1,1], [2,2] = 5 ways
        assert_eq!(solution.climb_stairs(4), 5);
        
        // n=5: 8 ways (manually verified)
        assert_eq!(solution.climb_stairs(5), 8);
        
        // n=6: 13 ways (Fibonacci sequence)
        assert_eq!(solution.climb_stairs(6), 13);
    }

    #[test]
    fn test_matrix_multiplication() {
        let solution = setup();
        
        // Test matrix multiplication helper
        let a = [[1, 2], [3, 4]];
        let b = [[5, 6], [7, 8]];
        let result = solution.matrix_multiply(&a, &b);
        
        // Verify matrix multiplication
        assert_eq!(result[0][0], 19); // 1*5 + 2*7
        assert_eq!(result[0][1], 22); // 1*6 + 2*8
        assert_eq!(result[1][0], 43); // 3*5 + 4*7
        assert_eq!(result[1][1], 50); // 3*6 + 4*8
    }

    #[test]
    fn test_performance_characteristics() {
        let solution = setup();
        
        // All non-recursive methods should handle n=45 efficiently
        let result_optimal = solution.climb_stairs(45);
        let result_dp = solution.climb_stairs_dp_array(45);
        let result_memo = solution.climb_stairs_memo(45);
        let result_matrix = solution.climb_stairs_matrix(45);
        let result_binet = solution.climb_stairs_binet(45);
        
        assert_eq!(result_optimal, 1836311903);
        assert_eq!(result_dp, 1836311903);
        assert_eq!(result_memo, 1836311903);
        assert_eq!(result_matrix, 1836311903);
        assert_eq!(result_binet, 1836311903);
    }
}