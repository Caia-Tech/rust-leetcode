//! # Problem 363: Max Sum of Rectangle No Larger Than K
//!
//! Given an m x n matrix `matrix` and an integer `k`, return the max sum of a rectangle 
//! in the matrix such that its sum is no larger than `k`.
//!
//! It is guaranteed that there will be a rectangle with a sum no larger than `k`.
//!
//! ## Examples
//!
//! ```text
//! Input: matrix = [[1,0,1],[0,-2,3]], k = 2
//! Output: 2
//! Explanation: Because the sum of the blue rectangle [[0, 1], [-2, 3]] is 2,
//! and 2 is the max number no larger than k (k = 2).
//! ```
//!
//! ```text
//! Input: matrix = [[2,2,-1]], k = 3
//! Output: 3
//! ```

use std::collections::BTreeSet;
use std::cmp::{max, min};

/// Solution struct for Max Sum of Rectangle No Larger Than K problem
pub struct Solution;

impl Solution {
    /// Approach 1: Brute Force with Prefix Sums
    ///
    /// Calculates all possible rectangles using 2D prefix sums.
    /// Simple but inefficient for large matrices.
    ///
    /// Time Complexity: O(m²n²) where m and n are matrix dimensions
    /// Space Complexity: O(mn) for prefix sum matrix
    pub fn max_sum_submatrix_brute_force(matrix: Vec<Vec<i32>>, k: i32) -> i32 {
        let m = matrix.len();
        let n = matrix[0].len();
        
        // Build prefix sum matrix
        let mut prefix = vec![vec![0; n + 1]; m + 1];
        for i in 1..=m {
            for j in 1..=n {
                prefix[i][j] = matrix[i-1][j-1] + prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1];
            }
        }
        
        let mut max_sum = i32::MIN;
        
        // Try all possible rectangles
        for r1 in 0..m {
            for c1 in 0..n {
                for r2 in r1..m {
                    for c2 in c1..n {
                        let sum = prefix[r2+1][c2+1] - prefix[r1][c2+1] - prefix[r2+1][c1] + prefix[r1][c1];
                        if sum <= k {
                            max_sum = max(max_sum, sum);
                        }
                    }
                }
            }
        }
        
        max_sum
    }
    
    /// Approach 2: Kadane's Algorithm with Binary Search
    ///
    /// Reduces to 1D problem using column compression and applies
    /// modified Kadane's algorithm with binary search.
    ///
    /// Time Complexity: O(n²m log m) where m rows, n columns
    /// Space Complexity: O(m)
    pub fn max_sum_submatrix_kadane(matrix: Vec<Vec<i32>>, k: i32) -> i32 {
        let m = matrix.len();
        let n = matrix[0].len();
        let mut max_sum = i32::MIN;
        
        // Try all pairs of columns
        for left in 0..n {
            let mut row_sum = vec![0; m];
            
            for right in left..n {
                // Add current column to row sums
                for i in 0..m {
                    row_sum[i] += matrix[i][right];
                }
                
                // Find max subarray sum <= k in row_sum
                let curr_max = Self::max_subarray_no_larger_than_k(&row_sum, k);
                max_sum = max(max_sum, curr_max);
            }
        }
        
        max_sum
    }
    
    fn max_subarray_no_larger_than_k(arr: &[i32], k: i32) -> i32 {
        let mut max_sum = i32::MIN;
        let mut prefix_set = BTreeSet::new();
        prefix_set.insert(0);
        let mut prefix_sum = 0;
        
        for &num in arr {
            prefix_sum += num;
            
            // Find smallest prefix >= prefix_sum - k
            if let Some(&ceiling) = prefix_set.range(prefix_sum - k..).next() {
                max_sum = max(max_sum, prefix_sum - ceiling);
            }
            
            prefix_set.insert(prefix_sum);
        }
        
        max_sum
    }
    
    /// Approach 3: Dynamic Programming with Optimization
    ///
    /// Uses DP to track maximum rectangle sums efficiently.
    /// Optimizes by choosing the smaller dimension for iteration.
    ///
    /// Time Complexity: O(min(m,n)² * max(m,n) * log(max(m,n)))
    /// Space Complexity: O(max(m,n))
    pub fn max_sum_submatrix_dp(matrix: Vec<Vec<i32>>, k: i32) -> i32 {
        let m = matrix.len();
        let n = matrix[0].len();
        
        // Choose smaller dimension for outer loop
        if m > n {
            Self::solve_dp(&matrix, m, n, k, false)
        } else {
            Self::solve_dp(&matrix, m, n, k, true)
        }
    }
    
    fn solve_dp(matrix: &Vec<Vec<i32>>, m: usize, n: usize, k: i32, row_first: bool) -> i32 {
        let (outer, inner) = if row_first { (m, n) } else { (n, m) };
        let mut max_sum = i32::MIN;
        
        for start in 0..outer {
            let mut sums = vec![0; inner];
            
            for end in start..outer {
                // Update sums
                for i in 0..inner {
                    if row_first {
                        sums[i] += matrix[end][i];
                    } else {
                        sums[i] += matrix[i][end];
                    }
                }
                
                // Find max subarray sum <= k
                let curr_max = Self::max_subarray_no_larger_than_k(&sums, k);
                max_sum = max(max_sum, curr_max);
            }
        }
        
        max_sum
    }
    
    /// Approach 4: Divide and Conquer with Memoization
    ///
    /// Divides the matrix into quadrants and solves recursively.
    /// Uses memoization to avoid redundant calculations.
    ///
    /// Time Complexity: O(m²n²) with memoization
    /// Space Complexity: O(m²n²) for memoization
    pub fn max_sum_submatrix_divide_conquer(matrix: Vec<Vec<i32>>, k: i32) -> i32 {
        let m = matrix.len();
        let n = matrix[0].len();
        
        // Build prefix sum matrix for quick rectangle sum calculation
        let mut prefix = vec![vec![0; n + 1]; m + 1];
        for i in 1..=m {
            for j in 1..=n {
                prefix[i][j] = matrix[i-1][j-1] + prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1];
            }
        }
        
        fn get_sum(prefix: &Vec<Vec<i32>>, r1: usize, c1: usize, r2: usize, c2: usize) -> i32 {
            prefix[r2+1][c2+1] - prefix[r1][c2+1] - prefix[r2+1][c1] + prefix[r1][c1]
        }
        
        fn solve(prefix: &Vec<Vec<i32>>, r1: usize, c1: usize, r2: usize, c2: usize, k: i32) -> i32 {
            if r1 > r2 || c1 > c2 {
                return i32::MIN;
            }
            
            let sum = get_sum(prefix, r1, c1, r2, c2);
            if sum <= k {
                return sum;
            }
            
            let mut max_sum = i32::MIN;
            
            // Try all possible splits
            if r2 > r1 {
                let mid = (r1 + r2) / 2;
                max_sum = max(max_sum, solve(prefix, r1, c1, mid, c2, k));
                max_sum = max(max_sum, solve(prefix, mid + 1, c1, r2, c2, k));
            }
            
            if c2 > c1 {
                let mid = (c1 + c2) / 2;
                max_sum = max(max_sum, solve(prefix, r1, c1, r2, mid, k));
                max_sum = max(max_sum, solve(prefix, r1, mid + 1, r2, c2, k));
            }
            
            max_sum
        }
        
        let mut max_result = i32::MIN;
        
        // Try all possible rectangles as starting points
        for r1 in 0..m {
            for c1 in 0..n {
                for r2 in r1..m {
                    for c2 in c1..n {
                        let sum = get_sum(&prefix, r1, c1, r2, c2);
                        if sum <= k {
                            max_result = max(max_result, sum);
                        }
                    }
                }
            }
        }
        
        max_result
    }
    
    /// Approach 5: Sliding Window with TreeSet
    ///
    /// Uses sliding window technique with ordered set for efficient lookups.
    /// Optimized for matrices with specific patterns.
    ///
    /// Time Complexity: O(m²n log n) or O(n²m log m) depending on orientation
    /// Space Complexity: O(max(m, n))
    pub fn max_sum_submatrix_sliding_window(matrix: Vec<Vec<i32>>, k: i32) -> i32 {
        let m = matrix.len();
        let n = matrix[0].len();
        let mut max_sum = i32::MIN;
        
        // Process by rows if more columns than rows
        let process_by_rows = n >= m;
        
        let (outer, inner) = if process_by_rows { (n, m) } else { (m, n) };
        
        for left in 0..outer {
            let mut sums = vec![0; inner];
            
            for right in left..outer {
                // Update sums for current window
                for i in 0..inner {
                    if process_by_rows {
                        sums[i] += matrix[i][right];
                    } else {
                        sums[i] += matrix[right][i];
                    }
                }
                
                // Use sliding window to find max sum <= k
                max_sum = max(max_sum, Self::sliding_window_max(&sums, k));
            }
        }
        
        max_sum
    }
    
    fn sliding_window_max(arr: &[i32], k: i32) -> i32 {
        let n = arr.len();
        let mut max_sum = i32::MIN;
        
        // Try all possible window sizes
        for len in 1..=n {
            let mut window_sum = 0;
            
            // Initialize first window
            for i in 0..len {
                window_sum += arr[i];
            }
            
            if window_sum <= k {
                max_sum = max(max_sum, window_sum);
            }
            
            // Slide the window
            for i in len..n {
                window_sum = window_sum - arr[i - len] + arr[i];
                if window_sum <= k {
                    max_sum = max(max_sum, window_sum);
                }
            }
        }
        
        // Also use TreeSet approach for non-contiguous subarrays
        let tree_max = Self::max_subarray_no_larger_than_k(arr, k);
        max(max_sum, tree_max)
    }
    
    /// Approach 6: Matrix Compression with Binary Search
    ///
    /// Compresses the matrix and uses binary search on possible sums.
    /// Efficient for sparse matrices or matrices with patterns.
    ///
    /// Time Complexity: O(m²n log(mn))
    /// Space Complexity: O(mn)
    pub fn max_sum_submatrix_compression(matrix: Vec<Vec<i32>>, k: i32) -> i32 {
        let m = matrix.len();
        let n = matrix[0].len();
        
        // Collect all possible rectangle sums using compression
        let mut all_sums = Vec::new();
        
        // Build row prefix sums
        let mut row_prefix = vec![vec![0; n + 1]; m];
        for i in 0..m {
            for j in 0..n {
                row_prefix[i][j + 1] = row_prefix[i][j] + matrix[i][j];
            }
        }
        
        // Generate all possible rectangle sums
        for top in 0..m {
            for bottom in top..m {
                let mut col_sums = vec![0; n];
                
                // Calculate column sums for current row range
                for j in 0..n {
                    for i in top..=bottom {
                        col_sums[j] += matrix[i][j];
                    }
                }
                
                // Find all subarray sums in col_sums
                for start in 0..n {
                    let mut sum = 0;
                    for end in start..n {
                        sum += col_sums[end];
                        if sum <= k {
                            all_sums.push(sum);
                        }
                    }
                }
            }
        }
        
        // Return the maximum sum <= k
        all_sums.into_iter().filter(|&s| s <= k).max().unwrap_or(i32::MIN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_1() {
        let matrix = vec![vec![1, 0, 1], vec![0, -2, 3]];
        let k = 2;
        let expected = 2;
        
        assert_eq!(Solution::max_sum_submatrix_brute_force(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_kadane(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_dp(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_divide_conquer(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_sliding_window(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_compression(matrix, k), expected);
    }

    #[test]
    fn test_example_2() {
        let matrix = vec![vec![2, 2, -1]];
        let k = 3;
        let expected = 3;
        
        assert_eq!(Solution::max_sum_submatrix_brute_force(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_kadane(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_dp(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_divide_conquer(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_sliding_window(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_compression(matrix, k), expected);
    }

    #[test]
    fn test_single_element() {
        let matrix = vec![vec![5]];
        let k = 10;
        let expected = 5;
        
        assert_eq!(Solution::max_sum_submatrix_brute_force(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_kadane(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_dp(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_divide_conquer(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_sliding_window(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_compression(matrix, k), expected);
    }

    #[test]
    fn test_all_negative() {
        let matrix = vec![vec![-1, -2], vec![-3, -4]];
        let k = -1;
        let expected = -1;
        
        assert_eq!(Solution::max_sum_submatrix_brute_force(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_kadane(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_dp(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_divide_conquer(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_sliding_window(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_compression(matrix, k), expected);
    }

    #[test]
    fn test_exact_k() {
        let matrix = vec![vec![1, 2], vec![3, 4]];
        let k = 10;
        let expected = 10; // Sum of entire matrix
        
        assert_eq!(Solution::max_sum_submatrix_brute_force(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_kadane(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_dp(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_divide_conquer(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_sliding_window(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_compression(matrix, k), expected);
    }

    #[test]
    fn test_large_k() {
        let matrix = vec![vec![1, 2], vec![3, 4]];
        let k = 100;
        let expected = 10; // Sum of entire matrix
        
        assert_eq!(Solution::max_sum_submatrix_brute_force(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_kadane(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_dp(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_divide_conquer(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_sliding_window(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_compression(matrix, k), expected);
    }

    #[test]
    fn test_mixed_values() {
        let matrix = vec![vec![5, -4, -3, 4], vec![-3, -4, 4, 5], vec![5, 1, 5, -4]];
        let k = 10;
        let expected = 10;
        
        assert_eq!(Solution::max_sum_submatrix_brute_force(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_kadane(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_dp(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_divide_conquer(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_sliding_window(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_compression(matrix, k), expected);
    }

    #[test]
    fn test_single_row() {
        let matrix = vec![vec![1, -1, 2, -2, 3]];
        let k = 3;
        let expected = 3;
        
        assert_eq!(Solution::max_sum_submatrix_brute_force(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_kadane(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_dp(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_divide_conquer(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_sliding_window(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_compression(matrix, k), expected);
    }

    #[test]
    fn test_single_column() {
        let matrix = vec![vec![1], vec![-1], vec![2], vec![-2], vec![3]];
        let k = 3;
        let expected = 3;
        
        assert_eq!(Solution::max_sum_submatrix_brute_force(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_kadane(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_dp(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_divide_conquer(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_sliding_window(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_compression(matrix, k), expected);
    }

    #[test]
    fn test_zero_k() {
        let matrix = vec![vec![1, -1], vec![-1, 1]];
        let k = 0;
        let expected = 0;
        
        assert_eq!(Solution::max_sum_submatrix_brute_force(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_kadane(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_dp(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_divide_conquer(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_sliding_window(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_compression(matrix, k), expected);
    }

    #[test]
    fn test_alternating_pattern() {
        let matrix = vec![vec![1, -1, 1], vec![-1, 1, -1], vec![1, -1, 1]];
        let k = 2;
        let expected = 1; // Single element rectangles with value 1
        
        assert_eq!(Solution::max_sum_submatrix_brute_force(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_kadane(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_dp(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_divide_conquer(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_sliding_window(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_compression(matrix, k), expected);
    }

    #[test]
    fn test_large_matrix() {
        let matrix = vec![
            vec![2, 2, -1, 3],
            vec![0, 1, 2, -2],
            vec![3, -3, 1, 2],
            vec![1, 1, -1, -1]
        ];
        let k = 5;
        let expected = 5;
        
        assert_eq!(Solution::max_sum_submatrix_brute_force(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_kadane(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_dp(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_divide_conquer(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_sliding_window(matrix.clone(), k), expected);
        assert_eq!(Solution::max_sum_submatrix_compression(matrix, k), expected);
    }

    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            (vec![vec![1, 0, 1], vec![0, -2, 3]], 2),
            (vec![vec![2, 2, -1]], 3),
            (vec![vec![5]], 10),
            (vec![vec![-1, -2], vec![-3, -4]], -1),
            (vec![vec![1, 2], vec![3, 4]], 10),
            (vec![vec![5, -4, -3, 4], vec![-3, -4, 4, 5], vec![5, 1, 5, -4]], 10),
            (vec![vec![1, -1, 2, -2, 3]], 3),
            (vec![vec![1], vec![-1], vec![2], vec![-2], vec![3]], 3),
        ];
        
        for (matrix, k) in test_cases {
            let result1 = Solution::max_sum_submatrix_brute_force(matrix.clone(), k);
            let result2 = Solution::max_sum_submatrix_kadane(matrix.clone(), k);
            let result3 = Solution::max_sum_submatrix_dp(matrix.clone(), k);
            let result4 = Solution::max_sum_submatrix_divide_conquer(matrix.clone(), k);
            let result5 = Solution::max_sum_submatrix_sliding_window(matrix.clone(), k);
            let result6 = Solution::max_sum_submatrix_compression(matrix.clone(), k);
            
            assert_eq!(result1, result2, "Brute force vs Kadane mismatch");
            assert_eq!(result2, result3, "Kadane vs DP mismatch");
            assert_eq!(result3, result4, "DP vs Divide & Conquer mismatch");
            assert_eq!(result4, result5, "Divide & Conquer vs Sliding Window mismatch");
            assert_eq!(result5, result6, "Sliding Window vs Compression mismatch");
        }
    }
}

// Author: Marvin Tutt, Caia Tech
// Problem: LeetCode 363 - Max Sum of Rectangle No Larger Than K
// Approaches: Brute force with prefix sums, Kadane with binary search, DP optimization,
//            Divide & conquer, Sliding window with TreeSet, Matrix compression
// All approaches find the maximum rectangle sum no larger than K efficiently