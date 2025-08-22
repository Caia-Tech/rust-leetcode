//! # Problem 354: Russian Doll Envelopes
//!
//! You are given a 2D array of integers `envelopes` where `envelopes[i] = [wᵢ, hᵢ]` represents 
//! the width and the height of an envelope.
//!
//! One envelope can fit into another if and only if both the width and height of one envelope 
//! are greater than the other envelope's width and height.
//!
//! Return the maximum number of envelopes you can Russian doll (i.e., put one inside the other).
//!
//! Note: You cannot rotate an envelope.
//!
//! ## Examples
//!
//! ```
//! Input: envelopes = [[5,4],[6,4],[6,7],[2,3]]
//! Output: 3
//! Explanation: The maximum number of envelopes you can Russian doll is 3 ([2,3] => [5,4] => [6,7]).
//! ```
//!
//! ```
//! Input: envelopes = [[1,1],[1,1],[1,1]]
//! Output: 1
//! ```

use std::cmp::{max, min, Ordering};
use std::collections::{HashMap, BTreeSet};

/// Solution struct for Russian Doll Envelopes problem
pub struct Solution;

impl Solution {
    /// Approach 1: Dynamic Programming with Sorting
    ///
    /// Sort envelopes by width (ascending) and height (descending for same width).
    /// Then find LIS (Longest Increasing Subsequence) on heights.
    ///
    /// Time Complexity: O(n²) where n is the number of envelopes
    /// Space Complexity: O(n) for DP array
    pub fn max_envelopes_dp(mut envelopes: Vec<Vec<i32>>) -> i32 {
        if envelopes.is_empty() {
            return 0;
        }
        
        // Sort by width ascending, height descending for same width
        envelopes.sort_by(|a, b| {
            if a[0] == b[0] {
                b[1].cmp(&a[1])
            } else {
                a[0].cmp(&b[0])
            }
        });
        
        let n = envelopes.len();
        let mut dp = vec![1; n];
        let mut max_len = 1;
        
        for i in 1..n {
            for j in 0..i {
                if envelopes[j][1] < envelopes[i][1] {
                    dp[i] = max(dp[i], dp[j] + 1);
                }
            }
            max_len = max(max_len, dp[i]);
        }
        
        max_len
    }
    
    /// Approach 2: Binary Search with LIS Optimization
    ///
    /// Uses binary search to find LIS more efficiently than O(n²) DP.
    /// Maintains a sorted subsequence for binary search.
    ///
    /// Time Complexity: O(n log n)
    /// Space Complexity: O(n)
    pub fn max_envelopes_binary_search(mut envelopes: Vec<Vec<i32>>) -> i32 {
        if envelopes.is_empty() {
            return 0;
        }
        
        // Sort by width ascending, height descending for same width
        envelopes.sort_by(|a, b| {
            if a[0] == b[0] {
                b[1].cmp(&a[1])
            } else {
                a[0].cmp(&b[0])
            }
        });
        
        // Find LIS on heights using binary search
        let mut lis = Vec::new();
        
        for envelope in envelopes {
            let height = envelope[1];
            
            match lis.binary_search(&height) {
                Ok(_) => {} // Height already exists, skip
                Err(pos) => {
                    if pos == lis.len() {
                        lis.push(height);
                    } else {
                        lis[pos] = height;
                    }
                }
            }
        }
        
        lis.len() as i32
    }
    
    /// Approach 3: Segment Tree Based Solution
    ///
    /// Uses a segment tree to efficiently query and update maximum values.
    /// More complex but demonstrates advanced data structures.
    ///
    /// Time Complexity: O(n log n)
    /// Space Complexity: O(n)
    pub fn max_envelopes_segment_tree(mut envelopes: Vec<Vec<i32>>) -> i32 {
        if envelopes.is_empty() {
            return 0;
        }
        
        // Sort and compress heights
        envelopes.sort_by(|a, b| {
            if a[0] == b[0] {
                b[1].cmp(&a[1])
            } else {
                a[0].cmp(&b[0])
            }
        });
        
        let heights: Vec<i32> = envelopes.iter().map(|e| e[1]).collect();
        let mut sorted_heights = heights.clone();
        sorted_heights.sort_unstable();
        sorted_heights.dedup();
        
        let height_map: HashMap<i32, usize> = sorted_heights
            .iter()
            .enumerate()
            .map(|(i, &h)| (h, i))
            .collect();
        
        let m = sorted_heights.len();
        let mut tree = vec![0; 4 * m];
        
        fn update(tree: &mut Vec<i32>, node: usize, start: usize, end: usize, idx: usize, val: i32) {
            if start == end {
                tree[node] = max(tree[node], val);
                return;
            }
            
            let mid = (start + end) / 2;
            if idx <= mid {
                update(tree, 2 * node + 1, start, mid, idx, val);
            } else {
                update(tree, 2 * node + 2, mid + 1, end, idx, val);
            }
            
            tree[node] = max(tree[2 * node + 1], tree[2 * node + 2]);
        }
        
        fn query(tree: &Vec<i32>, node: usize, start: usize, end: usize, l: usize, r: usize) -> i32 {
            if r < start || l > end {
                return 0;
            }
            
            if l <= start && end <= r {
                return tree[node];
            }
            
            let mid = (start + end) / 2;
            let left_max = query(tree, 2 * node + 1, start, mid, l, r);
            let right_max = query(tree, 2 * node + 2, mid + 1, end, l, r);
            
            max(left_max, right_max)
        }
        
        let mut result = 0;
        
        for envelope in envelopes {
            let height = envelope[1];
            let idx = height_map[&height];
            
            let max_prev = if idx > 0 {
                query(&tree, 0, 0, m - 1, 0, idx - 1)
            } else {
                0
            };
            
            let new_val = max_prev + 1;
            update(&mut tree, 0, 0, m - 1, idx, new_val);
            result = max(result, new_val);
        }
        
        result
    }
    
    /// Approach 4: Fenwick Tree (Binary Indexed Tree) Solution
    ///
    /// Uses a Fenwick tree for efficient range queries and updates.
    /// Alternative to segment tree with simpler implementation.
    ///
    /// Time Complexity: O(n log n)
    /// Space Complexity: O(n)
    pub fn max_envelopes_fenwick(mut envelopes: Vec<Vec<i32>>) -> i32 {
        if envelopes.is_empty() {
            return 0;
        }
        
        envelopes.sort_by(|a, b| {
            if a[0] == b[0] {
                b[1].cmp(&a[1])
            } else {
                a[0].cmp(&b[0])
            }
        });
        
        // Compress heights
        let heights: Vec<i32> = envelopes.iter().map(|e| e[1]).collect();
        let mut sorted_heights = heights.clone();
        sorted_heights.sort_unstable();
        sorted_heights.dedup();
        
        let height_map: HashMap<i32, usize> = sorted_heights
            .iter()
            .enumerate()
            .map(|(i, &h)| (h, i + 1))
            .collect();
        
        struct FenwickTree {
            tree: Vec<i32>,
        }
        
        impl FenwickTree {
            fn new(n: usize) -> Self {
                Self {
                    tree: vec![0; n + 1],
                }
            }
            
            fn update(&mut self, mut idx: usize, val: i32) {
                while idx < self.tree.len() {
                    self.tree[idx] = max(self.tree[idx], val);
                    idx += idx & (!idx + 1);
                }
            }
            
            fn query(&self, mut idx: usize) -> i32 {
                let mut result = 0;
                while idx > 0 {
                    result = max(result, self.tree[idx]);
                    idx -= idx & (!idx + 1);
                }
                result
            }
        }
        
        let mut fenwick = FenwickTree::new(sorted_heights.len());
        let mut result = 0;
        
        for envelope in envelopes {
            let height = envelope[1];
            let idx = height_map[&height];
            
            let max_prev = if idx > 1 {
                fenwick.query(idx - 1)
            } else {
                0
            };
            
            let new_val = max_prev + 1;
            fenwick.update(idx, new_val);
            result = max(result, new_val);
        }
        
        result
    }
    
    /// Approach 5: Divide and Conquer with Memoization
    ///
    /// Divides the problem into subproblems and uses memoization.
    /// Different perspective on the problem.
    ///
    /// Time Complexity: O(n²) with memoization
    /// Space Complexity: O(n²) for memoization
    pub fn max_envelopes_divide_conquer(mut envelopes: Vec<Vec<i32>>) -> i32 {
        if envelopes.is_empty() {
            return 0;
        }
        
        envelopes.sort_by(|a, b| {
            if a[0] == b[0] {
                a[1].cmp(&b[1])
            } else {
                a[0].cmp(&b[0])
            }
        });
        
        let n = envelopes.len();
        let mut memo: HashMap<(usize, Option<usize>), i32> = HashMap::new();
        
        fn solve(
            envelopes: &Vec<Vec<i32>>,
            idx: usize,
            prev: Option<usize>,
            memo: &mut HashMap<(usize, Option<usize>), i32>
        ) -> i32 {
            if idx == envelopes.len() {
                return 0;
            }
            
            if let Some(&result) = memo.get(&(idx, prev)) {
                return result;
            }
            
            // Option 1: Skip current envelope
            let skip = solve(envelopes, idx + 1, prev, memo);
            
            // Option 2: Take current envelope if valid
            let take = if let Some(p) = prev {
                if envelopes[p][0] < envelopes[idx][0] && envelopes[p][1] < envelopes[idx][1] {
                    1 + solve(envelopes, idx + 1, Some(idx), memo)
                } else {
                    0
                }
            } else {
                1 + solve(envelopes, idx + 1, Some(idx), memo)
            };
            
            let result = max(skip, take);
            memo.insert((idx, prev), result);
            result
        }
        
        solve(&envelopes, 0, None, &mut memo)
    }
    
    /// Approach 6: Patience Sorting Algorithm
    ///
    /// Uses the patience sorting algorithm to find LIS.
    /// Classic algorithm for finding longest increasing subsequence.
    ///
    /// Time Complexity: O(n log n)
    /// Space Complexity: O(n)
    pub fn max_envelopes_patience_sort(mut envelopes: Vec<Vec<i32>>) -> i32 {
        if envelopes.is_empty() {
            return 0;
        }
        
        // Sort by width ascending, height descending for same width
        envelopes.sort_by(|a, b| {
            match a[0].cmp(&b[0]) {
                Ordering::Equal => b[1].cmp(&a[1]),
                other => other,
            }
        });
        
        // Patience sorting on heights
        let mut piles: Vec<i32> = Vec::new();
        
        for envelope in envelopes {
            let height = envelope[1];
            
            // Binary search for the leftmost pile that can hold this height
            let pos = piles.binary_search(&height).unwrap_or_else(|e| e);
            
            if pos == piles.len() {
                piles.push(height);
            } else {
                piles[pos] = height;
            }
        }
        
        piles.len() as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_1() {
        let envelopes = vec![vec![5,4], vec![6,4], vec![6,7], vec![2,3]];
        let expected = 3;
        
        assert_eq!(Solution::max_envelopes_dp(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_binary_search(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_segment_tree(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_fenwick(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_divide_conquer(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_patience_sort(envelopes), expected);
    }

    #[test]
    fn test_example_2() {
        let envelopes = vec![vec![1,1], vec![1,1], vec![1,1]];
        let expected = 1;
        
        assert_eq!(Solution::max_envelopes_dp(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_binary_search(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_segment_tree(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_fenwick(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_divide_conquer(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_patience_sort(envelopes), expected);
    }

    #[test]
    fn test_single_envelope() {
        let envelopes = vec![vec![10, 20]];
        let expected = 1;
        
        assert_eq!(Solution::max_envelopes_dp(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_binary_search(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_segment_tree(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_fenwick(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_divide_conquer(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_patience_sort(envelopes), expected);
    }

    #[test]
    fn test_increasing_sequence() {
        let envelopes = vec![vec![1,1], vec![2,2], vec![3,3], vec![4,4]];
        let expected = 4;
        
        assert_eq!(Solution::max_envelopes_dp(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_binary_search(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_segment_tree(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_fenwick(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_divide_conquer(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_patience_sort(envelopes), expected);
    }

    #[test]
    fn test_decreasing_sequence() {
        let envelopes = vec![vec![4,4], vec![3,3], vec![2,2], vec![1,1]];
        let expected = 4; // After sorting becomes [1,1], [2,2], [3,3], [4,4] which all fit
        
        assert_eq!(Solution::max_envelopes_dp(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_binary_search(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_segment_tree(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_fenwick(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_divide_conquer(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_patience_sort(envelopes), expected);
    }

    #[test]
    fn test_same_width() {
        let envelopes = vec![vec![1,3], vec![1,4], vec![1,5], vec![2,6]];
        let expected = 2;
        
        assert_eq!(Solution::max_envelopes_dp(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_binary_search(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_segment_tree(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_fenwick(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_divide_conquer(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_patience_sort(envelopes), expected);
    }

    #[test]
    fn test_same_height() {
        let envelopes = vec![vec![1,1], vec![2,1], vec![3,1], vec![4,2]];
        let expected = 2;
        
        assert_eq!(Solution::max_envelopes_dp(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_binary_search(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_segment_tree(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_fenwick(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_divide_conquer(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_patience_sort(envelopes), expected);
    }

    #[test]
    fn test_complex_case() {
        let envelopes = vec![
            vec![10,8], vec![1,12], vec![6,15], vec![2,18], 
            vec![9,7], vec![2,9], vec![17,4], vec![18,3]
        ];
        // After sorting: [1,12], [2,18], [2,9], [6,15], [9,7], [10,8], [17,4], [18,3]
        // LIS on heights after sorting with proper handling: [3,4,7,8,12] = 2
        let expected = 2;
        
        assert_eq!(Solution::max_envelopes_dp(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_binary_search(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_segment_tree(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_fenwick(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_divide_conquer(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_patience_sort(envelopes), expected);
    }

    #[test]
    fn test_zigzag_pattern() {
        let envelopes = vec![vec![1,10], vec![2,1], vec![3,9], vec![4,2], vec![5,8]];
        let expected = 3;
        
        assert_eq!(Solution::max_envelopes_dp(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_binary_search(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_segment_tree(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_fenwick(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_divide_conquer(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_patience_sort(envelopes), expected);
    }

    #[test]
    fn test_large_values() {
        let envelopes = vec![
            vec![100000, 100000], vec![99999, 99999], 
            vec![100001, 100001], vec![1, 1]
        ];
        let expected = 4; // [1,1] -> [99999,99999] -> [100000,100000] -> [100001,100001]
        
        assert_eq!(Solution::max_envelopes_dp(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_binary_search(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_segment_tree(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_fenwick(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_divide_conquer(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_patience_sort(envelopes), expected);
    }

    #[test]
    fn test_empty_input() {
        let envelopes: Vec<Vec<i32>> = vec![];
        let expected = 0;
        
        assert_eq!(Solution::max_envelopes_dp(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_binary_search(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_segment_tree(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_fenwick(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_divide_conquer(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_patience_sort(envelopes), expected);
    }

    #[test]
    fn test_two_envelopes_fit() {
        let envelopes = vec![vec![1,1], vec![2,2]];
        let expected = 2;
        
        assert_eq!(Solution::max_envelopes_dp(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_binary_search(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_segment_tree(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_fenwick(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_divide_conquer(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_patience_sort(envelopes), expected);
    }

    #[test]
    fn test_two_envelopes_no_fit() {
        let envelopes = vec![vec![1,2], vec![2,1]];
        let expected = 1;
        
        assert_eq!(Solution::max_envelopes_dp(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_binary_search(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_segment_tree(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_fenwick(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_divide_conquer(envelopes.clone()), expected);
        assert_eq!(Solution::max_envelopes_patience_sort(envelopes), expected);
    }

    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            vec![vec![5,4], vec![6,4], vec![6,7], vec![2,3]],
            vec![vec![1,1], vec![1,1], vec![1,1]],
            vec![vec![1,1], vec![2,2], vec![3,3], vec![4,4]],
            vec![vec![4,4], vec![3,3], vec![2,2], vec![1,1]],
            vec![vec![1,3], vec![1,4], vec![1,5], vec![2,6]],
            vec![vec![1,1], vec![2,1], vec![3,1], vec![4,2]],
            vec![vec![10,8], vec![1,12], vec![6,15], vec![2,18], vec![9,7], vec![2,9], vec![17,4], vec![18,3]],
            vec![vec![1,10], vec![2,1], vec![3,9], vec![4,2], vec![5,8]],
        ];
        
        for envelopes in test_cases {
            let result1 = Solution::max_envelopes_dp(envelopes.clone());
            let result2 = Solution::max_envelopes_binary_search(envelopes.clone());
            let result3 = Solution::max_envelopes_segment_tree(envelopes.clone());
            let result4 = Solution::max_envelopes_fenwick(envelopes.clone());
            let result5 = Solution::max_envelopes_divide_conquer(envelopes.clone());
            let result6 = Solution::max_envelopes_patience_sort(envelopes.clone());
            
            assert_eq!(result1, result2, "DP vs Binary Search mismatch");
            assert_eq!(result2, result3, "Binary Search vs Segment Tree mismatch");
            assert_eq!(result3, result4, "Segment Tree vs Fenwick mismatch");
            assert_eq!(result4, result5, "Fenwick vs Divide & Conquer mismatch");
            assert_eq!(result5, result6, "Divide & Conquer vs Patience Sort mismatch");
        }
    }
}

// Author: Marvin Tutt, Caia Tech
// Problem: LeetCode 354 - Russian Doll Envelopes
// Approaches: DP with sorting, Binary search LIS, Segment tree, Fenwick tree,
//            Divide & conquer with memoization, Patience sorting
// All approaches find the maximum number of nested envelopes efficiently