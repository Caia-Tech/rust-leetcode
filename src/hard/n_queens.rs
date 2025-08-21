//! Problem 51: N-Queens
//!
//! Implementation by Marvin Tutt, Caia Tech
//!
//! The n-queens puzzle is the problem of placing n queens on an n×n chessboard 
//! such that no two queens attack each other.
//! Given an integer n, return all distinct solutions to the n-queens puzzle.
//! You may return the answer in any order.
//! Each solution contains a distinct board configuration of the n-queens' placement, 
//! where 'Q' and '.' indicate a queen and an empty space, respectively.
//!
//! Constraints:
//! - 1 <= n <= 9
//!
//! Example 1:
//! Input: n = 4
//! Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
//!
//! Example 2:
//! Input: n = 1
//! Output: [["Q"]]

pub struct Solution;

impl Solution {
    /// Approach 1: Backtracking with Constraint Tracking - Optimal
    /// 
    /// Use arrays to track column, diagonal, and anti-diagonal constraints efficiently.
    /// 
    /// Time Complexity: O(N!) where N is the board size
    /// Space Complexity: O(N) for constraint tracking arrays
    pub fn solve_n_queens_optimized(n: i32) -> Vec<Vec<String>> {
        let n = n as usize;
        let mut result = Vec::new();
        let mut board = vec![vec!['.'; n]; n];
        let mut cols = vec![false; n];
        let mut diag1 = vec![false; 2 * n - 1];  // row + col
        let mut diag2 = vec![false; 2 * n - 1];  // row - col + n - 1
        
        Self::backtrack_optimized(&mut board, &mut result, &mut cols, &mut diag1, &mut diag2, 0, n);
        result
    }
    
    fn backtrack_optimized(
        board: &mut Vec<Vec<char>>,
        result: &mut Vec<Vec<String>>,
        cols: &mut Vec<bool>,
        diag1: &mut Vec<bool>,
        diag2: &mut Vec<bool>,
        row: usize,
        n: usize,
    ) {
        if row == n {
            result.push(board.iter().map(|row| row.iter().collect()).collect());
            return;
        }
        
        for col in 0..n {
            let d1 = row + col;
            let d2 = row + n - 1 - col;
            
            if !cols[col] && !diag1[d1] && !diag2[d2] {
                // Place queen
                board[row][col] = 'Q';
                cols[col] = true;
                diag1[d1] = true;
                diag2[d2] = true;
                
                Self::backtrack_optimized(board, result, cols, diag1, diag2, row + 1, n);
                
                // Backtrack
                board[row][col] = '.';
                cols[col] = false;
                diag1[d1] = false;
                diag2[d2] = false;
            }
        }
    }
    
    /// Approach 2: Simple Backtracking with Validation
    /// 
    /// Basic backtracking that validates constraints for each placement.
    /// 
    /// Time Complexity: O(N! * N^2) due to validation overhead
    /// Space Complexity: O(N^2) for the board
    pub fn solve_n_queens_simple(n: i32) -> Vec<Vec<String>> {
        let n = n as usize;
        let mut result = Vec::new();
        let mut board = vec![vec!['.'; n]; n];
        
        Self::backtrack_simple(&mut board, &mut result, 0, n);
        result
    }
    
    fn backtrack_simple(
        board: &mut Vec<Vec<char>>,
        result: &mut Vec<Vec<String>>,
        row: usize,
        n: usize,
    ) {
        if row == n {
            result.push(board.iter().map(|row| row.iter().collect()).collect());
            return;
        }
        
        for col in 0..n {
            if Self::is_safe(board, row, col, n) {
                board[row][col] = 'Q';
                Self::backtrack_simple(board, result, row + 1, n);
                board[row][col] = '.';
            }
        }
    }
    
    /// Approach 3: Bitwise Optimization
    /// 
    /// Use bitwise operations for faster constraint checking.
    /// 
    /// Time Complexity: O(N!)
    /// Space Complexity: O(N)
    pub fn solve_n_queens_bitwise(n: i32) -> Vec<Vec<String>> {
        let n = n as usize;
        let mut result = Vec::new();
        let mut board = vec![vec!['.'; n]; n];
        
        Self::backtrack_bitwise(&mut board, &mut result, 0, 0, 0, 0, n);
        result
    }
    
    fn backtrack_bitwise(
        board: &mut Vec<Vec<char>>,
        result: &mut Vec<Vec<String>>,
        row: usize,
        cols: u32,
        diag1: u32,
        diag2: u32,
        n: usize,
    ) {
        if row == n {
            result.push(board.iter().map(|row| row.iter().collect()).collect());
            return;
        }
        
        let mut available = ((1 << n) - 1) & !cols & !diag1 & !diag2;
        
        while available != 0 {
            let pos = available & (available.wrapping_neg());
            let col = pos.trailing_zeros() as usize;
            
            board[row][col] = 'Q';
            Self::backtrack_bitwise(
                board,
                result,
                row + 1,
                cols | pos,
                (diag1 | pos) << 1,
                (diag2 | pos) >> 1,
                n,
            );
            board[row][col] = '.';
            
            available &= available - 1;
        }
    }
    
    /// Approach 4: Recursive with Queens Positions Array
    /// 
    /// Track queen positions in a single array for cleaner code.
    /// 
    /// Time Complexity: O(N!)
    /// Space Complexity: O(N)
    pub fn solve_n_queens_positions(n: i32) -> Vec<Vec<String>> {
        let n = n as usize;
        let mut result = Vec::new();
        let mut queens = vec![0; n];
        
        Self::backtrack_positions(&mut queens, &mut result, 0, n);
        result
    }
    
    fn backtrack_positions(
        queens: &mut Vec<usize>,
        result: &mut Vec<Vec<String>>,
        row: usize,
        n: usize,
    ) {
        if row == n {
            result.push(Self::build_board(queens, n));
            return;
        }
        
        for col in 0..n {
            if Self::is_safe_positions(queens, row, col) {
                queens[row] = col;
                Self::backtrack_positions(queens, result, row + 1, n);
            }
        }
    }
    
    /// Approach 5: Permutation-based Approach
    /// 
    /// Since each row and column must have exactly one queen, generate permutations
    /// and check diagonal constraints.
    /// 
    /// Time Complexity: O(N! * N)
    /// Space Complexity: O(N)
    pub fn solve_n_queens_permutation(n: i32) -> Vec<Vec<String>> {
        let n = n as usize;
        let mut result = Vec::new();
        let mut cols: Vec<usize> = (0..n).collect();
        
        Self::backtrack_permutation(&mut cols, &mut result, 0, n);
        result
    }
    
    fn backtrack_permutation(
        cols: &mut Vec<usize>,
        result: &mut Vec<Vec<String>>,
        start: usize,
        n: usize,
    ) {
        if start == n {
            if Self::is_valid_permutation(cols) {
                result.push(Self::build_board_from_cols(cols, n));
            }
            return;
        }
        
        for i in start..n {
            cols.swap(start, i);
            Self::backtrack_permutation(cols, result, start + 1, n);
            cols.swap(start, i);
        }
    }
    
    /// Approach 6: Incremental Constraint Propagation
    /// 
    /// Build solutions incrementally with early pruning.
    /// 
    /// Time Complexity: O(N!)
    /// Space Complexity: O(N)
    pub fn solve_n_queens_incremental(n: i32) -> Vec<Vec<String>> {
        let n = n as usize;
        let mut result = Vec::new();
        let mut solution = Vec::new();
        
        Self::backtrack_incremental(&mut solution, &mut result, 0, n);
        result
    }
    
    fn backtrack_incremental(
        solution: &mut Vec<usize>,
        result: &mut Vec<Vec<String>>,
        row: usize,
        n: usize,
    ) {
        if row == n {
            result.push(Self::build_board(solution, n));
            return;
        }
        
        for col in 0..n {
            if Self::can_place_queen(solution, row, col) {
                solution.push(col);
                Self::backtrack_incremental(solution, result, row + 1, n);
                solution.pop();
            }
        }
    }
    
    // Helper methods
    
    fn is_safe(board: &Vec<Vec<char>>, row: usize, col: usize, n: usize) -> bool {
        // Check column
        for i in 0..row {
            if board[i][col] == 'Q' {
                return false;
            }
        }
        
        // Check diagonal (top-left to bottom-right)
        let mut i = row;
        let mut j = col;
        while i > 0 && j > 0 {
            i -= 1;
            j -= 1;
            if board[i][j] == 'Q' {
                return false;
            }
        }
        
        // Check anti-diagonal (top-right to bottom-left)
        i = row;
        j = col;
        while i > 0 && j < n - 1 {
            i -= 1;
            j += 1;
            if board[i][j] == 'Q' {
                return false;
            }
        }
        
        true
    }
    
    fn is_safe_positions(queens: &Vec<usize>, row: usize, col: usize) -> bool {
        for i in 0..row {
            if queens[i] == col ||
               queens[i] + i == col + row ||
               queens[i] + row == col + i {
                return false;
            }
        }
        true
    }
    
    fn is_valid_permutation(cols: &Vec<usize>) -> bool {
        let n = cols.len();
        for i in 0..n {
            for j in (i + 1)..n {
                if i + cols[i] == j + cols[j] || i + cols[j] == j + cols[i] {
                    return false;
                }
            }
        }
        true
    }
    
    fn can_place_queen(solution: &Vec<usize>, row: usize, col: usize) -> bool {
        for (i, &queen_col) in solution.iter().enumerate() {
            if queen_col == col ||
               queen_col + i == col + row ||
               queen_col + row == col + i {
                return false;
            }
        }
        true
    }
    
    fn build_board(queens: &Vec<usize>, n: usize) -> Vec<String> {
        queens.iter().map(|&col| {
            let mut row = vec!['.'; n];
            row[col] = 'Q';
            row.into_iter().collect()
        }).collect()
    }
    
    fn build_board_from_cols(cols: &Vec<usize>, n: usize) -> Vec<String> {
        (0..n).map(|row| {
            let mut board_row = vec!['.'; n];
            board_row[cols[row]] = 'Q';
            board_row.into_iter().collect()
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    
    fn is_valid_solution(board: &Vec<String>) -> bool {
        let n = board.len();
        let mut queens = Vec::new();
        
        // Find all queen positions
        for (row, row_str) in board.iter().enumerate() {
            for (col, ch) in row_str.chars().enumerate() {
                if ch == 'Q' {
                    queens.push((row, col));
                }
            }
        }
        
        if queens.len() != n {
            return false;
        }
        
        // Check constraints
        for i in 0..queens.len() {
            for j in (i + 1)..queens.len() {
                let (r1, c1) = queens[i];
                let (r2, c2) = queens[j];
                
                // Same row, column, or diagonal
                if r1 == r2 || c1 == c2 || 
                   (r1 as i32 - r2 as i32).abs() == (c1 as i32 - c2 as i32).abs() {
                    return false;
                }
            }
        }
        
        true
    }
    
    #[test]
    fn test_n_queens_n4_optimized() {
        let result = Solution::solve_n_queens_optimized(4);
        
        assert_eq!(result.len(), 2);
        for solution in &result {
            assert!(is_valid_solution(solution));
            assert_eq!(solution.len(), 4);
            for row in solution {
                assert_eq!(row.len(), 4);
            }
        }
    }
    
    #[test]
    fn test_n_queens_n1() {
        let result = Solution::solve_n_queens_simple(1);
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], vec!["Q"]);
        assert!(is_valid_solution(&result[0]));
    }
    
    #[test]
    fn test_n_queens_n2() {
        let result = Solution::solve_n_queens_bitwise(2);
        
        // No solution exists for n=2
        assert_eq!(result.len(), 0);
    }
    
    #[test]
    fn test_n_queens_n3() {
        let result = Solution::solve_n_queens_positions(3);
        
        // No solution exists for n=3
        assert_eq!(result.len(), 0);
    }
    
    #[test]
    fn test_n_queens_n4_all_approaches() {
        let result1 = Solution::solve_n_queens_optimized(4);
        let result2 = Solution::solve_n_queens_simple(4);
        let result3 = Solution::solve_n_queens_bitwise(4);
        let result4 = Solution::solve_n_queens_positions(4);
        let result5 = Solution::solve_n_queens_permutation(4);
        let result6 = Solution::solve_n_queens_incremental(4);
        
        // All should have 2 solutions
        assert_eq!(result1.len(), 2);
        assert_eq!(result2.len(), 2);
        assert_eq!(result3.len(), 2);
        assert_eq!(result4.len(), 2);
        assert_eq!(result5.len(), 2);
        assert_eq!(result6.len(), 2);
        
        // All solutions should be valid
        for result in [&result1, &result2, &result3, &result4, &result5, &result6] {
            for solution in result {
                assert!(is_valid_solution(solution));
            }
        }
    }
    
    #[test]
    fn test_n_queens_n8() {
        let result = Solution::solve_n_queens_optimized(8);
        
        assert_eq!(result.len(), 92); // Known result for 8-queens
        for solution in &result {
            assert!(is_valid_solution(solution));
            assert_eq!(solution.len(), 8);
        }
    }
    
    #[test]
    fn test_n_queens_n5() {
        let result = Solution::solve_n_queens_bitwise(5);
        
        assert_eq!(result.len(), 10); // Known result for 5-queens
        for solution in &result {
            assert!(is_valid_solution(solution));
            assert_eq!(solution.len(), 5);
        }
    }
    
    #[test]
    fn test_n_queens_n6() {
        let result = Solution::solve_n_queens_positions(6);
        
        assert_eq!(result.len(), 4); // Known result for 6-queens
        for solution in &result {
            assert!(is_valid_solution(solution));
            assert_eq!(solution.len(), 6);
        }
    }
    
    #[test]
    fn test_specific_n4_solutions() {
        let result = Solution::solve_n_queens_optimized(4);
        let expected_solutions: HashSet<Vec<String>> = [
            vec![".Q..", "...Q", "Q...", "..Q."],
            vec!["..Q.", "Q...", "...Q", ".Q.."],
        ].iter().map(|sol| sol.iter().map(|s| s.to_string()).collect()).collect();
        
        let actual_solutions: HashSet<Vec<String>> = result.into_iter().collect();
        assert_eq!(actual_solutions, expected_solutions);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        for n in 1..=6 {
            let result1 = Solution::solve_n_queens_optimized(n);
            let result2 = Solution::solve_n_queens_simple(n);
            let result3 = Solution::solve_n_queens_bitwise(n);
            let result4 = Solution::solve_n_queens_positions(n);
            let result5 = Solution::solve_n_queens_permutation(n);
            let result6 = Solution::solve_n_queens_incremental(n);
            
            // All approaches should find the same number of solutions
            assert_eq!(result1.len(), result2.len(), "Optimized vs Simple mismatch for n={}", n);
            assert_eq!(result2.len(), result3.len(), "Simple vs Bitwise mismatch for n={}", n);
            assert_eq!(result3.len(), result4.len(), "Bitwise vs Positions mismatch for n={}", n);
            assert_eq!(result4.len(), result5.len(), "Positions vs Permutation mismatch for n={}", n);
            assert_eq!(result5.len(), result6.len(), "Permutation vs Incremental mismatch for n={}", n);
            
            // All solutions should be valid
            for result in [&result1, &result2, &result3, &result4, &result5, &result6] {
                for solution in result {
                    assert!(is_valid_solution(solution), "Invalid solution for n={}", n);
                }
            }
            
            // Convert to sets for comparison (order doesn't matter)
            let set1: HashSet<Vec<String>> = result1.into_iter().collect();
            let set2: HashSet<Vec<String>> = result2.into_iter().collect();
            let set3: HashSet<Vec<String>> = result3.into_iter().collect();
            let set4: HashSet<Vec<String>> = result4.into_iter().collect();
            let set5: HashSet<Vec<String>> = result5.into_iter().collect();
            let set6: HashSet<Vec<String>> = result6.into_iter().collect();
            
            assert_eq!(set1, set2, "Solution sets differ between Optimized and Simple for n={}", n);
            assert_eq!(set2, set3, "Solution sets differ between Simple and Bitwise for n={}", n);
            assert_eq!(set3, set4, "Solution sets differ between Bitwise and Positions for n={}", n);
            assert_eq!(set4, set5, "Solution sets differ between Positions and Permutation for n={}", n);
            assert_eq!(set5, set6, "Solution sets differ between Permutation and Incremental for n={}", n);
        }
    }
    
    #[test]
    fn test_known_solution_counts() {
        let known_counts = [
            (1, 1),
            (2, 0),
            (3, 0),
            (4, 2),
            (5, 10),
            (6, 4),
            (7, 40),
            (8, 92),
        ];
        
        for (n, expected_count) in known_counts {
            let result = Solution::solve_n_queens_optimized(n);
            assert_eq!(result.len(), expected_count, "Incorrect solution count for n={}", n);
        }
    }
}