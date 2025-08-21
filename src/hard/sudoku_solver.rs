//! Problem 37: Sudoku Solver
//!
//! Implementation by Marvin Tutt, Caia Tech
//!
//! Write a program to solve a Sudoku puzzle by filling the empty cells.
//! A sudoku solution must satisfy all of the following rules:
//! - Each of the digits 1-9 must occur exactly once in each row.
//! - Each of the digits 1-9 must occur exactly once in each column.
//! - Each of the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.
//! The '.' character indicates empty cells.
//!
//! Constraints:
//! - board.length == 9
//! - board[i].length == 9
//! - board[i][j] is a digit or '.'.
//! - It is guaranteed that the input board has only one solution.
//!
//! Example:
//! Input: board = [["5","3",".",".","7",".",".",".","."],
//!                 ["6",".",".","1","9","5",".",".","."],
//!                 [".","9","8",".",".",".",".","6","."],
//!                 ["8",".",".",".","6",".",".",".","3"],
//!                 ["4",".",".","8",".","3",".",".","1"],
//!                 ["7",".",".",".","2",".",".",".","6"],
//!                 [".","6",".",".",".",".","2","8","."],
//!                 [".",".",".","4","1","9",".",".","5"],
//!                 [".",".",".",".","8",".",".","7","9"]]
//! Output: [["5","3","4","6","7","8","9","1","2"],
//!          ["6","7","2","1","9","5","3","4","8"],
//!          ["1","9","8","3","4","2","5","6","7"],
//!          ["8","5","9","7","6","1","4","2","3"],
//!          ["4","2","6","8","5","3","7","9","1"],
//!          ["7","1","3","9","2","4","8","5","6"],
//!          ["9","6","1","5","3","7","2","8","4"],
//!          ["2","8","7","4","1","9","6","3","5"],
//!          ["3","4","5","2","8","6","1","7","9"]]

pub struct Solution;

impl Solution {
    /// Approach 1: Backtracking with Constraint Propagation - Optimal
    /// 
    /// Use backtracking with smart constraint tracking to solve efficiently.
    /// Track which numbers are available for each row, column, and box.
    /// 
    /// Time Complexity: O(9^(empty_cells)) in worst case, but much faster with pruning
    /// Space Complexity: O(1) additional space (board is modified in place)
    pub fn solve_sudoku_optimized(board: &mut Vec<Vec<char>>) {
        let mut rows = vec![vec![false; 9]; 9];
        let mut cols = vec![vec![false; 9]; 9];
        let mut boxes = vec![vec![false; 9]; 9];
        
        // Initialize constraint tracking
        for i in 0..9 {
            for j in 0..9 {
                if board[i][j] != '.' {
                    let num = (board[i][j] as u8 - b'1') as usize;
                    rows[i][num] = true;
                    cols[j][num] = true;
                    boxes[Self::box_index(i, j)][num] = true;
                }
            }
        }
        
        Self::backtrack_optimized(board, &mut rows, &mut cols, &mut boxes, 0, 0);
    }
    
    fn backtrack_optimized(
        board: &mut Vec<Vec<char>>,
        rows: &mut Vec<Vec<bool>>,
        cols: &mut Vec<Vec<bool>>,
        boxes: &mut Vec<Vec<bool>>,
        row: usize,
        col: usize,
    ) -> bool {
        if row == 9 {
            return true; // Successfully filled all cells
        }
        
        let (next_row, next_col) = if col == 8 { (row + 1, 0) } else { (row, col + 1) };
        
        if board[row][col] != '.' {
            return Self::backtrack_optimized(board, rows, cols, boxes, next_row, next_col);
        }
        
        let box_idx = Self::box_index(row, col);
        
        for num in 0..9 {
            if !rows[row][num] && !cols[col][num] && !boxes[box_idx][num] {
                // Place the number
                board[row][col] = (b'1' + num as u8) as char;
                rows[row][num] = true;
                cols[col][num] = true;
                boxes[box_idx][num] = true;
                
                if Self::backtrack_optimized(board, rows, cols, boxes, next_row, next_col) {
                    return true;
                }
                
                // Backtrack
                board[row][col] = '.';
                rows[row][num] = false;
                cols[col][num] = false;
                boxes[box_idx][num] = false;
            }
        }
        
        false
    }
    
    /// Approach 2: Simple Backtracking
    /// 
    /// Basic backtracking approach that validates constraints for each placement.
    /// 
    /// Time Complexity: O(9^(empty_cells))
    /// Space Complexity: O(empty_cells) for recursion stack
    pub fn solve_sudoku_simple(board: &mut Vec<Vec<char>>) {
        Self::backtrack_simple(board, 0, 0);
    }
    
    fn backtrack_simple(board: &mut Vec<Vec<char>>, row: usize, col: usize) -> bool {
        if row == 9 {
            return true;
        }
        
        let (next_row, next_col) = if col == 8 { (row + 1, 0) } else { (row, col + 1) };
        
        if board[row][col] != '.' {
            return Self::backtrack_simple(board, next_row, next_col);
        }
        
        for num in b'1'..=b'9' {
            let ch = num as char;
            if Self::is_valid_placement(board, row, col, ch) {
                board[row][col] = ch;
                
                if Self::backtrack_simple(board, next_row, next_col) {
                    return true;
                }
                
                board[row][col] = '.';
            }
        }
        
        false
    }
    
    /// Approach 3: Backtracking with Most Constrained Variable (MCV)
    /// 
    /// Choose the empty cell with the fewest possible values first.
    /// 
    /// Time Complexity: O(9^(empty_cells)) but with better pruning
    /// Space Complexity: O(empty_cells)
    pub fn solve_sudoku_mcv(board: &mut Vec<Vec<char>>) {
        Self::backtrack_mcv(board);
    }
    
    fn backtrack_mcv(board: &mut Vec<Vec<char>>) -> bool {
        let (row, col) = match Self::find_most_constrained_cell(board) {
            Some(pos) => pos,
            None => return true, // All cells filled
        };
        
        for num in b'1'..=b'9' {
            let ch = num as char;
            if Self::is_valid_placement(board, row, col, ch) {
                board[row][col] = ch;
                
                if Self::backtrack_mcv(board) {
                    return true;
                }
                
                board[row][col] = '.';
            }
        }
        
        false
    }
    
    /// Approach 4: Constraint Propagation with Arc Consistency
    /// 
    /// Use constraint propagation to reduce the search space before backtracking.
    /// 
    /// Time Complexity: O(9^k) where k < empty_cells due to propagation
    /// Space Complexity: O(1)
    pub fn solve_sudoku_constraint_propagation(board: &mut Vec<Vec<char>>) {
        let mut possibilities = vec![vec![vec![true; 9]; 9]; 9];
        
        // Initialize possibilities based on current board state
        Self::initialize_possibilities(board, &mut possibilities);
        Self::propagate_constraints(&mut possibilities);
        Self::backtrack_with_propagation(board, &mut possibilities);
    }
    
    /// Approach 5: Dancing Links Algorithm (Simplified)
    /// 
    /// For this implementation, we'll use the optimized backtracking approach
    /// as a full Dancing Links implementation would be quite complex.
    /// 
    /// Time Complexity: O(9^(empty_cells))
    /// Space Complexity: O(1)
    pub fn solve_sudoku_dancing_links(board: &mut Vec<Vec<char>>) {
        // Use the optimized backtracking as a simplified version
        Self::solve_sudoku_optimized(board);
    }
    
    /// Approach 6: Bitwise Optimization
    /// 
    /// Use bitwise operations for faster constraint checking.
    /// 
    /// Time Complexity: O(9^(empty_cells))
    /// Space Complexity: O(1)
    pub fn solve_sudoku_bitwise(board: &mut Vec<Vec<char>>) {
        let mut row_masks = vec![0u16; 9];
        let mut col_masks = vec![0u16; 9];
        let mut box_masks = vec![0u16; 9];
        
        // Initialize bitmasks
        for i in 0..9 {
            for j in 0..9 {
                if board[i][j] != '.' {
                    let bit = 1 << ((board[i][j] as u8 - b'1') as usize);
                    row_masks[i] |= bit;
                    col_masks[j] |= bit;
                    box_masks[Self::box_index(i, j)] |= bit;
                }
            }
        }
        
        Self::backtrack_bitwise(board, &mut row_masks, &mut col_masks, &mut box_masks, 0, 0);
    }
    
    fn backtrack_bitwise(
        board: &mut Vec<Vec<char>>,
        row_masks: &mut Vec<u16>,
        col_masks: &mut Vec<u16>,
        box_masks: &mut Vec<u16>,
        row: usize,
        col: usize,
    ) -> bool {
        if row == 9 {
            return true;
        }
        
        let (next_row, next_col) = if col == 8 { (row + 1, 0) } else { (row, col + 1) };
        
        if board[row][col] != '.' {
            return Self::backtrack_bitwise(board, row_masks, col_masks, box_masks, next_row, next_col);
        }
        
        let box_idx = Self::box_index(row, col);
        let used_mask = row_masks[row] | col_masks[col] | box_masks[box_idx];
        
        for num in 0..9 {
            let bit = 1 << num;
            if used_mask & bit == 0 {
                board[row][col] = (b'1' + num as u8) as char;
                row_masks[row] |= bit;
                col_masks[col] |= bit;
                box_masks[box_idx] |= bit;
                
                if Self::backtrack_bitwise(board, row_masks, col_masks, box_masks, next_row, next_col) {
                    return true;
                }
                
                board[row][col] = '.';
                row_masks[row] &= !bit;
                col_masks[col] &= !bit;
                box_masks[box_idx] &= !bit;
            }
        }
        
        false
    }
    
    // Helper methods
    
    fn box_index(row: usize, col: usize) -> usize {
        (row / 3) * 3 + col / 3
    }
    
    fn is_valid_placement(board: &Vec<Vec<char>>, row: usize, col: usize, ch: char) -> bool {
        // Check row
        for j in 0..9 {
            if board[row][j] == ch {
                return false;
            }
        }
        
        // Check column
        for i in 0..9 {
            if board[i][col] == ch {
                return false;
            }
        }
        
        // Check 3x3 box
        let box_row = (row / 3) * 3;
        let box_col = (col / 3) * 3;
        for i in box_row..box_row + 3 {
            for j in box_col..box_col + 3 {
                if board[i][j] == ch {
                    return false;
                }
            }
        }
        
        true
    }
    
    fn find_most_constrained_cell(board: &Vec<Vec<char>>) -> Option<(usize, usize)> {
        let mut min_options = 10;
        let mut best_cell = None;
        
        for i in 0..9 {
            for j in 0..9 {
                if board[i][j] == '.' {
                    let mut options = 0;
                    for num in b'1'..=b'9' {
                        if Self::is_valid_placement(board, i, j, num as char) {
                            options += 1;
                        }
                    }
                    
                    if options < min_options {
                        min_options = options;
                        best_cell = Some((i, j));
                        
                        if options == 0 {
                            return Some((i, j)); // Dead end, return immediately
                        }
                    }
                }
            }
        }
        
        best_cell
    }
    
    fn initialize_possibilities(board: &Vec<Vec<char>>, possibilities: &mut Vec<Vec<Vec<bool>>>) {
        for i in 0..9 {
            for j in 0..9 {
                if board[i][j] != '.' {
                    // Cell is already filled
                    possibilities[i][j].fill(false);
                } else {
                    // Check which numbers are possible
                    for num in 0..9 {
                        let ch = (b'1' + num as u8) as char;
                        possibilities[i][j][num] = Self::is_valid_placement(board, i, j, ch);
                    }
                }
            }
        }
    }
    
    fn propagate_constraints(possibilities: &mut Vec<Vec<Vec<bool>>>) {
        let mut changed = true;
        while changed {
            changed = false;
            
            // If a cell has only one possibility, remove that number from peers
            for i in 0..9 {
                for j in 0..9 {
                    let count = possibilities[i][j].iter().filter(|&&x| x).count();
                    if count == 1 {
                        if let Some(num) = possibilities[i][j].iter().position(|&x| x) {
                            // Remove this number from row, column, and box
                            for k in 0..9 {
                                if k != j && possibilities[i][k][num] {
                                    possibilities[i][k][num] = false;
                                    changed = true;
                                }
                                if k != i && possibilities[k][j][num] {
                                    possibilities[k][j][num] = false;
                                    changed = true;
                                }
                            }
                            
                            let box_row = (i / 3) * 3;
                            let box_col = (j / 3) * 3;
                            for bi in box_row..box_row + 3 {
                                for bj in box_col..box_col + 3 {
                                    if (bi != i || bj != j) && possibilities[bi][bj][num] {
                                        possibilities[bi][bj][num] = false;
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    fn backtrack_with_propagation(board: &mut Vec<Vec<char>>, possibilities: &mut Vec<Vec<Vec<bool>>>) -> bool {
        // Find cell with minimum possibilities
        let (row, col) = match Self::find_cell_min_possibilities(possibilities) {
            Some(pos) => pos,
            None => return true, // All cells filled
        };
        
        let possible_nums: Vec<usize> = possibilities[row][col]
            .iter()
            .enumerate()
            .filter_map(|(i, &possible)| if possible { Some(i) } else { None })
            .collect();
        
        for &num in &possible_nums {
            let ch = (b'1' + num as u8) as char;
            
            // Save current state
            let saved_possibilities = possibilities.clone();
            
            // Make move
            board[row][col] = ch;
            possibilities[row][col].fill(false);
            Self::propagate_constraints(possibilities);
            
            if Self::backtrack_with_propagation(board, possibilities) {
                return true;
            }
            
            // Backtrack
            board[row][col] = '.';
            *possibilities = saved_possibilities;
        }
        
        false
    }
    
    fn find_cell_min_possibilities(possibilities: &Vec<Vec<Vec<bool>>>) -> Option<(usize, usize)> {
        let mut min_count = 10;
        let mut best_cell = None;
        
        for i in 0..9 {
            for j in 0..9 {
                let count = possibilities[i][j].iter().filter(|&&x| x).count();
                if count > 0 && count < min_count {
                    min_count = count;
                    best_cell = Some((i, j));
                }
            }
        }
        
        best_cell
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_board() -> Vec<Vec<char>> {
        vec![
            vec!['5','3','.','.','7','.','.','.','.'],
            vec!['6','.','.','1','9','5','.','.','.'],
            vec!['.','9','8','.','.','.','.','6','.'],
            vec!['8','.','.','.','6','.','.','.','3'],
            vec!['4','.','.','8','.','3','.','.','1'],
            vec!['7','.','.','.','2','.','.','.','6'],
            vec!['.','6','.','.','.','.','2','8','.'],
            vec!['.','.','.','4','1','9','.','.','5'],
            vec!['.','.','.','.','8','.','.','7','9']
        ]
    }
    
    fn expected_solution() -> Vec<Vec<char>> {
        vec![
            vec!['5','3','4','6','7','8','9','1','2'],
            vec!['6','7','2','1','9','5','3','4','8'],
            vec!['1','9','8','3','4','2','5','6','7'],
            vec!['8','5','9','7','6','1','4','2','3'],
            vec!['4','2','6','8','5','3','7','9','1'],
            vec!['7','1','3','9','2','4','8','5','6'],
            vec!['9','6','1','5','3','7','2','8','4'],
            vec!['2','8','7','4','1','9','6','3','5'],
            vec!['3','4','5','2','8','6','1','7','9']
        ]
    }
    
    fn is_valid_solution(board: &Vec<Vec<char>>) -> bool {
        // Check rows
        for row in board {
            let mut seen = vec![false; 9];
            for &ch in row {
                if ch == '.' || seen[(ch as u8 - b'1') as usize] {
                    return false;
                }
                seen[(ch as u8 - b'1') as usize] = true;
            }
        }
        
        // Check columns
        for col in 0..9 {
            let mut seen = vec![false; 9];
            for row in 0..9 {
                let ch = board[row][col];
                if ch == '.' || seen[(ch as u8 - b'1') as usize] {
                    return false;
                }
                seen[(ch as u8 - b'1') as usize] = true;
            }
        }
        
        // Check boxes
        for box_row in (0..9).step_by(3) {
            for box_col in (0..9).step_by(3) {
                let mut seen = vec![false; 9];
                for i in box_row..box_row + 3 {
                    for j in box_col..box_col + 3 {
                        let ch = board[i][j];
                        if ch == '.' || seen[(ch as u8 - b'1') as usize] {
                            return false;
                        }
                        seen[(ch as u8 - b'1') as usize] = true;
                    }
                }
            }
        }
        
        true
    }
    
    #[test]
    fn test_solve_sudoku_optimized() {
        let mut board = create_test_board();
        Solution::solve_sudoku_optimized(&mut board);
        
        assert!(is_valid_solution(&board));
        assert_eq!(board, expected_solution());
    }
    
    #[test]
    fn test_solve_sudoku_simple() {
        let mut board = create_test_board();
        Solution::solve_sudoku_simple(&mut board);
        
        assert!(is_valid_solution(&board));
        assert_eq!(board, expected_solution());
    }
    
    #[test]
    fn test_solve_sudoku_mcv() {
        let mut board = create_test_board();
        Solution::solve_sudoku_mcv(&mut board);
        
        assert!(is_valid_solution(&board));
        assert_eq!(board, expected_solution());
    }
    
    #[test]
    fn test_solve_sudoku_constraint_propagation() {
        let mut board = create_test_board();
        Solution::solve_sudoku_constraint_propagation(&mut board);
        
        assert!(is_valid_solution(&board));
        assert_eq!(board, expected_solution());
    }
    
    #[test]
    fn test_solve_sudoku_dancing_links() {
        let mut board = create_test_board();
        Solution::solve_sudoku_dancing_links(&mut board);
        
        assert!(is_valid_solution(&board));
        assert_eq!(board, expected_solution());
    }
    
    #[test]
    fn test_solve_sudoku_bitwise() {
        let mut board = create_test_board();
        Solution::solve_sudoku_bitwise(&mut board);
        
        assert!(is_valid_solution(&board));
        assert_eq!(board, expected_solution());
    }
    
    #[test]
    fn test_easy_sudoku() {
        let mut board = vec![
            vec!['5','3','4','6','7','8','9','1','.'],
            vec!['6','7','2','1','9','5','3','4','8'],
            vec!['1','9','8','3','4','2','5','6','7'],
            vec!['8','5','9','7','6','1','4','2','3'],
            vec!['4','2','6','8','5','3','7','9','1'],
            vec!['7','1','3','9','2','4','8','5','6'],
            vec!['9','6','1','5','3','7','2','8','4'],
            vec!['2','8','7','4','1','9','6','3','5'],
            vec!['3','4','5','2','8','6','1','7','9']
        ];
        
        Solution::solve_sudoku_optimized(&mut board);
        assert!(is_valid_solution(&board));
        assert_eq!(board[0][8], '2');
    }
    
    #[test]
    fn test_near_complete_sudoku() {
        let mut board = vec![
            vec!['5','3','4','6','7','8','9','1','2'],
            vec!['6','7','2','1','9','5','3','4','8'],
            vec!['1','9','8','3','4','2','5','6','7'],
            vec!['8','5','9','7','6','1','4','2','3'],
            vec!['4','2','6','8','5','3','7','9','1'],
            vec!['7','1','3','9','2','4','8','5','6'],
            vec!['9','6','1','5','3','7','2','8','4'],
            vec!['2','8','7','4','1','9','6','3','5'],
            vec!['3','4','5','2','8','6','1','7','.']
        ];
        
        Solution::solve_sudoku_simple(&mut board);
        assert!(is_valid_solution(&board));
        assert_eq!(board[8][8], '9');
    }
    
    #[test]
    fn test_hard_sudoku() {
        let mut board = vec![
            vec!['.','.','.','.','.','.','.','.','.'],
            vec!['.','.','.','.','.','3','.','8','5'],
            vec!['.','.','.','7','.','.','.','.','.'],
            vec!['.','.','.','.','.','.','.','.','.'],
            vec!['4','.','.','.','1','.','.','.','6'],
            vec!['.','.','.','.','.','.','.','.','.'],
            vec!['.','.','.','1','.','.','.','.','.'],
            vec!['9','2','.','.','.','.','.','.','.'],
            vec!['.','.','.','.','.','.','.','.','7']
        ];
        
        Solution::solve_sudoku_mcv(&mut board);
        assert!(is_valid_solution(&board));
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_board = create_test_board();
        
        let mut board1 = test_board.clone();
        let mut board2 = test_board.clone();
        let mut board3 = test_board.clone();
        let mut board4 = test_board.clone();
        let mut board5 = test_board.clone();
        let mut board6 = test_board.clone();
        
        Solution::solve_sudoku_optimized(&mut board1);
        Solution::solve_sudoku_simple(&mut board2);
        Solution::solve_sudoku_mcv(&mut board3);
        Solution::solve_sudoku_constraint_propagation(&mut board4);
        Solution::solve_sudoku_dancing_links(&mut board5);
        Solution::solve_sudoku_bitwise(&mut board6);
        
        assert!(is_valid_solution(&board1));
        assert!(is_valid_solution(&board2));
        assert!(is_valid_solution(&board3));
        assert!(is_valid_solution(&board4));
        assert!(is_valid_solution(&board5));
        assert!(is_valid_solution(&board6));
        
        // All solutions should be the same (deterministic)
        assert_eq!(board1, board2);
        assert_eq!(board2, board3);
        assert_eq!(board3, board4);
        assert_eq!(board4, board5);
        assert_eq!(board5, board6);
    }
}