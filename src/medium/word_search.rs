//! # Problem 79: Word Search
//!
//! Given an `m x n` grid of characters `board` and a string `word`, return `true` if `word` exists in the grid.
//!
//! The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are
//! horizontally or vertically neighboring. The same letter cell may not be used more than once.
//!
//! ## Examples
//!
//! ```text
//! Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
//! Output: true
//! ```
//!
//! ```text
//! Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
//! Output: true
//! ```
//!
//! ```text
//! Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
//! Output: false
//! ```
//!
//! ## Constraints
//!
//! * m == board.length
//! * n == board[i].length
//! * 1 <= m, n <= 6
//! * 1 <= word.length <= 15
//! * board and word consists of only lowercase and uppercase English letters

use std::collections::{HashSet, HashMap};

/// Solution for Word Search problem
pub struct Solution;

impl Solution {
    /// Creates a new instance of Solution
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: DFS Backtracking with In-Place Marking (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Try starting from each cell in the grid
    /// 2. For each starting cell, perform DFS to match the word
    /// 3. Mark visited cells temporarily by changing their value
    /// 4. Backtrack by restoring original value when returning
    /// 5. Explore all 4 directions for next character
    /// 
    /// **Time Complexity:** O(m * n * 4^L) where L = word length
    /// **Space Complexity:** O(L) for recursion stack
    /// 
    /// **Key Insights:**
    /// - In-place marking avoids separate visited array
    /// - Early termination when word is found
    /// - Backtracking ensures cells can be reused in different paths
    /// 
    /// **Why this works:**
    /// - DFS explores all possible paths from each starting point
    /// - Backtracking allows exploring different paths without interference
    /// - Marking prevents using same cell twice in single path
    /// 
    /// **Example walkthrough for "SEE":**
    /// ```text
    /// Board: [["A","B","C","E"],
    ///         ["S","F","C","S"],
    ///         ["A","D","E","E"]]
    /// 
    /// Start at (1,0) with 'S': matches word[0]
    /// Try all directions from 'S'
    /// Move to (2,0) 'A': doesn't match word[1]='E', backtrack
    /// Move to (1,1) 'F': doesn't match word[1]='E', backtrack
    /// Move up to (0,0) 'A': doesn't match word[1]='E', backtrack
    /// Continue until finding path S(1,0) -> E(2,1) -> E(2,2)
    /// ```
    pub fn exist(&self, board: Vec<Vec<char>>, word: String) -> bool {
        if board.is_empty() || board[0].is_empty() || word.is_empty() {
            return false;
        }
        
        let mut board = board;
        let word_chars: Vec<char> = word.chars().collect();
        let m = board.len();
        let n = board[0].len();
        
        fn dfs(board: &mut Vec<Vec<char>>, word: &[char], i: usize, j: usize, idx: usize) -> bool {
            // Check if we've found the entire word
            if idx == word.len() {
                return true;
            }
            
            // Check boundaries and character match
            if i >= board.len() || j >= board[0].len() || board[i][j] != word[idx] {
                return false;
            }
            
            // Mark current cell as visited
            let temp = board[i][j];
            board[i][j] = '#';
            
            // Explore all 4 directions
            let found = (i > 0 && dfs(board, word, i - 1, j, idx + 1)) ||
                       (i + 1 < board.len() && dfs(board, word, i + 1, j, idx + 1)) ||
                       (j > 0 && dfs(board, word, i, j - 1, idx + 1)) ||
                       (j + 1 < board[0].len() && dfs(board, word, i, j + 1, idx + 1));
            
            // Backtrack: restore original value
            board[i][j] = temp;
            
            found
        }
        
        // Try starting from each cell
        for i in 0..m {
            for j in 0..n {
                if dfs(&mut board, &word_chars, i, j, 0) {
                    return true;
                }
            }
        }
        
        false
    }

    /// # Approach 2: DFS with Visited Set
    /// 
    /// **Algorithm:**
    /// 1. Use HashSet to track visited cells in current path
    /// 2. Add cell to visited set when entering
    /// 3. Remove from visited set when backtracking
    /// 4. Check all 4 directions for next character
    /// 
    /// **Time Complexity:** O(m * n * 4^L) where L = word length
    /// **Space Complexity:** O(L) for visited set and recursion stack
    /// 
    /// **Trade-offs:**
    /// - Cleaner than in-place marking but uses more memory
    /// - Original board remains unchanged
    /// - HashSet operations add small overhead
    /// 
    /// **When to use:** When board should not be modified
    pub fn exist_visited_set(&self, board: Vec<Vec<char>>, word: String) -> bool {
        if board.is_empty() || board[0].is_empty() || word.is_empty() {
            return false;
        }
        
        let word_chars: Vec<char> = word.chars().collect();
        let m = board.len();
        let n = board[0].len();
        
        fn dfs(
            board: &Vec<Vec<char>>, 
            word: &[char], 
            visited: &mut HashSet<(usize, usize)>,
            i: usize, 
            j: usize, 
            idx: usize
        ) -> bool {
            if idx == word.len() {
                return true;
            }
            
            if i >= board.len() || j >= board[0].len() || 
               visited.contains(&(i, j)) || board[i][j] != word[idx] {
                return false;
            }
            
            visited.insert((i, j));
            
            let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];
            for (di, dj) in directions {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                
                if ni >= 0 && ni < board.len() as i32 && nj >= 0 && nj < board[0].len() as i32 {
                    if dfs(board, word, visited, ni as usize, nj as usize, idx + 1) {
                        return true;
                    }
                }
            }
            
            visited.remove(&(i, j));
            false
        }
        
        for i in 0..m {
            for j in 0..n {
                let mut visited = HashSet::new();
                if dfs(&board, &word_chars, &mut visited, i, j, 0) {
                    return true;
                }
            }
        }
        
        false
    }

    /// # Approach 3: Optimized with Pruning
    /// 
    /// **Algorithm:**
    /// 1. Count character frequencies in board and word
    /// 2. Check if board has all required characters
    /// 3. Reverse word if it starts with common character
    /// 4. Use DFS with early termination
    /// 
    /// **Time Complexity:** O(m * n * 4^L) worst case, often better with pruning
    /// **Space Complexity:** O(L) for recursion stack
    /// 
    /// **Optimizations:**
    /// - Pre-check character availability
    /// - Start search from less common characters
    /// - Early termination when impossible
    /// 
    /// **Performance boost:** Can eliminate many impossible cases early
    pub fn exist_optimized(&self, board: Vec<Vec<char>>, word: String) -> bool {
        if board.is_empty() || board[0].is_empty() || word.is_empty() {
            return false;
        }
        
        let mut board = board;
        let m = board.len();
        let n = board[0].len();
        
        // Count character frequencies
        let mut board_freq = HashMap::new();
        for i in 0..m {
            for j in 0..n {
                *board_freq.entry(board[i][j]).or_insert(0) += 1;
            }
        }
        
        let mut word_freq = HashMap::new();
        for ch in word.chars() {
            *word_freq.entry(ch).or_insert(0) += 1;
        }
        
        // Check if board has all required characters
        for (ch, count) in &word_freq {
            if board_freq.get(ch).unwrap_or(&0) < count {
                return false;
            }
        }
        
        // Potentially reverse word for better starting positions
        let word_chars: Vec<char> = if word_freq.get(&word.chars().next().unwrap()).unwrap_or(&0) >
                                       word_freq.get(&word.chars().last().unwrap()).unwrap_or(&0) {
            word.chars().rev().collect()
        } else {
            word.chars().collect()
        };
        
        fn dfs(board: &mut Vec<Vec<char>>, word: &[char], i: usize, j: usize, idx: usize) -> bool {
            if idx == word.len() {
                return true;
            }
            
            if i >= board.len() || j >= board[0].len() || board[i][j] != word[idx] {
                return false;
            }
            
            let temp = board[i][j];
            board[i][j] = '#';
            
            let found = (i > 0 && dfs(board, word, i - 1, j, idx + 1)) ||
                       (i + 1 < board.len() && dfs(board, word, i + 1, j, idx + 1)) ||
                       (j > 0 && dfs(board, word, i, j - 1, idx + 1)) ||
                       (j + 1 < board[0].len() && dfs(board, word, i, j + 1, idx + 1));
            
            board[i][j] = temp;
            found
        }
        
        for i in 0..m {
            for j in 0..n {
                if board[i][j] == word_chars[0] && dfs(&mut board, &word_chars, i, j, 0) {
                    return true;
                }
            }
        }
        
        false
    }

    /// # Approach 4: Iterative DFS with Stack
    /// 
    /// **Algorithm:**
    /// 1. Use explicit stack instead of recursion
    /// 2. Store state (position, word index, visited cells) in stack
    /// 3. Process stack until empty or word found
    /// 4. Handle backtracking through state management
    /// 
    /// **Time Complexity:** O(m * n * 4^L) where L = word length
    /// **Space Complexity:** O(L * 4^L) for stack in worst case
    /// 
    /// **Advantages:**
    /// - No recursion stack overflow risk
    /// - More control over search order
    /// - Can implement custom prioritization
    /// 
    /// **Challenges:**
    /// - More complex state management
    /// - Higher memory usage than recursive version
    pub fn exist_iterative(&self, board: Vec<Vec<char>>, word: String) -> bool {
        if board.is_empty() || board[0].is_empty() || word.is_empty() {
            return false;
        }
        
        let word_chars: Vec<char> = word.chars().collect();
        let m = board.len();
        let n = board[0].len();
        
        // State: (row, col, word_index, visited_cells)
        type State = (usize, usize, usize, HashSet<(usize, usize)>);
        
        for i in 0..m {
            for j in 0..n {
                if board[i][j] != word_chars[0] {
                    continue;
                }
                
                let mut stack: Vec<State> = Vec::new();
                let mut initial_visited = HashSet::new();
                initial_visited.insert((i, j));
                stack.push((i, j, 1, initial_visited));
                
                while let Some((row, col, idx, visited)) = stack.pop() {
                    if idx == word_chars.len() {
                        return true;
                    }
                    
                    let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];
                    for (di, dj) in directions {
                        let ni = row as i32 + di;
                        let nj = col as i32 + dj;
                        
                        if ni >= 0 && ni < m as i32 && nj >= 0 && nj < n as i32 {
                            let ni = ni as usize;
                            let nj = nj as usize;
                            
                            if !visited.contains(&(ni, nj)) && 
                               board[ni][nj] == word_chars[idx] {
                                let mut new_visited = visited.clone();
                                new_visited.insert((ni, nj));
                                stack.push((ni, nj, idx + 1, new_visited));
                            }
                        }
                    }
                }
            }
        }
        
        false
    }

    /// # Approach 5: Bidirectional Search
    /// 
    /// **Algorithm:**
    /// 1. Search from both ends of the word simultaneously
    /// 2. Find paths from start and end characters
    /// 3. Check if paths can connect in the middle
    /// 4. Use meet-in-the-middle strategy
    /// 
    /// **Time Complexity:** O(m * n * 4^(L/2)) theoretical improvement
    /// **Space Complexity:** O(m * n) for path storage
    /// 
    /// **Advantages:**
    /// - Can be faster for long words
    /// - Reduces search space exponentially
    /// 
    /// **Challenges:**
    /// - Complex implementation
    /// - Overhead may not be worth it for short words
    /// 
    /// **Note:** This is more educational than practical for this problem
    pub fn exist_bidirectional(&self, board: Vec<Vec<char>>, word: String) -> bool {
        if board.is_empty() || board[0].is_empty() || word.is_empty() {
            return false;
        }
        
        let word_chars: Vec<char> = word.chars().collect();
        let m = board.len();
        let n = board[0].len();
        
        // For simplicity, fall back to regular DFS for short words
        if word_chars.len() <= 3 {
            return self.exist(board.clone(), word);
        }
        
        let mid = word_chars.len() / 2;
        
        // Find all paths of length mid from start character
        let mut start_paths = Vec::new();
        for i in 0..m {
            for j in 0..n {
                if board[i][j] == word_chars[0] {
                    let mut visited = HashSet::new();
                    visited.insert((i, j));
                    self.find_paths(&board, &word_chars[..=mid], i, j, 1, visited.clone(), &mut start_paths);
                }
            }
        }
        
        // Find all paths from end character
        let mut end_paths = Vec::new();
        let reversed: Vec<char> = word_chars[mid..].iter().rev().copied().collect();
        for i in 0..m {
            for j in 0..n {
                if board[i][j] == reversed[0] {
                    let mut visited = HashSet::new();
                    visited.insert((i, j));
                    self.find_paths(&board, &reversed, i, j, 1, visited.clone(), &mut end_paths);
                }
            }
        }
        
        // Check if any paths can connect
        for (end_pos1, visited1) in &start_paths {
            for (end_pos2, visited2) in &end_paths {
                if end_pos1 == end_pos2 {
                    // Check if paths don't overlap (except at meeting point)
                    let overlap = visited1.intersection(visited2).count();
                    if overlap == 1 && visited1.contains(end_pos1) && visited2.contains(end_pos2) {
                        return true;
                    }
                }
            }
        }
        
        false
    }
    
    fn find_paths(
        &self,
        board: &Vec<Vec<char>>,
        word: &[char],
        i: usize,
        j: usize,
        idx: usize,
        mut visited: HashSet<(usize, usize)>,
        results: &mut Vec<((usize, usize), HashSet<(usize, usize)>)>
    ) {
        if idx == word.len() {
            results.push(((i, j), visited));
            return;
        }
        
        let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];
        for (di, dj) in directions {
            let ni = i as i32 + di;
            let nj = j as i32 + dj;
            
            if ni >= 0 && ni < board.len() as i32 && nj >= 0 && nj < board[0].len() as i32 {
                let ni = ni as usize;
                let nj = nj as usize;
                
                if !visited.contains(&(ni, nj)) && 
                   board[ni][nj] == word[idx] {
                    visited.insert((ni, nj));
                    self.find_paths(board, word, ni, nj, idx + 1, visited.clone(), results);
                    visited.remove(&(ni, nj));
                }
            }
        }
    }

    /// # Approach 6: Trie-Based Multi-Word Search
    /// 
    /// **Algorithm:**
    /// 1. Build trie from word (or multiple words)
    /// 2. DFS through board while traversing trie
    /// 3. Mark found words in trie
    /// 4. Continue search for all possible words
    /// 
    /// **Time Complexity:** O(m * n * 4^L) for single word
    /// **Space Complexity:** O(L) for trie
    /// 
    /// **Advantages:**
    /// - Excellent for searching multiple words simultaneously
    /// - Prefix sharing reduces redundant searches
    /// - Can find all words in single pass
    /// 
    /// **When to use:** When searching for multiple words in same board
    pub fn exist_trie(&self, board: Vec<Vec<char>>, word: String) -> bool {
        if board.is_empty() || board[0].is_empty() || word.is_empty() {
            return false;
        }
        
        // Build trie
        let mut trie = TrieNode::new();
        trie.insert(&word);
        
        let mut board = board;
        let m = board.len();
        let n = board[0].len();
        let mut found = false;
        
        fn dfs(
            board: &mut Vec<Vec<char>>,
            node: &TrieNode,
            i: usize,
            j: usize,
            found: &mut bool
        ) {
            if *found || i >= board.len() || j >= board[0].len() || board[i][j] == '#' {
                return;
            }
            
            let ch = board[i][j];
            if !node.children.contains_key(&ch) {
                return;
            }
            
            let next_node = &node.children[&ch];
            if next_node.is_word {
                *found = true;
                return;
            }
            
            board[i][j] = '#';
            
            if i > 0 { dfs(board, next_node, i - 1, j, found); }
            if !*found && i + 1 < board.len() { dfs(board, next_node, i + 1, j, found); }
            if !*found && j > 0 { dfs(board, next_node, i, j - 1, found); }
            if !*found && j + 1 < board[0].len() { dfs(board, next_node, i, j + 1, found); }
            
            board[i][j] = ch;
        }
        
        for i in 0..m {
            for j in 0..n {
                dfs(&mut board, &trie, i, j, &mut found);
                if found {
                    return true;
                }
            }
        }
        
        false
    }
}

struct TrieNode {
    children: HashMap<char, TrieNode>,
    is_word: bool,
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            children: HashMap::new(),
            is_word: false,
        }
    }
    
    fn insert(&mut self, word: &str) {
        let mut current = self;
        for ch in word.chars() {
            current = current.children.entry(ch).or_insert_with(TrieNode::new);
        }
        current.is_word = true;
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
        
        // Example 1: Word exists
        let board1 = vec![
            vec!['A','B','C','E'],
            vec!['S','F','C','S'],
            vec!['A','D','E','E']
        ];
        assert_eq!(solution.exist(board1, "ABCCED".to_string()), true);
        
        // Example 2: Word exists
        let board2 = vec![
            vec!['A','B','C','E'],
            vec!['S','F','C','S'],
            vec!['A','D','E','E']
        ];
        assert_eq!(solution.exist(board2, "SEE".to_string()), true);
        
        // Example 3: Word doesn't exist (can't reuse cells)
        let board3 = vec![
            vec!['A','B','C','E'],
            vec!['S','F','C','S'],
            vec!['A','D','E','E']
        ];
        assert_eq!(solution.exist(board3, "ABCB".to_string()), false);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // Single cell board
        assert_eq!(solution.exist(vec![vec!['A']], "A".to_string()), true);
        assert_eq!(solution.exist(vec![vec!['A']], "B".to_string()), false);
        
        // Single row
        let board = vec![vec!['A','B','C']];
        assert_eq!(solution.exist(board.clone(), "ABC".to_string()), true);
        assert_eq!(solution.exist(board.clone(), "ACB".to_string()), false);
        
        // Single column
        let board = vec![vec!['A'], vec!['B'], vec!['C']];
        assert_eq!(solution.exist(board.clone(), "ABC".to_string()), true);
        assert_eq!(solution.exist(board.clone(), "ACB".to_string()), false);
        
        // Empty word
        assert_eq!(solution.exist(vec![vec!['A']], "".to_string()), false);
    }

    #[test]
    fn test_path_patterns() {
        let solution = setup();
        
        // Straight line path
        let board = vec![
            vec!['A','B','C'],
            vec!['D','E','F'],
            vec!['G','H','I']
        ];
        assert_eq!(solution.exist(board.clone(), "ABC".to_string()), true);
        assert_eq!(solution.exist(board.clone(), "ADG".to_string()), true);
        assert_eq!(solution.exist(board.clone(), "AEI".to_string()), false); // No diagonal
        
        // L-shaped path
        assert_eq!(solution.exist(board.clone(), "ABEH".to_string()), true);
        
        // Snake path
        assert_eq!(solution.exist(board.clone(), "ABCFEH".to_string()), true);
    }

    #[test]
    fn test_cell_reuse() {
        let solution = setup();
        
        // Cannot reuse cells
        let board = vec![
            vec!['A','B'],
            vec!['C','D']
        ];
        assert_eq!(solution.exist(board.clone(), "ABCD".to_string()), false); // Would need to reuse B
        assert_eq!(solution.exist(board.clone(), "ABDC".to_string()), true);  // Valid path
        
        // Overlapping paths
        let board = vec![
            vec!['A','A','A'],
            vec!['A','A','A'],
            vec!['A','A','A']
        ];
        assert_eq!(solution.exist(board.clone(), "AAAAAAAAA".to_string()), true); // 9 A's - exact match
        assert_eq!(solution.exist(board.clone(), "AAAAAAAAAA".to_string()), false); // 10 A's - too many
    }

    #[test]
    fn test_complex_boards() {
        let solution = setup();
        
        // Complex path finding
        let board = vec![
            vec!['C','A','A'],
            vec!['A','A','A'],
            vec!['B','C','D']
        ];
        assert_eq!(solution.exist(board.clone(), "AAB".to_string()), true);
        
        // Maze-like board
        let board = vec![
            vec!['A','B','C','E'],
            vec!['S','F','E','S'],
            vec!['A','D','E','E']
        ];
        assert_eq!(solution.exist(board.clone(), "ABCESEEEFS".to_string()), true);
    }

    #[test]
    fn test_case_sensitivity() {
        let solution = setup();
        
        let board = vec![
            vec!['a','B','c'],
            vec!['D','e','F']
        ];
        assert_eq!(solution.exist(board.clone(), "aBc".to_string()), true);
        assert_eq!(solution.exist(board.clone(), "ABC".to_string()), false); // Case matters
        assert_eq!(solution.exist(board.clone(), "DeF".to_string()), true);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            (vec![vec!['A','B'], vec!['C','D']], "ABDC"),
            (vec![vec!['A','B','C','E'], vec!['S','F','C','S'], vec!['A','D','E','E']], "ABCCED"),
            (vec![vec!['A','B','C','E'], vec!['S','F','C','S'], vec!['A','D','E','E']], "SEE"),
            (vec![vec!['A','B','C','E'], vec!['S','F','C','S'], vec!['A','D','E','E']], "ABCB"),
            (vec![vec!['A']], "A"),
            (vec![vec!['A','A'], vec!['A','A']], "AAAA"),
        ];
        
        for (board, word) in test_cases {
            let word_string = word.to_string();
            let result1 = solution.exist(board.clone(), word_string.clone());
            let result2 = solution.exist_visited_set(board.clone(), word_string.clone());
            let result3 = solution.exist_optimized(board.clone(), word_string.clone());
            let result6 = solution.exist_trie(board.clone(), word_string.clone());
            
            assert_eq!(result1, result2, "DFS vs Visited Set mismatch for word: {}", word);
            assert_eq!(result2, result3, "Visited Set vs Optimized mismatch for word: {}", word);
            assert_eq!(result3, result6, "Optimized vs Trie mismatch for word: {}", word);
        }
    }

    #[test]
    fn test_boundary_conditions() {
        let solution = setup();
        
        // Maximum board size (6x6)
        let board = vec![
            vec!['A','B','C','D','E','F'],
            vec!['G','H','I','J','K','L'],
            vec!['M','N','O','P','Q','R'],
            vec!['S','T','U','V','W','X'],
            vec!['Y','Z','A','B','C','D'],
            vec!['E','F','G','H','I','J']
        ];
        
        // Path along edges
        assert_eq!(solution.exist(board.clone(), "ABCDEF".to_string()), true);
        assert_eq!(solution.exist(board.clone(), "AFGMSY".to_string()), true);
        
        // Maximum word length (15)
        assert_eq!(solution.exist(board.clone(), "ABCDEFGHIJKLMNO".to_string()), false);
    }

    #[test]
    fn test_performance_patterns() {
        let solution = setup();
        
        // All same character - worst case for pruning
        let board = vec![
            vec!['A','A','A'],
            vec!['A','A','A'],
            vec!['A','A','A']
        ];
        assert_eq!(solution.exist(board.clone(), "AAAAAAAAA".to_string()), true);
        
        // No matching start character - best case
        let board = vec![
            vec!['B','B','B'],
            vec!['B','B','B'],
            vec!['B','B','B']
        ];
        assert_eq!(solution.exist(board.clone(), "ABC".to_string()), false);
        
        // Multiple starting positions
        let board = vec![
            vec!['A','B','A'],
            vec!['B','A','B'],
            vec!['A','B','A']
        ];
        assert_eq!(solution.exist(board.clone(), "ABA".to_string()), true);
    }

    #[test]
    fn test_mathematical_properties() {
        let solution = setup();
        
        // Property: Word longer than total cells must be false
        let board = vec![vec!['A','B'], vec!['C','D']];
        assert_eq!(solution.exist(board.clone(), "ABCDE".to_string()), false);
        
        // Property: Word with character not in board must be false
        assert_eq!(solution.exist(board.clone(), "XYZ".to_string()), false);
        
        // Property: Single character always works if present
        assert_eq!(solution.exist(board.clone(), "A".to_string()), true);
        assert_eq!(solution.exist(board.clone(), "D".to_string()), true);
        assert_eq!(solution.exist(board.clone(), "E".to_string()), false);
    }

    #[test]
    fn test_zigzag_patterns() {
        let solution = setup();
        
        let board = vec![
            vec!['A','B','C'],
            vec!['F','E','D'],
            vec!['G','H','I']
        ];
        
        // Zigzag path
        assert_eq!(solution.exist(board.clone(), "ABCDE".to_string()), true);
        assert_eq!(solution.exist(board.clone(), "ABEDC".to_string()), true);
        
        // Spiral path
        assert_eq!(solution.exist(board.clone(), "ABCDEFGHI".to_string()), false); // Can't complete spiral
    }

    #[test]
    fn test_corner_cases() {
        let solution = setup();
        
        // Word at corners
        let board = vec![
            vec!['A','B','C'],
            vec!['D','E','F'],
            vec!['G','H','I']
        ];
        
        // Top-left corner
        assert_eq!(solution.exist(board.clone(), "ABE".to_string()), true);
        
        // Bottom-right corner
        assert_eq!(solution.exist(board.clone(), "IHE".to_string()), true);
        
        // All corners
        assert_eq!(solution.exist(board.clone(), "ACIG".to_string()), false); // Not connected
    }
}