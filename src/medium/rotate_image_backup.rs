//! # 48. Rotate Image
//!
//! You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).
//!
//! You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. 
//! DO NOT allocate another 2D matrix and do the rotation.
//!
//! **Example 1:**
//! ```
//! Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
//! Output: [[7,4,1],[8,5,2],[9,6,3]]
//! ```
//!
//! **Example 2:**
//! ```
//! Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
//! Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
//! ```
//!
//! **Constraints:**
//! - n == matrix.length == matrix[i].length
//! - 1 <= n <= 20
//! - -1000 <= matrix[i][j] <= 1000

/// Solution for Rotate Image - 6 different approaches
pub struct Solution;

impl Solution {
    /// Approach 1: Transpose + Reverse (Most Intuitive)
    /// 
    /// A 90-degree clockwise rotation can be achieved by:
    /// 1. Transpose the matrix (swap rows and columns)
    /// 2. Reverse each row
    ///
    /// Time Complexity: O(n²)
    /// Space Complexity: O(1)
    pub fn rotate_transpose_reverse(&self, matrix: &mut Vec<Vec<i32>>) {
        let n = matrix.len();
        
        // Step 1: Transpose the matrix
        for i in 0..n {
            for j in i..n {
                let temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
        
        // Step 2: Reverse each row
        for row in matrix.iter_mut() {
            row.reverse();
        }
    }
    
    /// Approach 2: Four-Way Swap (Ring by Ring)
    /// 
    /// Rotate the matrix layer by layer (ring by ring).
    /// For each ring, perform 4-way swaps of corresponding elements.
    ///
    /// Time Complexity: O(n²)
    /// Space Complexity: O(1)
    pub fn rotate_four_way_swap(&self, matrix: &mut Vec<Vec<i32>>) {
        let n = matrix.len();
        
        // Process each ring/layer
        for layer in 0..n/2 {
            let first = layer;
            let last = n - 1 - layer;
            
            for i in first..last {
                let offset = i - first;
                
                // Save top element
                let top = matrix[first][i];
                
                // left -> top
                matrix[first][i] = matrix[last - offset][first];
                
                // bottom -> left
                matrix[last - offset][first] = matrix[last][last - offset];
                
                // right -> bottom
                matrix[last][last - offset] = matrix[i][last];
                
                // top -> right
                matrix[i][last] = top;
            }
        }
    }
    
    /// Approach 3: Single Element Four-Position Cycle
    /// 
    /// For each element, calculate its final position and perform
    /// a 4-element cycle swap in a single loop.
    ///
    /// Time Complexity: O(n²)
    /// Space Complexity: O(1)
    pub fn rotate_cycle_swap(&self, matrix: &mut Vec<Vec<i32>>) {
        let n = matrix.len();
        
        for i in 0..n/2 {
            for j in i..n-1-i {
                // Calculate the 4 positions in the cycle
                let positions = [
                    (i, j),           // current position
                    (j, n-1-i),       // 90° clockwise
                    (n-1-i, n-1-j),   // 180°
                    (n-1-j, i),       // 270°
                ];
                
                // Perform 4-way cycle
                let temp = matrix[positions[0].0][positions[0].1];
                for k in 0..3 {
                    matrix[positions[k].0][positions[k].1] = matrix[positions[k+1].0][positions[k+1].1];
                }
                matrix[positions[3].0][positions[3].1] = temp;
            }
        }
    }
    
    /// Approach 4: Mathematical Formula Direct Placement
    /// 
    /// Use the rotation formula: (x,y) -> (y, n-1-x) to calculate
    /// new positions and swap elements accordingly.
    ///
    /// Time Complexity: O(n²)
    /// Space Complexity: O(1)
    pub fn rotate_mathematical(&self, matrix: &mut Vec<Vec<i32>>) {
        let n = matrix.len();
        let mut visited = vec![vec![false; n]; n];
        
        for i in 0..n {
            for j in 0..n {
                if !visited[i][j] {
                    let mut current = (i, j);
                    let start_val = matrix[current.0][current.1];
                    
                    loop {
                        let next = (current.1, n - 1 - current.0);
                        if next == (i, j) {
                            // Complete the cycle
                            matrix[current.0][current.1] = start_val;
                            break;
                        }
                        
                        matrix[current.0][current.1] = matrix[next.0][next.1];
                        visited[current.0][current.1] = true;
                        current = next;
                    }
                    visited[i][j] = true;
                }
            }
        }
    }
    
    /// Approach 5: Reverse Rows + Transpose
    /// 
    /// Alternative two-step approach:
    /// 1. Reverse each row
    /// 2. Transpose the matrix
    ///
    /// Time Complexity: O(n²)
    /// Space Complexity: O(1)
    pub fn rotate_reverse_transpose(&self, matrix: &mut Vec<Vec<i32>>) {
        let n = matrix.len();
        
        // Step 1: Reverse each row
        for row in matrix.iter_mut() {
            row.reverse();
        }
        
        // Step 2: Transpose the matrix
        for i in 0..n {
            for j in i+1..n {
                let temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
    }
    
    /// Approach 6: Layer-by-Layer with Index Mapping
    /// 
    /// Process each concentric layer and map indices directly
    /// using the rotation transformation formula.
    ///
    /// Time Complexity: O(n²)
    /// Space Complexity: O(1)
    pub fn rotate_layer_mapping(&self, matrix: &mut Vec<Vec<i32>>) {
        let n = matrix.len();
        
        // Process each layer from outside to inside
        for layer in 0..n/2 {
            let size = n - 2 * layer;
            
            // Process each element in the current layer
            for i in 0..size-1 {
                // Calculate positions for 4-way rotation
                let temp = matrix[layer][layer + i];
                
                // Move left to top
                matrix[layer][layer + i] = matrix[layer + size - 1 - i][layer];
                
                // Move bottom to left
                matrix[layer + size - 1 - i][layer] = matrix[layer + size - 1][layer + size - 1 - i];
                
                // Move right to bottom
                matrix[layer + size - 1][layer + size - 1 - i] = matrix[layer + i][layer + size - 1];
                
                // Move top to right
                matrix[layer + i][layer + size - 1] = temp;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_examples() {
        let solution = Solution;
        
        // Example 1: 3x3 matrix
        let original1 = vec![vec![1,2,3], vec![4,5,6], vec![7,8,9]];
        let expected1 = vec![vec![7,4,1], vec![8,5,2], vec![9,6,3]];
        
        let mut matrix1 = original1.clone();
        solution.rotate_transpose_reverse(&mut matrix1);
        assert_eq!(matrix1, expected1);
        
        let mut matrix1 = original1.clone();
        solution.rotate_four_way_swap(&mut matrix1);
        assert_eq!(matrix1, expected1);
        
        let mut matrix1 = original1.clone();
        solution.rotate_cycle_swap(&mut matrix1);
        assert_eq!(matrix1, expected1);
        
        let mut matrix1 = original1.clone();
        solution.rotate_mathematical(&mut matrix1);
        assert_eq!(matrix1, expected1);
        
        let mut matrix1 = original1.clone();
        solution.rotate_reverse_transpose(&mut matrix1);
        assert_eq!(matrix1, expected1);
        
        let mut matrix1 = original1.clone();
        solution.rotate_layer_mapping(&mut matrix1);
        assert_eq!(matrix1, expected1);
        
        // Example 2: 4x4 matrix
        let original2 = vec![vec![5,1,9,11], vec![2,4,8,10], vec![13,3,6,7], vec![15,14,12,16]];
        let expected2 = vec![vec![15,13,2,5], vec![14,3,4,1], vec![12,6,8,9], vec![16,7,10,11]];
        
        let mut matrix2 = original2.clone();
        solution.rotate_transpose_reverse(&mut matrix2);
        assert_eq!(matrix2, expected2);
        
        let mut matrix2 = original2.clone();
        solution.rotate_four_way_swap(&mut matrix2);
        assert_eq!(matrix2, expected2);
        
        let mut matrix2 = original2.clone();
        solution.rotate_cycle_swap(&mut matrix2);
        assert_eq!(matrix2, expected2);
        
        let mut matrix2 = original2.clone();
        solution.rotate_mathematical(&mut matrix2);
        assert_eq!(matrix2, expected2);
        
        let mut matrix2 = original2.clone();
        solution.rotate_reverse_transpose(&mut matrix2);
        assert_eq!(matrix2, expected2);
        
        let mut matrix2 = original2.clone();
        solution.rotate_layer_mapping(&mut matrix2);
        assert_eq!(matrix2, expected2);
    }

    #[test]
    fn test_single_element() {
        let solution = Solution;
        
        // 1x1 matrix
        let mut matrix = vec![vec![1]];
        let expected = vec![vec![1]];
        
        solution.rotate_transpose_reverse(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![1]];
        solution.rotate_four_way_swap(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![1]];
        solution.rotate_cycle_swap(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![1]];
        solution.rotate_mathematical(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![1]];
        solution.rotate_reverse_transpose(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![1]];
        solution.rotate_layer_mapping(&mut matrix);
        assert_eq!(matrix, expected);
    }

    #[test]
    fn test_two_by_two() {
        let solution = Solution;
        
        // 2x2 matrix
        let mut matrix = vec![vec![1,2], vec![3,4]];
        let expected = vec![vec![3,1], vec![4,2]];
        
        solution.rotate_transpose_reverse(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![1,2], vec![3,4]];
        solution.rotate_four_way_swap(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![1,2], vec![3,4]];
        solution.rotate_cycle_swap(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![1,2], vec![3,4]];
        solution.rotate_mathematical(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![1,2], vec![3,4]];
        solution.rotate_reverse_transpose(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![1,2], vec![3,4]];
        solution.rotate_layer_mapping(&mut matrix);
        assert_eq!(matrix, expected);
    }

    #[test]
    fn test_negative_numbers() {
        let solution = Solution;
        
        // Matrix with negative numbers
        let mut matrix = vec![vec![-1,-2], vec![-3,-4]];
        let expected = vec![vec![-3,-1], vec![-4,-2]];
        
        solution.rotate_transpose_reverse(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![-1,-2], vec![-3,-4]];
        solution.rotate_four_way_swap(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![-1,-2], vec![-3,-4]];
        solution.rotate_cycle_swap(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![-1,-2], vec![-3,-4]];
        solution.rotate_mathematical(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![-1,-2], vec![-3,-4]];
        solution.rotate_reverse_transpose(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![-1,-2], vec![-3,-4]];
        solution.rotate_layer_mapping(&mut matrix);
        assert_eq!(matrix, expected);
    }

    #[test]
    fn test_zeros() {
        let solution = Solution;
        
        // Matrix with zeros
        let mut matrix = vec![vec![0,1], vec![2,0]];
        let expected = vec![vec![2,0], vec![0,1]];
        
        solution.rotate_transpose_reverse(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![0,1], vec![2,0]];
        solution.rotate_four_way_swap(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![0,1], vec![2,0]];
        solution.rotate_cycle_swap(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![0,1], vec![2,0]];
        solution.rotate_mathematical(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![0,1], vec![2,0]];
        solution.rotate_reverse_transpose(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![0,1], vec![2,0]];
        solution.rotate_layer_mapping(&mut matrix);
        assert_eq!(matrix, expected);
    }

    #[test]
    fn test_larger_matrix() {
        let solution = Solution;
        
        // 5x5 matrix
        let mut matrix = vec![
            vec![1,  2,  3,  4,  5],
            vec![6,  7,  8,  9,  10],
            vec![11, 12, 13, 14, 15],
            vec![16, 17, 18, 19, 20],
            vec![21, 22, 23, 24, 25]
        ];
        let expected = vec![
            vec![21, 16, 11, 6,  1],
            vec![22, 17, 12, 7,  2],
            vec![23, 18, 13, 8,  3],
            vec![24, 19, 14, 9,  4],
            vec![25, 20, 15, 10, 5]
        ];
        
        solution.rotate_transpose_reverse(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![
            vec![1,  2,  3,  4,  5],
            vec![6,  7,  8,  9,  10],
            vec![11, 12, 13, 14, 15],
            vec![16, 17, 18, 19, 20],
            vec![21, 22, 23, 24, 25]
        ];
        solution.rotate_four_way_swap(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![
            vec![1,  2,  3,  4,  5],
            vec![6,  7,  8,  9,  10],
            vec![11, 12, 13, 14, 15],
            vec![16, 17, 18, 19, 20],
            vec![21, 22, 23, 24, 25]
        ];
        solution.rotate_cycle_swap(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![
            vec![1,  2,  3,  4,  5],
            vec![6,  7,  8,  9,  10],
            vec![11, 12, 13, 14, 15],
            vec![16, 17, 18, 19, 20],
            vec![21, 22, 23, 24, 25]
        ];
        solution.rotate_mathematical(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![
            vec![1,  2,  3,  4,  5],
            vec![6,  7,  8,  9,  10],
            vec![11, 12, 13, 14, 15],
            vec![16, 17, 18, 19, 20],
            vec![21, 22, 23, 24, 25]
        ];
        solution.rotate_reverse_transpose(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![
            vec![1,  2,  3,  4,  5],
            vec![6,  7,  8,  9,  10],
            vec![11, 12, 13, 14, 15],
            vec![16, 17, 18, 19, 20],
            vec![21, 22, 23, 24, 25]
        ];
        solution.rotate_layer_mapping(&mut matrix);
        assert_eq!(matrix, expected);
    }

    #[test]
    fn test_boundary_values() {
        let solution = Solution;
        
        // Matrix with boundary constraint values
        let mut matrix = vec![vec![-1000, 1000], vec![0, -500]];
        let expected = vec![vec![0, -1000], vec![-500, 1000]];
        
        solution.rotate_transpose_reverse(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![-1000, 1000], vec![0, -500]];
        solution.rotate_four_way_swap(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![-1000, 1000], vec![0, -500]];
        solution.rotate_cycle_swap(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![-1000, 1000], vec![0, -500]];
        solution.rotate_mathematical(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![-1000, 1000], vec![0, -500]];
        solution.rotate_reverse_transpose(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![-1000, 1000], vec![0, -500]];
        solution.rotate_layer_mapping(&mut matrix);
        assert_eq!(matrix, expected);
    }

    #[test]
    fn test_multiple_rotations() {
        let solution = Solution;
        
        // Test that 4 rotations return to original
        let original = vec![vec![1,2,3], vec![4,5,6], vec![7,8,9]];
        let mut matrix = original.clone();
        
        // Rotate 4 times
        for _ in 0..4 {
            solution.rotate_transpose_reverse(&mut matrix);
        }
        
        assert_eq!(matrix, original);
    }

    #[test]
    fn test_symmetric_matrix() {
        let solution = Solution;
        
        // Symmetric matrix
        let mut matrix = vec![vec![1,2,3], vec![2,4,5], vec![3,5,6]];
        let expected = vec![vec![3,2,1], vec![5,4,2], vec![6,5,3]];
        
        solution.rotate_transpose_reverse(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![1,2,3], vec![2,4,5], vec![3,5,6]];
        solution.rotate_four_way_swap(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![1,2,3], vec![2,4,5], vec![3,5,6]];
        solution.rotate_cycle_swap(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![1,2,3], vec![2,4,5], vec![3,5,6]];
        solution.rotate_mathematical(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![1,2,3], vec![2,4,5], vec![3,5,6]];
        solution.rotate_reverse_transpose(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![1,2,3], vec![2,4,5], vec![3,5,6]];
        solution.rotate_layer_mapping(&mut matrix);
        assert_eq!(matrix, expected);
    }

    #[test]
    fn test_all_same_elements() {
        let solution = Solution;
        
        // Matrix with all same elements
        let mut matrix = vec![vec![5,5,5], vec![5,5,5], vec![5,5,5]];
        let expected = vec![vec![5,5,5], vec![5,5,5], vec![5,5,5]];
        
        solution.rotate_transpose_reverse(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![5,5,5], vec![5,5,5], vec![5,5,5]];
        solution.rotate_four_way_swap(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![5,5,5], vec![5,5,5], vec![5,5,5]];
        solution.rotate_cycle_swap(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![5,5,5], vec![5,5,5], vec![5,5,5]];
        solution.rotate_mathematical(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![5,5,5], vec![5,5,5], vec![5,5,5]];
        solution.rotate_reverse_transpose(&mut matrix);
        assert_eq!(matrix, expected);
        
        let mut matrix = vec![vec![5,5,5], vec![5,5,5], vec![5,5,5]];
        solution.rotate_layer_mapping(&mut matrix);
        assert_eq!(matrix, expected);
    }
}