//! # 48. Rotate Image
//!
//! You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).
//!
//! You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. 
//! DO NOT allocate another 2D matrix and do the rotation.
//!
//! **Example 1:**
//! ```text
//! Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
//! Output: [[7,4,1],[8,5,2],[9,6,3]]
//! ```
//!
//! **Example 2:**
//! ```text
//! Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
//! Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
//! ```
//!
//! **Constraints:**
//! - n == matrix.length == matrix[i].length
//! - 1 <= n <= 20
//! - -1000 <= matrix[i][j] <= 1000

/// Solution for Rotate Image - simplified working version
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
    pub fn rotate(&self, matrix: &mut Vec<Vec<i32>>) {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_examples() {
        let solution = Solution;
        
        // Example 1: 3x3 matrix
        let mut matrix1 = vec![vec![1,2,3], vec![4,5,6], vec![7,8,9]];
        let expected1 = vec![vec![7,4,1], vec![8,5,2], vec![9,6,3]];
        solution.rotate(&mut matrix1);
        assert_eq!(matrix1, expected1);
        
        // Example 2: 4x4 matrix
        let mut matrix2 = vec![vec![5,1,9,11], vec![2,4,8,10], vec![13,3,6,7], vec![15,14,12,16]];
        let expected2 = vec![vec![15,13,2,5], vec![14,3,4,1], vec![12,6,8,9], vec![16,7,10,11]];
        solution.rotate(&mut matrix2);
        assert_eq!(matrix2, expected2);
    }

    #[test]
    fn test_single_element() {
        let solution = Solution;
        
        // 1x1 matrix
        let mut matrix = vec![vec![1]];
        let expected = vec![vec![1]];
        solution.rotate(&mut matrix);
        assert_eq!(matrix, expected);
    }

    #[test]
    fn test_two_by_two() {
        let solution = Solution;
        
        // 2x2 matrix
        let mut matrix = vec![vec![1,2], vec![3,4]];
        let expected = vec![vec![3,1], vec![4,2]];
        solution.rotate(&mut matrix);
        assert_eq!(matrix, expected);
    }

    #[test]
    fn test_edge_cases() {
        let solution = Solution;
        
        // Matrix with negative numbers
        let mut matrix = vec![vec![-1,-2], vec![-3,-4]];
        let expected = vec![vec![-3,-1], vec![-4,-2]];
        solution.rotate(&mut matrix);
        assert_eq!(matrix, expected);
        
        // Matrix with zeros
        let mut matrix = vec![vec![0,1], vec![2,0]];
        let expected = vec![vec![2,0], vec![0,1]];
        solution.rotate(&mut matrix);
        assert_eq!(matrix, expected);
        
        // Matrix with boundary constraint values
        let mut matrix = vec![vec![-1000, 1000], vec![0, -500]];
        let expected = vec![vec![0, -1000], vec![-500, 1000]];
        solution.rotate(&mut matrix);
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
        solution.rotate(&mut matrix);
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
            solution.rotate(&mut matrix);
        }
        
        assert_eq!(matrix, original);
    }

    #[test]
    fn test_symmetric_matrix() {
        let solution = Solution;
        
        // Symmetric matrix
        let mut matrix = vec![vec![1,2,3], vec![2,4,5], vec![3,5,6]];
        let expected = vec![vec![3,2,1], vec![5,4,2], vec![6,5,3]];
        solution.rotate(&mut matrix);
        assert_eq!(matrix, expected);
    }

    #[test]
    fn test_all_same_elements() {
        let solution = Solution;
        
        // Matrix with all same elements
        let mut matrix = vec![vec![5,5,5], vec![5,5,5], vec![5,5,5]];
        let expected = vec![vec![5,5,5], vec![5,5,5], vec![5,5,5]];
        solution.rotate(&mut matrix);
        assert_eq!(matrix, expected);
    }
}