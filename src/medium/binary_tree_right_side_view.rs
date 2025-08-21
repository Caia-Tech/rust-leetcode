//! Problem 199: Binary Tree Right Side View
//! 
//! Given the root of a binary tree, imagine yourself standing on the right side of it,
//! return the values of the nodes you can see ordered from top to bottom.
//! 
//! Example 1:
//!     1
//!    / \
//!   2   3
//!    \   \
//!     5   4
//! Input: root = [1,2,3,null,5,null,4]
//! Output: [1,3,4]
//! 
//! Example 2:
//!     1
//!    /
//!   3
//! Input: root = [1,null,3]
//! Output: [1,3]
//! 
//! Example 3:
//! Input: root = []
//! Output: []
//! 
//! Constraints:
//! - The number of nodes in the tree is in the range [0, 100].
//! - -100 <= Node.val <= 100

use std::rc::Rc;
use std::cell::RefCell;
use std::collections::VecDeque;
use crate::utils::data_structures::TreeNode;

pub struct Solution;

impl Solution {
    /// Approach 1: BFS Level Order Traversal
    /// 
    /// Performs level order traversal and takes the rightmost node from each level.
    /// Uses a queue to process nodes level by level.
    /// 
    /// Time Complexity: O(n) where n is number of nodes
    /// Space Complexity: O(w) where w is maximum width of tree
    pub fn right_side_view_bfs(&self, root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        if root.is_none() {
            return vec![];
        }
        
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(root.unwrap());
        
        while !queue.is_empty() {
            let level_size = queue.len();
            let mut rightmost = 0;
            
            for _i in 0..level_size {
                let node = queue.pop_front().unwrap();
                let node_borrowed = node.borrow();
                rightmost = node_borrowed.val; // Last node in level will be rightmost
                
                if let Some(left) = &node_borrowed.left {
                    queue.push_back(left.clone());
                }
                if let Some(right) = &node_borrowed.right {
                    queue.push_back(right.clone());
                }
            }
            
            result.push(rightmost);
        }
        
        result
    }
    
    /// Approach 2: DFS with Level Tracking (Preorder)
    /// 
    /// Uses DFS but processes right child before left child. For each level,
    /// the first node we encounter will be the rightmost.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(h) for recursion stack, where h is height
    pub fn right_side_view_dfs_preorder(&self, root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        let mut result = Vec::new();
        self.dfs_preorder(root.as_ref(), 0, &mut result);
        result
    }
    
    fn dfs_preorder(&self, node: Option<&Rc<RefCell<TreeNode>>>, level: usize, result: &mut Vec<i32>) {
        if let Some(node) = node {
            let node_borrowed = node.borrow();
            
            // If this is the first node we see at this level, it's the rightmost
            if result.len() == level {
                result.push(node_borrowed.val);
            }
            
            // Visit right first, then left
            self.dfs_preorder(node_borrowed.right.as_ref(), level + 1, result);
            self.dfs_preorder(node_borrowed.left.as_ref(), level + 1, result);
        }
    }
    
    /// Approach 3: DFS with Maximum Depth Tracking
    /// 
    /// Uses DFS to process right subtree first, ensuring we see the rightmost
    /// node at each level before any left nodes at the same level.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(h)
    pub fn right_side_view_dfs_max_depth(&self, root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        let mut result = Vec::new();
        let mut max_depth = -1;
        self.dfs_max_depth(root.as_ref(), 0, &mut result, &mut max_depth);
        result
    }
    
    fn dfs_max_depth(&self, node: Option<&Rc<RefCell<TreeNode>>>, level: i32, result: &mut Vec<i32>, max_depth: &mut i32) {
        if let Some(node) = node {
            let node_borrowed = node.borrow();
            
            // If this is the first time we see this level, it must be rightmost
            if level > *max_depth {
                result.push(node_borrowed.val);
                *max_depth = level;
            }
            
            // Visit right first, then left
            self.dfs_max_depth(node_borrowed.right.as_ref(), level + 1, result, max_depth);
            self.dfs_max_depth(node_borrowed.left.as_ref(), level + 1, result, max_depth);
        }
    }
    
    /// Approach 4: BFS with Explicit Level Tracking
    /// 
    /// Uses BFS but explicitly tracks the level of each node in the queue.
    /// Maintains a map of level to rightmost node value.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(w)
    pub fn right_side_view_level_tracking(&self, root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        if root.is_none() {
            return vec![];
        }
        
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back((root.unwrap(), 0));
        
        while let Some((node, level)) = queue.pop_front() {
            let node_borrowed = node.borrow();
            
            // Ensure result has enough levels
            if result.len() <= level {
                result.resize(level + 1, 0);
            }
            
            // Always update with current node (rightmost will be last)
            result[level] = node_borrowed.val;
            
            if let Some(left) = &node_borrowed.left {
                queue.push_back((left.clone(), level + 1));
            }
            if let Some(right) = &node_borrowed.right {
                queue.push_back((right.clone(), level + 1));
            }
        }
        
        result
    }
    
    /// Approach 5: Iterative DFS with Stack
    /// 
    /// Uses an explicit stack to simulate DFS traversal. Processes nodes
    /// in right-to-left order to ensure rightmost nodes are seen first.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(h)
    pub fn right_side_view_iterative_dfs(&self, root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        if root.is_none() {
            return vec![];
        }
        
        let mut result = Vec::new();
        let mut stack = Vec::new();
        stack.push((root.unwrap(), 0));
        
        while let Some((node, level)) = stack.pop() {
            let node_borrowed = node.borrow();
            
            // If this is the first node at this level, it's the rightmost
            if result.len() == level {
                result.push(node_borrowed.val);
            }
            
            // Push left first, then right (so right is processed first)
            if let Some(left) = &node_borrowed.left {
                stack.push((left.clone(), level + 1));
            }
            if let Some(right) = &node_borrowed.right {
                stack.push((right.clone(), level + 1));
            }
        }
        
        result
    }
    
    /// Approach 6: Reverse BFS (Bottom-up)
    /// 
    /// Performs BFS but builds result from bottom-up perspective.
    /// Alternative approach that processes levels in reverse order conceptually.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(w) where w is maximum width
    pub fn right_side_view_reverse_bfs(&self, root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        if root.is_none() {
            return vec![];
        }
        
        let mut levels = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(root.unwrap());
        
        // Collect all levels first
        while !queue.is_empty() {
            let level_size = queue.len();
            let mut level = Vec::new();
            
            for _ in 0..level_size {
                let node = queue.pop_front().unwrap();
                let node_borrowed = node.borrow();
                level.push(node_borrowed.val);
                
                if let Some(left) = &node_borrowed.left {
                    queue.push_back(left.clone());
                }
                if let Some(right) = &node_borrowed.right {
                    queue.push_back(right.clone());
                }
            }
            
            levels.push(level);
        }
        
        // Extract rightmost from each level
        levels.into_iter().map(|level| *level.last().unwrap()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;
    use std::cell::RefCell;
    
    fn build_tree(vals: Vec<Option<i32>>) -> Option<Rc<RefCell<TreeNode>>> {
        if vals.is_empty() || vals[0].is_none() {
            return None;
        }
        
        let root = Rc::new(RefCell::new(TreeNode::new(vals[0].unwrap())));
        let mut queue = VecDeque::new();
        queue.push_back(root.clone());
        
        let mut i = 1;
        while !queue.is_empty() && i < vals.len() {
            if let Some(node) = queue.pop_front() {
                if i < vals.len() && vals[i].is_some() {
                    let left = Rc::new(RefCell::new(TreeNode::new(vals[i].unwrap())));
                    node.borrow_mut().left = Some(left.clone());
                    queue.push_back(left);
                }
                i += 1;
                
                if i < vals.len() && vals[i].is_some() {
                    let right = Rc::new(RefCell::new(TreeNode::new(vals[i].unwrap())));
                    node.borrow_mut().right = Some(right.clone());
                    queue.push_back(right);
                }
                i += 1;
            }
        }
        
        Some(root)
    }
    
    #[test]
    fn test_example_1() {
        let solution = Solution;
        
        // Example 1: [1,2,3,null,5,null,4] -> [1,3,4]
        let root = build_tree(vec![Some(1), Some(2), Some(3), None, Some(5), None, Some(4)]);
        let expected = vec![1, 3, 4];
        
        assert_eq!(solution.right_side_view_bfs(root), expected);
    }
    
    #[test]
    fn test_example_2() {
        let solution = Solution;
        
        // Example 2: [1,null,3] -> [1,3]
        let root = build_tree(vec![Some(1), None, Some(3)]);
        let expected = vec![1, 3];
        
        assert_eq!(solution.right_side_view_bfs(root), expected);
    }
    
    #[test]
    fn test_empty_tree() {
        let solution = Solution;
        
        // Example 3: [] -> []
        let root = None;
        let expected: Vec<i32> = vec![];
        
        assert_eq!(solution.right_side_view_bfs(root), expected);
    }
    
    #[test]
    fn test_single_node() {
        let solution = Solution;
        
        let root = build_tree(vec![Some(1)]);
        let expected = vec![1];
        
        assert_eq!(solution.right_side_view_bfs(root), expected);
    }
    
    #[test]
    fn test_dfs_preorder() {
        let solution = Solution;
        
        let root = build_tree(vec![Some(1), Some(2), Some(3), None, Some(5), None, Some(4)]);
        let expected = vec![1, 3, 4];
        
        assert_eq!(solution.right_side_view_dfs_preorder(root), expected);
    }
    
    #[test]
    fn test_dfs_max_depth() {
        let solution = Solution;
        
        let root = build_tree(vec![Some(1), Some(2), Some(3), None, Some(5), None, Some(4)]);
        let expected = vec![1, 3, 4];
        
        assert_eq!(solution.right_side_view_dfs_max_depth(root), expected);
    }
    
    #[test]
    fn test_level_tracking() {
        let solution = Solution;
        
        let root = build_tree(vec![Some(1), Some(2), Some(3), None, Some(5), None, Some(4)]);
        let expected = vec![1, 3, 4];
        
        assert_eq!(solution.right_side_view_level_tracking(root), expected);
    }
    
    #[test]
    fn test_iterative_dfs() {
        let solution = Solution;
        
        let root = build_tree(vec![Some(1), Some(2), Some(3), None, Some(5), None, Some(4)]);
        let expected = vec![1, 3, 4];
        
        assert_eq!(solution.right_side_view_iterative_dfs(root), expected);
    }
    
    #[test]
    fn test_left_skewed_tree() {
        let solution = Solution;
        
        // Tree: 1 -> 2 -> 3 -> 4 (all left children)
        let root = build_tree(vec![Some(1), Some(2), None, Some(3), None, Some(4)]);
        let expected = vec![1, 2, 3, 4];
        
        assert_eq!(solution.right_side_view_bfs(root), expected);
    }
    
    #[test]
    fn test_right_skewed_tree() {
        let solution = Solution;
        
        // Tree: 1 -> 2 -> 3 -> 4 (all right children)
        let root = build_tree(vec![Some(1), None, Some(2), None, Some(3), None, Some(4)]);
        let expected = vec![1, 2, 3, 4];
        
        assert_eq!(solution.right_side_view_bfs(root), expected);
    }
    
    #[test]
    fn test_perfect_binary_tree() {
        let solution = Solution;
        
        // Perfect binary tree
        let root = build_tree(vec![Some(1), Some(2), Some(3), Some(4), Some(5), Some(6), Some(7)]);
        let expected = vec![1, 3, 7];
        
        assert_eq!(solution.right_side_view_bfs(root), expected);
    }
    
    #[test]
    fn test_complex_tree() {
        let solution = Solution;
        
        // More complex structure
        let root = build_tree(vec![
            Some(1), Some(2), Some(3), Some(4), Some(5), Some(6), Some(7),
            Some(8), Some(9), Some(10), Some(11), Some(12), Some(13), Some(14), Some(15)
        ]);
        let expected = vec![1, 3, 7, 15];
        
        assert_eq!(solution.right_side_view_bfs(root), expected);
    }
    
    #[test]
    fn test_unbalanced_left_heavy() {
        let solution = Solution;
        
        // Tree heavily weighted to the left but with some right nodes
        let root = build_tree(vec![Some(1), Some(2), Some(3), Some(4), None, None, Some(7), Some(8)]);
        let expected = vec![1, 3, 7, 8];
        
        assert_eq!(solution.right_side_view_bfs(root), expected);
    }
    
    #[test]
    fn test_deep_tree() {
        let solution = Solution;
        
        // Deep tree structure
        let root = build_tree(vec![
            Some(1), Some(2), Some(3), Some(4), Some(5), Some(6), Some(7),
            None, None, Some(10), Some(11), None, None, Some(14), Some(15)
        ]);
        let expected = vec![1, 3, 7, 15];
        
        assert_eq!(solution.right_side_view_bfs(root), expected);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let solution = Solution;
        
        let test_cases = vec![
            vec![Some(1), Some(2), Some(3), None, Some(5), None, Some(4)],
            vec![Some(1), None, Some(3)],
            vec![Some(1)],
            vec![Some(1), Some(2), Some(3), Some(4), Some(5), Some(6), Some(7)],
            vec![Some(1), Some(2), None, Some(3), None, Some(4)],
            vec![Some(1), None, Some(2), None, Some(3), None, Some(4)],
        ];
        
        for tree_vals in test_cases {
            let root1 = build_tree(tree_vals.clone());
            let root2 = build_tree(tree_vals.clone());
            let root3 = build_tree(tree_vals.clone());
            let root4 = build_tree(tree_vals.clone());
            let root5 = build_tree(tree_vals.clone());
            
            let result1 = solution.right_side_view_bfs(root1);
            let result2 = solution.right_side_view_dfs_preorder(root2);
            let result3 = solution.right_side_view_dfs_max_depth(root3);
            let result4 = solution.right_side_view_level_tracking(root4);
            let result5 = solution.right_side_view_iterative_dfs(root5);
            
            assert_eq!(result1, result2, "BFS and DFS preorder differ for {:?}", tree_vals);
            assert_eq!(result1, result3, "BFS and DFS max depth differ for {:?}", tree_vals);
            assert_eq!(result1, result4, "BFS and level tracking differ for {:?}", tree_vals);
            assert_eq!(result1, result5, "BFS and iterative DFS differ for {:?}", tree_vals);
        }
    }
    
    #[test]
    fn test_edge_cases() {
        let solution = Solution;
        
        // Tree with negative values
        let root = build_tree(vec![Some(-1), Some(-2), Some(-3)]);
        let expected = vec![-1, -3];
        assert_eq!(solution.right_side_view_bfs(root), expected);
        
        // Tree with zero
        let root = build_tree(vec![Some(0), Some(-1), Some(1)]);
        let expected = vec![0, 1];
        assert_eq!(solution.right_side_view_bfs(root), expected);
        
        // Tree with maximum values (within constraints)
        let root = build_tree(vec![Some(100), Some(-100), Some(50)]);
        let expected = vec![100, 50];
        assert_eq!(solution.right_side_view_bfs(root), expected);
    }
    
    #[test]
    fn test_large_single_path() {
        let solution = Solution;
        
        // Create a path going only left
        let root = build_tree(vec![
            Some(1), Some(2), None, Some(3), None, Some(4), None,
            Some(5), None, Some(6), None, Some(7)
        ]);
        let expected = vec![1, 2, 3, 4, 5, 6, 7];
        
        assert_eq!(solution.right_side_view_dfs_preorder(root), expected);
    }
    
    #[test]
    fn test_alternating_pattern() {
        let solution = Solution;
        
        // Tree with alternating left-right pattern
        let root = build_tree(vec![Some(1), Some(2), None, None, Some(3), Some(4)]);
        let expected = vec![1, 2, 3, 4];
        
        assert_eq!(solution.right_side_view_bfs(root), expected);
    }
    
    #[test]
    fn test_sparse_tree() {
        let solution = Solution;
        
        // Simpler sparse tree - tree with some missing nodes
        let root = build_tree(vec![Some(1), Some(2), Some(3), Some(4), None, None, Some(7)]);
        let expected = vec![1, 3, 7];
        
        assert_eq!(solution.right_side_view_bfs(root), expected);
    }
    
    #[test]
    fn test_performance_with_wide_tree() {
        let solution = Solution;
        
        // Wide tree (more breadth than depth)
        let vals: Vec<Option<i32>> = (1..=31).map(Some).collect();
        let root = build_tree(vals);
        
        let result = solution.right_side_view_bfs(root);
        // Should see rightmost at each level: 1, 3, 7, 15, 31
        let expected = vec![1, 3, 7, 15, 31];
        
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_memory_efficiency() {
        let solution = Solution;
        
        // Test with a structure that could cause memory issues
        let root = build_tree(vec![
            Some(1), Some(2), Some(3), Some(4), Some(5), Some(6), Some(7),
            Some(8), Some(9), Some(10), Some(11), Some(12), Some(13), Some(14), Some(15),
            Some(16), Some(17), Some(18), Some(19), Some(20), Some(21), Some(22), Some(23),
            Some(24), Some(25), Some(26), Some(27), Some(28), Some(29), Some(30), Some(31)
        ]);
        
        // Should complete without memory issues
        let result = solution.right_side_view_iterative_dfs(root);
        assert!(!result.is_empty());
        assert_eq!(result, vec![1, 3, 7, 15, 31]);
    }
}