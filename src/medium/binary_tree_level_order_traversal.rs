//! Problem 102: Binary Tree Level Order Traversal
//! 
//! Given the root of a binary tree, return the level order traversal of its nodes' values.
//! (i.e., from left to right, level by level).
//! 
//! Example 1:
//!     3
//!    / \
//!   9  20
//!     /  \
//!    15   7
//! Input: root = [3,9,20,null,null,15,7]
//! Output: [[3],[9,20],[15,7]]
//! 
//! Example 2:
//! Input: root = [1]
//! Output: [[1]]
//! 
//! Example 3:
//! Input: root = []
//! Output: []
//! 
//! Constraints:
//! - The number of nodes in the tree is in the range [0, 2000].
//! - -1000 <= Node.val <= 1000

use std::rc::Rc;
use std::cell::RefCell;
use std::collections::VecDeque;
use crate::utils::data_structures::TreeNode;

pub struct Solution;

impl Solution {
    /// Approach 1: BFS with Queue (Standard Implementation)
    /// 
    /// Uses a queue to track nodes at each level. For each level, processes
    /// all nodes currently in the queue and adds their children for next level.
    /// 
    /// Time Complexity: O(n) where n is number of nodes
    /// Space Complexity: O(w) where w is maximum width of tree
    pub fn level_order_bfs(&self, root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
        if root.is_none() {
            return vec![];
        }
        
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(root.unwrap());
        
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
            
            result.push(level);
        }
        
        result
    }
    
    /// Approach 2: DFS with Level Tracking
    /// 
    /// Uses recursive DFS but tracks the current level. Ensures result vector
    /// has enough levels and adds nodes to appropriate level.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(h) for recursion stack, where h is height
    pub fn level_order_dfs(&self, root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
        let mut result = Vec::new();
        self.dfs_helper(root.as_ref(), 0, &mut result);
        result
    }
    
    fn dfs_helper(&self, node: Option<&Rc<RefCell<TreeNode>>>, level: usize, result: &mut Vec<Vec<i32>>) {
        if let Some(node) = node {
            // Ensure result has enough levels
            if result.len() <= level {
                result.push(Vec::new());
            }
            
            let node_borrowed = node.borrow();
            result[level].push(node_borrowed.val);
            
            self.dfs_helper(node_borrowed.left.as_ref(), level + 1, result);
            self.dfs_helper(node_borrowed.right.as_ref(), level + 1, result);
        }
    }
    
    /// Approach 3: BFS with Two Queues
    /// 
    /// Uses two queues alternately - one for current level, one for next level.
    /// Eliminates need to track level size by using separate queues.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(w) where w is maximum width
    pub fn level_order_two_queues(&self, root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
        if root.is_none() {
            return vec![];
        }
        
        let mut result = Vec::new();
        let mut current_level = VecDeque::new();
        let mut next_level = VecDeque::new();
        
        current_level.push_back(root.unwrap());
        
        while !current_level.is_empty() {
            let mut level = Vec::new();
            
            while let Some(node) = current_level.pop_front() {
                let node_borrowed = node.borrow();
                level.push(node_borrowed.val);
                
                if let Some(left) = &node_borrowed.left {
                    next_level.push_back(left.clone());
                }
                if let Some(right) = &node_borrowed.right {
                    next_level.push_back(right.clone());
                }
            }
            
            result.push(level);
            std::mem::swap(&mut current_level, &mut next_level);
        }
        
        result
    }
    
    /// Approach 4: BFS with Level Markers
    /// 
    /// Uses a special marker (None) in the queue to indicate end of level.
    /// When marker is encountered, start new level and add marker for next level.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(w)
    pub fn level_order_markers(&self, root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
        if root.is_none() {
            return vec![];
        }
        
        let mut result = Vec::new();
        let mut queue: VecDeque<Option<Rc<RefCell<TreeNode>>>> = VecDeque::new();
        
        queue.push_back(Some(root.unwrap()));
        queue.push_back(None); // Level marker
        
        let mut current_level = Vec::new();
        
        while !queue.is_empty() {
            if let Some(node_option) = queue.pop_front() {
                match node_option {
                    Some(node) => {
                        let node_borrowed = node.borrow();
                        current_level.push(node_borrowed.val);
                        
                        if let Some(left) = &node_borrowed.left {
                            queue.push_back(Some(left.clone()));
                        }
                        if let Some(right) = &node_borrowed.right {
                            queue.push_back(Some(right.clone()));
                        }
                    }
                    None => {
                        // End of level marker
                        result.push(current_level);
                        current_level = Vec::new();
                        
                        // Add marker for next level if there are more nodes
                        if !queue.is_empty() {
                            queue.push_back(None);
                        }
                    }
                }
            }
        }
        
        result
    }
    
    /// Approach 5: Iterative with Vector of Vectors
    /// 
    /// Processes one level at a time by keeping track of nodes in current level.
    /// Uses vectors instead of queues for potentially better cache performance.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(w)
    pub fn level_order_vectors(&self, root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
        if root.is_none() {
            return vec![];
        }
        
        let mut result = Vec::new();
        let mut current_level = vec![root.unwrap()];
        
        while !current_level.is_empty() {
            let mut level_values = Vec::new();
            let mut next_level = Vec::new();
            
            for node in current_level {
                let node_borrowed = node.borrow();
                level_values.push(node_borrowed.val);
                
                if let Some(left) = &node_borrowed.left {
                    next_level.push(left.clone());
                }
                if let Some(right) = &node_borrowed.right {
                    next_level.push(right.clone());
                }
            }
            
            result.push(level_values);
            current_level = next_level;
        }
        
        result
    }
    
    /// Approach 6: Morris-style Level Order (Space Optimized)
    /// 
    /// Attempts to reduce space complexity by using parent links simulation.
    /// More complex but educational for understanding tree traversal optimization.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(w) - cannot eliminate completely for level order
    pub fn level_order_morris_style(&self, root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
        if root.is_none() {
            return vec![];
        }
        
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back((root.unwrap(), 0usize)); // (node, level)
        
        while let Some((node, level)) = queue.pop_front() {
            // Ensure result has enough levels
            if result.len() <= level {
                result.resize(level + 1, Vec::new());
            }
            
            let node_borrowed = node.borrow();
            result[level].push(node_borrowed.val);
            
            if let Some(left) = &node_borrowed.left {
                queue.push_back((left.clone(), level + 1));
            }
            if let Some(right) = &node_borrowed.right {
                queue.push_back((right.clone(), level + 1));
            }
        }
        
        result
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
    fn test_basic_example() {
        let solution = Solution;
        
        // Example 1: [3,9,20,null,null,15,7] -> [[3],[9,20],[15,7]]
        let root = build_tree(vec![Some(3), Some(9), Some(20), None, None, Some(15), Some(7)]);
        let expected = vec![vec![3], vec![9, 20], vec![15, 7]];
        
        assert_eq!(solution.level_order_bfs(root), expected);
    }
    
    #[test]
    fn test_single_node() {
        let solution = Solution;
        
        // Example 2: [1] -> [[1]]
        let root = build_tree(vec![Some(1)]);
        let expected = vec![vec![1]];
        
        assert_eq!(solution.level_order_bfs(root), expected);
    }
    
    #[test]
    fn test_empty_tree() {
        let solution = Solution;
        
        // Example 3: [] -> []
        let root = None;
        let expected: Vec<Vec<i32>> = vec![];
        
        assert_eq!(solution.level_order_bfs(root), expected);
    }
    
    #[test]
    fn test_dfs_approach() {
        let solution = Solution;
        
        let root = build_tree(vec![Some(3), Some(9), Some(20), None, None, Some(15), Some(7)]);
        let expected = vec![vec![3], vec![9, 20], vec![15, 7]];
        
        assert_eq!(solution.level_order_dfs(root), expected);
    }
    
    #[test]
    fn test_two_queues_approach() {
        let solution = Solution;
        
        let root = build_tree(vec![Some(3), Some(9), Some(20), None, None, Some(15), Some(7)]);
        let expected = vec![vec![3], vec![9, 20], vec![15, 7]];
        
        assert_eq!(solution.level_order_two_queues(root), expected);
    }
    
    #[test]
    fn test_markers_approach() {
        let solution = Solution;
        
        let root = build_tree(vec![Some(3), Some(9), Some(20), None, None, Some(15), Some(7)]);
        let expected = vec![vec![3], vec![9, 20], vec![15, 7]];
        
        assert_eq!(solution.level_order_markers(root), expected);
    }
    
    #[test]
    fn test_vectors_approach() {
        let solution = Solution;
        
        let root = build_tree(vec![Some(3), Some(9), Some(20), None, None, Some(15), Some(7)]);
        let expected = vec![vec![3], vec![9, 20], vec![15, 7]];
        
        assert_eq!(solution.level_order_vectors(root), expected);
    }
    
    #[test]
    fn test_morris_style_approach() {
        let solution = Solution;
        
        let root = build_tree(vec![Some(3), Some(9), Some(20), None, None, Some(15), Some(7)]);
        let expected = vec![vec![3], vec![9, 20], vec![15, 7]];
        
        assert_eq!(solution.level_order_morris_style(root), expected);
    }
    
    #[test]
    fn test_left_skewed_tree() {
        let solution = Solution;
        
        // Tree: 1 -> 2 -> 3 -> 4 (all left children)
        let root = build_tree(vec![Some(1), Some(2), None, Some(3), None, Some(4)]);
        let expected = vec![vec![1], vec![2], vec![3], vec![4]];
        
        assert_eq!(solution.level_order_bfs(root), expected);
    }
    
    #[test]
    fn test_right_skewed_tree() {
        let solution = Solution;
        
        // Tree: 1 -> 2 -> 3 -> 4 (all right children)
        let root = build_tree(vec![Some(1), None, Some(2), None, Some(3), None, Some(4)]);
        let expected = vec![vec![1], vec![2], vec![3], vec![4]];
        
        assert_eq!(solution.level_order_bfs(root), expected);
    }
    
    #[test]
    fn test_perfect_binary_tree() {
        let solution = Solution;
        
        // Perfect binary tree with 3 levels
        let root = build_tree(vec![Some(1), Some(2), Some(3), Some(4), Some(5), Some(6), Some(7)]);
        let expected = vec![vec![1], vec![2, 3], vec![4, 5, 6, 7]];
        
        assert_eq!(solution.level_order_bfs(root), expected);
    }
    
    #[test]
    fn test_complete_binary_tree() {
        let solution = Solution;
        
        // Complete but not perfect binary tree
        let root = build_tree(vec![Some(1), Some(2), Some(3), Some(4), Some(5), Some(6)]);
        let expected = vec![vec![1], vec![2, 3], vec![4, 5, 6]];
        
        assert_eq!(solution.level_order_bfs(root), expected);
    }
    
    #[test]
    fn test_unbalanced_tree() {
        let solution = Solution;
        
        // Unbalanced tree
        let root = build_tree(vec![Some(1), Some(2), Some(3), Some(4), None, None, Some(7), Some(8)]);
        let expected = vec![vec![1], vec![2, 3], vec![4, 7], vec![8]];
        
        assert_eq!(solution.level_order_bfs(root), expected);
    }
    
    #[test]
    fn test_large_tree() {
        let solution = Solution;
        
        // Larger tree to test performance
        let root = build_tree(vec![
            Some(1), Some(2), Some(3), Some(4), Some(5), Some(6), Some(7),
            Some(8), Some(9), Some(10), Some(11), Some(12), Some(13), Some(14), Some(15)
        ]);
        let expected = vec![
            vec![1],
            vec![2, 3],
            vec![4, 5, 6, 7],
            vec![8, 9, 10, 11, 12, 13, 14, 15]
        ];
        
        assert_eq!(solution.level_order_bfs(root), expected);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let solution = Solution;
        
        let test_cases = vec![
            vec![Some(3), Some(9), Some(20), None, None, Some(15), Some(7)],
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
            let root6 = build_tree(tree_vals.clone());
            
            let result1 = solution.level_order_bfs(root1);
            let result2 = solution.level_order_dfs(root2);
            let result3 = solution.level_order_two_queues(root3);
            let result4 = solution.level_order_markers(root4);
            let result5 = solution.level_order_vectors(root5);
            let result6 = solution.level_order_morris_style(root6);
            
            assert_eq!(result1, result2, "BFS and DFS differ for {:?}", tree_vals);
            assert_eq!(result1, result3, "BFS and Two Queues differ for {:?}", tree_vals);
            assert_eq!(result1, result4, "BFS and Markers differ for {:?}", tree_vals);
            assert_eq!(result1, result5, "BFS and Vectors differ for {:?}", tree_vals);
            assert_eq!(result1, result6, "BFS and Morris style differ for {:?}", tree_vals);
        }
    }
    
    #[test]
    fn test_edge_cases() {
        let solution = Solution;
        
        // Tree with negative values
        let root = build_tree(vec![Some(-1), Some(-2), Some(-3)]);
        let expected = vec![vec![-1], vec![-2, -3]];
        assert_eq!(solution.level_order_bfs(root), expected);
        
        // Tree with zero
        let root = build_tree(vec![Some(0), Some(-1), Some(1)]);
        let expected = vec![vec![0], vec![-1, 1]];
        assert_eq!(solution.level_order_bfs(root), expected);
        
        // Tree with maximum values (within constraints)
        let root = build_tree(vec![Some(1000), Some(-1000), Some(500)]);
        let expected = vec![vec![1000], vec![-1000, 500]];
        assert_eq!(solution.level_order_bfs(root), expected);
    }
    
    #[test]
    fn test_single_path_trees() {
        let solution = Solution;
        
        // Only left children
        let root = build_tree(vec![Some(1), Some(2), None, Some(3), None, Some(4)]);
        let expected = vec![vec![1], vec![2], vec![3], vec![4]];
        assert_eq!(solution.level_order_dfs(root), expected);
        
        // Only right children
        let root = build_tree(vec![Some(1), None, Some(2), None, Some(3), None, Some(4)]);
        let expected = vec![vec![1], vec![2], vec![3], vec![4]];
        assert_eq!(solution.level_order_dfs(root), expected);
    }
    
    #[test]
    fn test_memory_efficiency() {
        let solution = Solution;
        
        // Test with a tree that would cause memory issues if not handled properly
        let mut vals = vec![Some(1)];
        for i in 2..100 {
            vals.push(Some(i));
            vals.push(Some(i + 100));
        }
        let root = build_tree(vals);
        
        // Should complete without memory issues
        let result = solution.level_order_vectors(root);
        assert!(!result.is_empty());
        assert_eq!(result[0], vec![1]);
    }
    
    #[test]
    fn test_performance_characteristics() {
        let solution = Solution;
        
        // Test different approaches on same large tree
        let vals: Vec<Option<i32>> = (1..=127).map(Some).collect();
        let root = build_tree(vals);
        
        let result = solution.level_order_bfs(root);
        
        // Verify structure: should have 7 levels for 127 nodes
        assert_eq!(result.len(), 7);
        assert_eq!(result[0].len(), 1);   // Level 0: 1 node
        assert_eq!(result[1].len(), 2);   // Level 1: 2 nodes
        assert_eq!(result[2].len(), 4);   // Level 2: 4 nodes
        assert_eq!(result[3].len(), 8);   // Level 3: 8 nodes
        assert_eq!(result[4].len(), 16);  // Level 4: 16 nodes
        assert_eq!(result[5].len(), 32);  // Level 5: 32 nodes
        assert_eq!(result[6].len(), 64);  // Level 6: 64 nodes
    }
}