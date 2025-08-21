//! Problem 98: Validate Binary Search Tree
//! 
//! Given the root of a binary tree, determine if it is a valid binary search tree (BST).
//! 
//! A valid BST is defined as follows:
//! - The left subtree of a node contains only nodes with keys less than the node's key.
//! - The right subtree of a node contains only nodes with keys greater than the node's key.
//! - Both the left and right subtrees must also be binary search trees.
//! 
//! Example 1:
//!     2
//!    / \
//!   1   3
//! Input: root = [2,1,3]
//! Output: true
//! 
//! Example 2:
//!     5
//!    / \
//!   1   4
//!      / \
//!     3   6
//! Input: root = [5,1,4,null,null,3,6]
//! Output: false
//! Explanation: The root node's value is 5 but its right child's value is 4.
//! 
//! Constraints:
//! - The number of nodes in the tree is in the range [1, 10^4].
//! - -2^31 <= Node.val <= 2^31 - 1

use std::rc::Rc;
use std::cell::RefCell;
use crate::utils::data_structures::TreeNode;

pub struct Solution;

impl Solution {
    /// Approach 1: Recursive Range Validation
    /// 
    /// Recursively validates each node by maintaining valid range bounds.
    /// Each node must be within its valid range based on its ancestors.
    /// 
    /// Time Complexity: O(n) where n is number of nodes
    /// Space Complexity: O(h) where h is height of tree (recursion stack)
    pub fn is_valid_bst_recursive(&self, root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        self.validate_range(root.as_ref(), None, None)
    }
    
    fn validate_range(&self, node: Option<&Rc<RefCell<TreeNode>>>, min: Option<i32>, max: Option<i32>) -> bool {
        match node {
            None => true,
            Some(node) => {
                let val = node.borrow().val;
                
                // Check if current node violates bounds
                if let Some(min_val) = min {
                    if val <= min_val {
                        return false;
                    }
                }
                if let Some(max_val) = max {
                    if val >= max_val {
                        return false;
                    }
                }
                
                // Recursively validate children with updated bounds
                self.validate_range(node.borrow().left.as_ref(), min, Some(val)) &&
                self.validate_range(node.borrow().right.as_ref(), Some(val), max)
            }
        }
    }
    
    /// Approach 2: Inorder Traversal
    /// 
    /// Performs inorder traversal and checks if values are in strictly increasing order.
    /// BST property ensures inorder traversal produces sorted sequence.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(h) for recursion stack
    pub fn is_valid_bst_inorder(&self, root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        let mut prev = None;
        self.inorder_check(root.as_ref(), &mut prev)
    }
    
    fn inorder_check(&self, node: Option<&Rc<RefCell<TreeNode>>>, prev: &mut Option<i32>) -> bool {
        match node {
            None => true,
            Some(node) => {
                let node_borrowed = node.borrow();
                
                // Check left subtree
                if !self.inorder_check(node_borrowed.left.as_ref(), prev) {
                    return false;
                }
                
                // Check current node
                if let Some(prev_val) = *prev {
                    if node_borrowed.val <= prev_val {
                        return false;
                    }
                }
                *prev = Some(node_borrowed.val);
                
                // Check right subtree
                self.inorder_check(node_borrowed.right.as_ref(), prev)
            }
        }
    }
    
    /// Approach 3: Iterative Inorder with Stack
    /// 
    /// Uses explicit stack to perform iterative inorder traversal.
    /// Eliminates recursion overhead while maintaining same logic.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(h) for stack
    pub fn is_valid_bst_iterative(&self, root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        let mut stack = Vec::new();
        let mut current = root;
        let mut prev = None;
        
        while current.is_some() || !stack.is_empty() {
            // Go to leftmost node
            while let Some(node) = current {
                stack.push(node.clone());
                current = node.borrow().left.clone();
            }
            
            // Process current node
            if let Some(node) = stack.pop() {
                let val = node.borrow().val;
                
                if let Some(prev_val) = prev {
                    if val <= prev_val {
                        return false;
                    }
                }
                prev = Some(val);
                
                current = node.borrow().right.clone();
            }
        }
        
        true
    }
    
    /// Approach 4: Morris Traversal
    /// 
    /// Performs inorder traversal using Morris algorithm with O(1) space.
    /// Temporarily modifies tree structure to eliminate recursion and stack.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn is_valid_bst_morris(&self, root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        let mut current = root;
        let mut prev = None;
        
        while let Some(node) = current.take() {
            let node_val = node.borrow().val;
            let left = node.borrow().left.clone();
            
            if left.is_none() {
                // No left subtree, process current node
                if let Some(prev_val) = prev {
                    if node_val <= prev_val {
                        return false;
                    }
                }
                prev = Some(node_val);
                current = node.borrow().right.clone();
            } else {
                // Find inorder predecessor
                let mut predecessor = left.clone();
                loop {
                    let pred_ref = predecessor.as_ref().unwrap();
                    let pred_right = pred_ref.borrow().right.clone();
                    if pred_right.is_none() || Rc::ptr_eq(pred_right.as_ref().unwrap(), &node) {
                        break;
                    }
                    predecessor = pred_right;
                }
                
                if let Some(pred) = predecessor {
                    let pred_right = pred.borrow().right.clone();
                    if pred_right.is_none() {
                        // Create thread
                        pred.borrow_mut().right = Some(node.clone());
                        current = left;
                    } else {
                        // Remove thread and process current node
                        pred.borrow_mut().right = None;
                        if let Some(prev_val) = prev {
                            if node_val <= prev_val {
                                return false;
                            }
                        }
                        prev = Some(node_val);
                        current = node.borrow().right.clone();
                    }
                }
            }
        }
        
        true
    }
    
    /// Approach 5: Postorder with Min/Max Tracking
    /// 
    /// Uses postorder traversal to compute min/max values for each subtree.
    /// Validates BST property by checking subtree bounds against parent.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(h)
    pub fn is_valid_bst_postorder(&self, root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        self.postorder_validate(root.as_ref()).is_some()
    }
    
    fn postorder_validate(&self, node: Option<&Rc<RefCell<TreeNode>>>) -> Option<(i32, i32)> {
        match node {
            None => Some((i32::MAX, i32::MIN)), // Empty subtree is valid
            Some(node) => {
                let node_borrowed = node.borrow();
                let val = node_borrowed.val;
                
                // Get bounds from left subtree
                let left_bounds = self.postorder_validate(node_borrowed.left.as_ref())?;
                // Get bounds from right subtree
                let right_bounds = self.postorder_validate(node_borrowed.right.as_ref())?;
                
                // Check BST property
                let has_left = node_borrowed.left.is_some();
                let has_right = node_borrowed.right.is_some();
                
                if has_left && left_bounds.1 >= val {
                    return None; // Left subtree max >= current node
                }
                if has_right && right_bounds.0 <= val {
                    return None; // Right subtree min <= current node
                }
                
                // Compute new bounds
                let min_val = if has_left { left_bounds.0 } else { val };
                let max_val = if has_right { right_bounds.1 } else { val };
                
                Some((min_val, max_val))
            }
        }
    }
    
    /// Approach 6: BFS Level Order Validation
    /// 
    /// Uses BFS to validate each node with its accumulated constraints.
    /// Maintains queue of (node, min_bound, max_bound) tuples.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(w) where w is maximum width of tree
    pub fn is_valid_bst_bfs(&self, root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        use std::collections::VecDeque;
        
        if root.is_none() {
            return true;
        }
        
        let mut queue = VecDeque::new();
        queue.push_back((root.unwrap(), None, None));
        
        while let Some((node, min_bound, max_bound)) = queue.pop_front() {
            let node_borrowed = node.borrow();
            let val = node_borrowed.val;
            
            // Check bounds
            if let Some(min_val) = min_bound {
                if val <= min_val {
                    return false;
                }
            }
            if let Some(max_val) = max_bound {
                if val >= max_val {
                    return false;
                }
            }
            
            // Add children to queue with updated bounds
            if let Some(left) = &node_borrowed.left {
                queue.push_back((left.clone(), min_bound, Some(val)));
            }
            if let Some(right) = &node_borrowed.right {
                queue.push_back((right.clone(), Some(val), max_bound));
            }
        }
        
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;
    use std::cell::RefCell;
    use std::collections::VecDeque;
    
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
    fn test_example_1_valid() {
        let solution = Solution;
        
        // Example 1: [2,1,3] -> true
        let root = build_tree(vec![Some(2), Some(1), Some(3)]);
        assert_eq!(solution.is_valid_bst_recursive(root), true);
    }
    
    #[test]
    fn test_example_2_invalid() {
        let solution = Solution;
        
        // Example 2: [5,1,4,null,null,3,6] -> false
        let root = build_tree(vec![Some(5), Some(1), Some(4), None, None, Some(3), Some(6)]);
        assert_eq!(solution.is_valid_bst_recursive(root), false);
    }
    
    #[test]
    fn test_single_node() {
        let solution = Solution;
        
        let root = build_tree(vec![Some(1)]);
        assert_eq!(solution.is_valid_bst_recursive(root), true);
    }
    
    #[test]
    fn test_empty_tree() {
        let solution = Solution;
        
        let root = None;
        assert_eq!(solution.is_valid_bst_recursive(root), true);
    }
    
    #[test]
    fn test_inorder_approach() {
        let solution = Solution;
        
        let root = build_tree(vec![Some(2), Some(1), Some(3)]);
        assert_eq!(solution.is_valid_bst_inorder(root), true);
        
        let root = build_tree(vec![Some(5), Some(1), Some(4), None, None, Some(3), Some(6)]);
        assert_eq!(solution.is_valid_bst_inorder(root), false);
    }
    
    #[test]
    fn test_iterative_approach() {
        let solution = Solution;
        
        let root = build_tree(vec![Some(2), Some(1), Some(3)]);
        assert_eq!(solution.is_valid_bst_iterative(root), true);
        
        let root = build_tree(vec![Some(5), Some(1), Some(4), None, None, Some(3), Some(6)]);
        assert_eq!(solution.is_valid_bst_iterative(root), false);
    }
    
    #[test]
    fn test_morris_approach() {
        let solution = Solution;
        
        let root = build_tree(vec![Some(2), Some(1), Some(3)]);
        assert_eq!(solution.is_valid_bst_morris(root), true);
        
        let root = build_tree(vec![Some(5), Some(1), Some(4), None, None, Some(3), Some(6)]);
        assert_eq!(solution.is_valid_bst_morris(root), false);
    }
    
    #[test]
    fn test_postorder_approach() {
        let solution = Solution;
        
        let root = build_tree(vec![Some(2), Some(1), Some(3)]);
        assert_eq!(solution.is_valid_bst_postorder(root), true);
        
        let root = build_tree(vec![Some(5), Some(1), Some(4), None, None, Some(3), Some(6)]);
        assert_eq!(solution.is_valid_bst_postorder(root), false);
    }
    
    #[test]
    fn test_bfs_approach() {
        let solution = Solution;
        
        let root = build_tree(vec![Some(2), Some(1), Some(3)]);
        assert_eq!(solution.is_valid_bst_bfs(root), true);
        
        let root = build_tree(vec![Some(5), Some(1), Some(4), None, None, Some(3), Some(6)]);
        assert_eq!(solution.is_valid_bst_bfs(root), false);
    }
    
    #[test]
    fn test_left_skewed_valid() {
        let solution = Solution;
        
        // Valid left-skewed tree: 3 -> 2 -> 1
        let root = build_tree(vec![Some(3), Some(2), None, Some(1)]);
        assert_eq!(solution.is_valid_bst_recursive(root), true);
    }
    
    #[test]
    fn test_right_skewed_valid() {
        let solution = Solution;
        
        // Valid right-skewed tree: 1 -> 2 -> 3
        let root = build_tree(vec![Some(1), None, Some(2), None, Some(3)]);
        assert_eq!(solution.is_valid_bst_recursive(root), true);
    }
    
    #[test]
    fn test_duplicate_values() {
        let solution = Solution;
        
        // BST cannot have duplicate values
        let root = build_tree(vec![Some(2), Some(2), Some(2)]);
        assert_eq!(solution.is_valid_bst_recursive(root), false);
        
        let root = build_tree(vec![Some(5), Some(1), Some(5)]);
        assert_eq!(solution.is_valid_bst_recursive(root), false);
    }
    
    #[test]
    fn test_subtle_violations() {
        let solution = Solution;
        
        // Subtle violation: left subtree has node greater than root
        let root = build_tree(vec![Some(10), Some(5), Some(15), None, None, Some(6), Some(20)]);
        assert_eq!(solution.is_valid_bst_recursive(root), false);
        
        // Another subtle violation
        let root = build_tree(vec![Some(10), Some(5), Some(15), Some(3), Some(12), None, None]);
        assert_eq!(solution.is_valid_bst_recursive(root), false);
    }
    
    #[test]
    fn test_valid_complex_tree() {
        let solution = Solution;
        
        // Valid complex BST
        let root = build_tree(vec![
            Some(8), Some(4), Some(12), Some(2), Some(6), Some(10), Some(14),
            Some(1), Some(3), Some(5), Some(7), Some(9), Some(11), Some(13), Some(15)
        ]);
        assert_eq!(solution.is_valid_bst_recursive(root), true);
    }
    
    #[test]
    fn test_boundary_values() {
        let solution = Solution;
        
        // Test with extreme values
        let root = build_tree(vec![Some(i32::MIN), None, Some(i32::MAX)]);
        assert_eq!(solution.is_valid_bst_recursive(root), true);
        
        let root = build_tree(vec![Some(0), Some(i32::MIN), Some(i32::MAX)]);
        assert_eq!(solution.is_valid_bst_recursive(root), true);
    }
    
    #[test]
    fn test_negative_values() {
        let solution = Solution;
        
        // Valid BST with negative values
        let root = build_tree(vec![Some(0), Some(-1), Some(1)]);
        assert_eq!(solution.is_valid_bst_recursive(root), true);
        
        let root = build_tree(vec![Some(-10), Some(-20), Some(-5)]);
        assert_eq!(solution.is_valid_bst_recursive(root), true);
        
        // Invalid BST with negative values
        let root = build_tree(vec![Some(-5), Some(-3), Some(-10)]);
        assert_eq!(solution.is_valid_bst_recursive(root), false);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let solution = Solution;
        
        let test_cases = vec![
            vec![Some(2), Some(1), Some(3)],
            vec![Some(5), Some(1), Some(4), None, None, Some(3), Some(6)],
            vec![Some(1)],
            vec![Some(10), Some(5), Some(15), None, None, Some(6), Some(20)],
            vec![Some(8), Some(4), Some(12), Some(2), Some(6), Some(10), Some(14)],
            vec![Some(-10), Some(-20), Some(-5)],
            vec![Some(i32::MIN), None, Some(i32::MAX)],
        ];
        
        for tree_vals in test_cases {
            let root1 = build_tree(tree_vals.clone());
            let root2 = build_tree(tree_vals.clone());
            let root3 = build_tree(tree_vals.clone());
            let root4 = build_tree(tree_vals.clone());
            let root5 = build_tree(tree_vals.clone());
            let root6 = build_tree(tree_vals.clone());
            
            let result1 = solution.is_valid_bst_recursive(root1);
            let result2 = solution.is_valid_bst_inorder(root2);
            let result3 = solution.is_valid_bst_iterative(root3);
            let result4 = solution.is_valid_bst_morris(root4);
            let result5 = solution.is_valid_bst_postorder(root5);
            let result6 = solution.is_valid_bst_bfs(root6);
            
            assert_eq!(result1, result2, "Recursive and Inorder differ for {:?}", tree_vals);
            assert_eq!(result1, result3, "Recursive and Iterative differ for {:?}", tree_vals);
            assert_eq!(result1, result4, "Recursive and Morris differ for {:?}", tree_vals);
            assert_eq!(result1, result5, "Recursive and Postorder differ for {:?}", tree_vals);
            assert_eq!(result1, result6, "Recursive and BFS differ for {:?}", tree_vals);
        }
    }
    
    #[test]
    fn test_edge_case_structures() {
        let solution = Solution;
        
        // Only left children
        let root = build_tree(vec![Some(5), Some(4), None, Some(3), None, Some(2)]);
        assert_eq!(solution.is_valid_bst_recursive(root), true);
        
        // Only right children
        let root = build_tree(vec![Some(1), None, Some(2), None, Some(3), None, Some(4)]);
        assert_eq!(solution.is_valid_bst_recursive(root), true);
        
        // Zigzag pattern (valid)
        let root = build_tree(vec![Some(4), Some(2), Some(6), Some(1), Some(3), Some(5), Some(7)]);
        assert_eq!(solution.is_valid_bst_recursive(root), true);
    }
    
    #[test]
    fn test_performance_with_large_tree() {
        let solution = Solution;
        
        // Create a large valid BST manually
        fn build_valid_bst(values: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
            if values.is_empty() {
                return None;
            }
            
            fn build_helper(values: &[i32]) -> Option<Rc<RefCell<TreeNode>>> {
                if values.is_empty() {
                    return None;
                }
                
                let mid = values.len() / 2;
                let node = Rc::new(RefCell::new(TreeNode::new(values[mid])));
                
                node.borrow_mut().left = build_helper(&values[..mid]);
                node.borrow_mut().right = build_helper(&values[mid + 1..]);
                
                Some(node)
            }
            
            build_helper(&values)
        }
        
        let values: Vec<i32> = (1..=31).collect(); // Smaller but still meaningful test
        let root = build_valid_bst(values);
        
        // All approaches should agree it's valid
        assert_eq!(solution.is_valid_bst_recursive(root.clone()), true);
        assert_eq!(solution.is_valid_bst_iterative(root.clone()), true);
        assert_eq!(solution.is_valid_bst_bfs(root), true);
    }
    
    #[test]
    fn test_tricky_cases() {
        let solution = Solution;
        
        // Case where simple comparison fails but BST property holds
        let root = build_tree(vec![Some(2147483647)]);
        assert_eq!(solution.is_valid_bst_recursive(root), true);
        
        // Case with one negative and one positive
        let root = build_tree(vec![Some(1), Some(-1), Some(2)]);
        assert_eq!(solution.is_valid_bst_recursive(root), true);
    }
    
    #[test]
    fn test_morris_tree_restoration() {
        let solution = Solution;
        
        // Test that Morris traversal properly restores tree structure
        let root = build_tree(vec![Some(4), Some(2), Some(6), Some(1), Some(3), Some(5), Some(7)]);
        let original_structure = format!("{:?}", root);
        
        let is_valid = solution.is_valid_bst_morris(root.clone());
        let final_structure = format!("{:?}", root);
        
        assert_eq!(is_valid, true);
        assert_eq!(original_structure, final_structure, "Morris traversal should restore tree");
    }
    
    #[test]
    fn test_memory_efficiency() {
        let solution = Solution;
        
        // Create tree that could cause stack overflow with deep recursion
        let mut root = Some(Rc::new(RefCell::new(TreeNode::new(50))));
        let mut current = root.clone();
        
        // Build a deep left-skewed tree
        for i in (1..50).rev() {
            if let Some(node) = current {
                let new_node = Rc::new(RefCell::new(TreeNode::new(i)));
                node.borrow_mut().left = Some(new_node.clone());
                current = Some(new_node);
            }
        }
        
        // Morris traversal should handle this efficiently
        assert_eq!(solution.is_valid_bst_morris(root), true);
    }
}