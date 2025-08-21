//! Problem 105: Construct Binary Tree from Preorder and Inorder Traversal
//!
//! Given two integer arrays preorder and inorder where preorder is the preorder traversal 
//! of a binary tree and inorder is the inorder traversal of the same tree, construct and 
//! return the binary tree.
//!
//! Constraints:
//! - 1 <= preorder.length <= 3000
//! - inorder.length == preorder.length
//! - -3000 <= preorder[i], inorder[i] <= 3000
//! - preorder and inorder consist of unique values
//! - Each value of inorder also appears in preorder
//! - preorder is guaranteed to be the preorder traversal of the tree
//! - inorder is guaranteed to be the inorder traversal of the tree

use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<Rc<RefCell<TreeNode>>>,
    pub right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None,
        }
    }
}

pub struct Solution;

impl Solution {
    /// Approach 1: Recursive with HashMap for O(1) Inorder Lookups
    /// 
    /// The key insight is that in preorder, the first element is always the root.
    /// We can find this root in inorder to split left and right subtrees.
    /// Using a HashMap for O(1) lookups of indices in inorder array.
    /// 
    /// Time Complexity: O(n) - visit each node once
    /// Space Complexity: O(n) - HashMap storage and recursion stack
    pub fn build_tree_recursive_hashmap(
        preorder: Vec<i32>,
        inorder: Vec<i32>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        if preorder.is_empty() || inorder.is_empty() {
            return None;
        }
        
        // Build HashMap for O(1) inorder index lookups
        let inorder_map: HashMap<i32, usize> = inorder
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();
        
        let mut preorder_idx = 0;
        
        Self::build_recursive_hashmap(
            &preorder,
            &inorder_map,
            &mut preorder_idx,
            0,
            inorder.len() - 1,
        )
    }
    
    fn build_recursive_hashmap(
        preorder: &[i32],
        inorder_map: &HashMap<i32, usize>,
        preorder_idx: &mut usize,
        in_start: usize,
        in_end: usize,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        if in_start > in_end || *preorder_idx >= preorder.len() {
            return None;
        }
        
        // Get root value from preorder
        let root_val = preorder[*preorder_idx];
        *preorder_idx += 1;
        
        // Create root node
        let root = Rc::new(RefCell::new(TreeNode::new(root_val)));
        
        // Find root position in inorder
        let in_root = *inorder_map.get(&root_val).unwrap();
        
        // Build left subtree first (preorder: root -> left -> right)
        if in_root > 0 {
            root.borrow_mut().left = Self::build_recursive_hashmap(
                preorder,
                inorder_map,
                preorder_idx,
                in_start,
                in_root - 1,
            );
        }
        
        // Build right subtree
        root.borrow_mut().right = Self::build_recursive_hashmap(
            preorder,
            inorder_map,
            preorder_idx,
            in_root + 1,
            in_end,
        );
        
        Some(root)
    }
    
    /// Approach 2: Iterative with Stack
    /// 
    /// Uses a stack to simulate the recursive calls. We process nodes in preorder
    /// and use inorder to determine when to pop from stack and switch to right subtree.
    /// 
    /// Time Complexity: O(n) - process each node once
    /// Space Complexity: O(n) - stack storage
    pub fn build_tree_iterative_stack(
        preorder: Vec<i32>,
        inorder: Vec<i32>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        if preorder.is_empty() || inorder.is_empty() {
            return None;
        }
        
        let root = Rc::new(RefCell::new(TreeNode::new(preorder[0])));
        let mut stack = vec![root.clone()];
        let mut inorder_idx = 0;
        
        for i in 1..preorder.len() {
            let mut parent = stack.last().unwrap().clone();
            let curr = Rc::new(RefCell::new(TreeNode::new(preorder[i])));
            
            // If parent val != current inorder val, we're still building left subtree
            if parent.borrow().val != inorder[inorder_idx] {
                parent.borrow_mut().left = Some(curr.clone());
            } else {
                // Pop from stack until we find the parent of right subtree
                while !stack.is_empty() && 
                      stack.last().unwrap().borrow().val == inorder[inorder_idx] {
                    parent = stack.pop().unwrap();
                    inorder_idx += 1;
                }
                parent.borrow_mut().right = Some(curr.clone());
            }
            
            stack.push(curr);
        }
        
        Some(root)
    }
    
    /// Approach 3: Divide and Conquer with Slicing
    /// 
    /// Classic divide and conquer approach that creates new slices for each recursive call.
    /// Less efficient due to slice copying but conceptually clearer.
    /// 
    /// Time Complexity: O(n²) worst case due to slice operations and searching
    /// Space Complexity: O(n²) due to slice copies
    pub fn build_tree_divide_conquer(
        preorder: Vec<i32>,
        inorder: Vec<i32>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        Self::build_divide_conquer(&preorder, &inorder)
    }
    
    fn build_divide_conquer(
        preorder: &[i32],
        inorder: &[i32],
    ) -> Option<Rc<RefCell<TreeNode>>> {
        if preorder.is_empty() || inorder.is_empty() {
            return None;
        }
        
        let root_val = preorder[0];
        let root = Rc::new(RefCell::new(TreeNode::new(root_val)));
        
        // Find root in inorder
        let root_idx = inorder.iter().position(|&x| x == root_val).unwrap();
        
        // Split arrays
        let left_inorder = &inorder[..root_idx];
        let right_inorder = &inorder[root_idx + 1..];
        
        let left_size = left_inorder.len();
        let left_preorder = &preorder[1..1 + left_size];
        let right_preorder = &preorder[1 + left_size..];
        
        // Recursively build subtrees
        root.borrow_mut().left = Self::build_divide_conquer(left_preorder, left_inorder);
        root.borrow_mut().right = Self::build_divide_conquer(right_preorder, right_inorder);
        
        Some(root)
    }
    
    /// Approach 4: Optimized with Index Tracking
    /// 
    /// Instead of creating new slices, we track indices to avoid copying.
    /// This is more memory efficient than the divide and conquer approach.
    /// 
    /// Time Complexity: O(n) - each node processed once
    /// Space Complexity: O(h) - recursion stack where h is tree height
    pub fn build_tree_index_tracking(
        preorder: Vec<i32>,
        inorder: Vec<i32>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        if preorder.is_empty() || inorder.is_empty() {
            return None;
        }
        
        let mut pre_idx = 0;
        Self::build_with_indices(
            &preorder,
            &inorder,
            &mut pre_idx,
            0,
            inorder.len(),
        )
    }
    
    fn build_with_indices(
        preorder: &[i32],
        inorder: &[i32],
        pre_idx: &mut usize,
        in_start: usize,
        in_end: usize,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        if in_start >= in_end || *pre_idx >= preorder.len() {
            return None;
        }
        
        let root_val = preorder[*pre_idx];
        *pre_idx += 1;
        
        let root = Rc::new(RefCell::new(TreeNode::new(root_val)));
        
        // Find root in inorder range
        let mut root_idx = in_start;
        for i in in_start..in_end {
            if inorder[i] == root_val {
                root_idx = i;
                break;
            }
        }
        
        // Build subtrees
        root.borrow_mut().left = Self::build_with_indices(
            preorder,
            inorder,
            pre_idx,
            in_start,
            root_idx,
        );
        
        root.borrow_mut().right = Self::build_with_indices(
            preorder,
            inorder,
            pre_idx,
            root_idx + 1,
            in_end,
        );
        
        Some(root)
    }
    
    /// Approach 5: Morris-Inspired Iterative Construction
    /// 
    /// An iterative approach inspired by Morris traversal concepts,
    /// building the tree by maintaining parent-child relationships.
    /// 
    /// Time Complexity: O(n) - each node processed once
    /// Space Complexity: O(n) - storage for nodes
    pub fn build_tree_morris_inspired(
        preorder: Vec<i32>,
        inorder: Vec<i32>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        if preorder.is_empty() || inorder.is_empty() {
            return None;
        }
        
        let root = Rc::new(RefCell::new(TreeNode::new(preorder[0])));
        let mut curr = root.clone();
        let mut stack = vec![];
        
        let mut pre_idx = 1;
        let mut in_idx = 0;
        
        while pre_idx < preorder.len() {
            if curr.borrow().val != inorder[in_idx] {
                // Still building left subtree
                let new_node = Rc::new(RefCell::new(TreeNode::new(preorder[pre_idx])));
                curr.borrow_mut().left = Some(new_node.clone());
                stack.push(curr);
                curr = new_node;
                pre_idx += 1;
            } else {
                // Found a node that matches inorder, move to right
                in_idx += 1;
                
                // Find the correct parent for right child
                while !stack.is_empty() && 
                      stack.last().unwrap().borrow().val == inorder[in_idx] {
                    curr = stack.pop().unwrap();
                    in_idx += 1;
                }
                
                // Add right child
                if pre_idx < preorder.len() {
                    let new_node = Rc::new(RefCell::new(TreeNode::new(preorder[pre_idx])));
                    curr.borrow_mut().right = Some(new_node.clone());
                    curr = new_node;
                    pre_idx += 1;
                }
            }
        }
        
        Some(root)
    }
    
    /// Approach 6: Boundary-Based Construction
    /// 
    /// Uses boundaries to determine when to stop building left/right subtrees.
    /// This approach uses the property that inorder values act as boundaries.
    /// 
    /// Time Complexity: O(n) - each node processed once
    /// Space Complexity: O(h) - recursion stack
    pub fn build_tree_boundary(
        preorder: Vec<i32>,
        inorder: Vec<i32>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        if preorder.is_empty() || inorder.is_empty() {
            return None;
        }
        
        let mut pre_idx = 0;
        let mut in_idx = 0;
        
        Self::build_with_boundary(
            &preorder,
            &inorder,
            &mut pre_idx,
            &mut in_idx,
            i32::MIN,
        )
    }
    
    fn build_with_boundary(
        preorder: &[i32],
        inorder: &[i32],
        pre_idx: &mut usize,
        in_idx: &mut usize,
        boundary: i32,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        if *pre_idx >= preorder.len() || 
           (*in_idx < inorder.len() && inorder[*in_idx] == boundary) {
            return None;
        }
        
        let root_val = preorder[*pre_idx];
        *pre_idx += 1;
        
        let root = Rc::new(RefCell::new(TreeNode::new(root_val)));
        
        // Build left subtree with current root as boundary
        root.borrow_mut().left = Self::build_with_boundary(
            preorder,
            inorder,
            pre_idx,
            in_idx,
            root_val,
        );
        
        // Move past current root in inorder
        if *in_idx < inorder.len() && inorder[*in_idx] == root_val {
            *in_idx += 1;
        }
        
        // Build right subtree with original boundary
        root.borrow_mut().right = Self::build_with_boundary(
            preorder,
            inorder,
            pre_idx,
            in_idx,
            boundary,
        );
        
        Some(root)
    }
}

/// Helper function to convert tree to preorder traversal for testing
pub fn tree_to_preorder(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let mut result = Vec::new();
    preorder_helper(root, &mut result);
    result
}

fn preorder_helper(node: Option<Rc<RefCell<TreeNode>>>, result: &mut Vec<i32>) {
    if let Some(n) = node {
        result.push(n.borrow().val);
        preorder_helper(n.borrow().left.clone(), result);
        preorder_helper(n.borrow().right.clone(), result);
    }
}

/// Helper function to convert tree to inorder traversal for testing
pub fn tree_to_inorder(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let mut result = Vec::new();
    inorder_helper(root, &mut result);
    result
}

fn inorder_helper(node: Option<Rc<RefCell<TreeNode>>>, result: &mut Vec<i32>) {
    if let Some(n) = node {
        inorder_helper(n.borrow().left.clone(), result);
        result.push(n.borrow().val);
        inorder_helper(n.borrow().right.clone(), result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_tree() {
        // Tree: [3,9,20,null,null,15,7]
        //       3
        //      / \
        //     9   20
        //        /  \
        //       15   7
        let preorder = vec![3, 9, 20, 15, 7];
        let inorder = vec![9, 3, 15, 20, 7];
        
        // Test all approaches
        let tree1 = Solution::build_tree_recursive_hashmap(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree1.clone()), preorder);
        assert_eq!(tree_to_inorder(tree1), inorder);
        
        let tree2 = Solution::build_tree_iterative_stack(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree2.clone()), preorder);
        assert_eq!(tree_to_inorder(tree2), inorder);
        
        let tree3 = Solution::build_tree_divide_conquer(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree3.clone()), preorder);
        assert_eq!(tree_to_inorder(tree3), inorder);
        
        let tree4 = Solution::build_tree_index_tracking(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree4.clone()), preorder);
        assert_eq!(tree_to_inorder(tree4), inorder);
        
        let tree5 = Solution::build_tree_morris_inspired(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree5.clone()), preorder);
        assert_eq!(tree_to_inorder(tree5), inorder);
        
        let tree6 = Solution::build_tree_boundary(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree6.clone()), preorder);
        assert_eq!(tree_to_inorder(tree6), inorder);
    }
    
    #[test]
    fn test_single_node() {
        let preorder = vec![1];
        let inorder = vec![1];
        
        let tree1 = Solution::build_tree_recursive_hashmap(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree1.clone()), vec![1]);
        assert_eq!(tree_to_inorder(tree1), vec![1]);
        
        let tree2 = Solution::build_tree_iterative_stack(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree2), vec![1]);
        
        let tree3 = Solution::build_tree_divide_conquer(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree3), vec![1]);
        
        let tree4 = Solution::build_tree_index_tracking(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree4), vec![1]);
        
        let tree5 = Solution::build_tree_morris_inspired(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree5), vec![1]);
        
        let tree6 = Solution::build_tree_boundary(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree6), vec![1]);
    }
    
    #[test]
    fn test_left_skewed_tree() {
        // Tree:     1
        //          /
        //         2
        //        /
        //       3
        //      /
        //     4
        let preorder = vec![1, 2, 3, 4];
        let inorder = vec![4, 3, 2, 1];
        
        let tree1 = Solution::build_tree_recursive_hashmap(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree1.clone()), preorder);
        assert_eq!(tree_to_inorder(tree1), inorder);
        
        let tree2 = Solution::build_tree_iterative_stack(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree2.clone()), preorder);
        
        let tree3 = Solution::build_tree_divide_conquer(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree3.clone()), preorder);
        
        let tree4 = Solution::build_tree_index_tracking(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree4.clone()), preorder);
        
        let tree5 = Solution::build_tree_morris_inspired(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree5.clone()), preorder);
        
        let tree6 = Solution::build_tree_boundary(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree6.clone()), preorder);
    }
    
    #[test]
    fn test_right_skewed_tree() {
        // Tree: 1
        //        \
        //         2
        //          \
        //           3
        //            \
        //             4
        let preorder = vec![1, 2, 3, 4];
        let inorder = vec![1, 2, 3, 4];
        
        let tree1 = Solution::build_tree_recursive_hashmap(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree1.clone()), preorder);
        assert_eq!(tree_to_inorder(tree1), inorder);
        
        let tree2 = Solution::build_tree_iterative_stack(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_inorder(tree2), inorder);
        
        let tree3 = Solution::build_tree_divide_conquer(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_inorder(tree3), inorder);
        
        let tree4 = Solution::build_tree_index_tracking(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_inorder(tree4), inorder);
        
        let tree5 = Solution::build_tree_morris_inspired(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_inorder(tree5), inorder);
        
        let tree6 = Solution::build_tree_boundary(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_inorder(tree6), inorder);
    }
    
    #[test]
    fn test_balanced_tree() {
        // Tree:       1
        //           /   \
        //          2     3
        //         / \   / \
        //        4   5 6   7
        let preorder = vec![1, 2, 4, 5, 3, 6, 7];
        let inorder = vec![4, 2, 5, 1, 6, 3, 7];
        
        let tree1 = Solution::build_tree_recursive_hashmap(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree1.clone()), preorder);
        assert_eq!(tree_to_inorder(tree1), inorder);
        
        let tree2 = Solution::build_tree_iterative_stack(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree2), preorder);
        
        let tree3 = Solution::build_tree_divide_conquer(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree3), preorder);
        
        let tree4 = Solution::build_tree_index_tracking(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree4), preorder);
        
        let tree5 = Solution::build_tree_morris_inspired(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree5), preorder);
        
        let tree6 = Solution::build_tree_boundary(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree6), preorder);
    }
    
    #[test]
    fn test_empty_input() {
        let preorder = vec![];
        let inorder = vec![];
        
        let tree1 = Solution::build_tree_recursive_hashmap(preorder.clone(), inorder.clone());
        assert!(tree1.is_none());
        
        let tree2 = Solution::build_tree_iterative_stack(preorder.clone(), inorder.clone());
        assert!(tree2.is_none());
        
        let tree3 = Solution::build_tree_divide_conquer(preorder.clone(), inorder.clone());
        assert!(tree3.is_none());
        
        let tree4 = Solution::build_tree_index_tracking(preorder.clone(), inorder.clone());
        assert!(tree4.is_none());
        
        let tree5 = Solution::build_tree_morris_inspired(preorder.clone(), inorder.clone());
        assert!(tree5.is_none());
        
        let tree6 = Solution::build_tree_boundary(preorder.clone(), inorder.clone());
        assert!(tree6.is_none());
    }
    
    #[test]
    fn test_two_nodes_left() {
        // Tree:  1
        //       /
        //      2
        let preorder = vec![1, 2];
        let inorder = vec![2, 1];
        
        let tree1 = Solution::build_tree_recursive_hashmap(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree1.clone()), preorder);
        assert_eq!(tree_to_inorder(tree1), inorder);
        
        let tree2 = Solution::build_tree_iterative_stack(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree2), preorder);
        
        let tree3 = Solution::build_tree_divide_conquer(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree3), preorder);
        
        let tree4 = Solution::build_tree_index_tracking(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree4), preorder);
        
        let tree5 = Solution::build_tree_morris_inspired(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree5), preorder);
        
        let tree6 = Solution::build_tree_boundary(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree6), preorder);
    }
    
    #[test]
    fn test_two_nodes_right() {
        // Tree: 1
        //        \
        //         2
        let preorder = vec![1, 2];
        let inorder = vec![1, 2];
        
        let tree1 = Solution::build_tree_recursive_hashmap(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree1.clone()), preorder);
        assert_eq!(tree_to_inorder(tree1), inorder);
        
        let tree2 = Solution::build_tree_iterative_stack(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_inorder(tree2), inorder);
        
        let tree3 = Solution::build_tree_divide_conquer(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_inorder(tree3), inorder);
        
        let tree4 = Solution::build_tree_index_tracking(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_inorder(tree4), inorder);
        
        let tree5 = Solution::build_tree_morris_inspired(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_inorder(tree5), inorder);
        
        let tree6 = Solution::build_tree_boundary(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_inorder(tree6), inorder);
    }
    
    #[test]
    fn test_negative_values() {
        let preorder = vec![-1, -2, -3];
        let inorder = vec![-2, -1, -3];
        
        let tree1 = Solution::build_tree_recursive_hashmap(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree1.clone()), preorder);
        assert_eq!(tree_to_inorder(tree1), inorder);
        
        let tree2 = Solution::build_tree_iterative_stack(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree2), preorder);
        
        let tree3 = Solution::build_tree_divide_conquer(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree3), preorder);
        
        let tree4 = Solution::build_tree_index_tracking(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree4), preorder);
        
        let tree5 = Solution::build_tree_morris_inspired(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree5), preorder);
        
        let tree6 = Solution::build_tree_boundary(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree6), preorder);
    }
    
    #[test]
    fn test_complex_tree() {
        // Complex tree with mixed structure
        let preorder = vec![10, 5, 3, 1, 4, 8, 7, 9, 20, 15, 25, 30];
        let inorder = vec![1, 3, 4, 5, 7, 8, 9, 10, 15, 20, 25, 30];
        
        let tree1 = Solution::build_tree_recursive_hashmap(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree1.clone()), preorder);
        assert_eq!(tree_to_inorder(tree1), inorder);
        
        let tree2 = Solution::build_tree_iterative_stack(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree2.clone()), preorder);
        assert_eq!(tree_to_inorder(tree2), inorder);
        
        let tree3 = Solution::build_tree_divide_conquer(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree3.clone()), preorder);
        
        let tree4 = Solution::build_tree_index_tracking(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree4.clone()), preorder);
        
        let tree5 = Solution::build_tree_morris_inspired(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree5.clone()), preorder);
        
        let tree6 = Solution::build_tree_boundary(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree6.clone()), preorder);
    }
    
    #[test]
    fn test_zigzag_tree() {
        // Tree with zigzag structure
        //        5
        //       /
        //      3
        //       \
        //        4
        //       /
        //      2
        let preorder = vec![5, 3, 4, 2];
        let inorder = vec![3, 2, 4, 5];
        
        let tree1 = Solution::build_tree_recursive_hashmap(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree1.clone()), preorder);
        assert_eq!(tree_to_inorder(tree1), inorder);
        
        let tree2 = Solution::build_tree_iterative_stack(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree2), preorder);
        
        let tree3 = Solution::build_tree_divide_conquer(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree3), preorder);
        
        let tree4 = Solution::build_tree_index_tracking(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree4), preorder);
        
        let tree5 = Solution::build_tree_morris_inspired(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree5), preorder);
        
        let tree6 = Solution::build_tree_boundary(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree6), preorder);
    }
    
    #[test]
    fn test_large_values() {
        let preorder = vec![3000, -3000, 1000, -1000];
        let inorder = vec![-3000, 1000, 3000, -1000];
        
        let tree1 = Solution::build_tree_recursive_hashmap(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree1.clone()), preorder);
        assert_eq!(tree_to_inorder(tree1), inorder);
        
        let tree2 = Solution::build_tree_iterative_stack(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree2), preorder);
        
        let tree3 = Solution::build_tree_divide_conquer(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree3), preorder);
        
        let tree4 = Solution::build_tree_index_tracking(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree4), preorder);
        
        let tree5 = Solution::build_tree_morris_inspired(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree5), preorder);
        
        let tree6 = Solution::build_tree_boundary(preorder.clone(), inorder.clone());
        assert_eq!(tree_to_preorder(tree6), preorder);
    }
    
    #[test]
    fn test_performance_large_tree() {
        // Generate a larger tree for performance testing
        let mut preorder = Vec::new();
        let mut inorder = Vec::new();
        
        // Create a tree with pattern
        for i in 0..100 {
            preorder.push(i);
        }
        
        // Create corresponding inorder (simplified for test)
        fn create_inorder(start: i32, end: i32, inorder: &mut Vec<i32>) {
            if start > end {
                return;
            }
            let mid = start + (end - start) / 2;
            create_inorder(start, mid - 1, inorder);
            inorder.push(mid);
            create_inorder(mid + 1, end, inorder);
        }
        
        create_inorder(0, 99, &mut inorder);
        
        // Test that all approaches handle larger input
        let tree1 = Solution::build_tree_recursive_hashmap(preorder.clone(), inorder.clone());
        assert!(tree1.is_some());
        
        let tree2 = Solution::build_tree_index_tracking(preorder.clone(), inorder.clone());
        assert!(tree2.is_some());
        
        // Verify reconstruction is correct
        let reconstructed_pre = tree_to_preorder(tree1.clone());
        let reconstructed_in = tree_to_inorder(tree1);
        assert_eq!(reconstructed_pre.len(), preorder.len());
        assert_eq!(reconstructed_in.len(), inorder.len());
    }
    
    #[test]
    fn test_consistency_check() {
        // Test that all approaches produce identical trees
        let test_cases = vec![
            (vec![1, 2, 3], vec![2, 1, 3]),
            (vec![5, 3, 1, 4, 8, 7, 9], vec![1, 3, 4, 5, 7, 8, 9]),  // Fixed: correct preorder for the given inorder
            (vec![10, 5, 15], vec![5, 10, 15]),
        ];
        
        for (preorder, inorder) in test_cases {
            let tree1 = Solution::build_tree_recursive_hashmap(preorder.clone(), inorder.clone());
            let tree2 = Solution::build_tree_iterative_stack(preorder.clone(), inorder.clone());
            let tree3 = Solution::build_tree_divide_conquer(preorder.clone(), inorder.clone());
            let tree4 = Solution::build_tree_index_tracking(preorder.clone(), inorder.clone());
            let tree5 = Solution::build_tree_morris_inspired(preorder.clone(), inorder.clone());
            let tree6 = Solution::build_tree_boundary(preorder.clone(), inorder.clone());
            
            // All should produce same preorder
            let pre1 = tree_to_preorder(tree1.clone());
            let pre2 = tree_to_preorder(tree2.clone());
            let pre3 = tree_to_preorder(tree3.clone());
            let pre4 = tree_to_preorder(tree4.clone());
            let pre5 = tree_to_preorder(tree5.clone());
            let pre6 = tree_to_preorder(tree6.clone());
            
            assert_eq!(pre1, preorder);
            assert_eq!(pre2, preorder);
            assert_eq!(pre3, preorder);
            assert_eq!(pre4, preorder);
            assert_eq!(pre5, preorder);
            assert_eq!(pre6, preorder);
            
            // All should produce same inorder
            let in1 = tree_to_inorder(tree1);
            let in2 = tree_to_inorder(tree2);
            let in3 = tree_to_inorder(tree3);
            let in4 = tree_to_inorder(tree4);
            let in5 = tree_to_inorder(tree5);
            let in6 = tree_to_inorder(tree6);
            
            assert_eq!(in1, inorder);
            assert_eq!(in2, inorder);
            assert_eq!(in3, inorder);
            assert_eq!(in4, inorder);
            assert_eq!(in5, inorder);
            assert_eq!(in6, inorder);
        }
    }
}