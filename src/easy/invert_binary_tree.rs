//! # Problem 226: Invert Binary Tree
//!
//! Given the root of a binary tree, invert the tree, and return its root.
//!
//! ## Examples
//!
//! ```text
//! Input: root = [4,2,7,1,3,6,9]
//! Output: [4,7,2,9,6,3,1]
//! ```
//!
//! ```text
//! Input: root = [2,1,3]
//! Output: [2,3,1]
//! ```
//!
//! ```text
//! Input: root = []
//! Output: []
//! ```
//!
//! ## Constraints
//!
//! * The number of nodes in the tree is in the range [0, 100]
//! * -100 <= Node.val <= 100

use crate::utils::data_structures::TreeNode;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::VecDeque;

/// Solution for Invert Binary Tree problem
pub struct Solution;

impl Solution {
    /// Creates a new instance of Solution
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Recursive DFS (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Base case: if node is null, return null
    /// 2. Recursively invert left and right subtrees
    /// 3. Swap the inverted left and right children
    /// 4. Return the current node
    /// 
    /// **Time Complexity:** O(n) - Visit each node once
    /// **Space Complexity:** O(h) - Recursion stack, h = height of tree
    /// 
    /// **Key Insights:**
    /// - Inversion means swapping left and right children at every node
    /// - Post-order processing: invert children first, then swap
    /// - Can also be done pre-order: swap first, then invert
    /// 
    /// **Why this works:**
    /// - Tree inversion is naturally recursive
    /// - Each subtree needs to be inverted independently
    /// - Then we swap the subtrees at current node
    /// 
    /// **Visualization:**
    /// ```text
    /// Original:     4           Inverted:     4
    ///              / \                       / \
    ///             2   7         =>          7   2
    ///            / \ / \                   / \ / \
    ///           1 3 6 9                   9 6 3 1
    /// ```
    pub fn invert_tree(&self, root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        if let Some(node) = root {
            let left = self.invert_tree(node.borrow().left.clone());
            let right = self.invert_tree(node.borrow().right.clone());
            
            // Swap the children
            node.borrow_mut().left = right;
            node.borrow_mut().right = left;
            
            Some(node)
        } else {
            None
        }
    }

    /// # Approach 2: Iterative DFS with Stack
    /// 
    /// **Algorithm:**
    /// 1. Use stack to simulate recursion
    /// 2. Push root to stack
    /// 3. While stack not empty:
    ///    - Pop node and swap its children
    ///    - Push non-null children to stack
    /// 4. Return original root
    /// 
    /// **Time Complexity:** O(n) - Visit each node once
    /// **Space Complexity:** O(h) - Stack size in worst case
    /// 
    /// **Advantages:**
    /// - Avoids recursion stack overflow
    /// - More explicit control over traversal
    /// - Easier to debug step by step
    /// 
    /// **When to use:** Very deep trees or when avoiding recursion
    pub fn invert_tree_iterative(&self, root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        if root.is_none() {
            return None;
        }
        
        let mut stack = Vec::new();
        stack.push(root.clone());
        
        while let Some(Some(node)) = stack.pop() {
            // Swap children
            let left = node.borrow().left.clone();
            let right = node.borrow().right.clone();
            
            node.borrow_mut().left = right.clone();
            node.borrow_mut().right = left.clone();
            
            // Add children to stack for processing
            if left.is_some() {
                stack.push(left);
            }
            if right.is_some() {
                stack.push(right);
            }
        }
        
        root
    }

    /// # Approach 3: Level-Order Traversal (BFS)
    /// 
    /// **Algorithm:**
    /// 1. Use queue for level-by-level processing
    /// 2. For each node in current level:
    ///    - Swap its left and right children
    ///    - Add children to queue for next level
    /// 3. Continue until queue is empty
    /// 
    /// **Time Complexity:** O(n) - Visit each node once
    /// **Space Complexity:** O(w) - Queue size, w = maximum width
    /// 
    /// **Characteristics:**
    /// - Processes tree level by level
    /// - Natural for problems requiring level-wise operations
    /// - Good memory locality for wide trees
    /// 
    /// **When to use:** When level-order processing is preferred
    pub fn invert_tree_bfs(&self, root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        if root.is_none() {
            return None;
        }
        
        let mut queue = VecDeque::new();
        queue.push_back(root.clone());
        
        while let Some(Some(node)) = queue.pop_front() {
            // Swap children
            let left = node.borrow().left.clone();
            let right = node.borrow().right.clone();
            
            node.borrow_mut().left = right.clone();
            node.borrow_mut().right = left.clone();
            
            // Add children to queue
            if left.is_some() {
                queue.push_back(left);
            }
            if right.is_some() {
                queue.push_back(right);
            }
        }
        
        root
    }

    /// # Approach 4: In-Place Swap with Preorder
    /// 
    /// **Algorithm:**
    /// 1. Visit node first (preorder)
    /// 2. Swap left and right children immediately
    /// 3. Recursively process swapped children
    /// 
    /// **Time Complexity:** O(n) - Visit each node once
    /// **Space Complexity:** O(h) - Recursion stack
    /// 
    /// **Difference from Approach 1:**
    /// - Swaps before recursive calls (preorder)
    /// - Approach 1 swaps after recursive calls (postorder)
    /// - Both achieve same result
    /// 
    /// **Educational value:**
    /// - Shows that swap timing doesn't matter for this problem
    /// - Demonstrates preorder vs postorder processing
    pub fn invert_tree_preorder(&self, root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        if let Some(node) = root {
            // Swap children first (preorder)
            let left = node.borrow().left.clone();
            let right = node.borrow().right.clone();
            
            node.borrow_mut().left = right;
            node.borrow_mut().right = left;
            
            // Then recurse on swapped children
            self.invert_tree_preorder(node.borrow().left.clone());
            self.invert_tree_preorder(node.borrow().right.clone());
            
            Some(node)
        } else {
            None
        }
    }

    /// # Approach 5: Iterative with Two Pointers
    /// 
    /// **Algorithm:**
    /// 1. Use iterative approach with explicit node tracking
    /// 2. Process nodes in reverse level order
    /// 3. Swap children for each node
    /// 
    /// **Time Complexity:** O(n) - Visit each node once
    /// **Space Complexity:** O(w) - Width of tree for queue
    /// 
    /// **Characteristics:**
    /// - Processes from bottom levels up
    /// - Alternative traversal order
    /// - Good for certain tree algorithms
    /// 
    /// **When to use:** When reverse level order is beneficial
    pub fn invert_tree_reverse_level(&self, root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        if root.is_none() {
            return None;
        }
        
        let mut queue = VecDeque::new();
        let mut nodes = Vec::new();
        
        queue.push_back(root.clone());
        
        // Collect all nodes first
        while let Some(Some(node)) = queue.pop_front() {
            nodes.push(node.clone());
            let node_ref = node.borrow();
            
            if node_ref.left.is_some() {
                queue.push_back(node_ref.left.clone());
            }
            if node_ref.right.is_some() {
                queue.push_back(node_ref.right.clone());
            }
        }
        
        // Process nodes in reverse order (bottom-up)
        for node in nodes.iter().rev() {
            let left = node.borrow().left.clone();
            let right = node.borrow().right.clone();
            
            node.borrow_mut().left = right;
            node.borrow_mut().right = left;
        }
        
        root
    }

    /// # Approach 6: Functional Style with Immutable Trees
    /// 
    /// **Algorithm:**
    /// 1. Create new tree nodes instead of modifying existing ones
    /// 2. For each node, create new node with swapped children
    /// 3. Recursively build inverted subtrees
    /// 
    /// **Time Complexity:** O(n) - Visit each node once
    /// **Space Complexity:** O(n) - New tree + recursion stack
    /// 
    /// **Characteristics:**
    /// - Preserves original tree structure
    /// - Functional programming style
    /// - Creates completely new tree
    /// 
    /// **When to use:** When original tree must be preserved
    pub fn invert_tree_immutable(&self, root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        match root {
            None => None,
            Some(node) => {
                let node_ref = node.borrow();
                let new_node = Rc::new(RefCell::new(TreeNode::new(node_ref.val)));
                
                // Create new node with swapped children
                new_node.borrow_mut().left = self.invert_tree_immutable(node_ref.right.clone());
                new_node.borrow_mut().right = self.invert_tree_immutable(node_ref.left.clone());
                
                Some(new_node)
            }
        }
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

    fn create_tree(values: Vec<Option<i32>>) -> Option<Rc<RefCell<TreeNode>>> {
        if values.is_empty() || values[0].is_none() {
            return None;
        }
        
        let root = Rc::new(RefCell::new(TreeNode::new(values[0].unwrap())));
        let mut queue = VecDeque::new();
        queue.push_back(root.clone());
        
        let mut i = 1;
        while !queue.is_empty() && i < values.len() {
            if let Some(node) = queue.pop_front() {
                // Left child
                if i < values.len() {
                    if let Some(val) = values[i] {
                        let left_child = Rc::new(RefCell::new(TreeNode::new(val)));
                        node.borrow_mut().left = Some(left_child.clone());
                        queue.push_back(left_child);
                    }
                    i += 1;
                }
                
                // Right child
                if i < values.len() {
                    if let Some(val) = values[i] {
                        let right_child = Rc::new(RefCell::new(TreeNode::new(val)));
                        node.borrow_mut().right = Some(right_child.clone());
                        queue.push_back(right_child);
                    }
                    i += 1;
                }
            }
        }
        
        Some(root)
    }

    fn tree_to_vec(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Option<i32>> {
        let mut result = Vec::new();
        if root.is_none() {
            return result;
        }
        
        let mut queue = VecDeque::new();
        queue.push_back(root);
        
        while !queue.is_empty() {
            if let Some(node_opt) = queue.pop_front() {
                if let Some(node) = node_opt {
                    result.push(Some(node.borrow().val));
                    queue.push_back(node.borrow().left.clone());
                    queue.push_back(node.borrow().right.clone());
                } else {
                    result.push(None);
                }
            }
        }
        
        // Remove trailing None values
        while result.last() == Some(&None) {
            result.pop();
        }
        
        result
    }

    #[test]
    fn test_basic_examples() {
        let solution = setup();
        
        // Example 1: [4,2,7,1,3,6,9] → [4,7,2,9,6,3,1]
        let tree1 = create_tree(vec![Some(4), Some(2), Some(7), Some(1), Some(3), Some(6), Some(9)]);
        let inverted1 = solution.invert_tree(tree1);
        let result1 = tree_to_vec(inverted1);
        let expected1 = vec![Some(4), Some(7), Some(2), Some(9), Some(6), Some(3), Some(1)];
        assert_eq!(result1, expected1);
        
        // Example 2: [2,1,3] → [2,3,1]
        let tree2 = create_tree(vec![Some(2), Some(1), Some(3)]);
        let inverted2 = solution.invert_tree(tree2);
        let result2 = tree_to_vec(inverted2);
        let expected2 = vec![Some(2), Some(3), Some(1)];
        assert_eq!(result2, expected2);
        
        // Example 3: [] → []
        let tree3 = None;
        let inverted3 = solution.invert_tree(tree3);
        assert_eq!(inverted3, None);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // Single node
        let single = create_tree(vec![Some(1)]);
        let inverted_single = solution.invert_tree(single);
        assert_eq!(tree_to_vec(inverted_single), vec![Some(1)]);
        
        // Only left child
        let left_only = create_tree(vec![Some(1), Some(2)]);
        let inverted_left = solution.invert_tree(left_only);
        assert_eq!(tree_to_vec(inverted_left), vec![Some(1), None, Some(2)]);
        
        // Only right child
        let right_only = create_tree(vec![Some(1), None, Some(2)]);
        let inverted_right = solution.invert_tree(right_only);
        assert_eq!(tree_to_vec(inverted_right), vec![Some(1), Some(2)]);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        let test_trees = vec![
            create_tree(vec![Some(4), Some(2), Some(7), Some(1), Some(3), Some(6), Some(9)]),
            create_tree(vec![Some(2), Some(1), Some(3)]),
            create_tree(vec![Some(1)]),
            None,
            create_tree(vec![Some(1), Some(2), Some(3), Some(4), Some(5)]),
            create_tree(vec![Some(1), Some(2), None, Some(3)]),
        ];
        
        for tree in test_trees {
            // Clone tree for each approach since they modify in-place
            let tree1 = clone_tree(tree.clone());
            let tree2 = clone_tree(tree.clone());
            let tree3 = clone_tree(tree.clone());
            let tree4 = clone_tree(tree.clone());
            let tree5 = clone_tree(tree.clone());
            let tree6 = clone_tree(tree.clone());
            
            let result1 = tree_to_vec(solution.invert_tree(tree1));
            let result2 = tree_to_vec(solution.invert_tree_iterative(tree2));
            let result3 = tree_to_vec(solution.invert_tree_bfs(tree3));
            let result4 = tree_to_vec(solution.invert_tree_preorder(tree4));
            let result5 = tree_to_vec(solution.invert_tree_reverse_level(tree5));
            let result6 = tree_to_vec(solution.invert_tree_immutable(tree6));
            
            assert_eq!(result1, result2, "Recursive vs Iterative mismatch");
            assert_eq!(result2, result3, "Iterative vs BFS mismatch");
            assert_eq!(result3, result4, "BFS vs Preorder mismatch");
            assert_eq!(result4, result5, "Preorder vs Reverse Level mismatch");
            assert_eq!(result5, result6, "Reverse Level vs Immutable mismatch");
        }
    }

    fn clone_tree(root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        match root {
            None => None,
            Some(node) => {
                let node_ref = node.borrow();
                let new_node = Rc::new(RefCell::new(TreeNode::new(node_ref.val)));
                new_node.borrow_mut().left = clone_tree(node_ref.left.clone());
                new_node.borrow_mut().right = clone_tree(node_ref.right.clone());
                Some(new_node)
            }
        }
    }

    #[test]
    fn test_symmetric_tree() {
        let solution = setup();
        
        // Symmetric tree should become symmetric when inverted
        let symmetric = create_tree(vec![Some(1), Some(2), Some(2), Some(3), Some(3), Some(3), Some(3)]);
        let inverted = solution.invert_tree_immutable(symmetric);
        let result = tree_to_vec(inverted);
        let expected = vec![Some(1), Some(2), Some(2), Some(3), Some(3), Some(3), Some(3)];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_deep_tree() {
        let solution = setup();
        
        // Deep left chain
        let deep_left = create_tree(vec![Some(1), Some(2), None, Some(3), None, Some(4)]);
        let inverted = solution.invert_tree_immutable(deep_left);
        let result = tree_to_vec(inverted);
        let expected = vec![Some(1), None, Some(2), None, Some(3), None, Some(4)];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_negative_values() {
        let solution = setup();
        
        // Tree with negative values
        let negative_tree = create_tree(vec![Some(-1), Some(-2), Some(-3)]);
        let inverted = solution.invert_tree_immutable(negative_tree);
        let result = tree_to_vec(inverted);
        let expected = vec![Some(-1), Some(-3), Some(-2)];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_double_inversion() {
        let solution = setup();
        
        // Inverting twice should give original tree
        let original = create_tree(vec![Some(1), Some(2), Some(3), Some(4), Some(5)]);
        let original_vec = tree_to_vec(original.clone());
        
        let inverted_once = solution.invert_tree_immutable(original);
        let inverted_twice = solution.invert_tree_immutable(inverted_once);
        let final_vec = tree_to_vec(inverted_twice);
        
        assert_eq!(original_vec, final_vec);
    }

    #[test]
    fn test_large_tree() {
        let solution = setup();
        
        // Complete binary tree with 7 nodes
        let large_tree = create_tree(vec![
            Some(1), Some(2), Some(3), Some(4), Some(5), Some(6), Some(7)
        ]);
        let inverted = solution.invert_tree_immutable(large_tree);
        let result = tree_to_vec(inverted);
        let expected = vec![Some(1), Some(3), Some(2), Some(7), Some(6), Some(5), Some(4)];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_boundary_values() {
        let solution = setup();
        
        // Minimum and maximum node values
        let boundary_tree = create_tree(vec![Some(0), Some(-100), Some(100)]);
        let inverted = solution.invert_tree_immutable(boundary_tree);
        let result = tree_to_vec(inverted);
        let expected = vec![Some(0), Some(100), Some(-100)];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_inversion_properties() {
        let solution = setup();
        
        // Property: left subtree becomes right subtree
        let tree = create_tree(vec![Some(1), Some(2), Some(3)]);
        let original_left = tree.as_ref().unwrap().borrow().left.as_ref().unwrap().borrow().val;
        let original_right = tree.as_ref().unwrap().borrow().right.as_ref().unwrap().borrow().val;
        
        let inverted = solution.invert_tree_immutable(tree);
        let new_left = inverted.as_ref().unwrap().borrow().left.as_ref().unwrap().borrow().val;
        let new_right = inverted.as_ref().unwrap().borrow().right.as_ref().unwrap().borrow().val;
        
        assert_eq!(original_left, new_right);
        assert_eq!(original_right, new_left);
    }
}