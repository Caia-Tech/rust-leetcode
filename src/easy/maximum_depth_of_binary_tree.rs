//! # Problem 104: Maximum Depth of Binary Tree
//!
//! Given the `root` of a binary tree, return its maximum depth.
//!
//! A binary tree's maximum depth is the number of nodes along the longest path from the root 
//! node down to the farthest leaf node.
//!
//! ## Examples
//!
//! ```text
//! Input: root = [3,9,20,null,null,15,7]
//! Output: 3
//! ```
//!
//! ```text
//! Input: root = [1,null,2]
//! Output: 2
//! ```
//!
//! ## Constraints
//!
//! * The number of nodes in the tree is in the range [0, 10^4]
//! * -100 <= Node.val <= 100

use crate::utils::data_structures::TreeNode;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::{VecDeque, HashMap};

/// Solution for Maximum Depth of Binary Tree problem
pub struct Solution;

impl Solution {
    /// Creates a new instance of Solution
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Recursive DFS (Optimal for most cases)
    /// 
    /// **Algorithm:**
    /// 1. Base case: if node is null, return 0
    /// 2. Recursively find depth of left and right subtrees
    /// 3. Return 1 + max(left_depth, right_depth)
    /// 
    /// **Time Complexity:** O(n) - Visit each node once
    /// **Space Complexity:** O(h) - Recursion stack depth, h = height of tree
    /// 
    /// **Key Insights:**
    /// - Classic divide and conquer approach
    /// - Maximum depth = 1 + maximum of subtree depths
    /// - Natural recursive structure matches tree structure
    /// 
    /// **Why this works:**
    /// - Tree depth is defined recursively
    /// - Each node contributes 1 to the depth
    /// - Take the longer path from left or right subtree
    /// 
    /// **Visualization:**
    /// ```text
    ///       3      ← depth 1
    ///      / \
    ///     9   20   ← depth 2  
    ///        / \
    ///       15  7  ← depth 3
    /// max_depth(3) = 1 + max(max_depth(9), max_depth(20))
    ///              = 1 + max(1, 2) = 3
    /// ```
    pub fn max_depth(&self, root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        match root {
            None => 0,
            Some(node) => {
                let node_ref = node.borrow();
                let left_depth = self.max_depth(node_ref.left.clone());
                let right_depth = self.max_depth(node_ref.right.clone());
                1 + left_depth.max(right_depth)
            }
        }
    }

    /// # Approach 2: Iterative DFS with Stack
    /// 
    /// **Algorithm:**
    /// 1. Use stack to simulate recursion
    /// 2. Push (node, current_depth) pairs
    /// 3. Track maximum depth seen so far
    /// 4. Process children with depth + 1
    /// 
    /// **Time Complexity:** O(n) - Visit each node once
    /// **Space Complexity:** O(h) - Stack size in worst case
    /// 
    /// **Advantages:**
    /// - Avoids recursion stack overflow for very deep trees
    /// - More control over memory usage
    /// - Can be easier to debug step by step
    /// 
    /// **When to use:** Very deep trees or stack overflow concerns
    pub fn max_depth_iterative_dfs(&self, root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        if root.is_none() {
            return 0;
        }
        
        let mut stack = Vec::new();
        let mut max_depth = 0;
        
        stack.push((root, 1));
        
        while let Some((node, depth)) = stack.pop() {
            if let Some(n) = node {
                max_depth = max_depth.max(depth);
                let node_ref = n.borrow();
                
                if node_ref.left.is_some() {
                    stack.push((node_ref.left.clone(), depth + 1));
                }
                if node_ref.right.is_some() {
                    stack.push((node_ref.right.clone(), depth + 1));
                }
            }
        }
        
        max_depth
    }

    /// # Approach 3: Level-Order Traversal (BFS)
    /// 
    /// **Algorithm:**
    /// 1. Use queue for level-by-level traversal
    /// 2. Process all nodes at current level
    /// 3. Increment depth after each level
    /// 4. Continue until queue is empty
    /// 
    /// **Time Complexity:** O(n) - Visit each node once
    /// **Space Complexity:** O(w) - Queue size, w = maximum width of tree
    /// 
    /// **Characteristics:**
    /// - Processes tree level by level
    /// - Naturally tracks depth as number of levels
    /// - Good for finding nodes at specific depths
    /// 
    /// **When to use:** When you need level-by-level processing
    pub fn max_depth_bfs(&self, root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        if root.is_none() {
            return 0;
        }
        
        let mut queue = VecDeque::new();
        let mut depth = 0;
        
        queue.push_back(root);
        
        while !queue.is_empty() {
            let level_size = queue.len();
            depth += 1;
            
            // Process all nodes at current level
            for _ in 0..level_size {
                if let Some(Some(node)) = queue.pop_front() {
                    let node_ref = node.borrow();
                    
                    if node_ref.left.is_some() {
                        queue.push_back(node_ref.left.clone());
                    }
                    if node_ref.right.is_some() {
                        queue.push_back(node_ref.right.clone());
                    }
                }
            }
        }
        
        depth
    }

    /// # Approach 4: Iterative with Two Stacks (Postorder-like)
    /// 
    /// **Algorithm:**
    /// 1. Use two stacks to simulate postorder traversal
    /// 2. Track depth for each node using parallel stack
    /// 3. Update maximum depth when processing leaves
    /// 
    /// **Time Complexity:** O(n) - Visit each node once
    /// **Space Complexity:** O(h) - Stack depth
    /// 
    /// **Characteristics:**
    /// - Processes nodes in postorder fashion
    /// - Good for bottom-up calculations
    /// - Alternative to recursive approach
    /// 
    /// **When to use:** When postorder processing is needed
    pub fn max_depth_two_stacks(&self, root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        if root.is_none() {
            return 0;
        }
        
        let mut stack = Vec::new();
        let mut depth_stack = Vec::new();
        let mut max_depth = 0;
        
        stack.push(root);
        depth_stack.push(1);
        
        while let Some(node) = stack.pop() {
            if let (Some(n), Some(depth)) = (node, depth_stack.pop()) {
                max_depth = max_depth.max(depth);
                let node_ref = n.borrow();
                
                if node_ref.left.is_some() {
                    stack.push(node_ref.left.clone());
                    depth_stack.push(depth + 1);
                }
                if node_ref.right.is_some() {
                    stack.push(node_ref.right.clone());
                    depth_stack.push(depth + 1);
                }
            }
        }
        
        max_depth
    }

    /// # Approach 5: Tail Recursive with Continuation
    /// 
    /// **Algorithm:**
    /// 1. Use tail recursion with explicit continuation
    /// 2. Track maximum depth through continuation chain
    /// 3. Optimize for tail call elimination
    /// 
    /// **Time Complexity:** O(n) - Visit each node once
    /// **Space Complexity:** O(h) - Continuation stack
    /// 
    /// **Educational value:**
    /// - Demonstrates functional programming concepts
    /// - Shows how to convert recursion to tail recursion
    /// - Illustrates continuation-passing style
    /// 
    /// **When to use:** Functional programming contexts or educational purposes
    pub fn max_depth_tail_recursive(&self, root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        fn tail_helper(node: Option<Rc<RefCell<TreeNode>>>, depth: i32, max_so_far: i32) -> i32 {
            match node {
                None => max_so_far,
                Some(n) => {
                    let node_ref = n.borrow();
                    let current_max = max_so_far.max(depth);
                    
                    let left_max = tail_helper(node_ref.left.clone(), depth + 1, current_max);
                    tail_helper(node_ref.right.clone(), depth + 1, left_max)
                }
            }
        }
        
        if root.is_none() {
            0
        } else {
            tail_helper(root, 1, 0)
        }
    }

    /// # Approach 6: Bottom-Up with Explicit Stack
    /// 
    /// **Algorithm:**
    /// 1. Use postorder traversal with explicit stack
    /// 2. Process nodes bottom-up after visiting children
    /// 3. Calculate depth based on children's depths
    /// 
    /// **Time Complexity:** O(n) - Each node processed once
    /// **Space Complexity:** O(h) - Stack depth
    /// 
    /// **Advantages:**
    /// - Natural bottom-up calculation
    /// - Mirrors recursive solution logic
    /// - Avoids recursion stack overflow
    /// 
    /// **When useful:** When you need bottom-up processing without recursion
    pub fn max_depth_bottom_up(&self, root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        if root.is_none() {
            return 0;
        }
        
        let mut stack = Vec::new();
        let mut depths = HashMap::new();
        let mut visited = std::collections::HashSet::new();
        
        stack.push(root.clone());
        
        while let Some(node_opt) = stack.last().cloned() {
            if let Some(node) = node_opt {
                let node_ptr = node.as_ptr() as usize;
                
                if visited.contains(&node_ptr) {
                    // Process node after children are processed
                    stack.pop();
                    let node_ref = node.borrow();
                    
                    let left_depth = if node_ref.left.is_some() {
                        *depths.get(&(node_ref.left.as_ref().unwrap().as_ptr() as usize)).unwrap_or(&0)
                    } else {
                        0
                    };
                    
                    let right_depth = if node_ref.right.is_some() {
                        *depths.get(&(node_ref.right.as_ref().unwrap().as_ptr() as usize)).unwrap_or(&0)
                    } else {
                        0
                    };
                    
                    depths.insert(node_ptr, 1 + left_depth.max(right_depth));
                } else {
                    // Mark as visited and add children to stack
                    visited.insert(node_ptr);
                    let node_ref = node.borrow();
                    
                    if node_ref.right.is_some() {
                        stack.push(node_ref.right.clone());
                    }
                    if node_ref.left.is_some() {
                        stack.push(node_ref.left.clone());
                    }
                }
            } else {
                stack.pop();
            }
        }
        
        if let Some(root_node) = root {
            *depths.get(&(root_node.as_ptr() as usize)).unwrap_or(&0)
        } else {
            0
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

    #[test]
    fn test_basic_examples() {
        let solution = setup();
        
        // Example 1: [3,9,20,null,null,15,7] → 3
        let tree1 = create_tree(vec![Some(3), Some(9), Some(20), None, None, Some(15), Some(7)]);
        assert_eq!(solution.max_depth(tree1), 3);
        
        // Example 2: [1,null,2] → 2
        let tree2 = create_tree(vec![Some(1), None, Some(2)]);
        assert_eq!(solution.max_depth(tree2), 2);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // Empty tree
        assert_eq!(solution.max_depth(None), 0);
        
        // Single node
        let single_node = create_tree(vec![Some(1)]);
        assert_eq!(solution.max_depth(single_node), 1);
        
        // Only left children (degenerate tree)
        let left_only = create_tree(vec![Some(1), Some(2), None, Some(3)]);
        assert_eq!(solution.max_depth(left_only), 3);
        
        // Only right children (degenerate tree)
        let right_only = create_tree(vec![Some(1), None, Some(2), None, Some(3)]);
        assert_eq!(solution.max_depth(right_only), 3);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        let test_trees = vec![
            create_tree(vec![Some(3), Some(9), Some(20), None, None, Some(15), Some(7)]),
            create_tree(vec![Some(1), None, Some(2)]),
            create_tree(vec![Some(1)]),
            None,
            create_tree(vec![Some(1), Some(2), Some(3), Some(4), Some(5)]),
            create_tree(vec![Some(1), Some(2), None, Some(3), None, Some(4)]),
        ];
        
        for tree in test_trees {
            let result1 = solution.max_depth(tree.clone());
            let result2 = solution.max_depth_iterative_dfs(tree.clone());
            let result3 = solution.max_depth_bfs(tree.clone());
            let result4 = solution.max_depth_two_stacks(tree.clone());
            let result5 = solution.max_depth_tail_recursive(tree.clone());
            let result6 = solution.max_depth_bottom_up(tree.clone());
            
            assert_eq!(result1, result2, "Recursive vs Iterative DFS mismatch");
            assert_eq!(result2, result3, "Iterative DFS vs BFS mismatch");
            assert_eq!(result3, result4, "BFS vs Two Stacks mismatch");
            assert_eq!(result4, result5, "Two Stacks vs Tail Recursive mismatch");
            assert_eq!(result5, result6, "Tail Recursive vs Bottom Up mismatch");
        }
    }

    #[test]
    fn test_balanced_trees() {
        let solution = setup();
        
        // Perfect binary tree of depth 3
        let balanced = create_tree(vec![
            Some(1), Some(2), Some(3), Some(4), Some(5), Some(6), Some(7)
        ]);
        assert_eq!(solution.max_depth(balanced), 3);
        
        // Complete binary tree
        let complete = create_tree(vec![
            Some(1), Some(2), Some(3), Some(4), Some(5), Some(6)
        ]);
        assert_eq!(solution.max_depth(complete), 3);
    }

    #[test]
    fn test_unbalanced_trees() {
        let solution = setup();
        
        // Left-heavy tree
        let left_heavy = create_tree(vec![
            Some(1), Some(2), Some(3), Some(4), Some(5), None, None, Some(6)
        ]);
        assert_eq!(solution.max_depth(left_heavy), 4);
        
        // Right-heavy tree
        let right_heavy = create_tree(vec![
            Some(1), Some(2), Some(3), None, None, Some(4), Some(5), None, None, None, Some(6)
        ]);
        assert_eq!(solution.max_depth(right_heavy), 4);
    }

    #[test]
    fn test_deep_trees() {
        let solution = setup();
        
        // Chain of left children (depth 5)
        let deep_left = create_tree(vec![
            Some(1), Some(2), None, Some(3), None, Some(4), None, Some(5)
        ]);
        assert_eq!(solution.max_depth(deep_left), 5);
        
        // Chain of right children (depth 4)
        let deep_right = create_tree(vec![
            Some(1), None, Some(2), None, Some(3), None, Some(4)
        ]);
        assert_eq!(solution.max_depth(deep_right), 4);
    }

    #[test]
    fn test_specific_structures() {
        let solution = setup();
        
        // T-shaped tree
        let t_shaped = create_tree(vec![
            Some(1), Some(2), Some(3), Some(4), Some(5), Some(6), Some(7)
        ]);
        assert_eq!(solution.max_depth(t_shaped), 3);
        
        // Y-shaped tree
        let y_shaped = create_tree(vec![
            Some(1), Some(2), Some(3), Some(4), None, None, Some(5)
        ]);
        assert_eq!(solution.max_depth(y_shaped), 3);
    }

    #[test]
    fn test_negative_values() {
        let solution = setup();
        
        // Tree with negative values
        let negative_tree = create_tree(vec![
            Some(-100), Some(-50), Some(0), Some(50), Some(100)
        ]);
        assert_eq!(solution.max_depth(negative_tree), 3);
        
        // All negative values
        let all_negative = create_tree(vec![
            Some(-1), Some(-2), Some(-3)
        ]);
        assert_eq!(solution.max_depth(all_negative), 2);
    }

    #[test]
    fn test_large_trees() {
        let solution = setup();
        
        // Build a tree with many nodes
        let mut values = Vec::new();
        for i in 1..=15 {
            values.push(Some(i));
        }
        let large_tree = create_tree(values);
        assert_eq!(solution.max_depth(large_tree), 4); // 2^4 - 1 = 15 nodes max in perfect tree
    }

    #[test]
    fn test_property_based() {
        let solution = setup();
        
        // Property: max_depth >= 1 for non-empty trees
        let non_empty = create_tree(vec![Some(42)]);
        assert!(solution.max_depth(non_empty) >= 1);
        
        // Property: adding a child increases depth by at most 1
        let parent = create_tree(vec![Some(1), Some(2)]);
        let parent_depth = solution.max_depth(parent);
        
        let with_child = create_tree(vec![Some(1), Some(2), None, Some(3)]);
        let with_child_depth = solution.max_depth(with_child);
        
        assert!(with_child_depth <= parent_depth + 1);
    }

    #[test]
    fn test_constraint_boundaries() {
        let solution = setup();
        
        // Minimum node value
        let min_val = create_tree(vec![Some(-100)]);
        assert_eq!(solution.max_depth(min_val), 1);
        
        // Maximum node value
        let max_val = create_tree(vec![Some(100)]);
        assert_eq!(solution.max_depth(max_val), 1);
        
        // Test with boundary values in larger tree
        let boundary_tree = create_tree(vec![
            Some(0), Some(-100), Some(100), Some(-50), Some(50)
        ]);
        assert_eq!(solution.max_depth(boundary_tree), 3);
    }
}