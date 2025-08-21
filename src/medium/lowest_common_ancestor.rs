//! Problem 236: Lowest Common Ancestor of Binary Tree
//!
//! Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.
//! According to the definition of LCA on Wikipedia: "The lowest common ancestor is defined 
//! between two nodes p and q as the lowest node in T that has both p and q as descendants 
//! (where we allow a node to be a descendant of itself)."
//!
//! All of the nodes' values will be unique.
//! p and q are different and both values will exist in the tree.

use std::rc::Rc;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};

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
    /// Approach 1: Recursive DFS with Postorder Traversal
    /// 
    /// This is the most intuitive approach. We recursively search for both nodes
    /// in the left and right subtrees. The LCA is found when:
    /// 1. Current node is one of p or q, and we found the other in a subtree
    /// 2. We found p in one subtree and q in another subtree
    /// 
    /// Time Complexity: O(n) - visit each node once
    /// Space Complexity: O(h) - recursion stack depth where h is tree height
    pub fn lowest_common_ancestor_recursive(
        root: Option<Rc<RefCell<TreeNode>>>,
        p: Option<Rc<RefCell<TreeNode>>>,
        q: Option<Rc<RefCell<TreeNode>>>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        Self::find_lca_recursive(root, p, q)
    }
    
    fn find_lca_recursive(
        node: Option<Rc<RefCell<TreeNode>>>,
        p: Option<Rc<RefCell<TreeNode>>>,
        q: Option<Rc<RefCell<TreeNode>>>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        match node {
            None => None,
            Some(current) => {
                let p_val = p.as_ref().unwrap().borrow().val;
                let q_val = q.as_ref().unwrap().borrow().val;
                let current_val = current.borrow().val;
                
                // If current node is either p or q, it could be the LCA
                if current_val == p_val || current_val == q_val {
                    return Some(current);
                }
                
                // Search in left and right subtrees
                let left_result = Self::find_lca_recursive(
                    current.borrow().left.clone(), 
                    p.clone(), 
                    q.clone()
                );
                let right_result = Self::find_lca_recursive(
                    current.borrow().right.clone(), 
                    p.clone(), 
                    q.clone()
                );
                
                // If both subtrees return non-null, current node is LCA
                match (left_result, right_result) {
                    (Some(_), Some(_)) => Some(current),
                    (Some(left), None) => Some(left),
                    (None, Some(right)) => Some(right),
                    (None, None) => None,
                }
            }
        }
    }
    
    /// Approach 2: Parent Pointer Approach
    /// 
    /// Build a map from child to parent, then find the path from p to root
    /// and from q to root. The first common node in these paths is the LCA.
    /// 
    /// Time Complexity: O(n) - traverse tree once to build parent map, then O(h) to find LCA
    /// Space Complexity: O(n) - parent map storage
    pub fn lowest_common_ancestor_parent_pointers(
        root: Option<Rc<RefCell<TreeNode>>>,
        p: Option<Rc<RefCell<TreeNode>>>,
        q: Option<Rc<RefCell<TreeNode>>>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        if root.is_none() || p.is_none() || q.is_none() {
            return None;
        }
        
        let mut parent_map: HashMap<i32, Option<Rc<RefCell<TreeNode>>>> = HashMap::new();
        Self::build_parent_map(root.clone(), None, &mut parent_map);
        
        // Get path from p to root
        let mut p_ancestors = std::collections::HashSet::new();
        let mut current = p.clone();
        
        while let Some(node) = current {
            let val = node.borrow().val;
            p_ancestors.insert(val);
            current = parent_map.get(&val).and_then(|parent| parent.clone());
        }
        
        // Find first common ancestor in q's path to root
        let mut current = q.clone();
        while let Some(node) = current {
            let val = node.borrow().val;
            if p_ancestors.contains(&val) {
                return Some(node);
            }
            current = parent_map.get(&val).and_then(|parent| parent.clone());
        }
        
        None
    }
    
    fn build_parent_map(
        node: Option<Rc<RefCell<TreeNode>>>,
        parent: Option<Rc<RefCell<TreeNode>>>,
        parent_map: &mut HashMap<i32, Option<Rc<RefCell<TreeNode>>>>,
    ) {
        if let Some(current) = node {
            let val = current.borrow().val;
            parent_map.insert(val, parent);
            
            Self::build_parent_map(current.borrow().left.clone(), Some(current.clone()), parent_map);
            Self::build_parent_map(current.borrow().right.clone(), Some(current.clone()), parent_map);
        }
    }
    
    /// Approach 3: Iterative DFS with Stack
    /// 
    /// Use an explicit stack to simulate the recursive approach.
    /// Track the state of each node (whether we've processed left/right subtrees).
    /// 
    /// Time Complexity: O(n) - visit each node once
    /// Space Complexity: O(h) - stack space
    pub fn lowest_common_ancestor_iterative(
        root: Option<Rc<RefCell<TreeNode>>>,
        p: Option<Rc<RefCell<TreeNode>>>,
        q: Option<Rc<RefCell<TreeNode>>>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        if root.is_none() || p.is_none() || q.is_none() {
            return None;
        }
        
        let p_val = p.as_ref().unwrap().borrow().val;
        let q_val = q.as_ref().unwrap().borrow().val;
        
        #[derive(Clone)]
        enum State {
            PreProcess,
            PostProcess(Option<Rc<RefCell<TreeNode>>>, Option<Rc<RefCell<TreeNode>>>),
        }
        
        let mut stack = vec![(root.clone(), State::PreProcess)];
        let mut results: HashMap<i32, Option<Rc<RefCell<TreeNode>>>> = HashMap::new();
        
        while let Some((node_opt, state)) = stack.pop() {
            if let Some(node) = node_opt {
                let val = node.borrow().val;
                
                match state {
                    State::PreProcess => {
                        // Check if this is one of our target nodes
                        if val == p_val || val == q_val {
                            results.insert(val, Some(node.clone()));
                            continue;
                        }
                        
                        // Push post-processing state
                        stack.push((Some(node.clone()), State::PostProcess(None, None)));
                        
                        // Push children for processing
                        if let Some(right) = node.borrow().right.clone() {
                            stack.push((Some(right), State::PreProcess));
                        }
                        if let Some(left) = node.borrow().left.clone() {
                            stack.push((Some(left), State::PreProcess));
                        }
                    }
                    State::PostProcess(_, _) => {
                        let left_result = node.borrow().left.as_ref()
                            .and_then(|left| results.get(&left.borrow().val))
                            .and_then(|result| result.clone());
                        
                        let right_result = node.borrow().right.as_ref()
                            .and_then(|right| results.get(&right.borrow().val))
                            .and_then(|result| result.clone());
                        
                        let result = match (left_result, right_result) {
                            (Some(_), Some(_)) => Some(node.clone()),
                            (Some(left), None) => Some(left),
                            (None, Some(right)) => Some(right),
                            (None, None) => None,
                        };
                        
                        results.insert(val, result);
                    }
                }
            }
        }
        
        root.as_ref()
            .and_then(|r| results.get(&r.borrow().val))
            .and_then(|result| result.clone())
    }
    
    /// Approach 4: Path Comparison Approach
    /// 
    /// Find the path from root to p and from root to q.
    /// The LCA is the last common node in both paths.
    /// 
    /// Time Complexity: O(n) - might traverse entire tree to find paths
    /// Space Complexity: O(h) - path storage
    pub fn lowest_common_ancestor_path_comparison(
        root: Option<Rc<RefCell<TreeNode>>>,
        p: Option<Rc<RefCell<TreeNode>>>,
        q: Option<Rc<RefCell<TreeNode>>>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        if root.is_none() || p.is_none() || q.is_none() {
            return None;
        }
        
        let p_val = p.as_ref().unwrap().borrow().val;
        let q_val = q.as_ref().unwrap().borrow().val;
        
        let path_to_p = Self::find_path(root.clone(), p_val);
        let path_to_q = Self::find_path(root.clone(), q_val);
        
        if path_to_p.is_empty() || path_to_q.is_empty() {
            return None;
        }
        
        // Find last common node in both paths
        let mut lca = None;
        let min_len = path_to_p.len().min(path_to_q.len());
        
        for i in 0..min_len {
            if path_to_p[i].borrow().val == path_to_q[i].borrow().val {
                lca = Some(path_to_p[i].clone());
            } else {
                break;
            }
        }
        
        lca
    }
    
    fn find_path(
        root: Option<Rc<RefCell<TreeNode>>>,
        target: i32,
    ) -> Vec<Rc<RefCell<TreeNode>>> {
        let mut path = Vec::new();
        Self::find_path_helper(root, target, &mut path);
        path
    }
    
    fn find_path_helper(
        node: Option<Rc<RefCell<TreeNode>>>,
        target: i32,
        path: &mut Vec<Rc<RefCell<TreeNode>>>,
    ) -> bool {
        if let Some(current) = node {
            path.push(current.clone());
            
            let val = current.borrow().val;
            if val == target {
                return true;
            }
            
            if Self::find_path_helper(current.borrow().left.clone(), target, path) ||
               Self::find_path_helper(current.borrow().right.clone(), target, path) {
                return true;
            }
            
            path.pop();
        }
        false
    }
    
    /// Approach 5: Level Order Traversal with Parent Tracking
    /// 
    /// Use BFS to traverse the tree level by level while building parent relationships.
    /// Once both nodes are found, trace back to find common ancestor.
    /// 
    /// Time Complexity: O(n) - BFS traversal
    /// Space Complexity: O(n) - queue and parent map
    pub fn lowest_common_ancestor_bfs(
        root: Option<Rc<RefCell<TreeNode>>>,
        p: Option<Rc<RefCell<TreeNode>>>,
        q: Option<Rc<RefCell<TreeNode>>>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        if root.is_none() || p.is_none() || q.is_none() {
            return None;
        }
        
        let p_val = p.as_ref().unwrap().borrow().val;
        let q_val = q.as_ref().unwrap().borrow().val;
        
        let mut queue = VecDeque::new();
        let mut parent_map: HashMap<i32, Option<Rc<RefCell<TreeNode>>>> = HashMap::new();
        
        queue.push_back(root.clone());
        parent_map.insert(root.as_ref().unwrap().borrow().val, None);
        
        let mut found_p = false;
        let mut found_q = false;
        
        // BFS until both nodes are found
        while !queue.is_empty() && (!found_p || !found_q) {
            if let Some(node) = queue.pop_front() {
                if let Some(current) = node {
                    let val = current.borrow().val;
                    
                    if val == p_val {
                        found_p = true;
                    }
                    if val == q_val {
                        found_q = true;
                    }
                    
                    // Add children to queue and parent map
                    if let Some(left) = current.borrow().left.clone() {
                        queue.push_back(Some(left.clone()));
                        parent_map.insert(left.borrow().val, Some(current.clone()));
                    }
                    
                    if let Some(right) = current.borrow().right.clone() {
                        queue.push_back(Some(right.clone()));
                        parent_map.insert(right.borrow().val, Some(current.clone()));
                    }
                }
            }
        }
        
        // Now find LCA using parent pointers
        let mut p_ancestors = std::collections::HashSet::new();
        let mut current_val = p_val;
        
        // Collect all ancestors of p
        loop {
            p_ancestors.insert(current_val);
            if let Some(parent_opt) = parent_map.get(&current_val) {
                if let Some(parent) = parent_opt {
                    current_val = parent.borrow().val;
                } else {
                    break; // Reached root
                }
            } else {
                break;
            }
        }
        
        // Find first common ancestor in q's path
        current_val = q_val;
        loop {
            if p_ancestors.contains(&current_val) {
                // Find the actual node with this value
                return Self::find_node_by_value(root, current_val);
            }
            if let Some(parent_opt) = parent_map.get(&current_val) {
                if let Some(parent) = parent_opt {
                    current_val = parent.borrow().val;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        
        None
    }
    
    fn find_node_by_value(
        root: Option<Rc<RefCell<TreeNode>>>,
        target_val: i32,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        if let Some(node) = root {
            if node.borrow().val == target_val {
                return Some(node);
            }
            
            if let Some(found) = Self::find_node_by_value(node.borrow().left.clone(), target_val) {
                return Some(found);
            }
            
            if let Some(found) = Self::find_node_by_value(node.borrow().right.clone(), target_val) {
                return Some(found);
            }
        }
        None
    }
    
    /// Approach 6: Optimized Single Pass with Early Termination
    /// 
    /// Enhanced recursive approach that stops searching once LCA is found.
    /// Uses a more efficient state tracking mechanism.
    /// 
    /// Time Complexity: O(n) worst case, but often better due to early termination
    /// Space Complexity: O(h) - recursion stack
    pub fn lowest_common_ancestor_optimized(
        root: Option<Rc<RefCell<TreeNode>>>,
        p: Option<Rc<RefCell<TreeNode>>>,
        q: Option<Rc<RefCell<TreeNode>>>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        if root.is_none() || p.is_none() || q.is_none() {
            return None;
        }
        
        let p_val = p.as_ref().unwrap().borrow().val;
        let q_val = q.as_ref().unwrap().borrow().val;
        
        // Special case: if p and q are the same node
        if p_val == q_val {
            return Self::find_node_by_value(root, p_val);
        }
        
        let mut result = None;
        Self::find_lca_optimized(root, p_val, q_val, &mut result);
        result
    }
    
    fn find_lca_optimized(
        node: Option<Rc<RefCell<TreeNode>>>,
        p_val: i32,
        q_val: i32,
        result: &mut Option<Rc<RefCell<TreeNode>>>,
    ) -> i32 {
        if node.is_none() || result.is_some() {
            return 0;
        }
        
        let current = node.unwrap();
        let val = current.borrow().val;
        
        // Check if current node is one of the targets
        let current_match = if val == p_val || val == q_val { 1 } else { 0 };
        
        // Search left and right subtrees
        let left_count = Self::find_lca_optimized(
            current.borrow().left.clone(),
            p_val,
            q_val,
            result,
        );
        
        if result.is_some() {
            return 0; // Early termination
        }
        
        let right_count = Self::find_lca_optimized(
            current.borrow().right.clone(),
            p_val,
            q_val,
            result,
        );
        
        let total_count = current_match + left_count + right_count;
        
        // If we found both nodes, this is the LCA
        if total_count == 2 && result.is_none() {
            *result = Some(current.clone());
        }
        
        total_count
    }
}

/// Helper function to create a tree node
pub fn create_node(val: i32) -> Option<Rc<RefCell<TreeNode>>> {
    Some(Rc::new(RefCell::new(TreeNode::new(val))))
}

/// Helper function to build a tree from array representation
pub fn build_tree(values: Vec<Option<i32>>) -> Option<Rc<RefCell<TreeNode>>> {
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

/// Helper function to find a node by value in the tree
pub fn find_node(root: Option<Rc<RefCell<TreeNode>>>, target: i32) -> Option<Rc<RefCell<TreeNode>>> {
    if let Some(node) = root {
        if node.borrow().val == target {
            return Some(node);
        }
        
        if let Some(found) = find_node(node.borrow().left.clone(), target) {
            return Some(found);
        }
        
        if let Some(found) = find_node(node.borrow().right.clone(), target) {
            return Some(found);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_lca() {
        // Tree: [3,5,1,6,2,0,8,null,null,7,4]
        //       3
        //      / \
        //     5   1
        //    / \ / \
        //   6  2 0  8
        //     / \
        //    7   4
        let root = build_tree(vec![
            Some(3), Some(5), Some(1), Some(6), Some(2), Some(0), Some(8),
            None, None, Some(7), Some(4)
        ]);
        
        let p = find_node(root.clone(), 5);
        let q = find_node(root.clone(), 1);
        
        // Test all approaches
        let result1 = Solution::lowest_common_ancestor_recursive(root.clone(), p.clone(), q.clone());
        assert_eq!(result1.unwrap().borrow().val, 3);
        
        let result2 = Solution::lowest_common_ancestor_parent_pointers(root.clone(), p.clone(), q.clone());
        assert_eq!(result2.unwrap().borrow().val, 3);
        
        let result3 = Solution::lowest_common_ancestor_iterative(root.clone(), p.clone(), q.clone());
        assert_eq!(result3.unwrap().borrow().val, 3);
        
        let result4 = Solution::lowest_common_ancestor_path_comparison(root.clone(), p.clone(), q.clone());
        assert_eq!(result4.unwrap().borrow().val, 3);
        
        let result5 = Solution::lowest_common_ancestor_bfs(root.clone(), p.clone(), q.clone());
        assert_eq!(result5.unwrap().borrow().val, 3);
        
        let result6 = Solution::lowest_common_ancestor_optimized(root.clone(), p.clone(), q.clone());
        assert_eq!(result6.unwrap().borrow().val, 3);
    }
    
    #[test]
    fn test_lca_deeper_nodes() {
        // Same tree: [3,5,1,6,2,0,8,null,null,7,4]
        let root = build_tree(vec![
            Some(3), Some(5), Some(1), Some(6), Some(2), Some(0), Some(8),
            None, None, Some(7), Some(4)
        ]);
        
        let p = find_node(root.clone(), 6);
        let q = find_node(root.clone(), 4);
        
        // LCA should be 5
        let result1 = Solution::lowest_common_ancestor_recursive(root.clone(), p.clone(), q.clone());
        assert_eq!(result1.unwrap().borrow().val, 5);
        
        let result2 = Solution::lowest_common_ancestor_parent_pointers(root.clone(), p.clone(), q.clone());
        assert_eq!(result2.unwrap().borrow().val, 5);
        
        let result3 = Solution::lowest_common_ancestor_iterative(root.clone(), p.clone(), q.clone());
        assert_eq!(result3.unwrap().borrow().val, 5);
        
        let result4 = Solution::lowest_common_ancestor_path_comparison(root.clone(), p.clone(), q.clone());
        assert_eq!(result4.unwrap().borrow().val, 5);
        
        let result5 = Solution::lowest_common_ancestor_bfs(root.clone(), p.clone(), q.clone());
        assert_eq!(result5.unwrap().borrow().val, 5);
        
        let result6 = Solution::lowest_common_ancestor_optimized(root.clone(), p.clone(), q.clone());
        assert_eq!(result6.unwrap().borrow().val, 5);
    }
    
    #[test]
    fn test_one_node_is_ancestor() {
        // Tree: [3,5,1,6,2,0,8,null,null,7,4]
        let root = build_tree(vec![
            Some(3), Some(5), Some(1), Some(6), Some(2), Some(0), Some(8),
            None, None, Some(7), Some(4)
        ]);
        
        let p = find_node(root.clone(), 5);
        let q = find_node(root.clone(), 4);
        
        // LCA should be 5 (p is ancestor of q)
        let result1 = Solution::lowest_common_ancestor_recursive(root.clone(), p.clone(), q.clone());
        assert_eq!(result1.unwrap().borrow().val, 5);
        
        let result2 = Solution::lowest_common_ancestor_parent_pointers(root.clone(), p.clone(), q.clone());
        assert_eq!(result2.unwrap().borrow().val, 5);
        
        let result3 = Solution::lowest_common_ancestor_iterative(root.clone(), p.clone(), q.clone());
        assert_eq!(result3.unwrap().borrow().val, 5);
        
        let result4 = Solution::lowest_common_ancestor_path_comparison(root.clone(), p.clone(), q.clone());
        assert_eq!(result4.unwrap().borrow().val, 5);
        
        let result5 = Solution::lowest_common_ancestor_bfs(root.clone(), p.clone(), q.clone());
        assert_eq!(result5.unwrap().borrow().val, 5);
        
        let result6 = Solution::lowest_common_ancestor_optimized(root.clone(), p.clone(), q.clone());
        assert_eq!(result6.unwrap().borrow().val, 5);
    }
    
    #[test]
    fn test_simple_tree() {
        // Tree: [1,2]
        //   1
        //  /
        // 2
        let root = build_tree(vec![Some(1), Some(2), None]);
        
        let p = find_node(root.clone(), 1);
        let q = find_node(root.clone(), 2);
        
        // LCA should be 1
        let result1 = Solution::lowest_common_ancestor_recursive(root.clone(), p.clone(), q.clone());
        assert_eq!(result1.unwrap().borrow().val, 1);
        
        let result2 = Solution::lowest_common_ancestor_parent_pointers(root.clone(), p.clone(), q.clone());
        assert_eq!(result2.unwrap().borrow().val, 1);
        
        let result3 = Solution::lowest_common_ancestor_iterative(root.clone(), p.clone(), q.clone());
        assert_eq!(result3.unwrap().borrow().val, 1);
        
        let result4 = Solution::lowest_common_ancestor_path_comparison(root.clone(), p.clone(), q.clone());
        assert_eq!(result4.unwrap().borrow().val, 1);
        
        let result5 = Solution::lowest_common_ancestor_bfs(root.clone(), p.clone(), q.clone());
        assert_eq!(result5.unwrap().borrow().val, 1);
        
        let result6 = Solution::lowest_common_ancestor_optimized(root.clone(), p.clone(), q.clone());
        assert_eq!(result6.unwrap().borrow().val, 1);
    }
    
    #[test]
    fn test_single_node_tree() {
        // Tree: [1]
        let root = build_tree(vec![Some(1)]);
        
        let p = find_node(root.clone(), 1);
        let q = find_node(root.clone(), 1);
        
        // Test each approach individually to identify the failing one
        
        // LCA should be 1 (same node)
        let result1 = Solution::lowest_common_ancestor_recursive(root.clone(), p.clone(), q.clone());
        assert!(result1.is_some(), "Recursive approach returned None");
        assert_eq!(result1.unwrap().borrow().val, 1);
        
        let result2 = Solution::lowest_common_ancestor_parent_pointers(root.clone(), p.clone(), q.clone());
        assert!(result2.is_some(), "Parent pointers approach returned None");
        assert_eq!(result2.unwrap().borrow().val, 1);
        
        let result3 = Solution::lowest_common_ancestor_iterative(root.clone(), p.clone(), q.clone());
        assert!(result3.is_some(), "Iterative approach returned None");
        assert_eq!(result3.unwrap().borrow().val, 1);
        
        let result4 = Solution::lowest_common_ancestor_path_comparison(root.clone(), p.clone(), q.clone());
        assert!(result4.is_some(), "Path comparison approach returned None");
        assert_eq!(result4.unwrap().borrow().val, 1);
        
        let result5 = Solution::lowest_common_ancestor_bfs(root.clone(), p.clone(), q.clone());
        assert!(result5.is_some(), "BFS approach returned None");
        assert_eq!(result5.unwrap().borrow().val, 1);
        
        let result6 = Solution::lowest_common_ancestor_optimized(root.clone(), p.clone(), q.clone());
        assert!(result6.is_some(), "Optimized approach returned None");
        assert_eq!(result6.unwrap().borrow().val, 1);
    }
    
    #[test]
    fn test_left_skewed_tree() {
        // Tree: [1,2,null,3,null,4]
        //     1
        //    /
        //   2
        //  /
        // 3
        //  /
        // 4
        let root = build_tree(vec![Some(1), Some(2), None, Some(3), None, Some(4), None]);
        
        let p = find_node(root.clone(), 3);
        let q = find_node(root.clone(), 4);
        
        // LCA should be 3
        let result1 = Solution::lowest_common_ancestor_recursive(root.clone(), p.clone(), q.clone());
        assert_eq!(result1.unwrap().borrow().val, 3);
        
        let result2 = Solution::lowest_common_ancestor_parent_pointers(root.clone(), p.clone(), q.clone());
        assert_eq!(result2.unwrap().borrow().val, 3);
        
        let result3 = Solution::lowest_common_ancestor_iterative(root.clone(), p.clone(), q.clone());
        assert_eq!(result3.unwrap().borrow().val, 3);
        
        let result4 = Solution::lowest_common_ancestor_path_comparison(root.clone(), p.clone(), q.clone());
        assert_eq!(result4.unwrap().borrow().val, 3);
        
        let result5 = Solution::lowest_common_ancestor_bfs(root.clone(), p.clone(), q.clone());
        assert_eq!(result5.unwrap().borrow().val, 3);
        
        let result6 = Solution::lowest_common_ancestor_optimized(root.clone(), p.clone(), q.clone());
        assert_eq!(result6.unwrap().borrow().val, 3);
    }
    
    #[test]
    fn test_right_skewed_tree() {
        // Tree: [1,null,2,null,3,null,4]
        // 1
        //  \
        //   2
        //    \
        //     3
        //      \
        //       4
        let root = build_tree(vec![Some(1), None, Some(2), None, Some(3), None, Some(4)]);
        
        let p = find_node(root.clone(), 2);
        let q = find_node(root.clone(), 4);
        
        // LCA should be 2
        let result1 = Solution::lowest_common_ancestor_recursive(root.clone(), p.clone(), q.clone());
        assert_eq!(result1.unwrap().borrow().val, 2);
        
        let result2 = Solution::lowest_common_ancestor_parent_pointers(root.clone(), p.clone(), q.clone());
        assert_eq!(result2.unwrap().borrow().val, 2);
        
        let result3 = Solution::lowest_common_ancestor_iterative(root.clone(), p.clone(), q.clone());
        assert_eq!(result3.unwrap().borrow().val, 2);
        
        let result4 = Solution::lowest_common_ancestor_path_comparison(root.clone(), p.clone(), q.clone());
        assert_eq!(result4.unwrap().borrow().val, 2);
        
        let result5 = Solution::lowest_common_ancestor_bfs(root.clone(), p.clone(), q.clone());
        assert_eq!(result5.unwrap().borrow().val, 2);
        
        let result6 = Solution::lowest_common_ancestor_optimized(root.clone(), p.clone(), q.clone());
        assert_eq!(result6.unwrap().borrow().val, 2);
    }
    
    #[test]
    fn test_balanced_tree() {
        // Tree: [1,2,3,4,5,6,7]
        //       1
        //      / \
        //     2   3
        //    / \ / \
        //   4  5 6  7
        let root = build_tree(vec![
            Some(1), Some(2), Some(3), Some(4), Some(5), Some(6), Some(7)
        ]);
        
        let p = find_node(root.clone(), 4);
        let q = find_node(root.clone(), 5);
        
        // LCA should be 2
        let result1 = Solution::lowest_common_ancestor_recursive(root.clone(), p.clone(), q.clone());
        assert_eq!(result1.unwrap().borrow().val, 2);
        
        let result2 = Solution::lowest_common_ancestor_parent_pointers(root.clone(), p.clone(), q.clone());
        assert_eq!(result2.unwrap().borrow().val, 2);
        
        let result3 = Solution::lowest_common_ancestor_iterative(root.clone(), p.clone(), q.clone());
        assert_eq!(result3.unwrap().borrow().val, 2);
        
        let result4 = Solution::lowest_common_ancestor_path_comparison(root.clone(), p.clone(), q.clone());
        assert_eq!(result4.unwrap().borrow().val, 2);
        
        let result5 = Solution::lowest_common_ancestor_bfs(root.clone(), p.clone(), q.clone());
        assert_eq!(result5.unwrap().borrow().val, 2);
        
        let result6 = Solution::lowest_common_ancestor_optimized(root.clone(), p.clone(), q.clone());
        assert_eq!(result6.unwrap().borrow().val, 2);
    }
    
    #[test]
    fn test_cross_subtree_lca() {
        // Tree: [1,2,3,4,5,6,7]
        let root = build_tree(vec![
            Some(1), Some(2), Some(3), Some(4), Some(5), Some(6), Some(7)
        ]);
        
        let p = find_node(root.clone(), 4);  // Left subtree
        let q = find_node(root.clone(), 7);  // Right subtree
        
        // LCA should be 1 (root)
        let result1 = Solution::lowest_common_ancestor_recursive(root.clone(), p.clone(), q.clone());
        assert_eq!(result1.unwrap().borrow().val, 1);
        
        let result2 = Solution::lowest_common_ancestor_parent_pointers(root.clone(), p.clone(), q.clone());
        assert_eq!(result2.unwrap().borrow().val, 1);
        
        let result3 = Solution::lowest_common_ancestor_iterative(root.clone(), p.clone(), q.clone());
        assert_eq!(result3.unwrap().borrow().val, 1);
        
        let result4 = Solution::lowest_common_ancestor_path_comparison(root.clone(), p.clone(), q.clone());
        assert_eq!(result4.unwrap().borrow().val, 1);
        
        let result5 = Solution::lowest_common_ancestor_bfs(root.clone(), p.clone(), q.clone());
        assert_eq!(result5.unwrap().borrow().val, 1);
        
        let result6 = Solution::lowest_common_ancestor_optimized(root.clone(), p.clone(), q.clone());
        assert_eq!(result6.unwrap().borrow().val, 1);
    }
    
    #[test]
    fn test_large_tree_performance() {
        // Build a larger tree for performance testing
        let mut values = vec![Some(1)];
        for i in 2..=31 {
            values.push(Some(i));
        }
        
        let root = build_tree(values);
        let p = find_node(root.clone(), 15);
        let q = find_node(root.clone(), 31);
        
        // Test that all approaches work on larger trees
        let result1 = Solution::lowest_common_ancestor_recursive(root.clone(), p.clone(), q.clone());
        let result2 = Solution::lowest_common_ancestor_parent_pointers(root.clone(), p.clone(), q.clone());
        let result3 = Solution::lowest_common_ancestor_iterative(root.clone(), p.clone(), q.clone());
        let result4 = Solution::lowest_common_ancestor_path_comparison(root.clone(), p.clone(), q.clone());
        let result5 = Solution::lowest_common_ancestor_bfs(root.clone(), p.clone(), q.clone());
        let result6 = Solution::lowest_common_ancestor_optimized(root.clone(), p.clone(), q.clone());
        
        // All should return same result
        let expected = result1.as_ref().unwrap().borrow().val;
        assert_eq!(result2.unwrap().borrow().val, expected);
        assert_eq!(result3.unwrap().borrow().val, expected);
        assert_eq!(result4.unwrap().borrow().val, expected);
        assert_eq!(result5.unwrap().borrow().val, expected);
        assert_eq!(result6.unwrap().borrow().val, expected);
    }
    
    #[test]
    fn test_edge_case_deep_nodes() {
        // Tree with specific structure for edge case testing
        let root = build_tree(vec![
            Some(3), Some(5), Some(1), Some(6), Some(2), Some(0), Some(8),
            None, None, Some(7), Some(4), None, None, None, None
        ]);
        
        let p = find_node(root.clone(), 7);
        let q = find_node(root.clone(), 4);
        
        // LCA should be 2
        let result1 = Solution::lowest_common_ancestor_recursive(root.clone(), p.clone(), q.clone());
        assert_eq!(result1.unwrap().borrow().val, 2);
        
        let result2 = Solution::lowest_common_ancestor_parent_pointers(root.clone(), p.clone(), q.clone());
        assert_eq!(result2.unwrap().borrow().val, 2);
        
        let result3 = Solution::lowest_common_ancestor_iterative(root.clone(), p.clone(), q.clone());
        assert_eq!(result3.unwrap().borrow().val, 2);
        
        let result4 = Solution::lowest_common_ancestor_path_comparison(root.clone(), p.clone(), q.clone());
        assert_eq!(result4.unwrap().borrow().val, 2);
        
        let result5 = Solution::lowest_common_ancestor_bfs(root.clone(), p.clone(), q.clone());
        assert_eq!(result5.unwrap().borrow().val, 2);
        
        let result6 = Solution::lowest_common_ancestor_optimized(root.clone(), p.clone(), q.clone());
        assert_eq!(result6.unwrap().borrow().val, 2);
    }
    
    #[test]
    fn test_negative_values() {
        // Tree with negative values: [-1,-2,-3]
        //     -1
        //    /  \
        //  -2   -3
        let root = build_tree(vec![Some(-1), Some(-2), Some(-3)]);
        
        let p = find_node(root.clone(), -2);
        let q = find_node(root.clone(), -3);
        
        // LCA should be -1
        let result1 = Solution::lowest_common_ancestor_recursive(root.clone(), p.clone(), q.clone());
        assert_eq!(result1.unwrap().borrow().val, -1);
        
        let result2 = Solution::lowest_common_ancestor_parent_pointers(root.clone(), p.clone(), q.clone());
        assert_eq!(result2.unwrap().borrow().val, -1);
        
        let result3 = Solution::lowest_common_ancestor_iterative(root.clone(), p.clone(), q.clone());
        assert_eq!(result3.unwrap().borrow().val, -1);
        
        let result4 = Solution::lowest_common_ancestor_path_comparison(root.clone(), p.clone(), q.clone());
        assert_eq!(result4.unwrap().borrow().val, -1);
        
        let result5 = Solution::lowest_common_ancestor_bfs(root.clone(), p.clone(), q.clone());
        assert_eq!(result5.unwrap().borrow().val, -1);
        
        let result6 = Solution::lowest_common_ancestor_optimized(root.clone(), p.clone(), q.clone());
        assert_eq!(result6.unwrap().borrow().val, -1);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        // Test multiple tree configurations to ensure all approaches give same results
        let test_cases = vec![
            (vec![Some(1), Some(2), Some(3)], 2, 3, 1),
            (vec![Some(1), Some(2), None], 1, 2, 1),
            (vec![Some(1), None, Some(2)], 1, 2, 1),
            (vec![Some(1), Some(2), Some(3), Some(4), Some(5)], 4, 5, 2),
            (vec![Some(1), Some(2), Some(3), Some(4), Some(5)], 2, 3, 1),
        ];
        
        for (tree_vals, p_val, q_val, expected_lca) in test_cases {
            let root = build_tree(tree_vals);
            let p = find_node(root.clone(), p_val);
            let q = find_node(root.clone(), q_val);
            
            let result1 = Solution::lowest_common_ancestor_recursive(root.clone(), p.clone(), q.clone());
            let result2 = Solution::lowest_common_ancestor_parent_pointers(root.clone(), p.clone(), q.clone());
            let result3 = Solution::lowest_common_ancestor_iterative(root.clone(), p.clone(), q.clone());
            let result4 = Solution::lowest_common_ancestor_path_comparison(root.clone(), p.clone(), q.clone());
            let result5 = Solution::lowest_common_ancestor_bfs(root.clone(), p.clone(), q.clone());
            let result6 = Solution::lowest_common_ancestor_optimized(root.clone(), p.clone(), q.clone());
            
            assert_eq!(result1.unwrap().borrow().val, expected_lca);
            assert_eq!(result2.unwrap().borrow().val, expected_lca);
            assert_eq!(result3.unwrap().borrow().val, expected_lca);
            assert_eq!(result4.unwrap().borrow().val, expected_lca);
            assert_eq!(result5.unwrap().borrow().val, expected_lca);
            assert_eq!(result6.unwrap().borrow().val, expected_lca);
        }
    }
}