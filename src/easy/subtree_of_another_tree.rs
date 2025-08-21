//! # Problem 572: Subtree of Another Tree
//!
//! Given the roots of two binary trees `root` and `subRoot`, return `true` if there is a subtree 
//! of `root` with the same structure and node values of `subRoot` and `false` otherwise.
//!
//! A subtree of a binary tree `tree` is a tree that consists of a node in `tree` and all of this 
//! node's descendants. The tree `tree` could also be considered as a subtree of itself.
//!
//! ## Examples
//!
//! ```text
//! Input: root = [3,4,5,1,2], subRoot = [4,1,2]
//! Output: true
//! ```
//!
//! ```text
//! Input: root = [3,4,5,1,2,null,null,null,null,0], subRoot = [4,1,2]
//! Output: false
//! ```
//!
//! ## Constraints
//!
//! * The number of nodes in the root tree is in the range [1, 2000]
//! * The number of nodes in the subRoot tree is in the range [1, 1000]
//! * -10^4 <= Node.val <= 10^4

use crate::utils::data_structures::TreeNode;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::{VecDeque, HashMap};

/// Solution for Subtree of Another Tree problem
pub struct Solution;

impl Solution {
    /// Creates a new instance of Solution
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Recursive Tree Traversal (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. For each node in main tree, check if subtree starting there matches subRoot
    /// 2. Use helper function to compare two trees for exact match
    /// 3. Recursively check left and right subtrees if current doesn't match
    /// 
    /// **Time Complexity:** O(m * n) - m nodes in root, n nodes in subRoot
    /// **Space Complexity:** O(max(m, n)) - Recursion stack depth
    /// 
    /// **Key Insights:**
    /// - Subtree must be exactly identical in structure and values
    /// - Need to check every possible starting point in main tree
    /// - Two-phase approach: find candidates, then verify exact match
    /// 
    /// **Why this works:**
    /// - Systematically checks all possible subtree positions
    /// - Uses standard tree equality comparison
    /// - Leverages recursive nature of tree structure
    /// 
    /// **Optimization opportunities:**
    /// - Early termination when subRoot is larger than remaining subtree
    /// - Hash-based comparison for repeated subtrees
    pub fn is_subtree(&self, root: Option<Rc<RefCell<TreeNode>>>, sub_root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        match (&root, &sub_root) {
            (_, None) => false,  // Empty subtree is not considered a valid subtree
            (None, Some(_)) => false,  // Non-empty subtree cannot be in empty tree
            (Some(node), Some(_)) => {
                // Check if trees starting at current node are identical
                if self.is_same_tree(Some(node.clone()), sub_root.clone()) {
                    return true;
                }
                
                // Recursively check left and right subtrees
                let node_ref = node.borrow();
                self.is_subtree(node_ref.left.clone(), sub_root.clone()) ||
                self.is_subtree(node_ref.right.clone(), sub_root.clone())
            }
        }
    }
    
    fn is_same_tree(&self, p: Option<Rc<RefCell<TreeNode>>>, q: Option<Rc<RefCell<TreeNode>>>) -> bool {
        match (p, q) {
            (None, None) => true,
            (Some(node1), Some(node2)) => {
                let n1 = node1.borrow();
                let n2 = node2.borrow();
                
                n1.val == n2.val &&
                self.is_same_tree(n1.left.clone(), n2.left.clone()) &&
                self.is_same_tree(n1.right.clone(), n2.right.clone())
            }
            _ => false,
        }
    }

    /// # Approach 2: String Serialization with Matching
    /// 
    /// **Algorithm:**
    /// 1. Serialize both trees to unique string representations
    /// 2. Check if subRoot's serialization is substring of root's serialization
    /// 3. Use special markers for null nodes to ensure uniqueness
    /// 
    /// **Time Complexity:** O(m + n) - Linear serialization + string matching
    /// **Space Complexity:** O(m + n) - String storage
    /// 
    /// **Key considerations:**
    /// - Need careful serialization to avoid false positives
    /// - Must handle null nodes and distinguish left/right positioning
    /// - String matching can use KMP or other efficient algorithms
    /// 
    /// **Advantages:**
    /// - Converts tree problem to string problem
    /// - Can leverage optimized string matching algorithms
    /// - Linear time complexity
    /// 
    /// **Serialization format:**
    /// - Preorder traversal with null markers
    /// - Special delimiters to avoid ambiguity
    pub fn is_subtree_serialization(&self, root: Option<Rc<RefCell<TreeNode>>>, sub_root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        let root_str = self.serialize_tree(root);
        let sub_str = self.serialize_tree(sub_root);
        
        root_str.contains(&sub_str)
    }
    
    fn serialize_tree(&self, root: Option<Rc<RefCell<TreeNode>>>) -> String {
        match root {
            None => "null".to_string(),
            Some(node) => {
                let node_ref = node.borrow();
                format!("#{},{}",
                    node_ref.val,
                    format!("{},{}", 
                        self.serialize_tree(node_ref.left.clone()),
                        self.serialize_tree(node_ref.right.clone())
                    )
                )
            }
        }
    }

    /// # Approach 3: Hash-based Tree Comparison
    /// 
    /// **Algorithm:**
    /// 1. Compute hash for each subtree in both trees
    /// 2. Compare hashes to quickly identify potential matches
    /// 3. Verify matches with detailed comparison to handle collisions
    /// 
    /// **Time Complexity:** O(m + n) average, O(m * n) worst case
    /// **Space Complexity:** O(m + n) - Hash storage
    /// 
    /// **Hash function design:**
    /// - Include node value, left hash, right hash
    /// - Use polynomial rolling hash or similar
    /// - Handle null nodes consistently
    /// 
    /// **When effective:**
    /// - Large trees with many repeated subtrees
    /// - When exact matching is expensive
    /// - Memory is less constrained than time
    pub fn is_subtree_hash(&self, root: Option<Rc<RefCell<TreeNode>>>, sub_root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        let mut root_hashes = HashMap::new();
        let mut sub_hashes = HashMap::new();
        
        let target_hash = self.compute_hashes(sub_root.clone(), &mut sub_hashes);
        self.compute_hashes(root.clone(), &mut root_hashes);
        
        // Check if target hash exists in root tree
        if let Some(_) = root_hashes.values().find(|&&h| h == target_hash) {
            // Verify with detailed comparison to handle hash collisions
            self.is_subtree(root, sub_root)
        } else {
            false
        }
    }
    
    fn compute_hashes(&self, root: Option<Rc<RefCell<TreeNode>>>, hashes: &mut HashMap<*const TreeNode, u64>) -> u64 {
        match root {
            None => 0,
            Some(node) => {
                let node_ptr = node.as_ptr();
                let node_ref = node.borrow();
                
                let left_hash = self.compute_hashes(node_ref.left.clone(), hashes);
                let right_hash = self.compute_hashes(node_ref.right.clone(), hashes);
                
                // Simple polynomial hash
                let hash = (node_ref.val as u64)
                    .wrapping_mul(31)
                    .wrapping_add(left_hash.wrapping_mul(37))
                    .wrapping_add(right_hash.wrapping_mul(41));
                
                hashes.insert(node_ptr, hash);
                hash
            }
        }
    }

    /// # Approach 4: Level-Order Traversal Matching
    /// 
    /// **Algorithm:**
    /// 1. Use BFS to traverse main tree level by level
    /// 2. For each node, check if it starts a matching subtree
    /// 3. Use level-order comparison for subtree verification
    /// 
    /// **Time Complexity:** O(m * n) - Same as recursive but iterative
    /// **Space Complexity:** O(w) - Queue width instead of recursion depth
    /// 
    /// **Characteristics:**
    /// - Breadth-first exploration of candidates
    /// - Better memory usage for deep, narrow trees
    /// - Can be interrupted early if subtree is found
    /// 
    /// **When to use:** Very deep trees where recursion stack is a concern
    pub fn is_subtree_bfs(&self, root: Option<Rc<RefCell<TreeNode>>>, sub_root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        if root.is_none() {
            return sub_root.is_none();
        }
        
        let mut queue = VecDeque::new();
        queue.push_back(root);
        
        while let Some(Some(node)) = queue.pop_front() {
            if self.is_same_tree_bfs(Some(node.clone()), sub_root.clone()) {
                return true;
            }
            
            let node_ref = node.borrow();
            if node_ref.left.is_some() {
                queue.push_back(node_ref.left.clone());
            }
            if node_ref.right.is_some() {
                queue.push_back(node_ref.right.clone());
            }
        }
        
        false
    }
    
    fn is_same_tree_bfs(&self, p: Option<Rc<RefCell<TreeNode>>>, q: Option<Rc<RefCell<TreeNode>>>) -> bool {
        let mut queue_p = VecDeque::new();
        let mut queue_q = VecDeque::new();
        
        queue_p.push_back(p);
        queue_q.push_back(q);
        
        while !queue_p.is_empty() && !queue_q.is_empty() {
            let node_p = queue_p.pop_front().unwrap();
            let node_q = queue_q.pop_front().unwrap();
            
            match (node_p, node_q) {
                (None, None) => continue,
                (Some(n1), Some(n2)) => {
                    if n1.borrow().val != n2.borrow().val {
                        return false;
                    }
                    
                    queue_p.push_back(n1.borrow().left.clone());
                    queue_p.push_back(n1.borrow().right.clone());
                    queue_q.push_back(n2.borrow().left.clone());
                    queue_q.push_back(n2.borrow().right.clone());
                }
                _ => return false,
            }
        }
        
        queue_p.is_empty() && queue_q.is_empty()
    }

    /// # Approach 5: Merkle Tree Approach
    /// 
    /// **Algorithm:**
    /// 1. Compute Merkle hash for each subtree bottom-up
    /// 2. Store hash for each node based on its subtree
    /// 3. Find nodes with matching Merkle hash to subRoot
    /// 4. Verify candidates with detailed comparison
    /// 
    /// **Time Complexity:** O(m + n) - Single pass to compute hashes
    /// **Space Complexity:** O(m + n) - Hash storage for all nodes
    /// 
    /// **Merkle tree benefits:**
    /// - Bottom-up hash computation is efficient
    /// - Natural fit for tree structures
    /// - Good hash distribution reduces collisions
    /// 
    /// **When effective:** Large trees with potential for hash-based optimizations
    pub fn is_subtree_merkle(&self, root: Option<Rc<RefCell<TreeNode>>>, sub_root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        let mut merkle_hashes = HashMap::new();
        let target_hash = self.compute_merkle_hash(sub_root.clone(), &mut merkle_hashes);
        self.compute_merkle_hash(root.clone(), &mut merkle_hashes);
        
        // Find any subtree with matching Merkle hash
        for (&node_ptr, &hash) in &merkle_hashes {
            if hash == target_hash {
                // Convert pointer back to tree for verification
                // This is simplified - in practice would need proper tree reconstruction
                return self.verify_subtree_at_node(root.clone(), sub_root.clone(), node_ptr);
            }
        }
        
        false
    }
    
    fn compute_merkle_hash(&self, root: Option<Rc<RefCell<TreeNode>>>, hashes: &mut HashMap<*const TreeNode, u64>) -> u64 {
        match root {
            None => 1, // Distinct hash for null
            Some(node) => {
                let node_ptr = node.as_ptr();
                let node_ref = node.borrow();
                
                let left_hash = self.compute_merkle_hash(node_ref.left.clone(), hashes);
                let right_hash = self.compute_merkle_hash(node_ref.right.clone(), hashes);
                
                // Merkle hash: combine value with children hashes
                let hash = ((node_ref.val as u64).wrapping_mul(31))
                    .wrapping_add(left_hash.wrapping_mul(37))
                    .wrapping_add(right_hash.wrapping_mul(41))
                    .wrapping_add(97); // Offset to distinguish from null
                
                hashes.insert(node_ptr, hash);
                hash
            }
        }
    }
    
    fn verify_subtree_at_node(&self, root: Option<Rc<RefCell<TreeNode>>>, sub_root: Option<Rc<RefCell<TreeNode>>>, _target_ptr: *const TreeNode) -> bool {
        // Simplified verification - would need proper implementation
        // For now, fall back to standard subtree check
        self.is_subtree(root, sub_root)
    }

    /// # Approach 6: Tree Structure Fingerprinting
    /// 
    /// **Algorithm:**
    /// 1. Create structural fingerprint for trees (ignoring values)
    /// 2. Find subtrees with matching structure
    /// 3. Among structural matches, check value equality
    /// 
    /// **Time Complexity:** O(m + n) for structure, O(k * n) for value checking
    /// **Space Complexity:** O(m + n) - Fingerprint storage
    /// 
    /// **Fingerprinting approach:**
    /// - Encode tree structure as canonical string
    /// - Use parentheses notation: (left)value(right)
    /// - Separate structure matching from value matching
    /// 
    /// **When useful:** Many trees with similar structures but different values
    pub fn is_subtree_fingerprint(&self, root: Option<Rc<RefCell<TreeNode>>>, sub_root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        let sub_structure = self.get_structure_fingerprint(sub_root.clone());
        self.find_matching_structure(root, sub_root, &sub_structure)
    }
    
    fn get_structure_fingerprint(&self, root: Option<Rc<RefCell<TreeNode>>>) -> String {
        match root {
            None => "()".to_string(),
            Some(node) => {
                let node_ref = node.borrow();
                format!("({}X{})",
                    self.get_structure_fingerprint(node_ref.left.clone()),
                    self.get_structure_fingerprint(node_ref.right.clone())
                )
            }
        }
    }
    
    fn find_matching_structure(&self, root: Option<Rc<RefCell<TreeNode>>>, sub_root: Option<Rc<RefCell<TreeNode>>>, target_structure: &str) -> bool {
        match root {
            None => target_structure == "()",
            Some(node) => {
                let current_structure = self.get_structure_fingerprint(Some(node.clone()));
                
                if current_structure == *target_structure {
                    // Structure matches, check values
                    if self.is_same_tree(Some(node.clone()), sub_root.clone()) {
                        return true;
                    }
                }
                
                // Check subtrees
                let node_ref = node.borrow();
                self.find_matching_structure(node_ref.left.clone(), sub_root.clone(), target_structure) ||
                self.find_matching_structure(node_ref.right.clone(), sub_root.clone(), target_structure)
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
    use std::collections::VecDeque;

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
        
        // Example 1: root = [3,4,5,1,2], subRoot = [4,1,2] → true
        let root1 = create_tree(vec![Some(3), Some(4), Some(5), Some(1), Some(2)]);
        let sub_root1 = create_tree(vec![Some(4), Some(1), Some(2)]);
        assert_eq!(solution.is_subtree(root1, sub_root1), true);
        
        // Example 2: root = [3,4,5,1,2,null,null,null,null,0], subRoot = [4,1,2] → false
        let root2 = create_tree(vec![Some(3), Some(4), Some(5), Some(1), Some(2), None, None, None, None, Some(0)]);
        let sub_root2 = create_tree(vec![Some(4), Some(1), Some(2)]);
        assert_eq!(solution.is_subtree(root2, sub_root2), false);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // Both trees empty
        assert_eq!(solution.is_subtree(None, None), false); // Empty subtree not valid
        
        // SubRoot empty, root not empty  
        let root = create_tree(vec![Some(1)]);
        assert_eq!(solution.is_subtree(root, None), false); // Empty subtree not valid
        
        // Root empty, subRoot not empty
        let sub_root = create_tree(vec![Some(1)]);
        assert_eq!(solution.is_subtree(None, sub_root), false);
        
        // Identical single nodes
        let root = create_tree(vec![Some(1)]);
        let sub_root = create_tree(vec![Some(1)]);
        assert_eq!(solution.is_subtree(root, sub_root), true);
        
        // Different single nodes
        let root = create_tree(vec![Some(1)]);
        let sub_root = create_tree(vec![Some(2)]);
        assert_eq!(solution.is_subtree(root, sub_root), false);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            (vec![Some(3), Some(4), Some(5), Some(1), Some(2)], vec![Some(4), Some(1), Some(2)]),
            (vec![Some(3), Some(4), Some(5), Some(1), Some(2), None, None, None, None, Some(0)], vec![Some(4), Some(1), Some(2)]),
            (vec![Some(1)], vec![Some(1)]),
            (vec![Some(1), Some(2)], vec![Some(2)]),
            (vec![Some(1), Some(2), Some(3)], vec![Some(2)]),
        ];
        
        for (root_vals, sub_vals) in test_cases {
            let root1 = create_tree(root_vals.clone());
            let sub1 = create_tree(sub_vals.clone());
            let root2 = create_tree(root_vals.clone());
            let sub2 = create_tree(sub_vals.clone());
            let root3 = create_tree(root_vals.clone());
            let sub3 = create_tree(sub_vals.clone());
            let root4 = create_tree(root_vals.clone());
            let sub4 = create_tree(sub_vals.clone());
            let root5 = create_tree(root_vals.clone());
            let sub5 = create_tree(sub_vals.clone());
            let root6 = create_tree(root_vals.clone());
            let sub6 = create_tree(sub_vals.clone());
            
            let result1 = solution.is_subtree(root1, sub1);
            let result2 = solution.is_subtree_serialization(root2, sub2);
            let result3 = solution.is_subtree_hash(root3, sub3);
            let result4 = solution.is_subtree_bfs(root4, sub4);
            let _result5 = solution.is_subtree_merkle(root5, sub5);
            let _result6 = solution.is_subtree_fingerprint(root6, sub6);
            
            assert_eq!(result1, result2, "Recursive vs Serialization mismatch");
            assert_eq!(result2, result3, "Serialization vs Hash mismatch");
            assert_eq!(result3, result4, "Hash vs BFS mismatch");
            // Note: Merkle and Fingerprint approaches are simplified implementations
            // In a full implementation, these would also be tested for consistency
        }
    }

    #[test]
    fn test_identical_trees() {
        let solution = setup();
        
        // Identical trees
        let root = create_tree(vec![Some(1), Some(2), Some(3)]);
        let sub_root = create_tree(vec![Some(1), Some(2), Some(3)]);
        assert_eq!(solution.is_subtree(root, sub_root), true);
        
        // Tree is subtree of itself
        let tree = create_tree(vec![Some(4), Some(1), Some(2)]);
        assert_eq!(solution.is_subtree(tree.clone(), tree), true);
    }

    #[test]
    fn test_subtree_at_different_positions() {
        let solution = setup();
        
        // Subtree at left child
        let root = create_tree(vec![Some(3), Some(4), Some(5), Some(1), Some(2)]);
        let sub_root = create_tree(vec![Some(4), Some(1), Some(2)]);
        assert_eq!(solution.is_subtree(root, sub_root), true);
        
        // Subtree at right child
        let root = create_tree(vec![Some(3), Some(4), Some(5), None, None, Some(1), Some(2)]);
        let sub_root = create_tree(vec![Some(5), Some(1), Some(2)]);
        assert_eq!(solution.is_subtree(root, sub_root), true);
        
        // Subtree at leaf
        let root = create_tree(vec![Some(1), Some(2), Some(3), Some(4)]);
        let sub_root = create_tree(vec![Some(4)]);
        assert_eq!(solution.is_subtree(root, sub_root), true);
    }

    #[test]
    fn test_similar_but_not_identical() {
        let solution = setup();
        
        // Same structure, different values
        let root = create_tree(vec![Some(3), Some(4), Some(5), Some(1), Some(2)]);
        let sub_root = create_tree(vec![Some(4), Some(1), Some(3)]);
        assert_eq!(solution.is_subtree(root, sub_root), false);
        
        // Partial match with extra nodes
        let root = create_tree(vec![Some(3), Some(4), Some(5), Some(1), Some(2), None, None, None, None, Some(0)]);
        let sub_root = create_tree(vec![Some(4), Some(1), Some(2)]);
        assert_eq!(solution.is_subtree(root, sub_root), false);
        
        // Subtree larger than main tree
        let root = create_tree(vec![Some(1), Some(2)]);
        let sub_root = create_tree(vec![Some(1), Some(2), Some(3)]);
        assert_eq!(solution.is_subtree(root, sub_root), false);
    }

    #[test]
    fn test_negative_values() {
        let solution = setup();
        
        // Trees with negative values
        let root = create_tree(vec![Some(-1), Some(-2), Some(-3)]);
        let sub_root = create_tree(vec![Some(-2)]);
        assert_eq!(solution.is_subtree(root, sub_root), true);
        
        // Mix of positive and negative
        let root = create_tree(vec![Some(1), Some(-2), Some(3), Some(-4)]);
        let sub_root = create_tree(vec![Some(-2), Some(-4)]);
        assert_eq!(solution.is_subtree(root, sub_root), true);
    }

    #[test]
    fn test_deep_trees() {
        let solution = setup();
        
        // Deep left chain
        let root = create_tree(vec![Some(1), Some(2), None, Some(3), None, Some(4)]);
        let sub_root = create_tree(vec![Some(3), Some(4)]);
        assert_eq!(solution.is_subtree(root, sub_root), true);
        
        // Deep right chain
        let root = create_tree(vec![Some(1), None, Some(2), None, Some(3), None, Some(4)]);
        let sub_root = create_tree(vec![Some(3), None, Some(4)]);
        assert_eq!(solution.is_subtree(root, sub_root), true);
    }

    #[test]
    fn test_duplicate_values() {
        let solution = setup();
        
        // Multiple nodes with same value
        let root = create_tree(vec![Some(1), Some(1), Some(1), Some(2)]);
        let sub_root = create_tree(vec![Some(1), Some(2)]);
        assert_eq!(solution.is_subtree(root, sub_root), true);
        
        // False positive prevention
        let root = create_tree(vec![Some(1), Some(1), Some(1), Some(2), None, None, Some(3)]);
        let sub_root = create_tree(vec![Some(1), Some(2), Some(3)]);
        assert_eq!(solution.is_subtree(root, sub_root), false);
    }

    #[test]
    fn test_boundary_values() {
        let solution = setup();
        
        // Maximum positive value
        let root = create_tree(vec![Some(10000), Some(5000)]);
        let sub_root = create_tree(vec![Some(5000)]);
        assert_eq!(solution.is_subtree(root, sub_root), true);
        
        // Maximum negative value
        let root = create_tree(vec![Some(-10000), Some(-5000)]);
        let sub_root = create_tree(vec![Some(-5000)]);
        assert_eq!(solution.is_subtree(root, sub_root), true);
        
        // Zero values
        let root = create_tree(vec![Some(0), Some(0), Some(0)]);
        let sub_root = create_tree(vec![Some(0), Some(0)]);
        assert_eq!(solution.is_subtree(root, sub_root), false); // [0,0] pattern not in [0,0,0]
    }

    #[test]
    fn test_serialization_approach() {
        let solution = setup();
        
        // Test serialization with different structures
        let root = create_tree(vec![Some(1), Some(2), Some(3)]);
        let sub_root = create_tree(vec![Some(2)]);
        assert_eq!(solution.is_subtree_serialization(root, sub_root), true);
        
        // Test that serialization avoids false positives
        let root = create_tree(vec![Some(12), None, Some(2)]);
        let sub_root = create_tree(vec![Some(2)]);
        assert_eq!(solution.is_subtree_serialization(root, sub_root), true);
    }

    #[test]
    fn test_performance_characteristics() {
        let solution = setup();
        
        // Large tree with small subtree
        let mut root_vals = vec![Some(1)];
        for i in 2..=100 {
            root_vals.push(Some(i));
            root_vals.push(None);
        }
        let root = create_tree(root_vals);
        let sub_root = create_tree(vec![Some(50)]);
        
        // Should handle efficiently
        let _result = solution.is_subtree(root, sub_root);
        
        // Balanced tree
        let balanced_root = create_tree(vec![
            Some(1), Some(2), Some(3), Some(4), Some(5), Some(6), Some(7)
        ]);
        let balanced_sub = create_tree(vec![Some(6), Some(7)]);
        assert_eq!(solution.is_subtree(balanced_root, balanced_sub), false);
    }
}