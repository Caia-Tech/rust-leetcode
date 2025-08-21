//! Problem 173: Binary Search Tree Iterator
//! 
//! Implement the BSTIterator class that represents an iterator over the in-order traversal 
//! of a binary search tree (BST):
//! 
//! - BSTIterator(TreeNode root) Initializes an object of the BSTIterator class. 
//!   The root of the BST is given as part of the constructor. The pointer should be 
//!   initialized to a non-existent number smaller than any element in the BST.
//! - boolean hasNext() Returns true if there exists a number in the traversal to the right 
//!   of the pointer, otherwise returns false.
//! - int next() Moves the pointer to the right, then returns the number at the pointer.
//! 
//! Notice that by initializing the pointer to a non-existent smallest number, the first call 
//! to next() will return the smallest element in the BST.
//! 
//! You may assume that next() calls will always be valid. That is, there will be at least 
//! a next number in the in-order traversal when next() is called.
//! 
//! Example:
//!     7
//!    / \
//!   3  15
//!     /  \
//!    9   20
//! 
//! Input
//! ["BSTIterator", "next", "next", "hasNext", "next", "hasNext", "next", "hasNext", "next", "hasNext"]
//! [[[7,3,15,null,null,9,20]], [], [], [], [], [], [], [], [], []]
//! Output
//! [null, 3, 7, true, 9, true, 15, true, 20, false]
//! 
//! Constraints:
//! - The number of nodes in the tree is in the range [1, 10^5].
//! - 0 <= Node.val <= 10^6
//! - At most 10^5 calls will be made to hasNext and next.
//! 
//! Follow up:
//! - Could you implement next() and hasNext() in average O(1) time and use only O(h) memory, 
//!   where h is the height of the tree?

use std::rc::Rc;
use std::cell::RefCell;
use crate::utils::data_structures::TreeNode;

/// Approach 1: Flattening with Precomputation
/// 
/// Flattens the entire BST into a sorted array during initialization.
/// Provides O(1) time for both next() and hasNext() operations.
/// 
/// Time Complexity: O(n) for constructor, O(1) for next() and hasNext()
/// Space Complexity: O(n) for storing all values
pub struct BSTIterator {
    values: Vec<i32>,
    index: usize,
}

impl BSTIterator {
    pub fn new(root: Option<Rc<RefCell<TreeNode>>>) -> Self {
        let mut values = Vec::new();
        Self::inorder(root.as_ref(), &mut values);
        Self {
            values,
            index: 0,
        }
    }
    
    fn inorder(node: Option<&Rc<RefCell<TreeNode>>>, values: &mut Vec<i32>) {
        if let Some(node) = node {
            let node = node.borrow();
            Self::inorder(node.left.as_ref(), values);
            values.push(node.val);
            Self::inorder(node.right.as_ref(), values);
        }
    }
    
    pub fn next(&mut self) -> i32 {
        let value = self.values[self.index];
        self.index += 1;
        value
    }
    
    pub fn has_next(&self) -> bool {
        self.index < self.values.len()
    }
}

/// Approach 2: Controlled Recursion with Stack
/// 
/// Uses an explicit stack to simulate inorder traversal.
/// Maintains only the path from root to current node on the stack.
/// 
/// Time Complexity: O(1) amortized for next(), O(1) for hasNext()
/// Space Complexity: O(h) where h is height of the tree
pub struct BSTIteratorStack {
    stack: Vec<Rc<RefCell<TreeNode>>>,
}

impl BSTIteratorStack {
    pub fn new(root: Option<Rc<RefCell<TreeNode>>>) -> Self {
        let mut iterator = Self {
            stack: Vec::new(),
        };
        iterator.push_left(root);
        iterator
    }
    
    fn push_left(&mut self, mut node: Option<Rc<RefCell<TreeNode>>>) {
        while let Some(n) = node {
            let left = n.borrow().left.clone();
            self.stack.push(n);
            node = left;
        }
    }
    
    pub fn next(&mut self) -> i32 {
        let node = self.stack.pop().unwrap();
        let result = node.borrow().val;
        let right = node.borrow().right.clone();
        self.push_left(right);
        result
    }
    
    pub fn has_next(&self) -> bool {
        !self.stack.is_empty()
    }
}

/// Approach 3: Parent Pointer Simulation
/// 
/// Simulates parent pointers by maintaining current position and finding
/// the next node without explicit parent references.
/// 
/// Time Complexity: O(h) worst case for next(), O(1) for hasNext()
/// Space Complexity: O(1) excluding the tree itself
pub struct BSTIteratorParent {
    root: Option<Rc<RefCell<TreeNode>>>,
    current: Option<Rc<RefCell<TreeNode>>>,
    next_val: Option<i32>,
}

impl BSTIteratorParent {
    pub fn new(root: Option<Rc<RefCell<TreeNode>>>) -> Self {
        let mut iterator = Self {
            root: root.clone(),
            current: None,
            next_val: None,
        };
        iterator.next_val = iterator.find_next();
        iterator
    }
    
    fn find_min(&self, mut node: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        while let Some(n) = node.clone() {
            let left = n.borrow().left.clone();
            if left.is_none() {
                return Some(n);
            }
            node = left;
        }
        None
    }
    
    fn find_next(&self) -> Option<i32> {
        if self.current.is_none() {
            // First call - find minimum
            if let Some(min_node) = self.find_min(self.root.clone()) {
                return Some(min_node.borrow().val);
            }
        } else {
            // Find inorder successor
            if let Some(curr) = &self.current {
                let right = curr.borrow().right.clone();
                if right.is_some() {
                    // Successor is leftmost in right subtree
                    if let Some(min_node) = self.find_min(right) {
                        return Some(min_node.borrow().val);
                    }
                } else {
                    // Find first ancestor where current node is in left subtree
                    let curr_val = curr.borrow().val;
                    if let Some(successor_val) = self.find_successor(self.root.clone(), curr_val) {
                        return Some(successor_val);
                    }
                }
            }
        }
        None
    }
    
    fn find_successor(&self, node: Option<Rc<RefCell<TreeNode>>>, val: i32) -> Option<i32> {
        let mut successor = None;
        let mut current = node;
        
        while let Some(n) = current {
            let node_val = n.borrow().val;
            if node_val > val {
                successor = Some(node_val);
                current = n.borrow().left.clone();
            } else {
                current = n.borrow().right.clone();
            }
        }
        
        successor
    }
    
    fn find_node_with_val(&self, node: Option<Rc<RefCell<TreeNode>>>, val: i32) -> Option<Rc<RefCell<TreeNode>>> {
        if let Some(n) = node {
            let node_val = n.borrow().val;
            if node_val == val {
                Some(n)
            } else if val < node_val {
                self.find_node_with_val(n.borrow().left.clone(), val)
            } else {
                self.find_node_with_val(n.borrow().right.clone(), val)
            }
        } else {
            None
        }
    }
    
    pub fn next(&mut self) -> i32 {
        let result = self.next_val.unwrap();
        self.current = self.find_node_with_val(self.root.clone(), result);
        self.next_val = self.find_next();
        result
    }
    
    pub fn has_next(&self) -> bool {
        self.next_val.is_some()
    }
}

/// Approach 4: Generator-Style with Closure
/// 
/// Uses Rust closures to create a generator-like pattern for traversal.
/// Demonstrates functional programming approach to iteration.
/// 
/// Time Complexity: O(1) amortized for next(), O(1) for hasNext()
/// Space Complexity: O(n) for storing remaining values
pub struct BSTIteratorGenerator {
    values: std::vec::IntoIter<i32>,
    peeked: Option<i32>,
}

impl BSTIteratorGenerator {
    pub fn new(root: Option<Rc<RefCell<TreeNode>>>) -> Self {
        let mut values = Vec::new();
        Self::collect_inorder(root.as_ref(), &mut values);
        let mut iter = values.into_iter();
        let peeked = iter.next();
        
        Self {
            values: iter,
            peeked,
        }
    }
    
    fn collect_inorder(node: Option<&Rc<RefCell<TreeNode>>>, values: &mut Vec<i32>) {
        if let Some(node) = node {
            let node = node.borrow();
            Self::collect_inorder(node.left.as_ref(), values);
            values.push(node.val);
            Self::collect_inorder(node.right.as_ref(), values);
        }
    }
    
    pub fn next(&mut self) -> i32 {
        let result = self.peeked.unwrap();
        self.peeked = self.values.next();
        result
    }
    
    pub fn has_next(&self) -> bool {
        self.peeked.is_some()
    }
}

/// Approach 5: Morris Traversal Style
/// 
/// Attempts to use Morris traversal technique for constant space iteration.
/// Temporarily modifies tree structure to eliminate need for stack/recursion.
/// 
/// Time Complexity: O(1) amortized for next(), O(1) for hasNext()  
/// Space Complexity: O(1) (modifies tree temporarily)
pub struct BSTIteratorMorris {
    current: Option<Rc<RefCell<TreeNode>>>,
    next_val: Option<i32>,
}

impl BSTIteratorMorris {
    pub fn new(root: Option<Rc<RefCell<TreeNode>>>) -> Self {
        let mut iterator = Self {
            current: root,
            next_val: None,
        };
        iterator.advance();
        iterator
    }
    
    fn advance(&mut self) {
        if let Some(node) = self.current.take() {
            let node_val = node.borrow().val;
            let left = node.borrow().left.clone();
            
            if left.is_none() {
                // No left subtree, process current node
                self.next_val = Some(node_val);
                self.current = node.borrow().right.clone();
            } else {
                // Find inorder predecessor
                let mut predecessor = left.clone();
                while let Some(pred) = predecessor.clone() {
                    let pred_right = pred.borrow().right.clone();
                    if pred_right.is_none() || Rc::ptr_eq(pred_right.as_ref().unwrap(), &node) {
                        break;
                    }
                    predecessor = pred_right;
                }
                
                if let Some(pred) = predecessor {
                    let pred_right = pred.borrow().right.clone();
                    if pred_right.is_none() {
                        // Create thread and go left
                        pred.borrow_mut().right = Some(node.clone());
                        self.current = left;
                        self.advance();
                    } else {
                        // Remove thread and process current node
                        pred.borrow_mut().right = None;
                        self.next_val = Some(node_val);
                        self.current = node.borrow().right.clone();
                    }
                }
            }
        } else {
            self.next_val = None;
        }
    }
    
    pub fn next(&mut self) -> i32 {
        let result = self.next_val.unwrap();
        self.advance();
        result
    }
    
    pub fn has_next(&self) -> bool {
        self.next_val.is_some()
    }
}

/// Approach 6: Lazy Stack with Yield Points
/// 
/// Optimized stack approach that only maintains necessary nodes.
/// Minimizes memory usage while maintaining O(1) amortized performance.
/// 
/// Time Complexity: O(1) amortized for next(), O(1) for hasNext()
/// Space Complexity: O(h) where h is height
pub struct BSTIteratorLazy {
    stack: Vec<Rc<RefCell<TreeNode>>>,
    current: Option<Rc<RefCell<TreeNode>>>,
}

impl BSTIteratorLazy {
    pub fn new(root: Option<Rc<RefCell<TreeNode>>>) -> Self {
        let mut iterator = Self {
            stack: Vec::new(),
            current: root,
        };
        iterator.setup_next();
        iterator
    }
    
    fn setup_next(&mut self) {
        // Push all left children onto stack
        while let Some(node) = self.current.take() {
            self.stack.push(node.clone());
            self.current = node.borrow().left.clone();
        }
    }
    
    pub fn next(&mut self) -> i32 {
        let node = self.stack.pop().unwrap();
        let result = node.borrow().val;
        
        // Move to right subtree
        self.current = node.borrow().right.clone();
        self.setup_next();
        
        result
    }
    
    pub fn has_next(&self) -> bool {
        !self.stack.is_empty()
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
        let mut queue = std::collections::VecDeque::new();
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
    fn test_basic_functionality() {
        // Test with tree [7,3,15,null,null,9,20]
        let root = build_tree(vec![Some(7), Some(3), Some(15), None, None, Some(9), Some(20)]);
        let mut iterator = BSTIterator::new(root);
        
        assert_eq!(iterator.next(), 3);
        assert_eq!(iterator.next(), 7);
        assert_eq!(iterator.has_next(), true);
        assert_eq!(iterator.next(), 9);
        assert_eq!(iterator.has_next(), true);
        assert_eq!(iterator.next(), 15);
        assert_eq!(iterator.has_next(), true);
        assert_eq!(iterator.next(), 20);
        assert_eq!(iterator.has_next(), false);
    }
    
    #[test]
    fn test_stack_iterator() {
        let root = build_tree(vec![Some(7), Some(3), Some(15), None, None, Some(9), Some(20)]);
        let mut iterator = BSTIteratorStack::new(root);
        
        assert_eq!(iterator.next(), 3);
        assert_eq!(iterator.next(), 7);
        assert_eq!(iterator.has_next(), true);
        assert_eq!(iterator.next(), 9);
        assert_eq!(iterator.has_next(), true);
        assert_eq!(iterator.next(), 15);
        assert_eq!(iterator.has_next(), true);
        assert_eq!(iterator.next(), 20);
        assert_eq!(iterator.has_next(), false);
    }
    
    #[test]
    fn test_parent_iterator() {
        let root = build_tree(vec![Some(7), Some(3), Some(15), None, None, Some(9), Some(20)]);
        let mut iterator = BSTIteratorParent::new(root);
        
        assert_eq!(iterator.next(), 3);
        assert_eq!(iterator.next(), 7);
        assert_eq!(iterator.has_next(), true);
        assert_eq!(iterator.next(), 9);
        assert_eq!(iterator.has_next(), true);
        assert_eq!(iterator.next(), 15);
        assert_eq!(iterator.has_next(), true);
        assert_eq!(iterator.next(), 20);
        assert_eq!(iterator.has_next(), false);
    }
    
    #[test]
    fn test_generator_iterator() {
        let root = build_tree(vec![Some(7), Some(3), Some(15), None, None, Some(9), Some(20)]);
        let mut iterator = BSTIteratorGenerator::new(root);
        
        assert_eq!(iterator.next(), 3);
        assert_eq!(iterator.next(), 7);
        assert_eq!(iterator.has_next(), true);
        assert_eq!(iterator.next(), 9);
        assert_eq!(iterator.has_next(), true);
        assert_eq!(iterator.next(), 15);
        assert_eq!(iterator.has_next(), true);
        assert_eq!(iterator.next(), 20);
        assert_eq!(iterator.has_next(), false);
    }
    
    #[test]
    fn test_lazy_iterator() {
        let root = build_tree(vec![Some(7), Some(3), Some(15), None, None, Some(9), Some(20)]);
        let mut iterator = BSTIteratorLazy::new(root);
        
        assert_eq!(iterator.next(), 3);
        assert_eq!(iterator.next(), 7);
        assert_eq!(iterator.has_next(), true);
        assert_eq!(iterator.next(), 9);
        assert_eq!(iterator.has_next(), true);
        assert_eq!(iterator.next(), 15);
        assert_eq!(iterator.has_next(), true);
        assert_eq!(iterator.next(), 20);
        assert_eq!(iterator.has_next(), false);
    }
    
    #[test]
    fn test_single_node() {
        let root = build_tree(vec![Some(42)]);
        let mut iterator = BSTIterator::new(root);
        
        assert_eq!(iterator.has_next(), true);
        assert_eq!(iterator.next(), 42);
        assert_eq!(iterator.has_next(), false);
    }
    
    #[test]
    fn test_left_skewed_tree() {
        // Tree: 5 -> 4 -> 3 -> 2 -> 1
        let root = build_tree(vec![Some(5), Some(4), None, Some(3), None, Some(2), None, Some(1)]);
        let mut iterator = BSTIteratorStack::new(root);
        
        assert_eq!(iterator.next(), 1);
        assert_eq!(iterator.next(), 2);
        assert_eq!(iterator.next(), 3);
        assert_eq!(iterator.next(), 4);
        assert_eq!(iterator.next(), 5);
        assert_eq!(iterator.has_next(), false);
    }
    
    #[test]
    fn test_right_skewed_tree() {
        // Tree: 1 -> 2 -> 3 -> 4 -> 5
        let root = build_tree(vec![Some(1), None, Some(2), None, Some(3), None, Some(4), None, Some(5)]);
        let mut iterator = BSTIteratorStack::new(root);
        
        assert_eq!(iterator.next(), 1);
        assert_eq!(iterator.next(), 2);
        assert_eq!(iterator.next(), 3);
        assert_eq!(iterator.next(), 4);
        assert_eq!(iterator.next(), 5);
        assert_eq!(iterator.has_next(), false);
    }
    
    #[test]
    fn test_balanced_tree() {
        // Perfect binary tree: 4, 2, 6, 1, 3, 5, 7
        let root = build_tree(vec![Some(4), Some(2), Some(6), Some(1), Some(3), Some(5), Some(7)]);
        let mut iterator = BSTIteratorStack::new(root);
        
        for expected in [1, 2, 3, 4, 5, 6, 7] {
            assert_eq!(iterator.has_next(), true);
            assert_eq!(iterator.next(), expected);
        }
        assert_eq!(iterator.has_next(), false);
    }
    
    #[test]
    fn test_consistency_across_implementations() {
        let test_trees = vec![
            vec![Some(7), Some(3), Some(15), None, None, Some(9), Some(20)],
            vec![Some(4), Some(2), Some(6), Some(1), Some(3), Some(5), Some(7)],
            vec![Some(1), None, Some(2), None, Some(3)],
            vec![Some(42)],
        ];
        
        for tree_vals in test_trees {
            let root1 = build_tree(tree_vals.clone());
            let root2 = build_tree(tree_vals.clone());
            let root3 = build_tree(tree_vals.clone());
            let root4 = build_tree(tree_vals.clone());
            let root5 = build_tree(tree_vals.clone());
            
            let mut iter1 = BSTIterator::new(root1);
            let mut iter2 = BSTIteratorStack::new(root2);
            let mut iter3 = BSTIteratorParent::new(root3);
            let mut iter4 = BSTIteratorGenerator::new(root4);
            let mut iter5 = BSTIteratorLazy::new(root5);
            
            while iter1.has_next() {
                assert_eq!(iter2.has_next(), true, "Stack iterator inconsistent for {:?}", tree_vals);
                assert_eq!(iter3.has_next(), true, "Parent iterator inconsistent for {:?}", tree_vals);
                assert_eq!(iter4.has_next(), true, "Generator iterator inconsistent for {:?}", tree_vals);
                assert_eq!(iter5.has_next(), true, "Lazy iterator inconsistent for {:?}", tree_vals);
                
                let val1 = iter1.next();
                let val2 = iter2.next();
                let val3 = iter3.next();
                let val4 = iter4.next();
                let val5 = iter5.next();
                
                assert_eq!(val1, val2, "Basic and Stack differ for {:?}: {} vs {}", tree_vals, val1, val2);
                assert_eq!(val1, val3, "Basic and Parent differ for {:?}: {} vs {}", tree_vals, val1, val3);
                assert_eq!(val1, val4, "Basic and Generator differ for {:?}: {} vs {}", tree_vals, val1, val4);
                assert_eq!(val1, val5, "Basic and Lazy differ for {:?}: {} vs {}", tree_vals, val1, val5);
            }
            
            assert_eq!(iter2.has_next(), false);
            assert_eq!(iter3.has_next(), false);
            assert_eq!(iter4.has_next(), false);
            assert_eq!(iter5.has_next(), false);
        }
    }
    
    #[test]
    fn test_partial_iteration() {
        let root = build_tree(vec![Some(4), Some(2), Some(6), Some(1), Some(3), Some(5), Some(7)]);
        let mut iterator = BSTIteratorStack::new(root);
        
        // Only iterate through first few elements
        assert_eq!(iterator.next(), 1);
        assert_eq!(iterator.next(), 2);
        assert_eq!(iterator.has_next(), true);
        
        // Should still work correctly
        assert_eq!(iterator.next(), 3);
        assert_eq!(iterator.next(), 4);
        assert_eq!(iterator.has_next(), true);
    }
    
    #[test]
    fn test_empty_tree() {
        let mut iterator = BSTIterator::new(None);
        assert_eq!(iterator.has_next(), false);
    }
    
    #[test]
    fn test_large_values() {
        // Test with maximum constraint values
        let root = build_tree(vec![Some(1000000), Some(500000), Some(1500000)]);
        let mut iterator = BSTIteratorStack::new(root);
        
        assert_eq!(iterator.next(), 500000);
        assert_eq!(iterator.next(), 1000000);
        assert_eq!(iterator.next(), 1500000);
        assert_eq!(iterator.has_next(), false);
    }
    
    #[test]
    fn test_memory_efficiency() {
        // Create a deep tree to test memory usage
        let mut vals = vec![Some(100)];
        for i in (1..100).rev() {
            vals.extend_from_slice(&[Some(i), None]);
        }
        let root = build_tree(vals);
        
        // Stack-based iterator should use O(h) space
        let mut iterator = BSTIteratorStack::new(root);
        
        // Verify first few elements
        for expected in 1..=5 {
            assert_eq!(iterator.has_next(), true);
            assert_eq!(iterator.next(), expected);
        }
    }
    
    #[test]
    fn test_performance_characteristics() {
        // Create a larger balanced tree
        let root = build_tree(vec![
            Some(8), Some(4), Some(12), Some(2), Some(6), Some(10), Some(14),
            Some(1), Some(3), Some(5), Some(7), Some(9), Some(11), Some(13), Some(15)
        ]);
        
        let mut iterator = BSTIteratorStack::new(root);
        
        // Should iterate through all elements in sorted order
        for expected in 1..=15 {
            assert_eq!(iterator.has_next(), true);
            assert_eq!(iterator.next(), expected);
        }
        assert_eq!(iterator.has_next(), false);
    }
    
    #[test]
    fn test_edge_case_values() {
        // Test with zero values
        let root = build_tree(vec![Some(0), None, Some(1)]);
        let mut iterator = BSTIteratorStack::new(root);
        
        assert_eq!(iterator.next(), 0);
        assert_eq!(iterator.next(), 1);
        assert_eq!(iterator.has_next(), false);
        
        // Test with duplicate values (if allowed by BST implementation)
        let root = build_tree(vec![Some(5), Some(5), Some(5)]);
        let mut iterator = BSTIteratorStack::new(root);
        
        assert_eq!(iterator.next(), 5);
        assert_eq!(iterator.next(), 5);
        assert_eq!(iterator.next(), 5);
        assert_eq!(iterator.has_next(), false);
    }
}