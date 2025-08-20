//! Common data structures used in LeetCode problems

use std::cell::RefCell;
use std::rc::Rc;

/// Definition for singly-linked list
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        ListNode { next: None, val }
    }
    
    /// Helper function to create a linked list from a vector
    pub fn from_vec(vals: Vec<i32>) -> Option<Box<ListNode>> {
        let mut head = None;
        for &val in vals.iter().rev() {
            let mut node = ListNode::new(val);
            node.next = head;
            head = Some(Box::new(node));
        }
        head
    }
    
    /// Helper function to convert linked list to vector
    pub fn to_vec(head: Option<Box<ListNode>>) -> Vec<i32> {
        let mut result = Vec::new();
        let mut current = head;
        while let Some(node) = current {
            result.push(node.val);
            current = node.next;
        }
        result
    }
}

/// Definition for a binary tree node
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