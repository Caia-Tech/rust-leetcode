//! Problem 297: Serialize and Deserialize Binary Tree
//!
//! Implementation by Marvin Tutt, Caia Tech
//!
//! Serialization is the process of converting a data structure or object into a sequence of bits
//! so that it can be stored in a file or memory buffer, or transmitted across a network connection
//! link to be reconstructed later in the same or another computer environment.
//!
//! Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how
//! your serialization/deserialization algorithm should work. You just need to ensure that a binary
//! tree can be serialized to a string and this string can be deserialized to the original tree structure.
//!
//! Clarification: The input/output format is the same as how LeetCode serializes a binary tree.
//! You do not necessarily need to follow this format, so please be creative and come up with
//! different approaches yourself.
//!
//! Example 1:
//! Input: root = [1,2,3,null,null,4,5]
//! Output: [1,2,3,null,null,4,5]
//!
//! Example 2:
//! Input: root = []
//! Output: []
//!
//! Constraints:
//! - The number of nodes in the tree is in the range [0, 10^4].
//! - -1000 <= Node.val <= 1000

use std::rc::Rc;
use std::cell::RefCell;
use std::collections::VecDeque;

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

/// Approach 1: Preorder Traversal with Delimiters - Optimal
/// 
/// Serialize using preorder traversal with comma delimiters and "null" for None nodes.
/// Deserialize by recursively building the tree from the serialized string.
/// 
/// Time Complexity: O(n) for both serialize and deserialize
/// Space Complexity: O(n) for the string and recursion stack
pub struct CodecPreorder;

impl CodecPreorder {
    pub fn new() -> Self {
        CodecPreorder
    }
    
    pub fn serialize(&self, root: Option<Rc<RefCell<TreeNode>>>) -> String {
        let mut result = Vec::new();
        self.serialize_helper(&root, &mut result);
        result.join(",")
    }
    
    fn serialize_helper(&self, node: &Option<Rc<RefCell<TreeNode>>>, result: &mut Vec<String>) {
        match node {
            Some(n) => {
                let n = n.borrow();
                result.push(n.val.to_string());
                self.serialize_helper(&n.left, result);
                self.serialize_helper(&n.right, result);
            }
            None => {
                result.push("null".to_string());
            }
        }
    }
    
    pub fn deserialize(&self, data: String) -> Option<Rc<RefCell<TreeNode>>> {
        if data.is_empty() {
            return None;
        }
        let mut values: VecDeque<String> = data.split(',').map(|s| s.to_string()).collect();
        self.deserialize_helper(&mut values)
    }
    
    fn deserialize_helper(&self, values: &mut VecDeque<String>) -> Option<Rc<RefCell<TreeNode>>> {
        if let Some(val) = values.pop_front() {
            if val == "null" {
                return None;
            }
            
            let val = val.parse::<i32>().unwrap();
            let node = Rc::new(RefCell::new(TreeNode::new(val)));
            node.borrow_mut().left = self.deserialize_helper(values);
            node.borrow_mut().right = self.deserialize_helper(values);
            Some(node)
        } else {
            None
        }
    }
}

/// Approach 2: Level-Order Traversal (BFS)
/// 
/// Serialize using level-order traversal with queue.
/// Deserialize by reconstructing level by level.
/// 
/// Time Complexity: O(n)
/// Space Complexity: O(n)
pub struct CodecLevelOrder;

impl CodecLevelOrder {
    pub fn new() -> Self {
        CodecLevelOrder
    }
    
    pub fn serialize(&self, root: Option<Rc<RefCell<TreeNode>>>) -> String {
        if root.is_none() {
            return String::new();
        }
        
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(root.clone());
        
        while !queue.is_empty() {
            if let Some(node) = queue.pop_front() {
                if let Some(n) = node {
                    let n = n.borrow();
                    result.push(n.val.to_string());
                    queue.push_back(n.left.clone());
                    queue.push_back(n.right.clone());
                } else {
                    result.push("null".to_string());
                }
            }
        }
        
        // Remove trailing nulls
        while result.last() == Some(&"null".to_string()) {
            result.pop();
        }
        
        result.join(",")
    }
    
    pub fn deserialize(&self, data: String) -> Option<Rc<RefCell<TreeNode>>> {
        if data.is_empty() {
            return None;
        }
        
        let values: Vec<&str> = data.split(',').collect();
        let root_val = values[0].parse::<i32>().unwrap();
        let root = Rc::new(RefCell::new(TreeNode::new(root_val)));
        let mut queue = VecDeque::new();
        queue.push_back(root.clone());
        
        let mut i = 1;
        while !queue.is_empty() && i < values.len() {
            if let Some(node) = queue.pop_front() {
                // Process left child
                if i < values.len() && values[i] != "null" {
                    let left_val = values[i].parse::<i32>().unwrap();
                    let left = Rc::new(RefCell::new(TreeNode::new(left_val)));
                    node.borrow_mut().left = Some(left.clone());
                    queue.push_back(left);
                }
                i += 1;
                
                // Process right child
                if i < values.len() && values[i] != "null" {
                    let right_val = values[i].parse::<i32>().unwrap();
                    let right = Rc::new(RefCell::new(TreeNode::new(right_val)));
                    node.borrow_mut().right = Some(right.clone());
                    queue.push_back(right);
                }
                i += 1;
            }
        }
        
        Some(root)
    }
}

/// Approach 3: Inorder + Preorder Traversal
/// 
/// Serialize using both inorder and preorder traversals.
/// Deserialize by reconstructing from both traversals.
/// 
/// Time Complexity: O(n)
/// Space Complexity: O(n)
pub struct CodecInorderPreorder;

impl CodecInorderPreorder {
    pub fn new() -> Self {
        CodecInorderPreorder
    }
    
    pub fn serialize(&self, root: Option<Rc<RefCell<TreeNode>>>) -> String {
        let mut inorder = Vec::new();
        let mut preorder = Vec::new();
        self.inorder_traversal(&root, &mut inorder);
        self.preorder_traversal(&root, &mut preorder);
        
        let inorder_str = inorder.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
        let preorder_str = preorder.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
        
        if inorder_str.is_empty() {
            String::new()
        } else {
            format!("{}#{}", inorder_str, preorder_str)
        }
    }
    
    fn inorder_traversal(&self, node: &Option<Rc<RefCell<TreeNode>>>, result: &mut Vec<i32>) {
        if let Some(n) = node {
            let n = n.borrow();
            self.inorder_traversal(&n.left, result);
            result.push(n.val);
            self.inorder_traversal(&n.right, result);
        }
    }
    
    fn preorder_traversal(&self, node: &Option<Rc<RefCell<TreeNode>>>, result: &mut Vec<i32>) {
        if let Some(n) = node {
            let n = n.borrow();
            result.push(n.val);
            self.preorder_traversal(&n.left, result);
            self.preorder_traversal(&n.right, result);
        }
    }
    
    pub fn deserialize(&self, data: String) -> Option<Rc<RefCell<TreeNode>>> {
        if data.is_empty() {
            return None;
        }
        
        let parts: Vec<&str> = data.split('#').collect();
        if parts.len() != 2 {
            return None;
        }
        
        let inorder: Vec<i32> = parts[0].split(',').map(|s| s.parse().unwrap()).collect();
        let preorder: Vec<i32> = parts[1].split(',').map(|s| s.parse().unwrap()).collect();
        
        self.build_tree(&preorder, &inorder, 0, 0, inorder.len())
    }
    
    fn build_tree(&self, preorder: &[i32], inorder: &[i32], pre_start: usize, in_start: usize, in_end: usize) -> Option<Rc<RefCell<TreeNode>>> {
        if pre_start >= preorder.len() || in_start >= in_end {
            return None;
        }
        
        let root_val = preorder[pre_start];
        let root = Rc::new(RefCell::new(TreeNode::new(root_val)));
        
        // Find root in inorder
        let root_idx = (in_start..in_end).find(|&i| inorder[i] == root_val)?;
        let left_size = root_idx - in_start;
        
        root.borrow_mut().left = self.build_tree(preorder, inorder, pre_start + 1, in_start, root_idx);
        root.borrow_mut().right = self.build_tree(preorder, inorder, pre_start + 1 + left_size, root_idx + 1, in_end);
        
        Some(root)
    }
}

/// Approach 4: JSON-like Format
/// 
/// Serialize to a JSON-like nested structure.
/// Deserialize by parsing the JSON-like string.
/// 
/// Time Complexity: O(n)
/// Space Complexity: O(n)
pub struct CodecJson;

impl CodecJson {
    pub fn new() -> Self {
        CodecJson
    }
    
    pub fn serialize(&self, root: Option<Rc<RefCell<TreeNode>>>) -> String {
        match root {
            Some(node) => {
                let n = node.borrow();
                format!("{{{},[{}],[{}]}}",
                    n.val,
                    self.serialize(n.left.clone()),
                    self.serialize(n.right.clone())
                )
            }
            None => "null".to_string()
        }
    }
    
    pub fn deserialize(&self, data: String) -> Option<Rc<RefCell<TreeNode>>> {
        if data == "null" || data.is_empty() {
            return None;
        }
        
        self.parse_node(&data)
    }
    
    fn parse_node(&self, s: &str) -> Option<Rc<RefCell<TreeNode>>> {
        if s == "null" {
            return None;
        }
        
        // Remove outer braces
        let s = &s[1..s.len()-1];
        
        // Find the value and subtrees
        let mut depth = 0;
        let mut comma_positions = Vec::new();
        
        for (i, ch) in s.chars().enumerate() {
            match ch {
                '[' => depth += 1,
                ']' => depth -= 1,
                ',' if depth == 0 => comma_positions.push(i),
                _ => {}
            }
        }
        
        if comma_positions.len() != 2 {
            return None;
        }
        
        let val = s[..comma_positions[0]].parse::<i32>().ok()?;
        let left_str = &s[comma_positions[0]+2..comma_positions[1]-1];
        let right_str = &s[comma_positions[1]+2..s.len()-1];
        
        let node = Rc::new(RefCell::new(TreeNode::new(val)));
        node.borrow_mut().left = self.parse_node(left_str);
        node.borrow_mut().right = self.parse_node(right_str);
        
        Some(node)
    }
}

/// Approach 5: Parentheses Representation
/// 
/// Serialize using parentheses to represent tree structure.
/// Format: val(left)(right)
/// 
/// Time Complexity: O(n)
/// Space Complexity: O(n)
pub struct CodecParentheses;

impl CodecParentheses {
    pub fn new() -> Self {
        CodecParentheses
    }
    
    pub fn serialize(&self, root: Option<Rc<RefCell<TreeNode>>>) -> String {
        match root {
            Some(node) => {
                let n = node.borrow();
                let left_str = self.serialize(n.left.clone());
                let right_str = self.serialize(n.right.clone());
                
                if left_str.is_empty() && right_str.is_empty() {
                    n.val.to_string()
                } else {
                    format!("{}({})({})", n.val, left_str, right_str)
                }
            }
            None => String::new()
        }
    }
    
    pub fn deserialize(&self, data: String) -> Option<Rc<RefCell<TreeNode>>> {
        if data.is_empty() {
            return None;
        }
        
        let mut chars: Vec<char> = data.chars().collect();
        self.deserialize_helper(&mut chars, &mut 0)
    }
    
    fn deserialize_helper(&self, chars: &[char], idx: &mut usize) -> Option<Rc<RefCell<TreeNode>>> {
        if *idx >= chars.len() {
            return None;
        }
        
        // Parse the value
        let mut val_str = String::new();
        while *idx < chars.len() && chars[*idx] != '(' && chars[*idx] != ')' {
            val_str.push(chars[*idx]);
            *idx += 1;
        }
        
        if val_str.is_empty() {
            return None;
        }
        
        let val = val_str.parse::<i32>().ok()?;
        let node = Rc::new(RefCell::new(TreeNode::new(val)));
        
        // Check for left subtree
        if *idx < chars.len() && chars[*idx] == '(' {
            *idx += 1; // Skip '('
            node.borrow_mut().left = self.deserialize_helper(chars, idx);
            *idx += 1; // Skip ')'
        }
        
        // Check for right subtree
        if *idx < chars.len() && chars[*idx] == '(' {
            *idx += 1; // Skip '('
            node.borrow_mut().right = self.deserialize_helper(chars, idx);
            *idx += 1; // Skip ')'
        }
        
        Some(node)
    }
}

/// Approach 6: Binary Format with Length Encoding
/// 
/// Serialize using a compact binary-like format with length encoding.
/// 
/// Time Complexity: O(n)
/// Space Complexity: O(n)
pub struct CodecBinary;

impl CodecBinary {
    pub fn new() -> Self {
        CodecBinary
    }
    
    pub fn serialize(&self, root: Option<Rc<RefCell<TreeNode>>>) -> String {
        let mut result = Vec::new();
        self.serialize_to_vec(&root, &mut result);
        result.join("|")
    }
    
    fn serialize_to_vec(&self, node: &Option<Rc<RefCell<TreeNode>>>, result: &mut Vec<String>) {
        match node {
            Some(n) => {
                let n = n.borrow();
                result.push(format!("V{}", n.val));
                self.serialize_to_vec(&n.left, result);
                self.serialize_to_vec(&n.right, result);
            }
            None => {
                result.push("N".to_string());
            }
        }
    }
    
    pub fn deserialize(&self, data: String) -> Option<Rc<RefCell<TreeNode>>> {
        if data.is_empty() {
            return None;
        }
        
        let mut parts: VecDeque<String> = data.split('|').map(|s| s.to_string()).collect();
        self.deserialize_from_vec(&mut parts)
    }
    
    fn deserialize_from_vec(&self, parts: &mut VecDeque<String>) -> Option<Rc<RefCell<TreeNode>>> {
        if let Some(part) = parts.pop_front() {
            if part == "N" {
                return None;
            }
            
            if part.starts_with('V') {
                let val = part[1..].parse::<i32>().ok()?;
                let node = Rc::new(RefCell::new(TreeNode::new(val)));
                node.borrow_mut().left = self.deserialize_from_vec(parts);
                node.borrow_mut().right = self.deserialize_from_vec(parts);
                Some(node)
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_tree_1() -> Option<Rc<RefCell<TreeNode>>> {
        // Tree: [1,2,3,null,null,4,5]
        let node1 = Rc::new(RefCell::new(TreeNode::new(1)));
        let node2 = Rc::new(RefCell::new(TreeNode::new(2)));
        let node3 = Rc::new(RefCell::new(TreeNode::new(3)));
        let node4 = Rc::new(RefCell::new(TreeNode::new(4)));
        let node5 = Rc::new(RefCell::new(TreeNode::new(5)));
        
        node1.borrow_mut().left = Some(node2);
        node1.borrow_mut().right = Some(node3.clone());
        node3.borrow_mut().left = Some(node4);
        node3.borrow_mut().right = Some(node5);
        
        Some(node1)
    }
    
    fn create_tree_2() -> Option<Rc<RefCell<TreeNode>>> {
        // Tree: [1,null,2]
        let node1 = Rc::new(RefCell::new(TreeNode::new(1)));
        let node2 = Rc::new(RefCell::new(TreeNode::new(2)));
        
        node1.borrow_mut().right = Some(node2);
        
        Some(node1)
    }
    
    fn trees_equal(tree1: &Option<Rc<RefCell<TreeNode>>>, tree2: &Option<Rc<RefCell<TreeNode>>>) -> bool {
        match (tree1, tree2) {
            (None, None) => true,
            (Some(n1), Some(n2)) => {
                let n1 = n1.borrow();
                let n2 = n2.borrow();
                n1.val == n2.val &&
                trees_equal(&n1.left, &n2.left) &&
                trees_equal(&n1.right, &n2.right)
            }
            _ => false
        }
    }
    
    #[test]
    fn test_preorder_codec() {
        let codec = CodecPreorder::new();
        
        let tree = create_tree_1();
        let serialized = codec.serialize(tree.clone());
        let deserialized = codec.deserialize(serialized.clone());
        
        assert!(trees_equal(&tree, &deserialized));
    }
    
    #[test]
    fn test_levelorder_codec() {
        let codec = CodecLevelOrder::new();
        
        let tree = create_tree_1();
        let serialized = codec.serialize(tree.clone());
        let deserialized = codec.deserialize(serialized.clone());
        
        assert!(trees_equal(&tree, &deserialized));
    }
    
    #[test]
    fn test_inorder_preorder_codec() {
        let codec = CodecInorderPreorder::new();
        
        let tree = create_tree_2();
        let serialized = codec.serialize(tree.clone());
        let deserialized = codec.deserialize(serialized.clone());
        
        assert!(trees_equal(&tree, &deserialized));
    }
    
    #[test]
    fn test_json_codec() {
        let codec = CodecJson::new();
        
        let tree = create_tree_2();
        let serialized = codec.serialize(tree.clone());
        let deserialized = codec.deserialize(serialized.clone());
        
        assert!(trees_equal(&tree, &deserialized));
    }
    
    #[test]
    fn test_parentheses_codec() {
        let codec = CodecParentheses::new();
        
        let tree = create_tree_1();
        let serialized = codec.serialize(tree.clone());
        let deserialized = codec.deserialize(serialized.clone());
        
        assert!(trees_equal(&tree, &deserialized));
    }
    
    #[test]
    fn test_binary_codec() {
        let codec = CodecBinary::new();
        
        let tree = create_tree_1();
        let serialized = codec.serialize(tree.clone());
        let deserialized = codec.deserialize(serialized.clone());
        
        assert!(trees_equal(&tree, &deserialized));
    }
    
    #[test]
    fn test_empty_tree() {
        let codec = CodecPreorder::new();
        
        let tree: Option<Rc<RefCell<TreeNode>>> = None;
        let serialized = codec.serialize(tree.clone());
        let deserialized = codec.deserialize(serialized);
        
        assert!(trees_equal(&tree, &deserialized));
    }
    
    #[test]
    fn test_single_node() {
        let codec = CodecLevelOrder::new();
        
        let tree = Some(Rc::new(RefCell::new(TreeNode::new(42))));
        let serialized = codec.serialize(tree.clone());
        let deserialized = codec.deserialize(serialized);
        
        assert!(trees_equal(&tree, &deserialized));
    }
    
    #[test]
    fn test_negative_values() {
        let codec = CodecPreorder::new();
        
        let node1 = Rc::new(RefCell::new(TreeNode::new(-1)));
        let node2 = Rc::new(RefCell::new(TreeNode::new(-2)));
        let node3 = Rc::new(RefCell::new(TreeNode::new(-3)));
        
        node1.borrow_mut().left = Some(node2);
        node1.borrow_mut().right = Some(node3);
        
        let tree = Some(node1);
        let serialized = codec.serialize(tree.clone());
        let deserialized = codec.deserialize(serialized);
        
        assert!(trees_equal(&tree, &deserialized));
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_trees = vec![
            create_tree_1(),
            create_tree_2(),
            None,
            Some(Rc::new(RefCell::new(TreeNode::new(0)))),
        ];
        
        for tree in test_trees {
            let codec1 = CodecPreorder::new();
            let codec2 = CodecLevelOrder::new();
            let codec3 = CodecInorderPreorder::new();
            let codec4 = CodecJson::new();
            let codec5 = CodecParentheses::new();
            let codec6 = CodecBinary::new();
            
            let s1 = codec1.serialize(tree.clone());
            let d1 = codec1.deserialize(s1);
            
            let s2 = codec2.serialize(tree.clone());
            let d2 = codec2.deserialize(s2);
            
            let s3 = codec3.serialize(tree.clone());
            let d3 = codec3.deserialize(s3);
            
            let s4 = codec4.serialize(tree.clone());
            let d4 = codec4.deserialize(s4);
            
            let s5 = codec5.serialize(tree.clone());
            let d5 = codec5.deserialize(s5);
            
            let s6 = codec6.serialize(tree.clone());
            let d6 = codec6.deserialize(s6);
            
            assert!(trees_equal(&tree, &d1), "Preorder codec failed");
            assert!(trees_equal(&tree, &d2), "LevelOrder codec failed");
            assert!(trees_equal(&tree, &d3), "InorderPreorder codec failed");
            assert!(trees_equal(&tree, &d4), "JSON codec failed");
            assert!(trees_equal(&tree, &d5), "Parentheses codec failed");
            assert!(trees_equal(&tree, &d6), "Binary codec failed");
        }
    }
}