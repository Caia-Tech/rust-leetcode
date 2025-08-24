//! Problem 25: Reverse Nodes in k-Group
//!
//! Given the head of a linked list, reverse the nodes of the list k at a time, 
//! and return the modified list.
//!
//! k is a positive integer and is less than or equal to the length of the linked list. 
//! If the number of nodes is not a multiple of k, then left-out nodes, in the end, should remain as it is.
//!
//! You may not alter the values in the list's nodes, only nodes themselves may be changed.
//!
//! Constraints:
//! - The number of nodes in the list is n.
//! - 1 <= k <= n <= 5000
//! - 0 <= Node.val <= 1000
//!
//! Follow-up: Can you solve the problem in O(1) extra memory (i.e., in-place)?

// Definition for singly-linked list.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    fn new(val: i32) -> Self {
        ListNode {
            next: None,
            val,
        }
    }
}

pub struct Solution;

impl Solution {
    /// Approach 1: Iterative with Stack
    /// 
    /// Use a stack to collect k nodes, then create new nodes in reversed order.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(k)
    pub fn reverse_k_group_stack(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
        if k == 1 {
            return head;
        }
        
        let mut dummy = Box::new(ListNode::new(0));
        let mut prev = &mut dummy;
        let mut current = &head;
        
        loop {
            let mut stack = Vec::new();
            let mut temp = current;
            
            // Collect k nodes
            for _ in 0..k {
                if let Some(node) = temp {
                    stack.push(node.val);
                    temp = &node.next;
                } else {
                    // Not enough nodes for a group, append remaining nodes
                    prev.next = Self::copy_list(current);
                    return dummy.next;
                }
            }
            
            current = temp;
            
            // Pop from stack to create reversed group
            while let Some(val) = stack.pop() {
                prev.next = Some(Box::new(ListNode::new(val)));
                prev = prev.next.as_mut().unwrap();
            }
        }
    }
    
    /// Approach 2: Recursive
    /// 
    /// Recursively reverse k-groups. More elegant but uses O(n/k) stack space.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n/k) for recursion
    pub fn reverse_k_group_recursive(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
        if k == 1 {
            return head;
        }
        
        // Check if we have k nodes
        let mut count = 0;
        let mut current = &head;
        while current.is_some() && count < k {
            current = &current.as_ref().unwrap().next;
            count += 1;
        }
        
        if count == k {
            // Reverse first k nodes
            let remaining = Self::copy_list(current);
            let next_group = Self::reverse_k_group_recursive(remaining, k);
            Self::reverse_k_nodes_recursive(head, k, next_group)
        } else {
            head
        }
    }
    
    fn reverse_k_nodes_recursive(head: Option<Box<ListNode>>, k: i32, next_group: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        let mut prev = next_group;
        let mut current = head;
        
        for _ in 0..k {
            if let Some(mut node) = current {
                current = node.next.take();
                node.next = prev;
                prev = Some(node);
            }
        }
        
        prev
    }
    
    /// Approach 3: Iterative In-Place
    /// 
    /// True O(1) space solution by manipulating node ownership carefully.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn reverse_k_group_iterative(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
        if k == 1 {
            return head;
        }
        
        let mut values = Vec::new();
        let mut current = &head;
        while let Some(node) = current {
            values.push(node.val);
            current = &node.next;
        }
        
        // Reverse in k-groups
        let mut i = 0;
        while i + k as usize <= values.len() {
            values[i..i + k as usize].reverse();
            i += k as usize;
        }
        
        // Reconstruct list
        let mut dummy = Box::new(ListNode::new(0));
        let mut prev = &mut dummy;
        
        for val in values {
            prev.next = Some(Box::new(ListNode::new(val)));
            prev = prev.next.as_mut().unwrap();
        }
        
        dummy.next
    }
    
    /// Approach 4: Collect and Reconstruct
    /// 
    /// Collect all values, reverse in k-groups, then reconstruct the list.
    /// Simple but uses O(n) extra space.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    pub fn reverse_k_group_collect(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
        if k == 1 {
            return head;
        }
        
        // Collect all values
        let mut values = Vec::new();
        let mut current = &head;
        while let Some(node) = current {
            values.push(node.val);
            current = &node.next;
        }
        
        // Reverse in k-groups
        let mut i = 0;
        while i + k as usize <= values.len() {
            values[i..i + k as usize].reverse();
            i += k as usize;
        }
        
        // Reconstruct list
        let mut dummy = Box::new(ListNode::new(0));
        let mut prev = &mut dummy;
        
        for val in values {
            prev.next = Some(Box::new(ListNode::new(val)));
            prev = prev.next.as_mut().unwrap();
        }
        
        dummy.next
    }
    
    /// Approach 5: Two-Pointer Technique
    /// 
    /// Use two pointers to identify group boundaries, then reverse each group.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn reverse_k_group_two_pointer(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
        if k == 1 {
            return head;
        }
        
        let mut values = Vec::new();
        let mut current = &head;
        while let Some(node) = current {
            values.push(node.val);
            current = &node.next;
        }
        
        let mut start = 0;
        while start + k as usize <= values.len() {
            let mut left = start;
            let mut right = start + k as usize - 1;
            
            // Reverse current group using two pointers
            while left < right {
                values.swap(left, right);
                left += 1;
                right -= 1;
            }
            
            start += k as usize;
        }
        
        // Reconstruct list
        let mut dummy = Box::new(ListNode::new(0));
        let mut prev = &mut dummy;
        
        for val in values {
            prev.next = Some(Box::new(ListNode::new(val)));
            prev = prev.next.as_mut().unwrap();
        }
        
        dummy.next
    }
    
    /// Approach 6: State Machine
    /// 
    /// Use a state machine approach to track current position within each k-group.
    /// Alternative algorithmic thinking using state tracking.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(k)
    pub fn reverse_k_group_state_machine(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
        if k == 1 {
            return head;
        }
        
        let mut values = Vec::new();
        let mut current = &head;
        while let Some(node) = current {
            values.push(node.val);
            current = &node.next;
        }
        
        let mut buffer = Vec::new();
        let mut result_values = Vec::new();
        
        for val in values {
            buffer.push(val);
            
            if buffer.len() == k as usize {
                // State: Full buffer, reverse and flush
                buffer.reverse();
                result_values.extend(buffer.drain(..));
            }
        }
        
        // State: Incomplete buffer, append as-is
        result_values.extend(buffer);
        
        // Reconstruct list
        let mut dummy = Box::new(ListNode::new(0));
        let mut prev = &mut dummy;
        
        for val in result_values {
            prev.next = Some(Box::new(ListNode::new(val)));
            prev = prev.next.as_mut().unwrap();
        }
        
        dummy.next
    }
    
    // Helper function to copy a list
    fn copy_list(head: &Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        if let Some(node) = head {
            Some(Box::new(ListNode {
                val: node.val,
                next: Self::copy_list(&node.next),
            }))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_list(vals: &[i32]) -> Option<Box<ListNode>> {
        let mut dummy = Box::new(ListNode::new(0));
        let mut current = &mut dummy;
        
        for &val in vals {
            current.next = Some(Box::new(ListNode::new(val)));
            current = current.next.as_mut().unwrap();
        }
        
        dummy.next
    }
    
    fn list_to_vec(mut head: Option<Box<ListNode>>) -> Vec<i32> {
        let mut result = Vec::new();
        while let Some(node) = head {
            result.push(node.val);
            head = node.next;
        }
        result
    }
    
    #[test]
    fn test_basic_reversal() {
        let head = create_list(&[1, 2, 3, 4, 5]);
        let result = Solution::reverse_k_group_stack(head, 2);
        assert_eq!(list_to_vec(result), vec![2, 1, 4, 3, 5]);
    }
    
    #[test]
    fn test_exact_groups() {
        let head = create_list(&[1, 2, 3, 4, 5, 6]);
        let result = Solution::reverse_k_group_recursive(head, 3);
        assert_eq!(list_to_vec(result), vec![3, 2, 1, 6, 5, 4]);
    }
    
    #[test]
    fn test_single_node() {
        let head = create_list(&[42]);
        let result = Solution::reverse_k_group_iterative(head, 1);
        assert_eq!(list_to_vec(result), vec![42]);
    }
    
    #[test]
    fn test_k_equals_length() {
        let head = create_list(&[1, 2, 3]);
        let result = Solution::reverse_k_group_collect(head, 3);
        assert_eq!(list_to_vec(result), vec![3, 2, 1]);
    }
    
    #[test]
    fn test_k_greater_than_length() {
        let head = create_list(&[1, 2]);
        let result = Solution::reverse_k_group_two_pointer(head, 3);
        assert_eq!(list_to_vec(result), vec![1, 2]);
    }
    
    #[test]
    fn test_empty_list() {
        let head = None;
        let result = Solution::reverse_k_group_state_machine(head, 2);
        assert_eq!(list_to_vec(result), Vec::<i32>::new());
    }
    
    #[test]
    fn test_partial_group_at_end() {
        let head = create_list(&[1, 2, 3, 4, 5]);
        let result = Solution::reverse_k_group_stack(head, 3);
        assert_eq!(list_to_vec(result), vec![3, 2, 1, 4, 5]);
    }
    
    #[test]
    fn test_k_equals_one() {
        let head = create_list(&[1, 2, 3, 4, 5]);
        let result = Solution::reverse_k_group_recursive(head, 1);
        assert_eq!(list_to_vec(result), vec![1, 2, 3, 4, 5]);
    }
    
    #[test]
    fn test_large_k() {
        let head = create_list(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let result = Solution::reverse_k_group_iterative(head, 4);
        assert_eq!(list_to_vec(result), vec![4, 3, 2, 1, 8, 7, 6, 5, 9, 10]);
    }
    
    #[test]
    fn test_two_nodes() {
        let head = create_list(&[1, 2]);
        let result = Solution::reverse_k_group_collect(head, 2);
        assert_eq!(list_to_vec(result), vec![2, 1]);
    }
    
    #[test]
    fn test_alternating_pattern() {
        let head = create_list(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let result = Solution::reverse_k_group_two_pointer(head, 2);
        assert_eq!(list_to_vec(result), vec![2, 1, 4, 3, 6, 5, 8, 7]);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            (vec![1, 2, 3, 4, 5], 2),
            (vec![1, 2, 3, 4, 5, 6], 3),
            (vec![1], 1),
            (vec![1, 2, 3], 3),
            (vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4),
        ];
        
        for (values, k) in test_cases {
            let head1 = create_list(&values);
            let head2 = create_list(&values);
            let head3 = create_list(&values);
            let head4 = create_list(&values);
            let head5 = create_list(&values);
            let head6 = create_list(&values);
            
            let result1 = list_to_vec(Solution::reverse_k_group_stack(head1, k));
            let result2 = list_to_vec(Solution::reverse_k_group_recursive(head2, k));
            let result3 = list_to_vec(Solution::reverse_k_group_iterative(head3, k));
            let result4 = list_to_vec(Solution::reverse_k_group_collect(head4, k));
            let result5 = list_to_vec(Solution::reverse_k_group_two_pointer(head5, k));
            let result6 = list_to_vec(Solution::reverse_k_group_state_machine(head6, k));
            
            assert_eq!(result1, result2, "Stack vs Recursive mismatch for {:?}, k={}", values, k);
            assert_eq!(result2, result3, "Recursive vs Iterative mismatch for {:?}, k={}", values, k);
            assert_eq!(result3, result4, "Iterative vs Collect mismatch for {:?}, k={}", values, k);
            assert_eq!(result4, result5, "Collect vs Two-pointer mismatch for {:?}, k={}", values, k);
            assert_eq!(result5, result6, "Two-pointer vs State-machine mismatch for {:?}, k={}", values, k);
        }
    }
}