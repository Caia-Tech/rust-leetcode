//! # Problem 21: Merge Two Sorted Lists
//!
//! You are given the heads of two sorted linked lists `list1` and `list2`.
//!
//! Merge the two lists in a one sorted list. The list should be made by splicing together 
//! the nodes of the first two lists.
//!
//! Return the head of the merged linked list.
//!
//! ## Examples
//!
//! ```text
//! Input: list1 = [1,2,4], list2 = [1,3,4]
//! Output: [1,1,2,3,4,4]
//! ```
//!
//! ```text
//! Input: list1 = [], list2 = []
//! Output: []
//! ```
//!
//! ```text
//! Input: list1 = [], list2 = [0]
//! Output: [0]
//! ```
//!
//! ## Constraints
//!
//! * The number of nodes in both lists is in the range [0, 50]
//! * -100 <= Node.val <= 100
//! * Both list1 and list2 are sorted in non-decreasing order

use crate::utils::data_structures::ListNode;

/// Solution for Merge Two Sorted Lists problem
pub struct Solution;

impl Solution {
    /// Creates a new instance of Solution
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Iterative Merge with Dummy Head (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Create dummy head node to simplify edge cases
    /// 2. Use current pointer to track tail of merged list
    /// 3. Compare values and link smaller node to result
    /// 4. Advance pointer in the list whose node was added
    /// 5. Append remaining nodes from non-empty list
    /// 
    /// **Time Complexity:** O(m + n) where m, n are lengths of input lists
    /// **Space Complexity:** O(1) - Only uses constant extra space
    /// 
    /// **Key Insights:**
    /// - Dummy head eliminates special handling of first node
    /// - Two-pointer technique processes both lists simultaneously
    /// - No additional storage needed - reuses existing nodes
    /// 
    /// **Why this works:**
    /// - Both input lists are already sorted
    /// - Always choosing smaller value maintains sorted order
    /// - Splicing preserves all original nodes
    /// 
    /// **Step-by-step for [1,2,4] and [1,3,4]:**
    /// ```text
    /// Initial: dummy -> None, current = dummy
    /// Step 1: Compare 1 vs 1, choose first 1: dummy -> 1
    /// Step 2: Compare 2 vs 1, choose 1: dummy -> 1 -> 1  
    /// Step 3: Compare 2 vs 3, choose 2: dummy -> 1 -> 1 -> 2
    /// Step 4: Compare 4 vs 3, choose 3: dummy -> 1 -> 1 -> 2 -> 3
    /// Step 5: Compare 4 vs 4, choose first 4: dummy -> 1 -> 1 -> 2 -> 3 -> 4
    /// Step 6: Append remaining [4]: dummy -> 1 -> 1 -> 2 -> 3 -> 4 -> 4
    /// ```
    pub fn merge_two_lists(&self, list1: Option<Box<ListNode>>, list2: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        let mut dummy = Box::new(ListNode::new(0));
        let mut current = &mut dummy;
        let mut l1 = list1;
        let mut l2 = list2;
        
        while l1.is_some() && l2.is_some() {
            if l1.as_ref().unwrap().val <= l2.as_ref().unwrap().val {
                let next = l1.as_mut().unwrap().next.take();
                current.next = l1;
                l1 = next;
            } else {
                let next = l2.as_mut().unwrap().next.take();
                current.next = l2;
                l2 = next;
            }
            current = current.next.as_mut().unwrap();
        }
        
        // Append remaining nodes
        current.next = l1.or(l2);
        
        dummy.next
    }

    /// # Approach 2: Recursive Merge
    /// 
    /// **Algorithm:**
    /// 1. Base cases: if one list is empty, return the other
    /// 2. Compare heads of both lists
    /// 3. Choose smaller head and recursively merge rest
    /// 4. Link chosen head to result of recursive merge
    /// 
    /// **Time Complexity:** O(m + n) where m, n are lengths of input lists
    /// **Space Complexity:** O(m + n) - Recursion stack depth
    /// 
    /// **Advantages:**
    /// - Clean, elegant solution
    /// - Natural recursive structure matches problem
    /// - Easy to understand and implement
    /// 
    /// **Disadvantages:**
    /// - Uses recursion stack space
    /// - May cause stack overflow for very long lists
    /// 
    /// **When to use:** For moderate-sized lists where clarity is preferred
    pub fn merge_two_lists_recursive(&self, list1: Option<Box<ListNode>>, list2: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        match (list1, list2) {
            (None, None) => None,
            (Some(l1), None) => Some(l1),
            (None, Some(l2)) => Some(l2),
            (Some(mut l1), Some(mut l2)) => {
                if l1.val <= l2.val {
                    l1.next = self.merge_two_lists_recursive(l1.next.take(), Some(l2));
                    Some(l1)
                } else {
                    l2.next = self.merge_two_lists_recursive(Some(l1), l2.next.take());
                    Some(l2)
                }
            }
        }
    }

    /// # Approach 3: Vector Collection and Rebuild
    /// 
    /// **Algorithm:**
    /// 1. Collect all values from both lists into vector
    /// 2. Sort the combined vector
    /// 3. Build new linked list from sorted values
    /// 
    /// **Time Complexity:** O(n log n) where n = total number of nodes
    /// **Space Complexity:** O(n) - Vector storage plus new list
    /// 
    /// **Disadvantages:**
    /// - Doesn't leverage pre-sorted nature of input
    /// - Uses more time and space than necessary
    /// - Creates entirely new nodes instead of reusing
    /// 
    /// **Educational value:** Shows alternative approach but not optimal
    pub fn merge_two_lists_vector(&self, list1: Option<Box<ListNode>>, list2: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        let mut values = Vec::new();
        
        // Collect values from first list
        let mut current = list1;
        while let Some(node) = current {
            values.push(node.val);
            current = node.next;
        }
        
        // Collect values from second list
        let mut current = list2;
        while let Some(node) = current {
            values.push(node.val);
            current = node.next;
        }
        
        // Sort all values
        values.sort_unstable();
        
        // Build new linked list from sorted values
        let mut head = None;
        for &val in values.iter().rev() {
            let mut new_node = Box::new(ListNode::new(val));
            new_node.next = head;
            head = Some(new_node);
        }
        
        head
    }

    /// # Approach 4: In-place Merge with List1 as Base
    /// 
    /// **Algorithm:**
    /// 1. Use list1 as the base list to merge into
    /// 2. Track previous and current nodes in list1
    /// 3. Insert nodes from list2 at appropriate positions
    /// 4. Handle remaining nodes from list2
    /// 
    /// **Time Complexity:** O(m + n) where m, n are lengths of input lists
    /// **Space Complexity:** O(1) - Only uses constant extra space
    /// 
    /// **Characteristics:**
    /// - Modifies list1 structure to accommodate list2 nodes
    /// - Preserves original nodes from both lists
    /// - Slightly more complex pointer manipulation
    /// 
    /// **When useful:** When you want to preserve the first list structure
    pub fn merge_two_lists_in_place(&self, list1: Option<Box<ListNode>>, list2: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        if list1.is_none() { return list2; }
        if list2.is_none() { return list1; }
        
        let mut result = list1;
        let mut l2 = list2;
        
        // Ensure result starts with the smaller value
        if result.as_ref().unwrap().val > l2.as_ref().unwrap().val {
            std::mem::swap(&mut result, &mut l2);
        }
        
        let mut current = result.as_mut().unwrap();
        
        while let Some(mut l2_node) = l2 {
            // Find position to insert l2_node
            while current.next.is_some() && current.next.as_ref().unwrap().val <= l2_node.val {
                current = current.next.as_mut().unwrap();
            }
            
            // Insert l2_node after current
            let next_l2 = l2_node.next.take();
            l2_node.next = current.next.take();
            current.next = Some(l2_node);
            
            current = current.next.as_mut().unwrap();
            l2 = next_l2;
        }
        
        result
    }

    /// # Approach 5: Priority Queue Simulation
    /// 
    /// **Algorithm:**
    /// 1. Simulate merge using priority queue concept
    /// 2. Compare current heads and always choose smaller
    /// 3. Build result list by linking chosen nodes
    /// 4. Handle edge cases with empty lists
    /// 
    /// **Time Complexity:** O(m + n) where m, n are lengths of input lists
    /// **Space Complexity:** O(1) - Only uses constant extra space
    /// 
    /// **Note:** This is conceptually similar to approach 1 but with different
    /// implementation style that mimics priority queue behavior
    /// 
    /// **Educational value:** Shows connection to merge step in merge sort
    pub fn merge_two_lists_priority_queue(&self, list1: Option<Box<ListNode>>, list2: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        let mut heads = [list1, list2];
        let mut result = None;
        let mut tail = &mut result;
        
        while heads[0].is_some() || heads[1].is_some() {
            let choice = match (&heads[0], &heads[1]) {
                (Some(l1), Some(l2)) => if l1.val <= l2.val { 0 } else { 1 },
                (Some(_), None) => 0,
                (None, Some(_)) => 1,
                (None, None) => break,
            };
            
            if let Some(mut node) = heads[choice].take() {
                heads[choice] = node.next.take();
                *tail = Some(node);
                tail = &mut tail.as_mut().unwrap().next;
            }
        }
        
        result
    }

    /// # Approach 6: Stack-Based Merge
    /// 
    /// **Algorithm:**
    /// 1. Use two stacks to hold nodes in reverse order
    /// 2. Pop from stacks and merge in correct order
    /// 3. Build result list from smallest to largest
    /// 
    /// **Time Complexity:** O(m + n) where m, n are lengths of input lists
    /// **Space Complexity:** O(m + n) - Stack storage
    /// 
    /// **Note:** This approach is mainly educational and not practical
    /// for this problem since the input is already sorted
    /// 
    /// **Educational value:** Shows how different data structures
    /// can be applied to the same problem
    pub fn merge_two_lists_stack(&self, list1: Option<Box<ListNode>>, list2: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        let mut stack1 = Vec::new();
        let mut stack2 = Vec::new();
        
        // Push all nodes to stacks (reversing order)
        let mut current = list1;
        while let Some(node) = current {
            let next = node.next.clone();
            stack1.push(node.val);
            current = next;
        }
        
        let mut current = list2;
        while let Some(node) = current {
            let next = node.next.clone();
            stack2.push(node.val);
            current = next;
        }
        
        // Merge by comparing stack tops (which are largest values)
        let mut result = None;
        
        while !stack1.is_empty() || !stack2.is_empty() {
            let val = match (stack1.last(), stack2.last()) {
                (Some(&v1), Some(&v2)) => {
                    if v1 >= v2 {
                        stack1.pop().unwrap()
                    } else {
                        stack2.pop().unwrap()
                    }
                },
                (Some(_), None) => stack1.pop().unwrap(),
                (None, Some(_)) => stack2.pop().unwrap(),
                (None, None) => break,
            };
            
            let mut new_node = Box::new(ListNode::new(val));
            new_node.next = result;
            result = Some(new_node);
        }
        
        result
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

    fn vec_to_list(values: Vec<i32>) -> Option<Box<ListNode>> {
        let mut head = None;
        for &val in values.iter().rev() {
            let mut new_node = Box::new(ListNode::new(val));
            new_node.next = head;
            head = Some(new_node);
        }
        head
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
    fn test_basic_examples() {
        let solution = setup();
        
        // Example 1: [1,2,4] + [1,3,4] = [1,1,2,3,4,4]
        let list1 = vec_to_list(vec![1, 2, 4]);
        let list2 = vec_to_list(vec![1, 3, 4]);
        let result = solution.merge_two_lists(list1, list2);
        assert_eq!(list_to_vec(result), vec![1, 1, 2, 3, 4, 4]);
        
        // Example 2: [] + [] = []
        let result = solution.merge_two_lists(None, None);
        assert_eq!(list_to_vec(result), Vec::<i32>::new());
        
        // Example 3: [] + [0] = [0]
        let list2 = vec_to_list(vec![0]);
        let result = solution.merge_two_lists(None, list2);
        assert_eq!(list_to_vec(result), vec![0]);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // One empty list
        let list1 = vec_to_list(vec![1, 2, 3]);
        let result = solution.merge_two_lists(list1, None);
        assert_eq!(list_to_vec(result), vec![1, 2, 3]);
        
        let list2 = vec_to_list(vec![4, 5, 6]);
        let result = solution.merge_two_lists(None, list2);
        assert_eq!(list_to_vec(result), vec![4, 5, 6]);
        
        // Single node lists
        let list1 = vec_to_list(vec![1]);
        let list2 = vec_to_list(vec![2]);
        let result = solution.merge_two_lists(list1, list2);
        assert_eq!(list_to_vec(result), vec![1, 2]);
        
        let list1 = vec_to_list(vec![2]);
        let list2 = vec_to_list(vec![1]);
        let result = solution.merge_two_lists(list1, list2);
        assert_eq!(list_to_vec(result), vec![1, 2]);
    }

    #[test]
    fn test_identical_values() {
        let solution = setup();
        
        // All same values
        let list1 = vec_to_list(vec![1, 1, 1]);
        let list2 = vec_to_list(vec![1, 1]);
        let result = solution.merge_two_lists(list1, list2);
        assert_eq!(list_to_vec(result), vec![1, 1, 1, 1, 1]);
        
        // Mixed with duplicates
        let list1 = vec_to_list(vec![1, 2, 2, 3]);
        let list2 = vec_to_list(vec![2, 2, 4]);
        let result = solution.merge_two_lists(list1, list2);
        assert_eq!(list_to_vec(result), vec![1, 2, 2, 2, 2, 3, 4]);
    }

    #[test]
    fn test_disjoint_ranges() {
        let solution = setup();
        
        // List1 completely before list2
        let list1 = vec_to_list(vec![1, 2, 3]);
        let list2 = vec_to_list(vec![4, 5, 6]);
        let result = solution.merge_two_lists(list1, list2);
        assert_eq!(list_to_vec(result), vec![1, 2, 3, 4, 5, 6]);
        
        // List2 completely before list1
        let list1 = vec_to_list(vec![4, 5, 6]);
        let list2 = vec_to_list(vec![1, 2, 3]);
        let result = solution.merge_two_lists(list1, list2);
        assert_eq!(list_to_vec(result), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_interleaved_values() {
        let solution = setup();
        
        // Perfect interleaving
        let list1 = vec_to_list(vec![1, 3, 5, 7]);
        let list2 = vec_to_list(vec![2, 4, 6, 8]);
        let result = solution.merge_two_lists(list1, list2);
        assert_eq!(list_to_vec(result), vec![1, 2, 3, 4, 5, 6, 7, 8]);
        
        // Complex interleaving
        let list1 = vec_to_list(vec![1, 4, 7, 10]);
        let list2 = vec_to_list(vec![2, 3, 8, 9, 11]);
        let result = solution.merge_two_lists(list1, list2);
        assert_eq!(list_to_vec(result), vec![1, 2, 3, 4, 7, 8, 9, 10, 11]);
    }

    #[test]
    fn test_negative_values() {
        let solution = setup();
        
        // Mixed positive and negative
        let list1 = vec_to_list(vec![-3, -1, 2]);
        let list2 = vec_to_list(vec![-2, 0, 4]);
        let result = solution.merge_two_lists(list1, list2);
        assert_eq!(list_to_vec(result), vec![-3, -2, -1, 0, 2, 4]);
        
        // All negative
        let list1 = vec_to_list(vec![-10, -5, -1]);
        let list2 = vec_to_list(vec![-8, -3]);
        let result = solution.merge_two_lists(list1, list2);
        assert_eq!(list_to_vec(result), vec![-10, -8, -5, -3, -1]);
    }

    #[test]
    fn test_different_lengths() {
        let solution = setup();
        
        // First list much longer
        let list1 = vec_to_list(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let list2 = vec_to_list(vec![2, 5]);
        let result = solution.merge_two_lists(list1, list2);
        assert_eq!(list_to_vec(result), vec![1, 2, 2, 3, 4, 5, 5, 6, 7, 8]);
        
        // Second list much longer
        let list1 = vec_to_list(vec![3, 7]);
        let list2 = vec_to_list(vec![1, 2, 4, 5, 6, 8, 9]);
        let result = solution.merge_two_lists(list1, list2);
        assert_eq!(list_to_vec(result), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            (vec![1, 2, 4], vec![1, 3, 4]),
            (vec![], vec![0]),
            (vec![1], vec![2]),
            (vec![1, 3, 5], vec![2, 4, 6]),
            (vec![-1, 0, 3], vec![-2, 1, 4]),
        ];
        
        for (vals1, vals2) in test_cases {
            let list1_1 = vec_to_list(vals1.clone());
            let list2_1 = vec_to_list(vals2.clone());
            let result1 = list_to_vec(solution.merge_two_lists(list1_1, list2_1));
            
            let list1_2 = vec_to_list(vals1.clone());
            let list2_2 = vec_to_list(vals2.clone());
            let result2 = list_to_vec(solution.merge_two_lists_recursive(list1_2, list2_2));
            
            let list1_3 = vec_to_list(vals1.clone());
            let list2_3 = vec_to_list(vals2.clone());
            let result3 = list_to_vec(solution.merge_two_lists_vector(list1_3, list2_3));
            
            let list1_4 = vec_to_list(vals1.clone());
            let list2_4 = vec_to_list(vals2.clone());
            let result4 = list_to_vec(solution.merge_two_lists_in_place(list1_4, list2_4));
            
            let list1_5 = vec_to_list(vals1.clone());
            let list2_5 = vec_to_list(vals2.clone());
            let result5 = list_to_vec(solution.merge_two_lists_priority_queue(list1_5, list2_5));
            
            let list1_6 = vec_to_list(vals1.clone());
            let list2_6 = vec_to_list(vals2.clone());
            let result6 = list_to_vec(solution.merge_two_lists_stack(list1_6, list2_6));
            
            assert_eq!(result1, result2, "Iterative vs Recursive mismatch for {:?}, {:?}", vals1, vals2);
            assert_eq!(result2, result3, "Recursive vs Vector mismatch for {:?}, {:?}", vals1, vals2);
            assert_eq!(result3, result4, "Vector vs In-place mismatch for {:?}, {:?}", vals1, vals2);
            assert_eq!(result4, result5, "In-place vs Priority Queue mismatch for {:?}, {:?}", vals1, vals2);
            assert_eq!(result5, result6, "Priority Queue vs Stack mismatch for {:?}, {:?}", vals1, vals2);
        }
    }

    #[test]
    fn test_boundary_conditions() {
        let solution = setup();
        
        // Maximum values within constraint
        let list1 = vec_to_list(vec![100]);
        let list2 = vec_to_list(vec![100]);
        let result = solution.merge_two_lists(list1, list2);
        assert_eq!(list_to_vec(result), vec![100, 100]);
        
        // Minimum values within constraint
        let list1 = vec_to_list(vec![-100]);
        let list2 = vec_to_list(vec![-100]);
        let result = solution.merge_two_lists(list1, list2);
        assert_eq!(list_to_vec(result), vec![-100, -100]);
        
        // Full range
        let list1 = vec_to_list(vec![-100, 0]);
        let list2 = vec_to_list(vec![-50, 100]);
        let result = solution.merge_two_lists(list1, list2);
        assert_eq!(list_to_vec(result), vec![-100, -50, 0, 100]);
    }

    #[test]
    fn test_mathematical_properties() {
        let solution = setup();
        
        // Property: Result length = sum of input lengths
        let list1 = vec_to_list(vec![1, 3, 5]);
        let list2 = vec_to_list(vec![2, 4, 6, 7]);
        let original_len = 3 + 4;
        let result = solution.merge_two_lists(list1, list2);
        assert_eq!(list_to_vec(result).len(), original_len);
        
        // Property: Result is sorted
        let list1 = vec_to_list(vec![1, 4, 7]);
        let list2 = vec_to_list(vec![2, 3, 8, 9]);
        let result = list_to_vec(solution.merge_two_lists(list1, list2));
        let mut sorted_result = result.clone();
        sorted_result.sort();
        assert_eq!(result, sorted_result);
        
        // Property: All elements preserved
        let vals1 = vec![1, 3, 5];
        let vals2 = vec![2, 4, 6];
        let list1 = vec_to_list(vals1.clone());
        let list2 = vec_to_list(vals2.clone());
        let result = list_to_vec(solution.merge_two_lists(list1, list2));
        
        let mut all_input = vals1;
        all_input.extend(vals2);
        all_input.sort();
        
        assert_eq!(result, all_input);
    }

    #[test]
    fn test_performance_patterns() {
        let solution = setup();
        
        // Many duplicates
        let list1 = vec_to_list(vec![1; 25]);
        let list2 = vec_to_list(vec![1; 25]);
        let result = solution.merge_two_lists(list1, list2);
        assert_eq!(list_to_vec(result), vec![1; 50]);
        
        // Ascending sequence
        let list1 = vec_to_list((0..26).step_by(2).collect());  // Even numbers: 0,2,4,...,24
        let list2 = vec_to_list((1..26).step_by(2).collect());  // Odd numbers: 1,3,5,...,25
        let result = list_to_vec(solution.merge_two_lists(list1, list2));
        assert_eq!(result, (0..26).collect::<Vec<i32>>());
        
        // Descending interleave
        let list1 = vec_to_list(vec![-50, -30, -10]);
        let list2 = vec_to_list(vec![-40, -20, 0]);
        let result = solution.merge_two_lists(list1, list2);
        assert_eq!(list_to_vec(result), vec![-50, -40, -30, -20, -10, 0]);
    }
}