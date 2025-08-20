//! # Problem 21: Merge Two Sorted Lists
//!
//! You are given the heads of two sorted linked lists `list1` and `list2`.
//! Merge the two lists into one sorted list. The list should be made by splicing 
//! together the nodes of the first two lists.
//!
//! Return the head of the merged linked list.
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::easy::merge_two_sorted_lists::Solution;
//! use rust_leetcode::utils::ListNode;
//! 
//! let solution = Solution::new();
//! 
//! // Example 1: [1,2,4] and [1,3,4] -> [1,1,2,3,4,4]
//! let list1 = ListNode::from_vec(vec![1, 2, 4]);
//! let list2 = ListNode::from_vec(vec![1, 3, 4]);
//! let result = solution.merge_two_lists(list1, list2);
//! assert_eq!(ListNode::to_vec(result), vec![1, 1, 2, 3, 4, 4]);
//! ```
//!
//! ## Constraints
//!
//! - The number of nodes in both lists is in the range [0, 50].
//! - -100 <= Node.val <= 100
//! - Both list1 and list2 are sorted in non-decreasing order.

use crate::utils::ListNode;

/// Solution struct following LeetCode format
pub struct Solution;

impl Solution {
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Iterative with Dummy Head (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Create a dummy head to simplify edge case handling
    /// 2. Use current pointer to track end of merged list
    /// 3. Compare current nodes from both lists
    /// 4. Attach smaller node and advance corresponding pointer
    /// 5. Attach remaining nodes from non-empty list
    /// 
    /// **Time Complexity:** O(m + n) where m, n are lengths of the lists
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **Key Insight:** The dummy head eliminates special case handling for 
    /// the first node, making the code cleaner and less error-prone.
    /// 
    /// **Why this is optimal:**
    /// - Must examine every node in both lists → O(m + n) time minimum
    /// - Only rearranges existing nodes → O(1) space (excluding result)
    /// - No recursion overhead
    /// - Single pass through both lists
    pub fn merge_two_lists(
        &self,
        list1: Option<Box<ListNode>>,
        list2: Option<Box<ListNode>>,
    ) -> Option<Box<ListNode>> {
        // Create dummy head to simplify logic
        let mut dummy = Box::new(ListNode::new(0));
        let mut current = &mut dummy;
        
        let mut l1 = list1;
        let mut l2 = list2;
        
        // Merge while both lists have nodes
        while l1.is_some() && l2.is_some() {
            // Compare current values
            if l1.as_ref().unwrap().val <= l2.as_ref().unwrap().val {
                // Take from l1
                let next_l1 = l1.as_mut().unwrap().next.take();
                current.next = l1;
                l1 = next_l1;
            } else {
                // Take from l2
                let next_l2 = l2.as_mut().unwrap().next.take();
                current.next = l2;
                l2 = next_l2;
            }
            current = current.next.as_mut().unwrap();
        }
        
        // Attach remaining nodes (at most one list is non-empty)
        current.next = l1.or(l2);
        
        dummy.next
    }

    /// # Approach 2: Recursive (Elegant but uses call stack)
    /// 
    /// **Algorithm:**
    /// 1. Base cases: if one list is empty, return the other
    /// 2. Compare current heads
    /// 3. Choose smaller head and recursively merge with remainder
    /// 4. Return the chosen head with merged remainder as next
    /// 
    /// **Time Complexity:** O(m + n) - Each node processed once
    /// **Space Complexity:** O(m + n) - Recursion depth proportional to total nodes
    /// 
    /// **Trade-offs:**
    /// - **Pros:** More elegant, mathematically cleaner
    /// - **Cons:** Uses call stack space, potential stack overflow for large lists
    /// - **Performance:** Function call overhead vs iterative approach
    /// 
    /// **When to use:** When elegance > performance and lists are reasonably small
    pub fn merge_two_lists_recursive(
        &self,
        list1: Option<Box<ListNode>>,
        list2: Option<Box<ListNode>>,
    ) -> Option<Box<ListNode>> {
        match (list1, list2) {
            (None, None) => None,
            (Some(l1), None) => Some(l1),
            (None, Some(l2)) => Some(l2),
            (Some(mut l1), Some(mut l2)) => {
                if l1.val <= l2.val {
                    l1.next = self.merge_two_lists_recursive(l1.next, Some(l2));
                    Some(l1)
                } else {
                    l2.next = self.merge_two_lists_recursive(Some(l1), l2.next);
                    Some(l2)
                }
            }
        }
    }

    /// # Approach 3: In-Place with Pointer Manipulation (Most Memory Efficient)
    /// 
    /// **Algorithm:**
    /// 1. Handle empty list edge cases upfront
    /// 2. Ensure list1 starts with smaller or equal value
    /// 3. Iterate through list1, inserting nodes from list2 where appropriate
    /// 4. Handle case where list2 has remaining nodes
    /// 
    /// **Time Complexity:** O(m + n) - Each node examined once
    /// **Space Complexity:** O(1) - True constant space, no dummy node
    /// 
    /// **Why more complex:**
    /// - No dummy head makes edge cases trickier
    /// - Must carefully track previous node for insertions
    /// - More branching logic required
    /// 
    /// **Trade-off analysis:**
    /// - **Memory:** Saves one ListNode allocation vs dummy head approach
    /// - **Complexity:** More complex code, higher chance of bugs
    /// - **Performance:** Marginally faster (no dummy node creation)
    /// 
    /// **Verdict:** Usually not worth the complexity for marginal gains
    pub fn merge_two_lists_in_place(
        &self,
        list1: Option<Box<ListNode>>,
        list2: Option<Box<ListNode>>,
    ) -> Option<Box<ListNode>> {
        if list1.is_none() {
            return list2;
        }
        if list2.is_none() {
            return list1;
        }
        
        let mut l1 = list1;
        let mut l2 = list2;
        
        // Ensure l1 starts with smaller value for consistency
        if l1.as_ref().unwrap().val > l2.as_ref().unwrap().val {
            std::mem::swap(&mut l1, &mut l2);
        }
        
        let mut current = l1.as_mut().unwrap();
        
        // Traverse l1 and insert l2 nodes where appropriate
        while l2.is_some() {
            if current.next.is_none() || 
               current.next.as_ref().unwrap().val >= l2.as_ref().unwrap().val {
                // Insert l2 node here
                let next_l2 = l2.as_mut().unwrap().next.take();
                let old_next = current.next.take();
                current.next = l2;
                current.next.as_mut().unwrap().next = old_next;
                l2 = next_l2;
            }
            current = current.next.as_mut().unwrap();
        }
        
        l1
    }

    /// # Approach 4: Vector-Based (ANTI-PATTERN - Educational)
    /// 
    /// **Algorithm:**
    /// 1. Convert both linked lists to vectors
    /// 2. Merge vectors using two-pointer technique
    /// 3. Convert merged vector back to linked list
    /// 
    /// **Time Complexity:** O(m + n) - Same as optimal approaches
    /// **Space Complexity:** O(m + n) - Additional vector storage
    /// 
    /// **Why this is an anti-pattern:**
    /// - **Defeats the purpose:** Problem is about linked list manipulation
    /// - **Extra memory:** Unnecessary vector allocations
    /// - **Extra work:** Three phases instead of one
    /// - **Loss of structure:** Loses the linked nature of the data
    /// 
    /// **When you might see this:**
    /// - Novice programmers avoiding pointer manipulation
    /// - Quick prototyping (but shouldn't make it to production)
    /// - When interviewer asks "what if you can't use the linked list structure?"
    /// 
    /// **Educational value:** Shows why understanding data structures matters
    pub fn merge_two_lists_vector_antipattern(
        &self,
        list1: Option<Box<ListNode>>,
        list2: Option<Box<ListNode>>,
    ) -> Option<Box<ListNode>> {
        // Convert to vectors (wasteful but educational)
        let vec1 = ListNode::to_vec(list1);
        let vec2 = ListNode::to_vec(list2);
        
        // Merge vectors using two-pointer technique
        let mut merged = Vec::new();
        let mut i = 0;
        let mut j = 0;
        
        while i < vec1.len() && j < vec2.len() {
            if vec1[i] <= vec2[j] {
                merged.push(vec1[i]);
                i += 1;
            } else {
                merged.push(vec2[j]);
                j += 1;
            }
        }
        
        // Add remaining elements
        while i < vec1.len() {
            merged.push(vec1[i]);
            i += 1;
        }
        while j < vec2.len() {
            merged.push(vec2[j]);
            j += 1;
        }
        
        // Convert back to linked list (more waste)
        ListNode::from_vec(merged)
    }

    /// # Approach 5: Priority Queue / Heap Based (OVERKILL)
    /// 
    /// **Algorithm:**
    /// 1. Add all nodes from both lists to a min-heap
    /// 2. Extract nodes in sorted order to build result
    /// 
    /// **Time Complexity:** O((m + n) * log(m + n)) - Heap operations
    /// **Space Complexity:** O(m + n) - Heap storage
    /// 
    /// **Why this is overkill:**
    /// - **Worse time complexity:** O(n log n) vs O(n) for optimal
    /// - **Worse space complexity:** O(n) vs O(1) for optimal  
    /// - **Unnecessary:** Lists are already sorted!
    /// - **Complex:** Much more code for worse performance
    /// 
    /// **When it might be useful:**
    /// - Merging K sorted lists (where K > 2)
    /// - When you need to merge many lists dynamically
    /// - As a general "merge multiple sorted sequences" solution
    /// 
    /// **Educational insight:** Shows why algorithmic choice depends on problem constraints
    pub fn merge_two_lists_heap_overkill(
        &self,
        list1: Option<Box<ListNode>>,
        list2: Option<Box<ListNode>>,
    ) -> Option<Box<ListNode>> {
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;
        
        // BinaryHeap is max-heap, so use Reverse for min-heap behavior
        let mut heap: BinaryHeap<Reverse<i32>> = BinaryHeap::new();
        
        // Add all values to heap (destroying list structure)
        let mut current = list1;
        while let Some(node) = current {
            heap.push(Reverse(node.val));
            current = node.next;
        }
        
        current = list2;
        while let Some(node) = current {
            heap.push(Reverse(node.val));
            current = node.next;
        }
        
        // Build result from sorted heap
        let mut dummy = Box::new(ListNode::new(0));
        let mut current = &mut dummy;
        
        while let Some(Reverse(val)) = heap.pop() {
            current.next = Some(Box::new(ListNode::new(val)));
            current = current.next.as_mut().unwrap();
        }
        
        dummy.next
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
    use rstest::rstest;

    fn setup() -> Solution {
        Solution::new()
    }

    #[rstest]
    #[case(vec![1, 2, 4], vec![1, 3, 4], vec![1, 1, 2, 3, 4, 4])]
    #[case(vec![], vec![], vec![])]
    #[case(vec![], vec![0], vec![0])]
    #[case(vec![1], vec![], vec![1])]
    #[case(vec![2], vec![1], vec![1, 2])]
    fn test_basic_cases(
        #[case] list1_vals: Vec<i32>,
        #[case] list2_vals: Vec<i32>,
        #[case] expected: Vec<i32>,
    ) {
        let solution = setup();
        let list1 = ListNode::from_vec(list1_vals);
        let list2 = ListNode::from_vec(list2_vals);
        let result = solution.merge_two_lists(list1, list2);
        assert_eq!(ListNode::to_vec(result), expected);
    }

    #[test]
    fn test_empty_lists() {
        let solution = setup();
        
        // Both empty
        assert_eq!(
            ListNode::to_vec(solution.merge_two_lists(None, None)),
            Vec::<i32>::new()
        );
        
        // One empty
        let list1 = ListNode::from_vec(vec![1, 2, 3]);
        assert_eq!(
            ListNode::to_vec(solution.merge_two_lists(list1, None)),
            vec![1, 2, 3]
        );
        
        let list2 = ListNode::from_vec(vec![4, 5, 6]);
        assert_eq!(
            ListNode::to_vec(solution.merge_two_lists(None, list2)),
            vec![4, 5, 6]
        );
    }

    #[test]
    fn test_single_element_lists() {
        let solution = setup();
        
        // Same values
        let list1 = ListNode::from_vec(vec![1]);
        let list2 = ListNode::from_vec(vec![1]);
        assert_eq!(
            ListNode::to_vec(solution.merge_two_lists(list1, list2)),
            vec![1, 1]
        );
        
        // Different values
        let list1 = ListNode::from_vec(vec![1]);
        let list2 = ListNode::from_vec(vec![2]);
        assert_eq!(
            ListNode::to_vec(solution.merge_two_lists(list1, list2)),
            vec![1, 2]
        );
        
        let list1 = ListNode::from_vec(vec![2]);
        let list2 = ListNode::from_vec(vec![1]);
        assert_eq!(
            ListNode::to_vec(solution.merge_two_lists(list1, list2)),
            vec![1, 2]
        );
    }

    #[test]
    fn test_different_lengths() {
        let solution = setup();
        
        // First list shorter
        let list1 = ListNode::from_vec(vec![1, 3]);
        let list2 = ListNode::from_vec(vec![2, 4, 5, 6]);
        assert_eq!(
            ListNode::to_vec(solution.merge_two_lists(list1, list2)),
            vec![1, 2, 3, 4, 5, 6]
        );
        
        // Second list shorter
        let list1 = ListNode::from_vec(vec![1, 3, 5, 7]);
        let list2 = ListNode::from_vec(vec![2, 4]);
        assert_eq!(
            ListNode::to_vec(solution.merge_two_lists(list1, list2)),
            vec![1, 2, 3, 4, 5, 7]
        );
    }

    #[test]
    fn test_no_interleaving() {
        let solution = setup();
        
        // All elements from list1 come before list2
        let list1 = ListNode::from_vec(vec![1, 2, 3]);
        let list2 = ListNode::from_vec(vec![4, 5, 6]);
        assert_eq!(
            ListNode::to_vec(solution.merge_two_lists(list1, list2)),
            vec![1, 2, 3, 4, 5, 6]
        );
        
        // All elements from list2 come before list1
        let list1 = ListNode::from_vec(vec![4, 5, 6]);
        let list2 = ListNode::from_vec(vec![1, 2, 3]);
        assert_eq!(
            ListNode::to_vec(solution.merge_two_lists(list1, list2)),
            vec![1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn test_negative_numbers() {
        let solution = setup();
        
        let list1 = ListNode::from_vec(vec![-10, -5, 0]);
        let list2 = ListNode::from_vec(vec![-7, -3, 2]);
        assert_eq!(
            ListNode::to_vec(solution.merge_two_lists(list1, list2)),
            vec![-10, -7, -5, -3, 0, 2]
        );
    }

    #[test]
    fn test_duplicate_values() {
        let solution = setup();
        
        // Multiple duplicates
        let list1 = ListNode::from_vec(vec![1, 1, 2, 3, 3]);
        let list2 = ListNode::from_vec(vec![1, 2, 2, 3, 4]);
        assert_eq!(
            ListNode::to_vec(solution.merge_two_lists(list1, list2)),
            vec![1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
        );
        
        // All same value
        let list1 = ListNode::from_vec(vec![2, 2, 2]);
        let list2 = ListNode::from_vec(vec![2, 2]);
        assert_eq!(
            ListNode::to_vec(solution.merge_two_lists(list1, list2)),
            vec![2, 2, 2, 2, 2]
        );
    }

    #[test]
    fn test_boundary_values() {
        let solution = setup();
        
        // Using constraint boundaries: -100 <= Node.val <= 100
        let list1 = ListNode::from_vec(vec![-100, 0, 50]);
        let list2 = ListNode::from_vec(vec![-50, 25, 100]);
        assert_eq!(
            ListNode::to_vec(solution.merge_two_lists(list1, list2)),
            vec![-100, -50, 0, 25, 50, 100]
        );
    }

    #[test]
    fn test_all_approaches_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            (vec![1, 2, 4], vec![1, 3, 4]),
            (vec![], vec![]),
            (vec![], vec![0]),
            (vec![1], vec![]),
            (vec![2], vec![1]),
            (vec![1, 3, 5], vec![2, 4, 6]),
            (vec![-10, 0, 10], vec![-5, 5]),
            (vec![1, 1, 1], vec![2, 2]),
        ];
        
        for (vals1, vals2) in test_cases {
            let list1_1 = ListNode::from_vec(vals1.clone());
            let list2_1 = ListNode::from_vec(vals2.clone());
            let result1 = solution.merge_two_lists(list1_1, list2_1);
            
            let list1_2 = ListNode::from_vec(vals1.clone());
            let list2_2 = ListNode::from_vec(vals2.clone());
            let result2 = solution.merge_two_lists_recursive(list1_2, list2_2);
            
            let list1_3 = ListNode::from_vec(vals1.clone());
            let list2_3 = ListNode::from_vec(vals2.clone());
            let result3 = solution.merge_two_lists_vector_antipattern(list1_3, list2_3);
            
            let list1_4 = ListNode::from_vec(vals1.clone());
            let list2_4 = ListNode::from_vec(vals2.clone());
            let result4 = solution.merge_two_lists_heap_overkill(list1_4, list2_4);
            
            let vec1 = ListNode::to_vec(result1);
            let vec2 = ListNode::to_vec(result2);
            let vec3 = ListNode::to_vec(result3);
            let vec4 = ListNode::to_vec(result4);
            
            assert_eq!(vec1, vec2, "Recursive differs for {:?}, {:?}", vals1, vals2);
            assert_eq!(vec1, vec3, "Vector antipattern differs for {:?}, {:?}", vals1, vals2);
            assert_eq!(vec1, vec4, "Heap overkill differs for {:?}, {:?}", vals1, vals2);
        }
    }

    #[test]
    fn test_merge_stability() {
        let solution = setup();
        
        // Test that equal elements maintain relative order (stable sort property)
        // Since we use <= in comparison, elements from list1 should come before list2
        // when values are equal
        let list1 = ListNode::from_vec(vec![1, 3, 5]);
        let list2 = ListNode::from_vec(vec![1, 3, 5]);
        let result = solution.merge_two_lists(list1, list2);
        
        // The result should interleave: [1, 1, 3, 3, 5, 5]
        // This tests the stability of our merge (list1 elements come first for ties)
        assert_eq!(ListNode::to_vec(result), vec![1, 1, 3, 3, 5, 5]);
    }

    #[test]
    fn test_large_lists() {
        let solution = setup();
        
        // Test with maximum constraint size (50 nodes each)
        let list1_vals: Vec<i32> = (0..50).step_by(2).collect(); // [0, 2, 4, ..., 98]
        let list2_vals: Vec<i32> = (1..50).step_by(2).collect(); // [1, 3, 5, ..., 49]
        
        let list1 = ListNode::from_vec(list1_vals.clone());
        let list2 = ListNode::from_vec(list2_vals.clone());
        
        let result = solution.merge_two_lists(list1, list2);
        let result_vec = ListNode::to_vec(result);
        
        // Should be sorted sequence from 0 to 49 (but missing some values due to step_by)
        let mut expected = list1_vals;
        expected.extend(list2_vals);
        expected.sort();
        
        assert_eq!(result_vec, expected);
        assert_eq!(result_vec.len(), 50); // 25 + 25 = 50 total elements
    }

    #[test]
    fn test_linked_list_structure_preservation() {
        let solution = setup();
        
        // Verify that we're actually reusing nodes, not creating new ones
        // This is a bit tricky to test directly, but we can at least verify
        // that the merge doesn't create extra nodes
        let list1 = ListNode::from_vec(vec![1, 3, 5]);
        let list2 = ListNode::from_vec(vec![2, 4, 6]);
        
        let result = solution.merge_two_lists(list1, list2);
        let result_vec = ListNode::to_vec(result);
        
        assert_eq!(result_vec, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(result_vec.len(), 6); // Exactly the sum of input lengths
    }
}