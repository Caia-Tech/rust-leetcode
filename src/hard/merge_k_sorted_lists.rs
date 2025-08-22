//! # Problem 23: Merge k Sorted Lists
//!
//! You are given an array of `k` linked-lists `lists`, each linked-list is sorted in ascending order.
//!
//! Merge all the linked-lists into one sorted linked-list and return it.
//!
//! ## Examples
//!
//! ```ignore
//! use rust_leetcode::hard::merge_k_sorted_lists::Solution;
//! use rust_leetcode::utils::data_structures::ListNode;
//!
//! let solution = Solution::new();
//!
//! // Example 1: lists = [[1,4,5],[1,3,4],[2,6]]
//! // Output: [1,1,2,3,4,4,5,6]
//! // Note: Creating ListNode examples in doc comments is complex, see tests for full examples
//! ```
//!
//! ## Constraints
//!
//! - k == lists.length
//! - 0 <= k <= 10^4
//! - 0 <= lists[i].length <= 500
//! - -10^4 <= lists[i][j] <= 10^4
//! - lists[i] is sorted in ascending order.
//! - The sum of lists[i].length will not exceed 10^4.

use crate::utils::data_structures::ListNode;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Solution struct following LeetCode format
pub struct Solution;

impl Solution {
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Divide and Conquer (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Recursively divide the array of lists into halves
    /// 2. Merge pairs of lists until only one remains
    /// 3. Use the efficient two-list merge from Problem 21
    /// 
    /// **Time Complexity:** O(N log k) where N is total number of nodes, k is number of lists
    /// **Space Complexity:** O(log k) - Recursion stack depth
    /// 
    /// **Key Insight:** Instead of merging lists one by one (which would be O(kN)),
    /// we can merge in a tournament-style tree structure, reducing the work significantly.
    /// 
    /// **Why this is optimal:**
    /// - Each node is processed exactly log k times (depth of merge tree)
    /// - Minimizes redundant comparisons
    /// - Leverages efficient pairwise merge algorithm
    /// - Natural divide-and-conquer structure
    /// 
    /// **Merge tree visualization for k=4:**
    /// ```text
    ///     [Final]
    ///    /       \
    ///  [0,1]    [2,3]
    ///  /  \     /  \
    /// [0][1]  [2][3]
    /// ```
    pub fn merge_k_lists(&self, lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
        if lists.is_empty() {
            return None;
        }
        
        self.divide_and_conquer(&lists, 0, lists.len() - 1)
    }
    
    fn divide_and_conquer(&self, lists: &[Option<Box<ListNode>>], left: usize, right: usize) -> Option<Box<ListNode>> {
        if left == right {
            return lists[left].clone();
        }
        
        if left > right {
            return None;
        }
        
        let mid = left + (right - left) / 2;
        let left_merged = self.divide_and_conquer(lists, left, mid);
        let right_merged = self.divide_and_conquer(lists, mid + 1, right);
        
        self.merge_two_lists(left_merged, right_merged)
    }
    
    fn merge_two_lists(&self, list1: Option<Box<ListNode>>, list2: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        match (list1, list2) {
            (None, None) => None,
            (Some(node), None) | (None, Some(node)) => Some(node),
            (Some(node1), Some(node2)) => {
                if node1.val <= node2.val {
                    Some(Box::new(ListNode {
                        val: node1.val,
                        next: self.merge_two_lists(node1.next, Some(node2)),
                    }))
                } else {
                    Some(Box::new(ListNode {
                        val: node2.val,
                        next: self.merge_two_lists(Some(node1), node2.next),
                    }))
                }
            }
        }
    }

    /// # Approach 2: Min-Heap/Priority Queue
    /// 
    /// **Algorithm:**
    /// 1. Insert the first node of each non-empty list into a min-heap
    /// 2. Extract minimum node, add to result, insert its next node
    /// 3. Repeat until heap is empty
    /// 
    /// **Time Complexity:** O(N log k) where N is total nodes, k is number of lists
    /// **Space Complexity:** O(k) - Heap stores at most k nodes
    /// 
    /// **Key Insight:** At any moment, we only need to track the "front" of each list.
    /// The min-heap efficiently gives us the global minimum among all fronts.
    /// 
    /// **Why this approach is elegant:**
    /// - Intuitive: always pick the smallest available element
    /// - Efficient: heap operations are O(log k)
    /// - Space-efficient: only stores k nodes at a time
    /// - Naturally handles variable-length lists
    /// 
    /// **Implementation challenge in Rust:**
    /// BinaryHeap is max-heap by default, so we use Reverse wrapper or custom Ord
    pub fn merge_k_lists_heap(&self, lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
        // Custom wrapper for min-heap behavior
        #[derive(Debug)]
        struct MinNode(Option<Box<ListNode>>);
        
        impl PartialEq for MinNode {
            fn eq(&self, other: &Self) -> bool {
                match (&self.0, &other.0) {
                    (Some(a), Some(b)) => a.val == b.val,
                    (None, None) => true,
                    _ => false,
                }
            }
        }
        
        impl Eq for MinNode {}
        
        impl PartialOrd for MinNode {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        
        impl Ord for MinNode {
            fn cmp(&self, other: &Self) -> Ordering {
                match (&self.0, &other.0) {
                    (Some(a), Some(b)) => b.val.cmp(&a.val), // Reverse for min-heap
                    (Some(_), None) => Ordering::Less,
                    (None, Some(_)) => Ordering::Greater,
                    (None, None) => Ordering::Equal,
                }
            }
        }
        
        let mut heap = BinaryHeap::new();
        
        // Add first node of each non-empty list to heap
        for list in lists {
            if list.is_some() {
                heap.push(MinNode(list));
            }
        }
        
        let mut dummy = Box::new(ListNode::new(0));
        let mut current = &mut dummy;
        
        while let Some(MinNode(Some(node))) = heap.pop() {
            // Add next node to heap if it exists
            if node.next.is_some() {
                heap.push(MinNode(node.next.clone()));
            }
            
            // Add current node to result
            current.next = Some(Box::new(ListNode::new(node.val)));
            current = current.next.as_mut().unwrap();
        }
        
        dummy.next
    }

    /// # Approach 3: Sequential Merging (Less Efficient)
    /// 
    /// **Algorithm:**
    /// 1. Start with first list as result
    /// 2. Merge second list with result, update result
    /// 3. Continue merging each subsequent list with accumulated result
    /// 
    /// **Time Complexity:** O(kN) where N is total nodes, k is number of lists
    /// **Space Complexity:** O(1) - Only using constant extra space (not counting recursion)
    /// 
    /// **Why this is less efficient:**
    /// - First few merges handle small lists (efficient)
    /// - Later merges handle increasingly large accumulated results (inefficient)
    /// - Each node in early lists gets processed k times
    /// 
    /// **Example analysis for k=4 lists of length n each:**
    /// - Merge 1: Process n + n = 2n nodes
    /// - Merge 2: Process 2n + n = 3n nodes  
    /// - Merge 3: Process 3n + n = 4n nodes
    /// - Total: 2n + 3n + 4n = 9n vs optimal 4n log 4 = 8n
    /// 
    /// **When this might be acceptable:**
    /// - Very few lists (k ≤ 3)
    /// - Memory-constrained environments
    /// - Implementation simplicity is paramount
    pub fn merge_k_lists_sequential(&self, lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
        if lists.is_empty() {
            return None;
        }
        
        let mut result = lists[0].clone();
        
        for i in 1..lists.len() {
            result = self.merge_two_lists(result, lists[i].clone());
        }
        
        result
    }

    /// # Approach 4: Iterative Divide and Conquer
    /// 
    /// **Algorithm:**
    /// 1. Use bottom-up approach instead of recursive top-down
    /// 2. In each iteration, merge adjacent pairs
    /// 3. Continue until only one list remains
    /// 
    /// **Time Complexity:** O(N log k) - Same as recursive divide and conquer
    /// **Space Complexity:** O(1) - No recursion stack
    /// 
    /// **Advantages over recursive approach:**
    /// - No risk of stack overflow for large k
    /// - More cache-friendly access pattern
    /// - Easier to reason about space usage
    /// 
    /// **Implementation pattern:**
    /// ```text
    /// Round 1: [0,1], [2,3], [4,5], [6,7] -> 4 lists
    /// Round 2: [01,23], [45,67]           -> 2 lists
    /// Round 3: [0123,4567]                -> 1 list
    /// ```
    pub fn merge_k_lists_iterative(&self, lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
        if lists.is_empty() {
            return None;
        }
        
        let mut lists = lists;
        let mut interval = 1;
        
        while interval < lists.len() {
            for i in (0..lists.len()).step_by(interval * 2) {
                let left = lists[i].clone();
                let right = if i + interval < lists.len() {
                    lists[i + interval].clone()
                } else {
                    None
                };
                
                lists[i] = self.merge_two_lists(left, right);
            }
            interval *= 2;
        }
        
        lists[0].clone()
    }

    /// # Approach 5: Flatten and Sort (Naive)
    /// 
    /// **Algorithm:**
    /// 1. Traverse all lists and collect all values
    /// 2. Sort the collected values
    /// 3. Build new linked list from sorted values
    /// 
    /// **Time Complexity:** O(N log N) where N is total number of nodes
    /// **Space Complexity:** O(N) - Store all values in vector
    /// 
    /// **Why this defeats the purpose:**
    /// - Ignores the fact that input lists are already sorted
    /// - Uses extra space unnecessarily
    /// - General sorting is overkill for merging sorted sequences
    /// 
    /// **When this might be considered:**
    /// - When lists are not actually sorted (problem misunderstanding)
    /// - Quick prototyping where correctness > efficiency
    /// - Educational demonstration of why leveraging constraints matters
    /// 
    /// **Educational value:** Shows the importance of exploiting problem constraints
    pub fn merge_k_lists_flatten_sort(&self, lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
        let mut values = Vec::new();
        
        // Collect all values
        for list in lists {
            let mut current = list;
            while let Some(node) = current {
                values.push(node.val);
                current = node.next;
            }
        }
        
        // Sort values
        values.sort();
        
        // Build new linked list
        if values.is_empty() {
            return None;
        }
        
        let mut dummy = Box::new(ListNode::new(0));
        let mut current = &mut dummy;
        
        for val in values {
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

    fn setup() -> Solution {
        Solution::new()
    }

    // Helper function to create linked list from vector
    fn create_list(vals: Vec<i32>) -> Option<Box<ListNode>> {
        if vals.is_empty() {
            return None;
        }
        
        let mut head = Box::new(ListNode::new(vals[0]));
        let mut current = &mut head;
        
        for &val in &vals[1..] {
            current.next = Some(Box::new(ListNode::new(val)));
            current = current.next.as_mut().unwrap();
        }
        
        Some(head)
    }

    // Helper function to convert linked list to vector for easy comparison
    fn list_to_vec(list: Option<Box<ListNode>>) -> Vec<i32> {
        let mut result = Vec::new();
        let mut current = list;
        
        while let Some(node) = current {
            result.push(node.val);
            current = node.next;
        }
        
        result
    }

    #[test]
    fn test_empty_input() {
        let solution = setup();
        
        // Empty list of lists
        assert_eq!(solution.merge_k_lists(vec![]), None);
        
        // List containing only empty lists
        let empty_lists = vec![None, None, None];
        assert_eq!(solution.merge_k_lists(empty_lists), None);
    }

    #[test]
    fn test_single_list() {
        let solution = setup();
        
        // Single non-empty list
        let lists = vec![create_list(vec![1, 2, 3])];
        let result = list_to_vec(solution.merge_k_lists(lists));
        assert_eq!(result, vec![1, 2, 3]);
        
        // Single empty list
        let lists = vec![None];
        assert_eq!(solution.merge_k_lists(lists), None);
    }

    #[test]
    fn test_basic_merge() {
        let solution = setup();
        
        // Example from problem: [[1,4,5],[1,3,4],[2,6]]
        let lists = vec![
            create_list(vec![1, 4, 5]),
            create_list(vec![1, 3, 4]),
            create_list(vec![2, 6]),
        ];
        
        let result = list_to_vec(solution.merge_k_lists(lists));
        assert_eq!(result, vec![1, 1, 2, 3, 4, 4, 5, 6]);
    }

    #[test]
    fn test_mixed_empty_lists() {
        let solution = setup();
        
        // Mix of empty and non-empty lists
        let lists = vec![
            None,
            create_list(vec![1, 3]),
            None,
            create_list(vec![2, 4]),
            None,
        ];
        
        let result = list_to_vec(solution.merge_k_lists(lists));
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_single_element_lists() {
        let solution = setup();
        
        // Multiple single-element lists
        let lists = vec![
            create_list(vec![5]),
            create_list(vec![1]),
            create_list(vec![3]),
            create_list(vec![2]),
        ];
        
        let result = list_to_vec(solution.merge_k_lists(lists));
        assert_eq!(result, vec![1, 2, 3, 5]);
    }

    #[test]
    fn test_varying_lengths() {
        let solution = setup();
        
        // Lists of different lengths
        let lists = vec![
            create_list(vec![1]),
            create_list(vec![2, 3, 4, 5]),
            create_list(vec![6, 7]),
            create_list(vec![8, 9, 10, 11, 12]),
        ];
        
        let result = list_to_vec(solution.merge_k_lists(lists));
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }

    #[test]
    fn test_duplicate_values() {
        let solution = setup();
        
        // Lists with duplicate values
        let lists = vec![
            create_list(vec![1, 1, 1]),
            create_list(vec![1, 2, 3]),
            create_list(vec![2, 2, 2]),
        ];
        
        let result = list_to_vec(solution.merge_k_lists(lists));
        assert_eq!(result, vec![1, 1, 1, 1, 2, 2, 2, 2, 3]);
    }

    #[test]
    fn test_negative_numbers() {
        let solution = setup();
        
        // Lists with negative numbers
        let lists = vec![
            create_list(vec![-5, -1, 2]),
            create_list(vec![-3, 0, 3]),
            create_list(vec![-2, 1, 4]),
        ];
        
        let result = list_to_vec(solution.merge_k_lists(lists));
        assert_eq!(result, vec![-5, -3, -2, -1, 0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_boundary_values() {
        let solution = setup();
        
        // Test constraint boundaries: -10^4 <= val <= 10^4
        let lists = vec![
            create_list(vec![-10000, 0, 10000]),
            create_list(vec![-5000, 5000]),
        ];
        
        let result = list_to_vec(solution.merge_k_lists(lists));
        assert_eq!(result, vec![-10000, -5000, 0, 5000, 10000]);
    }

    #[test]
    fn test_large_k() {
        let solution = setup();
        
        // Test with many lists (within constraint k <= 10^4)
        let mut lists = Vec::new();
        for i in 0..100 {
            lists.push(create_list(vec![i * 3, i * 3 + 1, i * 3 + 2]));
        }
        
        let result = list_to_vec(solution.merge_k_lists(lists));
        
        // Should be sorted sequence [0, 1, 2, ..., 299]
        let expected: Vec<i32> = (0..300).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_performance_patterns() {
        let solution = setup();
        
        // Pattern 1: Ascending list lengths (worst case for sequential merge)
        let lists = vec![
            create_list((0..10).collect()),
            create_list((10..30).collect()),
            create_list((30..60).collect()),
            create_list((60..100).collect()),
        ];
        
        let result = list_to_vec(solution.merge_k_lists(lists));
        let expected: Vec<i32> = (0..100).collect();
        assert_eq!(result, expected);
        
        // Pattern 2: Equal length lists (good case for divide and conquer)
        let lists = vec![
            create_list(vec![1, 5, 9]),
            create_list(vec![2, 6, 10]),
            create_list(vec![3, 7, 11]),
            create_list(vec![4, 8, 12]),
        ];
        
        let result = list_to_vec(solution.merge_k_lists(lists));
        assert_eq!(result, (1..13).collect::<Vec<i32>>());
    }

    #[test]
    fn test_all_approaches_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            // Empty cases
            vec![],
            vec![None, None],
            
            // Single list cases
            vec![create_list(vec![1, 2, 3])],
            
            // Multiple list cases
            vec![
                create_list(vec![1, 4, 5]),
                create_list(vec![1, 3, 4]),
                create_list(vec![2, 6]),
            ],
            
            // Mixed empty and non-empty
            vec![
                None,
                create_list(vec![1, 3]),
                create_list(vec![2, 4]),
            ],
            
            // Different lengths
            vec![
                create_list(vec![1]),
                create_list(vec![2, 3, 4]),
                create_list(vec![5, 6]),
            ],
        ];
        
        for (i, lists) in test_cases.into_iter().enumerate() {
            let result1 = list_to_vec(solution.merge_k_lists(lists.clone()));
            let result2 = list_to_vec(solution.merge_k_lists_heap(lists.clone()));
            let result3 = list_to_vec(solution.merge_k_lists_sequential(lists.clone()));
            let result4 = list_to_vec(solution.merge_k_lists_iterative(lists.clone()));
            let result5 = list_to_vec(solution.merge_k_lists_flatten_sort(lists.clone()));
            
            assert_eq!(result1, result2, "Heap approach differs for test case {}", i);
            assert_eq!(result1, result3, "Sequential approach differs for test case {}", i);
            assert_eq!(result1, result4, "Iterative approach differs for test case {}", i);
            assert_eq!(result1, result5, "Flatten-sort approach differs for test case {}", i);
        }
    }

    #[test]
    fn test_divide_and_conquer_specific() {
        let solution = setup();
        
        // Test cases that specifically exercise divide and conquer logic
        
        // Power of 2 number of lists (perfect binary tree)
        let lists = vec![
            create_list(vec![1, 8]),
            create_list(vec![2, 7]),
            create_list(vec![3, 6]),
            create_list(vec![4, 5]),
        ];
        
        let result = list_to_vec(solution.merge_k_lists(lists));
        assert_eq!(result, (1..9).collect::<Vec<i32>>());
        
        // Non-power of 2 (unbalanced tree)
        let lists = vec![
            create_list(vec![1, 6]),
            create_list(vec![2, 5]),
            create_list(vec![3, 4]),
        ];
        
        let result = list_to_vec(solution.merge_k_lists(lists));
        assert_eq!(result, (1..7).collect::<Vec<i32>>());
    }

    #[test]
    fn test_heap_specific_cases() {
        let solution = setup();
        
        // Cases that test heap behavior thoroughly
        
        // All lists start with same value
        let lists = vec![
            create_list(vec![1, 4, 7]),
            create_list(vec![1, 5, 8]),
            create_list(vec![1, 6, 9]),
        ];
        
        let result = list_to_vec(solution.merge_k_lists_heap(lists));
        assert_eq!(result, vec![1, 1, 1, 4, 5, 6, 7, 8, 9]);
        
        // Monotonic sequences (heap always has clear min)
        let lists = vec![
            create_list(vec![1, 2, 3]),
            create_list(vec![4, 5, 6]),
            create_list(vec![7, 8, 9]),
        ];
        
        let result = list_to_vec(solution.merge_k_lists_heap(lists));
        assert_eq!(result, (1..10).collect::<Vec<i32>>());
    }

    #[test]
    fn test_edge_cases_comprehensive() {
        let solution = setup();
        
        // Very large single list
        let large_list = create_list((0..1000).collect());
        let lists = vec![large_list];
        let result = list_to_vec(solution.merge_k_lists(lists));
        assert_eq!(result, (0..1000).collect::<Vec<i32>>());
        
        // Many tiny lists
        let tiny_lists: Vec<_> = (0..100).map(|i| create_list(vec![i])).collect();
        let result = list_to_vec(solution.merge_k_lists(tiny_lists));
        assert_eq!(result, (0..100).collect::<Vec<i32>>());
        
        // Interleaved pattern
        let lists = vec![
            create_list(vec![1, 3, 5, 7, 9]),
            create_list(vec![2, 4, 6, 8, 10]),
        ];
        
        let result = list_to_vec(solution.merge_k_lists(lists));
        assert_eq!(result, (1..11).collect::<Vec<i32>>());
    }
}