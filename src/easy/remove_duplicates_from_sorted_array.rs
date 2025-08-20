//! # Problem 26: Remove Duplicates from Sorted Array
//!
//! Given an integer array `nums` sorted in **non-decreasing order**, remove the duplicates
//! **in-place** such that each unique element appears only once. The **relative order** of
//! the elements should be kept the same.
//!
//! Since it is impossible to change the length of the array in some languages, you must
//! instead have the result be placed in the **first part** of the array `nums`. More formally,
//! if there are `k` elements after removing the duplicates, then the first `k` elements of
//! `nums` should hold the final result.
//!
//! Return `k`.
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::easy::remove_duplicates_from_sorted_array::Solution;
//! 
//! let solution = Solution::new();
//! 
//! // Example 1:
//! let mut nums = vec![1, 1, 2];
//! let k = solution.remove_duplicates(&mut nums);
//! assert_eq!(k, 2);
//! assert_eq!(nums[0..k as usize], [1, 2]);
//! 
//! // Example 2:
//! let mut nums = vec![0, 0, 1, 1, 1, 2, 2, 3, 3, 4];
//! let k = solution.remove_duplicates(&mut nums);
//! assert_eq!(k, 5);
//! assert_eq!(nums[0..k as usize], [0, 1, 2, 3, 4]);
//! ```
//!
//! ## Constraints
//!
//! - 1 <= nums.length <= 3 * 10^4
//! - -100 <= nums[i] <= 100  
//! - `nums` is sorted in **non-decreasing** order.

/// Solution struct following LeetCode format
pub struct Solution;

impl Solution {
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Two Pointers (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Use slow pointer to track position for next unique element
    /// 2. Use fast pointer to scan through the array
    /// 3. When fast finds a new unique element, copy it to slow position
    /// 4. Increment slow pointer after each unique element placed
    /// 
    /// **Time Complexity:** O(n) - Single pass through the array
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **Key Insight:** Since array is sorted, duplicates are adjacent.
    /// We only need to compare with the previous element to detect uniqueness.
    /// 
    /// **Why this is optimal:**
    /// - Must examine each element at least once → O(n) time minimum
    /// - In-place requirement → O(1) space optimal
    /// - Single pass with early placement → no redundant work
    /// 
    /// **Two-pointer pattern:** Classic technique for in-place array problems
    pub fn remove_duplicates(&self, nums: &mut Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let mut slow = 1; // Position for next unique element
        
        for fast in 1..nums.len() {
            // Found a new unique element
            if nums[fast] != nums[fast - 1] {
                nums[slow] = nums[fast];
                slow += 1;
            }
        }
        
        slow as i32
    }

    /// # Approach 2: Two Pointers with Explicit Tracking
    /// 
    /// **Algorithm:**
    /// 1. Track the last seen value explicitly
    /// 2. Use write pointer for next position
    /// 3. Compare each element with last seen value
    /// 4. Only write when element differs from last seen
    /// 
    /// **Time Complexity:** O(n) - Single pass
    /// **Space Complexity:** O(1) - Constant extra space
    /// 
    /// **Difference from Approach 1:**
    /// - Explicitly tracks "last seen" value vs comparing with previous array element
    /// - Slightly more readable logic flow
    /// - Same performance characteristics
    /// 
    /// **When to prefer:** When explicit state tracking improves code clarity
    pub fn remove_duplicates_explicit(&self, nums: &mut Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let mut write_pos = 1;
        let mut last_unique = nums[0];
        
        for read_pos in 1..nums.len() {
            if nums[read_pos] != last_unique {
                nums[write_pos] = nums[read_pos];
                last_unique = nums[read_pos];
                write_pos += 1;
            }
        }
        
        write_pos as i32
    }

    /// # Approach 3: Single Pointer with Counting (Alternative)
    /// 
    /// **Algorithm:**
    /// 1. Use single pointer to track unique elements placed
    /// 2. Iterate through array, comparing each element with the last unique
    /// 3. Place element only if different from last unique at position i-1
    /// 
    /// **Time Complexity:** O(n) - Single pass
    /// **Space Complexity:** O(1) - Constant extra space
    /// 
    /// **Subtle difference:** Uses the array itself as the "last seen" tracker
    /// - Compares nums[i] with nums[unique_count - 1]
    /// - Avoids separate variable for last unique value
    /// 
    /// **Trade-off:** Slightly less clear logic for marginal memory savings
    pub fn remove_duplicates_single_pointer(&self, nums: &mut Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let mut unique_count = 1;
        
        for i in 1..nums.len() {
            if nums[i] != nums[unique_count - 1] {
                nums[unique_count] = nums[i];
                unique_count += 1;
            }
        }
        
        unique_count as i32
    }

    /// # Approach 4: HashSet-Based (VIOLATES CONSTRAINTS - Educational)
    /// 
    /// **Algorithm:**
    /// 1. Use HashSet to track seen elements
    /// 2. Rebuild array with only unique elements
    /// 3. Return count of unique elements
    /// 
    /// **Time Complexity:** O(n) - Single pass, but with hash operations
    /// **Space Complexity:** O(n) - HashSet storage for unique elements
    /// 
    /// **Why this violates constraints:**
    /// - **Not in-place:** Uses additional O(n) space
    /// - **Ignores sorted property:** Doesn't leverage the key constraint
    /// - **Overkill:** HashSet unnecessary when duplicates are adjacent
    /// 
    /// **Educational value:**
    /// - Shows why understanding problem constraints matters
    /// - Demonstrates when simpler solutions exist
    /// - Highlights space complexity trade-offs
    /// 
    /// **Real-world use case:** When input is NOT sorted
    pub fn remove_duplicates_hashset(&self, nums: &mut Vec<i32>) -> i32 {
        use std::collections::HashSet;
        
        let mut seen = HashSet::new();
        let mut write_pos = 0;
        
        // Fix borrow checker issue by using indices instead of iterator
        for read_pos in 0..nums.len() {
            let num = nums[read_pos];
            if seen.insert(num) {  // Returns true if value was not already present
                nums[write_pos] = num;
                write_pos += 1;
            }
        }
        
        write_pos as i32
    }

    /// # Approach 5: Iterator-Based Functional (Rust Idiomatic)
    /// 
    /// **Algorithm:**
    /// 1. Use iterator with deduplication
    /// 2. Collect unique elements back into vector
    /// 3. Return count of unique elements
    /// 
    /// **Time Complexity:** O(n) - Single pass through iterator
    /// **Space Complexity:** O(n) - Creates new vector (violates in-place)
    /// 
    /// **Why this doesn't meet LeetCode requirements:**
    /// - **Not truly in-place:** Creates intermediate collections
    /// - **Different semantics:** Modifies vector structure vs content
    /// 
    /// **When this is appropriate:**
    /// - **Rust production code:** More idiomatic and readable
    /// - **When in-place not required:** Cleaner functional approach
    /// - **Immutable data preferred:** Functional programming patterns
    /// 
    /// **Educational insight:** Shows difference between algorithmic interviews vs production code
    pub fn remove_duplicates_functional(&self, nums: &mut Vec<i32>) -> i32 {
        // Dedup() only removes consecutive duplicates, perfect for sorted arrays
        nums.dedup();
        nums.len() as i32
    }

    /// # Approach 6: Manual Loop with Early Termination (Optimized)
    /// 
    /// **Algorithm:**
    /// 1. Track current unique value and count
    /// 2. Skip consecutive duplicates in inner loop  
    /// 3. Place unique element and continue
    /// 
    /// **Time Complexity:** O(n) - Each element processed exactly once despite nested appearance
    /// **Space Complexity:** O(1) - Constant extra space
    /// 
    /// **Optimization insight:** 
    /// - Can skip multiple consecutive duplicates in one step
    /// - Useful when there are many consecutive duplicates
    /// - Same O(n) but potentially fewer writes in practice
    /// 
    /// **Trade-off:** More complex code for potential performance gain in specific cases
    pub fn remove_duplicates_skip_duplicates(&self, nums: &mut Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let mut write_pos = 1;
        let mut read_pos = 1;
        
        while read_pos < nums.len() {
            // Skip all duplicates
            while read_pos < nums.len() && nums[read_pos] == nums[read_pos - 1] {
                read_pos += 1;
            }
            
            // Place the unique element if we found one
            if read_pos < nums.len() {
                nums[write_pos] = nums[read_pos];
                write_pos += 1;
            }
            
            read_pos += 1;
        }
        
        write_pos as i32
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
    #[case(vec![1, 1, 2], 2, vec![1, 2])]
    #[case(vec![0, 0, 1, 1, 1, 2, 2, 3, 3, 4], 5, vec![0, 1, 2, 3, 4])]
    #[case(vec![1], 1, vec![1])]
    #[case(vec![1, 2, 3, 4, 5], 5, vec![1, 2, 3, 4, 5])]  // No duplicates
    #[case(vec![1, 1, 1, 1, 1], 1, vec![1])]  // All same
    fn test_basic_cases(
        #[case] mut input: Vec<i32>,
        #[case] expected_k: i32,
        #[case] expected_unique: Vec<i32>,
    ) {
        let solution = setup();
        let original = input.clone();
        let k = solution.remove_duplicates(&mut input);
        
        assert_eq!(k, expected_k, "Wrong count for input {:?}", original);
        assert_eq!(
            &input[0..k as usize], 
            expected_unique.as_slice(),
            "Wrong unique elements for input {:?}", original
        );
    }

    #[test]
    fn test_single_element() {
        let solution = setup();
        
        // Single element
        let mut nums = vec![42];
        let k = solution.remove_duplicates(&mut nums);
        assert_eq!(k, 1);
        assert_eq!(nums[0], 42);
    }

    #[test]
    fn test_two_elements() {
        let solution = setup();
        
        // Two different elements
        let mut nums = vec![1, 2];
        let k = solution.remove_duplicates(&mut nums);
        assert_eq!(k, 2);
        assert_eq!(&nums[0..k as usize], [1, 2]);
        
        // Two same elements
        let mut nums = vec![3, 3];
        let k = solution.remove_duplicates(&mut nums);
        assert_eq!(k, 1);
        assert_eq!(&nums[0..k as usize], [3]);
    }

    #[test]
    fn test_boundary_values() {
        let solution = setup();
        
        // Using constraint boundaries: -100 <= nums[i] <= 100
        let mut nums = vec![-100, -100, -50, 0, 0, 50, 100, 100];
        let k = solution.remove_duplicates(&mut nums);
        assert_eq!(k, 5);
        assert_eq!(&nums[0..k as usize], [-100, -50, 0, 50, 100]);
    }

    #[test]
    fn test_consecutive_duplicates() {
        let solution = setup();
        
        // Many consecutive duplicates
        let mut nums = vec![1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3];
        let k = solution.remove_duplicates(&mut nums);
        assert_eq!(k, 3);
        assert_eq!(&nums[0..k as usize], [1, 2, 3]);
        
        // Mixed pattern
        let mut nums = vec![1, 2, 2, 3, 4, 4, 4, 5, 5, 6];
        let k = solution.remove_duplicates(&mut nums);
        assert_eq!(k, 6);
        assert_eq!(&nums[0..k as usize], [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_no_duplicates() {
        let solution = setup();
        
        // Already unique
        let mut nums = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let original = nums.clone();
        let k = solution.remove_duplicates(&mut nums);
        assert_eq!(k, 10);
        assert_eq!(&nums[0..k as usize], original.as_slice());
    }

    #[test]
    fn test_negative_numbers() {
        let solution = setup();
        
        let mut nums = vec![-5, -5, -3, -3, -1, -1, 0, 0, 2, 2];
        let k = solution.remove_duplicates(&mut nums);
        assert_eq!(k, 5);
        assert_eq!(&nums[0..k as usize], [-5, -3, -1, 0, 2]);
    }

    #[test]
    fn test_large_array() {
        let solution = setup();
        
        // Test with maximum constraint size (30,000 elements)
        let mut large_nums = Vec::new();
        for i in 0..1000 {
            // Add each number 30 times to create duplicates
            for _ in 0..30 {
                large_nums.push(i);
            }
        }
        // Total: 30,000 elements, 1000 unique
        
        let k = solution.remove_duplicates(&mut large_nums);
        assert_eq!(k, 1000);
        
        // Verify first few and last few elements
        assert_eq!(&large_nums[0..5], [0, 1, 2, 3, 4]);
        assert_eq!(&large_nums[995..1000], [995, 996, 997, 998, 999]);
    }

    #[test]
    fn test_all_approaches_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            vec![1, 1, 2],
            vec![0, 0, 1, 1, 1, 2, 2, 3, 3, 4],
            vec![1],
            vec![1, 2, 3, 4, 5],
            vec![1, 1, 1, 1, 1],
            vec![-10, -10, -5, 0, 5, 5, 10],
            vec![1, 2, 2, 3, 4, 4, 4, 5],
        ];
        
        for original in test_cases {
            let mut nums1 = original.clone();
            let mut nums2 = original.clone();
            let mut nums3 = original.clone();
            let mut nums4 = original.clone();
            let mut nums5 = original.clone();
            let mut nums6 = original.clone();
            
            let k1 = solution.remove_duplicates(&mut nums1);
            let k2 = solution.remove_duplicates_explicit(&mut nums2);
            let k3 = solution.remove_duplicates_single_pointer(&mut nums3);
            let k4 = solution.remove_duplicates_hashset(&mut nums4);
            let k5 = solution.remove_duplicates_functional(&mut nums5);
            let k6 = solution.remove_duplicates_skip_duplicates(&mut nums6);
            
            assert_eq!(k1, k2, "Explicit approach differs for {:?}", original);
            assert_eq!(k1, k3, "Single pointer approach differs for {:?}", original);
            assert_eq!(k1, k4, "HashSet approach differs for {:?}", original);
            assert_eq!(k1, k5, "Functional approach differs for {:?}", original);
            assert_eq!(k1, k6, "Skip duplicates approach differs for {:?}", original);
            
            // Verify the unique elements are the same
            let slice1 = &nums1[0..k1 as usize];
            let slice2 = &nums2[0..k2 as usize];
            let slice3 = &nums3[0..k3 as usize];
            let slice4 = &nums4[0..k4 as usize];
            let slice5 = &nums5[0..k5 as usize];
            let slice6 = &nums6[0..k6 as usize];
            
            assert_eq!(slice1, slice2, "Explicit elements differ for {:?}", original);
            assert_eq!(slice1, slice3, "Single pointer elements differ for {:?}", original);
            assert_eq!(slice1, slice4, "HashSet elements differ for {:?}", original);
            assert_eq!(slice1, slice5, "Functional elements differ for {:?}", original);
            assert_eq!(slice1, slice6, "Skip duplicates elements differ for {:?}", original);
        }
    }

    #[test]
    fn test_in_place_modification() {
        let solution = setup();
        
        // Verify that we're actually modifying in-place
        let mut nums = vec![1, 1, 2, 2, 2, 3, 3, 4];
        let original_capacity = nums.capacity();
        let original_ptr = nums.as_ptr();
        
        let k = solution.remove_duplicates(&mut nums);
        
        // Vector should be same object (same pointer and capacity)
        assert_eq!(nums.capacity(), original_capacity);
        assert_eq!(nums.as_ptr(), original_ptr);
        
        // But content should be modified
        assert_eq!(k, 4);
        assert_eq!(&nums[0..k as usize], [1, 2, 3, 4]);
    }

    #[test]
    fn test_sorted_property_preservation() {
        let solution = setup();
        
        // Verify that the sorted property is maintained
        let mut nums = vec![-10, -5, -5, -1, 0, 0, 0, 5, 10, 10];
        let k = solution.remove_duplicates(&mut nums);
        
        let unique_slice = &nums[0..k as usize];
        
        // Should still be sorted
        for i in 1..unique_slice.len() {
            assert!(unique_slice[i] > unique_slice[i - 1], 
                   "Result not sorted: {:?}", unique_slice);
        }
    }

    #[test]
    fn test_performance_patterns() {
        let solution = setup();
        
        // Pattern 1: No duplicates (best case - no movement needed)
        let mut no_dups: Vec<i32> = (0..1000).collect();
        let original_no_dups = no_dups.clone();
        let k1 = solution.remove_duplicates(&mut no_dups);
        assert_eq!(k1, 1000);
        assert_eq!(no_dups[0..k1 as usize], original_no_dups[..]);
        
        // Pattern 2: All duplicates (worst case - maximum skipping)
        let mut all_same = vec![42; 1000];
        let k2 = solution.remove_duplicates(&mut all_same);
        assert_eq!(k2, 1);
        assert_eq!(all_same[0], 42);
        
        // Pattern 3: Alternating duplicates (moderate case)
        let mut alternating = Vec::new();
        for i in 0..500 {
            alternating.push(i);
            alternating.push(i);
        }
        let k3 = solution.remove_duplicates(&mut alternating);
        assert_eq!(k3, 500);
        for i in 0..500 {
            assert_eq!(alternating[i], i as i32);
        }
    }

    #[test]
    fn test_edge_case_comprehensive() {
        let solution = setup();
        
        // Edge case: Two identical elements at boundaries
        let mut boundary_dups = vec![-100, 100, 100];
        let k = solution.remove_duplicates(&mut boundary_dups);
        assert_eq!(k, 2);
        assert_eq!(&boundary_dups[0..k as usize], [-100, 100]);
        
        // Edge case: Maximum duplicates of boundary values
        let mut max_boundary = vec![-100; 100];
        max_boundary.extend(vec![100; 100]);
        let k = solution.remove_duplicates(&mut max_boundary);
        assert_eq!(k, 2);
        assert_eq!(&max_boundary[0..k as usize], [-100, 100]);
    }
}