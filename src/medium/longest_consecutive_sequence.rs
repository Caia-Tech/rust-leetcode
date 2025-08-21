//! # Problem 128: Longest Consecutive Sequence
//!
//! Given an unsorted array of integers `nums`, return the length of the longest consecutive 
//! elements sequence.
//!
//! You must write an algorithm that runs in O(n) time complexity.
//!
//! ## Examples
//!
//! ```text
//! Input: nums = [100,4,200,1,3,2]
//! Output: 4
//! Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.
//! ```
//!
//! ```text
//! Input: nums = [0,3,7,2,5,8,4,6,0,1]
//! Output: 9
//! ```
//!
//! ## Constraints
//!
//! * 0 <= nums.length <= 10^5
//! * -10^9 <= nums[i] <= 10^9

use std::collections::{HashSet, HashMap};

/// Solution for Longest Consecutive Sequence problem
pub struct Solution;

impl Solution {
    /// Creates a new instance of Solution
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: HashSet with Smart Iteration (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Put all numbers in HashSet for O(1) lookup
    /// 2. For each number, check if it's the start of a sequence (no num-1 exists)
    /// 3. If it's a start, count consecutive numbers from there
    /// 4. Track maximum length found
    /// 
    /// **Time Complexity:** O(n) - Each number visited at most twice
    /// **Space Complexity:** O(n) - HashSet storage
    /// 
    /// **Key Insight:**
    /// - Only start counting from the beginning of sequences
    /// - If num-1 exists, current num is not a sequence start
    /// - This ensures each number is counted exactly once across all sequences
    /// 
    /// **Why this achieves O(n):**
    /// - Outer loop: O(n) iterations
    /// - Inner while loop: Each number visited at most once across all iterations
    /// - Total: O(n) + O(n) = O(n)
    /// 
    /// **Visualization:**
    /// ```text
    /// nums = [100, 4, 200, 1, 3, 2]
    /// set = {100, 4, 200, 1, 3, 2}
    /// 
    /// num=100: 99 not in set → start of sequence → count: 100 → length 1
    /// num=4: 3 in set → not start, skip
    /// num=200: 199 not in set → start of sequence → count: 200 → length 1  
    /// num=1: 0 not in set → start of sequence → count: 1,2,3,4 → length 4
    /// num=3: 2 in set → not start, skip
    /// num=2: 1 in set → not start, skip
    /// ```
    pub fn longest_consecutive(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let num_set: HashSet<i32> = nums.into_iter().collect();
        let mut max_length = 0;
        
        for &num in &num_set {
            // Only start counting if this is the beginning of a sequence
            if !num_set.contains(&(num - 1)) {
                let mut current_num = num;
                let mut current_length = 1;
                
                // Count consecutive numbers
                while num_set.contains(&(current_num + 1)) {
                    current_num += 1;
                    current_length += 1;
                }
                
                max_length = max_length.max(current_length);
            }
        }
        
        max_length
    }

    /// # Approach 2: Union-Find (Disjoint Set)
    /// 
    /// **Algorithm:**
    /// 1. Create union-find structure for all numbers
    /// 2. For each number, union with consecutive neighbors if they exist
    /// 3. Find the largest component size
    /// 
    /// **Time Complexity:** O(n α(n)) - Nearly linear with inverse Ackermann
    /// **Space Complexity:** O(n) - Union-find structure
    /// 
    /// **Advantages:**
    /// - Demonstrates union-find application
    /// - Good for dynamic connectivity problems
    /// - Extensible to more complex connectivity queries
    /// 
    /// **When to use:** When you need to track connected components dynamically
    pub fn longest_consecutive_union_find(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let unique_nums: Vec<i32> = nums.into_iter().collect::<HashSet<_>>().into_iter().collect();
        let n = unique_nums.len();
        let mut parent = (0..n).collect::<Vec<_>>();
        let mut size = vec![1; n];
        
        // Map number to index
        let num_to_idx: HashMap<i32, usize> = unique_nums.iter()
            .enumerate()
            .map(|(i, &num)| (num, i))
            .collect();
        
        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]); // Path compression
            }
            parent[x]
        }
        
        fn union(parent: &mut Vec<usize>, size: &mut Vec<usize>, x: usize, y: usize) {
            let root_x = find(parent, x);
            let root_y = find(parent, y);
            
            if root_x != root_y {
                // Union by size
                if size[root_x] < size[root_y] {
                    parent[root_x] = root_y;
                    size[root_y] += size[root_x];
                } else {
                    parent[root_y] = root_x;
                    size[root_x] += size[root_y];
                }
            }
        }
        
        // Union consecutive numbers
        for &num in &unique_nums {
            let idx = num_to_idx[&num];
            
            if let Some(&next_idx) = num_to_idx.get(&(num + 1)) {
                union(&mut parent, &mut size, idx, next_idx);
            }
        }
        
        // Find maximum component size
        size.into_iter().max().unwrap_or(0) as i32
    }

    /// # Approach 3: Sorting and Linear Scan
    /// 
    /// **Algorithm:**
    /// 1. Remove duplicates and sort array
    /// 2. Scan through sorted array, tracking consecutive sequences
    /// 3. Reset count when gap found, update maximum
    /// 
    /// **Time Complexity:** O(n log n) - Sorting dominates
    /// **Space Complexity:** O(1) - Only constant extra space after deduplication
    /// 
    /// **Characteristics:**
    /// - Simple and intuitive approach
    /// - Good when data is already sorted or nearly sorted
    /// - Falls back on well-tested sorting algorithms
    /// 
    /// **When to use:** When O(n log n) is acceptable and simplicity is valued
    pub fn longest_consecutive_sort(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let mut unique_nums: Vec<i32> = nums.into_iter().collect::<HashSet<_>>().into_iter().collect();
        unique_nums.sort_unstable();
        
        let mut max_length = 1;
        let mut current_length = 1;
        
        for i in 1..unique_nums.len() {
            if unique_nums[i] == unique_nums[i - 1] + 1 {
                current_length += 1;
            } else {
                max_length = max_length.max(current_length);
                current_length = 1;
            }
        }
        
        max_length.max(current_length)
    }

    /// # Approach 4: HashMap with Lazy Expansion
    /// 
    /// **Algorithm:**
    /// 1. Use HashMap to track sequence boundaries and lengths
    /// 2. For each new number, check if it extends existing sequences
    /// 3. Merge sequences when number connects two sequences
    /// 4. Update boundary information
    /// 
    /// **Time Complexity:** O(n) - Each number processed once
    /// **Space Complexity:** O(n) - HashMap storage
    /// 
    /// **Key insight:**
    /// - Track only sequence endpoints and their lengths
    /// - When adding number, check if it extends left or right sequences
    /// - Merge sequences by updating new endpoints
    /// 
    /// **When useful:** When you need to process numbers incrementally
    pub fn longest_consecutive_hashmap(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let mut lengths: HashMap<i32, i32> = HashMap::new();
        let mut max_length = 0;
        
        for num in nums {
            if lengths.contains_key(&num) {
                continue; // Skip duplicates
            }
            
            let left_length = *lengths.get(&(num - 1)).unwrap_or(&0);
            let right_length = *lengths.get(&(num + 1)).unwrap_or(&0);
            
            let new_length = left_length + right_length + 1;
            lengths.insert(num, new_length);
            
            // Update the boundaries of the new sequence
            lengths.insert(num - left_length, new_length);
            lengths.insert(num + right_length, new_length);
            
            max_length = max_length.max(new_length);
        }
        
        max_length
    }

    /// # Approach 5: Recursive with Memoization
    /// 
    /// **Algorithm:**
    /// 1. For each number, recursively find longest sequence starting from it
    /// 2. Use memoization to cache results
    /// 3. Base case: if next number doesn't exist, sequence length is 1
    /// 
    /// **Time Complexity:** O(n) - Each number computed once with memoization
    /// **Space Complexity:** O(n) - Memoization table + recursion stack
    /// 
    /// **Educational value:**
    /// - Shows recursive approach to sequence problems
    /// - Demonstrates top-down dynamic programming
    /// - Illustrates memoization benefits
    /// 
    /// **Limitations:**
    /// - Potential stack overflow for very long sequences
    /// - More complex than iterative approaches
    pub fn longest_consecutive_recursive(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let num_set: HashSet<i32> = nums.into_iter().collect();
        let mut memo: HashMap<i32, i32> = HashMap::new();
        let mut max_length = 0;
        
        fn find_length(num: i32, set: &HashSet<i32>, memo: &mut HashMap<i32, i32>) -> i32 {
            if let Some(&cached) = memo.get(&num) {
                return cached;
            }
            
            let length = if set.contains(&(num + 1)) {
                1 + find_length(num + 1, set, memo)
            } else {
                1
            };
            
            memo.insert(num, length);
            length
        }
        
        for &num in &num_set {
            let length = find_length(num, &num_set, &mut memo);
            max_length = max_length.max(length);
        }
        
        max_length
    }

    /// # Approach 6: Interval Merging
    /// 
    /// **Algorithm:**
    /// 1. Sort unique numbers
    /// 2. Merge consecutive intervals
    /// 3. Track maximum interval length
    /// 
    /// **Time Complexity:** O(n log n) - Sorting required
    /// **Space Complexity:** O(n) - Store intervals
    /// 
    /// **Connection to interval problems:**
    /// - Each number is a unit interval [num, num]
    /// - Consecutive numbers create merged intervals
    /// - Longest sequence = longest merged interval
    /// 
    /// **When useful:** When working with interval-based algorithms
    pub fn longest_consecutive_intervals(&self, nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let mut unique_nums: Vec<i32> = nums.into_iter().collect::<HashSet<_>>().into_iter().collect();
        unique_nums.sort_unstable();
        
        let mut intervals = Vec::new();
        let mut start = unique_nums[0];
        let mut end = unique_nums[0];
        
        for i in 1..unique_nums.len() {
            if unique_nums[i] == end + 1 {
                end = unique_nums[i];
            } else {
                intervals.push((start, end));
                start = unique_nums[i];
                end = unique_nums[i];
            }
        }
        intervals.push((start, end));
        
        intervals.into_iter()
            .map(|(start, end)| end - start + 1)
            .max()
            .unwrap_or(0)
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

    #[test]
    fn test_basic_examples() {
        let solution = setup();
        
        // Example 1: [100,4,200,1,3,2] → 4
        let result1 = solution.longest_consecutive(vec![100, 4, 200, 1, 3, 2]);
        assert_eq!(result1, 4);
        
        // Example 2: [0,3,7,2,5,8,4,6,0,1] → 9
        let result2 = solution.longest_consecutive(vec![0, 3, 7, 2, 5, 8, 4, 6, 0, 1]);
        assert_eq!(result2, 9);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // Empty array
        assert_eq!(solution.longest_consecutive(vec![]), 0);
        
        // Single element
        assert_eq!(solution.longest_consecutive(vec![1]), 1);
        
        // Two elements consecutive
        assert_eq!(solution.longest_consecutive(vec![1, 2]), 2);
        
        // Two elements non-consecutive
        assert_eq!(solution.longest_consecutive(vec![1, 3]), 1);
        
        // All same elements
        assert_eq!(solution.longest_consecutive(vec![1, 1, 1, 1]), 1);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            vec![100, 4, 200, 1, 3, 2],
            vec![0, 3, 7, 2, 5, 8, 4, 6, 0, 1],
            vec![1],
            vec![],
            vec![1, 2, 3, 4, 5],
            vec![5, 4, 3, 2, 1],
            vec![1, 1, 1, 1],
            vec![1, 3, 5, 7, 9],
            vec![-1, 0, 1, 2],
        ];
        
        for nums in test_cases {
            let result1 = solution.longest_consecutive(nums.clone());
            let result2 = solution.longest_consecutive_union_find(nums.clone());
            let result3 = solution.longest_consecutive_sort(nums.clone());
            let result4 = solution.longest_consecutive_hashmap(nums.clone());
            let result5 = solution.longest_consecutive_recursive(nums.clone());
            let result6 = solution.longest_consecutive_intervals(nums.clone());
            
            assert_eq!(result1, result2, "HashSet vs Union-Find mismatch for {:?}", nums);
            assert_eq!(result2, result3, "Union-Find vs Sort mismatch for {:?}", nums);
            assert_eq!(result3, result4, "Sort vs HashMap mismatch for {:?}", nums);
            assert_eq!(result4, result5, "HashMap vs Recursive mismatch for {:?}", nums);
            assert_eq!(result5, result6, "Recursive vs Intervals mismatch for {:?}", nums);
        }
    }

    #[test]
    fn test_duplicates() {
        let solution = setup();
        
        // Duplicates in sequence
        let result = solution.longest_consecutive(vec![1, 2, 2, 3, 4]);
        assert_eq!(result, 4);
        
        // Duplicates scattered
        let result = solution.longest_consecutive(vec![1, 1, 2, 2, 3, 3]);
        assert_eq!(result, 3);
        
        // Many duplicates
        let result = solution.longest_consecutive(vec![0, 0, 0, 0, 0]);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_negative_numbers() {
        let solution = setup();
        
        // All negative
        let result = solution.longest_consecutive(vec![-3, -2, -1]);
        assert_eq!(result, 3);
        
        // Mix of negative and positive
        let result = solution.longest_consecutive(vec![-1, 0, 1, 2]);
        assert_eq!(result, 4);
        
        // Negative sequence with gap
        let result = solution.longest_consecutive(vec![-5, -4, -1, 0, 1]);
        assert_eq!(result, 3); // -1, 0, 1
    }

    #[test]
    fn test_large_numbers() {
        let solution = setup();
        
        // Large positive numbers
        let result = solution.longest_consecutive(vec![1000000000, 999999999, 999999998]);
        assert_eq!(result, 3);
        
        // Large negative numbers
        let result = solution.longest_consecutive(vec![-1000000000, -999999999]);
        assert_eq!(result, 2);
        
        // Mix of large numbers
        let result = solution.longest_consecutive(vec![1000000000, -1000000000, 0]);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_multiple_sequences() {
        let solution = setup();
        
        // Two equal length sequences
        let result = solution.longest_consecutive(vec![1, 2, 3, 10, 11, 12]);
        assert_eq!(result, 3);
        
        // One longer sequence
        let result = solution.longest_consecutive(vec![1, 2, 10, 11, 12, 13]);
        assert_eq!(result, 4);
        
        // Multiple short sequences
        let result = solution.longest_consecutive(vec![1, 3, 5, 7, 9, 2, 4, 6, 8]);
        assert_eq!(result, 9); // 1,2,3,4,5,6,7,8,9
    }

    #[test]
    fn test_sequence_patterns() {
        let solution = setup();
        
        // Ascending consecutive
        let result = solution.longest_consecutive(vec![1, 2, 3, 4, 5]);
        assert_eq!(result, 5);
        
        // Descending consecutive (order doesn't matter)
        let result = solution.longest_consecutive(vec![5, 4, 3, 2, 1]);
        assert_eq!(result, 5);
        
        // Shuffled consecutive
        let result = solution.longest_consecutive(vec![3, 1, 4, 2, 5]);
        assert_eq!(result, 5);
        
        // Gaps between sequences
        let result = solution.longest_consecutive(vec![1, 2, 4, 5, 7, 8, 9]);
        assert_eq!(result, 3); // 7, 8, 9
    }

    #[test]
    fn test_boundary_values() {
        let solution = setup();
        
        // Minimum possible value
        let result = solution.longest_consecutive(vec![-1000000000, -999999999]);
        assert_eq!(result, 2);
        
        // Maximum possible value
        let result = solution.longest_consecutive(vec![999999999, 1000000000]);
        assert_eq!(result, 2);
        
        // Zero and neighbors
        let result = solution.longest_consecutive(vec![-1, 0, 1]);
        assert_eq!(result, 3);
    }

    #[test]
    fn test_no_consecutive() {
        let solution = setup();
        
        // All isolated numbers
        let result = solution.longest_consecutive(vec![1, 3, 5, 7, 9]);
        assert_eq!(result, 1);
        
        // Large gaps
        let result = solution.longest_consecutive(vec![1, 100, 1000, 10000]);
        assert_eq!(result, 1);
        
        // Powers of 2
        let result = solution.longest_consecutive(vec![1, 2, 4, 8, 16]);
        assert_eq!(result, 2); // 1, 2
    }

    #[test]
    fn test_performance_characteristics() {
        let solution = setup();
        
        // Large consecutive sequence
        let large_seq: Vec<i32> = (1..=1000).collect();
        let result = solution.longest_consecutive(large_seq);
        assert_eq!(result, 1000);
        
        // Large scattered sequence
        let scattered: Vec<i32> = (0..1000).step_by(2).collect(); // Even numbers
        let result = solution.longest_consecutive(scattered);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_mathematical_properties() {
        let solution = setup();
        
        // Property: longest sequence <= array length
        let nums = vec![1, 2, 3, 7, 8, 9];
        let result = solution.longest_consecutive(nums.clone());
        assert!(result <= nums.len() as i32);
        
        // Property: if all unique and consecutive, result = length
        let consecutive = vec![5, 6, 7, 8];
        let result = solution.longest_consecutive(consecutive.clone());
        assert_eq!(result, consecutive.len() as i32);
        
        // Property: result >= 1 for non-empty arrays
        let single = vec![42];
        let result = solution.longest_consecutive(single);
        assert!(result >= 1);
    }

    #[test]
    fn test_specific_edge_cases() {
        let solution = setup();
        
        // Array length 1
        assert_eq!(solution.longest_consecutive(vec![0]), 1);
        
        // Array length 2, non-consecutive
        assert_eq!(solution.longest_consecutive(vec![0, 2]), 1);
        
        // Array length 2, consecutive
        assert_eq!(solution.longest_consecutive(vec![0, 1]), 2);
        
        // Maximum array size (conceptual test)
        let max_array: Vec<i32> = (0..100).collect();
        let result = solution.longest_consecutive(max_array);
        assert_eq!(result, 100);
    }
}