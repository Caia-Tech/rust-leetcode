//! Problem 128: Longest Consecutive Sequence
//!
//! Given an unsorted array of integers nums, return the length of the longest 
//! consecutive elements sequence.
//!
//! You must write an algorithm that runs in O(n) time complexity.
//!
//! Constraints:
//! - 0 <= nums.length <= 10^5
//! - -10^9 <= nums[i] <= 10^9
//!
//! Example 1:
//! Input: nums = [100,4,200,1,3,2]
//! Output: 4
//! Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. 
//! Therefore its length is 4.
//!
//! Example 2:
//! Input: nums = [0,3,7,2,5,8,4,6,0,1]
//! Output: 9

use std::collections::{HashMap, HashSet};

pub struct Solution;

impl Solution {
    /// Approach 1: HashSet with Sequence Start Detection
    /// 
    /// Use HashSet for O(1) lookups. For each number, check if it's the start
    /// of a sequence (no num-1 exists), then count consecutive numbers.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    pub fn longest_consecutive_hashset(nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let num_set: HashSet<i32> = nums.into_iter().collect();
        let mut max_length = 0;
        
        for &num in &num_set {
            // Check if this is the start of a sequence
            if !num.checked_sub(1).map_or(false, |prev| num_set.contains(&prev)) {
                let mut current_num = num;
                let mut current_length = 1;
                
                // Count consecutive numbers
                while let Some(next) = current_num.checked_add(1) {
                    if num_set.contains(&next) {
                        current_num = next;
                        current_length += 1;
                    } else {
                        break;
                    }
                }
                
                max_length = max_length.max(current_length);
            }
        }
        
        max_length
    }
    
    /// Approach 2: Union-Find (Disjoint Set Union)
    /// 
    /// Use Union-Find to group consecutive numbers together.
    /// Each group represents a consecutive sequence.
    /// 
    /// Time Complexity: O(n α(n)) ≈ O(n)
    /// Space Complexity: O(n)
    pub fn longest_consecutive_union_find(nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let mut uf = UnionFind::new();
        let num_set: HashSet<i32> = nums.into_iter().collect();
        
        // Add all numbers to union-find
        for &num in &num_set {
            uf.add(num);
        }
        
        // Union consecutive numbers
        for &num in &num_set {
            if let Some(next) = num.checked_add(1) {
                if num_set.contains(&next) {
                    uf.union(num, next);
                }
            }
        }
        
        // Find the largest component size
        uf.max_component_size()
    }
    
    /// Approach 3: HashMap with Boundary Tracking
    /// 
    /// For each number, track the length of sequence ending at that number.
    /// Update boundaries when adding new numbers.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    pub fn longest_consecutive_boundary_map(nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let mut boundary_map = HashMap::new(); // num -> length of sequence ending at num
        let mut max_length = 0;
        
        for num in nums {
            if boundary_map.contains_key(&num) {
                continue; // Skip duplicates
            }
            
            let left_length = num.checked_sub(1)
                .and_then(|prev| boundary_map.get(&prev))
                .copied()
                .unwrap_or(0);
            let right_length = num.checked_add(1)
                .and_then(|next| boundary_map.get(&next))
                .copied()
                .unwrap_or(0);
            let total_length = left_length + right_length + 1;
            
            boundary_map.insert(num, total_length);
            max_length = max_length.max(total_length);
            
            // Update boundaries of the merged sequence
            if let Some(left_boundary) = num.checked_sub(left_length) {
                boundary_map.insert(left_boundary, total_length);
            }
            if let Some(right_boundary) = num.checked_add(right_length) {
                boundary_map.insert(right_boundary, total_length);
            }
        }
        
        max_length
    }
    
    /// Approach 4: Sorting with Deduplication
    /// 
    /// Sort the array and find longest consecutive subsequence.
    /// Note: This violates O(n) requirement but is included for completeness.
    /// 
    /// Time Complexity: O(n log n)
    /// Space Complexity: O(1) excluding input
    pub fn longest_consecutive_sorting(nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let mut sorted_nums = nums;
        sorted_nums.sort_unstable();
        
        let mut max_length = 1;
        let mut current_length = 1;
        
        for i in 1..sorted_nums.len() {
            if sorted_nums[i] == sorted_nums[i - 1] {
                continue; // Skip duplicates
            } else if sorted_nums[i] == sorted_nums[i - 1] + 1 {
                current_length += 1;
                max_length = max_length.max(current_length);
            } else {
                current_length = 1;
            }
        }
        
        max_length
    }
    
    /// Approach 5: HashMap with Range Merging (Fixed)
    /// 
    /// Use simpler approach: for each number, check left and right extensions.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    pub fn longest_consecutive_range_merge(nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let num_set: HashSet<i32> = nums.into_iter().collect();
        let mut visited = HashSet::new();
        let mut max_length = 0;
        
        for &num in &num_set {
            if visited.contains(&num) {
                continue;
            }
            
            let mut start = num;
            let mut end = num;
            
            // Expand left
            while let Some(prev) = start.checked_sub(1) {
                if num_set.contains(&prev) {
                    start = prev;
                } else {
                    break;
                }
            }
            
            // Expand right
            while let Some(next) = end.checked_add(1) {
                if num_set.contains(&next) {
                    end = next;
                } else {
                    break;
                }
            }
            
            // Mark all numbers in this range as visited
            for i in start..=end {
                visited.insert(i);
            }
            
            max_length = max_length.max(end - start + 1);
        }
        
        max_length
    }
    
    /// Approach 6: HashSet with Bidirectional Expansion
    /// 
    /// For each unvisited number, expand in both directions to find
    /// the full consecutive sequence.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    pub fn longest_consecutive_bidirectional(nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        
        let mut num_set: HashSet<i32> = nums.into_iter().collect();
        let mut max_length = 0;
        
        while !num_set.is_empty() {
            let &start_num = num_set.iter().next().unwrap();
            num_set.remove(&start_num);
            
            let mut length = 1;
            
            // Expand left
            let mut left = start_num;
            while let Some(prev) = left.checked_sub(1) {
                if num_set.contains(&prev) {
                    num_set.remove(&prev);
                    length += 1;
                    left = prev;
                } else {
                    break;
                }
            }
            
            // Expand right
            let mut right = start_num;
            while let Some(next) = right.checked_add(1) {
                if num_set.contains(&next) {
                    num_set.remove(&next);
                    length += 1;
                    right = next;
                } else {
                    break;
                }
            }
            
            max_length = max_length.max(length);
        }
        
        max_length
    }
}

struct UnionFind {
    parent: HashMap<i32, i32>,
    size: HashMap<i32, i32>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parent: HashMap::new(),
            size: HashMap::new(),
        }
    }
    
    fn add(&mut self, x: i32) {
        if !self.parent.contains_key(&x) {
            self.parent.insert(x, x);
            self.size.insert(x, 1);
        }
    }
    
    fn find(&mut self, x: i32) -> i32 {
        if self.parent[&x] != x {
            let root = self.find(self.parent[&x]);
            self.parent.insert(x, root);
        }
        self.parent[&x]
    }
    
    fn union(&mut self, x: i32, y: i32) {
        let root_x = self.find(x);
        let root_y = self.find(y);
        
        if root_x != root_y {
            let size_x = self.size[&root_x];
            let size_y = self.size[&root_y];
            
            if size_x < size_y {
                self.parent.insert(root_x, root_y);
                self.size.insert(root_y, size_x + size_y);
            } else {
                self.parent.insert(root_y, root_x);
                self.size.insert(root_x, size_x + size_y);
            }
        }
    }
    
    fn max_component_size(&mut self) -> i32 {
        let keys: Vec<i32> = self.parent.keys().copied().collect();
        let mut roots = HashSet::new();
        for key in keys {
            roots.insert(self.find(key));
        }
        roots.into_iter().map(|root| self.size[&root]).max().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_example_1() {
        let nums = vec![100, 4, 200, 1, 3, 2];
        assert_eq!(Solution::longest_consecutive_hashset(nums.clone()), 4);
        assert_eq!(Solution::longest_consecutive_union_find(nums), 4);
    }
    
    #[test]
    fn test_example_2() {
        let nums = vec![0, 3, 7, 2, 5, 8, 4, 6, 0, 1];
        assert_eq!(Solution::longest_consecutive_boundary_map(nums.clone()), 9);
        assert_eq!(Solution::longest_consecutive_sorting(nums), 9);
    }
    
    #[test]
    fn test_empty_array() {
        assert_eq!(Solution::longest_consecutive_hashset(vec![]), 0);
        assert_eq!(Solution::longest_consecutive_range_merge(vec![]), 0);
    }
    
    #[test]
    fn test_single_element() {
        assert_eq!(Solution::longest_consecutive_bidirectional(vec![1]), 1);
        assert_eq!(Solution::longest_consecutive_union_find(vec![1]), 1);
    }
    
    #[test]
    fn test_no_consecutive() {
        let nums = vec![1, 3, 5, 7, 9];
        assert_eq!(Solution::longest_consecutive_hashset(nums.clone()), 1);
        assert_eq!(Solution::longest_consecutive_boundary_map(nums), 1);
    }
    
    #[test]
    fn test_all_consecutive() {
        let nums = vec![1, 2, 3, 4, 5];
        assert_eq!(Solution::longest_consecutive_range_merge(nums.clone()), 5);
        assert_eq!(Solution::longest_consecutive_sorting(nums), 5);
    }
    
    #[test]
    fn test_duplicates() {
        let nums = vec![1, 2, 2, 3, 4, 4, 5];
        assert_eq!(Solution::longest_consecutive_bidirectional(nums.clone()), 5);
        assert_eq!(Solution::longest_consecutive_union_find(nums), 5);
    }
    
    #[test]
    fn test_negative_numbers() {
        let nums = vec![-1, -2, -3, 0, 1, 2];
        assert_eq!(Solution::longest_consecutive_hashset(nums.clone()), 6);
        assert_eq!(Solution::longest_consecutive_boundary_map(nums), 6);
    }
    
    #[test]
    fn test_mixed_sequence() {
        let nums = vec![1, 9, 3, 10, 4, 20, 2];
        assert_eq!(Solution::longest_consecutive_range_merge(nums.clone()), 4);
        assert_eq!(Solution::longest_consecutive_sorting(nums), 4);
    }
    
    #[test]
    fn test_large_gap() {
        let nums = vec![1, 1000000, 2, 999999, 3];
        assert_eq!(Solution::longest_consecutive_bidirectional(nums.clone()), 3);
        assert_eq!(Solution::longest_consecutive_union_find(nums), 3);
    }
    
    #[test]
    fn test_reverse_order() {
        let nums = vec![5, 4, 3, 2, 1];
        assert_eq!(Solution::longest_consecutive_hashset(nums.clone()), 5);
        assert_eq!(Solution::longest_consecutive_boundary_map(nums), 5);
    }
    
    #[test]
    fn test_multiple_sequences() {
        let nums = vec![1, 2, 3, 10, 11, 12, 13, 20, 21];
        assert_eq!(Solution::longest_consecutive_range_merge(nums.clone()), 4);
        assert_eq!(Solution::longest_consecutive_sorting(nums), 4);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            vec![100, 4, 200, 1, 3, 2],
            vec![0, 3, 7, 2, 5, 8, 4, 6, 0, 1],
            vec![],
            vec![1],
            vec![1, 3, 5, 7, 9],
            vec![1, 2, 3, 4, 5],
            vec![1, 2, 2, 3, 4, 4, 5],
            vec![-1, -2, -3, 0, 1, 2],
            vec![1, 9, 3, 10, 4, 20, 2],
            vec![1, 1000000, 2, 999999, 3],
            vec![5, 4, 3, 2, 1],
            vec![1, 2, 3, 10, 11, 12, 13, 20, 21],
            vec![-5, -4, -3, -2, -1],
            vec![0],
            vec![1, 1, 1, 1],
            vec![2147483647, -2147483648, 0],
        ];
        
        for nums in test_cases {
            let result1 = Solution::longest_consecutive_hashset(nums.clone());
            let result2 = Solution::longest_consecutive_union_find(nums.clone());
            let result3 = Solution::longest_consecutive_boundary_map(nums.clone());
            let result4 = Solution::longest_consecutive_sorting(nums.clone());
            let result5 = Solution::longest_consecutive_range_merge(nums.clone());
            let result6 = Solution::longest_consecutive_bidirectional(nums.clone());
            
            assert_eq!(result1, result2, "HashSet vs UnionFind mismatch for {:?}", nums);
            assert_eq!(result2, result3, "UnionFind vs BoundaryMap mismatch for {:?}", nums);
            assert_eq!(result3, result4, "BoundaryMap vs Sorting mismatch for {:?}", nums);
            assert_eq!(result4, result5, "Sorting vs RangeMerge mismatch for {:?}", nums);
            assert_eq!(result5, result6, "RangeMerge vs Bidirectional mismatch for {:?}", nums);
        }
    }
}