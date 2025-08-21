//! Problem 56: Merge Intervals
//! 
//! Given an array of intervals where intervals[i] = [start_i, end_i], merge all overlapping intervals,
//! and return an array of the non-overlapping intervals that cover all the intervals in the input.
//! 
//! Example 1:
//! Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
//! Output: [[1,6],[8,10],[15,18]]
//! Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
//! 
//! Example 2:
//! Input: intervals = [[1,4],[4,5]]
//! Output: [[1,5]]
//! Explanation: Intervals [1,4] and [4,5] are considered overlapping.

use std::collections::BTreeMap;

pub struct Solution;

impl Solution {
    /// Approach 1: Sort and Merge
    /// 
    /// The classic approach: sort intervals by start time, then iterate and merge overlapping ones.
    /// Two intervals [a,b] and [c,d] overlap if b >= c (assuming a <= c after sorting).
    /// 
    /// Time Complexity: O(n log n) for sorting + O(n) for merging = O(n log n)
    /// Space Complexity: O(1) excluding output space
    pub fn merge_sort_and_merge(&self, mut intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        if intervals.is_empty() {
            return Vec::new();
        }
        
        // Sort intervals by start time
        intervals.sort_by_key(|interval| interval[0]);
        
        let mut result = Vec::new();
        let mut current = intervals[0].clone();
        
        for interval in intervals.into_iter().skip(1) {
            // If current interval overlaps with the next interval
            if current[1] >= interval[0] {
                // Merge them by extending the end time
                current[1] = current[1].max(interval[1]);
            } else {
                // No overlap, add current to result and move to next
                result.push(current);
                current = interval;
            }
        }
        
        // Don't forget the last interval
        result.push(current);
        result
    }
    
    /// Approach 2: Stack-based Merging
    /// 
    /// Uses a stack to maintain the merged intervals. For each new interval,
    /// check if it overlaps with the top of the stack.
    /// 
    /// Time Complexity: O(n log n) for sorting + O(n) for processing
    /// Space Complexity: O(n) for the stack
    pub fn merge_stack(&self, mut intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        if intervals.is_empty() {
            return Vec::new();
        }
        
        intervals.sort_by_key(|interval| interval[0]);
        
        let mut stack: Vec<Vec<i32>> = Vec::new();
        
        for interval in intervals {
            if stack.is_empty() || stack.last().unwrap()[1] < interval[0] {
                // No overlap, push new interval
                stack.push(interval);
            } else {
                // Overlap detected, merge with the top of stack
                let last_idx = stack.len() - 1;
                stack[last_idx][1] = stack[last_idx][1].max(interval[1]);
            }
        }
        
        stack
    }
    
    /// Approach 3: Sweep Line Algorithm
    /// 
    /// Creates events for interval starts and ends, then processes them in order.
    /// Maintains a count of active intervals to determine when merging should occur.
    /// 
    /// Time Complexity: O(n log n) for sorting events
    /// Space Complexity: O(n) for events storage
    pub fn merge_sweep_line(&self, intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        if intervals.is_empty() {
            return Vec::new();
        }
        
        let mut events = Vec::new();
        
        // Create events: (time, type, original_index)
        // type: 0 = start, 1 = end
        for (i, interval) in intervals.iter().enumerate() {
            events.push((interval[0], 0, i)); // start event
            events.push((interval[1], 1, i)); // end event
        }
        
        // Sort events by time, with start events before end events for same time
        events.sort();
        
        let mut result = Vec::new();
        let mut active_count = 0;
        let mut merge_start = 0;
        
        for (time, event_type, _) in events {
            if event_type == 0 { // start event
                if active_count == 0 {
                    merge_start = time;
                }
                active_count += 1;
            } else { // end event
                active_count -= 1;
                if active_count == 0 {
                    result.push(vec![merge_start, time]);
                }
            }
        }
        
        result
    }
    
    /// Approach 4: Coordinate Compression with Array
    /// 
    /// For intervals with bounded coordinates, we can use coordinate compression
    /// and mark active ranges in an array, then reconstruct intervals.
    /// 
    /// Time Complexity: O(n log n) for sorting + O(R) where R is coordinate range
    /// Space Complexity: O(R) for the range array
    pub fn merge_coordinate_compression(&self, intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        if intervals.is_empty() {
            return Vec::new();
        }
        
        // Get all unique coordinates
        let mut coords = std::collections::BTreeSet::new();
        for interval in &intervals {
            coords.insert(interval[0]);
            coords.insert(interval[1]);
        }
        
        let coord_vec: Vec<i32> = coords.into_iter().collect();
        let coord_to_idx: std::collections::HashMap<i32, usize> = 
            coord_vec.iter().enumerate().map(|(i, &x)| (x, i)).collect();
        
        // Mark active ranges
        let mut active = vec![false; coord_vec.len()];
        for interval in &intervals {
            let start_idx = coord_to_idx[&interval[0]];
            let end_idx = coord_to_idx[&interval[1]];
            
            for i in start_idx..end_idx {
                active[i] = true;
            }
        }
        
        // Reconstruct intervals from active ranges
        let mut result = Vec::new();
        let mut i = 0;
        
        while i < active.len() {
            if active[i] {
                let start = coord_vec[i];
                while i < active.len() && active[i] {
                    i += 1;
                }
                let end = coord_vec[i.min(coord_vec.len() - 1)];
                result.push(vec![start, end]);
            } else {
                i += 1;
            }
        }
        
        result
    }
    
    /// Approach 5: Union-Find (Disjoint Set)
    /// 
    /// Treats overlapping intervals as connected components.
    /// Each interval is a node, and overlapping intervals are connected.
    /// 
    /// Time Complexity: O(n²) for checking all pairs + O(n α(n)) for union-find
    /// Space Complexity: O(n) for union-find structure
    pub fn merge_union_find(&self, intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        if intervals.is_empty() {
            return Vec::new();
        }
        
        let n = intervals.len();
        let mut parent = (0..n).collect::<Vec<usize>>();
        
        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }
        
        fn union(parent: &mut Vec<usize>, x: usize, y: usize) {
            let px = find(parent, x);
            let py = find(parent, y);
            if px != py {
                parent[px] = py;
            }
        }
        
        // Union overlapping intervals
        for i in 0..n {
            for j in i + 1..n {
                // Check if intervals[i] and intervals[j] overlap
                let max_start = intervals[i][0].max(intervals[j][0]);
                let min_end = intervals[i][1].min(intervals[j][1]);
                
                if max_start <= min_end {
                    union(&mut parent, i, j);
                }
            }
        }
        
        // Group intervals by their root parent
        let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            groups.entry(root).or_insert(Vec::new()).push(i);
        }
        
        // Merge intervals in each group
        let mut result = Vec::new();
        for indices in groups.values() {
            let mut min_start = i32::MAX;
            let mut max_end = i32::MIN;
            
            for &idx in indices {
                min_start = min_start.min(intervals[idx][0]);
                max_end = max_end.max(intervals[idx][1]);
            }
            
            result.push(vec![min_start, max_end]);
        }
        
        // Sort result by start time
        result.sort_by_key(|interval| interval[0]);
        result
    }
    
    /// Approach 6: TreeMap-based Range Merging
    /// 
    /// Uses a TreeMap to maintain non-overlapping intervals and efficiently
    /// find and merge overlapping ranges as new intervals are added.
    /// 
    /// Time Complexity: O(n log n) for n insertions into TreeMap
    /// Space Complexity: O(n) for TreeMap storage
    pub fn merge_treemap(&self, intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        if intervals.is_empty() {
            return Vec::new();
        }
        
        let mut tree_map: BTreeMap<i32, i32> = BTreeMap::new();
        
        for interval in intervals {
            let start = interval[0];
            let end = interval[1];
            
            // Find all intervals that overlap with [start, end]
            let mut to_remove = Vec::new();
            let mut new_start = start;
            let mut new_end = end;
            
            // Check intervals that might overlap
            for (&existing_start, &existing_end) in &tree_map {
                if existing_end < start {
                    continue; // No overlap, too early
                }
                if existing_start > end {
                    break; // No overlap, too late
                }
                
                // Overlap found
                new_start = new_start.min(existing_start);
                new_end = new_end.max(existing_end);
                to_remove.push(existing_start);
            }
            
            // Remove overlapping intervals
            for key in to_remove {
                tree_map.remove(&key);
            }
            
            // Insert merged interval
            tree_map.insert(new_start, new_end);
        }
        
        // Convert TreeMap to result vector
        tree_map.into_iter()
            .map(|(start, end)| vec![start, end])
            .collect()
    }
    
    /// Helper function to check if two intervals overlap
    fn intervals_overlap(a: &[i32], b: &[i32]) -> bool {
        a[0].max(b[0]) <= a[1].min(b[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn sort_intervals(mut intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        intervals.sort();
        intervals
    }
    
    #[test]
    fn test_sort_and_merge() {
        let solution = Solution;
        
        assert_eq!(
            sort_intervals(solution.merge_sort_and_merge(vec![vec![1,3], vec![2,6], vec![8,10], vec![15,18]])),
            vec![vec![1,6], vec![8,10], vec![15,18]]
        );
        
        assert_eq!(
            sort_intervals(solution.merge_sort_and_merge(vec![vec![1,4], vec![4,5]])),
            vec![vec![1,5]]
        );
        
        assert_eq!(
            sort_intervals(solution.merge_sort_and_merge(vec![vec![1,3]])),
            vec![vec![1,3]]
        );
        
        assert_eq!(
            solution.merge_sort_and_merge(vec![]),
            Vec::<Vec<i32>>::new()
        );
    }
    
    #[test]
    fn test_stack() {
        let solution = Solution;
        
        assert_eq!(
            sort_intervals(solution.merge_stack(vec![vec![1,3], vec![2,6], vec![8,10], vec![15,18]])),
            vec![vec![1,6], vec![8,10], vec![15,18]]
        );
        
        assert_eq!(
            sort_intervals(solution.merge_stack(vec![vec![1,4], vec![4,5]])),
            vec![vec![1,5]]
        );
    }
    
    #[test]
    fn test_sweep_line() {
        let solution = Solution;
        
        assert_eq!(
            sort_intervals(solution.merge_sweep_line(vec![vec![1,3], vec![2,6], vec![8,10], vec![15,18]])),
            vec![vec![1,6], vec![8,10], vec![15,18]]
        );
        
        assert_eq!(
            sort_intervals(solution.merge_sweep_line(vec![vec![1,4], vec![4,5]])),
            vec![vec![1,5]]
        );
    }
    
    #[test]
    fn test_coordinate_compression() {
        let solution = Solution;
        
        assert_eq!(
            sort_intervals(solution.merge_coordinate_compression(vec![vec![1,3], vec![2,6], vec![8,10], vec![15,18]])),
            vec![vec![1,6], vec![8,10], vec![15,18]]
        );
        
        assert_eq!(
            sort_intervals(solution.merge_coordinate_compression(vec![vec![1,4], vec![4,5]])),
            vec![vec![1,5]]
        );
    }
    
    #[test]
    fn test_union_find() {
        let solution = Solution;
        
        assert_eq!(
            sort_intervals(solution.merge_union_find(vec![vec![1,3], vec![2,6], vec![8,10], vec![15,18]])),
            vec![vec![1,6], vec![8,10], vec![15,18]]
        );
        
        assert_eq!(
            sort_intervals(solution.merge_union_find(vec![vec![1,4], vec![4,5]])),
            vec![vec![1,5]]
        );
    }
    
    #[test]
    fn test_treemap() {
        let solution = Solution;
        
        assert_eq!(
            sort_intervals(solution.merge_treemap(vec![vec![1,3], vec![2,6], vec![8,10], vec![15,18]])),
            vec![vec![1,6], vec![8,10], vec![15,18]]
        );
        
        assert_eq!(
            sort_intervals(solution.merge_treemap(vec![vec![1,4], vec![4,5]])),
            vec![vec![1,5]]
        );
    }
    
    #[test]
    fn test_edge_cases() {
        let solution = Solution;
        
        // Single interval
        assert_eq!(
            solution.merge_sort_and_merge(vec![vec![1,4]]),
            vec![vec![1,4]]
        );
        
        // No overlapping intervals
        assert_eq!(
            solution.merge_sort_and_merge(vec![vec![1,2], vec![3,4], vec![5,6]]),
            vec![vec![1,2], vec![3,4], vec![5,6]]
        );
        
        // All intervals overlap
        assert_eq!(
            solution.merge_sort_and_merge(vec![vec![1,4], vec![2,5], vec![3,6]]),
            vec![vec![1,6]]
        );
        
        // Intervals with same start
        assert_eq!(
            sort_intervals(solution.merge_sort_and_merge(vec![vec![1,3], vec![1,5], vec![6,7]])),
            vec![vec![1,5], vec![6,7]]
        );
        
        // Intervals with same end
        assert_eq!(
            sort_intervals(solution.merge_sort_and_merge(vec![vec![1,3], vec![2,3], vec![4,5]])),
            vec![vec![1,3], vec![4,5]]
        );
    }
    
    #[test]
    fn test_complex_merging() {
        let solution = Solution;
        
        // Multiple merges in sequence
        assert_eq!(
            solution.merge_sort_and_merge(vec![vec![1,3], vec![2,6], vec![5,10], vec![9,12], vec![15,18]]),
            vec![vec![1,12], vec![15,18]]
        );
        
        // Nested intervals
        assert_eq!(
            solution.merge_sort_and_merge(vec![vec![1,10], vec![2,3], vec![4,5], vec![6,7], vec![8,9]]),
            vec![vec![1,10]]
        );
        
        // Unsorted input
        assert_eq!(
            sort_intervals(solution.merge_sort_and_merge(vec![vec![8,10], vec![1,3], vec![15,18], vec![2,6]])),
            vec![vec![1,6], vec![8,10], vec![15,18]]
        );
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let solution = Solution;
        
        let test_cases = vec![
            vec![vec![1,3], vec![2,6], vec![8,10], vec![15,18]],
            vec![vec![1,4], vec![4,5]],
            vec![vec![1,3]],
            vec![],
            vec![vec![1,4], vec![2,5], vec![3,6]],
            vec![vec![1,2], vec![3,4], vec![5,6]],
            vec![vec![8,10], vec![1,3], vec![15,18], vec![2,6]],
        ];
        
        for intervals in test_cases {
            let sort_merge = sort_intervals(solution.merge_sort_and_merge(intervals.clone()));
            let stack = sort_intervals(solution.merge_stack(intervals.clone()));
            let sweep_line = sort_intervals(solution.merge_sweep_line(intervals.clone()));
            let union_find = sort_intervals(solution.merge_union_find(intervals.clone()));
            let treemap = sort_intervals(solution.merge_treemap(intervals.clone()));
            
            assert_eq!(sort_merge, stack, "Sort-merge and stack differ for {:?}", intervals);
            assert_eq!(sort_merge, sweep_line, "Sort-merge and sweep-line differ for {:?}", intervals);
            assert_eq!(sort_merge, union_find, "Sort-merge and union-find differ for {:?}", intervals);
            assert_eq!(sort_merge, treemap, "Sort-merge and treemap differ for {:?}", intervals);
        }
    }
    
    #[test]
    fn test_large_ranges() {
        let solution = Solution;
        
        // Large coordinate values
        let intervals = vec![vec![1000000, 2000000], vec![1500000, 2500000], vec![3000000, 4000000]];
        let expected = vec![vec![1000000, 2500000], vec![3000000, 4000000]];
        
        assert_eq!(sort_intervals(solution.merge_sort_and_merge(intervals.clone())), expected);
        assert_eq!(sort_intervals(solution.merge_stack(intervals.clone())), expected);
        assert_eq!(sort_intervals(solution.merge_sweep_line(intervals.clone())), expected);
        assert_eq!(sort_intervals(solution.merge_union_find(intervals.clone())), expected);
        assert_eq!(sort_intervals(solution.merge_treemap(intervals)), expected);
    }
    
    #[test]
    fn test_point_intervals() {
        let solution = Solution;
        
        // Point intervals (start == end) and adjacent intervals
        let intervals = vec![vec![1,1], vec![2,2], vec![1,2]];
        // [1,1] and [1,2] overlap (since 1 >= 1), resulting in [1,2]
        // [2,2] touches [1,2] at endpoint 2, so they merge into [1,2]
        let expected = vec![vec![1,2]];
        
        assert_eq!(sort_intervals(solution.merge_sort_and_merge(intervals)), expected);
    }
}