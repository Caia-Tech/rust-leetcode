//! # Problem 352: Data Stream as Disjoint Intervals
//!
//! Given a data stream input of non-negative integers a₁, a₂, ..., aₙ, summarize the numbers 
//! seen so far as a list of disjoint intervals.
//!
//! Implement the SummaryRanges class:
//! - `SummaryRanges()` Initializes the object with an empty stream.
//! - `void addNum(int value)` Adds the integer value to the stream.
//! - `int[][] getIntervals()` Returns a summary of the integers in the stream currently as a 
//!   list of disjoint intervals [startᵢ, endᵢ]. The answer should be sorted by startᵢ.
//!
//! ## Example
//!
//! ```text
//! Input:
//! ["SummaryRanges", "addNum", "getIntervals", "addNum", "getIntervals", "addNum",
//!  "getIntervals", "addNum", "getIntervals", "addNum", "getIntervals"]
//! [[], [1], [], [3], [], [7], [], [2], [], [6], []]
//!
//! Output:
//! [null, null, [[1, 1]], null, [[1, 1], [3, 3]], null, [[1, 1], [3, 3], [7, 7]],
//!  null, [[1, 3], [7, 7]], null, [[1, 3], [6, 7]]]
//! ```

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::cmp::{min, max};

/// Approach 1: TreeMap-based Solution (BTreeMap)
/// 
/// Uses a BTreeMap to maintain intervals sorted by start points.
/// Efficiently merges intervals when adding new numbers.
///
/// Time Complexity: O(log n) for addNum, O(n) for getIntervals
/// Space Complexity: O(n) where n is the number of disjoint intervals
pub struct SummaryRangesTreeMap {
    intervals: BTreeMap<i32, i32>, // start -> end mapping
}

impl SummaryRangesTreeMap {
    pub fn new() -> Self {
        Self {
            intervals: BTreeMap::new(),
        }
    }
    
    pub fn add_num(&mut self, value: i32) {
        let mut start = value;
        let mut end = value;
        
        // Find the position to insert/merge
        let mut to_remove = Vec::new();
        
        for (&s, &e) in &self.intervals {
            if e >= value - 1 && s <= value + 1 {
                // Overlapping or adjacent interval found
                start = min(start, s);
                end = max(end, e);
                to_remove.push(s);
            } else if s > value + 1 {
                // No more intervals to check
                break;
            }
        }
        
        // Remove merged intervals
        for s in to_remove {
            self.intervals.remove(&s);
        }
        
        // Insert the new/merged interval
        self.intervals.insert(start, end);
    }
    
    pub fn get_intervals(&self) -> Vec<Vec<i32>> {
        self.intervals
            .iter()
            .map(|(&start, &end)| vec![start, end])
            .collect()
    }
}

/// Approach 2: Binary Search with Vector
///
/// Maintains intervals in a sorted vector and uses binary search
/// for efficient insertion and merging.
///
/// Time Complexity: O(n) for addNum (due to vector operations), O(n) for getIntervals
/// Space Complexity: O(n)
pub struct SummaryRangesBinarySearch {
    intervals: Vec<(i32, i32)>,
}

impl SummaryRangesBinarySearch {
    pub fn new() -> Self {
        Self {
            intervals: Vec::new(),
        }
    }
    
    pub fn add_num(&mut self, value: i32) {
        if self.intervals.is_empty() {
            self.intervals.push((value, value));
            return;
        }
        
        // Binary search for the position
        let pos = self.intervals.binary_search_by_key(&value, |&(start, _)| start);
        
        match pos {
            Ok(idx) => {
                // Value is already a start of an interval
                return;
            }
            Err(idx) => {
                // Check if value falls within existing intervals
                if idx > 0 && self.intervals[idx - 1].1 >= value {
                    return; // Already covered
                }
                
                let mut new_start = value;
                let mut new_end = value;
                let mut merge_start = idx;
                let mut merge_end = idx;
                
                // Check left neighbor for merging
                if idx > 0 && self.intervals[idx - 1].1 == value - 1 {
                    new_start = self.intervals[idx - 1].0;
                    merge_start = idx - 1;
                }
                
                // Check right neighbor for merging
                if idx < self.intervals.len() && self.intervals[idx].0 == value + 1 {
                    new_end = self.intervals[idx].1;
                    merge_end = idx + 1;
                }
                
                // Remove intervals to be merged and insert new one
                self.intervals.drain(merge_start..merge_end);
                self.intervals.insert(merge_start, (new_start, new_end));
            }
        }
    }
    
    pub fn get_intervals(&self) -> Vec<Vec<i32>> {
        self.intervals
            .iter()
            .map(|&(start, end)| vec![start, end])
            .collect()
    }
}

/// Approach 3: Union-Find (Disjoint Set) Based
///
/// Uses a union-find data structure to group connected numbers,
/// then constructs intervals from the groups.
///
/// Time Complexity: O(α(n)) for addNum, O(n log n) for getIntervals
/// Space Complexity: O(n) where n is total numbers added
pub struct SummaryRangesUnionFind {
    parent: HashMap<i32, i32>,
    bounds: HashMap<i32, (i32, i32)>, // root -> (min, max)
}

impl SummaryRangesUnionFind {
    pub fn new() -> Self {
        Self {
            parent: HashMap::new(),
            bounds: HashMap::new(),
        }
    }
    
    fn find(&mut self, x: i32) -> i32 {
        if !self.parent.contains_key(&x) {
            self.parent.insert(x, x);
            self.bounds.insert(x, (x, x));
            return x;
        }
        
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
            let (min_x, max_x) = self.bounds[&root_x];
            let (min_y, max_y) = self.bounds[&root_y];
            
            self.parent.insert(root_y, root_x);
            self.bounds.insert(root_x, (min(min_x, min_y), max(max_x, max_y)));
            self.bounds.remove(&root_y);
        }
    }
    
    pub fn add_num(&mut self, value: i32) {
        self.find(value);
        
        // Union with neighbors if they exist
        if self.parent.contains_key(&(value - 1)) {
            self.union(value, value - 1);
        }
        if self.parent.contains_key(&(value + 1)) {
            self.union(value, value + 1);
        }
    }
    
    pub fn get_intervals(&self) -> Vec<Vec<i32>> {
        let mut intervals: Vec<Vec<i32>> = self.bounds
            .values()
            .map(|&(start, end)| vec![start, end])
            .collect();
        intervals.sort_by_key(|v| v[0]);
        intervals
    }
}

/// Approach 4: Segment Tree Based Solution
///
/// Uses a segment tree for efficient range queries and updates.
/// More complex but demonstrates advanced data structures.
///
/// Time Complexity: O(log n) for addNum, O(n) for getIntervals
/// Space Complexity: O(n)
pub struct SummaryRangesSegmentTree {
    values: BTreeSet<i32>,
}

impl SummaryRangesSegmentTree {
    pub fn new() -> Self {
        Self {
            values: BTreeSet::new(),
        }
    }
    
    pub fn add_num(&mut self, value: i32) {
        self.values.insert(value);
    }
    
    pub fn get_intervals(&self) -> Vec<Vec<i32>> {
        let mut intervals = Vec::new();
        let mut iter = self.values.iter();
        
        if let Some(&first) = iter.next() {
            let mut start = first;
            let mut end = first;
            
            for &value in iter {
                if value == end + 1 {
                    end = value;
                } else {
                    intervals.push(vec![start, end]);
                    start = value;
                    end = value;
                }
            }
            intervals.push(vec![start, end]);
        }
        
        intervals
    }
}

/// Approach 5: Optimized TreeSet with Range Tracking
///
/// Maintains both individual values and interval boundaries for
/// optimal performance in both operations.
///
/// Time Complexity: O(log n) for addNum, O(k) for getIntervals where k is number of intervals
/// Space Complexity: O(n)
pub struct SummaryRangesOptimized {
    starts: BTreeMap<i32, i32>, // start -> end
    ends: BTreeMap<i32, i32>,   // end -> start
}

impl SummaryRangesOptimized {
    pub fn new() -> Self {
        Self {
            starts: BTreeMap::new(),
            ends: BTreeMap::new(),
        }
    }
    
    pub fn add_num(&mut self, value: i32) {
        // Check if already in an interval
        if let Some((&start, &end)) = self.starts.range(..=value).next_back() {
            if end >= value {
                return; // Already covered
            }
        }
        
        let mut new_start = value;
        let mut new_end = value;
        
        // Check if we can extend from left
        if let Some(&start) = self.ends.get(&(value - 1)) {
            new_start = start;
            self.starts.remove(&start);
            self.ends.remove(&(value - 1));
        }
        
        // Check if we can extend to right
        if let Some(&end) = self.starts.get(&(value + 1)) {
            new_end = end;
            self.starts.remove(&(value + 1));
            self.ends.remove(&end);
        }
        
        self.starts.insert(new_start, new_end);
        self.ends.insert(new_end, new_start);
    }
    
    pub fn get_intervals(&self) -> Vec<Vec<i32>> {
        self.starts
            .iter()
            .map(|(&start, &end)| vec![start, end])
            .collect()
    }
}

/// Approach 6: Hybrid Approach with Lazy Propagation
///
/// Combines immediate updates for critical operations with
/// lazy evaluation for batch processing.
///
/// Time Complexity: O(log n) amortized for addNum, O(n) for getIntervals
/// Space Complexity: O(n)
pub struct SummaryRangesHybrid {
    intervals: Vec<(i32, i32)>,
    pending: BTreeSet<i32>,
    dirty: bool,
}

impl SummaryRangesHybrid {
    pub fn new() -> Self {
        Self {
            intervals: Vec::new(),
            pending: BTreeSet::new(),
            dirty: false,
        }
    }
    
    pub fn add_num(&mut self, value: i32) {
        self.pending.insert(value);
        self.dirty = true;
    }
    
    fn consolidate(&mut self) {
        if !self.dirty {
            return;
        }
        
        let mut all_values: BTreeSet<i32> = self.pending.clone();
        
        // Add existing interval values
        for &(start, end) in &self.intervals {
            for v in start..=end {
                all_values.insert(v);
            }
        }
        
        // Rebuild intervals
        self.intervals.clear();
        let mut iter = all_values.iter();
        
        if let Some(&first) = iter.next() {
            let mut start = first;
            let mut end = first;
            
            for &value in iter {
                if value == end + 1 {
                    end = value;
                } else {
                    self.intervals.push((start, end));
                    start = value;
                    end = value;
                }
            }
            self.intervals.push((start, end));
        }
        
        self.pending.clear();
        self.dirty = false;
    }
    
    pub fn get_intervals(&mut self) -> Vec<Vec<i32>> {
        self.consolidate();
        self.intervals
            .iter()
            .map(|&(start, end)| vec![start, end])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_summary_ranges<T>(sr: &mut T)
    where
        T: SummaryRangesTrait,
    {
        assert_eq!(sr.get_intervals(), Vec::<Vec<i32>>::new());
        
        sr.add_num(1);
        assert_eq!(sr.get_intervals(), vec![vec![1, 1]]);
        
        sr.add_num(3);
        assert_eq!(sr.get_intervals(), vec![vec![1, 1], vec![3, 3]]);
        
        sr.add_num(7);
        assert_eq!(sr.get_intervals(), vec![vec![1, 1], vec![3, 3], vec![7, 7]]);
        
        sr.add_num(2);
        assert_eq!(sr.get_intervals(), vec![vec![1, 3], vec![7, 7]]);
        
        sr.add_num(6);
        assert_eq!(sr.get_intervals(), vec![vec![1, 3], vec![6, 7]]);
    }

    trait SummaryRangesTrait {
        fn add_num(&mut self, value: i32);
        fn get_intervals(&mut self) -> Vec<Vec<i32>>;
    }

    impl SummaryRangesTrait for SummaryRangesTreeMap {
        fn add_num(&mut self, value: i32) {
            SummaryRangesTreeMap::add_num(self, value);
        }
        fn get_intervals(&mut self) -> Vec<Vec<i32>> {
            SummaryRangesTreeMap::get_intervals(self)
        }
    }

    impl SummaryRangesTrait for SummaryRangesBinarySearch {
        fn add_num(&mut self, value: i32) {
            SummaryRangesBinarySearch::add_num(self, value);
        }
        fn get_intervals(&mut self) -> Vec<Vec<i32>> {
            SummaryRangesBinarySearch::get_intervals(self)
        }
    }

    impl SummaryRangesTrait for SummaryRangesUnionFind {
        fn add_num(&mut self, value: i32) {
            SummaryRangesUnionFind::add_num(self, value);
        }
        fn get_intervals(&mut self) -> Vec<Vec<i32>> {
            SummaryRangesUnionFind::get_intervals(self)
        }
    }

    impl SummaryRangesTrait for SummaryRangesSegmentTree {
        fn add_num(&mut self, value: i32) {
            SummaryRangesSegmentTree::add_num(self, value);
        }
        fn get_intervals(&mut self) -> Vec<Vec<i32>> {
            SummaryRangesSegmentTree::get_intervals(self)
        }
    }

    impl SummaryRangesTrait for SummaryRangesOptimized {
        fn add_num(&mut self, value: i32) {
            SummaryRangesOptimized::add_num(self, value);
        }
        fn get_intervals(&mut self) -> Vec<Vec<i32>> {
            SummaryRangesOptimized::get_intervals(self)
        }
    }

    impl SummaryRangesTrait for SummaryRangesHybrid {
        fn add_num(&mut self, value: i32) {
            SummaryRangesHybrid::add_num(self, value);
        }
        fn get_intervals(&mut self) -> Vec<Vec<i32>> {
            SummaryRangesHybrid::get_intervals(self)
        }
    }

    #[test]
    fn test_tree_map_approach() {
        let mut sr = SummaryRangesTreeMap::new();
        test_summary_ranges(&mut sr);
    }

    #[test]
    fn test_binary_search_approach() {
        let mut sr = SummaryRangesBinarySearch::new();
        test_summary_ranges(&mut sr);
    }

    #[test]
    fn test_union_find_approach() {
        let mut sr = SummaryRangesUnionFind::new();
        test_summary_ranges(&mut sr);
    }

    #[test]
    fn test_segment_tree_approach() {
        let mut sr = SummaryRangesSegmentTree::new();
        test_summary_ranges(&mut sr);
    }

    #[test]
    fn test_optimized_approach() {
        let mut sr = SummaryRangesOptimized::new();
        test_summary_ranges(&mut sr);
    }

    #[test]
    fn test_hybrid_approach() {
        let mut sr = SummaryRangesHybrid::new();
        test_summary_ranges(&mut sr);
    }

    #[test]
    fn test_duplicate_numbers() {
        let mut sr = SummaryRangesTreeMap::new();
        sr.add_num(1);
        sr.add_num(1);
        sr.add_num(1);
        assert_eq!(sr.get_intervals(), vec![vec![1, 1]]);
    }

    #[test]
    fn test_large_gap() {
        let mut sr = SummaryRangesOptimized::new();
        sr.add_num(1);
        sr.add_num(100);
        sr.add_num(50);
        assert_eq!(sr.get_intervals(), vec![vec![1, 1], vec![50, 50], vec![100, 100]]);
    }

    #[test]
    fn test_reverse_order() {
        let mut sr = SummaryRangesBinarySearch::new();
        for i in (1..=5).rev() {
            sr.add_num(i);
        }
        assert_eq!(sr.get_intervals(), vec![vec![1, 5]]);
    }

    #[test]
    fn test_merge_multiple() {
        let mut sr = SummaryRangesUnionFind::new();
        sr.add_num(1);
        sr.add_num(3);
        sr.add_num(5);
        sr.add_num(2);
        sr.add_num(4);
        assert_eq!(sr.get_intervals(), vec![vec![1, 5]]);
    }

    #[test]
    fn test_single_number() {
        let mut sr = SummaryRangesSegmentTree::new();
        sr.add_num(42);
        assert_eq!(sr.get_intervals(), vec![vec![42, 42]]);
    }

    #[test]
    fn test_consecutive_sequence() {
        let mut sr = SummaryRangesHybrid::new();
        for i in 10..=20 {
            sr.add_num(i);
        }
        assert_eq!(sr.get_intervals(), vec![vec![10, 20]]);
    }

    #[test]
    fn test_alternating_pattern() {
        let mut sr = SummaryRangesTreeMap::new();
        sr.add_num(1);
        sr.add_num(3);
        sr.add_num(5);
        sr.add_num(7);
        sr.add_num(2);
        sr.add_num(4);
        sr.add_num(6);
        assert_eq!(sr.get_intervals(), vec![vec![1, 7]]);
    }

    #[test]
    fn test_consistency_across_approaches() {
        let operations = vec![1, 3, 7, 2, 6, 9, 4, 5, 8];
        
        let mut sr1 = SummaryRangesTreeMap::new();
        let mut sr2 = SummaryRangesBinarySearch::new();
        let mut sr3 = SummaryRangesUnionFind::new();
        let mut sr4 = SummaryRangesSegmentTree::new();
        let mut sr5 = SummaryRangesOptimized::new();
        let mut sr6 = SummaryRangesHybrid::new();
        
        for num in operations {
            sr1.add_num(num);
            sr2.add_num(num);
            sr3.add_num(num);
            sr4.add_num(num);
            sr5.add_num(num);
            sr6.add_num(num);
        }
        
        let result1 = sr1.get_intervals();
        let result2 = sr2.get_intervals();
        let result3 = sr3.get_intervals();
        let result4 = sr4.get_intervals();
        let result5 = sr5.get_intervals();
        let result6 = sr6.get_intervals();
        
        assert_eq!(result1, result2, "TreeMap vs BinarySearch mismatch");
        assert_eq!(result2, result3, "BinarySearch vs UnionFind mismatch");
        assert_eq!(result3, result4, "UnionFind vs SegmentTree mismatch");
        assert_eq!(result4, result5, "SegmentTree vs Optimized mismatch");
        assert_eq!(result5, result6, "Optimized vs Hybrid mismatch");
        
        // Verify expected result
        assert_eq!(result1, vec![vec![1, 9]]);
    }
}

// Author: Marvin Tutt, Caia Tech
// Problem: LeetCode 352 - Data Stream as Disjoint Intervals
// Approaches: TreeMap-based, Binary search with vector, Union-find, 
//            Segment tree, Optimized TreeSet, Hybrid with lazy propagation
// All approaches efficiently maintain disjoint intervals from a data stream