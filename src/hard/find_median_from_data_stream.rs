//! Problem 295: Find Median from Data Stream
//!
//! The median is the middle value in an ordered integer list. If the size of the list is even,
//! there is no middle value, and the median is the mean of the two middle values.
//!
//! Implement the MedianFinder class:
//! - MedianFinder() initializes the MedianFinder object.
//! - void addNum(int num) adds the integer num from the data stream to the data structure.
//! - double findMedian() returns the median of all elements so far.
//!
//! Answers within 10^-5 of the actual answer will be accepted.
//!
//! Constraints:
//! - -10^5 <= num <= 10^5
//! - There will be at least one element in the data structure before calling findMedian.
//! - At most 5 * 10^4 calls will be made to addNum and findMedian.
//!
//! Follow up:
//! - If all integer numbers from the stream are in the range [0, 100], how would you optimize it?
//! - If 99% of all integer numbers from the stream are in the range [0, 100], how would you optimize it?

use std::collections::BinaryHeap;
use std::cmp::Reverse;

/// Approach 1: Two Heaps (Max Heap + Min Heap)
/// 
/// Use max heap for smaller half and min heap for larger half.
/// Keep heaps balanced with size difference <= 1.
/// 
/// Time Complexity: addNum O(log n), findMedian O(1)
/// Space Complexity: O(n)
pub struct MedianFinderTwoHeaps {
    max_heap: BinaryHeap<i32>,          // smaller half (max heap)
    min_heap: BinaryHeap<Reverse<i32>>, // larger half (min heap)
}

impl MedianFinderTwoHeaps {
    pub fn new() -> Self {
        Self {
            max_heap: BinaryHeap::new(),
            min_heap: BinaryHeap::new(),
        }
    }
    
    pub fn add_num(&mut self, num: i32) {
        // Add to max heap first if empty, or if num <= max of max_heap
        if self.max_heap.is_empty() || num <= *self.max_heap.peek().unwrap() {
            self.max_heap.push(num);
        } else {
            self.min_heap.push(Reverse(num));
        }
        
        // Balance the heaps
        if self.max_heap.len() > self.min_heap.len() + 1 {
            let val = self.max_heap.pop().unwrap();
            self.min_heap.push(Reverse(val));
        } else if self.min_heap.len() > self.max_heap.len() + 1 {
            let Reverse(val) = self.min_heap.pop().unwrap();
            self.max_heap.push(val);
        }
    }
    
    pub fn find_median(&self) -> f64 {
        if self.max_heap.len() == self.min_heap.len() {
            (*self.max_heap.peek().unwrap() as f64 + self.min_heap.peek().unwrap().0 as f64) / 2.0
        } else if self.max_heap.len() > self.min_heap.len() {
            *self.max_heap.peek().unwrap() as f64
        } else {
            self.min_heap.peek().unwrap().0 as f64
        }
    }
}

/// Approach 2: Sorted Array with Binary Search
/// 
/// Maintain a sorted array and use binary search for insertion.
/// 
/// Time Complexity: addNum O(n), findMedian O(1)
/// Space Complexity: O(n)
pub struct MedianFinderSortedArray {
    nums: Vec<i32>,
}

impl MedianFinderSortedArray {
    pub fn new() -> Self {
        Self {
            nums: Vec::new(),
        }
    }
    
    pub fn add_num(&mut self, num: i32) {
        let pos = self.nums.binary_search(&num).unwrap_or_else(|x| x);
        self.nums.insert(pos, num);
    }
    
    pub fn find_median(&self) -> f64 {
        let n = self.nums.len();
        if n % 2 == 1 {
            self.nums[n / 2] as f64
        } else {
            (self.nums[n / 2 - 1] as f64 + self.nums[n / 2] as f64) / 2.0
        }
    }
}

/// Approach 3: Multiset using BTreeMap (Frequency Map)
/// 
/// Use BTreeMap to maintain sorted order with frequencies.
/// 
/// Time Complexity: addNum O(log n), findMedian O(log n)
/// Space Complexity: O(k) where k is number of distinct elements
use std::collections::BTreeMap;

pub struct MedianFinderBTreeMap {
    map: BTreeMap<i32, usize>,
    total_count: usize,
}

impl MedianFinderBTreeMap {
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
            total_count: 0,
        }
    }
    
    pub fn add_num(&mut self, num: i32) {
        *self.map.entry(num).or_insert(0) += 1;
        self.total_count += 1;
    }
    
    pub fn find_median(&self) -> f64 {
        let target = if self.total_count % 2 == 1 {
            vec![self.total_count / 2]
        } else {
            vec![self.total_count / 2 - 1, self.total_count / 2]
        };
        
        let mut count = 0;
        let mut results = Vec::new();
        
        for (&num, &freq) in &self.map {
            if count + freq > target[0] {
                results.push(num);
                if target.len() == 1 {
                    break;
                }
                if target.len() == 2 && (count + freq > target[1] || results.len() == 2) {
                    if results.len() == 1 {
                        results.push(num);
                    }
                    break;
                }
            }
            count += freq;
            if count > target[0] && target.len() == 2 && results.is_empty() {
                results.push(num);
            }
        }
        
        if results.len() == 1 {
            results[0] as f64
        } else {
            (results[0] as f64 + results[1] as f64) / 2.0
        }
    }
}

/// Approach 4: Optimized for Range [0, 100]
/// 
/// Use counting array for small range optimization.
/// 
/// Time Complexity: addNum O(1), findMedian O(100) = O(1)
/// Space Complexity: O(100) = O(1)
pub struct MedianFinderSmallRange {
    counts: [i32; 201], // -100 to 100 inclusive
    total_count: i32,
}

impl MedianFinderSmallRange {
    pub fn new() -> Self {
        Self {
            counts: [0; 201],
            total_count: 0,
        }
    }
    
    pub fn add_num(&mut self, num: i32) {
        self.counts[(num + 100) as usize] += 1;
        self.total_count += 1;
    }
    
    pub fn find_median(&self) -> f64 {
        let target = if self.total_count % 2 == 1 {
            vec![self.total_count / 2]
        } else {
            vec![self.total_count / 2 - 1, self.total_count / 2]
        };
        
        let mut count = 0;
        let mut results = Vec::new();
        
        for i in 0..201 {
            if self.counts[i] > 0 {
                let num = i as i32 - 100;
                if count + self.counts[i] > target[0] {
                    results.push(num);
                    if target.len() == 1 || count + self.counts[i] > target[1] {
                        if target.len() == 2 && results.len() == 1 {
                            results.push(num);
                        }
                        break;
                    }
                }
                count += self.counts[i];
                if target.len() == 2 && count > target[0] && results.is_empty() {
                    results.push(num);
                }
            }
        }
        
        if results.len() == 1 {
            results[0] as f64
        } else {
            (results[0] as f64 + results[1] as f64) / 2.0
        }
    }
}

/// Approach 5: Bucket-Based for Mixed Range
/// 
/// Use buckets for common range [0,100] and fallback structures for outliers.
/// Optimized for 99% in [0,100] scenario.
/// 
/// Time Complexity: addNum O(log n), findMedian O(100 + log n)
/// Space Complexity: O(100 + k) where k is outliers
pub struct MedianFinderBuckets {
    bucket_counts: [i32; 101], // 0 to 100 inclusive
    bucket_total: i32,
    outliers: Vec<i32>, // sorted outliers
    total_count: i32,
}

impl MedianFinderBuckets {
    pub fn new() -> Self {
        Self {
            bucket_counts: [0; 101],
            bucket_total: 0,
            outliers: Vec::new(),
            total_count: 0,
        }
    }
    
    pub fn add_num(&mut self, num: i32) {
        if num >= 0 && num <= 100 {
            self.bucket_counts[num as usize] += 1;
            self.bucket_total += 1;
        } else {
            let pos = self.outliers.binary_search(&num).unwrap_or_else(|x| x);
            self.outliers.insert(pos, num);
        }
        self.total_count += 1;
    }
    
    pub fn find_median(&self) -> f64 {
        let mid = self.total_count / 2;
        let is_even = self.total_count % 2 == 0;
        
        if is_even {
            let left_idx = mid - 1;
            let right_idx = mid;
            
            let left_val = self.find_kth_element(left_idx);
            let right_val = self.find_kth_element(right_idx);
            
            (left_val as f64 + right_val as f64) / 2.0
        } else {
            self.find_kth_element(mid) as f64
        }
    }
    
    fn find_kth_element(&self, k: i32) -> i32 {
        let neg_count = self.outliers.iter().take_while(|&&x| x < 0).count() as i32;
        
        if k < neg_count {
            return self.outliers[k as usize];
        }
        
        let k_in_bucket = k - neg_count;
        if k_in_bucket < self.bucket_total {
            // Find in bucket
            let mut count = 0;
            for i in 0..101 {
                count += self.bucket_counts[i];
                if count > k_in_bucket {
                    return i as i32;
                }
            }
        }
        
        // Find in positive outliers
        let pos_start = neg_count + self.bucket_total;
        let pos_idx = (k - pos_start) as usize;
        self.outliers[neg_count as usize + pos_idx]
    }
}

/// Approach 6: Segment Tree for Range Queries
/// 
/// Use segment tree to efficiently find k-th element and maintain counts.
/// More complex but supports efficient range operations.
/// 
/// Time Complexity: addNum O(log range), findMedian O(log range)
/// Space Complexity: O(range)
pub struct MedianFinderSegmentTree {
    tree: Vec<i32>,
    offset: i32, // to handle negative numbers
    size: usize,
    total_count: i32,
}

impl MedianFinderSegmentTree {
    pub fn new() -> Self {
        const RANGE: usize = 200001; // -100000 to 100000
        const OFFSET: i32 = 100000;
        
        Self {
            tree: vec![0; 4 * RANGE],
            offset: OFFSET,
            size: RANGE,
            total_count: 0,
        }
    }
    
    pub fn add_num(&mut self, num: i32) {
        let idx = (num + self.offset) as usize;
        self.update(1, 0, self.size - 1, idx, 1);
        self.total_count += 1;
    }
    
    pub fn find_median(&self) -> f64 {
        if self.total_count % 2 == 1 {
            let k = self.total_count / 2;
            let val = self.find_kth(1, 0, self.size - 1, k + 1);
            (val as i32 - self.offset) as f64
        } else {
            let k1 = self.total_count / 2 - 1;
            let k2 = self.total_count / 2;
            let val1 = self.find_kth(1, 0, self.size - 1, k1 + 1);
            let val2 = self.find_kth(1, 0, self.size - 1, k2 + 1);
            ((val1 as i32 - self.offset) as f64 + (val2 as i32 - self.offset) as f64) / 2.0
        }
    }
    
    fn update(&mut self, node: usize, start: usize, end: usize, idx: usize, val: i32) {
        if start == end {
            self.tree[node] += val;
        } else {
            let mid = (start + end) / 2;
            if idx <= mid {
                self.update(2 * node, start, mid, idx, val);
            } else {
                self.update(2 * node + 1, mid + 1, end, idx, val);
            }
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1];
        }
    }
    
    fn find_kth(&self, node: usize, start: usize, end: usize, k: i32) -> i32 {
        if start == end {
            return start as i32;
        }
        
        let mid = (start + end) / 2;
        let left_count = self.tree[2 * node];
        
        if k <= left_count {
            self.find_kth(2 * node, start, mid, k)
        } else {
            self.find_kth(2 * node + 1, mid + 1, end, k - left_count)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_two_heaps() {
        let mut finder = MedianFinderTwoHeaps::new();
        finder.add_num(1);
        assert_eq!(finder.find_median(), 1.0);
        
        finder.add_num(2);
        assert_eq!(finder.find_median(), 1.5);
        
        finder.add_num(3);
        assert_eq!(finder.find_median(), 2.0);
    }
    
    #[test]
    fn test_sorted_array() {
        let mut finder = MedianFinderSortedArray::new();
        finder.add_num(1);
        assert_eq!(finder.find_median(), 1.0);
        
        finder.add_num(2);
        assert_eq!(finder.find_median(), 1.5);
        
        finder.add_num(3);
        assert_eq!(finder.find_median(), 2.0);
    }
    
    #[test]
    fn test_btree_map() {
        let mut finder = MedianFinderBTreeMap::new();
        finder.add_num(1);
        assert_eq!(finder.find_median(), 1.0);
        
        finder.add_num(2);
        assert_eq!(finder.find_median(), 1.5);
        
        finder.add_num(3);
        assert_eq!(finder.find_median(), 2.0);
    }
    
    #[test]
    fn test_small_range() {
        let mut finder = MedianFinderSmallRange::new();
        finder.add_num(6);
        assert_eq!(finder.find_median(), 6.0);
        
        finder.add_num(10);
        assert_eq!(finder.find_median(), 8.0);
        
        finder.add_num(2);
        assert_eq!(finder.find_median(), 6.0);
        
        finder.add_num(6);
        assert_eq!(finder.find_median(), 6.0);
        
        finder.add_num(5);
        assert_eq!(finder.find_median(), 6.0);
    }
    
    #[test]
    fn test_buckets() {
        let mut finder = MedianFinderBuckets::new();
        finder.add_num(12);
        assert_eq!(finder.find_median(), 12.0);
        
        finder.add_num(10);
        assert!((finder.find_median() - 11.0).abs() < 1e-9);
        
        finder.add_num(13);
        assert_eq!(finder.find_median(), 12.0);
    }
    
    #[test]
    fn test_segment_tree() {
        let mut finder = MedianFinderSegmentTree::new();
        finder.add_num(1);
        assert_eq!(finder.find_median(), 1.0);
        
        finder.add_num(2);
        assert_eq!(finder.find_median(), 1.5);
        
        finder.add_num(3);
        assert_eq!(finder.find_median(), 2.0);
    }
    
    #[test]
    fn test_negative_numbers() {
        let mut finder = MedianFinderTwoHeaps::new();
        finder.add_num(-1);
        assert_eq!(finder.find_median(), -1.0);
        
        finder.add_num(-2);
        assert_eq!(finder.find_median(), -1.5);
        
        finder.add_num(-3);
        assert_eq!(finder.find_median(), -2.0);
        
        finder.add_num(-4);
        assert_eq!(finder.find_median(), -2.5);
    }
    
    #[test]
    fn test_large_numbers() {
        let mut finder = MedianFinderTwoHeaps::new();
        finder.add_num(100000);
        finder.add_num(-100000);
        assert_eq!(finder.find_median(), 0.0);
        
        finder.add_num(50000);
        assert_eq!(finder.find_median(), 50000.0);
    }
    
    #[test]
    fn test_duplicates() {
        let mut finder = MedianFinderTwoHeaps::new();
        finder.add_num(1);
        finder.add_num(1);
        assert_eq!(finder.find_median(), 1.0);
        
        finder.add_num(1);
        assert_eq!(finder.find_median(), 1.0);
        
        finder.add_num(2);
        assert_eq!(finder.find_median(), 1.0);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_sequences = vec![
            vec![1, 2, 3, 4, 5],
            vec![-1, -2, -3],
            vec![1, 1, 1, 1],
            vec![5, 4, 3, 2, 1],
            vec![1, 3, 2, 4],
            vec![0, 0, 0],
        ];
        
        for sequence in test_sequences {
            let mut finder1 = MedianFinderTwoHeaps::new();
            let mut finder2 = MedianFinderSortedArray::new();
            let mut finder3 = MedianFinderBTreeMap::new();
            let mut finder6 = MedianFinderSegmentTree::new();
            
            for num in &sequence {
                finder1.add_num(*num);
                finder2.add_num(*num);
                finder3.add_num(*num);
                finder6.add_num(*num);
            }
            
            let result1 = finder1.find_median();
            let result2 = finder2.find_median();
            let result3 = finder3.find_median();
            let result6 = finder6.find_median();
            
            assert!((result1 - result2).abs() < 1e-9, "TwoHeaps vs SortedArray mismatch for {:?}: {} vs {}", sequence, result1, result2);
            assert!((result2 - result3).abs() < 1e-9, "SortedArray vs BTreeMap mismatch for {:?}: {} vs {}", sequence, result2, result3);
            assert!((result1 - result6).abs() < 1e-9, "TwoHeaps vs SegmentTree mismatch for {:?}: {} vs {}", sequence, result1, result6);
        }
    }
}