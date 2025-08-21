//! # Problem 347: Top K Frequent Elements
//!
//! Given an integer array `nums` and an integer `k`, return the `k` most frequent elements. 
//! You may return the answer in any order.
//!
//! ## Examples
//!
//! ```text
//! Input: nums = [1,1,1,2,2,3], k = 2
//! Output: [1,2]
//! ```
//!
//! ```text
//! Input: nums = [1], k = 1
//! Output: [1]
//! ```
//!
//! ## Constraints
//!
//! * 1 <= nums.length <= 10^5
//! * -10^4 <= nums[i] <= 10^4
//! * k is in the range [1, the number of unique elements in the array]
//! * It is guaranteed that the answer is unique

use std::collections::{HashMap, BinaryHeap, BTreeMap};
use std::cmp::Reverse;

/// Solution for Top K Frequent Elements problem
pub struct Solution;

impl Solution {
    /// Creates a new instance of Solution
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Min Heap of Size K (Optimal for small K)
    /// 
    /// **Algorithm:**
    /// 1. Count frequency of each element using HashMap
    /// 2. Use min heap to maintain top K frequent elements
    /// 3. If heap size > K, remove minimum frequency element
    /// 4. Extract remaining elements from heap
    /// 
    /// **Time Complexity:** O(n log k) - n elements, heap operations are log k
    /// **Space Complexity:** O(n + k) - HashMap + heap of size k
    /// 
    /// **Key Insights:**
    /// - Min heap keeps least frequent element at top
    /// - When heap size exceeds k, remove least frequent
    /// - Remaining elements are top k frequent
    /// 
    /// **Why min heap works:**
    /// - Want to remove least frequent when heap is full
    /// - Min heap gives quick access to minimum element
    /// - Efficiently maintains k largest frequencies
    /// 
    /// **When optimal:** When k is much smaller than number of unique elements
    pub fn top_k_frequent(&self, nums: Vec<i32>, k: i32) -> Vec<i32> {
        let mut freq_map = HashMap::new();
        
        // Count frequencies
        for num in nums {
            *freq_map.entry(num).or_insert(0) += 1;
        }
        
        // Use min heap to keep top k elements
        let mut heap = BinaryHeap::new();
        
        for (num, freq) in freq_map {
            heap.push(Reverse((freq, num)));
            
            if heap.len() > k as usize {
                heap.pop();
            }
        }
        
        // Extract results
        heap.into_iter().map(|Reverse((_, num))| num).collect()
    }

    /// # Approach 2: Bucket Sort by Frequency
    /// 
    /// **Algorithm:**
    /// 1. Count frequencies using HashMap
    /// 2. Create buckets where index = frequency
    /// 3. Place elements in corresponding frequency buckets
    /// 4. Traverse buckets from highest frequency, collect k elements
    /// 
    /// **Time Complexity:** O(n) - Linear in array size
    /// **Space Complexity:** O(n) - HashMap + buckets array
    /// 
    /// **Key Insights:**
    /// - Maximum frequency is at most n (all elements same)
    /// - Can use array indexing instead of heap
    /// - Traverse from high to low frequency
    /// 
    /// **Why bucket sort works:**
    /// - Frequency range is bounded [1, n]
    /// - Can use counting sort principle
    /// - No need for comparison-based sorting
    /// 
    /// **When optimal:** When k is large or want guaranteed O(n) time
    pub fn top_k_frequent_bucket(&self, nums: Vec<i32>, k: i32) -> Vec<i32> {
        let n = nums.len();
        let mut freq_map = HashMap::new();
        
        // Count frequencies
        for num in nums {
            *freq_map.entry(num).or_insert(0) += 1;
        }
        let mut buckets: Vec<Vec<i32>> = vec![Vec::new(); n + 1];
        
        // Place elements in frequency buckets
        for (num, freq) in freq_map {
            buckets[freq].push(num);
        }
        
        // Collect top k elements
        let mut result = Vec::new();
        
        for freq in (1..=n).rev() {
            for &num in &buckets[freq] {
                result.push(num);
                if result.len() == k as usize {
                    return result;
                }
            }
        }
        
        result
    }

    /// # Approach 3: Max Heap (Simple but less efficient)
    /// 
    /// **Algorithm:**
    /// 1. Count frequencies
    /// 2. Push all (frequency, element) pairs to max heap
    /// 3. Pop k elements from heap
    /// 
    /// **Time Complexity:** O(n log n) - All elements in heap
    /// **Space Complexity:** O(n) - HashMap + heap
    /// 
    /// **Characteristics:**
    /// - Simpler to implement
    /// - Less memory efficient than min heap approach
    /// - Good when k is close to number of unique elements
    /// 
    /// **When to use:** When simplicity is preferred over efficiency
    pub fn top_k_frequent_max_heap(&self, nums: Vec<i32>, k: i32) -> Vec<i32> {
        let mut freq_map = HashMap::new();
        
        // Count frequencies
        for num in nums {
            *freq_map.entry(num).or_insert(0) += 1;
        }
        
        // Max heap of (frequency, element)
        let mut heap = BinaryHeap::new();
        
        for (num, freq) in freq_map {
            heap.push((freq, num));
        }
        
        // Extract top k elements
        let mut result = Vec::new();
        for _ in 0..k {
            if let Some((_, num)) = heap.pop() {
                result.push(num);
            }
        }
        
        result
    }

    /// # Approach 4: Sort and Take K (Simple but effective)
    /// 
    /// **Algorithm:**
    /// 1. Count frequencies
    /// 2. Convert to vector of (frequency, element) pairs
    /// 3. Sort by frequency in descending order
    /// 4. Take first k elements
    /// 
    /// **Time Complexity:** O(n log n) - Sorting dominates
    /// **Space Complexity:** O(n) - HashMap + vector
    /// 
    /// **Advantages:**
    /// - Simple and reliable
    /// - Good performance for moderate sizes
    /// - Easy to understand and debug
    /// 
    /// **When effective:** When simplicity and reliability are preferred
    pub fn top_k_frequent_quickselect(&self, nums: Vec<i32>, k: i32) -> Vec<i32> {
        let mut freq_map = HashMap::new();
        
        // Count frequencies
        for num in nums {
            *freq_map.entry(num).or_insert(0) += 1;
        }
        
        let mut freq_pairs: Vec<(i32, i32)> = freq_map.into_iter().collect();
        
        // Sort by frequency in descending order (second element is frequency)
        freq_pairs.sort_by(|a, b| b.1.cmp(&a.1));
        
        // Return top k elements (first element is the number)
        freq_pairs[..k as usize].iter().map(|(num, _)| *num).collect()
    }

    /// # Approach 5: TreeMap (Frequency-sorted)
    /// 
    /// **Algorithm:**
    /// 1. Count frequencies
    /// 2. Use TreeMap to automatically sort by frequency
    /// 3. Collect elements from highest frequencies
    /// 
    /// **Time Complexity:** O(n log u) where u = unique elements
    /// **Space Complexity:** O(n) - HashMap + TreeMap
    /// 
    /// **Characteristics:**
    /// - Automatic sorting by frequency
    /// - Good for maintaining sorted order
    /// - Useful when need both frequency and sorted access
    /// 
    /// **When useful:** When you need sorted frequency information
    pub fn top_k_frequent_treemap(&self, nums: Vec<i32>, k: i32) -> Vec<i32> {
        let mut freq_map = HashMap::new();
        
        // Count frequencies
        for num in nums {
            *freq_map.entry(num).or_insert(0) += 1;
        }
        
        // TreeMap: frequency -> list of elements with that frequency
        let mut freq_tree: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
        
        for (num, freq) in freq_map {
            freq_tree.entry(freq).or_insert_with(Vec::new).push(num);
        }
        
        // Collect from highest frequencies
        let mut result = Vec::new();
        
        for (_, nums_with_freq) in freq_tree.iter().rev() {
            for &num in nums_with_freq {
                result.push(num);
                if result.len() == k as usize {
                    return result;
                }
            }
        }
        
        result
    }

    /// # Approach 6: Linear Scan with Nth Element
    /// 
    /// **Algorithm:**
    /// 1. Count frequencies
    /// 2. Find kth largest frequency without full sorting
    /// 3. Collect all elements with frequency >= kth largest
    /// 4. Handle ties appropriately
    /// 
    /// **Time Complexity:** O(n + u log u) where u = unique elements
    /// **Space Complexity:** O(n) - HashMap + frequency vector
    /// 
    /// **Approach:**
    /// - Sort unique frequencies instead of all elements
    /// - Use nth_element-like algorithm
    /// - More cache-friendly than quickselect on large arrays
    /// 
    /// **When effective:** When number of unique elements is small
    pub fn top_k_frequent_nth_element(&self, nums: Vec<i32>, k: i32) -> Vec<i32> {
        let mut freq_map = HashMap::new();
        
        // Count frequencies
        for num in nums {
            *freq_map.entry(num).or_insert(0) += 1;
        }
        
        // Get all unique frequencies
        let mut frequencies: Vec<i32> = freq_map.values().cloned().collect();
        frequencies.sort_unstable();
        frequencies.reverse(); // Sort in descending order
        
        // Find the kth largest frequency (may have ties)
        let mut count = 0;
        let mut threshold_freq = 0;
        
        for &freq in &frequencies {
            let elements_with_freq = freq_map.values().filter(|&&f| f == freq).count();
            
            if count + elements_with_freq >= k as usize {
                threshold_freq = freq;
                break;
            }
            count += elements_with_freq;
        }
        
        // Collect elements with frequency >= threshold
        let mut result = Vec::new();
        let mut collected = 0;
        
        // First collect elements with frequency > threshold
        for (&num, &freq) in &freq_map {
            if freq > threshold_freq {
                result.push(num);
                collected += 1;
            }
        }
        
        // Then collect elements with frequency == threshold (up to k total)
        for (&num, &freq) in &freq_map {
            if freq == threshold_freq && collected < k as usize {
                result.push(num);
                collected += 1;
            }
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

    fn sort_result(mut result: Vec<i32>) -> Vec<i32> {
        result.sort_unstable();
        result
    }

    #[test]
    fn test_basic_examples() {
        let solution = setup();
        
        // Example 1: [1,1,1,2,2,3], k=2 → [1,2]
        let result1 = solution.top_k_frequent(vec![1, 1, 1, 2, 2, 3], 2);
        assert_eq!(sort_result(result1), vec![1, 2]);
        
        // Example 2: [1], k=1 → [1]
        let result2 = solution.top_k_frequent(vec![1], 1);
        assert_eq!(result2, vec![1]);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // All elements same
        let result = solution.top_k_frequent(vec![1, 1, 1, 1], 1);
        assert_eq!(result, vec![1]);
        
        // All elements different, k = all
        let result = solution.top_k_frequent(vec![1, 2, 3, 4], 4);
        assert_eq!(sort_result(result), vec![1, 2, 3, 4]);
        
        // k = 1, multiple elements with same max frequency
        let result = solution.top_k_frequent(vec![1, 2], 1);
        assert!(result.len() == 1 && (result[0] == 1 || result[0] == 2));
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            (vec![1, 1, 1, 2, 2, 3], 2),
            (vec![1], 1),
            (vec![1, 2, 3, 1, 2, 1], 2),
            (vec![4, 1, -1, 2, -1, 2, 3], 2),
            (vec![1, 1, 1, 2, 2, 3, 3, 3, 3], 2),
        ];
        
        for (nums, k) in test_cases {
            let result1 = sort_result(solution.top_k_frequent(nums.clone(), k));
            let result2 = sort_result(solution.top_k_frequent_bucket(nums.clone(), k));
            let result3 = sort_result(solution.top_k_frequent_max_heap(nums.clone(), k));
            let result4 = sort_result(solution.top_k_frequent_quickselect(nums.clone(), k));
            let result5 = sort_result(solution.top_k_frequent_treemap(nums.clone(), k));
            let result6 = sort_result(solution.top_k_frequent_nth_element(nums.clone(), k));
            
            assert_eq!(result1, result2, "Min heap vs Bucket sort mismatch for {:?}", nums);
            assert_eq!(result2, result3, "Bucket sort vs Max heap mismatch for {:?}", nums);
            assert_eq!(result3, result4, "Max heap vs Quickselect mismatch for {:?}", nums);
            assert_eq!(result4, result5, "Quickselect vs TreeMap mismatch for {:?}", nums);
            assert_eq!(result5, result6, "TreeMap vs Nth element mismatch for {:?}", nums);
        }
    }

    #[test]
    fn test_frequency_patterns() {
        let solution = setup();
        
        // Decreasing frequency pattern
        let result = solution.top_k_frequent(vec![1, 1, 1, 2, 2, 3], 3);
        assert_eq!(sort_result(result), vec![1, 2, 3]);
        
        // Equal frequencies
        let result = solution.top_k_frequent(vec![1, 2, 3, 4], 2);
        assert_eq!(result.len(), 2);
        
        // Single high frequency element
        let result = solution.top_k_frequent(vec![1, 1, 1, 1, 2, 3, 4], 1);
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn test_negative_numbers() {
        let solution = setup();
        
        // Mix of positive and negative
        let result = solution.top_k_frequent(vec![-1, -1, 2, 2, 3], 2);
        assert_eq!(sort_result(result), vec![-1, 2]);
        
        // All negative
        let result = solution.top_k_frequent(vec![-1, -2, -1, -3, -2, -1], 2);
        assert_eq!(sort_result(result), vec![-2, -1]);
    }

    #[test]
    fn test_large_frequencies() {
        let solution = setup();
        
        // One element appears many times
        let mut nums = vec![1; 1000];
        nums.extend(vec![2; 500]);
        nums.extend(vec![3; 100]);
        
        let result = solution.top_k_frequent(nums, 2);
        assert_eq!(sort_result(result), vec![1, 2]);
    }

    #[test]
    fn test_boundary_values() {
        let solution = setup();
        
        // Minimum value
        let result = solution.top_k_frequent(vec![-10000, -10000, 1], 1);
        assert_eq!(result, vec![-10000]);
        
        // Maximum value  
        let result = solution.top_k_frequent(vec![10000, 10000, 1], 1);
        assert_eq!(result, vec![10000]);
        
        // Mixed boundaries
        let result = solution.top_k_frequent(vec![-10000, 10000, -10000], 1);
        assert_eq!(result, vec![-10000]);
    }

    #[test]
    fn test_k_variations() {
        let solution = setup();
        
        let nums = vec![1, 1, 1, 2, 2, 3, 4, 4, 4, 4];
        
        // k = 1 (most frequent)
        let result = solution.top_k_frequent(nums.clone(), 1);
        assert_eq!(result, vec![4]);
        
        // k = 2 (top 2)
        let result = solution.top_k_frequent(nums.clone(), 2);
        assert_eq!(sort_result(result), vec![1, 4]);
        
        // k = 3 (top 3)
        let result = solution.top_k_frequent(nums.clone(), 3);
        assert_eq!(sort_result(result), vec![1, 2, 4]);
    }

    #[test]
    fn test_duplicate_frequencies() {
        let solution = setup();
        
        // Multiple elements with same frequency
        let result = solution.top_k_frequent(vec![1, 2, 3, 1, 2, 3], 3);
        assert_eq!(sort_result(result), vec![1, 2, 3]);
        
        // Ties in frequency
        let result = solution.top_k_frequent(vec![1, 1, 2, 2, 3, 3, 4], 3);
        assert_eq!(result.len(), 3);
        // Should contain 3 elements with highest frequencies
        for &num in &result {
            assert!(vec![1, 2, 3].contains(&num));
        }
    }

    #[test]
    fn test_performance_characteristics() {
        let solution = setup();
        
        // Large array with few unique elements
        let mut nums = Vec::new();
        for i in 0..1000 {
            nums.push(i % 10); // Only 10 unique values
        }
        
        let result = solution.top_k_frequent(nums, 3);
        assert_eq!(result.len(), 3);
        
        // Should handle efficiently
        assert!(result.iter().all(|&x| x >= 0 && x < 10));
    }

    #[test]
    fn test_mathematical_properties() {
        let solution = setup();
        
        let nums = vec![1, 1, 2, 2, 2, 3, 3, 3, 3];
        
        // Property: more frequent elements should appear in smaller k
        let result_k1 = solution.top_k_frequent(nums.clone(), 1);
        let result_k2 = solution.top_k_frequent(nums.clone(), 2);
        
        // Element in k=1 should also be in k=2
        assert!(result_k2.contains(&result_k1[0]));
        
        // Property: total unique elements bound
        let all_unique = solution.top_k_frequent(nums.clone(), 10);
        assert_eq!(sort_result(all_unique), vec![1, 2, 3]); // Only 3 unique elements
    }

    #[test]
    fn test_stability_and_ordering() {
        let solution = setup();
        
        // Test that results are consistent across multiple runs
        let nums = vec![1, 1, 2, 2, 3, 4, 5];
        
        let result1 = solution.top_k_frequent(nums.clone(), 2);
        let result2 = solution.top_k_frequent(nums.clone(), 2);
        
        // Results should be consistent (though order may vary for ties)
        assert_eq!(sort_result(result1), sort_result(result2));
    }
}