//! Problem 274: H-Index
//! 
//! Given an array of integers citations where citations[i] is the number of citations 
//! a researcher received for their ith paper, return the researcher's h-index.
//! 
//! According to the definition of h-index on Wikipedia: The h-index is defined as the 
//! maximum value of h such that the given researcher has published h papers that have 
//! each been cited at least h times.
//! 
//! Example 1:
//! Input: citations = [3,0,6,1,5]
//! Output: 3
//! Explanation: [3,0,6,1,5] means the researcher has 5 papers in total and each of them 
//! had received 3, 0, 6, 1, 5 citations respectively.
//! Since the researcher has 3 papers with at least 3 citations each and the remaining 
//! two with no more than 3 citations each, their h-index is 3.
//! 
//! Example 2:
//! Input: citations = [1,3,1]
//! Output: 1

pub struct Solution;

impl Solution {
    /// Approach 1: Sorting Approach
    /// 
    /// Sort the citations in descending order and find the maximum h where
    /// at least h papers have h or more citations.
    /// 
    /// Time Complexity: O(n log n)
    /// Space Complexity: O(1) if in-place sort, O(n) otherwise
    pub fn h_index_sorting(&self, citations: Vec<i32>) -> i32 {
        let mut citations = citations;
        citations.sort_unstable_by(|a, b| b.cmp(a)); // Sort in descending order
        
        let mut h_index = 0;
        for (i, &citation) in citations.iter().enumerate() {
            let papers_count = (i + 1) as i32;
            if citation >= papers_count {
                h_index = papers_count;
            } else {
                break;
            }
        }
        
        h_index
    }
    
    /// Approach 2: Counting Sort
    /// 
    /// Uses counting sort since h-index is bounded by the number of papers.
    /// For n papers, h-index cannot exceed n.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    pub fn h_index_counting(&self, citations: Vec<i32>) -> i32 {
        let n = citations.len();
        let mut buckets = vec![0; n + 1];
        
        // Count papers by citation count (cap at n)
        for citation in citations {
            let bucket_index = (citation as usize).min(n);
            buckets[bucket_index] += 1;
        }
        
        // Find h-index by checking from highest to lowest
        let mut papers_with_at_least_h = 0;
        for h in (0..=n).rev() {
            papers_with_at_least_h += buckets[h];
            if papers_with_at_least_h >= h {
                return h as i32;
            }
        }
        
        0
    }
    
    /// Approach 3: Binary Search
    /// 
    /// Binary search on the possible h-index values (0 to n).
    /// For each candidate h, check if there are at least h papers with h+ citations.
    /// 
    /// Time Complexity: O(n log n)
    /// Space Complexity: O(1)
    pub fn h_index_binary_search(&self, citations: Vec<i32>) -> i32 {
        let n = citations.len();
        let mut left = 0;
        let mut right = n;
        
        while left <= right {
            let mid = left + (right - left) / 2;
            let papers_with_at_least_mid = citations.iter()
                .filter(|&&c| c >= mid as i32)
                .count();
            
            if papers_with_at_least_mid >= mid {
                if mid == n || citations.iter().filter(|&&c| c >= (mid + 1) as i32).count() < mid + 1 {
                    return mid as i32;
                }
                left = mid + 1;
            } else {
                if mid == 0 {
                    break;
                }
                right = mid - 1;
            }
        }
        
        right as i32
    }
    
    /// Approach 4: Hash Map Frequency
    /// 
    /// Uses a hash map to count citations and then processes the frequencies
    /// to find the h-index.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    pub fn h_index_hashmap(&self, citations: Vec<i32>) -> i32 {
        use std::collections::HashMap;
        
        let n = citations.len();
        let mut freq = HashMap::new();
        
        // Count frequency of each citation count
        for citation in citations {
            *freq.entry(citation).or_insert(0) += 1;
        }
        
        // Calculate cumulative counts from high to low
        let mut papers_with_at_least_h = 0;
        let mut max_h = 0;
        
        for h in (0..=n as i32).rev() {
            // Add papers with exactly h citations
            if let Some(&count) = freq.get(&h) {
                papers_with_at_least_h += count;
            }
            
            // Also add papers with more than h citations
            for (&citation, &count) in &freq {
                if citation > h {
                    // This is already counted in previous iterations
                }
            }
            
            // Recalculate for current h
            papers_with_at_least_h = freq.iter()
                .filter(|(&citation, _)| citation >= h)
                .map(|(_, &count)| count)
                .sum();
            
            if papers_with_at_least_h >= h {
                max_h = h;
                break;
            }
        }
        
        max_h
    }
    
    /// Approach 5: Linear Scan with Optimization
    /// 
    /// Directly checks each possible h-index value from n down to 0.
    /// Optimized by early termination.
    /// 
    /// Time Complexity: O(n²) worst case, but often better in practice
    /// Space Complexity: O(1)
    pub fn h_index_linear(&self, citations: Vec<i32>) -> i32 {
        let n = citations.len();
        
        for h in (0..=n).rev() {
            let papers_with_at_least_h = citations.iter()
                .filter(|&&c| c >= h as i32)
                .count();
            
            if papers_with_at_least_h >= h {
                return h as i32;
            }
        }
        
        0
    }
    
    /// Approach 6: Bucket Sort Variation
    /// 
    /// Similar to counting sort but organizes papers into buckets more efficiently.
    /// Uses the insight that h-index is at most n.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    pub fn h_index_bucket(&self, citations: Vec<i32>) -> i32 {
        let n = citations.len();
        let mut bucket = vec![0; n + 1];
        
        // Place each paper in appropriate bucket
        for citation in citations {
            if citation >= n as i32 {
                bucket[n] += 1;  // Papers with n+ citations
            } else {
                bucket[citation as usize] += 1;
            }
        }
        
        // Find h-index by scanning from right to left
        let mut total = 0;
        for i in (0..=n).rev() {
            total += bucket[i];
            if total >= i {
                return i as i32;
            }
        }
        
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sorting() {
        let solution = Solution;
        
        assert_eq!(solution.h_index_sorting(vec![3,0,6,1,5]), 3);
        assert_eq!(solution.h_index_sorting(vec![1,3,1]), 1);
        assert_eq!(solution.h_index_sorting(vec![100]), 1);
        assert_eq!(solution.h_index_sorting(vec![0,0]), 0);
        assert_eq!(solution.h_index_sorting(vec![0,1,3,5,6]), 3);
    }
    
    #[test]
    fn test_counting() {
        let solution = Solution;
        
        assert_eq!(solution.h_index_counting(vec![3,0,6,1,5]), 3);
        assert_eq!(solution.h_index_counting(vec![1,3,1]), 1);
        assert_eq!(solution.h_index_counting(vec![100]), 1);
        assert_eq!(solution.h_index_counting(vec![0,0]), 0);
        assert_eq!(solution.h_index_counting(vec![0,1,3,5,6]), 3);
    }
    
    #[test]
    fn test_binary_search() {
        let solution = Solution;
        
        assert_eq!(solution.h_index_binary_search(vec![3,0,6,1,5]), 3);
        assert_eq!(solution.h_index_binary_search(vec![1,3,1]), 1);
        assert_eq!(solution.h_index_binary_search(vec![100]), 1);
        assert_eq!(solution.h_index_binary_search(vec![0,0]), 0);
        assert_eq!(solution.h_index_binary_search(vec![0,1,3,5,6]), 3);
    }
    
    #[test]
    fn test_hashmap() {
        let solution = Solution;
        
        assert_eq!(solution.h_index_hashmap(vec![3,0,6,1,5]), 3);
        assert_eq!(solution.h_index_hashmap(vec![1,3,1]), 1);
        assert_eq!(solution.h_index_hashmap(vec![100]), 1);
        assert_eq!(solution.h_index_hashmap(vec![0,0]), 0);
        assert_eq!(solution.h_index_hashmap(vec![0,1,3,5,6]), 3);
    }
    
    #[test]
    fn test_linear() {
        let solution = Solution;
        
        assert_eq!(solution.h_index_linear(vec![3,0,6,1,5]), 3);
        assert_eq!(solution.h_index_linear(vec![1,3,1]), 1);
        assert_eq!(solution.h_index_linear(vec![100]), 1);
        assert_eq!(solution.h_index_linear(vec![0,0]), 0);
        assert_eq!(solution.h_index_linear(vec![0,1,3,5,6]), 3);
    }
    
    #[test]
    fn test_bucket() {
        let solution = Solution;
        
        assert_eq!(solution.h_index_bucket(vec![3,0,6,1,5]), 3);
        assert_eq!(solution.h_index_bucket(vec![1,3,1]), 1);
        assert_eq!(solution.h_index_bucket(vec![100]), 1);
        assert_eq!(solution.h_index_bucket(vec![0,0]), 0);
        assert_eq!(solution.h_index_bucket(vec![0,1,3,5,6]), 3);
    }
    
    #[test]
    fn test_edge_cases() {
        let solution = Solution;
        
        // Single paper with high citations
        assert_eq!(solution.h_index_sorting(vec![1000]), 1);
        
        // All papers have 0 citations
        assert_eq!(solution.h_index_sorting(vec![0,0,0,0]), 0);
        
        // All papers have same citations
        assert_eq!(solution.h_index_sorting(vec![5,5,5,5,5]), 5);
        
        // Decreasing citations
        assert_eq!(solution.h_index_sorting(vec![10,8,5,4,3]), 4);
        
        // Increasing citations
        assert_eq!(solution.h_index_sorting(vec![1,2,3,4,5]), 3);
        
        // Large number of papers
        let many_papers = vec![1; 1000];
        assert_eq!(solution.h_index_bucket(many_papers), 1);
    }
    
    #[test]
    fn test_h_index_definition() {
        let solution = Solution;
        
        // Test case: [3,0,6,1,5]
        // Sorted: [6,5,3,1,0]
        // h=1: 5 papers with >=1 citations ✓
        // h=2: 3 papers with >=2 citations ✓  
        // h=3: 3 papers with >=3 citations ✓
        // h=4: 2 papers with >=4 citations ✗
        // So h-index = 3
        assert_eq!(solution.h_index_sorting(vec![3,0,6,1,5]), 3);
        
        // Test case: [1,3,1]
        // Sorted: [3,1,1]
        // h=1: 3 papers with >=1 citations ✓
        // h=2: 1 paper with >=2 citations ✗
        // So h-index = 1
        assert_eq!(solution.h_index_sorting(vec![1,3,1]), 1);
        
        // Test perfect h-index case
        assert_eq!(solution.h_index_sorting(vec![5,4,3,2,1]), 3);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let solution = Solution;
        
        let test_cases = vec![
            vec![3,0,6,1,5],
            vec![1,3,1],
            vec![100],
            vec![0,0],
            vec![0,1,3,5,6],
            vec![1000],
            vec![0,0,0,0],
            vec![5,5,5,5,5],
            vec![10,8,5,4,3],
            vec![1,2,3,4,5],
            vec![0],
            vec![1,1,1],
            vec![10,10,10,10,10,10,10,10,10,10],
        ];
        
        for citations in test_cases {
            let sorting = solution.h_index_sorting(citations.clone());
            let counting = solution.h_index_counting(citations.clone());
            let binary_search = solution.h_index_binary_search(citations.clone());
            let hashmap = solution.h_index_hashmap(citations.clone());
            let linear = solution.h_index_linear(citations.clone());
            let bucket = solution.h_index_bucket(citations.clone());
            
            assert_eq!(sorting, counting, "Sorting and counting differ for {:?}", citations);
            assert_eq!(sorting, binary_search, "Sorting and binary search differ for {:?}", citations);
            assert_eq!(sorting, hashmap, "Sorting and hashmap differ for {:?}", citations);
            assert_eq!(sorting, linear, "Sorting and linear differ for {:?}", citations);
            assert_eq!(sorting, bucket, "Sorting and bucket differ for {:?}", citations);
        }
    }
    
    #[test]
    fn test_mathematical_properties() {
        let solution = Solution;
        
        // h-index cannot exceed number of papers
        let citations = vec![100, 200, 300];
        let h = solution.h_index_sorting(citations.clone());
        assert!(h <= citations.len() as i32);
        
        // Verify h-index definition
        let papers_with_at_least_h = citations.iter().filter(|&&c| c >= h).count();
        assert!(papers_with_at_least_h >= h as usize);
        
        // Verify maximality (h+1 should not satisfy the condition)
        if h < citations.len() as i32 {
            let papers_with_at_least_h_plus_1 = citations.iter().filter(|&&c| c >= h + 1).count();
            assert!(papers_with_at_least_h_plus_1 < (h + 1) as usize);
        }
    }
    
    #[test]
    fn test_performance_characteristics() {
        let solution = Solution;
        
        // Large array with varied citations
        let large_citations: Vec<i32> = (0..1000).map(|i| i % 100).collect();
        
        let sorting_result = solution.h_index_sorting(large_citations.clone());
        let counting_result = solution.h_index_counting(large_citations.clone());
        let bucket_result = solution.h_index_bucket(large_citations);
        
        assert_eq!(sorting_result, counting_result);
        assert_eq!(sorting_result, bucket_result);
        
        // Test with all high citations
        let high_citations = vec![1000; 500];
        assert_eq!(solution.h_index_bucket(high_citations), 500);
    }
    
    #[test]
    fn test_boundary_conditions() {
        let solution = Solution;
        
        // Empty array (if allowed)
        // Note: Problem constraints typically ensure non-empty array
        
        // Single element cases
        assert_eq!(solution.h_index_sorting(vec![0]), 0);
        assert_eq!(solution.h_index_sorting(vec![1]), 1);
        assert_eq!(solution.h_index_sorting(vec![5]), 1);
        
        // Two elements
        assert_eq!(solution.h_index_sorting(vec![0,0]), 0);
        assert_eq!(solution.h_index_sorting(vec![1,1]), 1);
        assert_eq!(solution.h_index_sorting(vec![2,2]), 2);
        assert_eq!(solution.h_index_sorting(vec![0,1]), 1);
        assert_eq!(solution.h_index_sorting(vec![1,2]), 1);
        
        // All zeros
        assert_eq!(solution.h_index_sorting(vec![0,0,0,0,0]), 0);
        
        // Sequential numbers
        assert_eq!(solution.h_index_sorting(vec![1,2,3,4,5,6,7,8,9,10]), 5);
    }
}