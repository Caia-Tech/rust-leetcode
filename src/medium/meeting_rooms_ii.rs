//! # Problem 253: Meeting Rooms II
//!
//! **Difficulty**: Medium
//! **Topics**: Array, Two Pointers, Greedy, Sorting, Heap (Priority Queue), Prefix Sum
//! **Acceptance Rate**: 49.4%

/// Solution struct following LeetCode format
pub struct Solution;

impl Solution {
    /// Create a new solution instance
    pub fn new() -> Self {
        Solution
    }

    /// Main solution approach using heap/priority queue
    /// 
    /// Time Complexity: O(n log n) - sorting + heap operations
    /// Space Complexity: O(n) - for the heap
    pub fn min_meeting_rooms(&self, intervals: Vec<Vec<i32>>) -> i32 {
        if intervals.is_empty() {
            return 0;
        }

        // Sort intervals by start time
        let mut intervals = intervals;
        intervals.sort_by_key(|interval| interval[0]);
        
        // Use a min heap to track end times of ongoing meetings
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;
        let mut heap = BinaryHeap::new();
        
        for interval in intervals {
            let start = interval[0];
            let end = interval[1];
            
            // Remove all meetings that have ended before current starts
            while let Some(&Reverse(earliest_end)) = heap.peek() {
                if earliest_end <= start {
                    heap.pop();
                } else {
                    break;
                }
            }
            
            // Add current meeting's end time
            heap.push(Reverse(end));
        }
        
        heap.len() as i32
    }

    /// Alternative solution using chronological ordering
    /// 
    /// Time Complexity: O(n log n) - sorting events
    /// Space Complexity: O(n) - for events array
    pub fn min_meeting_rooms_chronological(&self, intervals: Vec<Vec<i32>>) -> i32 {
        if intervals.is_empty() {
            return 0;
        }

        let mut events = Vec::new();
        
        // Create events for each start and end time
        for interval in intervals {
            events.push((interval[0], 1));  // Meeting starts (+1 room)
            events.push((interval[1], -1)); // Meeting ends (-1 room)
        }
        
        // Sort events by time, with end events before start events for same time
        events.sort_by(|a, b| {
            if a.0 == b.0 {
                a.1.cmp(&b.1) // End events (-1) come before start events (1)
            } else {
                a.0.cmp(&b.0)
            }
        });
        
        let mut active_meetings = 0;
        let mut max_rooms = 0;
        
        for (_, event_type) in events {
            active_meetings += event_type;
            max_rooms = max_rooms.max(active_meetings);
        }
        
        max_rooms
    }

    /// Brute force solution (for comparison)
    /// 
    /// Time Complexity: O(n³) - for each time point, check all overlapping meetings
    /// Space Complexity: O(1) - constant space
    pub fn min_meeting_rooms_brute_force(&self, intervals: Vec<Vec<i32>>) -> i32 {
        if intervals.is_empty() {
            return 0;
        }

        let mut max_rooms = 0;
        
        // Collect all unique time points
        let mut time_points = Vec::new();
        for interval in &intervals {
            time_points.push(interval[0]); // start times
            time_points.push(interval[1]); // end times
        }
        time_points.sort_unstable();
        time_points.dedup();
        
        // For each time point, count how many meetings are active
        for &time in &time_points {
            let mut active_meetings = 0;
            
            for interval in &intervals {
                // Meeting is active if time is in [start, end)
                if time >= interval[0] && time < interval[1] {
                    active_meetings += 1;
                }
            }
            
            max_rooms = max_rooms.max(active_meetings);
        }
        
        max_rooms
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

    #[test]
    fn test_basic_cases() {
        let solution = Solution::new();
        
        // Test case 1: [[0,30],[5,10],[15,20]]
        assert_eq!(solution.min_meeting_rooms(vec![vec![0,30], vec![5,10], vec![15,20]]), 2);
        
        // Test case 2: [[7,10],[2,4]]
        assert_eq!(solution.min_meeting_rooms(vec![vec![7,10], vec![2,4]]), 1);
        
        // Test case 3: Single meeting
        assert_eq!(solution.min_meeting_rooms(vec![vec![1,5]]), 1);
        
        // Test case 4: No meetings
        assert_eq!(solution.min_meeting_rooms(vec![]), 0);
    }

    #[test]
    fn test_edge_cases() {
        let solution = Solution::new();
        
        // Back-to-back meetings (should need only 1 room)
        assert_eq!(solution.min_meeting_rooms(vec![vec![1,5], vec![5,10]]), 1);
        
        // All overlapping meetings
        assert_eq!(solution.min_meeting_rooms(vec![
            vec![1,10], vec![2,6], vec![3,7], vec![4,8]
        ]), 4);
        
        // Complex overlapping pattern
        assert_eq!(solution.min_meeting_rooms(vec![
            vec![9,10], vec![4,9], vec![4,17]
        ]), 2);
        
        // Many meetings, some overlapping
        assert_eq!(solution.min_meeting_rooms(vec![
            vec![0,30], vec![5,10], vec![15,20], vec![25,30]
        ]), 2);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = Solution::new();
        
        let test_cases = vec![
            vec![vec![0,30], vec![5,10], vec![15,20]],
            vec![vec![7,10], vec![2,4]],
            vec![vec![1,5], vec![5,10]],
            vec![vec![9,10], vec![4,9], vec![4,17]],
            vec![vec![1,10], vec![2,6], vec![3,7], vec![4,8]],
        ];

        for case in test_cases {
            let result1 = solution.min_meeting_rooms(case.clone());
            let result2 = solution.min_meeting_rooms_chronological(case.clone());
            let result3 = solution.min_meeting_rooms_brute_force(case.clone());
            
            assert_eq!(result1, result2, "Heap and chronological approaches should match");
            assert_eq!(result1, result3, "Heap and brute force approaches should match");
        }
    }
    
    #[test]
    fn test_performance_scenarios() {
        let solution = Solution::new();
        
        // Large number of non-overlapping meetings
        let mut large_non_overlapping = Vec::new();
        for i in 0..100 {
            large_non_overlapping.push(vec![i * 10, i * 10 + 5]);
        }
        assert_eq!(solution.min_meeting_rooms(large_non_overlapping), 1);
        
        // All meetings start at same time
        let simultaneous = vec![
            vec![0, 10], vec![0, 20], vec![0, 30], vec![0, 40]
        ];
        assert_eq!(solution.min_meeting_rooms(simultaneous), 4);
    }
}

#[cfg(test)]
mod benchmarks {
    // Note: Benchmarks will be added to benches/ directory for Criterion integration
}