//! Problem 76: Minimum Window Substring
//!
//! Given two strings s and t of lengths m and n respectively, return the minimum window 
//! substring of s such that every character in t (including duplicates) is included in the window. 
//! If there is no such window, return the empty string "".
//!
//! The testcases will be generated such that the answer is unique.
//!
//! Constraints:
//! - m == s.length
//! - n == t.length
//! - 1 <= m, n <= 10^5
//! - s and t consist of uppercase and lowercase English letters.
//!
//! Follow up: Could you find an algorithm that runs in O(m + n) time?
//!
//! Example 1:
//! Input: s = "ADOBECODEBANC", t = "ABC"
//! Output: "BANC"
//! Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
//!
//! Example 2:
//! Input: s = "a", t = "a"
//! Output: "a"
//! Explanation: The entire string s is the minimum window.
//!
//! Example 3:
//! Input: s = "a", t = "aa"
//! Output: ""
//! Explanation: Both 'a's from t must be included in the window.

use std::collections::HashMap;

pub struct Solution;

impl Solution {
    /// Approach 1: Sliding Window with HashMap
    /// 
    /// Use two pointers to maintain a sliding window and HashMap to track character counts.
    /// Expand right pointer until all characters are covered, then contract left pointer.
    /// 
    /// Time Complexity: O(|s| + |t|)
    /// Space Complexity: O(|s| + |t|)
    pub fn min_window_sliding_window(s: String, t: String) -> String {
        if s.is_empty() || t.is_empty() || s.len() < t.len() {
            return String::new();
        }
        
        let s_chars: Vec<char> = s.chars().collect();
        let t_chars: Vec<char> = t.chars().collect();
        
        // Count characters in t
        let mut t_count = HashMap::new();
        for &ch in &t_chars {
            *t_count.entry(ch).or_insert(0) += 1;
        }
        
        let required = t_count.len();
        let mut formed = 0;
        let mut window_counts = HashMap::new();
        
        let mut left = 0;
        let mut min_len = usize::MAX;
        let mut min_left = 0;
        
        for right in 0..s_chars.len() {
            let ch = s_chars[right];
            *window_counts.entry(ch).or_insert(0) += 1;
            
            if let Some(&target_count) = t_count.get(&ch) {
                if window_counts[&ch] == target_count {
                    formed += 1;
                }
            }
            
            // Try to contract window from left
            while left <= right && formed == required {
                let current_len = right - left + 1;
                if current_len < min_len {
                    min_len = current_len;
                    min_left = left;
                }
                
                let left_ch = s_chars[left];
                *window_counts.get_mut(&left_ch).unwrap() -= 1;
                
                if let Some(&target_count) = t_count.get(&left_ch) {
                    if window_counts[&left_ch] < target_count {
                        formed -= 1;
                    }
                }
                
                left += 1;
            }
        }
        
        if min_len == usize::MAX {
            String::new()
        } else {
            s_chars[min_left..min_left + min_len].iter().collect()
        }
    }
    
    /// Approach 2: Optimized Sliding Window with Filtered String
    /// 
    /// Pre-filter the string to only include characters present in t.
    /// This reduces the search space when t is much smaller than s.
    /// 
    /// Time Complexity: O(|s| + |t|)
    /// Space Complexity: O(|s| + |t|)
    pub fn min_window_optimized(s: String, t: String) -> String {
        if s.is_empty() || t.is_empty() || s.len() < t.len() {
            return String::new();
        }
        
        let s_chars: Vec<char> = s.chars().collect();
        let t_chars: Vec<char> = t.chars().collect();
        
        // Count characters in t
        let mut t_count = HashMap::new();
        for &ch in &t_chars {
            *t_count.entry(ch).or_insert(0) += 1;
        }
        
        // Filter s to only include characters from t
        let mut filtered_s = Vec::new();
        for (i, &ch) in s_chars.iter().enumerate() {
            if t_count.contains_key(&ch) {
                filtered_s.push((i, ch));
            }
        }
        
        if filtered_s.is_empty() {
            return String::new();
        }
        
        let required = t_count.len();
        let mut formed = 0;
        let mut window_counts = HashMap::new();
        
        let mut left = 0;
        let mut min_len = usize::MAX;
        let mut min_start = 0;
        
        for right in 0..filtered_s.len() {
            let (_, ch) = filtered_s[right];
            *window_counts.entry(ch).or_insert(0) += 1;
            
            if window_counts[&ch] == t_count[&ch] {
                formed += 1;
            }
            
            while left <= right && formed == required {
                let (start_idx, _) = filtered_s[left];
                let (end_idx, _) = filtered_s[right];
                let current_len = end_idx - start_idx + 1;
                
                if current_len < min_len {
                    min_len = current_len;
                    min_start = start_idx;
                }
                
                let (_, left_ch) = filtered_s[left];
                *window_counts.get_mut(&left_ch).unwrap() -= 1;
                
                if window_counts[&left_ch] < t_count[&left_ch] {
                    formed -= 1;
                }
                
                left += 1;
            }
        }
        
        if min_len == usize::MAX {
            String::new()
        } else {
            s_chars[min_start..min_start + min_len].iter().collect()
        }
    }
    
    /// Approach 3: Array-based Character Counting
    /// 
    /// Use arrays instead of HashMap for character counting (ASCII optimization).
    /// 
    /// Time Complexity: O(|s| + |t|)
    /// Space Complexity: O(1) - fixed size arrays
    pub fn min_window_array_count(s: String, t: String) -> String {
        if s.is_empty() || t.is_empty() || s.len() < t.len() {
            return String::new();
        }
        
        let s_chars: Vec<char> = s.chars().collect();
        let t_chars: Vec<char> = t.chars().collect();
        
        // Count characters in t using array
        let mut t_count = [0; 128]; // ASCII
        let mut unique_chars = 0;
        
        for &ch in &t_chars {
            let idx = ch as usize;
            if t_count[idx] == 0 {
                unique_chars += 1;
            }
            t_count[idx] += 1;
        }
        
        let mut window_count = [0; 128];
        let mut formed = 0;
        let mut left = 0;
        let mut min_len = usize::MAX;
        let mut min_left = 0;
        
        for right in 0..s_chars.len() {
            let ch_idx = s_chars[right] as usize;
            window_count[ch_idx] += 1;
            
            if t_count[ch_idx] > 0 && window_count[ch_idx] == t_count[ch_idx] {
                formed += 1;
            }
            
            while left <= right && formed == unique_chars {
                let current_len = right - left + 1;
                if current_len < min_len {
                    min_len = current_len;
                    min_left = left;
                }
                
                let left_ch_idx = s_chars[left] as usize;
                window_count[left_ch_idx] -= 1;
                
                if t_count[left_ch_idx] > 0 && window_count[left_ch_idx] < t_count[left_ch_idx] {
                    formed -= 1;
                }
                
                left += 1;
            }
        }
        
        if min_len == usize::MAX {
            String::new()
        } else {
            s_chars[min_left..min_left + min_len].iter().collect()
        }
    }
    
    /// Approach 4: Two-Pointer with Early Termination
    /// 
    /// Add early termination optimizations and bounds checking.
    /// 
    /// Time Complexity: O(|s| + |t|)
    /// Space Complexity: O(|s| + |t|)
    pub fn min_window_early_termination(s: String, t: String) -> String {
        if s.is_empty() || t.is_empty() || s.len() < t.len() {
            return String::new();
        }
        
        let s_chars: Vec<char> = s.chars().collect();
        let t_chars: Vec<char> = t.chars().collect();
        
        // Early termination: check if all characters in t exist in s
        let mut s_char_set = std::collections::HashSet::new();
        for &ch in &s_chars {
            s_char_set.insert(ch);
        }
        
        for &ch in &t_chars {
            if !s_char_set.contains(&ch) {
                return String::new();
            }
        }
        
        let mut t_count = HashMap::new();
        for &ch in &t_chars {
            *t_count.entry(ch).or_insert(0) += 1;
        }
        
        let mut left = 0;
        let mut min_len = usize::MAX;
        let mut min_start = 0;
        let mut window_count = HashMap::new();
        let mut valid_chars = 0;
        
        for right in 0..s_chars.len() {
            let ch = s_chars[right];
            
            if t_count.contains_key(&ch) {
                let count = window_count.entry(ch).or_insert(0);
                *count += 1;
                
                if *count == t_count[&ch] {
                    valid_chars += 1;
                }
                
                while valid_chars == t_count.len() {
                    let current_len = right - left + 1;
                    if current_len < min_len {
                        min_len = current_len;
                        min_start = left;
                    }
                    
                    // Early termination: if we found the theoretical minimum
                    if min_len == t_chars.len() {
                        break;
                    }
                    
                    let left_ch = s_chars[left];
                    if let Some(count) = window_count.get_mut(&left_ch) {
                        *count -= 1;
                        if *count < t_count[&left_ch] {
                            valid_chars -= 1;
                        }
                    }
                    
                    left += 1;
                }
            } else {
                // Skip characters not in t when contracting
                while left <= right && !t_count.contains_key(&s_chars[left]) {
                    left += 1;
                }
            }
        }
        
        if min_len == usize::MAX {
            String::new()
        } else {
            s_chars[min_start..min_start + min_len].iter().collect()
        }
    }
    
    /// Approach 5: Substring Generation with Validation
    /// 
    /// Generate all possible substrings and validate each one.
    /// 
    /// Time Complexity: O(|s|³)
    /// Space Complexity: O(|t|)
    pub fn min_window_brute_force(s: String, t: String) -> String {
        if s.is_empty() || t.is_empty() || s.len() < t.len() {
            return String::new();
        }
        
        let s_chars: Vec<char> = s.chars().collect();
        let t_chars: Vec<char> = t.chars().collect();
        
        let mut t_count = HashMap::new();
        for &ch in &t_chars {
            *t_count.entry(ch).or_insert(0) += 1;
        }
        
        let mut min_window = String::new();
        let mut min_len = usize::MAX;
        
        for i in 0..s_chars.len() {
            for j in i + t_chars.len() - 1..s_chars.len() {
                let window: String = s_chars[i..=j].iter().collect();
                if Self::contains_all_chars(&window, &t_count) {
                    if window.len() < min_len {
                        min_len = window.len();
                        min_window = window;
                    }
                    break; // Found minimum for this starting position
                }
            }
        }
        
        min_window
    }
    
    fn contains_all_chars(window: &str, t_count: &HashMap<char, i32>) -> bool {
        let mut window_count = HashMap::new();
        for ch in window.chars() {
            *window_count.entry(ch).or_insert(0) += 1;
        }
        
        for (&ch, &count) in t_count {
            if window_count.get(&ch).unwrap_or(&0) < &count {
                return false;
            }
        }
        
        true
    }
    
    /// Approach 6: Binary Search on Answer Length
    /// 
    /// Binary search on the length of the minimum window.
    /// For each length, check if a valid window exists.
    /// 
    /// Time Complexity: O(|s| * log(|s|))
    /// Space Complexity: O(|t|)
    pub fn min_window_binary_search(s: String, t: String) -> String {
        if s.is_empty() || t.is_empty() || s.len() < t.len() {
            return String::new();
        }
        
        let s_chars: Vec<char> = s.chars().collect();
        let t_chars: Vec<char> = t.chars().collect();
        
        let mut t_count = HashMap::new();
        for &ch in &t_chars {
            *t_count.entry(ch).or_insert(0) += 1;
        }
        
        let mut left = t_chars.len();
        let mut right = s_chars.len();
        let mut result = String::new();
        
        while left <= right {
            let mid = left + (right - left) / 2;
            
            if let Some(window) = Self::find_window_of_length(&s_chars, &t_count, mid) {
                result = window;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        
        result
    }
    
    fn find_window_of_length(s_chars: &[char], t_count: &HashMap<char, i32>, length: usize) -> Option<String> {
        if length > s_chars.len() {
            return None;
        }
        
        for i in 0..=s_chars.len() - length {
            let window = &s_chars[i..i + length];
            let window_str: String = window.iter().collect();
            
            if Self::contains_all_chars(&window_str, t_count) {
                return Some(window_str);
            }
        }
        
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_case() {
        assert_eq!(Solution::min_window_sliding_window("ADOBECODEBANC".to_string(), "ABC".to_string()), "BANC");
        assert_eq!(Solution::min_window_optimized("ADOBECODEBANC".to_string(), "ABC".to_string()), "BANC");
    }
    
    #[test]
    fn test_single_character() {
        assert_eq!(Solution::min_window_array_count("a".to_string(), "a".to_string()), "a");
        assert_eq!(Solution::min_window_early_termination("a".to_string(), "a".to_string()), "a");
    }
    
    #[test]
    fn test_no_valid_window() {
        assert_eq!(Solution::min_window_brute_force("a".to_string(), "aa".to_string()), "");
        assert_eq!(Solution::min_window_binary_search("a".to_string(), "aa".to_string()), "");
    }
    
    #[test]
    fn test_entire_string() {
        assert_eq!(Solution::min_window_sliding_window("abc".to_string(), "abc".to_string()), "abc");
        assert_eq!(Solution::min_window_optimized("cba".to_string(), "abc".to_string()), "cba");
    }
    
    #[test]
    fn test_duplicates() {
        assert_eq!(Solution::min_window_array_count("ADOBECODEBANC".to_string(), "AABC".to_string()), "ADOBECODEBA");
        assert_eq!(Solution::min_window_early_termination("aa".to_string(), "aa".to_string()), "aa");
    }
    
    #[test]
    fn test_empty_strings() {
        assert_eq!(Solution::min_window_sliding_window("".to_string(), "a".to_string()), "");
        assert_eq!(Solution::min_window_optimized("a".to_string(), "".to_string()), "");
    }
    
    #[test]
    fn test_case_sensitivity() {
        assert_eq!(Solution::min_window_array_count("Aa".to_string(), "A".to_string()), "A");
        assert_eq!(Solution::min_window_early_termination("aA".to_string(), "A".to_string()), "A");
    }
    
    #[test]
    fn test_multiple_valid_windows() {
        assert_eq!(Solution::min_window_brute_force("ab".to_string(), "b".to_string()), "b");
        assert_eq!(Solution::min_window_binary_search("ba".to_string(), "b".to_string()), "b");
    }
    
    #[test]
    fn test_long_pattern() {
        assert_eq!(Solution::min_window_sliding_window("abcdef".to_string(), "abcdef".to_string()), "abcdef");
        assert_eq!(Solution::min_window_optimized("fedcba".to_string(), "abcdef".to_string()), "fedcba");
    }
    
    #[test]
    fn test_pattern_longer_than_string() {
        assert_eq!(Solution::min_window_array_count("a".to_string(), "abc".to_string()), "");
        assert_eq!(Solution::min_window_early_termination("ab".to_string(), "abc".to_string()), "");
    }
    
    #[test]
    fn test_repeated_characters() {
        assert_eq!(Solution::min_window_sliding_window("aaab".to_string(), "aab".to_string()), "aab");
        assert_eq!(Solution::min_window_optimized("baaab".to_string(), "aab".to_string()), "baa");
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            ("ADOBECODEBANC", "ABC"),
            ("a", "a"),
            ("a", "aa"),
            ("abc", "abc"),
            ("ADOBECODEBANC", "AABC"),
            ("aa", "aa"),
            ("", "a"),
            ("a", ""),
            ("Aa", "A"),
            ("ab", "b"),
            ("abcdef", "abcdef"),
            ("a", "abc"),
            ("aaab", "aab"),
            ("baaab", "aab"),
            ("pwwkew", "wke"),
        ];
        
        for (s, t) in test_cases {
            let s = s.to_string();
            let t = t.to_string();
            
            let result1 = Solution::min_window_sliding_window(s.clone(), t.clone());
            let result2 = Solution::min_window_optimized(s.clone(), t.clone());
            let result3 = Solution::min_window_array_count(s.clone(), t.clone());
            let result4 = Solution::min_window_early_termination(s.clone(), t.clone());
            let result5 = Solution::min_window_brute_force(s.clone(), t.clone());
            let result6 = Solution::min_window_binary_search(s.clone(), t.clone());
            
            // All should have same length (though actual string might differ for ties)
            assert_eq!(result1.len(), result2.len(), "SlidingWindow vs Optimized length mismatch for s='{}', t='{}'", s, t);
            assert_eq!(result2.len(), result3.len(), "Optimized vs ArrayCount length mismatch for s='{}', t='{}'", s, t);
            assert_eq!(result3.len(), result4.len(), "ArrayCount vs EarlyTermination length mismatch for s='{}', t='{}'", s, t);
            assert_eq!(result4.len(), result5.len(), "EarlyTermination vs BruteForce length mismatch for s='{}', t='{}'", s, t);
            assert_eq!(result5.len(), result6.len(), "BruteForce vs BinarySearch length mismatch for s='{}', t='{}'", s, t);
            
            // All non-empty results should contain all characters from t
            if !result1.is_empty() {
                let mut t_count = HashMap::new();
                for ch in t.chars() { 
                    *t_count.entry(ch).or_insert(0) += 1; 
                }
                assert!(Solution::contains_all_chars(&result1, &t_count), 
                    "Result '{}' doesn't contain all chars from '{}'", result1, t);
            }
        }
    }
}