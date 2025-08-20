use rust_leetcode::easy::{
    valid_parentheses::Solution as ParenthesesSolution,
    palindrome_number::Solution as PalindromeSolution,
};
use rust_leetcode::medium::{
    longest_substring_without_repeating_characters::Solution as LongestSubstringSolution,
    longest_palindromic_substring::Solution as PalindromicSubstringSolution,
    three_sum::Solution as ThreeSumSolution,
};
use rust_leetcode::hard::{
    median_of_two_sorted_arrays::Solution as MedianSolution,
    trapping_rain_water::Solution as TrappingRainWaterSolution,
};
use proptest::prelude::*;

// Property-based tests for Valid Parentheses
proptest! {
    #[test]
    fn test_valid_parentheses_properties(
        pairs in prop::collection::vec(0..3u8, 0..50)  // 0=(), 1=[], 2={}
    ) {
        let mut s = String::new();
        let mut stack = Vec::new();
        
        // Build a valid parentheses string
        for &pair_type in &pairs {
            let (open, close) = match pair_type {
                0 => ('(', ')'),
                1 => ('[', ']'),
                2 => ('{', '}'),
                _ => unreachable!(),
            };
            s.push(open);
            stack.push(close);
        }
        
        // Close in reverse order (making it valid)
        while let Some(close) = stack.pop() {
            s.push(close);
        }
        
        let solution = ParenthesesSolution::new();
        let result = solution.is_valid(s);
        prop_assert!(result); // Should always be valid
    }
    
    #[test] 
    fn test_invalid_parentheses_single_char(c in "[(){}]") {
        let solution = ParenthesesSolution::new();
        let result = solution.is_valid(c.to_string());
        prop_assert!(!result); // Single bracket should be invalid
    }
}

// Property-based tests for Palindrome Number
proptest! {
    #[test]
    fn test_palindrome_number_properties(n in 0..1000000i32) {
        let solution = PalindromeSolution::new();
        let result = solution.is_palindrome(n);
        
        // Verify by string comparison
        let s = n.to_string();
        let reversed: String = s.chars().rev().collect();
        let expected = s == reversed;
        
        prop_assert_eq!(result, expected);
    }
    
    #[test]
    fn test_palindrome_negative_numbers(n in -1000000..-1i32) {
        let solution = PalindromeSolution::new();
        let result = solution.is_palindrome(n);
        prop_assert!(!result); // Negative numbers should never be palindromes
    }
    
    #[test]
    fn test_palindrome_single_digit(n in 0..10i32) {
        let solution = PalindromeSolution::new();
        let result = solution.is_palindrome(n);
        prop_assert!(result); // Single digits are always palindromes
    }
}

// Property-based tests for Longest Substring Without Repeating Characters
proptest! {
    #[test]
    fn test_longest_substring_length_property(
        s in "[a-z]{0,100}"
    ) {
        let solution = LongestSubstringSolution::new();
        let result1 = solution.length_of_longest_substring(s.clone());
        let result2 = solution.length_of_longest_substring_hashset(s.clone());
        let result3 = solution.length_of_longest_substring_brute_force(s.clone());
        
        // All approaches should give same result
        prop_assert_eq!(result1, result2);
        prop_assert_eq!(result2, result3);
        
        // Result should not exceed string length
        prop_assert!(result1 <= s.len() as i32);
        prop_assert!(result1 >= 0);
        
        // If string is empty, result should be 0
        if s.is_empty() {
            prop_assert_eq!(result1, 0);
        }
    }
    
    #[test]
    fn test_longest_substring_unique_chars(
        chars in prop::collection::vec(prop::char::range('a', 'z'), 1..20)
    ) {
        // Create string with unique characters
        let unique_chars: std::collections::HashSet<_> = chars.into_iter().collect();
        let s: String = unique_chars.into_iter().collect();
        
        let solution = LongestSubstringSolution::new();
        let result = solution.length_of_longest_substring(s.clone());
        
        // Should equal the string length since all chars are unique
        prop_assert_eq!(result, s.len() as i32);
    }
    
    #[test]
    fn test_longest_substring_all_same(
        c in prop::char::range('a', 'z'),
        len in 1..50usize
    ) {
        let s = c.to_string().repeat(len);
        let solution = LongestSubstringSolution::new();
        let result = solution.length_of_longest_substring(s);
        
        // Should be 1 since all characters are the same
        prop_assert_eq!(result, 1);
    }
}

// Property-based tests for 3Sum
proptest! {
    #[test]
    fn test_three_sum_approaches_consistency(
        nums in prop::collection::vec(-100..100i32, 3..50)
    ) {
        let solution = ThreeSumSolution::new();
        let result1 = solution.three_sum(nums.clone());
        let result2 = solution.three_sum_hashset(nums.clone());
        
        // Both approaches should find same number of solutions
        prop_assert_eq!(result1.len(), result2.len());
        
        // All triplets should sum to zero
        for triplet in &result1 {
            prop_assert_eq!(triplet.len(), 3);
            prop_assert_eq!(triplet[0] + triplet[1] + triplet[2], 0);
        }
        
        for triplet in &result2 {
            prop_assert_eq!(triplet.len(), 3);
            prop_assert_eq!(triplet[0] + triplet[1] + triplet[2], 0);
        }
    }
    
    #[test]
    fn test_three_sum_no_duplicates(
        nums in prop::collection::vec(-50..50i32, 3..30)
    ) {
        let solution = ThreeSumSolution::new();
        let result = solution.three_sum(nums);
        
        // No duplicate triplets should exist
        let mut seen = std::collections::HashSet::new();
        for triplet in result {
            let mut sorted_triplet = triplet.clone();
            sorted_triplet.sort();
            prop_assert!(seen.insert(sorted_triplet));
        }
    }
    
    #[test]
    fn test_three_sum_small_arrays(
        a in -10..10i32,
        b in -10..10i32,
    ) {
        let nums = vec![a, b];
        let solution = ThreeSumSolution::new();
        let result = solution.three_sum(nums);
        
        // Should return empty for arrays with less than 3 elements
        prop_assert!(result.is_empty());
    }
}

// Property-based tests for Median of Two Sorted Arrays
proptest! {
    #[test] 
    fn test_median_approaches_consistency(
        mut nums1 in prop::collection::vec(-1000..1000i32, 0..20),
        mut nums2 in prop::collection::vec(-1000..1000i32, 0..20)
    ) {
        // Skip test if both arrays are empty (invalid input)
        if nums1.is_empty() && nums2.is_empty() {
            return Ok(());
        }
        
        // Ensure arrays are sorted
        nums1.sort();
        nums2.sort();
        
        let solution = MedianSolution::new();
        let result1 = solution.find_median_sorted_arrays(nums1.clone(), nums2.clone());
        let result2 = solution.find_median_sorted_arrays_merge(nums1.clone(), nums2.clone());
        
        // Both approaches should give same result (within floating point precision)
        let diff = (result1 - result2).abs();
        prop_assert!(diff < 1e-10);
        
        // Verify by merging and finding median manually
        let mut merged = nums1;
        merged.extend(nums2);
        merged.sort();
        
        let expected_median = if merged.len() % 2 == 1 {
            merged[merged.len() / 2] as f64
        } else {
            let mid = merged.len() / 2;
            (merged[mid - 1] + merged[mid]) as f64 / 2.0
        };
        
        let diff = (result1 - expected_median).abs();
        prop_assert!(diff < 1e-10);
    }
    
    #[test]
    fn test_median_single_element_arrays(
        a in -1000..1000i32,
        b in -1000..1000i32
    ) {
        let solution = MedianSolution::new();
        let result = solution.find_median_sorted_arrays(vec![a], vec![b]);
        let expected = (a + b) as f64 / 2.0;
        
        let diff = (result - expected).abs();
        prop_assert!(diff < 1e-10);
    }
    
    #[test]
    fn test_median_empty_array(
        mut nums in prop::collection::vec(-100..100i32, 1..20)
    ) {
        nums.sort();
        let solution = MedianSolution::new();
        
        let result1 = solution.find_median_sorted_arrays(vec![], nums.clone());
        let result2 = solution.find_median_sorted_arrays(nums.clone(), vec![]);
        
        let expected = if nums.len() % 2 == 1 {
            nums[nums.len() / 2] as f64
        } else {
            let mid = nums.len() / 2;
            (nums[mid - 1] + nums[mid]) as f64 / 2.0
        };
        
        let diff1 = (result1 - expected).abs();
        let diff2 = (result2 - expected).abs();
        prop_assert!(diff1 < 1e-10);
        prop_assert!(diff2 < 1e-10);
    }
}

// Property-based tests for Trapping Rain Water
proptest! {
    #[test]
    fn test_trapping_rain_water_approaches_consistency(
        height in prop::collection::vec(0..10i32, 0..50)
    ) {
        let solution = TrappingRainWaterSolution::new();
        let result1 = solution.trap(height.clone());
        let result2 = solution.trap_dp(height.clone());
        let result3 = solution.trap_stack(height.clone());
        let result4 = solution.trap_brute_force(height.clone());
        
        // All approaches should give same result
        prop_assert_eq!(result1, result2);
        prop_assert_eq!(result2, result3);
        prop_assert_eq!(result3, result4);
        
        // Result should be non-negative
        prop_assert!(result1 >= 0);
    }
    
    #[test]
    fn test_trapping_rain_water_monotonic_increasing(
        start in 0..5i32,
        len in 1..20usize
    ) {
        let height: Vec<i32> = (start..start + len as i32).collect();
        let solution = TrappingRainWaterSolution::new();
        let result = solution.trap(height);
        
        // Monotonic increasing should trap no water
        prop_assert_eq!(result, 0);
    }
    
    #[test]
    fn test_trapping_rain_water_monotonic_decreasing(
        start in 10..20i32,
        len in 2..10usize
    ) {
        // Create decreasing sequence, ensuring all values are non-negative
        let height: Vec<i32> = (0..len).map(|i| (start - i as i32).max(0)).collect();
        
        // Ensure it's actually monotonic decreasing (skip if not)
        let mut is_decreasing = true;
        for i in 1..height.len() {
            if height[i] > height[i-1] {
                is_decreasing = false;
                break;
            }
        }
        
        if !is_decreasing {
            return Ok(());
        }
        
        let solution = TrappingRainWaterSolution::new();
        let result = solution.trap(height);
        
        // Monotonic decreasing should trap no water
        prop_assert_eq!(result, 0);
    }
    
    #[test]
    fn test_trapping_rain_water_single_valley(
        left in 1..10i32,
        right in 1..10i32,
        width in 1..20usize
    ) {
        let mut height = vec![left];
        height.extend(vec![0; width]);
        height.push(right);
        
        let solution = TrappingRainWaterSolution::new();
        let result = solution.trap(height);
        
        // Should trap width * min(left, right) units
        let expected = width as i32 * std::cmp::min(left, right);
        prop_assert_eq!(result, expected);
    }
    
    #[test]
    fn test_trapping_rain_water_flat_terrain(
        h in 0..10i32,
        len in 1..50usize
    ) {
        let height = vec![h; len];
        let solution = TrappingRainWaterSolution::new();
        let result = solution.trap(height);
        
        // Flat terrain should trap no water
        prop_assert_eq!(result, 0);
    }
}

// Property-based tests for Longest Palindromic Substring
proptest! {
    #[test]
    fn test_longest_palindromic_substring_approaches_consistency(
        s in "[a-c]{0,50}" // Limited alphabet to increase palindrome likelihood
    ) {
        let solution = PalindromicSubstringSolution::new();
        let result1 = solution.longest_palindrome(s.clone());
        let result2 = solution.longest_palindrome_dp(s.clone());
        let result3 = solution.longest_palindrome_manacher(s.clone());
        
        // All approaches should give same length results
        prop_assert_eq!(result1.len(), result2.len());
        prop_assert_eq!(result2.len(), result3.len());
        
        // All results should be palindromes
        prop_assert!(is_palindrome(&result1));
        prop_assert!(is_palindrome(&result2));
        prop_assert!(is_palindrome(&result3));
        
        // Results should be substrings of original
        if !s.is_empty() {
            prop_assert!(s.contains(&result1));
            prop_assert!(s.contains(&result2));
            prop_assert!(s.contains(&result3));
        }
    }
    
    #[test]
    fn test_longest_palindromic_substring_single_char(
        s in "[a-z]"
    ) {
        let solution = PalindromicSubstringSolution::new();
        let result = solution.longest_palindrome(s.clone());
        
        // Single character should return itself
        prop_assert_eq!(result, s);
    }
    
    #[test]
    fn test_longest_palindromic_substring_all_same(
        c in prop::char::range('a', 'z'),
        len in 1..30usize
    ) {
        let s = c.to_string().repeat(len);
        let solution = PalindromicSubstringSolution::new();
        let result = solution.longest_palindrome(s.clone());
        
        // Should return the entire string since it's all the same character
        prop_assert_eq!(result, s);
    }
}

fn is_palindrome(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    for i in 0..len / 2 {
        if chars[i] != chars[len - 1 - i] {
            return false;
        }
    }
    true
}