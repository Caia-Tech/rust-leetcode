//! # Problem 13: Roman to Integer
//!
//! Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.
//!
//! | Symbol | Value |
//! |--------|-------|
//! | I      | 1     |
//! | V      | 5     |
//! | X      | 10    |
//! | L      | 50    |
//! | C      | 100   |
//! | D      | 500   |
//! | M      | 1000  |
//!
//! For example, `2` is written as `II`, just two ones added together. `12` is written as `XII`, 
//! which is simply `X + II`. The number `27` is written as `XXVII`, which is `XX + V + II`.
//!
//! Roman numerals are usually written largest to smallest from left to right. However, the numeral 
//! for four is not `IIII`. Instead, the number four is written as `IV`. Because the one is before 
//! the five we subtract it making four. The same principle applies to the number nine, which is 
//! written as `IX`. There are six instances where subtraction is used:
//!
//! - `I` can be placed before `V` (5) and `X` (10) to make 4 and 9. 
//! - `X` can be placed before `L` (50) and `C` (100) to make 40 and 90. 
//! - `C` can be placed before `D` (500) and `M` (1000) to make 400 and 900.
//!
//! Given a roman numeral, convert it to an integer.
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::easy::roman_to_integer::Solution;
//! 
//! let solution = Solution::new();
//! 
//! // Example 1:
//! assert_eq!(solution.roman_to_int("III".to_string()), 3);
//! 
//! // Example 2: 
//! assert_eq!(solution.roman_to_int("LVIII".to_string()), 58);
//! 
//! // Example 3:
//! assert_eq!(solution.roman_to_int("MCMXC".to_string()), 1990);
//! ```
//!
//! ## Constraints
//!
//! - 1 <= s.length <= 15
//! - s contains only the characters ('I', 'V', 'X', 'L', 'C', 'D', 'M').
//! - It is guaranteed that s is a valid roman numeral in the range [1, 3999].

use std::collections::HashMap;

/// Solution struct following LeetCode format
pub struct Solution;

impl Solution {
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Right-to-Left Traversal (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Process the Roman numeral from right to left
    /// 2. If current value is less than the previously seen maximum, subtract it
    /// 3. Otherwise, add it to the result
    /// 4. Keep track of the maximum value seen so far
    /// 
    /// **Time Complexity:** O(n) - Single pass through the string
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **Key Insight:** When reading right-to-left, if we encounter a smaller value
    /// than what we've seen before, it means we're in a subtraction case (like IV, IX).
    /// 
    /// **Why this works:**
    /// - "IV": Process V(5) first, then I(1). Since 1 < 5, subtract: 5 - 1 = 4
    /// - "VI": Process I(1) first, then V(5). Since 5 > 1, add: 1 + 5 = 6
    /// - "MCMXC": M(1000) + C(100) - M(1000) + X(10) - C(100) = 1990
    pub fn roman_to_int(&self, s: String) -> i32 {
        let chars: Vec<char> = s.chars().collect();
        let mut result = 0;
        let mut prev_value = 0;
        
        // Process from right to left
        for &ch in chars.iter().rev() {
            let current_value = self.char_to_value(ch);
            
            if current_value < prev_value {
                // Subtraction case (e.g., I in IV, X in XC)
                result -= current_value;
            } else {
                // Addition case (normal or dominant symbol)
                result += current_value;
            }
            
            prev_value = current_value;
        }
        
        result
    }

    /// # Approach 2: Left-to-Right with Lookahead
    /// 
    /// **Algorithm:**
    /// 1. Process from left to right
    /// 2. Look ahead to the next character
    /// 3. If current < next, subtract current (subtraction case)
    /// 4. Otherwise, add current
    /// 
    /// **Time Complexity:** O(n) - Single pass with lookahead
    /// **Space Complexity:** O(1) - Constant extra space
    /// 
    /// **Pattern Recognition:**
    /// - If we see I followed by V or X: subtract I
    /// - If we see X followed by L or C: subtract X  
    /// - If we see C followed by D or M: subtract C
    pub fn roman_to_int_lookahead(&self, s: String) -> i32 {
        let chars: Vec<char> = s.chars().collect();
        let mut result = 0;
        let len = chars.len();
        
        for i in 0..len {
            let current_value = self.char_to_value(chars[i]);
            
            // Look ahead to next character
            if i + 1 < len {
                let next_value = self.char_to_value(chars[i + 1]);
                
                if current_value < next_value {
                    // Subtraction case
                    result -= current_value;
                } else {
                    // Addition case
                    result += current_value;
                }
            } else {
                // Last character, always add
                result += current_value;
            }
        }
        
        result
    }

    /// # Approach 3: HashMap with Pattern Matching
    /// 
    /// **Algorithm:**
    /// 1. Create mappings for both single characters and subtraction patterns
    /// 2. Process string by checking for two-character patterns first
    /// 3. Fall back to single characters
    /// 
    /// **Time Complexity:** O(n) - Single pass with pattern checking
    /// **Space Complexity:** O(1) - Fixed-size HashMaps
    /// 
    /// **Advantage:** Very explicit about all valid Roman numeral patterns
    pub fn roman_to_int_hashmap(&self, s: String) -> i32 {
        let mut values = HashMap::new();
        
        // Single character mappings
        values.insert("I", 1);
        values.insert("V", 5);
        values.insert("X", 10);
        values.insert("L", 50);
        values.insert("C", 100);
        values.insert("D", 500);
        values.insert("M", 1000);
        
        // Subtraction patterns
        values.insert("IV", 4);
        values.insert("IX", 9);
        values.insert("XL", 40);
        values.insert("XC", 90);
        values.insert("CD", 400);
        values.insert("CM", 900);
        
        let mut result = 0;
        let mut i = 0;
        let chars: Vec<char> = s.chars().collect();
        
        while i < chars.len() {
            // Try two-character pattern first
            if i + 1 < chars.len() {
                let two_char: String = chars[i..i+2].iter().collect();
                if let Some(&value) = values.get(two_char.as_str()) {
                    result += value;
                    i += 2;
                    continue;
                }
            }
            
            // Single character
            let one_char: String = chars[i..i+1].iter().collect();
            if let Some(&value) = values.get(one_char.as_str()) {
                result += value;
            }
            i += 1;
        }
        
        result
    }

    /// # Approach 4: Mathematical Replacement
    /// 
    /// **Algorithm:**
    /// 1. Replace all subtraction patterns with equivalent addition patterns
    /// 2. Sum all individual character values
    /// 
    /// **Time Complexity:** O(n) - String replacement operations
    /// **Space Complexity:** O(n) - String manipulation creates new strings
    /// 
    /// **Note:** Less efficient due to string operations, but conceptually clear
    pub fn roman_to_int_replacement(&self, s: String) -> i32 {
        let mut modified = s;
        
        // Replace subtraction patterns with equivalent values
        modified = modified.replace("CM", "DCCCC");  // 900 = 500 + 400
        modified = modified.replace("CD", "CCCC");   // 400 = 100 * 4
        modified = modified.replace("XC", "LXXXX");  // 90 = 50 + 40
        modified = modified.replace("XL", "XXXX");   // 40 = 10 * 4
        modified = modified.replace("IX", "VIIII");  // 9 = 5 + 4
        modified = modified.replace("IV", "IIII");   // 4 = 1 * 4
        
        // Now sum all characters
        modified.chars()
            .map(|ch| self.char_to_value(ch))
            .sum()
    }

    /// Helper function to convert a Roman character to its integer value
    fn char_to_value(&self, ch: char) -> i32 {
        match ch {
            'I' => 1,
            'V' => 5,
            'X' => 10,
            'L' => 50,
            'C' => 100,
            'D' => 500,
            'M' => 1000,
            _ => 0, // Should never happen with valid input
        }
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
    use rstest::rstest;

    fn setup() -> Solution {
        Solution::new()
    }

    #[rstest]
    #[case("III", 3)]
    #[case("LVIII", 58)]  // L=50, V=5, III=3
    #[case("MCMXC", 1990)] // M=1000, CM=900, XC=90
    #[case("IV", 4)]
    #[case("IX", 9)]
    #[case("XL", 40)]
    #[case("XC", 90)]
    #[case("CD", 400)]
    #[case("CM", 900)]
    fn test_basic_cases(#[case] input: &str, #[case] expected: i32) {
        let solution = setup();
        assert_eq!(solution.roman_to_int(input.to_string()), expected);
    }

    #[rstest]
    #[case("I", 1)]
    #[case("V", 5)]
    #[case("X", 10)]
    #[case("L", 50)]
    #[case("C", 100)]
    #[case("D", 500)]
    #[case("M", 1000)]
    fn test_single_characters(#[case] input: &str, #[case] expected: i32) {
        let solution = setup();
        assert_eq!(solution.roman_to_int(input.to_string()), expected);
    }

    #[test]
    fn test_complex_numbers() {
        let solution = setup();
        
        let test_cases = vec![
            ("MCDXLIV", 1444),    // 1000 + 400 + 40 + 4
            ("MMMDCCCLXXXVIII", 3888), // 3000 + 800 + 80 + 8
            ("MMMCMXCIX", 3999),  // 3000 + 900 + 90 + 9 (maximum)
            ("DCXXI", 621),       // 500 + 100 + 20 + 1
            ("CMLIV", 954),       // 900 + 50 + 4
        ];
        
        for (roman, expected) in test_cases {
            assert_eq!(solution.roman_to_int(roman.to_string()), expected,
                      "Failed for {}", roman);
        }
    }

    #[test]
    fn test_subtraction_patterns() {
        let solution = setup();
        
        // Test all valid subtraction patterns
        let patterns = vec![
            ("IV", 4),   // I before V
            ("IX", 9),   // I before X
            ("XL", 40),  // X before L
            ("XC", 90),  // X before C
            ("CD", 400), // C before D
            ("CM", 900), // C before M
        ];
        
        for (pattern, expected) in patterns {
            assert_eq!(solution.roman_to_int(pattern.to_string()), expected,
                      "Subtraction pattern {} failed", pattern);
        }
    }

    #[test]
    fn test_multiple_subtraction_patterns() {
        let solution = setup();
        
        let test_cases = vec![
            ("MCMXC", 1990),     // M + CM + XC = 1000 + 900 + 90
            ("MCDXLIV", 1444),   // M + CD + XL + IV = 1000 + 400 + 40 + 4
            ("CMXC", 990),       // CM + XC = 900 + 90
            ("CDXLIV", 444),     // CD + XL + IV = 400 + 40 + 4
        ];
        
        for (roman, expected) in test_cases {
            assert_eq!(solution.roman_to_int(roman.to_string()), expected,
                      "Multiple subtraction test failed for {}", roman);
        }
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // Minimum and maximum values
        assert_eq!(solution.roman_to_int("I".to_string()), 1);
        assert_eq!(solution.roman_to_int("MMMCMXCIX".to_string()), 3999);
        
        // All same character
        assert_eq!(solution.roman_to_int("III".to_string()), 3);
        assert_eq!(solution.roman_to_int("XXX".to_string()), 30);
        assert_eq!(solution.roman_to_int("CCC".to_string()), 300);
        assert_eq!(solution.roman_to_int("MMM".to_string()), 3000);
    }

    #[test]
    fn test_all_approaches_consistency() {
        let solution = setup();
        let test_cases = vec![
            "III", "LVIII", "MCMXC", "IV", "IX", "XL", "XC", "CD", "CM",
            "MCDXLIV", "MMMDCCCLXXXVIII", "MMMCMXCIX", "DCXXI", "CMLIV"
        ];
        
        for case in test_cases {
            let result1 = solution.roman_to_int(case.to_string());
            let result2 = solution.roman_to_int_lookahead(case.to_string());
            let result3 = solution.roman_to_int_hashmap(case.to_string());
            let result4 = solution.roman_to_int_replacement(case.to_string());
            
            assert_eq!(result1, result2, "Lookahead approach differs for {}", case);
            assert_eq!(result1, result3, "HashMap approach differs for {}", case);
            assert_eq!(result1, result4, "Replacement approach differs for {}", case);
        }
    }

    #[test]
    fn test_ascending_and_descending_patterns() {
        let solution = setup();
        
        // Pure ascending (no subtraction)
        assert_eq!(solution.roman_to_int("MDC".to_string()), 1600); // 1000 + 500 + 100
        assert_eq!(solution.roman_to_int("CLX".to_string()), 160);  // 100 + 50 + 10
        
        // Mixed patterns
        assert_eq!(solution.roman_to_int("MCDL".to_string()), 1450); // 1000 + 400 + 50
        assert_eq!(solution.roman_to_int("DCXLIV".to_string()), 644); // 500 + 100 + 40 + 4
    }

    #[test]
    fn test_performance_characteristics() {
        let solution = setup();
        
        // Test with maximum length string (15 characters)
        let long_roman = "MMMCMXCVIII"; // 3000 + 900 + 90 + 8 = 3998
        let result = solution.roman_to_int(long_roman.to_string());
        
        // Verify all approaches handle long strings correctly
        let result2 = solution.roman_to_int_lookahead(long_roman.to_string());
        let result3 = solution.roman_to_int_hashmap(long_roman.to_string());
        let result4 = solution.roman_to_int_replacement(long_roman.to_string());
        
        assert_eq!(result, result2);
        assert_eq!(result, result3);
        assert_eq!(result, result4);
    }
}