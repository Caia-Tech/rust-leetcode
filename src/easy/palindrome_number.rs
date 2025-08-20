//! # Problem 9: Palindrome Number
//!
//! Given an integer `x`, return `true` if `x` is a palindrome, and `false` otherwise.
//! 
//! An integer is a palindrome when it reads the same backward as forward.
//! - For example, `121` is a palindrome while `123` is not.
//!
//! **Follow up:** Could you solve it without converting the integer to a string?
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::easy::palindrome_number::Solution;
//! 
//! let solution = Solution::new();
//! 
//! // Example 1:
//! assert_eq!(solution.is_palindrome(121), true);
//! 
//! // Example 2: 
//! assert_eq!(solution.is_palindrome(-121), false);
//! 
//! // Example 3:
//! assert_eq!(solution.is_palindrome(10), false);
//! ```
//!
//! ## Constraints
//!
//! - -2^31 <= x <= 2^31 - 1

/// Solution struct following LeetCode format
pub struct Solution;

impl Solution {
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Half Number Reversal (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Handle edge cases: negative numbers and numbers ending in 0
    /// 2. Reverse only half of the number to avoid overflow
    /// 3. Compare first half with reversed second half
    /// 4. Handle odd-length numbers by dividing middle digit
    /// 
    /// **Time Complexity:** O(log n) - Process half the digits
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **Key Insight:** We only need to reverse half the number. When the reversed
    /// half becomes greater than or equal to the remaining number, we've reached the middle.
    /// 
    /// **Why this works:**
    /// - For even-length numbers: 1221 -> original=12, reversed=12
    /// - For odd-length numbers: 12321 -> original=12, reversed=123, but we check 12 == 123/10
    pub fn is_palindrome(&self, x: i32) -> bool {
        // Negative numbers are not palindromes
        // Numbers ending in 0 (except 0 itself) are not palindromes
        if x < 0 || (x % 10 == 0 && x != 0) {
            return false;
        }
        
        let mut original = x;
        let mut reversed_half = 0;
        
        // Reverse half of the number
        // Stop when reversed_half >= original (we've reached or passed the middle)
        while original > reversed_half {
            reversed_half = reversed_half * 10 + original % 10;
            original /= 10;
        }
        
        // For even-length numbers: original == reversed_half
        // For odd-length numbers: original == reversed_half / 10 (ignore middle digit)
        original == reversed_half || original == reversed_half / 10
    }

    /// # Approach 2: String Conversion (Simple but uses extra space)
    /// 
    /// **Algorithm:**
    /// 1. Convert number to string
    /// 2. Use two pointers to compare characters from both ends
    /// 3. Move pointers towards center
    /// 
    /// **Time Complexity:** O(log n) - Number of digits
    /// **Space Complexity:** O(log n) - String storage
    /// 
    /// **Note:** While intuitive, this approach uses extra space and string operations
    pub fn is_palindrome_string(&self, x: i32) -> bool {
        if x < 0 {
            return false;
        }
        
        let s = x.to_string();
        let chars: Vec<char> = s.chars().collect();
        let len = chars.len();
        
        for i in 0..len / 2 {
            if chars[i] != chars[len - 1 - i] {
                return false;
            }
        }
        
        true
    }

    /// # Approach 3: Full Number Reversal (Brute Force)
    /// 
    /// **Algorithm:**
    /// 1. Reverse the entire number
    /// 2. Compare with original
    /// 3. Handle potential overflow during reversal
    /// 
    /// **Time Complexity:** O(log n) - All digits processed
    /// **Space Complexity:** O(1) - Constant extra space
    /// 
    /// **Drawback:** Potential overflow issues for large numbers
    pub fn is_palindrome_full_reverse(&self, x: i32) -> bool {
        if x < 0 {
            return false;
        }
        
        let original = x;
        let mut num = x;
        let mut reversed = 0i64; // Use i64 to handle potential overflow
        
        while num > 0 {
            reversed = reversed * 10 + (num % 10) as i64;
            num /= 10;
        }
        
        original as i64 == reversed
    }

    /// # Approach 4: Digit Array Comparison (Educational)
    /// 
    /// **Algorithm:**
    /// 1. Extract all digits into a vector
    /// 2. Compare vector with its reverse
    /// 
    /// **Time Complexity:** O(log n) - Extract and compare digits
    /// **Space Complexity:** O(log n) - Store all digits
    /// 
    /// **Use Case:** Good for understanding digit manipulation
    pub fn is_palindrome_digit_array(&self, x: i32) -> bool {
        if x < 0 {
            return false;
        }
        
        let mut digits = Vec::new();
        let mut num = x;
        
        // Handle zero separately
        if num == 0 {
            return true;
        }
        
        // Extract digits
        while num > 0 {
            digits.push(num % 10);
            num /= 10;
        }
        
        // Compare with reverse (digits is already in reverse order)
        let len = digits.len();
        for i in 0..len / 2 {
            if digits[i] != digits[len - 1 - i] {
                return false;
            }
        }
        
        true
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
    #[case(121, true)]
    #[case(-121, false)]
    #[case(10, false)]
    #[case(0, true)]
    #[case(1, true)]
    #[case(11, true)]
    #[case(123, false)]
    #[case(12321, true)]
    #[case(1221, true)]
    #[case(12345, false)]
    fn test_is_palindrome_basic(#[case] input: i32, #[case] expected: bool) {
        let solution = setup();
        assert_eq!(solution.is_palindrome(input), expected);
    }

    #[rstest]
    #[case(-1, false)]
    #[case(-121, false)]
    #[case(-12321, false)]
    #[case(i32::MIN, false)]
    fn test_negative_numbers(#[case] input: i32, #[case] expected: bool) {
        let solution = setup();
        assert_eq!(solution.is_palindrome(input), expected);
    }

    #[rstest]
    #[case(10, false)]
    #[case(100, false)]
    #[case(1000, false)]
    #[case(12340, false)]
    fn test_numbers_ending_in_zero(#[case] input: i32, #[case] expected: bool) {
        let solution = setup();
        assert_eq!(solution.is_palindrome(input), expected);
    }

    #[test]
    fn test_single_digit_numbers() {
        let solution = setup();
        
        // All single digit numbers should be palindromes
        for i in 0..=9 {
            assert!(solution.is_palindrome(i), "Single digit {} should be palindrome", i);
        }
    }

    #[test]
    fn test_double_digit_palindromes() {
        let solution = setup();
        
        let palindromes = vec![11, 22, 33, 44, 55, 66, 77, 88, 99];
        for num in palindromes {
            assert!(solution.is_palindrome(num), "{} should be palindrome", num);
        }
        
        let non_palindromes = vec![10, 12, 13, 21, 23, 34, 45, 56, 67, 78, 89, 98];
        for num in non_palindromes {
            assert!(!solution.is_palindrome(num), "{} should not be palindrome", num);
        }
    }

    #[test]
    fn test_large_palindromes() {
        let solution = setup();
        
        let large_palindromes = vec![
            1234321,
            123454321,
            1111111,
            987656789,
            1000000001,
        ];
        
        for num in large_palindromes {
            assert!(solution.is_palindrome(num), "{} should be palindrome", num);
        }
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // Maximum and minimum values
        assert!(!solution.is_palindrome(i32::MAX)); // 2147483647
        assert!(!solution.is_palindrome(i32::MIN)); // -2147483648
        
        // Powers of 10
        assert!(solution.is_palindrome(0));
        assert!(!solution.is_palindrome(10));
        assert!(!solution.is_palindrome(100));
        assert!(!solution.is_palindrome(1000));
        
        // Numbers with repeating digits
        assert!(solution.is_palindrome(1111));
        assert!(solution.is_palindrome(2222));
        assert!(!solution.is_palindrome(1112));
    }

    #[test]
    fn test_all_approaches_consistency() {
        let solution = setup();
        let test_cases = vec![
            121, -121, 10, 0, 1, 11, 123, 12321, 1221, 12345,
            1234321, 987656789, 1000, 9999, 1001, 12344321
        ];
        
        for case in test_cases {
            let result1 = solution.is_palindrome(case);
            let result2 = solution.is_palindrome_string(case);
            let result3 = solution.is_palindrome_full_reverse(case);
            let result4 = solution.is_palindrome_digit_array(case);
            
            assert_eq!(result1, result2, "String approach differs for {}", case);
            assert_eq!(result1, result3, "Full reverse approach differs for {}", case);
            assert_eq!(result1, result4, "Digit array approach differs for {}", case);
        }
    }

    #[test]
    fn test_palindrome_properties() {
        let solution = setup();
        
        // Test palindromic patterns
        let patterns = vec![
            (1, true),
            (11, true),
            (121, true),
            (1221, true),
            (12321, true),
            (123321, true),
            (1234321, true),
        ];
        
        for (num, expected) in patterns {
            assert_eq!(solution.is_palindrome(num), expected, "Pattern test failed for {}", num);
        }
    }

    #[test]
    fn test_performance_critical_cases() {
        let solution = setup();
        
        // Test cases that might cause performance issues
        let large_cases = vec![
            -987654321,  // Large negative within i32 range
            1999999991,  // Large palindrome
            1000000001,  // Palindrome with zeros
            2000000002,  // Another large palindrome
        ];
        
        for case in large_cases {
            // Just ensure it doesn't panic and returns consistent results
            let result = solution.is_palindrome(case);
            let string_result = solution.is_palindrome_string(case);
            assert_eq!(result, string_result, "Inconsistent results for large case {}", case);
        }
    }
}