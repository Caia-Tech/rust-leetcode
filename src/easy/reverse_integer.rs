//! # Problem 7: Reverse Integer
//!
//! Given a signed 32-bit integer `x`, return `x` with its digits reversed. If reversing `x` causes 
//! the value to go outside the signed 32-bit integer range `[-2^31, 2^31 - 1]`, then return `0`.
//!
//! **Assume the environment does not allow you to store 64-bit integers (signed or unsigned).**
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::easy::reverse_integer::Solution;
//! 
//! let solution = Solution::new();
//! 
//! // Example 1:
//! assert_eq!(solution.reverse(123), 321);
//! 
//! // Example 2: 
//! assert_eq!(solution.reverse(-123), -321);
//! 
//! // Example 3:
//! assert_eq!(solution.reverse(120), 21);
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

    /// # Approach 1: Mathematical Reversal with Overflow Detection
    /// 
    /// **Algorithm:**
    /// 1. Extract digits one by one using modulo operation
    /// 2. Build reversed number by multiplying by 10 and adding digit
    /// 3. Check for overflow before each multiplication
    /// 4. Handle negative numbers by tracking sign separately
    /// 
    /// **Time Complexity:** O(log n) - Number of digits in the input
    /// **Space Complexity:** O(1) - Only using constant extra space
    /// 
    /// **Key Insight:** We must detect overflow BEFORE it happens, not after.
    /// For a 32-bit signed integer, the range is [-2,147,483,648, 2,147,483,647].
    /// 
    /// **Overflow Detection Logic:**
    /// - If result > i32::MAX / 10, then result * 10 will overflow
    /// - If result == i32::MAX / 10, then result * 10 + digit might overflow if digit > 7
    /// - Similar logic applies for underflow with i32::MIN
    pub fn reverse(&self, x: i32) -> i32 {
        let mut num = x;
        let mut result = 0i32;
        
        while num != 0 {
            let digit = num % 10;
            num /= 10;
            
            // Check for overflow before multiplication
            // i32::MAX = 2,147,483,647, so i32::MAX / 10 = 214,748,364
            if result > i32::MAX / 10 || (result == i32::MAX / 10 && digit > 7) {
                return 0;
            }
            
            // Check for underflow
            // i32::MIN = -2,147,483,648, so i32::MIN / 10 = -214,748,364
            if result < i32::MIN / 10 || (result == i32::MIN / 10 && digit < -8) {
                return 0;
            }
            
            result = result * 10 + digit;
        }
        
        result
    }

    /// # Approach 2: String-based Reversal (Alternative but less efficient)
    /// 
    /// **Algorithm:**
    /// 1. Convert number to string
    /// 2. Handle negative sign separately
    /// 3. Reverse the string
    /// 4. Parse back to integer with overflow checking
    /// 
    /// **Time Complexity:** O(log n) - String operations on digits
    /// **Space Complexity:** O(log n) - String storage for digits
    /// 
    /// **Note:** This approach uses more memory and is generally less preferred
    /// for competitive programming, but demonstrates alternative thinking.
    pub fn reverse_string_approach(&self, x: i32) -> i32 {
        let is_negative = x < 0;
        let mut s = x.abs().to_string();
        
        // Reverse the string
        s = s.chars().rev().collect();
        
        // Parse back to integer
        match s.parse::<i32>() {
            Ok(mut result) => {
                if is_negative {
                    result = -result;
                }
                result
            }
            Err(_) => 0, // Overflow occurred during parsing
        }
    }

    /// # Approach 3: Digit-by-digit with i64 intermediate (Conceptual)
    /// 
    /// **Algorithm:**
    /// 1. Use i64 to avoid overflow during calculation
    /// 2. Check bounds before returning i32
    /// 
    /// **Note:** This violates the problem constraint of not using 64-bit integers,
    /// but shows how the problem would be solved if that constraint didn't exist.
    /// 
    /// **Time Complexity:** O(log n)
    /// **Space Complexity:** O(1)
    pub fn reverse_with_i64(&self, x: i32) -> i32 {
        let mut num = x as i64;
        let mut result = 0i64;
        
        while num != 0 {
            result = result * 10 + num % 10;
            num /= 10;
        }
        
        // Check if result fits in i32 range
        if result > i32::MAX as i64 || result < i32::MIN as i64 {
            0
        } else {
            result as i32
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
    #[case(123, 321)]
    #[case(-123, -321)]
    #[case(120, 21)]
    #[case(0, 0)]
    #[case(1, 1)]
    #[case(-1, -1)]
    #[case(10, 1)]
    #[case(-10, -1)]
    fn test_reverse_basic_cases(#[case] input: i32, #[case] expected: i32) {
        let solution = setup();
        assert_eq!(solution.reverse(input), expected);
    }

    #[rstest]
    #[case(1534236469, 0)] // Would result in 9646324351, which overflows
    #[case(-1563847412, 0)] // Would result in -2147483651, which underflows
    #[case(i32::MAX, 0)] // 2147483647 -> 7463847412 (overflow)
    #[case(i32::MIN, 0)] // -2147483648 -> would underflow
    fn test_overflow_cases(#[case] input: i32, #[case] expected: i32) {
        let solution = setup();
        assert_eq!(solution.reverse(input), expected);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // Single digit numbers
        for i in -9..=9 {
            assert_eq!(solution.reverse(i), i);
        }
        
        // Numbers ending in zero
        assert_eq!(solution.reverse(1000), 1);
        assert_eq!(solution.reverse(-1000), -1);
        assert_eq!(solution.reverse(10200), 201);
        
        // Palindromic numbers
        assert_eq!(solution.reverse(121), 121);
        assert_eq!(solution.reverse(-121), -121);
    }

    #[test]
    fn test_boundary_values() {
        let solution = setup();
        
        // Values close to overflow boundary
        assert_eq!(solution.reverse(1463847412), 2147483641); // Just within range
        assert_eq!(solution.reverse(-1463847412), -2147483641); // Just within range
        
        // Test the exact boundary case
        // 2147483647 reversed would be 7463847412 (overflow)
        assert_eq!(solution.reverse(2147483647), 0);
        assert_eq!(solution.reverse(-2147483648), 0);
    }

    #[test]
    fn test_alternative_approaches_consistency() {
        let solution = setup();
        let test_cases = vec![123, -123, 120, 0, 1, -1, 1000, 121];
        
        for case in test_cases {
            let result1 = solution.reverse(case);
            let result2 = solution.reverse_string_approach(case);
            let result3 = solution.reverse_with_i64(case);
            
            assert_eq!(result1, result2, "String approach differs for {}", case);
            assert_eq!(result1, result3, "i64 approach differs for {}", case);
        }
    }

    #[test]
    fn test_mathematical_properties() {
        let solution = setup();
        
        // Reversing twice should give original (if no overflow)
        let test_values = vec![123, -123, 12, -12, 5];
        for val in test_values {
            let reversed = solution.reverse(val);
            if reversed != 0 { // Skip if overflow occurred
                let double_reversed = solution.reverse(reversed);
                assert_eq!(double_reversed, val, "Double reversal failed for {}", val);
            }
        }
    }

    #[test]
    fn test_overflow_detection_precision() {
        let solution = setup();
        
        // Test values right at the boundary
        // 214748364 * 10 + 7 = 2147483647 (i32::MAX) ✓
        // 214748364 * 10 + 8 = 2147483648 (overflow) ✗
        
        // Construct number that will test boundary: 463847412
        // Reverse: 214748364, next digit would be 6, so 2147483646 ✓
        assert_eq!(solution.reverse(463847412), 214748364);
        
        // Construct number that will overflow: 7463847412 -> but this itself overflows i32
        // So we test 1534236469 which reverses to 9646324351 (overflow)
        assert_eq!(solution.reverse(1534236469), 0);
    }
}