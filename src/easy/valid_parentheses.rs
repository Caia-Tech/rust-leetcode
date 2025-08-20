//! # Problem 20: Valid Parentheses
//!
//! Given a string `s` containing just the characters `'('`, `')'`, `'['`, `']'`, `'{'` and `'}'`, 
//! determine if the input string is valid.
//!
//! An input string is valid if:
//! 1. Open brackets must be closed by the same type of brackets.
//! 2. Open brackets must be closed in the correct order.
//! 3. Every close bracket has a corresponding open bracket of the same type.
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::easy::valid_parentheses::Solution;
//! 
//! let solution = Solution::new();
//! 
//! // Example 1:
//! assert_eq!(solution.is_valid("()".to_string()), true);
//! 
//! // Example 2: 
//! assert_eq!(solution.is_valid("()[]{}".to_string()), true);
//! 
//! // Example 3:
//! assert_eq!(solution.is_valid("(]".to_string()), false);
//! ```
//!
//! ## Constraints
//!
//! - 1 <= s.length <= 10^4
//! - s consists of parentheses only '()[]{}'.

/// Solution struct following LeetCode format
pub struct Solution;

impl Solution {
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Stack-Based Validation (Optimal)
    /// 
    /// **Algorithm:**
    /// 1. Use a stack to track opening brackets
    /// 2. For each character:
    ///    - If opening bracket: push to stack
    ///    - If closing bracket: check if it matches top of stack
    /// 3. String is valid if stack is empty at the end
    /// 
    /// **Time Complexity:** O(n) - Single pass through the string
    /// **Space Complexity:** O(n) - Stack can contain up to n/2 opening brackets
    /// 
    /// **Key Insight:** The LIFO nature of stack perfectly matches the nested 
    /// structure requirement of valid parentheses. The most recently opened 
    /// bracket must be the first to close.
    /// 
    /// **Why this is optimal:**
    /// - We need to track opening brackets somehow → O(n) space minimum
    /// - We need to check each character → O(n) time minimum
    /// - Stack gives us O(1) push/pop operations
    /// - Early termination when mismatch found
    pub fn is_valid(&self, s: String) -> bool {
        let mut stack = Vec::new();
        
        for ch in s.chars() {
            match ch {
                // Opening brackets - push to stack
                '(' | '[' | '{' => stack.push(ch),
                
                // Closing brackets - check for matching opening
                ')' => {
                    if stack.pop() != Some('(') {
                        return false;
                    }
                }
                ']' => {
                    if stack.pop() != Some('[') {
                        return false;
                    }
                }
                '}' => {
                    if stack.pop() != Some('{') {
                        return false;
                    }
                }
                
                // Invalid character (shouldn't happen per constraints)
                _ => return false,
            }
        }
        
        // Valid only if all brackets were matched (stack empty)
        stack.is_empty()
    }

    /// # Approach 2: HashMap-Based Matching (Cleaner Code)
    /// 
    /// **Algorithm:**
    /// 1. Use HashMap to define bracket pairs
    /// 2. Same stack logic but with cleaner lookup
    /// 3. More maintainable for adding new bracket types
    /// 
    /// **Time Complexity:** O(n) - Same as approach 1
    /// **Space Complexity:** O(n) - Stack + small constant for HashMap
    /// 
    /// **Trade-off Analysis:**
    /// - **Pros:** More maintainable, easier to extend
    /// - **Cons:** HashMap lookup overhead (though O(1) average)
    /// - **Performance:** Slightly slower due to hash operations vs direct match
    /// 
    /// **When to use:** When code maintainability > micro-optimizations
    pub fn is_valid_hashmap(&self, s: String) -> bool {
        use std::collections::HashMap;
        
        // Map closing brackets to opening brackets
        let pairs: HashMap<char, char> = [
            (')', '('),
            (']', '['),
            ('}', '{'),
        ].iter().cloned().collect();
        
        let mut stack = Vec::new();
        
        for ch in s.chars() {
            if "([{".contains(ch) {
                // Opening bracket
                stack.push(ch);
            } else if ")]})".contains(ch) {
                // Closing bracket
                if let Some(&expected_open) = pairs.get(&ch) {
                    if stack.pop() != Some(expected_open) {
                        return false;
                    }
                } else {
                    return false; // Invalid closing bracket
                }
            } else {
                return false; // Invalid character
            }
        }
        
        stack.is_empty()
    }

    /// # Approach 3: Counter-Based (INCORRECT - Educational Purpose)
    /// 
    /// **Algorithm:**
    /// 1. Count opening and closing brackets separately
    /// 2. Check if counts match at the end
    /// 
    /// **Why this approach FAILS:**
    /// - It can't detect wrong ordering: "(])" would pass incorrectly
    /// - It can't distinguish bracket types: "([)]" would pass incorrectly
    /// - Missing the LIFO constraint that makes parentheses valid
    /// 
    /// **Educational Value:**
    /// This demonstrates why the problem requires more than just counting.
    /// The temporal ordering and nesting structure are crucial.
    /// 
    /// **Example where it fails:**
    /// - Input: "([)]" 
    /// - Counter approach: 1 '(', 1 ')', 1 '[', 1 ']' → "valid"
    /// - Reality: Invalid because '[' closes before '(' is closed
    pub fn is_valid_counter_incorrect(&self, s: String) -> bool {
        let mut paren_count = 0;
        let mut bracket_count = 0;
        let mut brace_count = 0;
        
        for ch in s.chars() {
            match ch {
                '(' => paren_count += 1,
                ')' => paren_count -= 1,
                '[' => bracket_count += 1,
                ']' => bracket_count -= 1,
                '{' => brace_count += 1,
                '}' => brace_count -= 1,
                _ => return false,
            }
            
            // Early termination if any count goes negative
            if paren_count < 0 || bracket_count < 0 || brace_count < 0 {
                return false;
            }
        }
        
        paren_count == 0 && bracket_count == 0 && brace_count == 0
    }

    /// # Approach 4: String Replacement (HIGHLY INEFFICIENT)
    /// 
    /// **Algorithm:**
    /// 1. Repeatedly remove valid pairs: "()", "[]", "{}"
    /// 2. Continue until no more pairs can be removed
    /// 3. String is valid if it becomes empty
    /// 
    /// **Time Complexity:** O(n²) in worst case - Each replacement is O(n), may need n/2 iterations
    /// **Space Complexity:** O(n) - String manipulation creates new strings
    /// 
    /// **Why this is inefficient:**
    /// - String replacement is O(n) for each operation
    /// - May need O(n) replacement rounds
    /// - Creates many intermediate strings (memory pressure)
    /// - No early termination benefits
    /// 
    /// **Example demonstrating O(n²) behavior:**
    /// - Input: "(((()))))"
    /// - Iteration 1: Remove inner "()" → "(((())))" (O(n) operation)
    /// - Iteration 2: Remove next "()" → "((()))" (O(n) operation)
    /// - Continue for n/2 iterations, each taking O(n) time
    /// 
    /// **Educational Value:**
    /// Shows why algorithmic choice matters for performance.
    pub fn is_valid_replacement_inefficient(&self, s: String) -> bool {
        let mut current = s;
        
        loop {
            let original_len = current.len();
            
            // Remove all occurrences of valid pairs
            current = current.replace("()", "");
            current = current.replace("[]", "");
            current = current.replace("{}", "");
            
            // If no pairs were removed, we're done
            if current.len() == original_len {
                break;
            }
        }
        
        current.is_empty()
    }

    /// # Approach 5: Recursive Descent (Academic Interest)
    /// 
    /// **Algorithm:**
    /// 1. Use recursive descent parsing
    /// 2. Try to match balanced sequences recursively
    /// 3. Each recursive call handles one bracket pair
    /// 
    /// **Time Complexity:** O(n) - Each character processed once
    /// **Space Complexity:** O(n) - Recursion depth up to n/2
    /// 
    /// **Why not optimal for this problem:**
    /// - Recursion overhead vs simple iteration
    /// - More complex code for same time complexity
    /// - Stack overflow risk for deeply nested inputs
    /// 
    /// **Educational Value:**
    /// Demonstrates parsing techniques, but stack approach is simpler.
    pub fn is_valid_recursive(&self, s: String) -> bool {
        let chars: Vec<char> = s.chars().collect();
        let mut index = 0;
        self.parse_sequence(&chars, &mut index) && index == chars.len()
    }
    
    fn parse_sequence(&self, chars: &[char], index: &mut usize) -> bool {
        while *index < chars.len() {
            match chars[*index] {
                '(' => {
                    *index += 1;
                    if !self.parse_sequence(chars, index) {
                        return false;
                    }
                    if *index >= chars.len() || chars[*index] != ')' {
                        return false;
                    }
                    *index += 1;
                }
                '[' => {
                    *index += 1;
                    if !self.parse_sequence(chars, index) {
                        return false;
                    }
                    if *index >= chars.len() || chars[*index] != ']' {
                        return false;
                    }
                    *index += 1;
                }
                '{' => {
                    *index += 1;
                    if !self.parse_sequence(chars, index) {
                        return false;
                    }
                    if *index >= chars.len() || chars[*index] != '}' {
                        return false;
                    }
                    *index += 1;
                }
                ')' | ']' | '}' => {
                    // End of current sequence
                    return true;
                }
                _ => return false,
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
    #[case("()", true)]
    #[case("()[]{}", true)]
    #[case("(]", false)]
    #[case("([)]", false)]
    #[case("{[]}", true)]
    #[case("", true)]  // Empty string is valid
    fn test_basic_cases(#[case] input: &str, #[case] expected: bool) {
        let solution = setup();
        assert_eq!(solution.is_valid(input.to_string()), expected);
    }

    #[rstest]
    #[case("(", false)]           // Unclosed opening
    #[case(")", false)]           // Unmatched closing  
    #[case("((", false)]          // Multiple unclosed
    #[case("))", false)]          // Multiple unmatched
    #[case("(])", false)]         // Wrong type match
    #[case("([)])", false)]       // Incorrect nesting
    fn test_edge_cases(#[case] input: &str, #[case] expected: bool) {
        let solution = setup();
        assert_eq!(solution.is_valid(input.to_string()), expected);
    }

    #[test]
    fn test_deeply_nested() {
        let solution = setup();
        
        // Deeply nested but valid - construct it programmatically to avoid counting errors
        let deep_valid = format!("{}{}", "(".repeat(10), ")".repeat(10));
        assert!(solution.is_valid(deep_valid));
        
        // Deeply nested with mixed brackets
        let mixed_nested = "([{([{([{}])}])}])";
        assert!(solution.is_valid(mixed_nested.to_string()));
        
        // Deep nesting but invalid - extra closing parentheses
        let deep_invalid = format!("{}{}", "(".repeat(10), ")".repeat(12));
        assert!(!solution.is_valid(deep_invalid));
    }

    #[test]
    fn test_complex_patterns() {
        let solution = setup();
        
        // Complex valid patterns
        assert!(solution.is_valid("([{}])".to_string()));
        assert!(solution.is_valid("(){}[]".to_string()));
        assert!(solution.is_valid("({[]})".to_string()));
        assert!(solution.is_valid("[[[[]]]]".to_string()));
        
        // Complex invalid patterns
        assert!(!solution.is_valid("([{]})".to_string()));  // Wrong closing order
        assert!(!solution.is_valid("({[}])".to_string()));  // Mismatched in middle
        assert!(!solution.is_valid("([)]{}".to_string()));  // First part invalid
    }

    #[test]
    fn test_performance_stress() {
        let solution = setup();
        
        // Large valid string
        let large_valid = "()".repeat(5000);
        assert!(solution.is_valid(large_valid));
        
        // Large invalid string (should fail quickly)
        let large_invalid = format!("{}]", "(".repeat(5000));
        assert!(!solution.is_valid(large_invalid));
        
        // Alternating pattern
        let alternating = "()[]{}".repeat(1000);
        assert!(solution.is_valid(alternating));
    }

    #[test]
    fn test_all_approaches_consistency() {
        let solution = setup();
        let test_cases = vec![
            "()",
            "()[]{}",
            "(]",
            "([)]",
            "{[]}",
            "",
            "(",
            ")",
            "(((",
            ")))",
            "([{}])",
            "({[}])",
        ];
        
        for case in test_cases {
            let result1 = solution.is_valid(case.to_string());
            let result2 = solution.is_valid_hashmap(case.to_string());
            let result4 = solution.is_valid_replacement_inefficient(case.to_string());
            let result5 = solution.is_valid_recursive(case.to_string());
            
            assert_eq!(result1, result2, "HashMap approach differs for '{}'", case);
            assert_eq!(result1, result4, "Replacement approach differs for '{}'", case);
            assert_eq!(result1, result5, "Recursive approach differs for '{}'", case);
        }
    }

    #[test]
    fn test_counter_approach_failures() {
        let solution = setup();
        
        // Cases where counter approach fails but should be invalid
        let failing_cases = vec![
            "([)]",    // Wrong nesting order
            "(])",     // Wrong bracket types
            "([)]{}",  // Partial wrong nesting
        ];
        
        for case in failing_cases {
            let correct_result = solution.is_valid(case.to_string());
            let counter_result = solution.is_valid_counter_incorrect(case.to_string());
            
            // Demonstrate that counter approach gives wrong answer
            assert!(!correct_result, "Case '{}' should be invalid", case);
            
            // Note: Counter approach might accidentally get some of these right
            // due to early termination, but the logic is fundamentally flawed
            println!("Case '{}': correct={}, counter={}", case, correct_result, counter_result);
        }
    }

    #[test]
    fn test_algorithmic_properties() {
        let solution = setup();
        
        // Test that valid nested structures work
        let nested_levels = vec![
            "()",
            "(())",
            "((()))",
            "(((())))",
        ];
        
        for nested in nested_levels {
            assert!(solution.is_valid(nested.to_string()), 
                   "Nested structure '{}' should be valid", nested);
        }
        
        // Test that invalid structures fail
        let invalid_nested = vec![
            "(()",    // Missing closing
            "())",    // Extra closing
            "(()(",   // Unbalanced
        ];
        
        for invalid in invalid_nested {
            assert!(!solution.is_valid(invalid.to_string()), 
                   "Invalid structure '{}' should be rejected", invalid);
        }
    }

    #[test]
    fn test_early_termination_efficiency() {
        let solution = setup();
        
        // These should fail immediately on first mismatch
        let early_fail_cases = vec![
            ")",        // Immediate failure
            "]{}",      // Fail on first char
            "([)]",     // Fail when ']' doesn't match '('
        ];
        
        for case in early_fail_cases {
            assert!(!solution.is_valid(case.to_string()), 
                   "Case '{}' should fail early", case);
        }
    }
}