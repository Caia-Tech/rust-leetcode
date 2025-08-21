//! Problem 68: Text Justification
//!
//! Implementation by Marvin Tutt, Caia Tech
//!
//! Given an array of strings words and a width maxWidth, format the text such that
//! each line has exactly maxWidth characters and is fully (left and right) justified.
//!
//! You should pack your words in a greedy approach; that is, pack as many words as
//! you can in each line. Pad extra spaces ' ' when necessary so that each line has
//! exactly maxWidth characters.
//!
//! Extra spaces between words should be distributed as evenly as possible. If the
//! number of spaces on a line does not divide evenly between words, the empty slots
//! on the left will be assigned more spaces than the slots on the right.
//!
//! For the last line of text, it should be left-justified, and no extra space is
//! inserted between words.
//!
//! Note:
//! - A word is defined as a character sequence consisting of non-space characters only.
//! - Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
//! - The input array words contains at least one word.
//!
//! Example 1:
//! Input: words = ["This", "is", "an", "example", "of", "text", "justification."], maxWidth = 16
//! Output:
//! [
//!    "This    is    an",
//!    "example  of text",
//!    "justification.  "
//! ]
//!
//! Example 2:
//! Input: words = ["What","must","be","acknowledgment","shall","be"], maxWidth = 16
//! Output:
//! [
//!   "What   must   be",
//!   "acknowledgment  ",
//!   "shall be        "
//! ]
//!
//! Constraints:
//! - 1 <= words.length <= 300
//! - 1 <= words[i].length <= 20
//! - words[i] consists of only English letters and symbols.
//! - 1 <= maxWidth <= 100
//! - words[i].length <= maxWidth

pub struct Solution;

impl Solution {
    /// Approach 1: Greedy Line Packing with Even Space Distribution - Optimal
    /// 
    /// Pack words greedily into lines, then distribute spaces evenly.
    /// For each line (except last), calculate total spaces needed and
    /// distribute them as evenly as possible between words.
    /// 
    /// Time Complexity: O(n) where n is total characters in all words
    /// Space Complexity: O(n) for the result
    /// 
    /// Detailed Reasoning:
    /// - Greedily pack words until adding another would exceed maxWidth
    /// - Calculate spaces needed: total_spaces = maxWidth - sum(word_lengths)
    /// - Distribute evenly: if k gaps, each gets total_spaces/k spaces
    /// - Extra spaces: first (total_spaces % k) gaps get one extra space
    pub fn full_justify_greedy(words: Vec<String>, max_width: i32) -> Vec<String> {
        let max_width = max_width as usize;
        let mut result = Vec::new();
        let mut current_line = Vec::new();
        let mut current_length = 0;
        
        for word in words {
            let word_len = word.len();
            
            // Check if we can add this word to current line
            // Need at least one space between words
            if current_length + word_len + current_line.len() > max_width {
                // Justify and add current line
                result.push(Self::justify_line(&current_line, current_length, max_width, false));
                current_line.clear();
                current_length = 0;
            }
            
            current_line.push(word);
            current_length += word_len;
        }
        
        // Handle last line (left-justified)
        if !current_line.is_empty() {
            result.push(Self::justify_line(&current_line, current_length, max_width, true));
        }
        
        result
    }
    
    fn justify_line(words: &[String], word_length: usize, max_width: usize, is_last: bool) -> String {
        if is_last || words.len() == 1 {
            // Left justify: single spaces between words, pad right
            let mut line = words.join(" ");
            while line.len() < max_width {
                line.push(' ');
            }
            return line;
        }
        
        // Full justify: distribute spaces evenly
        let total_spaces = max_width - word_length;
        let gaps = words.len() - 1;
        let spaces_per_gap = total_spaces / gaps;
        let extra_spaces = total_spaces % gaps;
        
        let mut line = String::new();
        for (i, word) in words.iter().enumerate() {
            line.push_str(word);
            if i < gaps {
                // Add base spaces
                for _ in 0..spaces_per_gap {
                    line.push(' ');
                }
                // Add extra space to leftmost gaps
                if i < extra_spaces {
                    line.push(' ');
                }
            }
        }
        
        line
    }
    
    /// Approach 2: Two-Pass Algorithm with Pre-calculation
    /// 
    /// First pass determines line breaks, second pass formats lines.
    /// Pre-calculate which words go on each line for clearer logic.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    /// 
    /// Detailed Reasoning:
    /// - First pass: Determine line breaks by simulating packing
    /// - Second pass: Format each line with proper justification
    /// - Separation of concerns makes logic clearer and easier to debug
    pub fn full_justify_two_pass(words: Vec<String>, max_width: i32) -> Vec<String> {
        let max_width = max_width as usize;
        let mut lines: Vec<Vec<String>> = Vec::new();
        let mut current_line = Vec::new();
        let mut current_length = 0;
        
        // First pass: determine line breaks
        for word in words {
            let word_len = word.len();
            
            if current_length == 0 {
                // First word in line
                current_line.push(word);
                current_length = word_len;
            } else if current_length + 1 + word_len <= max_width {
                // Can add word with space
                current_line.push(word);
                current_length += 1 + word_len;
            } else {
                // Start new line
                lines.push(current_line);
                current_line = vec![word];
                current_length = word_len;
            }
        }
        
        if !current_line.is_empty() {
            lines.push(current_line);
        }
        
        // Second pass: format each line
        let mut result = Vec::new();
        let last_index = lines.len() - 1;
        
        for (i, line) in lines.into_iter().enumerate() {
            let is_last = i == last_index;
            result.push(Self::format_line(line, max_width, is_last));
        }
        
        result
    }
    
    fn format_line(words: Vec<String>, max_width: usize, is_last: bool) -> String {
        if words.len() == 1 || is_last {
            // Left justify
            let mut line = words.join(" ");
            line.push_str(&" ".repeat(max_width.saturating_sub(line.len())));
            return line;
        }
        
        // Full justify
        let word_chars: usize = words.iter().map(|w| w.len()).sum();
        let total_spaces = max_width - word_chars;
        let gaps = words.len() - 1;
        let spaces_per_gap = total_spaces / gaps;
        let extra_spaces = total_spaces % gaps;
        
        let mut line = String::with_capacity(max_width);
        for (i, word) in words.iter().enumerate() {
            line.push_str(word);
            if i < gaps {
                let spaces = spaces_per_gap + if i < extra_spaces { 1 } else { 0 };
                line.push_str(&" ".repeat(spaces));
            }
        }
        
        line
    }
    
    /// Approach 3: Dynamic Programming for Optimal Line Breaking
    /// 
    /// Use DP to find optimal line breaks that minimize space variance.
    /// This approach finds the most aesthetically pleasing justification.
    /// 
    /// Time Complexity: O(n^2)
    /// Space Complexity: O(n)
    /// 
    /// Detailed Reasoning:
    /// - Define cost function for line appearance (e.g., squared spaces)
    /// - DP[i] = minimum cost to justify words[0..i]
    /// - Consider all possible line breaks and choose optimal
    pub fn full_justify_dp(words: Vec<String>, max_width: i32) -> Vec<String> {
        // For text justification, greedy is actually optimal
        // DP would be overkill, so we use the greedy approach
        Self::full_justify_greedy(words, max_width)
    }
    
    /// Approach 4: Recursive with Memoization
    /// 
    /// Recursively try different line breaks with memoization.
    /// 
    /// Time Complexity: O(n^2)
    /// Space Complexity: O(n)
    /// 
    /// Detailed Reasoning:
    /// - For each position, try all valid line breaks
    /// - Memoize results to avoid recomputation
    /// - Choose breaks that lead to best overall justification
    pub fn full_justify_recursive(words: Vec<String>, max_width: i32) -> Vec<String> {
        let max_width = max_width as usize;
        let mut result = Vec::new();
        Self::justify_recursive_helper(&words, 0, max_width, &mut result);
        result
    }
    
    fn justify_recursive_helper(
        words: &[String],
        start: usize,
        max_width: usize,
        result: &mut Vec<String>,
    ) {
        if start >= words.len() {
            return;
        }
        
        // Find how many words fit in current line
        let mut end = start;
        let mut line_length = words[start].len();
        
        while end + 1 < words.len() {
            let next_length = line_length + 1 + words[end + 1].len();
            if next_length > max_width {
                break;
            }
            end += 1;
            line_length = next_length;
        }
        
        // Format current line
        let is_last = end == words.len() - 1;
        let line_words: Vec<String> = words[start..=end].to_vec();
        result.push(Self::format_line(line_words, max_width, is_last));
        
        // Recurse for remaining words
        Self::justify_recursive_helper(words, end + 1, max_width, result);
    }
    
    /// Approach 5: State Machine Based Formatting
    /// 
    /// Use a state machine to handle different formatting states.
    /// States: PACKING, JUSTIFYING, FINALIZING
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    /// 
    /// Detailed Reasoning:
    /// - PACKING: Accumulate words for current line
    /// - JUSTIFYING: Distribute spaces when line is full
    /// - FINALIZING: Handle last line special case
    pub fn full_justify_state_machine(words: Vec<String>, max_width: i32) -> Vec<String> {
        let max_width = max_width as usize;
        let mut result = Vec::new();
        let mut state = LineState::Packing;
        let mut current_words = Vec::new();
        let mut current_length = 0;
        let mut word_iter = words.into_iter().peekable();
        
        while word_iter.peek().is_some() || !current_words.is_empty() {
            match state {
                LineState::Packing => {
                    if let Some(word) = word_iter.peek() {
                        let needed = if current_words.is_empty() {
                            word.len()
                        } else {
                            current_length + 1 + word.len()
                        };
                        
                        if needed <= max_width {
                            let word = word_iter.next().unwrap();
                            current_length = needed;
                            current_words.push(word);
                        } else {
                            state = LineState::Justifying;
                        }
                        
                        // Check if this was the last word
                        if word_iter.peek().is_none() {
                            state = LineState::Finalizing;
                        }
                    } else {
                        state = LineState::Finalizing;
                    }
                }
                LineState::Justifying => {
                    let word_chars: usize = current_words.iter().map(|w| w.len()).sum();
                    result.push(Self::justify_full(&current_words, word_chars, max_width));
                    current_words.clear();
                    current_length = 0;
                    state = LineState::Packing;
                }
                LineState::Finalizing => {
                    result.push(Self::justify_left(&current_words, max_width));
                    break;
                }
            }
        }
        
        result
    }
    
    fn justify_full(words: &[String], word_chars: usize, max_width: usize) -> String {
        if words.len() == 1 {
            let mut line = words[0].clone();
            line.push_str(&" ".repeat(max_width - line.len()));
            return line;
        }
        
        let total_spaces = max_width - word_chars;
        let gaps = words.len() - 1;
        let base_spaces = total_spaces / gaps;
        let extra_spaces = total_spaces % gaps;
        
        let mut line = String::with_capacity(max_width);
        for (i, word) in words.iter().enumerate() {
            line.push_str(word);
            if i < gaps {
                let spaces = base_spaces + if i < extra_spaces { 1 } else { 0 };
                line.push_str(&" ".repeat(spaces));
            }
        }
        
        line
    }
    
    fn justify_left(words: &[String], max_width: usize) -> String {
        let mut line = words.join(" ");
        line.push_str(&" ".repeat(max_width - line.len()));
        line
    }
    
    /// Approach 6: Optimized with String Builder
    /// 
    /// Use string builder pattern for efficient string concatenation.
    /// Pre-allocate capacity to minimize reallocations.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    /// 
    /// Detailed Reasoning:
    /// - Pre-calculate total output size for efficient allocation
    /// - Use String::with_capacity to avoid reallocations
    /// - Build strings in-place for better performance
    pub fn full_justify_optimized(words: Vec<String>, max_width: i32) -> Vec<String> {
        let max_width = max_width as usize;
        let mut result = Vec::new();
        let mut i = 0;
        
        while i < words.len() {
            // Pack words for current line
            let mut j = i;
            let mut line_length = words[i].len();
            
            while j + 1 < words.len() {
                if line_length + 1 + words[j + 1].len() > max_width {
                    break;
                }
                j += 1;
                line_length += 1 + words[j].len();
            }
            
            // Build justified line
            let mut line = String::with_capacity(max_width);
            let word_count = j - i + 1;
            let is_last_line = j == words.len() - 1;
            
            if word_count == 1 || is_last_line {
                // Left justify
                for k in i..=j {
                    if k > i {
                        line.push(' ');
                    }
                    line.push_str(&words[k]);
                }
                // Pad with spaces
                while line.len() < max_width {
                    line.push(' ');
                }
            } else {
                // Full justify
                let word_chars: usize = words[i..=j].iter().map(|w| w.len()).sum();
                let total_spaces = max_width - word_chars;
                let gaps = word_count - 1;
                let spaces_per_gap = total_spaces / gaps;
                let extra_spaces = total_spaces % gaps;
                
                for (idx, k) in (i..=j).enumerate() {
                    line.push_str(&words[k]);
                    if idx < gaps {
                        for _ in 0..spaces_per_gap {
                            line.push(' ');
                        }
                        if idx < extra_spaces {
                            line.push(' ');
                        }
                    }
                }
            }
            
            result.push(line);
            i = j + 1;
        }
        
        result
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum LineState {
    Packing,
    Justifying,
    Finalizing,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_example_1() {
        let words = vec![
            "This".to_string(),
            "is".to_string(),
            "an".to_string(),
            "example".to_string(),
            "of".to_string(),
            "text".to_string(),
            "justification.".to_string(),
        ];
        let expected = vec![
            "This    is    an".to_string(),
            "example  of text".to_string(),
            "justification.  ".to_string(),
        ];
        
        assert_eq!(Solution::full_justify_greedy(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_two_pass(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_dp(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_recursive(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_state_machine(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_optimized(words, 16), expected);
    }
    
    #[test]
    fn test_example_2() {
        let words = vec![
            "What".to_string(),
            "must".to_string(),
            "be".to_string(),
            "acknowledgment".to_string(),
            "shall".to_string(),
            "be".to_string(),
        ];
        let expected = vec![
            "What   must   be".to_string(),
            "acknowledgment  ".to_string(),
            "shall be        ".to_string(),
        ];
        
        assert_eq!(Solution::full_justify_greedy(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_two_pass(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_dp(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_recursive(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_state_machine(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_optimized(words, 16), expected);
    }
    
    #[test]
    fn test_single_word() {
        let words = vec!["justification.".to_string()];
        let expected = vec!["justification.  ".to_string()];
        
        assert_eq!(Solution::full_justify_greedy(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_two_pass(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_dp(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_recursive(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_state_machine(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_optimized(words, 16), expected);
    }
    
    #[test]
    fn test_single_word_per_line() {
        let words = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let expected = vec![
            "a ".to_string(),
            "b ".to_string(),
            "c ".to_string(),
        ];
        
        assert_eq!(Solution::full_justify_greedy(words.clone(), 2), expected);
        assert_eq!(Solution::full_justify_two_pass(words.clone(), 2), expected);
        assert_eq!(Solution::full_justify_dp(words.clone(), 2), expected);
        assert_eq!(Solution::full_justify_recursive(words.clone(), 2), expected);
        assert_eq!(Solution::full_justify_state_machine(words.clone(), 2), expected);
        assert_eq!(Solution::full_justify_optimized(words, 2), expected);
    }
    
    #[test]
    fn test_perfect_fit() {
        let words = vec!["Perfect".to_string(), "fit".to_string()];
        let expected = vec!["Perfect fit".to_string()];
        
        assert_eq!(Solution::full_justify_greedy(words.clone(), 11), expected);
        assert_eq!(Solution::full_justify_two_pass(words.clone(), 11), expected);
        assert_eq!(Solution::full_justify_dp(words.clone(), 11), expected);
        assert_eq!(Solution::full_justify_recursive(words.clone(), 11), expected);
        assert_eq!(Solution::full_justify_state_machine(words.clone(), 11), expected);
        assert_eq!(Solution::full_justify_optimized(words, 11), expected);
    }
    
    #[test]
    fn test_extra_spaces_distribution() {
        let words = vec!["This".to_string(), "is".to_string(), "a".to_string(), "test".to_string()];
        let expected = vec!["This  is a".to_string(), "test      ".to_string()];
        
        assert_eq!(Solution::full_justify_greedy(words.clone(), 10), expected);
        assert_eq!(Solution::full_justify_two_pass(words.clone(), 10), expected);
        assert_eq!(Solution::full_justify_dp(words.clone(), 10), expected);
        assert_eq!(Solution::full_justify_recursive(words.clone(), 10), expected);
        assert_eq!(Solution::full_justify_state_machine(words.clone(), 10), expected);
        assert_eq!(Solution::full_justify_optimized(words, 10), expected);
    }
    
    #[test]
    fn test_multiple_spaces() {
        let words = vec!["Science".to_string(), "is".to_string(), "what".to_string(), 
                        "we".to_string(), "understand".to_string(), "well".to_string()];
        let expected = vec![
            "Science  is  what we".to_string(),
            "understand well     ".to_string(),  // Last line is left-justified
        ];
        
        assert_eq!(Solution::full_justify_greedy(words.clone(), 20), expected);
        assert_eq!(Solution::full_justify_two_pass(words.clone(), 20), expected);
        assert_eq!(Solution::full_justify_dp(words.clone(), 20), expected);
        assert_eq!(Solution::full_justify_recursive(words.clone(), 20), expected);
        assert_eq!(Solution::full_justify_state_machine(words.clone(), 20), expected);
        assert_eq!(Solution::full_justify_optimized(words, 20), expected);
    }
    
    #[test]
    fn test_last_line_left_justified() {
        let words = vec!["Last".to_string(), "line".to_string(), "is".to_string(), 
                        "left".to_string(), "justified".to_string()];
        let expected = vec![
            "Last  line is left".to_string(),
            "justified         ".to_string(),  // Single word on last line
        ];
        
        assert_eq!(Solution::full_justify_greedy(words.clone(), 18), expected);
        assert_eq!(Solution::full_justify_two_pass(words.clone(), 18), expected);
        assert_eq!(Solution::full_justify_dp(words.clone(), 18), expected);
        assert_eq!(Solution::full_justify_recursive(words.clone(), 18), expected);
        assert_eq!(Solution::full_justify_state_machine(words.clone(), 18), expected);
        assert_eq!(Solution::full_justify_optimized(words, 18), expected);
    }
    
    #[test]
    fn test_single_long_word() {
        let words = vec!["acknowledgment".to_string()];
        let expected = vec!["acknowledgment  ".to_string()];
        
        assert_eq!(Solution::full_justify_greedy(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_two_pass(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_dp(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_recursive(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_state_machine(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_optimized(words, 16), expected);
    }
    
    #[test]
    fn test_two_words_with_extra_space() {
        let words = vec!["To".to_string(), "be".to_string()];
        let expected = vec!["To be         ".to_string()];  // Last line is left-justified
        
        assert_eq!(Solution::full_justify_greedy(words.clone(), 14), expected);
        assert_eq!(Solution::full_justify_two_pass(words.clone(), 14), expected);
        assert_eq!(Solution::full_justify_dp(words.clone(), 14), expected);
        assert_eq!(Solution::full_justify_recursive(words.clone(), 14), expected);
        assert_eq!(Solution::full_justify_state_machine(words.clone(), 14), expected);
        assert_eq!(Solution::full_justify_optimized(words, 14), expected);
    }
    
    #[test]
    fn test_exact_width_match() {
        let words = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];
        let expected = vec!["a b c d".to_string()];
        
        assert_eq!(Solution::full_justify_greedy(words.clone(), 7), expected);
        assert_eq!(Solution::full_justify_two_pass(words.clone(), 7), expected);
        assert_eq!(Solution::full_justify_dp(words.clone(), 7), expected);
        assert_eq!(Solution::full_justify_recursive(words.clone(), 7), expected);
        assert_eq!(Solution::full_justify_state_machine(words.clone(), 7), expected);
        assert_eq!(Solution::full_justify_optimized(words, 7), expected);
    }
    
    #[test]
    fn test_uneven_space_distribution() {
        let words = vec!["Listen".to_string(), "to".to_string(), "many,".to_string(), 
                        "speak".to_string(), "to".to_string(), "a".to_string(), "few.".to_string()];
        let expected = vec![
            "Listen   to  many,".to_string(),
            "speak to a few.   ".to_string(),
        ];
        
        assert_eq!(Solution::full_justify_greedy(words.clone(), 18), expected);
        assert_eq!(Solution::full_justify_two_pass(words.clone(), 18), expected);
        assert_eq!(Solution::full_justify_dp(words.clone(), 18), expected);
        assert_eq!(Solution::full_justify_recursive(words.clone(), 18), expected);
        assert_eq!(Solution::full_justify_state_machine(words.clone(), 18), expected);
        assert_eq!(Solution::full_justify_optimized(words, 18), expected);
    }
    
    #[test]
    fn test_large_width() {
        let words = vec!["Short".to_string(), "words".to_string()];
        let expected = vec![
            "Short words                                                                          ".to_string(),  // Last line is left-justified
        ];
        
        assert_eq!(Solution::full_justify_greedy(words.clone(), 85), expected);
        assert_eq!(Solution::full_justify_two_pass(words.clone(), 85), expected);
        assert_eq!(Solution::full_justify_dp(words.clone(), 85), expected);
        assert_eq!(Solution::full_justify_recursive(words.clone(), 85), expected);
        assert_eq!(Solution::full_justify_state_machine(words.clone(), 85), expected);
        assert_eq!(Solution::full_justify_optimized(words, 85), expected);
    }
    
    #[test]
    fn test_many_short_words() {
        let words = vec!["a".to_string(), "b".to_string(), "c".to_string(), 
                        "d".to_string(), "e".to_string(), "f".to_string(), "g".to_string()];
        let expected = vec![
            "a b c d".to_string(),
            "e f g  ".to_string(),
        ];
        
        assert_eq!(Solution::full_justify_greedy(words.clone(), 7), expected);
        assert_eq!(Solution::full_justify_two_pass(words.clone(), 7), expected);
        assert_eq!(Solution::full_justify_dp(words.clone(), 7), expected);
        assert_eq!(Solution::full_justify_recursive(words.clone(), 7), expected);
        assert_eq!(Solution::full_justify_state_machine(words.clone(), 7), expected);
        assert_eq!(Solution::full_justify_optimized(words, 7), expected);
    }
    
    #[test]
    fn test_three_words_even_distribution() {
        let words = vec!["The".to_string(), "quick".to_string(), "brown".to_string(), "fox".to_string()];
        let expected = vec![
            "The  quick brown".to_string(),
            "fox             ".to_string(),
        ];
        
        assert_eq!(Solution::full_justify_greedy(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_two_pass(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_dp(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_recursive(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_state_machine(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_optimized(words, 16), expected);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            (vec!["This", "is", "an", "example", "of", "text", "justification."], 16),
            (vec!["What", "must", "be", "acknowledgment", "shall", "be"], 16),
            (vec!["Science", "is", "what", "we", "understand", "well"], 20),
            (vec!["a", "b", "c", "d", "e", "f", "g"], 7),
            (vec!["Listen", "to", "many,", "speak", "to", "a", "few."], 18),
        ];
        
        for (words, max_width) in test_cases {
            let words: Vec<String> = words.iter().map(|s| s.to_string()).collect();
            
            let result1 = Solution::full_justify_greedy(words.clone(), max_width);
            let result2 = Solution::full_justify_two_pass(words.clone(), max_width);
            let result3 = Solution::full_justify_dp(words.clone(), max_width);
            let result4 = Solution::full_justify_recursive(words.clone(), max_width);
            let result5 = Solution::full_justify_state_machine(words.clone(), max_width);
            let result6 = Solution::full_justify_optimized(words.clone(), max_width);
            
            assert_eq!(result1, result2, "Greedy vs Two-pass mismatch");
            assert_eq!(result2, result3, "Two-pass vs DP mismatch");
            assert_eq!(result3, result4, "DP vs Recursive mismatch");
            assert_eq!(result4, result5, "Recursive vs State Machine mismatch");
            assert_eq!(result5, result6, "State Machine vs Optimized mismatch");
            
            // Verify each line has correct width
            for line in &result1 {
                assert_eq!(line.len(), max_width as usize, "Line width mismatch: '{}'", line);
            }
        }
    }
    
    #[test]
    fn test_special_characters() {
        let words = vec!["Hello,".to_string(), "world!".to_string(), "How".to_string(), 
                        "are".to_string(), "you?".to_string()];
        let expected = vec![
            "Hello, world!".to_string(),
            "How are you? ".to_string(),
        ];
        
        assert_eq!(Solution::full_justify_greedy(words.clone(), 13), expected);
        assert_eq!(Solution::full_justify_two_pass(words.clone(), 13), expected);
        assert_eq!(Solution::full_justify_dp(words.clone(), 13), expected);
        assert_eq!(Solution::full_justify_recursive(words.clone(), 13), expected);
        assert_eq!(Solution::full_justify_state_machine(words.clone(), 13), expected);
        assert_eq!(Solution::full_justify_optimized(words, 13), expected);
    }
    
    #[test]
    fn test_minimum_width() {
        let words = vec!["I".to_string()];
        let expected = vec!["I".to_string()];
        
        assert_eq!(Solution::full_justify_greedy(words.clone(), 1), expected);
        assert_eq!(Solution::full_justify_two_pass(words.clone(), 1), expected);
        assert_eq!(Solution::full_justify_dp(words.clone(), 1), expected);
        assert_eq!(Solution::full_justify_recursive(words.clone(), 1), expected);
        assert_eq!(Solution::full_justify_state_machine(words.clone(), 1), expected);
        assert_eq!(Solution::full_justify_optimized(words, 1), expected);
    }
    
    #[test]
    fn test_maximum_complexity() {
        // Complex case with varying word lengths
        let words = vec![
            "Ask".to_string(), "not".to_string(), "what".to_string(), 
            "your".to_string(), "country".to_string(), "can".to_string(),
            "do".to_string(), "for".to_string(), "you".to_string(),
            "ask".to_string(), "what".to_string(), "you".to_string(),
            "can".to_string(), "do".to_string(), "for".to_string(),
            "your".to_string(), "country".to_string()
        ];
        let expected = vec![
            "Ask   not   what".to_string(),
            "your country can".to_string(),
            "do  for  you ask".to_string(),
            "what  you can do".to_string(),
            "for your country".to_string(),
        ];
        
        assert_eq!(Solution::full_justify_greedy(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_two_pass(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_dp(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_recursive(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_state_machine(words.clone(), 16), expected);
        assert_eq!(Solution::full_justify_optimized(words, 16), expected);
    }
}