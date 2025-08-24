//! # Problem 772: Basic Calculator III
//!
//! Implement a basic calculator to evaluate a simple expression string.
//!
//! The expression string contains only non-negative integers, '+', '-', '*', '/' operators, 
//! open '(' and closing ')' parentheses and empty spaces ' '. The integer division should 
//! truncate toward zero.
//!
//! You may assume that the given expression is always valid. All intermediate results will 
//! be in the range of [-2^31, 2^31 - 1].
//!
//! ## Examples
//!
//! ```
//! Input: s = "1+1"
//! Output: 2
//! ```
//!
//! ```
//! Input: s = "6-4/2"
//! Output: 4
//! ```
//!
//! ```
//! Input: s = "2*(5+5*2)/3+(6/2+8)"
//! Output: 21
//! ```


/// Solution struct for Basic Calculator III problem
pub struct Solution;

impl Solution {
    /// Approach 1: Recursive Descent Parser with Operator Precedence
    ///
    /// Uses recursive descent parsing to handle operator precedence correctly.
    /// Parentheses are handled by recursive calls.
    ///
    /// Time Complexity: O(n) where n is the length of the expression
    /// Space Complexity: O(n) for recursion stack in worst case (nested parentheses)
    pub fn calculate_recursive(s: String) -> i32 {
        let chars: Vec<char> = s.chars().collect();
        let mut index = 0;
        Self::parse_expression(&chars, &mut index)
    }
    
    fn parse_expression(chars: &[char], index: &mut usize) -> i32 {
        let mut result = Self::parse_term(chars, index);
        
        while *index < chars.len() {
            Self::skip_whitespace(chars, index);
            if *index >= chars.len() || chars[*index] == ')' {
                break;
            }
            
            let op = chars[*index];
            if op == '+' || op == '-' {
                *index += 1;
                let term = Self::parse_term(chars, index);
                if op == '+' {
                    result += term;
                } else {
                    result -= term;
                }
            } else {
                break;
            }
        }
        
        result
    }
    
    fn parse_term(chars: &[char], index: &mut usize) -> i32 {
        let mut result = Self::parse_factor(chars, index);
        
        while *index < chars.len() {
            Self::skip_whitespace(chars, index);
            if *index >= chars.len() || chars[*index] == ')' || chars[*index] == '+' || chars[*index] == '-' {
                break;
            }
            
            let op = chars[*index];
            if op == '*' || op == '/' {
                *index += 1;
                let factor = Self::parse_factor(chars, index);
                if op == '*' {
                    result *= factor;
                } else {
                    result /= factor;
                }
            } else {
                break;
            }
        }
        
        result
    }
    
    fn parse_factor(chars: &[char], index: &mut usize) -> i32 {
        Self::skip_whitespace(chars, index);
        
        if chars[*index] == '(' {
            *index += 1; // skip '('
            let result = Self::parse_expression(chars, index);
            Self::skip_whitespace(chars, index);
            *index += 1; // skip ')'
            result
        } else {
            Self::parse_number(chars, index)
        }
    }
    
    fn parse_number(chars: &[char], index: &mut usize) -> i32 {
        Self::skip_whitespace(chars, index);
        let mut num = 0;
        
        while *index < chars.len() && chars[*index].is_ascii_digit() {
            num = num * 10 + (chars[*index] as i32 - '0' as i32);
            *index += 1;
        }
        
        num
    }
    
    fn skip_whitespace(chars: &[char], index: &mut usize) {
        while *index < chars.len() && chars[*index] == ' ' {
            *index += 1;
        }
    }
    
    /// Approach 2: Stack-based Evaluation with Operator Precedence
    ///
    /// Uses two stacks: one for operands and one for operators.
    /// Handles precedence by evaluating higher precedence operations first.
    ///
    /// Time Complexity: O(n) where n is the length of the expression
    /// Space Complexity: O(n) for the stacks
    pub fn calculate_stack(s: String) -> i32 {
        let mut operands = Vec::new();
        let mut operators = Vec::new();
        let chars: Vec<char> = s.chars().collect();
        let mut i = 0;
        
        while i < chars.len() {
            if chars[i] == ' ' {
                i += 1;
                continue;
            }
            
            if chars[i].is_ascii_digit() {
                let mut num = 0;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    num = num * 10 + (chars[i] as i32 - '0' as i32);
                    i += 1;
                }
                operands.push(num);
            } else if chars[i] == '(' {
                operators.push(chars[i]);
                i += 1;
            } else if chars[i] == ')' {
                while let Some(op) = operators.pop() {
                    if op == '(' {
                        break;
                    }
                    Self::apply_operator(&mut operands, op);
                }
                i += 1;
            } else if Self::is_operator(chars[i]) {
                while !operators.is_empty() && Self::precedence(operators[operators.len() - 1]) >= Self::precedence(chars[i]) {
                    Self::apply_operator(&mut operands, operators.pop().unwrap());
                }
                operators.push(chars[i]);
                i += 1;
            }
        }
        
        while !operators.is_empty() {
            Self::apply_operator(&mut operands, operators.pop().unwrap());
        }
        
        operands[0]
    }
    
    fn is_operator(c: char) -> bool {
        c == '+' || c == '-' || c == '*' || c == '/'
    }
    
    fn precedence(op: char) -> i32 {
        match op {
            '+' | '-' => 1,
            '*' | '/' => 2,
            _ => 0,
        }
    }
    
    fn apply_operator(operands: &mut Vec<i32>, op: char) {
        let b = operands.pop().unwrap();
        let a = operands.pop().unwrap();
        
        let result = match op {
            '+' => a + b,
            '-' => a - b,
            '*' => a * b,
            '/' => a / b,
            _ => 0,
        };
        
        operands.push(result);
    }
    
    /// Approach 3: Shunting Yard Algorithm (Dijkstra's Algorithm)
    ///
    /// Converts infix notation to postfix (RPN) then evaluates.
    /// Classic algorithm for parsing mathematical expressions.
    ///
    /// Time Complexity: O(n) for conversion + O(n) for evaluation = O(n)
    /// Space Complexity: O(n) for output queue and operator stack
    pub fn calculate_shunting_yard(s: String) -> i32 {
        let postfix = Self::infix_to_postfix(s);
        Self::evaluate_postfix(postfix)
    }
    
    fn infix_to_postfix(s: String) -> Vec<String> {
        let mut output = Vec::new();
        let mut operators = Vec::new();
        let chars: Vec<char> = s.chars().collect();
        let mut i = 0;
        
        while i < chars.len() {
            if chars[i] == ' ' {
                i += 1;
                continue;
            }
            
            if chars[i].is_ascii_digit() {
                let mut num = String::new();
                while i < chars.len() && chars[i].is_ascii_digit() {
                    num.push(chars[i]);
                    i += 1;
                }
                output.push(num);
            } else if chars[i] == '(' {
                operators.push(chars[i]);
                i += 1;
            } else if chars[i] == ')' {
                while let Some(op) = operators.pop() {
                    if op == '(' {
                        break;
                    }
                    output.push(op.to_string());
                }
                i += 1;
            } else if Self::is_operator(chars[i]) {
                while !operators.is_empty() && 
                      operators[operators.len() - 1] != '(' &&
                      Self::precedence(operators[operators.len() - 1]) >= Self::precedence(chars[i]) {
                    output.push(operators.pop().unwrap().to_string());
                }
                operators.push(chars[i]);
                i += 1;
            }
        }
        
        while !operators.is_empty() {
            output.push(operators.pop().unwrap().to_string());
        }
        
        output
    }
    
    fn evaluate_postfix(postfix: Vec<String>) -> i32 {
        let mut stack = Vec::new();
        
        for token in postfix {
            if token.chars().all(|c| c.is_ascii_digit()) {
                stack.push(token.parse::<i32>().unwrap());
            } else {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                
                let result = match token.as_str() {
                    "+" => a + b,
                    "-" => a - b,
                    "*" => a * b,
                    "/" => a / b,
                    _ => 0,
                };
                
                stack.push(result);
            }
        }
        
        stack[0]
    }
    
    /// Approach 4: Single Pass with Immediate Calculation
    ///
    /// Processes the expression in a single pass, maintaining running totals
    /// for different precedence levels.
    ///
    /// Time Complexity: O(n) where n is the length of the expression
    /// Space Complexity: O(n) for recursion stack in worst case
    pub fn calculate_single_pass(s: String) -> i32 {
        let chars: Vec<char> = s.chars().collect();
        let mut index = 0;
        Self::evaluate(&chars, &mut index)
    }
    
    fn evaluate(chars: &[char], index: &mut usize) -> i32 {
        let mut stack = Vec::new();
        let mut num = 0;
        let mut sign = '+';
        
        while *index < chars.len() {
            let c = chars[*index];
            
            if c.is_ascii_digit() {
                num = num * 10 + (c as i32 - '0' as i32);
            }
            
            if c == '(' {
                *index += 1;
                num = Self::evaluate(chars, index);
            }
            
            if Self::is_operator(c) || c == ')' || *index == chars.len() - 1 {
                match sign {
                    '+' => stack.push(num),
                    '-' => stack.push(-num),
                    '*' => {
                        let prev = stack.pop().unwrap();
                        stack.push(prev * num);
                    }
                    '/' => {
                        let prev = stack.pop().unwrap();
                        stack.push(prev / num);
                    }
                    _ => {}
                }
                
                if c == ')' {
                    break;
                }
                
                sign = c;
                num = 0;
            }
            
            *index += 1;
        }
        
        stack.iter().sum()
    }
    
    /// Approach 5: Expression Tree Building
    ///
    /// Builds an expression tree then evaluates it recursively.
    /// Good for understanding the structure of the expression.
    ///
    /// Time Complexity: O(n) for building + O(n) for evaluation = O(n)
    /// Space Complexity: O(n) for the expression tree
    pub fn calculate_expression_tree(s: String) -> i32 {
        let chars: Vec<char> = s.chars().collect();
        let mut index = 0;
        let tree = Self::build_expression_tree(&chars, &mut index);
        Self::evaluate_tree(&tree)
    }
    
    fn build_expression_tree(chars: &[char], index: &mut usize) -> ExprNode {
        let mut left = Self::build_term_tree(chars, index);
        
        while *index < chars.len() {
            Self::skip_whitespace(chars, index);
            if *index >= chars.len() || chars[*index] == ')' {
                break;
            }
            
            let op = chars[*index];
            if op == '+' || op == '-' {
                *index += 1;
                let right = Self::build_term_tree(chars, index);
                left = ExprNode::Operator {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        
        left
    }
    
    fn build_term_tree(chars: &[char], index: &mut usize) -> ExprNode {
        let mut left = Self::build_factor_tree(chars, index);
        
        while *index < chars.len() {
            Self::skip_whitespace(chars, index);
            if *index >= chars.len() || chars[*index] == ')' || chars[*index] == '+' || chars[*index] == '-' {
                break;
            }
            
            let op = chars[*index];
            if op == '*' || op == '/' {
                *index += 1;
                let right = Self::build_factor_tree(chars, index);
                left = ExprNode::Operator {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        
        left
    }
    
    fn build_factor_tree(chars: &[char], index: &mut usize) -> ExprNode {
        Self::skip_whitespace(chars, index);
        
        if chars[*index] == '(' {
            *index += 1; // skip '('
            let expr = Self::build_expression_tree(chars, index);
            Self::skip_whitespace(chars, index);
            *index += 1; // skip ')'
            expr
        } else {
            ExprNode::Number(Self::parse_number(chars, index))
        }
    }
    
    fn evaluate_tree(node: &ExprNode) -> i32 {
        match node {
            ExprNode::Number(n) => *n,
            ExprNode::Operator { op, left, right } => {
                let left_val = Self::evaluate_tree(left);
                let right_val = Self::evaluate_tree(right);
                
                match op {
                    '+' => left_val + right_val,
                    '-' => left_val - right_val,
                    '*' => left_val * right_val,
                    '/' => left_val / right_val,
                    _ => 0,
                }
            }
        }
    }
    
    /// Approach 6: Token-based Parser with State Machine
    ///
    /// Tokenizes the input first, then uses a state machine to parse.
    /// Clean separation of lexical analysis and parsing.
    ///
    /// Time Complexity: O(n) for tokenization + O(n) for parsing = O(n)
    /// Space Complexity: O(n) for token storage
    pub fn calculate_tokenized(s: String) -> i32 {
        let tokens = Self::tokenize(s);
        let mut index = 0;
        Self::parse_tokens(&tokens, &mut index)
    }
    
    fn tokenize(s: String) -> Vec<Token> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = s.chars().collect();
        let mut i = 0;
        
        while i < chars.len() {
            match chars[i] {
                ' ' => i += 1,
                '(' => {
                    tokens.push(Token::LeftParen);
                    i += 1;
                }
                ')' => {
                    tokens.push(Token::RightParen);
                    i += 1;
                }
                '+' | '-' | '*' | '/' => {
                    tokens.push(Token::Operator(chars[i]));
                    i += 1;
                }
                c if c.is_ascii_digit() => {
                    let mut num = 0;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        num = num * 10 + (chars[i] as i32 - '0' as i32);
                        i += 1;
                    }
                    tokens.push(Token::Number(num));
                }
                _ => i += 1,
            }
        }
        
        tokens
    }
    
    fn parse_tokens(tokens: &[Token], index: &mut usize) -> i32 {
        let mut result = Self::parse_token_term(tokens, index);
        
        while *index < tokens.len() {
            if let Token::Operator(op) = tokens[*index] {
                if op == '+' || op == '-' {
                    *index += 1;
                    let term = Self::parse_token_term(tokens, index);
                    if op == '+' {
                        result += term;
                    } else {
                        result -= term;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        
        result
    }
    
    fn parse_token_term(tokens: &[Token], index: &mut usize) -> i32 {
        let mut result = Self::parse_token_factor(tokens, index);
        
        while *index < tokens.len() {
            if let Token::Operator(op) = tokens[*index] {
                if op == '*' || op == '/' {
                    *index += 1;
                    let factor = Self::parse_token_factor(tokens, index);
                    if op == '*' {
                        result *= factor;
                    } else {
                        result /= factor;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        
        result
    }
    
    fn parse_token_factor(tokens: &[Token], index: &mut usize) -> i32 {
        match &tokens[*index] {
            Token::Number(n) => {
                *index += 1;
                *n
            }
            Token::LeftParen => {
                *index += 1; // skip '('
                let result = Self::parse_tokens(tokens, index);
                *index += 1; // skip ')'
                result
            }
            _ => 0,
        }
    }
}

#[derive(Debug)]
enum ExprNode {
    Number(i32),
    Operator {
        op: char,
        left: Box<ExprNode>,
        right: Box<ExprNode>,
    },
}

#[derive(Debug, Clone)]
enum Token {
    Number(i32),
    Operator(char),
    LeftParen,
    RightParen,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_addition() {
        let s = "1+1".to_string();
        let expected = 2;
        
        assert_eq!(Solution::calculate_recursive(s.clone()), expected);
        assert_eq!(Solution::calculate_stack(s.clone()), expected);
        assert_eq!(Solution::calculate_shunting_yard(s.clone()), expected);
        assert_eq!(Solution::calculate_single_pass(s.clone()), expected);
        assert_eq!(Solution::calculate_expression_tree(s.clone()), expected);
        assert_eq!(Solution::calculate_tokenized(s), expected);
    }

    #[test]
    fn test_subtraction_with_division() {
        let s = "6-4/2".to_string();
        let expected = 4;
        
        assert_eq!(Solution::calculate_recursive(s.clone()), expected);
        assert_eq!(Solution::calculate_stack(s.clone()), expected);
        assert_eq!(Solution::calculate_shunting_yard(s.clone()), expected);
        assert_eq!(Solution::calculate_single_pass(s.clone()), expected);
        assert_eq!(Solution::calculate_expression_tree(s.clone()), expected);
        assert_eq!(Solution::calculate_tokenized(s), expected);
    }

    #[test]
    fn test_complex_expression() {
        let s = "2*(5+5*2)/3+(6/2+8)".to_string();
        let expected = 21;
        
        assert_eq!(Solution::calculate_recursive(s.clone()), expected);
        assert_eq!(Solution::calculate_stack(s.clone()), expected);
        assert_eq!(Solution::calculate_shunting_yard(s.clone()), expected);
        assert_eq!(Solution::calculate_single_pass(s.clone()), expected);
        assert_eq!(Solution::calculate_expression_tree(s.clone()), expected);
        assert_eq!(Solution::calculate_tokenized(s), expected);
    }

    #[test]
    fn test_single_number() {
        let s = "42".to_string();
        let expected = 42;
        
        assert_eq!(Solution::calculate_recursive(s.clone()), expected);
        assert_eq!(Solution::calculate_stack(s.clone()), expected);
        assert_eq!(Solution::calculate_shunting_yard(s.clone()), expected);
        assert_eq!(Solution::calculate_single_pass(s.clone()), expected);
        assert_eq!(Solution::calculate_expression_tree(s.clone()), expected);
        assert_eq!(Solution::calculate_tokenized(s), expected);
    }

    #[test]
    fn test_parentheses_priority() {
        let s = "(1+2)*3".to_string();
        let expected = 9;
        
        assert_eq!(Solution::calculate_recursive(s.clone()), expected);
        assert_eq!(Solution::calculate_stack(s.clone()), expected);
        assert_eq!(Solution::calculate_shunting_yard(s.clone()), expected);
        assert_eq!(Solution::calculate_single_pass(s.clone()), expected);
        assert_eq!(Solution::calculate_expression_tree(s.clone()), expected);
        assert_eq!(Solution::calculate_tokenized(s), expected);
    }

    #[test]
    fn test_nested_parentheses() {
        let s = "((1+2)*3)/3".to_string();
        let expected = 3;
        
        assert_eq!(Solution::calculate_recursive(s.clone()), expected);
        assert_eq!(Solution::calculate_stack(s.clone()), expected);
        assert_eq!(Solution::calculate_shunting_yard(s.clone()), expected);
        assert_eq!(Solution::calculate_single_pass(s.clone()), expected);
        assert_eq!(Solution::calculate_expression_tree(s.clone()), expected);
        assert_eq!(Solution::calculate_tokenized(s), expected);
    }

    #[test]
    fn test_multiplication_precedence() {
        let s = "2+3*4".to_string();
        let expected = 14;
        
        assert_eq!(Solution::calculate_recursive(s.clone()), expected);
        assert_eq!(Solution::calculate_stack(s.clone()), expected);
        assert_eq!(Solution::calculate_shunting_yard(s.clone()), expected);
        assert_eq!(Solution::calculate_single_pass(s.clone()), expected);
        assert_eq!(Solution::calculate_expression_tree(s.clone()), expected);
        assert_eq!(Solution::calculate_tokenized(s), expected);
    }

    #[test]
    fn test_division_truncation() {
        let s = "7/3".to_string();
        let expected = 2;
        
        assert_eq!(Solution::calculate_recursive(s.clone()), expected);
        assert_eq!(Solution::calculate_stack(s.clone()), expected);
        assert_eq!(Solution::calculate_shunting_yard(s.clone()), expected);
        assert_eq!(Solution::calculate_single_pass(s.clone()), expected);
        assert_eq!(Solution::calculate_expression_tree(s.clone()), expected);
        assert_eq!(Solution::calculate_tokenized(s), expected);
    }

    #[test]
    fn test_negative_division() {
        let s = "14-3*2".to_string();
        let expected = 8;
        
        assert_eq!(Solution::calculate_recursive(s.clone()), expected);
        assert_eq!(Solution::calculate_stack(s.clone()), expected);
        assert_eq!(Solution::calculate_shunting_yard(s.clone()), expected);
        assert_eq!(Solution::calculate_single_pass(s.clone()), expected);
        assert_eq!(Solution::calculate_expression_tree(s.clone()), expected);
        assert_eq!(Solution::calculate_tokenized(s), expected);
    }

    #[test]
    fn test_spaces_in_expression() {
        let s = " 2 + 3 * 4 ".to_string();
        let expected = 14;
        
        assert_eq!(Solution::calculate_recursive(s.clone()), expected);
        assert_eq!(Solution::calculate_stack(s.clone()), expected);
        assert_eq!(Solution::calculate_shunting_yard(s.clone()), expected);
        assert_eq!(Solution::calculate_single_pass(s.clone()), expected);
        assert_eq!(Solution::calculate_expression_tree(s.clone()), expected);
        assert_eq!(Solution::calculate_tokenized(s), expected);
    }

    #[test]
    fn test_multiple_operations() {
        let s = "1*2-3/4+5*6-7*8+9/10".to_string();
        let expected = 1*2 - 3/4 + 5*6 - 7*8 + 9/10; // = 2 - 0 + 30 - 56 + 0 = -24
        
        assert_eq!(Solution::calculate_recursive(s.clone()), expected);
        assert_eq!(Solution::calculate_stack(s.clone()), expected);
        assert_eq!(Solution::calculate_shunting_yard(s.clone()), expected);
        assert_eq!(Solution::calculate_single_pass(s.clone()), expected);
        assert_eq!(Solution::calculate_expression_tree(s.clone()), expected);
        assert_eq!(Solution::calculate_tokenized(s), expected);
    }

    #[test]
    fn test_complex_nested() {
        let s = "(2+6*3+5-(3*14/7+2)*5)+3".to_string();
        let expected = (2 + 6*3 + 5 - (3*14/7 + 2)*5) + 3; // = (2 + 18 + 5 - (6 + 2)*5) + 3 = (25 - 40) + 3 = -12
        
        assert_eq!(Solution::calculate_recursive(s.clone()), expected);
        assert_eq!(Solution::calculate_stack(s.clone()), expected);
        assert_eq!(Solution::calculate_shunting_yard(s.clone()), expected);
        assert_eq!(Solution::calculate_single_pass(s.clone()), expected);
        assert_eq!(Solution::calculate_expression_tree(s.clone()), expected);
        assert_eq!(Solution::calculate_tokenized(s), expected);
    }

    #[test]
    fn test_zero_operations() {
        let s = "0*5+2".to_string();
        let expected = 2;
        
        assert_eq!(Solution::calculate_recursive(s.clone()), expected);
        assert_eq!(Solution::calculate_stack(s.clone()), expected);
        assert_eq!(Solution::calculate_shunting_yard(s.clone()), expected);
        assert_eq!(Solution::calculate_single_pass(s.clone()), expected);
        assert_eq!(Solution::calculate_expression_tree(s.clone()), expected);
        assert_eq!(Solution::calculate_tokenized(s), expected);
    }

    #[test]
    fn test_large_numbers() {
        let s = "1000000+2000000*3".to_string();
        let expected = 1000000 + 2000000 * 3; // = 7000000
        
        assert_eq!(Solution::calculate_recursive(s.clone()), expected);
        assert_eq!(Solution::calculate_stack(s.clone()), expected);
        assert_eq!(Solution::calculate_shunting_yard(s.clone()), expected);
        assert_eq!(Solution::calculate_single_pass(s.clone()), expected);
        assert_eq!(Solution::calculate_expression_tree(s.clone()), expected);
        assert_eq!(Solution::calculate_tokenized(s), expected);
    }

    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            "1+1",
            "6-4/2",
            "2*(5+5*2)/3+(6/2+8)",
            "(1+2)*3",
            "2+3*4",
            "14-3*2",
            "7/3",
            "1*2-3/4+5*6-7*8+9/10",
            "(2+6*3+5-(3*14/7+2)*5)+3",
            "0*5+2",
            "1000000+2000000*3",
            "((1+2)*3)/3",
            " 2 + 3 * 4 ",
            "42",
        ];
        
        for expr in test_cases {
            let s = expr.to_string();
            
            let result1 = Solution::calculate_recursive(s.clone());
            let result2 = Solution::calculate_stack(s.clone());
            let result3 = Solution::calculate_shunting_yard(s.clone());
            let result4 = Solution::calculate_single_pass(s.clone());
            let result5 = Solution::calculate_expression_tree(s.clone());
            let result6 = Solution::calculate_tokenized(s.clone());
            
            assert_eq!(result1, result2, "Recursive vs Stack mismatch for '{}'", expr);
            assert_eq!(result2, result3, "Stack vs Shunting Yard mismatch for '{}'", expr);
            assert_eq!(result3, result4, "Shunting Yard vs Single Pass mismatch for '{}'", expr);
            assert_eq!(result4, result5, "Single Pass vs Expression Tree mismatch for '{}'", expr);
            assert_eq!(result5, result6, "Expression Tree vs Tokenized mismatch for '{}'", expr);
        }
    }
}

// Author: Marvin Tutt, Caia Tech
// Problem: LeetCode 772 - Basic Calculator III
// Approaches: Recursive descent parser, Stack-based evaluation, Shunting yard algorithm, 
//            Single pass calculation, Expression tree building, Token-based parser
// All approaches correctly handle operator precedence, parentheses, and integer division truncation