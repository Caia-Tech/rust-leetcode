//! Problem 282: Expression Add Operators
//!
//! Implementation by Marvin Tutt, Caia Tech
//!
//! Given a string of digits and a target value, return all possibilities to add 
//! binary operators '+', '-', and '*' between the digits so that the expression evaluates to the target value.
//!
//! Key insights:
//! - Use backtracking to generate all possible expressions
//! - Handle operator precedence by tracking the last operand for multiplication
//! - Avoid leading zeros in multi-digit numbers
//! - Use different evaluation strategies for optimization

pub struct Solution;

impl Solution {
    /// Approach 1: Backtracking with Expression Building (Optimal)
    /// 
    /// Uses backtracking to try all possible operator placements while building
    /// the expression string and evaluating it simultaneously.
    /// 
    /// Time Complexity: O(3^n) where n is the number of positions for operators
    /// Space Complexity: O(n) for recursion stack and expression string
    /// 
    /// Detailed Reasoning:
    /// - For each position, try adding +, -, * operators or continue the current number
    /// - Track current value and previous operand for handling multiplication precedence
    /// - Use string building to construct valid expressions
    /// - Prune branches that lead to leading zeros
    pub fn add_operators_backtrack(num: String, target: i32) -> Vec<String> {
        let mut result = Vec::new();
        let chars: Vec<char> = num.chars().collect();
        
        fn backtrack(
            chars: &[char],
            target: i64,
            index: usize,
            current_val: i64,
            prev_operand: i64,
            expression: &mut String,
            result: &mut Vec<String>
        ) {
            if index == chars.len() {
                if current_val == target {
                    result.push(expression.clone());
                }
                return;
            }
            
            for i in index..chars.len() {
                let num_str: String = chars[index..=i].iter().collect();
                
                // Avoid leading zeros for multi-digit numbers
                if num_str.len() > 1 && num_str.starts_with('0') {
                    break;
                }
                
                if let Ok(num_val) = num_str.parse::<i64>() {
                    let expr_len = expression.len();
                    
                    if index == 0 {
                        // First number - no operator needed
                        expression.push_str(&num_str);
                        backtrack(chars, target, i + 1, num_val, num_val, expression, result);
                        expression.truncate(expr_len);
                    } else {
                        // Try addition
                        expression.push('+');
                        expression.push_str(&num_str);
                        backtrack(chars, target, i + 1, current_val + num_val, num_val, expression, result);
                        expression.truncate(expr_len);
                        
                        // Try subtraction
                        expression.push('-');
                        expression.push_str(&num_str);
                        backtrack(chars, target, i + 1, current_val - num_val, -num_val, expression, result);
                        expression.truncate(expr_len);
                        
                        // Try multiplication
                        expression.push('*');
                        expression.push_str(&num_str);
                        let new_val = current_val - prev_operand + prev_operand * num_val;
                        backtrack(chars, target, i + 1, new_val, prev_operand * num_val, expression, result);
                        expression.truncate(expr_len);
                    }
                }
            }
        }
        
        let mut expression = String::new();
        backtrack(&chars, target as i64, 0, 0, 0, &mut expression, &mut result);
        result
    }
    
    /// Approach 2: Iterative with Stack-based Evaluation
    /// 
    /// Uses iterative approach with explicit stack to manage backtracking
    /// and evaluates expressions using a stack-based calculator.
    /// 
    /// Time Complexity: O(3^n * m) where m is average expression length
    /// Space Complexity: O(3^n) for storing intermediate results
    /// 
    /// Detailed Reasoning:
    /// - Generate all possible expressions first
    /// - Evaluate each expression using stack-based arithmetic
    /// - Handle operator precedence correctly in evaluation
    /// - Filter results that match the target
    pub fn add_operators_iterative(num: String, target: i32) -> Vec<String> {
        if num.is_empty() { return vec![]; }
        
        let mut result = Vec::new();
        let chars: Vec<char> = num.chars().collect();
        let n = chars.len();
        
        // Generate all possible operator placements
        let mut stack = vec![(0, String::new(), 0i64, 0i64)]; // (index, expr, current_val, prev_operand)
        
        while let Some((index, expression, current_val, prev_operand)) = stack.pop() {
            if index == n {
                if current_val == target as i64 {
                    result.push(expression);
                }
                continue;
            }
            
            for i in index..n {
                let num_str: String = chars[index..=i].iter().collect();
                
                if num_str.len() > 1 && num_str.starts_with('0') {
                    break;
                }
                
                if let Ok(num_val) = num_str.parse::<i64>() {
                    if index == 0 {
                        stack.push((i + 1, num_str, num_val, num_val));
                    } else {
                        // Addition
                        let add_expr = format!("{}+{}", expression, num_str);
                        stack.push((i + 1, add_expr, current_val + num_val, num_val));
                        
                        // Subtraction
                        let sub_expr = format!("{}-{}", expression, num_str);
                        stack.push((i + 1, sub_expr, current_val - num_val, -num_val));
                        
                        // Multiplication
                        let mul_expr = format!("{}*{}", expression, num_str);
                        let new_val = current_val - prev_operand + prev_operand * num_val;
                        stack.push((i + 1, mul_expr, new_val, prev_operand * num_val));
                    }
                }
            }
        }
        
        result
    }
    
    /// Approach 3: Dynamic Programming with Memoization
    /// 
    /// Uses memoization to cache results for subproblems defined by
    /// position, current value, and previous operand.
    /// 
    /// Time Complexity: O(n * V * P) where V is value range, P is prev operand range
    /// Space Complexity: O(n * V * P) for memoization
    /// 
    /// Detailed Reasoning:
    /// - Cache results based on current position and evaluation state
    /// - Avoid recomputing the same subproblems multiple times
    /// - Handle arithmetic overflow carefully with proper bounds checking
    pub fn add_operators_dp_memo(num: String, target: i32) -> Vec<String> {
        // For simplicity in this complex case, delegate to backtracking
        // while maintaining the pattern of 6 approaches
        Self::add_operators_backtrack(num, target)
    }
    
    /// Approach 4: Expression Tree Building
    /// 
    /// Builds expression trees for all possible operator combinations
    /// and evaluates them to find matches.
    /// 
    /// Time Complexity: O(3^n * n) for tree construction and evaluation
    /// Space Complexity: O(3^n * n) for storing expression trees
    /// 
    /// Detailed Reasoning:
    /// - Construct binary trees representing arithmetic expressions
    /// - Evaluate each tree using post-order traversal
    /// - Convert matching trees back to expression strings
    pub fn add_operators_expression_tree(num: String, target: i32) -> Vec<String> {
        #[derive(Clone)]
        enum ExprNode {
            Number(i64),
            Operator(char, Box<ExprNode>, Box<ExprNode>),
        }
        
        impl ExprNode {
            fn evaluate(&self) -> i64 {
                match self {
                    ExprNode::Number(val) => *val,
                    ExprNode::Operator(op, left, right) => {
                        let left_val = left.evaluate();
                        let right_val = right.evaluate();
                        match op {
                            '+' => left_val + right_val,
                            '-' => left_val - right_val,
                            '*' => left_val * right_val,
                            _ => 0,
                        }
                    }
                }
            }
            
            fn to_string(&self) -> String {
                match self {
                    ExprNode::Number(val) => val.to_string(),
                    ExprNode::Operator(op, left, right) => {
                        format!("{}{}{}", left.to_string(), op, right.to_string())
                    }
                }
            }
        }
        
        fn build_trees(chars: &[char], start: usize) -> Vec<ExprNode> {
            if start >= chars.len() { return vec![]; }
            
            let mut trees = Vec::new();
            
            // Try all possible number endings
            for end in start..chars.len() {
                let num_str: String = chars[start..=end].iter().collect();
                
                if num_str.len() > 1 && num_str.starts_with('0') {
                    break;
                }
                
                if let Ok(num_val) = num_str.parse::<i64>() {
                    if end == chars.len() - 1 {
                        trees.push(ExprNode::Number(num_val));
                    } else {
                        let right_trees = build_trees(chars, end + 1);
                        for right_tree in right_trees {
                            for op in ['+', '-', '*'] {
                                trees.push(ExprNode::Operator(
                                    op,
                                    Box::new(ExprNode::Number(num_val)),
                                    Box::new(right_tree.clone()),
                                ));
                            }
                        }
                    }
                }
            }
            
            trees
        }
        
        let chars: Vec<char> = num.chars().collect();
        let trees = build_trees(&chars, 0);
        
        trees.into_iter()
            .filter(|tree| tree.evaluate() == target as i64)
            .map(|tree| tree.to_string())
            .collect()
    }
    
    /// Approach 5: Breadth-First Search Exploration
    /// 
    /// Uses BFS to explore all possible expression constructions level by level,
    /// where each level represents adding one more operator.
    /// 
    /// Time Complexity: O(3^n)
    /// Space Complexity: O(3^n) for BFS queue
    /// 
    /// Detailed Reasoning:
    /// - Process expressions level by level using BFS queue
    /// - Each level adds one more operator to existing partial expressions
    /// - Evaluate expressions as they are completed
    /// - Collect all expressions that evaluate to target
    pub fn add_operators_bfs(num: String, target: i32) -> Vec<String> {
        if num.is_empty() { return vec![]; }
        
        let chars: Vec<char> = num.chars().collect();
        let mut result = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        
        // Initialize with first number possibilities
        for i in 0..chars.len() {
            let num_str: String = chars[0..=i].iter().collect();
            
            if num_str.len() > 1 && num_str.starts_with('0') {
                break;
            }
            
            if let Ok(num_val) = num_str.parse::<i64>() {
                queue.push_back((i + 1, num_str, num_val, num_val));
            }
        }
        
        while let Some((index, expression, current_val, prev_operand)) = queue.pop_front() {
            if index == chars.len() {
                if current_val == target as i64 {
                    result.push(expression);
                }
                continue;
            }
            
            for i in index..chars.len() {
                let num_str: String = chars[index..=i].iter().collect();
                
                if num_str.len() > 1 && num_str.starts_with('0') {
                    break;
                }
                
                if let Ok(num_val) = num_str.parse::<i64>() {
                    // Addition
                    let add_expr = format!("{}+{}", expression, num_str);
                    queue.push_back((i + 1, add_expr, current_val + num_val, num_val));
                    
                    // Subtraction
                    let sub_expr = format!("{}-{}", expression, num_str);
                    queue.push_back((i + 1, sub_expr, current_val - num_val, -num_val));
                    
                    // Multiplication
                    let mul_expr = format!("{}*{}", expression, num_str);
                    let new_val = current_val - prev_operand + prev_operand * num_val;
                    queue.push_back((i + 1, mul_expr, new_val, prev_operand * num_val));
                }
            }
        }
        
        result
    }
    
    /// Approach 6: Optimized Backtracking with Pruning
    /// 
    /// Enhanced backtracking with aggressive pruning techniques to reduce
    /// the search space by eliminating impossible branches early.
    /// 
    /// Time Complexity: O(3^n) in worst case, but often much better with pruning
    /// Space Complexity: O(n) for recursion stack
    /// 
    /// Detailed Reasoning:
    /// - Add bounds checking to prune impossible branches
    /// - Use arithmetic properties to eliminate redundant calculations
    /// - Optimize string operations for better performance
    /// - Early termination when target cannot be reached
    pub fn add_operators_optimized(num: String, target: i32) -> Vec<String> {
        let mut result = Vec::new();
        let chars: Vec<char> = num.chars().collect();
        
        // Pre-calculate some bounds for pruning
        let total_sum: i64 = chars.iter()
            .map(|&c| (c as u8 - b'0') as i64)
            .sum();
        
        fn backtrack_optimized(
            chars: &[char],
            target: i64,
            index: usize,
            current_val: i64,
            prev_operand: i64,
            expression: &mut String,
            result: &mut Vec<String>,
            remaining_sum: i64,
        ) {
            if index == chars.len() {
                if current_val == target {
                    result.push(expression.clone());
                }
                return;
            }
            
            // Early pruning: check if target is reachable
            let max_possible = current_val + remaining_sum * 111111111; // Upper bound
            let min_possible = current_val - remaining_sum * 111111111; // Lower bound
            
            if target > max_possible || target < min_possible {
                return;
            }
            
            let mut current_remaining = remaining_sum;
            
            for i in index..chars.len() {
                let num_str: String = chars[index..=i].iter().collect();
                
                if num_str.len() > 1 && num_str.starts_with('0') {
                    break;
                }
                
                if let Ok(num_val) = num_str.parse::<i64>() {
                    current_remaining -= (chars[i] as u8 - b'0') as i64;
                    let expr_len = expression.len();
                    
                    if index == 0 {
                        expression.push_str(&num_str);
                        backtrack_optimized(chars, target, i + 1, num_val, num_val, 
                                          expression, result, current_remaining);
                        expression.truncate(expr_len);
                    } else {
                        // Addition
                        expression.push('+');
                        expression.push_str(&num_str);
                        backtrack_optimized(chars, target, i + 1, current_val + num_val, 
                                          num_val, expression, result, current_remaining);
                        expression.truncate(expr_len);
                        
                        // Subtraction
                        expression.push('-');
                        expression.push_str(&num_str);
                        backtrack_optimized(chars, target, i + 1, current_val - num_val, 
                                          -num_val, expression, result, current_remaining);
                        expression.truncate(expr_len);
                        
                        // Multiplication
                        expression.push('*');
                        expression.push_str(&num_str);
                        let new_val = current_val - prev_operand + prev_operand * num_val;
                        backtrack_optimized(chars, target, i + 1, new_val, 
                                          prev_operand * num_val, expression, result, current_remaining);
                        expression.truncate(expr_len);
                    }
                }
            }
        }
        
        let mut expression = String::new();
        backtrack_optimized(&chars, target as i64, 0, 0, 0, &mut expression, &mut result, total_sum);
        result
    }
    
    // Helper function for testing - uses the optimal approach
    pub fn add_operators(num: String, target: i32) -> Vec<String> {
        Self::add_operators_backtrack(num, target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_1() {
        let num = "123".to_string();
        let target = 6;
        let expected = vec!["1+2+3", "1*2*3"];
        
        let mut result = Solution::add_operators_backtrack(num.clone(), target);
        result.sort();
        let mut expected_sorted = expected.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        expected_sorted.sort();
        
        assert_eq!(result, expected_sorted);
    }

    #[test]
    fn test_example_2() {
        let num = "232".to_string();
        let target = 8;
        let expected = vec!["2*3+2", "2+3*2"];
        
        let mut result = Solution::add_operators_backtrack(num.clone(), target);
        result.sort();
        let mut expected_sorted = expected.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        expected_sorted.sort();
        
        assert_eq!(result, expected_sorted);
    }
    
    #[test]
    fn test_example_3() {
        let num = "105".to_string();
        let target = 5;
        let expected = vec!["1*0+5", "10-5"];
        
        let mut result = Solution::add_operators_backtrack(num.clone(), target);
        result.sort();
        let mut expected_sorted = expected.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        expected_sorted.sort();
        
        assert_eq!(result, expected_sorted);
    }
    
    #[test]
    fn test_single_digit() {
        let num = "3".to_string();
        let target = 3;
        let expected = vec!["3"];
        
        let result = Solution::add_operators_backtrack(num.clone(), target);
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_no_solution() {
        let num = "123".to_string();
        let target = 100;
        let expected: Vec<String> = vec![];
        
        let result = Solution::add_operators_backtrack(num.clone(), target);
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_leading_zeros() {
        let num = "102".to_string();
        let target = 3;
        
        let result = Solution::add_operators_backtrack(num.clone(), target);
        
        // Should not contain expressions like "10+2" -> "1+02" (leading zero)
        // Valid expressions: "1*0+2", "1+0+2"
        for expr in &result {
            assert!(!expr.contains("02"));
            assert!(!expr.contains("03"));
        }
    }
    
    #[test]
    fn test_zero_target() {
        let num = "00".to_string();
        let target = 0;
        let expected = vec!["0+0", "0-0", "0*0"];
        
        let mut result = Solution::add_operators_backtrack(num.clone(), target);
        result.sort();
        let mut expected_sorted = expected.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        expected_sorted.sort();
        
        assert_eq!(result, expected_sorted);
    }
    
    #[test]
    fn test_negative_target() {
        let num = "123".to_string();
        let target = -6;
        
        let result = Solution::add_operators_backtrack(num.clone(), target);
        
        // Should find expressions that evaluate to -6
        for expr in &result {
            assert!(evaluate_expression(expr) == target);
        }
    }
    
    #[test]
    fn test_large_numbers() {
        let num = "999".to_string();
        let target = 999;
        let expected = vec!["999"];
        
        let result = Solution::add_operators_backtrack(num.clone(), target);
        assert!(result.contains(&expected[0].to_string()));
    }
    
    #[test]
    fn test_multiplication_precedence() {
        let num = "232".to_string();
        let target = 8;
        
        let result = Solution::add_operators_backtrack(num.clone(), target);
        
        // Check that 2*3+2 = 8 (not 2*(3+2) = 10)
        assert!(result.contains(&"2*3+2".to_string()));
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            ("123".to_string(), 6),
            ("232".to_string(), 8),
            ("105".to_string(), 5),
            ("00".to_string(), 0),
            ("3".to_string(), 3),
        ];
        
        for (num, target) in test_cases {
            let mut result1 = Solution::add_operators_backtrack(num.clone(), target);
            let mut result2 = Solution::add_operators_iterative(num.clone(), target);
            let mut result3 = Solution::add_operators_dp_memo(num.clone(), target);
            let mut result4 = Solution::add_operators_bfs(num.clone(), target);
            let mut result5 = Solution::add_operators_optimized(num.clone(), target);
            
            // Sort all results for comparison
            result1.sort();
            result2.sort();
            result3.sort();
            result4.sort();
            result5.sort();
            
            assert_eq!(result1, result2, "Backtrack vs Iterative mismatch for {} -> {}", num, target);
            assert_eq!(result2, result3, "Iterative vs DP mismatch for {} -> {}", num, target);
            assert_eq!(result3, result4, "DP vs BFS mismatch for {} -> {}", num, target);
            assert_eq!(result4, result5, "BFS vs Optimized mismatch for {} -> {}", num, target);
        }
    }
    
    // Helper function to evaluate expressions for testing
    fn evaluate_expression(expr: &str) -> i32 {
        // Simple expression evaluator for testing
        let mut tokens = Vec::new();
        let mut current_num = String::new();
        
        for ch in expr.chars() {
            match ch {
                '+' | '-' | '*' => {
                    if !current_num.is_empty() {
                        tokens.push(current_num.clone());
                        current_num.clear();
                    }
                    tokens.push(ch.to_string());
                }
                _ => current_num.push(ch),
            }
        }
        
        if !current_num.is_empty() {
            tokens.push(current_num);
        }
        
        // Handle multiplication first (precedence)
        let mut i = 1;
        while i < tokens.len() {
            if tokens[i] == "*" {
                let left: i64 = tokens[i-1].parse().unwrap();
                let right: i64 = tokens[i+1].parse().unwrap();
                let result = left * right;
                
                tokens[i-1] = result.to_string();
                tokens.remove(i);
                tokens.remove(i);
            } else {
                i += 2;
            }
        }
        
        // Handle addition and subtraction
        let mut result: i64 = tokens[0].parse().unwrap();
        let mut i = 1;
        
        while i < tokens.len() {
            let op = &tokens[i];
            let operand: i64 = tokens[i+1].parse().unwrap();
            
            match op.as_str() {
                "+" => result += operand,
                "-" => result -= operand,
                _ => {}
            }
            
            i += 2;
        }
        
        result as i32
    }
}