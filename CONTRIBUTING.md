# Contributing to Rust LeetCode Solutions

Thank you for your interest in contributing! This guide will help you add new problems or improve existing solutions.

## Adding a New Problem

### 1. File Structure

Create a new file in the appropriate difficulty folder:
- `src/easy/problem_name.rs`
- `src/medium/problem_name.rs`
- `src/hard/problem_name.rs`

### 2. Solution Template

```rust
//! # Problem [NUMBER]: [TITLE]
//!
//! [Problem description]
//!
//! ## Examples
//!
//! ```
//! [Example input/output]
//! ```
//!
//! ## Constraints
//!
//! * [List constraints]

/// Solution for [Problem Name]
pub struct Solution;

impl Solution {
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: [Name] (Optimal/Alternative)
    /// 
    /// **Algorithm:**
    /// 1. [Step 1]
    /// 2. [Step 2]
    /// 
    /// **Time Complexity:** O(?)
    /// **Space Complexity:** O(?)
    /// 
    /// **Key Insight:** [Explain why this works]
    pub fn method_name(&self, input: Type) -> ReturnType {
        // Implementation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_cases() {
        // Test implementation
    }
}
```

### 3. Requirements

Each solution must include:

- [ ] Problem description with examples
- [ ] At least one optimal solution
- [ ] Time and space complexity analysis
- [ ] Clear explanation of the approach
- [ ] Comprehensive unit tests
- [ ] Edge case handling

Bonus points for:
- [ ] Multiple approach implementations
- [ ] Benchmark comparisons
- [ ] Property-based tests
- [ ] Visual explanations in comments

### 4. Testing

Ensure all tests pass:
```bash
cargo test problem_name
cargo clippy -- -D warnings
cargo fmt
```

### 5. Documentation

Add your problem to:
1. The module file (`src/{difficulty}/mod.rs`)
2. The README.md problems list
3. Update the problem count in README

## Code Style

- Use idiomatic Rust patterns
- Prefer iterators over loops when clearer
- Use descriptive variable names
- Add comments for complex logic
- Format with `cargo fmt`

## Commit Guidelines

- Use clear commit messages
- Reference the LeetCode problem number
- Example: `Add Problem #42: Trapping Rain Water with multiple approaches`

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b add-problem-123`
3. Commit your changes
4. Push to your fork
5. Create a Pull Request with:
   - Problem number and title
   - Brief description of approaches implemented
   - Test coverage confirmation

## Questions?

Feel free to open an issue for discussion before implementing a complex solution.

Thank you for contributing!