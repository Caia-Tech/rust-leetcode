# Rust LeetCode Solutions

A comprehensive collection of LeetCode solutions in Rust, featuring multiple algorithmic approaches, extensive testing, and performance benchmarking.

## Features

- **13 Problems Solved** across Easy, Medium, and Hard difficulties
- **Multiple Approaches** per problem with complexity analysis
- **280+ Unit Tests** ensuring correctness
- **Property-Based Testing** with proptest
- **Performance Benchmarking** with Criterion
- **Educational Focus** - detailed explanations of why certain approaches work

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/rust-leetcode.git
cd rust-leetcode

# Run all tests
cargo test

# Run benchmarks
cargo bench

# Run specific problem tests
cargo test two_sum
```

## Problems Implemented

### Easy (9)
- Two Sum (#1)
- Reverse Integer (#7)
- Palindrome Number (#9)
- Roman to Integer (#13)
- Longest Common Prefix (#14)
- Valid Parentheses (#20)
- Merge Two Sorted Lists (#21)
- Remove Duplicates from Sorted Array (#26)
- Best Time to Buy and Sell Stock (#121)

### Medium (6)
- Longest Substring Without Repeating Characters (#3)
- Longest Palindromic Substring (#5)
- Container With Most Water (#11)
- 3Sum (#15)
- LRU Cache (#146)

### Hard (3)
- Median of Two Sorted Arrays (#4)
- Merge k Sorted Lists (#23)
- Trapping Rain Water (#42)

## Project Structure

```
src/
├── easy/           # Easy difficulty problems
├── medium/         # Medium difficulty problems
├── hard/           # Hard difficulty problems
├── utils/          # Shared data structures (ListNode, TreeNode)
└── lib.rs          # Library root

tests/
├── integration_tests.rs  # Integration tests
└── property_tests.rs     # Property-based tests

benches/
└── solutions.rs    # Performance benchmarks
```

## Example Usage

```rust
use rust_leetcode::easy::two_sum::Solution;

let solution = Solution::new();
let result = solution.two_sum(vec![2, 7, 11, 15], 9);
assert_eq!(result, vec![0, 1]);
```

## Approach Philosophy

Each problem includes multiple solutions:
- **Optimal** - The best time/space complexity solution
- **Alternative** - Different approaches with trade-offs
- **Educational** - Suboptimal but instructive implementations

## Testing

```bash
# Run all tests with output
cargo test -- --nocapture

# Run property-based tests
cargo test --test property_tests

# Run with specific test filter
cargo test container_with_most_water
```

## Benchmarking

See [README_BENCHMARKS.md](README_BENCHMARKS.md) for detailed benchmarking guide.

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench two_sum
```

## Contributing

Contributions are welcome! Please ensure:
- Multiple algorithmic approaches when applicable
- Comprehensive test coverage
- Clear documentation with complexity analysis
- Benchmark additions for new problems

## License

MIT

## Author

Built with Rust 🦀 for learning and interview preparation.