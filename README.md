# Rust LeetCode Solutions

A curated collection of [LeetCode](https://leetcode.com/) solutions implemented in Rust.
Problems are organized by difficulty and each module exposes a `Solution` type with multiple algorithmic approaches and tests.

## Repository Layout

```
src/                # Problem implementations grouped by difficulty
├── easy/
├── medium/
└── hard/
src/utils/          # Shared data structures and helpers
examples/           # Small executable usage examples
tests/              # Integration and property tests
benches/            # Criterion benchmarks
```

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for progress and
[README_BENCHMARKS.md](README_BENCHMARKS.md) for benchmarking notes.

## Usage

Add the crate to your project and call solution methods:

```rust
use rust_leetcode::easy::two_sum::Solution;

fn main() {
    let solution = Solution::new();
    let result = solution.two_sum(vec![2, 7, 11, 15], 9);
    assert_eq!(result, vec![0, 1]);
}
```

## Development

Run tests with:

```bash
cargo test
```

Run benchmarks (see [README_BENCHMARKS.md](README_BENCHMARKS.md)):

```bash
cargo bench
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT

## Author

Marvin Tutt, Caia Tech

