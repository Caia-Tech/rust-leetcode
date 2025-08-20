# Performance Benchmarking Guide

This project includes comprehensive benchmarks for all implemented LeetCode solutions using the [Criterion](https://crates.io/crates/criterion) framework.

## Running Benchmarks

### Basic Usage

```bash
# Run all benchmarks (may take several minutes)
cargo bench

# Run benchmarks for a specific problem
cargo bench two_sum
cargo bench three_sum
cargo bench trapping_rain_water

# Run with shorter measurement time (faster but less precise)
cargo bench -- --measurement-time 5
```

### Understanding Results

Benchmark output shows:
- **Time**: Average execution time per operation
- **Throughput**: Operations processed per second (when applicable)
- **Outliers**: Measurements that deviate significantly from the average

Example output:
```
two_sum/hash_map/100    time:   [1.9830 µs 1.9864 µs 1.9899 µs]
                        thrpt:  [50.253 Melem/s 50.343 Melem/s 50.429 Melem/s]
```

This means the hash map approach for 100 elements takes ~1.98 microseconds and processes ~50 million elements per second.

## Benchmark Categories

### 1. Two Sum (`benchmark_two_sum_approaches`)
- **Algorithms**: Brute force O(n²), Hash map O(n), Two-pass O(n)
- **Input Sizes**: 10, 100, 1,000 elements
- **Key Insight**: Hash map approach scales much better than brute force

### 2. Valid Parentheses (`benchmark_valid_parentheses`)
- **Algorithms**: Stack-based validation
- **Input Sizes**: 100, 1,000, 10,000 characters
- **Key Insight**: Linear time complexity with different string patterns

### 3. Longest Substring Without Repeating Characters (`benchmark_longest_substring`)
- **Algorithms**: Sliding window O(n), Hash set O(n), Brute force O(n³)
- **Input Patterns**: Unique chars, repeating patterns, mixed strings
- **Key Insight**: Sliding window consistently outperforms other approaches

### 4. Longest Palindromic Substring (`benchmark_longest_palindrome`)
- **Algorithms**: Expand around center O(n²), Manacher O(n), DP O(n²)
- **Input Patterns**: Short strings, medium strings, palindromic strings
- **Key Insight**: Manacher's algorithm provides linear time for all cases

### 5. 3Sum (`benchmark_three_sum`)
- **Algorithms**: Two pointers O(n²), Hash set O(n²), Brute force O(n³)
- **Input Sizes**: 50, 100, 200 elements
- **Key Insight**: Two pointers approach is most memory efficient

### 6. Median of Two Sorted Arrays (`benchmark_median_arrays`)
- **Algorithms**: Binary search O(log(min(m,n))), Merge O(m+n)
- **Input Sizes**: 10, 100, 1,000 elements per array
- **Key Insight**: Binary search maintains logarithmic complexity

### 7. Trapping Rain Water (`benchmark_trapping_rain_water`)
- **Algorithms**: Two pointers O(n), DP O(n), Stack O(n), Brute force O(n²)
- **Input Patterns**: Classic example, sawtooth, mountain patterns
- **Key Insight**: Two pointers is optimal in both time and space

### 8. Merge k Sorted Lists (`benchmark_merge_k_lists`)
- **Algorithms**: Divide & conquer O(n log k), Min-heap O(n log k), Sequential O(k*n)
- **Test Cases**: Various combinations of list count and size
- **Key Insight**: Divide & conquer and heap approaches scale similarly

## Performance Analysis Guidelines

### Comparing Approaches
1. **Time Complexity**: Check if real-world performance matches theoretical complexity
2. **Space Efficiency**: Consider memory usage patterns (not directly measured but observable)
3. **Input Size Scaling**: Look for performance cliffs as input grows
4. **Cache Effects**: Small inputs may show different patterns due to CPU cache

### Expected Performance Patterns

| Problem | Optimal Algorithm | Expected Complexity | Scaling Behavior |
|---------|------------------|-------------------|------------------|
| Two Sum | Hash Map | O(n) | Linear scaling |
| 3Sum | Two Pointers | O(n²) | Quadratic scaling |
| Median Arrays | Binary Search | O(log min(m,n)) | Logarithmic scaling |
| Rain Water | Two Pointers | O(n) | Linear scaling |
| Merge k Lists | Divide & Conquer | O(n log k) | Log-linear scaling |

### Benchmark Configuration

The benchmarks use:
- **Statistical rigor**: Multiple measurements with outlier detection
- **Warm-up**: JIT compilation and CPU cache warming
- **Throughput metrics**: Elements processed per second where applicable
- **Multiple input patterns**: Different data characteristics to test robustness

### Interpreting Results

- **Green flags**: Performance matches expected complexity
- **Red flags**: Unexpected performance cliffs or scaling issues
- **Investigation points**: Large variance in measurements or unusual patterns

## Advanced Usage

### Custom Benchmark Runs
```bash
# Save results to file
cargo bench > benchmark_results.txt

# Compare with baseline (after making changes)
cargo bench --save-baseline main
# ... make changes ...
cargo bench --baseline main
```

### Profiling Integration
The benchmark framework integrates well with profiling tools:
```bash
# Profile with perf (Linux)
cargo bench --no-run
perf record target/release/deps/solutions-* --bench --profile-time=1

# Profile with Instruments (macOS)
cargo bench --no-run
xcrun xctrace record --template 'CPU Profiler' --launch target/release/deps/solutions-*
```

## Contributing Benchmark Improvements

When adding new problems or optimizations:

1. **Add comprehensive benchmarks** for all approaches
2. **Use representative input patterns** that test different scenarios  
3. **Include multiple input sizes** to verify scaling behavior
4. **Document expected performance characteristics** in comments
5. **Verify results match theoretical complexity** analysis

This benchmarking suite helps ensure our implementations are not just correct, but performant across a wide range of inputs and use cases.