# Algorithm Complexity Analysis Guide

## Overview
This guide provides detailed complexity analysis for all implemented solutions, helping you understand the trade-offs between different algorithmic approaches and choose the optimal solution for your use case.

## Complexity Analysis Framework

### Time Complexity Categories

#### **O(1) - Constant Time**
**Problems with O(1) solutions:**
- Cache operations (LRU Cache, LFU Cache)
- Array element access
- Hash table operations (average case)

**Implementation Examples:**
```rust
// Hash table lookup - O(1) average
fn get(&self, key: i32) -> i32 {
    *self.map.get(&key).unwrap_or(&-1)
}
```

#### **O(log n) - Logarithmic Time**  
**Common Patterns:**
- Binary search variants
- Heap operations
- Balanced tree operations

**Problems:**
```
Binary Search in Rotated Array:     O(log n)
Find Minimum in Rotated Array:      O(log n) 
Kth Largest Element (Heap):          O(log n) per operation
```

#### **O(n) - Linear Time**
**Most Common Category (35+ problems)**

**Pattern Classifications:**
1. **Single Pass Array Processing**
   - Two Sum (with hash map)
   - Maximum Subarray (Kadane's)
   - Product of Array Except Self

2. **Linear String Processing**
   - Longest Substring (sliding window)
   - Valid Parentheses (stack)
   - Roman to Integer

3. **Tree/Graph Traversal**
   - Maximum Depth of Binary Tree
   - Number of Islands (DFS/BFS)
   - Clone Graph

#### **O(n log n) - Linearithmic Time**
**Sorting-Based Solutions:**
```
3Sum:                    Sort + Two Pointers = O(n log n)
Merge Intervals:         Sort + Linear Merge = O(n log n)
Top K Frequent Elements: Heap operations = O(n log k)
```

#### **O(n²) - Quadratic Time**
**Dynamic Programming Patterns:**
```
Longest Palindromic Substring:  O(n²) space, O(n²) time
Edit Distance:                  O(m×n) DP table
Longest Common Subsequence:     O(m×n) comparisons
```

**Nested Loop Algorithms:**
```
3Sum (Brute Force):            O(n³) → O(n²) optimized
Container With Most Water:     O(n²) → O(n) two pointers
```

#### **O(2^n) - Exponential Time**
**Backtracking Problems:**
```
Generate Parentheses:          O(2^n) combinations
Subsets:                       O(2^n) power set
N-Queens:                      O(n!) with pruning
```

## Space Complexity Analysis

### **O(1) - Constant Space**
**In-Place Algorithms (25+ implementations):**

**Array Manipulation:**
```rust
// Rotate Array - O(1) space
fn rotate(&self, nums: &mut Vec<i32>, k: i32) {
    let n = nums.len();
    let k = k as usize % n;
    nums.reverse();
    nums[..k].reverse();
    nums[k..].reverse();
}
```

**Two Pointer Techniques:**
```rust
// Remove Duplicates - O(1) space
fn remove_duplicates(&self, nums: &mut Vec<i32>) -> i32 {
    let mut write_idx = 1;
    for read_idx in 1..nums.len() {
        if nums[read_idx] != nums[read_idx - 1] {
            nums[write_idx] = nums[read_idx];
            write_idx += 1;
        }
    }
    write_idx as i32
}
```

### **O(n) - Linear Space**
**Hash Table Solutions:**
- Two Sum: Store value→index mapping
- Group Anagrams: Store sorted_string→anagram_list mapping
- LRU Cache: Hash map + doubly linked list

**Auxiliary Arrays:**
```rust
// Product Except Self - O(n) auxiliary space
fn product_except_self(&self, nums: Vec<i32>) -> Vec<i32> {
    let n = nums.len();
    let mut result = vec![1; n];
    
    // Left products
    for i in 1..n {
        result[i] = result[i-1] * nums[i-1];
    }
    
    // Right products (in-place)
    let mut right = 1;
    for i in (0..n).rev() {
        result[i] *= right;
        right *= nums[i];
    }
    
    result
}
```

### **O(n²) - Quadratic Space**
**2D Dynamic Programming:**
```rust
// Edit Distance - O(m×n) DP table
fn min_distance(&self, word1: String, word2: String) -> i32 {
    let m = word1.len();
    let n = word2.len();
    let mut dp = vec![vec![0; n + 1]; m + 1];
    // ... DP logic
}
```

## Problem-Specific Complexity Analysis

### Easy Problems Complexity Summary

| Problem | Time Complexity | Space Complexity | Optimization Notes |
|---------|----------------|------------------|-------------------|
| **Two Sum** | O(n) | O(n) | Hash map trade-off: space for time |
| **Valid Parentheses** | O(n) | O(n) | Stack size bounded by string length |
| **Merge Two Lists** | O(m+n) | O(1) | In-place merge, optimal |
| **Climbing Stairs** | O(n) | O(1) | Space-optimized Fibonacci |
| **Maximum Depth** | O(n) | O(h) | Height-balanced trees: O(log n) |

### Medium Problems Complexity Deep Dive

#### **Dynamic Programming Problems**

**Pattern: 1D DP → O(1) Space Optimization**
```rust
// House Robber - Space optimized
fn rob(&self, nums: Vec<i32>) -> i32 {
    let (mut prev, mut curr) = (0, 0);
    for num in nums {
        let temp = curr;
        curr = curr.max(prev + num);
        prev = temp;
    }
    curr
}
```

**Complexity Evolution:**
- Naive Recursion: O(2^n) time, O(n) space
- Memoized Recursion: O(n) time, O(n) space  
- Bottom-up DP: O(n) time, O(n) space
- Space-optimized: O(n) time, O(1) space ✓

#### **String Processing Optimizations**

**Longest Substring Without Repeating Characters:**
```
Brute Force:     O(n³) time - Check all substrings
Hash Set:        O(n²) time - Sliding window with set operations  
Sliding Window:  O(n) time  - Optimal single pass
```

**Implementation Comparison:**
```rust
// O(n) Sliding Window - Optimal
fn length_of_longest_substring(&self, s: String) -> i32 {
    let chars: Vec<char> = s.chars().collect();
    let mut char_index = std::collections::HashMap::new();
    let mut max_len = 0;
    let mut start = 0;
    
    for (end, &ch) in chars.iter().enumerate() {
        if let Some(&idx) = char_index.get(&ch) {
            start = start.max(idx + 1);
        }
        char_index.insert(ch, end);
        max_len = max_len.max(end - start + 1);
    }
    
    max_len as i32
}
```

### Hard Problems Advanced Analysis

#### **String Algorithm Complexities**

**Edit Distance (Levenshtein Distance):**
```
Standard DP:        O(m×n) time, O(m×n) space
Space Optimized:    O(m×n) time, O(min(m,n)) space
Rolling Array:      O(m×n) time, O(n) space
```

**Implementation with Space Optimization:**
```rust
fn min_distance(&self, word1: String, word2: String) -> i32 {
    let (word1, word2) = if word1.len() < word2.len() { 
        (word2, word1) 
    } else { 
        (word1, word2) 
    };
    
    let (m, n) = (word1.len(), word2.len());
    let mut prev = (0..=n as i32).collect::<Vec<_>>();
    let mut curr = vec![0; n + 1];
    
    for (i, ch1) in word1.chars().enumerate() {
        curr[0] = (i + 1) as i32;
        for (j, ch2) in word2.chars().enumerate() {
            curr[j + 1] = if ch1 == ch2 {
                prev[j]
            } else {
                1 + prev[j].min(prev[j + 1]).min(curr[j])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    
    prev[n]
}
```

#### **Advanced Tree Algorithm Analysis**

**Morris Traversal - O(1) Space Tree Traversal:**
```
Traditional Inorder:  O(n) time, O(h) space (recursion/stack)
Morris Traversal:     O(n) time, O(1) space (threading)
```

**Complexity Trade-offs:**
- **Time**: Morris is 2-3x slower due to tree modification overhead
- **Space**: Constant space vs. O(h) stack space
- **Code Complexity**: Morris requires careful pointer manipulation

#### **System Design Problem Analysis**

**LRU Cache Implementation:**
```
Operations Required:  get(), put() both in O(1)
Data Structures:      HashMap + Doubly Linked List
Space Complexity:     O(capacity)
```

**Design Analysis:**
```rust
struct LRUCache {
    capacity: usize,
    map: HashMap<i32, Rc<RefCell<Node>>>,  // O(1) access
    head: Rc<RefCell<Node>>,               // Dummy head
    tail: Rc<RefCell<Node>>,               // Dummy tail
}

// O(1) operations through direct node manipulation
impl LRUCache {
    fn get(&mut self, key: i32) -> i32 {
        if let Some(node) = self.map.get(&key) {
            let value = node.borrow().value;
            self.move_to_head(node.clone());
            value
        } else {
            -1
        }
    }
}
```

## Optimization Strategies by Pattern

### **Array Problems**

**Pattern Recognition:**
1. **Sorted Array**: Binary search, two pointers
2. **Unsorted Array**: Hash table, sorting preprocessing
3. **Subarray Problems**: Sliding window, prefix sums
4. **In-Place Modification**: Two pointers, cyclic replacements

### **String Problems**

**Optimization Patterns:**
1. **Character Frequency**: Hash maps, arrays (for ASCII)
2. **Pattern Matching**: KMP, Rabin-Karp, Z-algorithm
3. **Palindromes**: Expand from center, Manacher's algorithm
4. **Subsequences**: Dynamic programming with space optimization

### **Tree Problems**

**Space-Time Trade-offs:**
1. **Recursive**: Clean code, O(h) space
2. **Iterative**: Stack-based, explicit control
3. **Morris**: O(1) space, complex implementation
4. **Level-order**: Queue-based, good for breadth-first needs

### **Graph Problems**

**Algorithm Selection:**
1. **Unweighted Shortest Path**: BFS - O(V+E)
2. **Weighted Shortest Path**: Dijkstra - O((V+E)logV)
3. **Connectivity**: Union-Find - O(α(V)) per operation
4. **Topological Sort**: DFS/Kahn's - O(V+E)

## Performance Measurement Integration

### **Benchmark Integration**
Each complexity analysis is validated through the benchmark suite:
```bash
cargo bench                    # Run all benchmarks
cargo bench two_sum           # Benchmark specific problem
cargo bench -- --save-baseline main  # Save performance baseline
```

### **Complexity Verification**
```rust
// Example benchmark validating O(n) scaling
fn benchmark_linear_scaling(c: &mut Criterion) {
    let sizes = [100, 1000, 10000];
    for size in sizes.iter() {
        let input = generate_input(*size);
        c.bench_function(&format!("algorithm_{}", size), |b| {
            b.iter(|| algorithm(black_box(input.clone())))
        });
    }
}
```

## Recommendations by Use Case

### **Interview Preparation**
**Focus on understanding complexity trade-offs:**
1. Can always explain time/space complexity
2. Know when to optimize for time vs. space
3. Understand amortized analysis for data structures
4. Practice complexity analysis under pressure

### **Production Code**
**Prioritize maintainability with adequate performance:**
1. Choose O(n log n) over complex O(n) if simpler
2. Profile before optimizing
3. Consider worst-case vs. average-case performance
4. Document complexity assumptions

### **Competitive Programming**
**Optimize for execution speed:**
1. Know constant factor optimizations
2. Pre-compute when possible
3. Use faster I/O methods
4. Template commonly used algorithms

This complexity guide provides the foundation for making informed algorithmic choices across all problem domains represented in the codebase.