# Algorithm Pattern Recognition Guide

## Overview
This guide helps you recognize common algorithmic patterns in LeetCode problems and provides systematic approaches to solving them. Master these patterns to improve your problem-solving speed and accuracy.

## Core Patterns

### 1. **Two Pointers Pattern**
**When to Use:** Array/string problems with sorted data or need to find pairs/triplets.

**Key Indicators:**
- Sorted array or string
- Finding pairs with specific sum
- Removing duplicates in-place
- Palindrome checking
- Merging sorted arrays

**Template:**
```rust
fn two_pointers_template(arr: Vec<i32>, target: i32) -> bool {
    let mut left = 0;
    let mut right = arr.len() - 1;
    
    while left < right {
        let sum = arr[left] + arr[right];
        if sum == target {
            return true;
        } else if sum < target {
            left += 1;
        } else {
            right -= 1;
        }
    }
    false
}
```

**Problems:**
- Two Sum (sorted array variant)
- 3Sum (#15)
- Container With Most Water (#11)
- Remove Duplicates from Sorted Array (#26)
- Valid Palindrome

**Time Complexity:** O(n)
**Space Complexity:** O(1)

### 2. **Sliding Window Pattern**
**When to Use:** Contiguous subarray/substring problems with optimization criteria.

**Key Indicators:**
- "Maximum/minimum subarray/substring"
- "Longest/shortest subarray with condition"
- Fixed or variable window size
- String pattern matching

**Fixed Window Template:**
```rust
fn fixed_sliding_window(arr: Vec<i32>, k: usize) -> i32 {
    let mut window_sum = 0;
    let mut max_sum = 0;
    
    // Initialize first window
    for i in 0..k {
        window_sum += arr[i];
    }
    max_sum = window_sum;
    
    // Slide the window
    for i in k..arr.len() {
        window_sum = window_sum - arr[i - k] + arr[i];
        max_sum = max_sum.max(window_sum);
    }
    
    max_sum
}
```

**Variable Window Template:**
```rust
fn variable_sliding_window(s: String, k: i32) -> i32 {
    let chars: Vec<char> = s.chars().collect();
    let mut char_count = HashMap::new();
    let mut left = 0;
    let mut max_length = 0;
    
    for right in 0..chars.len() {
        // Expand window
        *char_count.entry(chars[right]).or_insert(0) += 1;
        
        // Contract window if needed
        while char_count.len() > k as usize {
            let left_char = chars[left];
            *char_count.get_mut(&left_char).unwrap() -= 1;
            if char_count[&left_char] == 0 {
                char_count.remove(&left_char);
            }
            left += 1;
        }
        
        max_length = max_length.max(right - left + 1);
    }
    
    max_length as i32
}
```

**Problems:**
- Longest Substring Without Repeating Characters (#3)
- Minimum Window Substring (#76)
- Maximum Subarray (#53)
- Sliding Window Maximum (#239)

**Time Complexity:** O(n)
**Space Complexity:** O(k) for hash map

### 3. **Hash Map Pattern**
**When to Use:** Need fast lookups, counting, or tracking relationships.

**Key Indicators:**
- Finding complements or pairs
- Counting frequency
- Detecting duplicates
- Mapping relationships
- Anagram problems

**Template:**
```rust
fn hash_map_pattern(nums: Vec<i32>, target: i32) -> Vec<i32> {
    let mut map: HashMap<i32, usize> = HashMap::new();
    
    for (i, &num) in nums.iter().enumerate() {
        let complement = target - num;
        if let Some(&complement_index) = map.get(&complement) {
            return vec![complement_index as i32, i as i32];
        }
        map.insert(num, i);
    }
    
    vec![]
}
```

**Problems:**
- Two Sum (#1)
- Group Anagrams (#49)
- Top K Frequent Elements (#347)
- LRU Cache (#146)
- Longest Consecutive Sequence (#128)

**Time Complexity:** O(n) average
**Space Complexity:** O(n)

### 4. **Dynamic Programming Pattern**

#### **1D DP (Linear)**
**When to Use:** Optimal substructure with overlapping subproblems in sequence.

**Template:**
```rust
fn dp_1d_pattern(nums: Vec<i32>) -> i32 {
    if nums.is_empty() { return 0; }
    
    let n = nums.len();
    let mut dp = vec![0; n];
    dp[0] = nums[0];
    
    for i in 1..n {
        dp[i] = dp[i-1].max(nums[i]); // Example: choose max
    }
    
    dp[n-1]
}
```

**Space Optimized:**
```rust
fn dp_1d_optimized(nums: Vec<i32>) -> i32 {
    if nums.is_empty() { return 0; }
    
    let mut prev = nums[0];
    let mut curr = nums[0];
    
    for i in 1..nums.len() {
        let temp = curr;
        curr = prev.max(nums[i]); // Example logic
        prev = temp;
    }
    
    curr
}
```

**Problems:**
- Climbing Stairs (#70)
- House Robber (#198)
- Maximum Subarray (#53)
- Coin Change (#322)

#### **2D DP (Grid/Table)**
**When to Use:** Two-dimensional choices or string/array combinations.

**Template:**
```rust
fn dp_2d_pattern(word1: String, word2: String) -> i32 {
    let m = word1.len();
    let n = word2.len();
    let mut dp = vec![vec![0; n + 1]; m + 1];
    
    // Initialize base cases
    for i in 0..=m {
        dp[i][0] = i as i32;
    }
    for j in 0..=n {
        dp[0][j] = j as i32;
    }
    
    let word1_chars: Vec<char> = word1.chars().collect();
    let word2_chars: Vec<char> = word2.chars().collect();
    
    for i in 1..=m {
        for j in 1..=n {
            if word1_chars[i-1] == word2_chars[j-1] {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + dp[i-1][j].min(dp[i][j-1]).min(dp[i-1][j-1]);
            }
        }
    }
    
    dp[m][n]
}
```

**Problems:**
- Edit Distance (#72)
- Longest Common Subsequence
- Unique Paths (#62)
- Regular Expression Matching (#10)

### 5. **Tree Traversal Patterns**

#### **DFS (Depth-First Search)**
**When to Use:** Need to explore paths, find depth, or process nodes in specific order.

**Recursive Template:**
```rust
use std::rc::Rc;
use std::cell::RefCell;

fn dfs_recursive(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    match root {
        None => 0,
        Some(node) => {
            let node = node.borrow();
            let left_result = dfs_recursive(node.left.clone());
            let right_result = dfs_recursive(node.right.clone());
            
            // Process current node
            1 + left_result.max(right_result)
        }
    }
}
```

**Iterative Template:**
```rust
fn dfs_iterative(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let mut result = Vec::new();
    let mut stack = Vec::new();
    
    if let Some(node) = root {
        stack.push(node);
    }
    
    while let Some(node) = stack.pop() {
        let node = node.borrow();
        result.push(node.val);
        
        // Right first (will be processed last)
        if let Some(right) = node.right.clone() {
            stack.push(right);
        }
        if let Some(left) = node.left.clone() {
            stack.push(left);
        }
    }
    
    result
}
```

#### **BFS (Breadth-First Search)**
**When to Use:** Level-order processing, shortest path in unweighted tree.

**Template:**
```rust
use std::collections::VecDeque;

fn bfs_pattern(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
    let mut result = Vec::new();
    if root.is_none() { return result; }
    
    let mut queue = VecDeque::new();
    queue.push_back(root.unwrap());
    
    while !queue.is_empty() {
        let level_size = queue.len();
        let mut level = Vec::new();
        
        for _ in 0..level_size {
            if let Some(node) = queue.pop_front() {
                let node = node.borrow();
                level.push(node.val);
                
                if let Some(left) = node.left.clone() {
                    queue.push_back(left);
                }
                if let Some(right) = node.right.clone() {
                    queue.push_back(right);
                }
            }
        }
        
        result.push(level);
    }
    
    result
}
```

**Problems:**
- Maximum Depth of Binary Tree (#104)
- Binary Tree Level Order Traversal (#102)
- Path Sum problems
- Binary Tree Maximum Path Sum (#124)

### 6. **Binary Search Pattern**
**When to Use:** Sorted data, finding boundaries, optimization problems.

**Key Indicators:**
- Sorted array/matrix
- "Find first/last occurrence"
- "Find minimum/maximum value that satisfies condition"
- Search in rotated array

**Standard Template:**
```rust
fn binary_search_pattern(nums: Vec<i32>, target: i32) -> i32 {
    let mut left = 0;
    let mut right = nums.len() as i32 - 1;
    
    while left <= right {
        let mid = left + (right - left) / 2;
        
        if nums[mid as usize] == target {
            return mid;
        } else if nums[mid as usize] < target {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    -1 // Not found
}
```

**Find Boundary Template:**
```rust
fn find_first_occurrence(nums: Vec<i32>, target: i32) -> i32 {
    let mut left = 0;
    let mut right = nums.len() as i32 - 1;
    let mut result = -1;
    
    while left <= right {
        let mid = left + (right - left) / 2;
        
        if nums[mid as usize] == target {
            result = mid;
            right = mid - 1; // Continue searching left
        } else if nums[mid as usize] < target {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    result
}
```

**Problems:**
- Binary Search
- Search in Rotated Sorted Array (#33)
- Find First and Last Position (#34)
- Find Minimum in Rotated Sorted Array (#153)

### 7. **Backtracking Pattern**
**When to Use:** Generate all possible solutions, constraint satisfaction.

**Key Indicators:**
- "Find all combinations/permutations"
- "Generate all valid..."
- Decision tree exploration
- Constraint satisfaction (N-Queens, Sudoku)

**Template:**
```rust
fn backtrack_pattern(nums: Vec<i32>) -> Vec<Vec<i32>> {
    let mut result = Vec::new();
    let mut current_path = Vec::new();
    
    backtrack(&nums, &mut current_path, &mut result, 0);
    result
}

fn backtrack(nums: &[i32], path: &mut Vec<i32>, result: &mut Vec<Vec<i32>>, start: usize) {
    // Base case - add valid solution
    if is_valid_solution(path) {
        result.push(path.clone());
        return;
    }
    
    for i in start..nums.len() {
        // Choose
        path.push(nums[i]);
        
        // Explore
        if is_valid_partial(path) {
            backtrack(nums, path, result, i + 1);
        }
        
        // Unchoose (backtrack)
        path.pop();
    }
}

fn is_valid_solution(path: &[i32]) -> bool {
    // Define when solution is complete
    path.len() == 3 // Example condition
}

fn is_valid_partial(path: &[i32]) -> bool {
    // Define when partial solution is still viable
    true // Example - always continue
}
```

**Problems:**
- Generate Parentheses (#22)
- Permutations (#46)
- Subsets (#78)
- N-Queens (#51)
- Sudoku Solver (#37)

### 8. **Graph Traversal Patterns**

#### **DFS for Graphs**
```rust
use std::collections::{HashMap, HashSet};

fn graph_dfs(graph: HashMap<i32, Vec<i32>>, start: i32) -> Vec<i32> {
    let mut visited = HashSet::new();
    let mut result = Vec::new();
    
    dfs_helper(&graph, start, &mut visited, &mut result);
    result
}

fn dfs_helper(graph: &HashMap<i32, Vec<i32>>, node: i32, 
              visited: &mut HashSet<i32>, result: &mut Vec<i32>) {
    visited.insert(node);
    result.push(node);
    
    if let Some(neighbors) = graph.get(&node) {
        for &neighbor in neighbors {
            if !visited.contains(&neighbor) {
                dfs_helper(graph, neighbor, visited, result);
            }
        }
    }
}
```

#### **BFS for Graphs**
```rust
use std::collections::{HashMap, HashSet, VecDeque};

fn graph_bfs(graph: HashMap<i32, Vec<i32>>, start: i32) -> Vec<i32> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut result = Vec::new();
    
    queue.push_back(start);
    visited.insert(start);
    
    while let Some(node) = queue.pop_front() {
        result.push(node);
        
        if let Some(neighbors) = graph.get(&node) {
            for &neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
    }
    
    result
}
```

**Problems:**
- Number of Islands (#200)
- Clone Graph
- Course Schedule (#207)
- Pacific Atlantic Water Flow (#417)

### 9. **Union-Find Pattern**
**When to Use:** Dynamic connectivity, grouping elements, cycle detection.

**Template:**
```rust
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }
    
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]); // Path compression
        }
        self.parent[x]
    }
    
    fn union(&mut self, x: usize, y: usize) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);
        
        if root_x == root_y {
            return false; // Already connected
        }
        
        // Union by rank
        if self.rank[root_x] < self.rank[root_y] {
            self.parent[root_x] = root_y;
        } else if self.rank[root_x] > self.rank[root_y] {
            self.parent[root_y] = root_x;
        } else {
            self.parent[root_y] = root_x;
            self.rank[root_x] += 1;
        }
        
        true
    }
    
    fn is_connected(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }
}
```

**Problems:**
- Number of Islands (alternative approach)
- Redundant Connection
- Graph Valid Tree

### 10. **Heap Pattern**
**When to Use:** Finding K-th element, maintaining order in stream, priority-based problems.

**Template:**
```rust
use std::collections::BinaryHeap;
use std::cmp::Reverse;

fn heap_pattern(nums: Vec<i32>, k: usize) -> Vec<i32> {
    // Max heap (default)
    let mut max_heap = BinaryHeap::new();
    
    // Min heap (using Reverse)
    let mut min_heap = BinaryHeap::new();
    
    for num in nums {
        max_heap.push(num);
        min_heap.push(Reverse(num));
        
        // Maintain heap size
        if max_heap.len() > k {
            max_heap.pop();
        }
    }
    
    max_heap.into_sorted_vec()
}
```

**Problems:**
- Kth Largest Element in Array (#215)
- Top K Frequent Elements (#347)
- Find Median from Data Stream (#295)
- Merge k Sorted Lists (#23)

## Pattern Recognition Strategy

### 1. **Problem Classification Steps**
1. **Read the problem carefully** - Identify input/output types
2. **Look for keywords** - "maximum", "minimum", "all combinations", "shortest path"
3. **Analyze constraints** - Array size, value ranges, time limits
4. **Identify data structures** - What's given? What's needed?
5. **Match to patterns** - Which template fits best?

### 2. **Common Problem Categories**

#### **Array/String Problems**
- **Two Pointers**: Sorted arrays, palindromes, pair finding
- **Sliding Window**: Contiguous subarrays, pattern matching
- **Hash Map**: Frequency counting, fast lookups
- **Binary Search**: Sorted arrays, finding boundaries

#### **Tree Problems**
- **DFS**: Path problems, depth calculations, tree modification
- **BFS**: Level-order, shortest path, tree width
- **Binary Search Tree**: Inorder traversal, validation

#### **Graph Problems**
- **DFS**: Connected components, cycle detection, path finding
- **BFS**: Shortest path, level processing
- **Union-Find**: Dynamic connectivity, grouping

#### **Dynamic Programming Problems**
- **1D DP**: Sequential decisions, optimal substructure
- **2D DP**: String matching, grid problems, combinations
- **Memoization**: Top-down recursive solutions

### 3. **Pattern Selection Flowchart**

```
Input Analysis
├── Sorted Array/String
│   ├── Finding pairs/triplets → Two Pointers
│   ├── Search problem → Binary Search
│   └── Contiguous subarray → Sliding Window
├── Unsorted Array/String
│   ├── Frequency/counting → Hash Map
│   ├── Contiguous subarray → Sliding Window
│   └── All combinations → Backtracking
├── Tree Structure
│   ├── Path/depth problems → DFS
│   ├── Level processing → BFS
│   └── BST operations → Inorder DFS
├── Graph Structure
│   ├── Connected components → DFS/Union-Find
│   ├── Shortest path → BFS
│   └── Dynamic connectivity → Union-Find
├── Optimization Problem
│   ├── Overlapping subproblems → DP
│   ├── Greedy choice → Greedy Algorithm
│   └── Multiple solutions → Backtracking
└── Stream/Priority
    ├── K-th element → Heap
    ├── Running median → Two Heaps
    └── LRU/LFU → Hash + LinkedList
```

### 4. **Time Complexity Quick Reference**

| Pattern | Best Case | Average | Worst Case |
|---------|-----------|---------|------------|
| Two Pointers | O(n) | O(n) | O(n) |
| Sliding Window | O(n) | O(n) | O(n) |
| Hash Map | O(1) lookup | O(n) space | O(n²) collisions |
| Binary Search | O(log n) | O(log n) | O(log n) |
| DFS/BFS | O(V+E) | O(V+E) | O(V+E) |
| Union-Find | O(α(n)) | O(α(n)) | O(log n) |
| Heap Operations | O(1) peek | O(log n) insert/delete | O(log n) |
| 1D DP | O(n) | O(n) | O(n) |
| 2D DP | O(m×n) | O(m×n) | O(m×n) |
| Backtracking | O(2^n) | O(b^d) | O(n!) |

### 5. **Practice Strategy**

#### **Week 1-2: Foundation Patterns**
1. Two Pointers (5 problems)
2. Hash Map (5 problems)
3. Sliding Window (5 problems)

#### **Week 3-4: Search & Traversal**
1. Binary Search (5 problems)
2. DFS/BFS Trees (5 problems)
3. DFS/BFS Graphs (5 problems)

#### **Week 5-6: Advanced Patterns**
1. Dynamic Programming (10 problems)
2. Backtracking (5 problems)
3. Heap/Priority Queue (5 problems)

#### **Week 7-8: Integration**
1. Mixed pattern problems
2. System design problems
3. Complex algorithm combinations

## Common Pitfalls and Tips

### **Two Pointers**
- ❌ Forgetting to handle empty arrays
- ❌ Off-by-one errors in boundary conditions
- ✅ Always check `left < right` condition
- ✅ Consider edge cases like single element

### **Sliding Window**
- ❌ Not updating window state correctly
- ❌ Expanding/contracting window in wrong order
- ✅ Track window state with hash map or counters
- ✅ Use while loop for contracting variable windows

### **Dynamic Programming**
- ❌ Not identifying base cases properly
- ❌ Incorrect state transition
- ✅ Start with recursive solution, then memoize
- ✅ Always consider space optimization

### **Backtracking**
- ❌ Not properly backtracking (removing choices)
- ❌ Not pruning invalid paths early
- ✅ Always undo choices after recursive calls
- ✅ Add early termination conditions

### **Tree Traversal**
- ❌ Forgetting null checks
- ❌ Stack overflow with deep recursion
- ✅ Handle null nodes explicitly
- ✅ Use iterative solution for very deep trees

This pattern recognition guide provides a systematic approach to tackling LeetCode problems efficiently. Practice identifying patterns quickly and implementing the corresponding templates to improve your problem-solving speed and accuracy.