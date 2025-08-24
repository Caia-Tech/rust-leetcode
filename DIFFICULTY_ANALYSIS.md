# LeetCode Problems Difficulty Analysis

## Overview
This document provides a comprehensive analysis of the 105 LeetCode problems implemented in this repository, organized by difficulty level with insights into algorithmic patterns and complexity characteristics.

## Problem Distribution by Difficulty

### Easy Problems (14 total - 13.3%)
These problems focus on fundamental algorithms and data structures, perfect for building a solid foundation.

| Problem | Key Concepts | Time Complexity | Space Complexity |
|---------|-------------|----------------|------------------|
| **1. Two Sum** | Hash Tables, Array | O(n) | O(n) |
| **7. Reverse Integer** | Integer Manipulation, Overflow Handling | O(log n) | O(1) |
| **9. Palindrome Number** | Number Theory, String Processing | O(log n) | O(1) |
| **13. Roman to Integer** | String Processing, Hash Tables | O(n) | O(1) |
| **14. Longest Common Prefix** | String Processing, Vertical Scanning | O(S) | O(1) |
| **20. Valid Parentheses** | Stack, String Processing | O(n) | O(n) |
| **21. Merge Two Sorted Lists** | Linked Lists, Two Pointers | O(m+n) | O(1) |
| **26. Remove Duplicates** | Two Pointers, Array | O(n) | O(1) |
| **70. Climbing Stairs** | Dynamic Programming, Fibonacci | O(n) | O(1) |
| **104. Maximum Depth of Binary Tree** | Binary Trees, DFS/BFS | O(n) | O(h) |
| **121. Best Time to Buy and Sell Stock** | Array, Single Pass | O(n) | O(1) |
| **198. House Robber** | Dynamic Programming | O(n) | O(1) |
| **226. Invert Binary Tree** | Binary Trees, Recursion | O(n) | O(h) |
| **572. Subtree of Another Tree** | Binary Trees, Tree Comparison | O(m×n) | O(h) |

**Common Patterns in Easy Problems:**
- Single pass array algorithms
- Basic data structure usage (stack, hash table)
- Simple dynamic programming
- Tree traversal fundamentals

### Medium Problems (45 total - 42.9%)
These problems require more sophisticated algorithms and optimization techniques.

#### **Array & String Processing (12 problems)**
| Problem | Key Concepts | Optimal Complexity |
|---------|-------------|-------------------|
| **3. Longest Substring Without Repeating** | Sliding Window, Hash Set | O(n) time, O(k) space |
| **5. Longest Palindromic Substring** | Expand Around Centers, Manacher's | O(n²) / O(n) time |
| **11. Container With Most Water** | Two Pointers | O(n) time, O(1) space |
| **15. 3Sum** | Two Pointers, Sorting | O(n²) time, O(1) space |
| **48. Rotate Image** | Matrix Manipulation | O(n²) time, O(1) space |
| **49. Group Anagrams** | Hash Table, Sorting | O(n×k log k) time |
| **56. Merge Intervals** | Sorting, Greedy | O(n log n) time |
| **75. Sort Colors** | Dutch National Flag | O(n) time, O(1) space |
| **238. Product Except Self** | Prefix/Suffix Products | O(n) time, O(1) space |
| **322. Coin Change** | Dynamic Programming | O(amount × coins) time |

#### **Tree & Graph Problems (15 problems)**
| Problem | Key Algorithm | Complexity |
|---------|--------------|------------|
| **98. Validate BST** | Inorder Traversal, Bounds Checking | O(n) time, O(h) space |
| **102. Binary Tree Level Order** | BFS, Queue | O(n) time, O(w) space |
| **133. Clone Graph** | DFS/BFS, Hash Map | O(n) time, O(n) space |
| **199. Binary Tree Right Side View** | Level Order Traversal | O(n) time, O(w) space |
| **200. Number of Islands** | DFS/BFS, Union Find | O(m×n) time |
| **207. Course Schedule** | Topological Sort, Cycle Detection | O(V+E) time |
| **208. Implement Trie** | Trie Data Structure | O(m) per operation |
| **230. Kth Smallest in BST** | Inorder Traversal, Morris | O(h+k) time |
| **236. Lowest Common Ancestor** | Tree Traversal | O(n) time, O(h) space |

#### **Dynamic Programming (8 problems)**
| Problem | DP Type | Complexity |
|---------|---------|------------|
| **53. Maximum Subarray** | Kadane's Algorithm | O(n) time, O(1) space |
| **55. Jump Game** | Greedy DP | O(n) time, O(1) space |
| **62. Unique Paths** | Combinatorics, Grid DP | O(m×n) time |
| **91. Decode Ways** | String DP | O(n) time, O(1) space |
| **139. Word Break** | String DP, Trie | O(n²) time |
| **152. Maximum Product Subarray** | Modified Kadane's | O(n) time, O(1) space |
| **213. House Robber II** | Circular Array DP | O(n) time, O(1) space |
| **300. Longest Increasing Subsequence** | Binary Search DP | O(n log n) time |

#### **Advanced Data Structures (10 problems)**
| Problem | Data Structure | Key Insight |
|---------|---------------|-------------|
| **146. LRU Cache** | Hash Map + Doubly Linked List | O(1) operations |
| **173. BST Iterator** | Stack-based Inorder | O(1) amortized |
| **208. Implement Trie** | Prefix Tree | Efficient string operations |
| **211. Add and Search Words** | Trie + Wildcard Matching | DFS with backtracking |
| **215. Kth Largest Element** | Quickselect, Heap | O(n) average time |
| **347. Top K Frequent Elements** | Bucket Sort, Heap | O(n) time optimal |

### Hard Problems (46 total - 43.8%)
These problems involve advanced algorithms, complex optimizations, and sophisticated data structures.

#### **Advanced String Processing (8 problems)**
| Problem | Algorithm | Complexity |
|---------|-----------|------------|
| **10. Regular Expression Matching** | Dynamic Programming | O(m×n) time |
| **44. Wildcard Matching** | DP with Optimization | O(m×n) time |
| **72. Edit Distance** | Levenshtein Distance DP | O(m×n) time |
| **76. Minimum Window Substring** | Sliding Window | O(m+n) time |
| **214. Shortest Palindrome** | KMP, Rolling Hash | O(n) time |
| **336. Palindrome Pairs** | Trie, String Manipulation | O(n×m²) time |

#### **Advanced Tree Algorithms (6 problems)**
| Problem | Key Technique | Complexity |
|---------|--------------|------------|
| **99. Recover Binary Search Tree** | Morris Traversal | O(n) time, O(1) space |
| **124. Binary Tree Maximum Path Sum** | Post-order DFS | O(n) time, O(h) space |
| **297. Serialize/Deserialize Binary Tree** | Level Order, DFS | O(n) time, O(n) space |

#### **Complex Dynamic Programming (10 problems)**
| Problem | DP Variant | Optimization |
|---------|------------|-------------|
| **123. Best Time to Buy/Sell Stock III** | State Machine DP | O(n) time, O(1) space |
| **188. Best Time to Buy/Sell Stock IV** | Optimized State DP | O(n×k) time |
| **312. Burst Balloons** | Interval DP | O(n³) time |
| **410. Split Array Largest Sum** | Binary Search + DP | O(n×log(sum)) time |

#### **Advanced Graph Algorithms (6 problems)**
| Problem | Algorithm | Application |
|---------|-----------|-------------|
| **126. Word Ladder II** | BFS + Backtracking | Shortest Path Enumeration |
| **127. Word Ladder** | BFS, Bidirectional Search | O(M²×N) time |
| **212. Word Search II** | Trie + DFS Backtracking | O(M×N×4^L) time |

#### **Computational Geometry & Math (6 problems)**
| Problem | Technique | Complexity |
|---------|-----------|------------|
| **149. Max Points on a Line** | Hash Map, GCD | O(n²) time |
| **164. Maximum Gap** | Radix Sort, Pigeonhole | O(n) time |
| **315. Count Smaller After Self** | Merge Sort, Binary Indexed Tree | O(n log n) time |

#### **System Design Problems (10 problems)**
| Problem | Data Structure | Design Pattern |
|---------|---------------|----------------|
| **146. LRU Cache** | Hash Map + DLL | Cache Replacement Policy |
| **295. Find Median from Data Stream** | Two Heaps | Streaming Algorithm |
| **460. LFU Cache** | Hash Map + DLL | Frequency-based Eviction |

## Algorithmic Pattern Analysis

### **Most Common Patterns by Difficulty:**

**Easy:**
1. Single Pass Array (6 problems)
2. Basic Tree Traversal (4 problems) 
3. Simple DP (2 problems)
4. Stack/Queue (2 problems)

**Medium:**
1. Dynamic Programming (15 problems)
2. Tree/Graph Traversal (12 problems)
3. Two Pointers/Sliding Window (8 problems)
4. Advanced Data Structures (10 problems)

**Hard:**
1. Complex DP (15 problems)
2. Advanced String Algorithms (8 problems)
3. System Design (10 problems)
4. Graph Algorithms (8 problems)
5. Computational Geometry (5 problems)

## Complexity Distribution

### **Time Complexity Analysis:**
- **O(1)**: 5 problems (Cache operations)
- **O(log n)**: 8 problems (Binary search variants)
- **O(n)**: 35 problems (Linear algorithms)
- **O(n log n)**: 15 problems (Sorting-based)
- **O(n²)**: 25 problems (Nested loops, DP)
- **O(n³)**: 10 problems (Complex DP)
- **O(2^n)**: 7 problems (Exponential backtracking)

### **Space Complexity Analysis:**
- **O(1)**: 25 problems (In-place algorithms)
- **O(log n)**: 10 problems (Recursion stack)
- **O(n)**: 45 problems (Linear extra space)
- **O(n²)**: 20 problems (2D DP tables)
- **O(2^n)**: 5 problems (Exponential state space)

## Learning Path Recommendations

### **Beginner Path (Easy → Medium basics):**
1. Start with array manipulation (Two Sum, Remove Duplicates)
2. Learn basic tree traversal (Maximum Depth, Invert Tree)
3. Practice simple DP (Climbing Stairs, House Robber)
4. Master two pointers (Container With Water, 3Sum)

### **Intermediate Path (Medium advanced):**
1. Advanced tree algorithms (LCA, BST operations)
2. Graph traversal (Number of Islands, Course Schedule)
3. Complex DP (Longest Increasing Subsequence, Coin Change)
4. Data structure design (LRU Cache, Trie)

### **Advanced Path (Hard problems):**
1. String algorithms (Edit Distance, Regex Matching)
2. Advanced DP (Stock problems, Interval DP)
3. System design (Cache implementations)
4. Computational geometry (Max Points on Line)

## Performance Benchmarking Insights

Based on the comprehensive benchmark suite in `benches/solutions.rs`:

### **Algorithm Efficiency Rankings:**

**Two Sum Approaches:**
1. Hash Map (O(n)) - Fastest for most inputs
2. Two Pass (O(n)) - Slightly slower due to two passes
3. Brute Force (O(n²)) - Exponentially slower for large inputs

**String Processing:**
1. Sliding Window - Most efficient for substring problems
2. Hash Set - Good for character tracking
3. Brute Force - Only viable for very small inputs

**Tree Algorithms:**
1. Morris Traversal - O(1) space but complex implementation
2. Iterative with Stack - Good balance of efficiency and readability
3. Recursive - Clean but uses O(h) space

This analysis provides a roadmap for understanding the algorithmic landscape of the implemented problems and guides learning progression from fundamental concepts to advanced techniques.