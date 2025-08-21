//! # Problem 207: Course Schedule
//!
//! There are a total of `numCourses` courses you have to take, labeled from `0` to `numCourses - 1`.
//! You are given an array `prerequisites` where `prerequisites[i] = [a_i, b_i]` indicates that you
//! must take course `b_i` first if you want to take course `a_i`.
//!
//! For example, the pair `[0, 1]` indicates that to take course `0` you have to first take course `1`.
//!
//! Return `true` if you can finish all courses. Otherwise, return `false`.
//!
//! ## Examples
//!
//! ```text
//! Input: numCourses = 2, prerequisites = [[1,0]]
//! Output: true
//! Explanation: There are a total of 2 courses to take. 
//! To take course 1 you should have finished course 0. So it is possible.
//! ```
//!
//! ```text
//! Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
//! Output: false
//! Explanation: There are a total of 2 courses to take. 
//! To take course 1 you should have finished course 0, and to take course 0 you 
//! should also have finished course 1. So it is impossible.
//! ```
//!
//! ## Constraints
//!
//! * 1 <= numCourses <= 2000
//! * 0 <= prerequisites.length <= 5000
//! * prerequisites[i].length == 2
//! * 0 <= a_i, b_i < numCourses
//! * All the pairs prerequisites[i] are unique

use std::collections::{HashSet, VecDeque};

/// Solution for Course Schedule problem
pub struct Solution;

impl Solution {
    /// Creates a new instance of Solution
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Kahn's Algorithm (BFS Topological Sort) - Optimal
    /// 
    /// **Algorithm:**
    /// 1. Build adjacency list and calculate in-degrees for all nodes
    /// 2. Add all nodes with in-degree 0 to queue (no prerequisites)
    /// 3. Process queue: remove node, decrement in-degrees of neighbors
    /// 4. If neighbor's in-degree becomes 0, add to queue
    /// 5. Count processed nodes - if equals numCourses, no cycle exists
    /// 
    /// **Time Complexity:** O(V + E) where V = numCourses, E = prerequisites
    /// **Space Complexity:** O(V + E) for adjacency list and in-degree array
    /// 
    /// **Key Insights:**
    /// - Cycle detection through topological sorting
    /// - In-degree 0 means no remaining prerequisites
    /// - If we can process all nodes, graph is acyclic (DAG)
    /// 
    /// **Why this works:**
    /// - DAG (Directed Acyclic Graph) has valid topological ordering
    /// - Cycle means some courses can never be completed
    /// - Kahn's algorithm processes nodes in topological order
    /// 
    /// **Step-by-step for [[1,0]]:**
    /// ```text
    /// Courses: [0, 1], Prerequisites: 1 depends on 0
    /// In-degrees: [0, 1] (course 0 has no prereqs, course 1 has 1 prereq)
    /// Queue starts with: [0] (in-degree 0)
    /// Process 0: remove from queue, reduce in-degree of 1 to 0
    /// Add 1 to queue: [1]
    /// Process 1: remove from queue
    /// All courses processed -> true
    /// ```
    pub fn can_finish(&self, num_courses: i32, prerequisites: Vec<Vec<i32>>) -> bool {
        let n = num_courses as usize;
        let mut graph = vec![Vec::new(); n];
        let mut in_degree = vec![0; n];
        
        // Build adjacency list and calculate in-degrees
        for prerequisite in prerequisites {
            let course = prerequisite[0] as usize;
            let prereq = prerequisite[1] as usize;
            
            graph[prereq].push(course);
            in_degree[course] += 1;
        }
        
        // Find all courses with no prerequisites
        let mut queue = VecDeque::new();
        for i in 0..n {
            if in_degree[i] == 0 {
                queue.push_back(i);
            }
        }
        
        let mut processed = 0;
        
        // Process courses in topological order
        while let Some(course) = queue.pop_front() {
            processed += 1;
            
            // Remove this course and update dependent courses
            for &dependent in &graph[course] {
                in_degree[dependent] -= 1;
                if in_degree[dependent] == 0 {
                    queue.push_back(dependent);
                }
            }
        }
        
        processed == n
    }

    /// # Approach 2: DFS Cycle Detection with Colors
    /// 
    /// **Algorithm:**
    /// 1. Use three colors: WHITE (unvisited), GRAY (visiting), BLACK (visited)
    /// 2. For each unvisited node, start DFS
    /// 3. Mark node GRAY when entering, BLACK when exiting
    /// 4. If we encounter GRAY node during DFS, cycle exists
    /// 
    /// **Time Complexity:** O(V + E) where V = numCourses, E = prerequisites
    /// **Space Complexity:** O(V + E) for adjacency list and color array
    /// 
    /// **Color semantics:**
    /// - WHITE: Never visited
    /// - GRAY: Currently being processed (in recursion stack)
    /// - BLACK: Completely processed
    /// 
    /// **Cycle detection:** Finding a back edge (GRAY node during DFS)
    /// 
    /// **When to use:** When you need the actual cycle or path information
    pub fn can_finish_dfs(&self, num_courses: i32, prerequisites: Vec<Vec<i32>>) -> bool {
        let n = num_courses as usize;
        let mut graph = vec![Vec::new(); n];
        
        // Build adjacency list
        for prerequisite in prerequisites {
            let course = prerequisite[0] as usize;
            let prereq = prerequisite[1] as usize;
            graph[prereq].push(course);
        }
        
        // 0: WHITE (unvisited), 1: GRAY (visiting), 2: BLACK (visited)
        let mut color = vec![0; n];
        
        fn has_cycle(graph: &Vec<Vec<usize>>, color: &mut Vec<i32>, node: usize) -> bool {
            if color[node] == 1 { // GRAY - back edge found
                return true;
            }
            if color[node] == 2 { // BLACK - already processed
                return false;
            }
            
            color[node] = 1; // Mark as GRAY (visiting)
            
            for &neighbor in &graph[node] {
                if has_cycle(graph, color, neighbor) {
                    return true;
                }
            }
            
            color[node] = 2; // Mark as BLACK (visited)
            false
        }
        
        // Check each component for cycles
        for i in 0..n {
            if color[i] == 0 && has_cycle(&graph, &mut color, i) {
                return false;
            }
        }
        
        true
    }

    /// # Approach 3: DFS with Recursion Stack
    /// 
    /// **Algorithm:**
    /// 1. Maintain visited set and recursion stack set
    /// 2. For each unvisited node, perform DFS
    /// 3. Add node to recursion stack when entering DFS
    /// 4. If we visit a node already in recursion stack, cycle exists
    /// 5. Remove from recursion stack when backtracking
    /// 
    /// **Time Complexity:** O(V + E) where V = numCourses, E = prerequisites
    /// **Space Complexity:** O(V + E) for adjacency list and sets
    /// 
    /// **Difference from approach 2:**
    /// - Uses explicit recursion stack instead of color coding
    /// - More intuitive understanding of cycle detection
    /// - Similar performance characteristics
    /// 
    /// **Educational value:** Shows different implementation styles for same concept
    pub fn can_finish_recursion_stack(&self, num_courses: i32, prerequisites: Vec<Vec<i32>>) -> bool {
        let n = num_courses as usize;
        let mut graph = vec![Vec::new(); n];
        
        // Build adjacency list
        for prerequisite in prerequisites {
            let course = prerequisite[0] as usize;
            let prereq = prerequisite[1] as usize;
            graph[prereq].push(course);
        }
        
        let mut visited = vec![false; n];
        let mut rec_stack = vec![false; n];
        
        fn has_cycle_rec(
            graph: &Vec<Vec<usize>>, 
            visited: &mut Vec<bool>, 
            rec_stack: &mut Vec<bool>, 
            node: usize
        ) -> bool {
            visited[node] = true;
            rec_stack[node] = true;
            
            for &neighbor in &graph[node] {
                if !visited[neighbor] {
                    if has_cycle_rec(graph, visited, rec_stack, neighbor) {
                        return true;
                    }
                } else if rec_stack[neighbor] {
                    return true; // Back edge found
                }
            }
            
            rec_stack[node] = false; // Remove from recursion stack
            false
        }
        
        for i in 0..n {
            if !visited[i] && has_cycle_rec(&graph, &mut visited, &mut rec_stack, i) {
                return false;
            }
        }
        
        true
    }

    /// # Approach 4: DFS with Memoization
    /// 
    /// **Algorithm:**
    /// 1. Use DFS to detect cycles with memoization
    /// 2. Cache results for each node to avoid recomputation
    /// 3. Three states: unvisited, visiting (in current path), visited
    /// 4. If we reach a visiting node, cycle detected
    /// 
    /// **Time Complexity:** O(V + E) where V = numCourses, E = prerequisites
    /// **Space Complexity:** O(V + E) for adjacency list and memoization
    /// 
    /// **Advantages:**
    /// - Avoids redundant computation through memoization
    /// - Clear state management
    /// - Works correctly for directed graphs
    /// 
    /// **When useful:** When graph has many shared substructures
    pub fn can_finish_memoized(&self, num_courses: i32, prerequisites: Vec<Vec<i32>>) -> bool {
        let n = num_courses as usize;
        let mut graph = vec![Vec::new(); n];
        
        // Build adjacency list
        for prerequisite in prerequisites {
            let course = prerequisite[0] as usize;
            let prereq = prerequisite[1] as usize;
            graph[prereq].push(course);
        }
        
        // 0: unvisited, 1: visiting, 2: visited (no cycle)
        let mut memo = vec![0; n];
        
        fn has_cycle_memo(
            graph: &Vec<Vec<usize>>, 
            memo: &mut Vec<i32>, 
            node: usize
        ) -> bool {
            if memo[node] == 1 { // Currently visiting - cycle found
                return true;
            }
            if memo[node] == 2 { // Already verified no cycle
                return false;
            }
            
            memo[node] = 1; // Mark as visiting
            
            for &neighbor in &graph[node] {
                if has_cycle_memo(graph, memo, neighbor) {
                    return true;
                }
            }
            
            memo[node] = 2; // Mark as verified (no cycle)
            false
        }
        
        for i in 0..n {
            if memo[i] == 0 && has_cycle_memo(&graph, &mut memo, i) {
                return false;
            }
        }
        
        true
    }

    /// # Approach 5: Modified DFS with Path Tracking
    /// 
    /// **Algorithm:**
    /// 1. Track current path during DFS traversal
    /// 2. If we revisit a node in current path, cycle exists
    /// 3. Use path set for O(1) cycle detection
    /// 4. Remove nodes from path when backtracking
    /// 
    /// **Time Complexity:** O(V + E) where V = numCourses, E = prerequisites
    /// **Space Complexity:** O(V + E) for adjacency list and path tracking
    /// 
    /// **Advantages:**
    /// - Can easily reconstruct the cycle if needed
    /// - Clear separation between global visited and current path
    /// - Intuitive understanding of cycle detection
    /// 
    /// **When useful:** When you need to report the actual cycle
    pub fn can_finish_path_tracking(&self, num_courses: i32, prerequisites: Vec<Vec<i32>>) -> bool {
        let n = num_courses as usize;
        let mut graph = vec![Vec::new(); n];
        
        // Build adjacency list
        for prerequisite in prerequisites {
            let course = prerequisite[0] as usize;
            let prereq = prerequisite[1] as usize;
            graph[prereq].push(course);
        }
        
        let mut visited = vec![false; n];
        
        fn has_cycle_path(
            graph: &Vec<Vec<usize>>, 
            visited: &mut Vec<bool>, 
            path: &mut HashSet<usize>, 
            node: usize
        ) -> bool {
            if path.contains(&node) {
                return true; // Cycle detected
            }
            
            if visited[node] {
                return false; // Already processed
            }
            
            visited[node] = true;
            path.insert(node);
            
            for &neighbor in &graph[node] {
                if has_cycle_path(graph, visited, path, neighbor) {
                    return true;
                }
            }
            
            path.remove(&node);
            false
        }
        
        for i in 0..n {
            if !visited[i] {
                let mut path = HashSet::new();
                if has_cycle_path(&graph, &mut visited, &mut path, i) {
                    return false;
                }
            }
        }
        
        true
    }

    /// # Approach 6: Iterative DFS with Stack
    /// 
    /// **Algorithm:**
    /// 1. Use explicit stack instead of recursion
    /// 2. Track state of each node (unvisited, visiting, visited)
    /// 3. Push neighbors to stack and continue DFS
    /// 4. Detect cycles when encountering visiting node
    /// 
    /// **Time Complexity:** O(V + E) where V = numCourses, E = prerequisites
    /// **Space Complexity:** O(V + E) for adjacency list and stack
    /// 
    /// **Advantages:**
    /// - Avoids recursion stack overflow for large graphs
    /// - More control over traversal order
    /// - Can be easier to debug step by step
    /// 
    /// **Challenges:**
    /// - More complex state management
    /// - Need to carefully handle backtracking logic
    /// 
    /// **When useful:** Very large graphs where recursion depth is a concern
    pub fn can_finish_iterative_dfs(&self, num_courses: i32, prerequisites: Vec<Vec<i32>>) -> bool {
        let n = num_courses as usize;
        let mut graph = vec![Vec::new(); n];
        
        // Build adjacency list
        for prerequisite in prerequisites {
            let course = prerequisite[0] as usize;
            let prereq = prerequisite[1] as usize;
            graph[prereq].push(course);
        }
        
        // 0: unvisited, 1: visiting, 2: visited
        let mut state = vec![0; n];
        
        for start in 0..n {
            if state[start] != 0 {
                continue;
            }
            
            let mut stack = vec![start];
            
            while let Some(node) = stack.last().copied() {
                if state[node] == 0 {
                    state[node] = 1; // Mark as visiting
                    
                    // Add unvisited neighbors to stack
                    for &neighbor in &graph[node] {
                        if state[neighbor] == 1 {
                            return false; // Cycle detected
                        }
                        if state[neighbor] == 0 {
                            stack.push(neighbor);
                        }
                    }
                } else {
                    // All neighbors processed, mark as visited
                    state[node] = 2;
                    stack.pop();
                }
            }
        }
        
        true
    }
}

impl Default for Solution {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Solution {
        Solution::new()
    }

    #[test]
    fn test_basic_examples() {
        let solution = setup();
        
        // Example 1: Can finish - linear dependency
        assert_eq!(solution.can_finish(2, vec![vec![1, 0]]), true);
        
        // Example 2: Cannot finish - cyclic dependency
        assert_eq!(solution.can_finish(2, vec![vec![1, 0], vec![0, 1]]), false);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // Single course, no prerequisites
        assert_eq!(solution.can_finish(1, vec![]), true);
        
        // Multiple courses, no prerequisites
        assert_eq!(solution.can_finish(3, vec![]), true);
        
        // Self-dependency (impossible)
        assert_eq!(solution.can_finish(1, vec![vec![0, 0]]), false);
        
        // Empty prerequisites
        assert_eq!(solution.can_finish(5, vec![]), true);
    }

    #[test]
    fn test_linear_dependencies() {
        let solution = setup();
        
        // Linear chain: 0 -> 1 -> 2 -> 3
        let prerequisites = vec![vec![1, 0], vec![2, 1], vec![3, 2]];
        assert_eq!(solution.can_finish(4, prerequisites), true);
        
        // Reverse chain: 3 -> 2 -> 1 -> 0
        let prerequisites = vec![vec![2, 3], vec![1, 2], vec![0, 1]];
        assert_eq!(solution.can_finish(4, prerequisites), true);
    }

    #[test]
    fn test_tree_dependencies() {
        let solution = setup();
        
        // Tree structure: 0 is root, 1,2 depend on 0, 3,4 depend on 1
        let prerequisites = vec![
            vec![1, 0], vec![2, 0],
            vec![3, 1], vec![4, 1]
        ];
        assert_eq!(solution.can_finish(5, prerequisites), true);
    }

    #[test]
    fn test_cycles() {
        let solution = setup();
        
        // Simple 2-cycle
        assert_eq!(solution.can_finish(2, vec![vec![0, 1], vec![1, 0]]), false);
        
        // 3-cycle: 0 -> 1 -> 2 -> 0
        let prerequisites = vec![vec![1, 0], vec![2, 1], vec![0, 2]];
        assert_eq!(solution.can_finish(3, prerequisites), false);
        
        // Larger cycle: 0 -> 1 -> 2 -> 3 -> 0
        let prerequisites = vec![vec![1, 0], vec![2, 1], vec![3, 2], vec![0, 3]];
        assert_eq!(solution.can_finish(4, prerequisites), false);
    }

    #[test]
    fn test_complex_graphs() {
        let solution = setup();
        
        // DAG with multiple paths
        let prerequisites = vec![
            vec![1, 0], vec![2, 0],  // 1,2 depend on 0
            vec![3, 1], vec![3, 2],  // 3 depends on both 1,2
            vec![4, 3]               // 4 depends on 3
        ];
        assert_eq!(solution.can_finish(5, prerequisites), true);
        
        // Complex cycle hidden in larger graph
        let prerequisites = vec![
            vec![1, 0], vec![2, 1], vec![3, 2],  // Linear chain
            vec![4, 3], vec![5, 4],              // Continue chain
            vec![2, 5]                           // Create cycle: 2->3->4->5->2
        ];
        assert_eq!(solution.can_finish(6, prerequisites), false);
    }

    #[test]
    fn test_disconnected_components() {
        let solution = setup();
        
        // Two separate linear chains
        let prerequisites = vec![
            vec![1, 0], vec![2, 1],  // Chain: 0->1->2
            vec![4, 3], vec![5, 4]   // Chain: 3->4->5
        ];
        assert_eq!(solution.can_finish(6, prerequisites), true);
        
        // One good component, one with cycle
        let prerequisites = vec![
            vec![1, 0], vec![2, 1],  // Good chain: 0->1->2
            vec![4, 3], vec![3, 4]   // Cycle: 3->4->3
        ];
        assert_eq!(solution.can_finish(5, prerequisites), false);
    }

    #[test]
    fn test_duplicate_prerequisites() {
        let solution = setup();
        
        // Duplicate prerequisites should be handled correctly
        let prerequisites = vec![
            vec![1, 0], vec![1, 0],  // Duplicate
            vec![2, 1]
        ];
        assert_eq!(solution.can_finish(3, prerequisites), true);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            (2, vec![vec![1, 0]]),
            (2, vec![vec![1, 0], vec![0, 1]]),
            (4, vec![vec![1, 0], vec![2, 1], vec![3, 2]]),
            (3, vec![vec![1, 0], vec![2, 1], vec![0, 2]]),
            (5, vec![]),
            (1, vec![vec![0, 0]]),
            (6, vec![vec![1, 0], vec![2, 1], vec![3, 2], vec![4, 3], vec![5, 4], vec![2, 5]]),
        ];
        
        for (num_courses, prerequisites) in test_cases {
            let result1 = solution.can_finish(num_courses, prerequisites.clone());
            let result2 = solution.can_finish_dfs(num_courses, prerequisites.clone());
            let result3 = solution.can_finish_recursion_stack(num_courses, prerequisites.clone());
            let result4 = solution.can_finish_memoized(num_courses, prerequisites.clone());
            let result5 = solution.can_finish_path_tracking(num_courses, prerequisites.clone());
            let result6 = solution.can_finish_iterative_dfs(num_courses, prerequisites.clone());
            
            assert_eq!(result1, result2, "Kahn vs DFS mismatch for {} courses, {:?}", num_courses, prerequisites);
            assert_eq!(result2, result3, "DFS vs Recursion Stack mismatch for {} courses, {:?}", num_courses, prerequisites);
            assert_eq!(result3, result4, "Recursion Stack vs Memoized mismatch for {} courses, {:?}", num_courses, prerequisites);
            assert_eq!(result4, result5, "Memoized vs Path Tracking mismatch for {} courses, {:?}", num_courses, prerequisites);
            assert_eq!(result5, result6, "Path Tracking vs Iterative DFS mismatch for {} courses, {:?}", num_courses, prerequisites);
        }
    }

    #[test]
    fn test_boundary_conditions() {
        let solution = setup();
        
        // Minimum courses
        assert_eq!(solution.can_finish(1, vec![]), true);
        
        // Maximum courses with no prerequisites
        assert_eq!(solution.can_finish(2000, vec![]), true);
        
        // Maximum courses with linear dependency
        let mut prerequisites = Vec::new();
        for i in 1..100 {
            prerequisites.push(vec![i, i-1]);
        }
        assert_eq!(solution.can_finish(100, prerequisites), true);
    }

    #[test]
    fn test_performance_patterns() {
        let solution = setup();
        
        // Star pattern: all courses depend on course 0
        let mut prerequisites = Vec::new();
        for i in 1..50 {
            prerequisites.push(vec![i, 0]);
        }
        assert_eq!(solution.can_finish(50, prerequisites), true);
        
        // Reverse star: course 0 depends on all others
        let mut prerequisites = Vec::new();
        for i in 1..50 {
            prerequisites.push(vec![0, i]);
        }
        assert_eq!(solution.can_finish(50, prerequisites), true);
        
        // Complete DAG: each course depends on all previous courses
        let mut prerequisites = Vec::new();
        for i in 1..20 {
            for j in 0..i {
                prerequisites.push(vec![i, j]);
            }
        }
        assert_eq!(solution.can_finish(20, prerequisites), true);
    }

    #[test]
    fn test_mathematical_properties() {
        let solution = setup();
        
        // Property: If graph is acyclic, can always finish
        let acyclic_prereqs = vec![vec![1, 0], vec![2, 0], vec![3, 1], vec![3, 2]];
        assert_eq!(solution.can_finish(4, acyclic_prereqs), true);
        
        // Property: If graph has cycle, cannot finish
        let cyclic_prereqs = vec![vec![1, 0], vec![2, 1], vec![0, 2]];
        assert_eq!(solution.can_finish(3, cyclic_prereqs), false);
        
        // Property: Adding edge to acyclic graph may create cycle
        let mut prereqs = vec![vec![1, 0], vec![2, 1]];
        assert_eq!(solution.can_finish(3, prereqs.clone()), true);
        
        prereqs.push(vec![0, 2]); // This creates a cycle
        assert_eq!(solution.can_finish(3, prereqs), false);
    }

    #[test]
    fn test_course_numbering() {
        let solution = setup();
        
        // Courses numbered from 0 to n-1
        let prerequisites = vec![
            vec![0, 1], vec![1, 2], vec![2, 3]  // 1->0, 2->1, 3->2
        ];
        assert_eq!(solution.can_finish(4, prerequisites), true);
        
        // Different ordering should work the same
        let prerequisites = vec![
            vec![2, 3], vec![1, 2], vec![0, 1]  // Same dependencies, different order
        ];
        assert_eq!(solution.can_finish(4, prerequisites), true);
    }
}