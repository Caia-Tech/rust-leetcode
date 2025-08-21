//! Problem 133: Clone Graph
//!
//! Given a reference of a node in a connected undirected graph.
//! Return a deep copy (clone) of the graph.
//!
//! Each node in the graph contains a value (int) and a list (Vec<Rc<RefCell<Node>>>) of its neighbors.
//!
//! The graph is represented in the test case using an adjacency list.
//! An adjacency list is a collection of unordered lists used to represent a finite graph.
//! Each list describes the set of neighbors of a node in the graph.
//!
//! The given node will always be the first node with val = 1.
//! You must return the copy of the given node as a reference to the cloned graph.
//!
//! Constraints:
//! - The number of nodes in the graph is in the range [0, 100].
//! - 1 <= Node.val <= 100
//! - Node.val is unique for each node.
//! - There are no repeated edges and no self-loops in the graph.
//! - The Graph is connected and all nodes can be visited starting from the given node.

use std::rc::Rc;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone)]
pub struct Node {
    pub val: i32,
    pub neighbors: RefCell<Vec<Rc<Node>>>,
}

impl Node {
    pub fn new(val: i32) -> Self {
        Node {
            val,
            neighbors: RefCell::new(Vec::new()),
        }
    }
    
    pub fn add_neighbor(&self, neighbor: Rc<Node>) {
        self.neighbors.borrow_mut().push(neighbor);
    }
}

pub struct Solution;

impl Solution {
    /// Approach 1: DFS with HashMap
    /// 
    /// Uses DFS to traverse the graph and a HashMap to track cloned nodes.
    /// This prevents infinite loops and ensures each node is cloned exactly once.
    /// 
    /// Time Complexity: O(N + E) where N is nodes and E is edges
    /// Space Complexity: O(N) for the HashMap and recursion stack
    pub fn clone_graph_dfs(node: Option<Rc<Node>>) -> Option<Rc<Node>> {
        node.as_ref().map(|n| {
            let mut visited = HashMap::new();
            Self::dfs_clone(n, &mut visited)
        })
    }
    
    fn dfs_clone(node: &Rc<Node>, visited: &mut HashMap<i32, Rc<Node>>) -> Rc<Node> {
        // If already cloned, return the clone
        if let Some(cloned) = visited.get(&node.val) {
            return cloned.clone();
        }
        
        // Create new node
        let cloned = Rc::new(Node::new(node.val));
        visited.insert(node.val, cloned.clone());
        
        // Clone all neighbors
        for neighbor in node.neighbors.borrow().iter() {
            let cloned_neighbor = Self::dfs_clone(neighbor, visited);
            cloned.add_neighbor(cloned_neighbor);
        }
        
        cloned
    }
    
    /// Approach 2: BFS with Queue
    /// 
    /// Uses BFS to traverse the graph level by level.
    /// Maintains a queue for traversal and HashMap for tracking clones.
    /// 
    /// Time Complexity: O(N + E)
    /// Space Complexity: O(N)
    pub fn clone_graph_bfs(node: Option<Rc<Node>>) -> Option<Rc<Node>> {
        node.as_ref().map(|n| {
            let mut visited = HashMap::new();
            let mut queue = VecDeque::new();
            
            // Create clone of starting node
            let cloned = Rc::new(Node::new(n.val));
            visited.insert(n.val, cloned.clone());
            queue.push_back(n.clone());
            
            // BFS traversal
            while let Some(current) = queue.pop_front() {
                let current_clone = visited.get(&current.val).unwrap().clone();
                
                for neighbor in current.neighbors.borrow().iter() {
                    let neighbor_clone = visited.entry(neighbor.val)
                        .or_insert_with(|| {
                            queue.push_back(neighbor.clone());
                            Rc::new(Node::new(neighbor.val))
                        })
                        .clone();
                    
                    current_clone.add_neighbor(neighbor_clone);
                }
            }
            
            cloned
        })
    }
    
    /// Approach 3: Iterative DFS with Stack
    /// 
    /// Uses an explicit stack instead of recursion for DFS.
    /// Helpful for avoiding stack overflow on very deep graphs.
    /// 
    /// Time Complexity: O(N + E)
    /// Space Complexity: O(N)
    pub fn clone_graph_iterative_dfs(node: Option<Rc<Node>>) -> Option<Rc<Node>> {
        node.as_ref().map(|n| {
            let mut visited = HashMap::new();
            let mut stack = vec![n.clone()];
            
            // Create clone of starting node
            let cloned = Rc::new(Node::new(n.val));
            visited.insert(n.val, cloned.clone());
            
            while let Some(current) = stack.pop() {
                let current_clone = visited.get(&current.val).unwrap().clone();
                
                for neighbor in current.neighbors.borrow().iter() {
                    let neighbor_clone = if let Some(existing) = visited.get(&neighbor.val) {
                        existing.clone()
                    } else {
                        let new_clone = Rc::new(Node::new(neighbor.val));
                        visited.insert(neighbor.val, new_clone.clone());
                        stack.push(neighbor.clone());
                        new_clone
                    };
                    
                    // Check if neighbor already added
                    let already_added = current_clone.neighbors.borrow()
                        .iter()
                        .any(|n| n.val == neighbor_clone.val);
                    
                    if !already_added {
                        current_clone.add_neighbor(neighbor_clone);
                    }
                }
            }
            
            cloned
        })
    }
    
    /// Approach 4: Two-Pass Approach
    /// 
    /// First pass: Create all nodes without connections
    /// Second pass: Establish all connections
    /// This separation can be cleaner for complex graphs.
    /// 
    /// Time Complexity: O(N + E)
    /// Space Complexity: O(N)
    pub fn clone_graph_two_pass(node: Option<Rc<Node>>) -> Option<Rc<Node>> {
        node.as_ref().map(|n| {
            let mut node_map = HashMap::new();
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();
            
            // First pass: Create all nodes
            queue.push_back(n.clone());
            visited.insert(n.val);
            
            while let Some(current) = queue.pop_front() {
                node_map.entry(current.val)
                    .or_insert_with(|| Rc::new(Node::new(current.val)));
                
                for neighbor in current.neighbors.borrow().iter() {
                    if visited.insert(neighbor.val) {
                        queue.push_back(neighbor.clone());
                    }
                }
            }
            
            // Second pass: Connect all nodes
            visited.clear();
            queue.push_back(n.clone());
            visited.insert(n.val);
            
            while let Some(current) = queue.pop_front() {
                let current_clone = node_map.get(&current.val).unwrap().clone();
                
                for neighbor in current.neighbors.borrow().iter() {
                    let neighbor_clone = node_map.get(&neighbor.val).unwrap().clone();
                    current_clone.add_neighbor(neighbor_clone);
                    
                    if visited.insert(neighbor.val) {
                        queue.push_back(neighbor.clone());
                    }
                }
            }
            
            node_map.get(&n.val).unwrap().clone()
        })
    }
    
    /// Approach 5: Optimized DFS with Early Termination
    /// 
    /// DFS that optimizes by checking if all nodes have been visited
    /// and terminates early if possible.
    /// 
    /// Time Complexity: O(N + E)
    /// Space Complexity: O(N)
    pub fn clone_graph_optimized_dfs(node: Option<Rc<Node>>) -> Option<Rc<Node>> {
        node.as_ref().map(|n| {
            let mut visited = HashMap::new();
            let total_nodes = Self::count_nodes(n);
            Self::dfs_clone_optimized(n, &mut visited, total_nodes)
        })
    }
    
    fn count_nodes(start: &Rc<Node>) -> usize {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start.clone());
        visited.insert(start.val);
        
        while let Some(current) = queue.pop_front() {
            for neighbor in current.neighbors.borrow().iter() {
                if visited.insert(neighbor.val) {
                    queue.push_back(neighbor.clone());
                }
            }
        }
        
        visited.len()
    }
    
    fn dfs_clone_optimized(
        node: &Rc<Node>,
        visited: &mut HashMap<i32, Rc<Node>>,
        total_nodes: usize,
    ) -> Rc<Node> {
        if let Some(cloned) = visited.get(&node.val) {
            return cloned.clone();
        }
        
        let cloned = Rc::new(Node::new(node.val));
        visited.insert(node.val, cloned.clone());
        
        // Don't skip neighbor connections even if all nodes are visited
        for neighbor in node.neighbors.borrow().iter() {
            let cloned_neighbor = Self::dfs_clone_optimized(neighbor, visited, total_nodes);
            cloned.add_neighbor(cloned_neighbor);
        }
        
        cloned
    }
    
    /// Approach 6: Functional Style with Immutable Structure
    /// 
    /// A more functional approach using immutable data structures
    /// where possible, minimizing side effects.
    /// 
    /// Time Complexity: O(N + E)
    /// Space Complexity: O(N)
    pub fn clone_graph_functional(node: Option<Rc<Node>>) -> Option<Rc<Node>> {
        node.as_ref().map(|n| {
            let mut cache = HashMap::new();
            Self::functional_clone(n, &mut cache)
        })
    }
    
    fn functional_clone(node: &Rc<Node>, cache: &mut HashMap<i32, Rc<Node>>) -> Rc<Node> {
        cache.get(&node.val).cloned().unwrap_or_else(|| {
            let new_node = Rc::new(Node::new(node.val));
            cache.insert(node.val, new_node.clone());
            
            let neighbors = node.neighbors.borrow()
                .iter()
                .map(|n| Self::functional_clone(n, cache))
                .collect::<Vec<_>>();
            
            for neighbor in neighbors {
                new_node.add_neighbor(neighbor);
            }
            
            new_node
        })
    }
}

/// Helper function to build a graph from adjacency list
pub fn build_graph(adj_list: Vec<Vec<i32>>) -> Option<Rc<Node>> {
    if adj_list.is_empty() {
        return None;
    }
    
    let mut nodes: HashMap<i32, Rc<Node>> = HashMap::new();
    
    // Create all nodes
    for (i, _) in adj_list.iter().enumerate() {
        nodes.insert((i + 1) as i32, Rc::new(Node::new((i + 1) as i32)));
    }
    
    // Connect neighbors
    for (i, neighbors) in adj_list.iter().enumerate() {
        let node = nodes.get(&((i + 1) as i32)).unwrap();
        for &neighbor_val in neighbors {
            if let Some(neighbor) = nodes.get(&neighbor_val) {
                node.add_neighbor(neighbor.clone());
            }
        }
    }
    
    nodes.get(&1).cloned()
}

/// Helper function to convert graph to adjacency list
pub fn graph_to_adj_list(node: Option<Rc<Node>>) -> Vec<Vec<i32>> {
    if node.is_none() {
        return vec![];
    }
    
    let mut visited = HashMap::new();
    let mut queue = VecDeque::new();
    
    let start = node.unwrap();
    queue.push_back(start.clone());
    visited.insert(start.val, start.neighbors.borrow().clone());
    
    while let Some(current) = queue.pop_front() {
        for neighbor in current.neighbors.borrow().iter() {
            if !visited.contains_key(&neighbor.val) {
                visited.insert(neighbor.val, neighbor.neighbors.borrow().clone());
                queue.push_back(neighbor.clone());
            }
        }
    }
    
    let max_val = visited.keys().max().copied().unwrap_or(0);
    let mut result = vec![vec![]; max_val as usize];
    
    for (val, neighbors) in visited {
        result[(val - 1) as usize] = neighbors.iter().map(|n| n.val).collect();
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn verify_clone(original: Option<Rc<Node>>, cloned: Option<Rc<Node>>) -> bool {
        match (original, cloned) {
            (None, None) => true,
            (Some(_), None) | (None, Some(_)) => false,
            (Some(orig), Some(clone)) => {
                // Convert both to adjacency lists and compare
                let orig_adj = graph_to_adj_list(Some(orig));
                let clone_adj = graph_to_adj_list(Some(clone));
                orig_adj == clone_adj
            }
        }
    }
    
    #[test]
    fn test_simple_graph() {
        // Graph: 1 -- 2
        //        |    |
        //        4 -- 3
        let adj_list = vec![
            vec![2, 4],
            vec![1, 3],
            vec![2, 4],
            vec![1, 3],
        ];
        let graph = build_graph(adj_list.clone());
        
        let clone1 = Solution::clone_graph_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone1));
        
        let clone2 = Solution::clone_graph_bfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone2));
        
        let clone3 = Solution::clone_graph_iterative_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone3));
        
        let clone4 = Solution::clone_graph_two_pass(graph.clone());
        assert!(verify_clone(graph.clone(), clone4));
        
        let clone5 = Solution::clone_graph_optimized_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone5));
        
        let clone6 = Solution::clone_graph_functional(graph.clone());
        assert!(verify_clone(graph.clone(), clone6));
    }
    
    #[test]
    fn test_single_node() {
        let adj_list = vec![vec![]];
        let graph = build_graph(adj_list);
        
        let clone1 = Solution::clone_graph_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone1));
        
        let clone2 = Solution::clone_graph_bfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone2));
        
        let clone3 = Solution::clone_graph_iterative_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone3));
        
        let clone4 = Solution::clone_graph_two_pass(graph.clone());
        assert!(verify_clone(graph.clone(), clone4));
        
        let clone5 = Solution::clone_graph_optimized_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone5));
        
        let clone6 = Solution::clone_graph_functional(graph.clone());
        assert!(verify_clone(graph.clone(), clone6));
    }
    
    #[test]
    fn test_empty_graph() {
        let graph = None;
        
        let clone1 = Solution::clone_graph_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone1));
        
        let clone2 = Solution::clone_graph_bfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone2));
        
        let clone3 = Solution::clone_graph_iterative_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone3));
        
        let clone4 = Solution::clone_graph_two_pass(graph.clone());
        assert!(verify_clone(graph.clone(), clone4));
        
        let clone5 = Solution::clone_graph_optimized_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone5));
        
        let clone6 = Solution::clone_graph_functional(graph.clone());
        assert!(verify_clone(graph.clone(), clone6));
    }
    
    #[test]
    fn test_linear_graph() {
        // Graph: 1 -- 2 -- 3
        let adj_list = vec![
            vec![2],
            vec![1, 3],
            vec![2],
        ];
        let graph = build_graph(adj_list);
        
        let clone1 = Solution::clone_graph_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone1));
        
        let clone2 = Solution::clone_graph_bfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone2));
        
        let clone3 = Solution::clone_graph_iterative_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone3));
        
        let clone4 = Solution::clone_graph_two_pass(graph.clone());
        assert!(verify_clone(graph.clone(), clone4));
        
        let clone5 = Solution::clone_graph_optimized_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone5));
        
        let clone6 = Solution::clone_graph_functional(graph.clone());
        assert!(verify_clone(graph.clone(), clone6));
    }
    
    #[test]
    fn test_star_graph() {
        // Graph: Center node connected to all others
        //        2
        //        |
        //    3-- 1 --4
        //        |
        //        5
        let adj_list = vec![
            vec![2, 3, 4, 5],
            vec![1],
            vec![1],
            vec![1],
            vec![1],
        ];
        let graph = build_graph(adj_list);
        
        let clone1 = Solution::clone_graph_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone1));
        
        let clone2 = Solution::clone_graph_bfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone2));
        
        let clone3 = Solution::clone_graph_iterative_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone3));
        
        let clone4 = Solution::clone_graph_two_pass(graph.clone());
        assert!(verify_clone(graph.clone(), clone4));
        
        let clone5 = Solution::clone_graph_optimized_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone5));
        
        let clone6 = Solution::clone_graph_functional(graph.clone());
        assert!(verify_clone(graph.clone(), clone6));
    }
    
    #[test]
    fn test_complete_graph() {
        // Complete graph K4: Every node connected to every other node
        let adj_list = vec![
            vec![2, 3, 4],
            vec![1, 3, 4],
            vec![1, 2, 4],
            vec![1, 2, 3],
        ];
        let graph = build_graph(adj_list);
        
        let clone1 = Solution::clone_graph_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone1));
        
        let clone2 = Solution::clone_graph_bfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone2));
        
        let clone3 = Solution::clone_graph_iterative_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone3));
        
        let clone4 = Solution::clone_graph_two_pass(graph.clone());
        assert!(verify_clone(graph.clone(), clone4));
        
        let clone5 = Solution::clone_graph_optimized_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone5));
        
        let clone6 = Solution::clone_graph_functional(graph.clone());
        assert!(verify_clone(graph.clone(), clone6));
    }
    
    #[test]
    fn test_tree_structure() {
        // Binary tree structure (as undirected graph)
        //       1
        //      / \
        //     2   3
        //    / \
        //   4   5
        let adj_list = vec![
            vec![2, 3],
            vec![1, 4, 5],
            vec![1],
            vec![2],
            vec![2],
        ];
        let graph = build_graph(adj_list);
        
        let clone1 = Solution::clone_graph_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone1));
        
        let clone2 = Solution::clone_graph_bfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone2));
        
        let clone3 = Solution::clone_graph_iterative_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone3));
        
        let clone4 = Solution::clone_graph_two_pass(graph.clone());
        assert!(verify_clone(graph.clone(), clone4));
        
        let clone5 = Solution::clone_graph_optimized_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone5));
        
        let clone6 = Solution::clone_graph_functional(graph.clone());
        assert!(verify_clone(graph.clone(), clone6));
    }
    
    #[test]
    fn test_cyclic_graph() {
        // Graph with multiple cycles
        //   1 -- 2
        //   |  X |
        //   3 -- 4
        let adj_list = vec![
            vec![2, 3, 4],
            vec![1, 3, 4],
            vec![1, 2, 4],
            vec![1, 2, 3],
        ];
        let graph = build_graph(adj_list);
        
        let clone1 = Solution::clone_graph_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone1));
        
        let clone2 = Solution::clone_graph_bfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone2));
        
        let clone3 = Solution::clone_graph_iterative_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone3));
        
        let clone4 = Solution::clone_graph_two_pass(graph.clone());
        assert!(verify_clone(graph.clone(), clone4));
        
        let clone5 = Solution::clone_graph_optimized_dfs(graph.clone());
        assert!(verify_clone(graph.clone(), clone5));
        
        let clone6 = Solution::clone_graph_functional(graph.clone());
        assert!(verify_clone(graph.clone(), clone6));
    }
    
    #[test]
    fn test_deep_references() {
        // Test that cloned graph has no shared references with original
        let adj_list = vec![
            vec![2, 3],
            vec![1, 3],
            vec![1, 2],
        ];
        let graph = build_graph(adj_list);
        
        let clone = Solution::clone_graph_dfs(graph.clone());
        
        // Verify they're different objects
        if let (Some(orig), Some(cloned)) = (graph, clone) {
            assert!(!Rc::ptr_eq(&orig, &cloned));
        }
    }
    
    #[test]
    fn test_large_graph() {
        // Create a larger graph for performance testing
        let mut adj_list = vec![vec![]; 20];
        
        // Create a grid-like structure
        for i in 0..20 {
            if i > 0 {
                adj_list[i].push(i as i32);
            }
            if i < 19 {
                adj_list[i].push((i + 2) as i32);
            }
            if i % 5 != 0 {
                adj_list[i].push((i) as i32);
            }
            if i % 5 != 4 {
                adj_list[i].push((i + 2) as i32);
            }
        }
        
        let graph = build_graph(adj_list);
        
        // Test all approaches handle larger graphs
        let clone1 = Solution::clone_graph_dfs(graph.clone());
        assert!(clone1.is_some());
        
        let clone2 = Solution::clone_graph_bfs(graph.clone());
        assert!(clone2.is_some());
        
        let clone3 = Solution::clone_graph_iterative_dfs(graph.clone());
        assert!(clone3.is_some());
        
        let clone4 = Solution::clone_graph_two_pass(graph.clone());
        assert!(clone4.is_some());
        
        let clone5 = Solution::clone_graph_optimized_dfs(graph.clone());
        assert!(clone5.is_some());
        
        let clone6 = Solution::clone_graph_functional(graph.clone());
        assert!(clone6.is_some());
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            vec![vec![2, 4], vec![1, 3], vec![2, 4], vec![1, 3]],
            vec![vec![2], vec![1]],
            vec![vec![]],
            vec![vec![2, 3, 4], vec![1, 3, 4], vec![1, 2, 4], vec![1, 2, 3]],
        ];
        
        for adj_list in test_cases {
            let graph = build_graph(adj_list);
            
            let clone1 = Solution::clone_graph_dfs(graph.clone());
            let clone2 = Solution::clone_graph_bfs(graph.clone());
            let clone3 = Solution::clone_graph_iterative_dfs(graph.clone());
            let clone4 = Solution::clone_graph_two_pass(graph.clone());
            let clone5 = Solution::clone_graph_optimized_dfs(graph.clone());
            let clone6 = Solution::clone_graph_functional(graph.clone());
            
            // All clones should be equivalent
            let adj1 = graph_to_adj_list(clone1);
            let adj2 = graph_to_adj_list(clone2);
            let adj3 = graph_to_adj_list(clone3);
            let adj4 = graph_to_adj_list(clone4);
            let adj5 = graph_to_adj_list(clone5);
            let adj6 = graph_to_adj_list(clone6);
            
            assert_eq!(adj1, adj2);
            assert_eq!(adj2, adj3);
            assert_eq!(adj3, adj4);
            assert_eq!(adj4, adj5);
            assert_eq!(adj5, adj6);
        }
    }
}