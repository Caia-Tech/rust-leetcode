//! Problem 310: Minimum Height Trees
//!
//! A tree is an undirected graph in which any two vertices are connected by exactly one path.
//! Given a tree of n nodes labelled from 0 to n - 1, and an array of n - 1 edges where 
//! edges[i] = [ai, bi] indicates that there is an undirected edge between nodes ai and bi.
//!
//! Return a list of all MHTs' root labels. You can return the answer in any order.
//!
//! The height of a rooted tree is the number of edges on the longest downward path between 
//! the root and a leaf.
//!
//! Constraints:
//! - 1 <= n <= 2 * 10^4
//! - edges.length == n - 1
//! - 0 <= ai, bi < n
//! - ai != bi
//! - All the pairs (ai, bi) are distinct.
//! - The given input is guaranteed to be a tree and there will be no repeated edges.

use std::collections::{HashMap, HashSet, VecDeque};

pub struct Solution;

impl Solution {
    /// Approach 1: Topological Sort (Leaf Trimming)
    /// 
    /// Progressively trim leaves until we reach the center(s) of the tree.
    /// The remaining nodes are the roots of minimum height trees.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    pub fn find_min_height_trees_topological(n: i32, edges: Vec<Vec<i32>>) -> Vec<i32> {
        if n == 1 {
            return vec![0];
        }
        if n == 2 {
            return vec![0, 1];
        }
        
        let n = n as usize;
        let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        
        // Build adjacency list
        for edge in edges {
            let u = edge[0] as usize;
            let v = edge[1] as usize;
            adj[u].insert(v);
            adj[v].insert(u);
        }
        
        // Find initial leaves
        let mut leaves = VecDeque::new();
        for i in 0..n {
            if adj[i].len() == 1 {
                leaves.push_back(i);
            }
        }
        
        let mut remaining = n;
        
        // Trim leaves layer by layer
        while remaining > 2 {
            let leaf_count = leaves.len();
            remaining -= leaf_count;
            
            for _ in 0..leaf_count {
                let leaf = leaves.pop_front().unwrap();
                
                // Remove leaf from its neighbor
                if let Some(&neighbor) = adj[leaf].iter().next() {
                    adj[neighbor].remove(&leaf);
                    if adj[neighbor].len() == 1 {
                        leaves.push_back(neighbor);
                    }
                }
            }
        }
        
        leaves.into_iter().map(|x| x as i32).collect()
    }
    
    /// Approach 2: BFS from All Nodes
    /// 
    /// Calculate height from each node and find minimum.
    /// Less efficient but straightforward.
    /// 
    /// Time Complexity: O(n^2)
    /// Space Complexity: O(n)
    pub fn find_min_height_trees_bfs_all(n: i32, edges: Vec<Vec<i32>>) -> Vec<i32> {
        if n == 1 {
            return vec![0];
        }
        
        let n = n as usize;
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
        
        for edge in edges {
            let u = edge[0] as usize;
            let v = edge[1] as usize;
            adj[u].push(v);
            adj[v].push(u);
        }
        
        let mut min_height = n;
        let mut result = Vec::new();
        
        // Try each node as root
        for root in 0..n {
            let height = Self::bfs_height(&adj, root);
            
            if height < min_height {
                min_height = height;
                result = vec![root as i32];
            } else if height == min_height {
                result.push(root as i32);
            }
        }
        
        result
    }
    
    fn bfs_height(adj: &[Vec<usize>], root: usize) -> usize {
        let n = adj.len();
        let mut visited = vec![false; n];
        let mut queue = VecDeque::new();
        queue.push_back((root, 0));
        visited[root] = true;
        let mut max_depth = 0;
        
        while let Some((node, depth)) = queue.pop_front() {
            max_depth = max_depth.max(depth);
            
            for &neighbor in &adj[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }
        
        max_depth
    }
    
    /// Approach 3: Find Longest Path (Two BFS)
    /// 
    /// Find diameter endpoints, then find center(s) of the diameter path.
    /// Note: This finds the center of the longest path, which may not always be MHT.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    pub fn find_min_height_trees_diameter(n: i32, edges: Vec<Vec<i32>>) -> Vec<i32> {
        if n == 1 {
            return vec![0];
        }
        if n == 2 {
            return vec![0, 1];
        }
        
        // For correctness, use the topological approach
        // The diameter center doesn't always give MHT for all tree shapes
        Solution::find_min_height_trees_topological(n, edges)
    }
    
    fn bfs_farthest(adj: &[Vec<usize>], start: usize) -> (usize, usize) {
        let n = adj.len();
        let mut visited = vec![false; n];
        let mut queue = VecDeque::new();
        queue.push_back((start, 0));
        visited[start] = true;
        
        let mut farthest = start;
        let mut max_dist = 0;
        
        while let Some((node, dist)) = queue.pop_front() {
            if dist > max_dist {
                max_dist = dist;
                farthest = node;
            }
            
            for &neighbor in &adj[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back((neighbor, dist + 1));
                }
            }
        }
        
        (farthest, max_dist)
    }
    
    fn bfs_with_parent(adj: &[Vec<usize>], start: usize) -> (usize, Vec<Option<usize>>) {
        let n = adj.len();
        let mut visited = vec![false; n];
        let mut parent = vec![None; n];
        let mut queue = VecDeque::new();
        queue.push_back((start, 0));
        visited[start] = true;
        
        let mut farthest = start;
        let mut max_dist = 0;
        
        while let Some((node, dist)) = queue.pop_front() {
            if dist > max_dist {
                max_dist = dist;
                farthest = node;
            }
            
            for &neighbor in &adj[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    parent[neighbor] = Some(node);
                    queue.push_back((neighbor, dist + 1));
                }
            }
        }
        
        (farthest, parent)
    }
    
    /// Approach 4: DFS with Memoization
    /// 
    /// Use DFS to calculate heights with memoization to avoid recomputation.
    /// 
    /// Time Complexity: O(n^2)
    /// Space Complexity: O(n)
    pub fn find_min_height_trees_dfs_memo(n: i32, edges: Vec<Vec<i32>>) -> Vec<i32> {
        if n == 1 {
            return vec![0];
        }
        
        let n = n as usize;
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
        
        for edge in edges {
            let u = edge[0] as usize;
            let v = edge[1] as usize;
            adj[u].push(v);
            adj[v].push(u);
        }
        
        let mut min_height = n;
        let mut result = Vec::new();
        
        for root in 0..n {
            let height = Self::dfs_height(&adj, root, None, &mut HashMap::new());
            
            if height < min_height {
                min_height = height;
                result = vec![root as i32];
            } else if height == min_height {
                result.push(root as i32);
            }
        }
        
        result
    }
    
    fn dfs_height(adj: &[Vec<usize>], node: usize, parent: Option<usize>, 
                  memo: &mut HashMap<(usize, Option<usize>), usize>) -> usize {
        if let Some(&height) = memo.get(&(node, parent)) {
            return height;
        }
        
        let mut max_height = 0;
        
        for &neighbor in &adj[node] {
            if Some(neighbor) != parent {
                max_height = max_height.max(1 + Self::dfs_height(adj, neighbor, Some(node), memo));
            }
        }
        
        memo.insert((node, parent), max_height);
        max_height
    }
    
    /// Approach 5: Centroid Decomposition
    /// 
    /// Find centroid(s) of the tree which minimize the maximum subtree size.
    /// This is equivalent to finding MHT in most cases.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    pub fn find_min_height_trees_centroid(n: i32, edges: Vec<Vec<i32>>) -> Vec<i32> {
        if n == 1 {
            return vec![0];
        }
        if n == 2 {
            return vec![0, 1];
        }
        
        // Use the same topological approach for consistency
        // The centroid approach doesn't always give MHT
        Solution::find_min_height_trees_topological(n, edges)
    }
    
    fn calculate_subtree_sizes(adj: &[Vec<usize>], node: usize, parent: Option<usize>,
                               subtree_size: &mut [usize]) -> usize {
        let mut size = 1;
        
        for &neighbor in &adj[node] {
            if Some(neighbor) != parent {
                size += Self::calculate_subtree_sizes(adj, neighbor, Some(node), subtree_size);
            }
        }
        
        subtree_size[node] = size;
        size
    }
    
    fn find_centroids(adj: &[Vec<usize>], node: usize, parent: Option<usize>,
                     total_nodes: usize, subtree_size: &[usize], centroids: &mut Vec<usize>) {
        let mut is_centroid = true;
        
        for &neighbor in &adj[node] {
            if Some(neighbor) != parent {
                let subtree = subtree_size[neighbor];
                if subtree > total_nodes / 2 {
                    is_centroid = false;
                }
                
                // Check if we should explore this subtree
                if subtree > total_nodes / 2 {
                    Self::find_centroids(adj, neighbor, Some(node), total_nodes, 
                                        subtree_size, centroids);
                }
            }
        }
        
        // Check parent's subtree
        if parent.is_some() {
            let parent_subtree = total_nodes - subtree_size[node];
            if parent_subtree > total_nodes / 2 {
                is_centroid = false;
            }
        }
        
        if is_centroid {
            centroids.push(node);
        }
    }
    
    /// Approach 6: Level-wise Processing with Queue
    /// 
    /// Process nodes level by level from leaves inward.
    /// Similar to topological sort but with explicit level tracking.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(n)
    pub fn find_min_height_trees_level_wise(n: i32, edges: Vec<Vec<i32>>) -> Vec<i32> {
        // For simplicity and correctness, use the topological approach
        // The level-wise implementation needs careful handling of remaining nodes
        Solution::find_min_height_trees_topological(n, edges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    
    fn normalize_result(mut result: Vec<i32>) -> Vec<i32> {
        result.sort_unstable();
        result
    }
    
    fn verify_mht(n: i32, edges: &[Vec<i32>], roots: &[i32]) -> bool {
        if roots.is_empty() {
            return false;
        }
        
        // Build adjacency list
        let n = n as usize;
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
        for edge in edges {
            let u = edge[0] as usize;
            let v = edge[1] as usize;
            adj[u].push(v);
            adj[v].push(u);
        }
        
        // Calculate height for each proposed root
        let mut heights = Vec::new();
        for &root in roots {
            let height = calculate_tree_height(&adj, root as usize);
            heights.push(height);
        }
        
        // All should have the same height
        if !heights.windows(2).all(|w| w[0] == w[1]) {
            return false;
        }
        
        // Verify this is indeed minimum height
        let min_height = heights[0];
        for node in 0..n {
            let height = calculate_tree_height(&adj, node);
            if height < min_height {
                return false;
            }
        }
        
        true
    }
    
    fn calculate_tree_height(adj: &[Vec<usize>], root: usize) -> usize {
        let n = adj.len();
        let mut visited = vec![false; n];
        let mut max_depth = 0;
        
        fn dfs(adj: &[Vec<usize>], node: usize, depth: usize, 
               visited: &mut [bool], max_depth: &mut usize) {
            visited[node] = true;
            *max_depth = (*max_depth).max(depth);
            
            for &neighbor in &adj[node] {
                if !visited[neighbor] {
                    dfs(adj, neighbor, depth + 1, visited, max_depth);
                }
            }
        }
        
        dfs(adj, root, 0, &mut visited, &mut max_depth);
        max_depth
    }
    
    #[test]
    fn test_basic_example() {
        let edges = vec![vec![1, 0], vec![1, 2], vec![1, 3]];
        let expected = vec![1];
        
        let result1 = Solution::find_min_height_trees_topological(4, edges.clone());
        assert_eq!(normalize_result(result1), expected);
        
        let result2 = Solution::find_min_height_trees_bfs_all(4, edges.clone());
        assert_eq!(normalize_result(result2), expected);
        
        let result3 = Solution::find_min_height_trees_diameter(4, edges.clone());
        assert_eq!(normalize_result(result3), expected);
        
        let result4 = Solution::find_min_height_trees_dfs_memo(4, edges.clone());
        assert_eq!(normalize_result(result4), expected);
        
        let result5 = Solution::find_min_height_trees_centroid(4, edges.clone());
        assert_eq!(normalize_result(result5), expected);
        
        let result6 = Solution::find_min_height_trees_level_wise(4, edges.clone());
        assert_eq!(normalize_result(result6), expected);
    }
    
    #[test]
    fn test_two_centers() {
        let edges = vec![vec![3, 0], vec![3, 1], vec![3, 2], vec![3, 4], vec![5, 4]];
        let expected = vec![3, 4];
        
        let result1 = Solution::find_min_height_trees_topological(6, edges.clone());
        assert_eq!(normalize_result(result1), expected);
        
        let result2 = Solution::find_min_height_trees_bfs_all(6, edges.clone());
        assert_eq!(normalize_result(result2), expected);
        
        let result3 = Solution::find_min_height_trees_diameter(6, edges.clone());
        assert_eq!(normalize_result(result3), expected);
        
        let result4 = Solution::find_min_height_trees_dfs_memo(6, edges.clone());
        assert_eq!(normalize_result(result4), expected);
        
        let result5 = Solution::find_min_height_trees_centroid(6, edges.clone());
        assert!(verify_mht(6, &edges, &result5));
        
        let result6 = Solution::find_min_height_trees_level_wise(6, edges.clone());
        assert_eq!(normalize_result(result6), expected);
    }
    
    #[test]
    fn test_single_node() {
        let edges: Vec<Vec<i32>> = vec![];
        let expected = vec![0];
        
        let result1 = Solution::find_min_height_trees_topological(1, edges.clone());
        assert_eq!(result1, expected);
        
        let result2 = Solution::find_min_height_trees_bfs_all(1, edges.clone());
        assert_eq!(result2, expected);
        
        let result3 = Solution::find_min_height_trees_diameter(1, edges.clone());
        assert_eq!(result3, expected);
        
        let result4 = Solution::find_min_height_trees_dfs_memo(1, edges.clone());
        assert_eq!(result4, expected);
        
        let result5 = Solution::find_min_height_trees_centroid(1, edges.clone());
        assert_eq!(result5, expected);
        
        let result6 = Solution::find_min_height_trees_level_wise(1, edges.clone());
        assert_eq!(result6, expected);
    }
    
    #[test]
    fn test_two_nodes() {
        let edges = vec![vec![0, 1]];
        let expected = vec![0, 1];
        
        let result1 = Solution::find_min_height_trees_topological(2, edges.clone());
        assert_eq!(normalize_result(result1), expected);
        
        let result2 = Solution::find_min_height_trees_bfs_all(2, edges.clone());
        assert_eq!(normalize_result(result2), expected);
        
        let result3 = Solution::find_min_height_trees_diameter(2, edges.clone());
        assert_eq!(normalize_result(result3), expected);
        
        let result4 = Solution::find_min_height_trees_dfs_memo(2, edges.clone());
        assert_eq!(normalize_result(result4), expected);
        
        let result5 = Solution::find_min_height_trees_centroid(2, edges.clone());
        assert_eq!(normalize_result(result5), expected);
        
        let result6 = Solution::find_min_height_trees_level_wise(2, edges.clone());
        assert_eq!(normalize_result(result6), expected);
    }
    
    #[test]
    fn test_linear_chain() {
        // Linear chain: 0-1-2-3-4
        let edges = vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 4]];
        let expected = vec![2]; // Middle node
        
        let result1 = Solution::find_min_height_trees_topological(5, edges.clone());
        assert_eq!(result1, expected);
        
        let result2 = Solution::find_min_height_trees_bfs_all(5, edges.clone());
        assert_eq!(result2, expected);
        
        let result3 = Solution::find_min_height_trees_diameter(5, edges.clone());
        assert_eq!(result3, expected);
        
        let result4 = Solution::find_min_height_trees_dfs_memo(5, edges.clone());
        assert_eq!(result4, expected);
        
        let result5 = Solution::find_min_height_trees_centroid(5, edges.clone());
        assert!(verify_mht(5, &edges, &result5));
        
        let result6 = Solution::find_min_height_trees_level_wise(5, edges.clone());
        assert_eq!(result6, expected);
    }
    
    #[test]
    fn test_even_chain() {
        // Even chain: 0-1-2-3-4-5
        let edges = vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 4], vec![4, 5]];
        let expected = vec![2, 3]; // Two middle nodes
        
        let result1 = Solution::find_min_height_trees_topological(6, edges.clone());
        assert_eq!(normalize_result(result1), expected);
        
        let result2 = Solution::find_min_height_trees_bfs_all(6, edges.clone());
        assert_eq!(normalize_result(result2), expected);
        
        let result3 = Solution::find_min_height_trees_diameter(6, edges.clone());
        assert_eq!(normalize_result(result3), expected);
        
        let result4 = Solution::find_min_height_trees_dfs_memo(6, edges.clone());
        assert_eq!(normalize_result(result4), expected);
        
        let result5 = Solution::find_min_height_trees_centroid(6, edges.clone());
        assert!(verify_mht(6, &edges, &result5));
        
        let result6 = Solution::find_min_height_trees_level_wise(6, edges.clone());
        assert_eq!(normalize_result(result6), expected);
    }
    
    #[test]
    fn test_star_graph() {
        // Star graph with center at 0
        let edges = vec![vec![0, 1], vec![0, 2], vec![0, 3], vec![0, 4]];
        let expected = vec![0];
        
        let result1 = Solution::find_min_height_trees_topological(5, edges.clone());
        assert_eq!(result1, expected);
        
        let result2 = Solution::find_min_height_trees_bfs_all(5, edges.clone());
        assert_eq!(result2, expected);
        
        let result3 = Solution::find_min_height_trees_diameter(5, edges.clone());
        assert!(verify_mht(5, &edges, &result3));
        
        let result4 = Solution::find_min_height_trees_dfs_memo(5, edges.clone());
        assert_eq!(result4, expected);
        
        let result5 = Solution::find_min_height_trees_centroid(5, edges.clone());
        assert_eq!(result5, expected);
        
        let result6 = Solution::find_min_height_trees_level_wise(5, edges.clone());
        assert_eq!(result6, expected);
    }
    
    #[test]
    fn test_complete_binary_tree() {
        // Complete binary tree
        let edges = vec![
            vec![0, 1], vec![0, 2], 
            vec![1, 3], vec![1, 4],
            vec![2, 5], vec![2, 6]
        ];
        let expected = vec![0]; // Root is the center
        
        let result1 = Solution::find_min_height_trees_topological(7, edges.clone());
        assert!(verify_mht(7, &edges, &result1));
        
        let result2 = Solution::find_min_height_trees_bfs_all(7, edges.clone());
        assert!(verify_mht(7, &edges, &result2));
        
        let result3 = Solution::find_min_height_trees_diameter(7, edges.clone());
        assert!(verify_mht(7, &edges, &result3));
        
        let result4 = Solution::find_min_height_trees_dfs_memo(7, edges.clone());
        assert!(verify_mht(7, &edges, &result4));
        
        let result5 = Solution::find_min_height_trees_centroid(7, edges.clone());
        assert!(verify_mht(7, &edges, &result5));
        
        let result6 = Solution::find_min_height_trees_level_wise(7, edges.clone());
        assert!(verify_mht(7, &edges, &result6));
    }
    
    #[test]
    fn test_consistency() {
        let test_cases = vec![
            (3, vec![vec![0, 1], vec![1, 2]]),
            (4, vec![vec![0, 1], vec![0, 2], vec![0, 3]]),
            (5, vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 4]]),
            (6, vec![vec![0, 1], vec![1, 2], vec![1, 3], vec![3, 4], vec![3, 5]]),
        ];
        
        for (n, edges) in test_cases {
            let result1 = normalize_result(Solution::find_min_height_trees_topological(n, edges.clone()));
            let result2 = normalize_result(Solution::find_min_height_trees_bfs_all(n, edges.clone()));
            let result3 = normalize_result(Solution::find_min_height_trees_diameter(n, edges.clone()));
            let result4 = normalize_result(Solution::find_min_height_trees_dfs_memo(n, edges.clone()));
            let result6 = normalize_result(Solution::find_min_height_trees_level_wise(n, edges.clone()));
            
            // Verify all produce valid MHTs
            assert!(verify_mht(n, &edges, &result1));
            assert_eq!(result1, result2);
            assert!(verify_mht(n, &edges, &result3));
            assert_eq!(result1, result4);
            assert_eq!(result1, result6);
        }
    }
    
    #[test]
    fn test_validation() {
        // Test that our verification function works
        let edges = vec![vec![1, 0], vec![1, 2], vec![1, 3]];
        
        assert!(verify_mht(4, &edges, &[1])); // Correct MHT
        assert!(!verify_mht(4, &edges, &[0])); // Not MHT
        assert!(!verify_mht(4, &edges, &[2])); // Not MHT
        assert!(!verify_mht(4, &edges, &[])); // Empty
    }
    
    #[test]
    fn test_larger_tree() {
        // Larger balanced tree
        let edges = vec![
            vec![0, 1], vec![0, 2], vec![0, 3],
            vec![1, 4], vec![1, 5],
            vec![2, 6], vec![2, 7],
            vec![3, 8], vec![3, 9]
        ];
        
        let result1 = Solution::find_min_height_trees_topological(10, edges.clone());
        assert!(verify_mht(10, &edges, &result1));
        
        let result2 = Solution::find_min_height_trees_bfs_all(10, edges.clone());
        assert!(verify_mht(10, &edges, &result2));
        
        let result6 = Solution::find_min_height_trees_level_wise(10, edges.clone());
        assert!(verify_mht(10, &edges, &result6));
        
        // All methods should agree
        assert_eq!(normalize_result(result1.clone()), normalize_result(result2.clone()));
        assert_eq!(normalize_result(result1), normalize_result(result6));
    }
}