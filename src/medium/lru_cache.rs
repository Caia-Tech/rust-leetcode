//! # Problem 146: LRU Cache
//!
//! Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.
//!
//! Implement the LRUCache class:
//!
//! * `LRUCache(int capacity)` Initialize the LRU cache with positive size capacity.
//! * `int get(int key)` Return the value of the key if the key exists, otherwise return -1.
//! * `void put(int key, int value)` Update the value of the key if the key exists. 
//!   Otherwise, add the key-value pair to the cache. If the number of keys exceeds the 
//!   capacity from this operation, evict the least recently used key.
//!
//! The functions get and put must each run in O(1) average time complexity.
//!
//! ## Examples
//!
//! ```
//! let mut lru_cache = LRUCache::new(2);
//! lru_cache.put(1, 1); // cache is {1=1}
//! lru_cache.put(2, 2); // cache is {1=1, 2=2}
//! assert_eq!(lru_cache.get(1), 1);    // return 1
//! lru_cache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
//! assert_eq!(lru_cache.get(2), -1);   // returns -1 (not found)
//! lru_cache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
//! assert_eq!(lru_cache.get(1), -1);   // return -1 (not found)
//! assert_eq!(lru_cache.get(3), 3);    // return 3
//! assert_eq!(lru_cache.get(4), 4);    // return 4
//! ```
//!
//! ## Constraints
//!
//! * 1 <= capacity <= 3000
//! * 0 <= key <= 10000
//! * 0 <= value <= 10^5
//! * At most 2 * 10^5 calls will be made to get and put.

use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

type NodeRef = Rc<RefCell<Node>>;

/// Internal node structure for doubly linked list
#[derive(Debug)]
struct Node {
    key: i32,
    value: i32,
    prev: Option<NodeRef>,
    next: Option<NodeRef>,
}

impl Node {
    fn new(key: i32, value: i32) -> Self {
        Node {
            key,
            value,
            prev: None,
            next: None,
        }
    }
}

/// # Approach 1: Hash Map + Doubly Linked List (Optimal)
/// 
/// **Algorithm:**
/// 1. Use HashMap for O(1) key lookup
/// 2. Use doubly linked list to maintain order (most recent to least recent)
/// 3. Head of list = most recently used, tail = least recently used
/// 4. When accessing a node, move it to head
/// 5. When adding new node, add to head
/// 6. When evicting, remove from tail
/// 
/// **Time Complexity:** 
/// - get: O(1) average
/// - put: O(1) average
/// 
/// **Space Complexity:** O(capacity) - HashMap + linked list nodes
/// 
/// **Key Design Decisions:**
/// - Doubly linked list allows O(1) insertion/deletion at any position
/// - HashMap provides O(1) access to any node
/// - Dummy head/tail nodes simplify edge case handling
/// 
/// **Why this approach is optimal:**
/// - Achieves required O(1) time complexity for both operations
/// - Space usage is exactly what's needed (no wasted space)
/// - Clean separation of concerns (HashMap for lookup, list for ordering)
pub struct LRUCache {
    capacity: usize,
    cache: HashMap<i32, NodeRef>,
    head: NodeRef,  // Dummy head (most recent)
    tail: NodeRef,  // Dummy tail (least recent)
}

impl LRUCache {
    /// Create a new LRU cache with the specified capacity
    pub fn new(capacity: i32) -> Self {
        let head = Rc::new(RefCell::new(Node::new(0, 0)));
        let tail = Rc::new(RefCell::new(Node::new(0, 0)));
        
        // Connect dummy nodes
        head.borrow_mut().next = Some(tail.clone());
        tail.borrow_mut().prev = Some(head.clone());
        
        LRUCache {
            capacity: capacity as usize,
            cache: HashMap::new(),
            head,
            tail,
        }
    }

    /// Get value by key, return -1 if not found
    /// Moves the accessed node to head (mark as most recently used)
    pub fn get(&mut self, key: i32) -> i32 {
        if let Some(node) = self.cache.get(&key) {
            let node = node.clone();
            let value = node.borrow().value;
            
            // Move to head (mark as most recently used)
            self.move_to_head(node);
            
            value
        } else {
            -1
        }
    }

    /// Put key-value pair into cache
    /// If key exists, update value and move to head
    /// If key doesn't exist, add new node at head
    /// If capacity exceeded, remove least recently used (tail)
    pub fn put(&mut self, key: i32, value: i32) {
        if let Some(node) = self.cache.get(&key) {
            // Key exists, update value and move to head
            let node = node.clone();
            node.borrow_mut().value = value;
            self.move_to_head(node);
        } else {
            // Key doesn't exist, create new node
            let new_node = Rc::new(RefCell::new(Node::new(key, value)));
            
            // Check if we need to evict
            if self.cache.len() >= self.capacity {
                // Remove least recently used (tail)
                let tail_prev = self.tail.borrow().prev.as_ref().unwrap().clone();
                let evicted_key = tail_prev.borrow().key;
                self.remove_node(tail_prev);
                self.cache.remove(&evicted_key);
            }
            
            // Add new node at head and to cache
            self.add_to_head(new_node.clone());
            self.cache.insert(key, new_node);
        }
    }

    /// Add node right after head (most recently used position)
    fn add_to_head(&mut self, node: NodeRef) {
        let first = self.head.borrow().next.as_ref().unwrap().clone();
        
        node.borrow_mut().prev = Some(self.head.clone());
        node.borrow_mut().next = Some(first.clone());
        
        self.head.borrow_mut().next = Some(node.clone());
        first.borrow_mut().prev = Some(node);
    }

    /// Remove node from its current position in the list
    fn remove_node(&mut self, node: NodeRef) {
        let prev = node.borrow().prev.as_ref().unwrap().clone();
        let next = node.borrow().next.as_ref().unwrap().clone();
        
        prev.borrow_mut().next = Some(next.clone());
        next.borrow_mut().prev = Some(prev);
    }

    /// Move existing node to head (mark as most recently used)
    fn move_to_head(&mut self, node: NodeRef) {
        self.remove_node(node.clone());
        self.add_to_head(node);
    }
}

/// # Approach 2: HashMap + Vec with timestamps (Alternative)
/// 
/// **Algorithm:**
/// 1. Use HashMap to store key -> (value, timestamp) pairs
/// 2. Keep a global timestamp counter
/// 3. On access, update timestamp
/// 4. On eviction, find entry with minimum timestamp
/// 
/// **Time Complexity:** 
/// - get: O(1) average
/// - put: O(n) worst case (when evicting, need to find min timestamp)
/// 
/// **Space Complexity:** O(capacity)
/// 
/// **Why this approach is suboptimal:**
/// - Eviction requires O(n) scan to find least recently used
/// - Doesn't meet the O(1) requirement for put operation
pub struct LRUCacheTimestamp {
    capacity: usize,
    cache: HashMap<i32, (i32, u64)>, // key -> (value, timestamp)
    timestamp: u64,
}

impl LRUCacheTimestamp {
    pub fn new(capacity: i32) -> Self {
        LRUCacheTimestamp {
            capacity: capacity as usize,
            cache: HashMap::new(),
            timestamp: 0,
        }
    }

    pub fn get(&mut self, key: i32) -> i32 {
        if let Some((value, _)) = self.cache.get(&key) {
            let value = *value;
            self.timestamp += 1;
            self.cache.insert(key, (value, self.timestamp));
            value
        } else {
            -1
        }
    }

    pub fn put(&mut self, key: i32, value: i32) {
        self.timestamp += 1;
        
        if self.cache.contains_key(&key) {
            // Update existing
            self.cache.insert(key, (value, self.timestamp));
        } else {
            // Check capacity
            if self.cache.len() >= self.capacity {
                // Find and remove least recently used (minimum timestamp)
                let mut min_key = key;
                let mut min_timestamp = self.timestamp;
                
                for (&k, &(_, ts)) in &self.cache {
                    if ts < min_timestamp {
                        min_timestamp = ts;
                        min_key = k;
                    }
                }
                
                self.cache.remove(&min_key);
            }
            
            self.cache.insert(key, (value, self.timestamp));
        }
    }
}

/// # Approach 3: HashMap + VecDeque (Queue-based)
/// 
/// **Algorithm:**
/// 1. Use HashMap for key -> value mapping
/// 2. Use VecDeque to maintain access order
/// 3. On access, remove from queue and push to back
/// 4. On eviction, pop from front
/// 
/// **Time Complexity:** 
/// - get: O(n) - need to find and remove from middle of queue
/// - put: O(n) - same issue
/// 
/// **Space Complexity:** O(capacity)
/// 
/// **Why this approach is suboptimal:**
/// - VecDeque doesn't support O(1) removal from middle
/// - Requires linear scan to find element to remove
use std::collections::VecDeque;

pub struct LRUCacheQueue {
    capacity: usize,
    cache: HashMap<i32, i32>,
    order: VecDeque<i32>,
}

impl LRUCacheQueue {
    pub fn new(capacity: i32) -> Self {
        LRUCacheQueue {
            capacity: capacity as usize,
            cache: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    pub fn get(&mut self, key: i32) -> i32 {
        if let Some(&value) = self.cache.get(&key) {
            // Remove from current position and move to back
            if let Some(pos) = self.order.iter().position(|&x| x == key) {
                self.order.remove(pos);
            }
            self.order.push_back(key);
            value
        } else {
            -1
        }
    }

    pub fn put(&mut self, key: i32, value: i32) {
        if self.cache.contains_key(&key) {
            // Update existing
            self.cache.insert(key, value);
            // Remove from current position and move to back
            if let Some(pos) = self.order.iter().position(|&x| x == key) {
                self.order.remove(pos);
            }
            self.order.push_back(key);
        } else {
            // Check capacity
            if self.cache.len() >= self.capacity {
                // Remove least recently used (front of queue)
                if let Some(evicted_key) = self.order.pop_front() {
                    self.cache.remove(&evicted_key);
                }
            }
            
            self.cache.insert(key, value);
            self.order.push_back(key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        let mut cache = LRUCache::new(2);
        
        cache.put(1, 1);
        cache.put(2, 2);
        assert_eq!(cache.get(1), 1);
        
        cache.put(3, 3);    // Evicts key 2
        assert_eq!(cache.get(2), -1);
        
        cache.put(4, 4);    // Evicts key 1
        assert_eq!(cache.get(1), -1);
        assert_eq!(cache.get(3), 3);
        assert_eq!(cache.get(4), 4);
    }

    #[test]
    fn test_update_existing_key() {
        let mut cache = LRUCache::new(2);
        
        cache.put(1, 1);
        cache.put(2, 2);
        cache.put(1, 10);   // Update key 1
        
        assert_eq!(cache.get(1), 10);
        assert_eq!(cache.get(2), 2);
    }

    #[test]
    fn test_capacity_one() {
        let mut cache = LRUCache::new(1);
        
        cache.put(1, 1);
        assert_eq!(cache.get(1), 1);
        
        cache.put(2, 2);    // Evicts key 1
        assert_eq!(cache.get(1), -1);
        assert_eq!(cache.get(2), 2);
    }

    #[test]
    fn test_get_nonexistent() {
        let mut cache = LRUCache::new(2);
        
        assert_eq!(cache.get(1), -1);
        
        cache.put(1, 1);
        assert_eq!(cache.get(1), 1);
        assert_eq!(cache.get(2), -1);
    }

    #[test]
    fn test_lru_ordering() {
        let mut cache = LRUCache::new(3);
        
        cache.put(1, 1);
        cache.put(2, 2);
        cache.put(3, 3);
        
        // Access 1, making it most recently used
        assert_eq!(cache.get(1), 1);
        
        // Add 4, should evict 2 (least recently used)
        cache.put(4, 4);
        
        assert_eq!(cache.get(1), 1);   // Still present
        assert_eq!(cache.get(2), -1);  // Evicted
        assert_eq!(cache.get(3), 3);   // Still present
        assert_eq!(cache.get(4), 4);   // New entry
    }

    #[test]
    fn test_approach_consistency() {
        // Test that all approaches give same results for basic operations
        let mut cache1 = LRUCache::new(2);
        let mut cache2 = LRUCacheTimestamp::new(2);
        let mut cache3 = LRUCacheQueue::new(2);
        
        // Sequence of operations
        cache1.put(1, 1);
        cache2.put(1, 1);
        cache3.put(1, 1);
        
        cache1.put(2, 2);
        cache2.put(2, 2);
        cache3.put(2, 2);
        
        assert_eq!(cache1.get(1), cache2.get(1));
        assert_eq!(cache2.get(1), cache3.get(1));
        
        cache1.put(3, 3);
        cache2.put(3, 3);
        cache3.put(3, 3);
        
        assert_eq!(cache1.get(2), cache2.get(2));
        assert_eq!(cache2.get(2), cache3.get(2));
        assert_eq!(cache1.get(1), cache2.get(1));
        assert_eq!(cache2.get(1), cache3.get(1));
        assert_eq!(cache1.get(3), cache2.get(3));
        assert_eq!(cache2.get(3), cache3.get(3));
    }

    #[test]
    fn test_large_capacity() {
        let mut cache = LRUCache::new(1000);
        
        // Fill cache
        for i in 0..1000 {
            cache.put(i, i * 2);
        }
        
        // Check all values are present
        for i in 0..1000 {
            assert_eq!(cache.get(i), i * 2);
        }
        
        // Add one more, should evict first entry (key 0)
        cache.put(1000, 2000);
        assert_eq!(cache.get(0), -1);
        assert_eq!(cache.get(1000), 2000);
        assert_eq!(cache.get(999), 999 * 2);
    }

    #[test]
    fn test_mixed_operations() {
        let mut cache = LRUCache::new(3);
        
        cache.put(1, 100);
        cache.put(2, 200);
        assert_eq!(cache.get(1), 100);
        
        cache.put(3, 300);
        cache.put(4, 400);  // Evicts 2
        
        assert_eq!(cache.get(2), -1);
        assert_eq!(cache.get(1), 100);
        assert_eq!(cache.get(3), 300);
        assert_eq!(cache.get(4), 400);
        
        cache.put(5, 500);  // Evicts 1 (since 1 was accessed earlier but 3,4 accessed later)
        assert_eq!(cache.get(1), -1);
    }

    #[test]
    fn test_stress_operations() {
        let mut cache = LRUCache::new(10);
        
        // Perform many operations
        for i in 0..100 {
            cache.put(i % 15, i);  // Some keys will be updated multiple times
        }
        
        // Check that only last 10 unique keys are present
        let mut present_count = 0;
        for i in 0..15 {
            if cache.get(i) != -1 {
                present_count += 1;
            }
        }
        
        assert!(present_count <= 10);
    }

    #[test]
    fn test_edge_cases() {
        // Test with minimum capacity
        let mut cache = LRUCache::new(1);
        cache.put(0, 0);
        assert_eq!(cache.get(0), 0);
        
        // Test key/value at boundaries
        let mut cache2 = LRUCache::new(2);
        cache2.put(0, 0);
        cache2.put(10000, 100000);
        
        assert_eq!(cache2.get(0), 0);
        assert_eq!(cache2.get(10000), 100000);
    }
}