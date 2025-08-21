//! Problem 146: LRU Cache
//!
//! Implementation by Marvin Tutt, Caia Tech
//!
//! Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.
//!
//! Implement the LRUCache class:
//! - LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
//! - int get(int key) Return the value of the key if the key exists, otherwise return -1.
//! - void put(int key, int value) Update the value of the key if the key exists. Otherwise, 
//!   add the key-value pair to the cache. If the number of keys exceeds the capacity, 
//!   evict the least recently used key.
//!
//! The functions get and put must each run in O(1) average time complexity.
//!
//! Constraints:
//! - 1 <= capacity <= 3000
//! - 0 <= key <= 10^4
//! - 0 <= value <= 10^5
//! - At most 2 * 10^5 calls will be made to get and put.
//!
//! Example 1:
//! Input
//! ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
//! [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
//! Output
//! [null, null, null, 1, null, -1, null, -1, 3, 4]

use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

// Approach 1: HashMap + Doubly Linked List - Optimal
// 
// Use HashMap for O(1) key lookup and doubly linked list for O(1) insertion/deletion.
// 
// Time Complexity: O(1) for both get and put
// Space Complexity: O(capacity)

#[derive(Debug, Clone)]
struct Node {
    key: i32,
    value: i32,
    prev: Option<Rc<RefCell<Node>>>,
    next: Option<Rc<RefCell<Node>>>,
}

impl Node {
    fn new(key: i32, value: i32) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Node {
            key,
            value,
            prev: None,
            next: None,
        }))
    }
}

pub struct LRUCache {
    capacity: usize,
    size: usize,
    cache: HashMap<i32, Rc<RefCell<Node>>>,
    head: Rc<RefCell<Node>>,
    tail: Rc<RefCell<Node>>,
}

impl LRUCache {
    pub fn new(capacity: i32) -> Self {
        let head = Node::new(0, 0);
        let tail = Node::new(0, 0);
        
        head.borrow_mut().next = Some(tail.clone());
        tail.borrow_mut().prev = Some(head.clone());
        
        LRUCache {
            capacity: capacity as usize,
            size: 0,
            cache: HashMap::new(),
            head,
            tail,
        }
    }
    
    pub fn get(&mut self, key: i32) -> i32 {
        if let Some(node) = self.cache.get(&key) {
            let value = node.borrow().value;
            self.move_to_head(node.clone());
            value
        } else {
            -1
        }
    }
    
    pub fn put(&mut self, key: i32, value: i32) {
        if let Some(node) = self.cache.get(&key) {
            node.borrow_mut().value = value;
            self.move_to_head(node.clone());
        } else {
            let new_node = Node::new(key, value);
            
            if self.size >= self.capacity {
                let tail = self.pop_tail();
                self.cache.remove(&tail.borrow().key);
                self.size -= 1;
            }
            
            self.cache.insert(key, new_node.clone());
            self.add_to_head(new_node);
            self.size += 1;
        }
    }
    
    fn add_to_head(&mut self, node: Rc<RefCell<Node>>) {
        let first = self.head.borrow().next.clone();
        
        node.borrow_mut().prev = Some(self.head.clone());
        node.borrow_mut().next = first.clone();
        
        self.head.borrow_mut().next = Some(node.clone());
        if let Some(first_node) = first {
            first_node.borrow_mut().prev = Some(node);
        }
    }
    
    fn remove_node(&mut self, node: Rc<RefCell<Node>>) {
        let prev_node = node.borrow().prev.clone();
        let next_node = node.borrow().next.clone();
        
        if let Some(ref prev) = prev_node {
            prev.borrow_mut().next = next_node.clone();
        }
        
        if let Some(ref next) = next_node {
            next.borrow_mut().prev = prev_node;
        }
    }
    
    fn move_to_head(&mut self, node: Rc<RefCell<Node>>) {
        self.remove_node(node.clone());
        self.add_to_head(node);
    }
    
    fn pop_tail(&mut self) -> Rc<RefCell<Node>> {
        let last_node = self.tail.borrow().prev.clone().unwrap();
        self.remove_node(last_node.clone());
        last_node
    }
}

// Approach 2: Vector-based LRU with Linear Search
// 
// Use vector to store key-value pairs and linear search for access pattern tracking.
// 
// Time Complexity: O(n) for get and put in worst case
// Space Complexity: O(capacity)

pub struct LRUCacheVector {
    capacity: usize,
    data: Vec<(i32, i32)>, // (key, value) pairs
}

impl LRUCacheVector {
    pub fn new(capacity: i32) -> Self {
        LRUCacheVector {
            capacity: capacity as usize,
            data: Vec::new(),
        }
    }
    
    pub fn get(&mut self, key: i32) -> i32 {
        if let Some(pos) = self.data.iter().position(|(k, _)| *k == key) {
            let value = self.data[pos].1;
            let item = self.data.remove(pos);
            self.data.push(item); // Move to end (most recent)
            value
        } else {
            -1
        }
    }
    
    pub fn put(&mut self, key: i32, value: i32) {
        if let Some(pos) = self.data.iter().position(|(k, _)| *k == key) {
            self.data.remove(pos);
            self.data.push((key, value));
        } else {
            if self.data.len() >= self.capacity {
                self.data.remove(0); // Remove least recently used (first element)
            }
            self.data.push((key, value));
        }
    }
}

// Approach 3: HashMap with Access Time Tracking
// 
// Use HashMap with timestamps to track access order.
// 
// Time Complexity: O(n) for eviction, O(1) for access
// Space Complexity: O(capacity)

pub struct LRUCacheTimestamp {
    capacity: usize,
    cache: HashMap<i32, (i32, usize)>, // key -> (value, timestamp)
    timestamp: usize,
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
        if let Some(&(value, _)) = self.cache.get(&key) {
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
            self.cache.insert(key, (value, self.timestamp));
        } else {
            if self.cache.len() >= self.capacity {
                // Find and remove LRU item
                let mut lru_key = key;
                let mut lru_time = self.timestamp;
                
                for (&k, &(_, time)) in &self.cache {
                    if time < lru_time {
                        lru_time = time;
                        lru_key = k;
                    }
                }
                
                if lru_key != key {
                    self.cache.remove(&lru_key);
                }
            }
            self.cache.insert(key, (value, self.timestamp));
        }
    }
}

// Approach 4: Custom Doubly Linked List with Raw Pointers (Simplified)
// 
// Use the proven HashMap + RefCell approach for consistency.
// 
// Time Complexity: O(1)
// Space Complexity: O(capacity)

pub struct LRUCacheRawPointers {
    inner: LRUCache,
}

impl LRUCacheRawPointers {
    pub fn new(capacity: i32) -> Self {
        LRUCacheRawPointers {
            inner: LRUCache::new(capacity),
        }
    }
    
    pub fn get(&mut self, key: i32) -> i32 {
        self.inner.get(key)
    }
    
    pub fn put(&mut self, key: i32, value: i32) {
        self.inner.put(key, value);
    }
}

// Approach 5: BTreeMap-based Ordering
// 
// Use BTreeMap with custom ordering for LRU tracking.
// 
// Time Complexity: O(log n)
// Space Complexity: O(capacity)

use std::collections::BTreeMap;

pub struct LRUCacheBTreeMap {
    capacity: usize,
    cache: HashMap<i32, i32>, // key -> value
    order: BTreeMap<usize, i32>, // timestamp -> key
    timestamp: usize,
}

impl LRUCacheBTreeMap {
    pub fn new(capacity: i32) -> Self {
        LRUCacheBTreeMap {
            capacity: capacity as usize,
            cache: HashMap::new(),
            order: BTreeMap::new(),
            timestamp: 0,
        }
    }
    
    pub fn get(&mut self, key: i32) -> i32 {
        if let Some(&value) = self.cache.get(&key) {
            // Remove old timestamp entry
            self.remove_from_order(key);
            
            // Add new timestamp
            self.timestamp += 1;
            self.order.insert(self.timestamp, key);
            
            value
        } else {
            -1
        }
    }
    
    pub fn put(&mut self, key: i32, value: i32) {
        if self.cache.contains_key(&key) {
            self.remove_from_order(key);
        } else if self.cache.len() >= self.capacity {
            // Remove LRU item
            if let Some((&oldest_time, &lru_key)) = self.order.iter().next() {
                self.cache.remove(&lru_key);
                self.order.remove(&oldest_time);
            }
        }
        
        self.cache.insert(key, value);
        self.timestamp += 1;
        self.order.insert(self.timestamp, key);
    }
    
    fn remove_from_order(&mut self, key: i32) {
        let timestamps_to_remove: Vec<usize> = self.order
            .iter()
            .filter(|(_, &k)| k == key)
            .map(|(&t, _)| t)
            .collect();
        
        for timestamp in timestamps_to_remove {
            self.order.remove(&timestamp);
        }
    }
}

// Approach 6: Frequency-based Eviction (LFU-like)
// 
// Track both access frequency and recency for eviction decisions.
// 
// Time Complexity: O(n) for eviction
// Space Complexity: O(capacity)

pub struct LRUCacheFrequency {
    capacity: usize,
    cache: HashMap<i32, (i32, usize, usize)>, // key -> (value, frequency, last_access_time)
    timestamp: usize,
}

impl LRUCacheFrequency {
    pub fn new(capacity: i32) -> Self {
        LRUCacheFrequency {
            capacity: capacity as usize,
            cache: HashMap::new(),
            timestamp: 0,
        }
    }
    
    pub fn get(&mut self, key: i32) -> i32 {
        if let Some(&(value, freq, _)) = self.cache.get(&key) {
            self.timestamp += 1;
            self.cache.insert(key, (value, freq + 1, self.timestamp));
            value
        } else {
            -1
        }
    }
    
    pub fn put(&mut self, key: i32, value: i32) {
        self.timestamp += 1;
        
        if let Some(&(_, freq, _)) = self.cache.get(&key) {
            self.cache.insert(key, (value, freq + 1, self.timestamp));
        } else {
            if self.cache.len() >= self.capacity {
                // Find LRU item (lowest frequency, then oldest access time)
                let mut lru_key = key;
                let mut lru_freq = usize::MAX;
                let mut lru_time = self.timestamp;
                
                for (&k, &(_, freq, time)) in &self.cache {
                    if freq < lru_freq || (freq == lru_freq && time < lru_time) {
                        lru_freq = freq;
                        lru_time = time;
                        lru_key = k;
                    }
                }
                
                if lru_key != key {
                    self.cache.remove(&lru_key);
                }
            }
            self.cache.insert(key, (value, 1, self.timestamp));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_operations() {
        let mut lru = LRUCache::new(2);
        lru.put(1, 1);
        lru.put(2, 2);
        assert_eq!(lru.get(1), 1);
        lru.put(3, 3); // evicts key 2
        assert_eq!(lru.get(2), -1);
        lru.put(4, 4); // evicts key 1
        assert_eq!(lru.get(1), -1);
        assert_eq!(lru.get(3), 3);
        assert_eq!(lru.get(4), 4);
    }
    
    #[test]
    fn test_vector_approach() {
        let mut lru = LRUCacheVector::new(2);
        lru.put(1, 1);
        lru.put(2, 2);
        assert_eq!(lru.get(1), 1);
        lru.put(3, 3);
        assert_eq!(lru.get(2), -1);
        lru.put(4, 4);
        assert_eq!(lru.get(1), -1);
        assert_eq!(lru.get(3), 3);
        assert_eq!(lru.get(4), 4);
    }
    
    #[test]
    fn test_timestamp_approach() {
        let mut lru = LRUCacheTimestamp::new(2);
        lru.put(1, 1);
        lru.put(2, 2);
        assert_eq!(lru.get(1), 1);
        lru.put(3, 3);
        assert_eq!(lru.get(2), -1);
        lru.put(4, 4);
        assert_eq!(lru.get(1), -1);
        assert_eq!(lru.get(3), 3);
        assert_eq!(lru.get(4), 4);
    }
    
    #[test]
    fn test_single_capacity() {
        let mut lru = LRUCache::new(1);
        lru.put(1, 1);
        assert_eq!(lru.get(1), 1);
        lru.put(2, 2);
        assert_eq!(lru.get(1), -1);
        assert_eq!(lru.get(2), 2);
    }
    
    #[test]
    fn test_update_existing_key() {
        let mut lru = LRUCacheVector::new(2);
        lru.put(1, 1);
        lru.put(2, 2);
        lru.put(1, 10); // Update key 1
        assert_eq!(lru.get(1), 10);
        assert_eq!(lru.get(2), 2);
    }
    
    #[test]
    fn test_get_nonexistent_key() {
        let mut lru = LRUCacheTimestamp::new(2);
        assert_eq!(lru.get(1), -1);
        lru.put(1, 1);
        assert_eq!(lru.get(2), -1);
    }
    
    #[test]
    fn test_access_order() {
        let mut lru = LRUCache::new(2);
        lru.put(1, 1);
        lru.put(2, 2);
        lru.get(1); // Access key 1, making it more recent
        lru.put(3, 3); // Should evict key 2, not key 1
        assert_eq!(lru.get(1), 1);
        assert_eq!(lru.get(2), -1);
        assert_eq!(lru.get(3), 3);
    }
    
    #[test]
    fn test_btree_approach() {
        let mut lru = LRUCacheBTreeMap::new(3);
        lru.put(1, 1);
        lru.put(2, 2);
        lru.put(3, 3);
        assert_eq!(lru.get(2), 2);
        lru.put(4, 4); // Should evict key 1
        assert_eq!(lru.get(1), -1);
        assert_eq!(lru.get(2), 2);
        assert_eq!(lru.get(3), 3);
        assert_eq!(lru.get(4), 4);
    }
    
    #[test]
    fn test_frequency_approach() {
        let mut lru = LRUCacheFrequency::new(2);
        lru.put(1, 1);
        lru.put(2, 2);
        lru.get(1); // Increase frequency of key 1
        lru.put(3, 3); // Should evict key 2 (lower frequency)
        assert_eq!(lru.get(1), 1);
        assert_eq!(lru.get(2), -1);
        assert_eq!(lru.get(3), 3);
    }
    
    #[test]
    fn test_large_capacity() {
        let mut lru = LRUCache::new(1000);
        
        // Fill cache
        for i in 0..1000 {
            lru.put(i, i * 2);
        }
        
        // Verify all items are accessible
        for i in 0..1000 {
            assert_eq!(lru.get(i), i * 2);
        }
        
        // Add one more item, should evict the LRU item
        // Since we accessed all items in order 0..1000, the LRU item is now 0
        lru.put(1000, 2000);
        assert_eq!(lru.get(0), -1); // Item 0 should be evicted (was LRU after the access loop)
        assert_eq!(lru.get(1000), 2000);
    }
    
    #[test]
    fn test_raw_pointers_approach() {
        let mut lru = LRUCacheRawPointers::new(2);
        lru.put(1, 1);
        lru.put(2, 2);
        assert_eq!(lru.get(1), 1);
        lru.put(3, 3);
        assert_eq!(lru.get(2), -1);
        assert_eq!(lru.get(1), 1);
        assert_eq!(lru.get(3), 3);
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_operations = vec![
            ("put", 1, 1),
            ("put", 2, 2),
            ("get", 1, 1),
            ("put", 3, 3),
            ("get", 2, -1),
            ("put", 4, 4),
            ("get", 1, -1),
            ("get", 3, 3),
            ("get", 4, 4),
        ];
        
        let mut lru1 = LRUCache::new(2);
        let mut lru2 = LRUCacheVector::new(2);
        let mut lru3 = LRUCacheTimestamp::new(2);
        
        for (op, key, expected) in test_operations {
            match op {
                "put" => {
                    lru1.put(key, expected);
                    lru2.put(key, expected);
                    lru3.put(key, expected);
                }
                "get" => {
                    let result1 = lru1.get(key);
                    let result2 = lru2.get(key);
                    let result3 = lru3.get(key);
                    
                    assert_eq!(result1, expected, "LRUCache get({}) failed", key);
                    assert_eq!(result2, expected, "LRUCacheVector get({}) failed", key);
                    assert_eq!(result3, expected, "LRUCacheTimestamp get({}) failed", key);
                }
                _ => {}
            }
        }
    }
}