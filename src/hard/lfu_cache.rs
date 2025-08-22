//! Problem 460: LFU Cache
//!
//! Design and implement a data structure for a Least Frequently Used (LFU) cache.
//!
//! Implement the LFUCache class:
//! - LFUCache(int capacity) Initializes the object with the capacity of the data structure.
//! - int get(int key) Gets the value of the key if the key exists in the cache. Otherwise, returns -1.
//! - void put(int key, int value) Update the value of the key if present, or inserts the key if not already present. 
//!   When the cache reaches its capacity, it should invalidate and remove the least frequently used key before inserting a new item. 
//!   For this problem, when there is a tie (i.e., two or more keys with the same frequency), the least recently used key would be invalidated.
//!
//! To determine the least frequently used key, a use counter is maintained for each key in the cache. 
//! The key with the smallest use counter is the least frequently used key.
//!
//! When a key is first inserted into the cache, its use counter is set to 1 (due to the put operation). 
//! The use counter for a key in the cache is incremented either a get or put operation is called on it.
//!
//! The functions get and put must each run in O(1) average time complexity.
//!
//! Constraints:
//! - 1 <= capacity <= 10^4
//! - 0 <= key <= 10^5
//! - 0 <= value <= 10^9
//! - At most 2 * 10^5 calls will be made to get and put.
//!
//! Example 1:
//! Input
//! ["LFUCache", "put", "put", "get", "put", "get", "get", "put", "get", "get", "get"]
//! [[2], [1, 1], [2, 2], [1], [3, 3], [2], [3], [4, 4], [1], [3], [4]]
//! Output
//! [null, null, null, 1, null, -1, 3, null, -1, 3, 4]

use std::collections::HashMap;
use std::collections::LinkedList;
use std::collections::BTreeMap;
use std::cell::RefCell;
use std::rc::Rc;

/// Approach 1: HashMap + DoublyLinkedList with Frequency Groups
/// 
/// Use HashMap for O(1) key lookup and frequency-indexed doubly linked lists
/// to maintain LRU order within each frequency group.
/// 
/// Time Complexity: O(1) for both get and put
/// Space Complexity: O(capacity)
pub struct LFUCacheHashMap {
    capacity: usize,
    min_freq: i32,
    key_to_val: HashMap<i32, i32>,
    key_to_freq: HashMap<i32, i32>,
    freq_to_keys: HashMap<i32, LinkedList<i32>>,
}

impl LFUCacheHashMap {
    pub fn new(capacity: i32) -> Self {
        Self {
            capacity: capacity as usize,
            min_freq: 0,
            key_to_val: HashMap::new(),
            key_to_freq: HashMap::new(),
            freq_to_keys: HashMap::new(),
        }
    }
    
    pub fn get(&mut self, key: i32) -> i32 {
        if let Some(&val) = self.key_to_val.get(&key) {
            self.increase_freq(key);
            val
        } else {
            -1
        }
    }
    
    pub fn put(&mut self, key: i32, value: i32) {
        if self.capacity == 0 {
            return;
        }
        
        if self.key_to_val.contains_key(&key) {
            self.key_to_val.insert(key, value);
            self.increase_freq(key);
            return;
        }
        
        if self.key_to_val.len() >= self.capacity {
            self.remove_min_freq_key();
        }
        
        self.key_to_val.insert(key, value);
        self.key_to_freq.insert(key, 1);
        self.freq_to_keys.entry(1).or_insert_with(LinkedList::new).push_back(key);
        self.min_freq = 1;
    }
    
    fn increase_freq(&mut self, key: i32) {
        let freq = self.key_to_freq[&key];
        self.key_to_freq.insert(key, freq + 1);
        
        // Remove from current frequency list
        if let Some(keys) = self.freq_to_keys.get_mut(&freq) {
            if let Some(pos) = keys.iter().position(|&k| k == key) {
                let mut split_list = keys.split_off(pos);
                split_list.pop_front();
                keys.append(&mut split_list);
            }
            if keys.is_empty() {
                self.freq_to_keys.remove(&freq);
            }
        }
        
        // Add to new frequency list
        self.freq_to_keys.entry(freq + 1).or_insert_with(LinkedList::new).push_back(key);
        
        // Update min_freq if necessary
        if freq == self.min_freq && !self.freq_to_keys.contains_key(&freq) {
            self.min_freq += 1;
        }
    }
    
    fn remove_min_freq_key(&mut self) {
        if let Some(keys) = self.freq_to_keys.get_mut(&self.min_freq) {
            if let Some(key) = keys.pop_front() {
                self.key_to_val.remove(&key);
                self.key_to_freq.remove(&key);
            }
            if keys.is_empty() {
                self.freq_to_keys.remove(&self.min_freq);
            }
        }
    }
}

/// Approach 2: Node-based Doubly Linked List
/// 
/// Use actual node pointers with Rc<RefCell<>> for true doubly linked list behavior.
/// 
/// Time Complexity: O(1) for both get and put
/// Space Complexity: O(capacity)
#[derive(Debug)]
struct Node {
    key: i32,
    value: i32,
    freq: i32,
    prev: Option<Rc<RefCell<Node>>>,
    next: Option<Rc<RefCell<Node>>>,
}

pub struct LFUCacheNodes {
    capacity: usize,
    size: usize,
    min_freq: i32,
    key_map: HashMap<i32, Rc<RefCell<Node>>>,
    freq_map: HashMap<i32, (Rc<RefCell<Node>>, Rc<RefCell<Node>>)>, // (head, tail)
}

impl LFUCacheNodes {
    pub fn new(capacity: i32) -> Self {
        Self {
            capacity: capacity as usize,
            size: 0,
            min_freq: 0,
            key_map: HashMap::new(),
            freq_map: HashMap::new(),
        }
    }
    
    pub fn get(&mut self, key: i32) -> i32 {
        if let Some(node) = self.key_map.get(&key).cloned() {
            let value = node.borrow().value;
            self.update_freq(node);
            value
        } else {
            -1
        }
    }
    
    pub fn put(&mut self, key: i32, value: i32) {
        if self.capacity == 0 {
            return;
        }
        
        if let Some(node) = self.key_map.get(&key).cloned() {
            node.borrow_mut().value = value;
            self.update_freq(node);
            return;
        }
        
        if self.size >= self.capacity {
            self.remove_lfu();
        }
        
        let new_node = Rc::new(RefCell::new(Node {
            key,
            value,
            freq: 1,
            prev: None,
            next: None,
        }));
        
        self.key_map.insert(key, new_node.clone());
        self.add_to_freq_list(new_node, 1);
        self.size += 1;
        self.min_freq = 1;
    }
    
    fn update_freq(&mut self, node: Rc<RefCell<Node>>) {
        let old_freq = node.borrow().freq;
        let new_freq = old_freq + 1;
        
        self.remove_from_freq_list(node.clone(), old_freq);
        node.borrow_mut().freq = new_freq;
        self.add_to_freq_list(node, new_freq);
        
        if old_freq == self.min_freq && !self.freq_map.contains_key(&old_freq) {
            self.min_freq += 1;
        }
    }
    
    fn add_to_freq_list(&mut self, node: Rc<RefCell<Node>>, freq: i32) {
        if !self.freq_map.contains_key(&freq) {
            let head = Rc::new(RefCell::new(Node {
                key: -1, value: -1, freq: -1,
                prev: None, next: None,
            }));
            let tail = Rc::new(RefCell::new(Node {
                key: -1, value: -1, freq: -1,
                prev: None, next: None,
            }));
            
            head.borrow_mut().next = Some(tail.clone());
            tail.borrow_mut().prev = Some(head.clone());
            
            self.freq_map.insert(freq, (head, tail));
        }
        
        let (head, _) = self.freq_map[&freq].clone();
        let first = head.borrow().next.clone();
        
        head.borrow_mut().next = Some(node.clone());
        node.borrow_mut().prev = Some(head);
        node.borrow_mut().next = first.clone();
        if let Some(first_node) = first {
            first_node.borrow_mut().prev = Some(node);
        }
    }
    
    fn remove_from_freq_list(&mut self, node: Rc<RefCell<Node>>, freq: i32) {
        let prev = node.borrow().prev.clone();
        let next = node.borrow().next.clone();
        
        if let Some(prev_node) = &prev {
            prev_node.borrow_mut().next = next.clone();
        }
        if let Some(next_node) = &next {
            next_node.borrow_mut().prev = prev.clone();
        }
        
        // Check if frequency list is empty
        if let Some((head, tail)) = self.freq_map.get(&freq) {
            if Rc::ptr_eq(&head.borrow().next.as_ref().unwrap(), tail) {
                self.freq_map.remove(&freq);
            }
        }
    }
    
    fn remove_lfu(&mut self) {
        if let Some((_, tail)) = self.freq_map.get(&self.min_freq).cloned() {
            let lfu_node = tail.borrow().prev.clone().unwrap();
            let key = lfu_node.borrow().key;
            
            self.remove_from_freq_list(lfu_node, self.min_freq);
            self.key_map.remove(&key);
            self.size -= 1;
        }
    }
}

/// Approach 3: BTreeMap for Ordered Frequencies
/// 
/// Use BTreeMap to maintain frequencies in sorted order for easy min finding.
/// 
/// Time Complexity: O(log F) where F is number of unique frequencies
/// Space Complexity: O(capacity)
pub struct LFUCacheBTreeMap {
    capacity: usize,
    counter: i32,
    key_to_val: HashMap<i32, i32>,
    key_to_freq: HashMap<i32, i32>,
    key_to_time: HashMap<i32, i32>,
    freq_to_time_keys: BTreeMap<i32, BTreeMap<i32, i32>>,
}

impl LFUCacheBTreeMap {
    pub fn new(capacity: i32) -> Self {
        Self {
            capacity: capacity as usize,
            counter: 0,
            key_to_val: HashMap::new(),
            key_to_freq: HashMap::new(),
            key_to_time: HashMap::new(),
            freq_to_time_keys: BTreeMap::new(),
        }
    }
    
    pub fn get(&mut self, key: i32) -> i32 {
        if let Some(&val) = self.key_to_val.get(&key) {
            self.increase_freq(key);
            val
        } else {
            -1
        }
    }
    
    pub fn put(&mut self, key: i32, value: i32) {
        if self.capacity == 0 {
            return;
        }
        
        if self.key_to_val.contains_key(&key) {
            self.key_to_val.insert(key, value);
            self.increase_freq(key);
            return;
        }
        
        if self.key_to_val.len() >= self.capacity {
            self.remove_lfu();
        }
        
        self.counter += 1;
        self.key_to_val.insert(key, value);
        self.key_to_freq.insert(key, 1);
        self.key_to_time.insert(key, self.counter);
        self.freq_to_time_keys.entry(1).or_insert_with(BTreeMap::new)
            .insert(self.counter, key);
    }
    
    fn increase_freq(&mut self, key: i32) {
        let freq = self.key_to_freq[&key];
        let time = self.key_to_time[&key];
        
        // Remove from current frequency
        if let Some(time_keys) = self.freq_to_time_keys.get_mut(&freq) {
            time_keys.remove(&time);
            if time_keys.is_empty() {
                self.freq_to_time_keys.remove(&freq);
            }
        }
        
        // Update frequency and time
        self.counter += 1;
        self.key_to_freq.insert(key, freq + 1);
        self.key_to_time.insert(key, self.counter);
        
        // Add to new frequency
        self.freq_to_time_keys.entry(freq + 1).or_insert_with(BTreeMap::new)
            .insert(self.counter, key);
    }
    
    fn remove_lfu(&mut self) {
        if let Some((&min_freq, _)) = self.freq_to_time_keys.iter().next() {
            if let Some(time_keys) = self.freq_to_time_keys.get_mut(&min_freq) {
                if let Some((&min_time, &key)) = time_keys.iter().next() {
                    time_keys.remove(&min_time);
                    if time_keys.is_empty() {
                        self.freq_to_time_keys.remove(&min_freq);
                    }
                    
                    self.key_to_val.remove(&key);
                    self.key_to_freq.remove(&key);
                    self.key_to_time.remove(&key);
                }
            }
        }
    }
}

/// Approach 4: Vector-based Implementation
/// 
/// Use vectors to simulate linked lists for simpler implementation.
/// 
/// Time Complexity: O(n) worst case for removal operations
/// Space Complexity: O(capacity)
pub struct LFUCacheVector {
    capacity: usize,
    counter: i32,
    items: Vec<(i32, i32, i32, i32)>, // (key, value, freq, time)
}

impl LFUCacheVector {
    pub fn new(capacity: i32) -> Self {
        Self {
            capacity: capacity as usize,
            counter: 0,
            items: Vec::new(),
        }
    }
    
    pub fn get(&mut self, key: i32) -> i32 {
        if let Some(pos) = self.items.iter().position(|(k, _, _, _)| *k == key) {
            let (k, v, freq, _) = self.items[pos];
            self.counter += 1;
            self.items[pos] = (k, v, freq + 1, self.counter);
            v
        } else {
            -1
        }
    }
    
    pub fn put(&mut self, key: i32, value: i32) {
        if self.capacity == 0 {
            return;
        }
        
        if let Some(pos) = self.items.iter().position(|(k, _, _, _)| *k == key) {
            let (k, _, freq, _) = self.items[pos];
            self.counter += 1;
            self.items[pos] = (k, value, freq + 1, self.counter);
            return;
        }
        
        if self.items.len() >= self.capacity {
            // Find LFU item (min freq, then min time)
            let mut min_pos = 0;
            for i in 1..self.items.len() {
                let (_, _, freq_i, time_i) = self.items[i];
                let (_, _, freq_min, time_min) = self.items[min_pos];
                
                if freq_i < freq_min || (freq_i == freq_min && time_i < time_min) {
                    min_pos = i;
                }
            }
            self.items.remove(min_pos);
        }
        
        self.counter += 1;
        self.items.push((key, value, 1, self.counter));
    }
}

/// Approach 5: Two-Level HashMap
/// 
/// Use nested HashMaps for frequency and recency tracking.
/// 
/// Time Complexity: O(1) average case
/// Space Complexity: O(capacity)
pub struct LFUCacheTwoLevel {
    capacity: usize,
    min_freq: i32,
    counter: i32,
    key_to_val: HashMap<i32, i32>,
    key_to_freq: HashMap<i32, i32>,
    key_to_time: HashMap<i32, i32>,
    freq_to_keys: HashMap<i32, HashMap<i32, i32>>, // freq -> (time -> key)
}

impl LFUCacheTwoLevel {
    pub fn new(capacity: i32) -> Self {
        Self {
            capacity: capacity as usize,
            min_freq: 0,
            counter: 0,
            key_to_val: HashMap::new(),
            key_to_freq: HashMap::new(),
            key_to_time: HashMap::new(),
            freq_to_keys: HashMap::new(),
        }
    }
    
    pub fn get(&mut self, key: i32) -> i32 {
        if let Some(&val) = self.key_to_val.get(&key) {
            self.increase_freq(key);
            val
        } else {
            -1
        }
    }
    
    pub fn put(&mut self, key: i32, value: i32) {
        if self.capacity == 0 {
            return;
        }
        
        if self.key_to_val.contains_key(&key) {
            self.key_to_val.insert(key, value);
            self.increase_freq(key);
            return;
        }
        
        if self.key_to_val.len() >= self.capacity {
            self.remove_lfu();
        }
        
        self.counter += 1;
        self.key_to_val.insert(key, value);
        self.key_to_freq.insert(key, 1);
        self.key_to_time.insert(key, self.counter);
        
        self.freq_to_keys.entry(1).or_insert_with(HashMap::new)
            .insert(self.counter, key);
        self.min_freq = 1;
    }
    
    fn increase_freq(&mut self, key: i32) {
        let freq = self.key_to_freq[&key];
        let time = self.key_to_time[&key];
        
        // Remove from current frequency
        if let Some(time_keys) = self.freq_to_keys.get_mut(&freq) {
            time_keys.remove(&time);
            if time_keys.is_empty() {
                self.freq_to_keys.remove(&freq);
                if freq == self.min_freq {
                    self.min_freq += 1;
                }
            }
        }
        
        // Update frequency and time
        self.counter += 1;
        self.key_to_freq.insert(key, freq + 1);
        self.key_to_time.insert(key, self.counter);
        
        // Add to new frequency
        self.freq_to_keys.entry(freq + 1).or_insert_with(HashMap::new)
            .insert(self.counter, key);
    }
    
    fn remove_lfu(&mut self) {
        if let Some(time_keys) = self.freq_to_keys.get_mut(&self.min_freq) {
            let min_time = *time_keys.keys().min().unwrap();
            let key = time_keys.remove(&min_time).unwrap();
            
            if time_keys.is_empty() {
                self.freq_to_keys.remove(&self.min_freq);
            }
            
            self.key_to_val.remove(&key);
            self.key_to_freq.remove(&key);
            self.key_to_time.remove(&key);
        }
    }
}

/// Approach 6: Custom Balanced Tree
/// 
/// Implement a simple balanced structure for frequency management.
/// 
/// Time Complexity: O(log n) for operations
/// Space Complexity: O(capacity)
pub struct LFUCacheBalanced {
    capacity: usize,
    size: usize,
    global_time: i32,
    items: HashMap<i32, (i32, i32, i32)>, // key -> (value, freq, time)
}

impl LFUCacheBalanced {
    pub fn new(capacity: i32) -> Self {
        Self {
            capacity: capacity as usize,
            size: 0,
            global_time: 0,
            items: HashMap::new(),
        }
    }
    
    pub fn get(&mut self, key: i32) -> i32 {
        if let Some(&(value, freq, _)) = self.items.get(&key) {
            self.global_time += 1;
            self.items.insert(key, (value, freq + 1, self.global_time));
            value
        } else {
            -1
        }
    }
    
    pub fn put(&mut self, key: i32, value: i32) {
        if self.capacity == 0 {
            return;
        }
        
        self.global_time += 1;
        
        if let Some(&(_, freq, _)) = self.items.get(&key) {
            self.items.insert(key, (value, freq + 1, self.global_time));
            return;
        }
        
        if self.size >= self.capacity {
            // Find LFU key
            let mut lfu_key = None;
            let mut min_freq = i32::MAX;
            let mut min_time = i32::MAX;
            
            for (&k, &(_, freq, time)) in &self.items {
                if freq < min_freq || (freq == min_freq && time < min_time) {
                    lfu_key = Some(k);
                    min_freq = freq;
                    min_time = time;
                }
            }
            
            if let Some(k) = lfu_key {
                self.items.remove(&k);
                self.size -= 1;
            }
        }
        
        self.items.insert(key, (value, 1, self.global_time));
        self.size += 1;
    }
}

// Wrapper for testing all approaches
pub struct LFUCache {
    cache: LFUCacheHashMap,
}

impl LFUCache {
    pub fn new(capacity: i32) -> Self {
        Self {
            cache: LFUCacheHashMap::new(capacity),
        }
    }
    
    pub fn get(&mut self, key: i32) -> i32 {
        self.cache.get(key)
    }
    
    pub fn put(&mut self, key: i32, value: i32) {
        self.cache.put(key, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_operations() {
        let mut cache = LFUCache::new(2);
        
        cache.put(1, 1);
        cache.put(2, 2);
        assert_eq!(cache.get(1), 1);
        
        cache.put(3, 3);    // evicts key 2
        assert_eq!(cache.get(2), -1);
        assert_eq!(cache.get(3), 3);
        assert_eq!(cache.get(1), 1);
        
        cache.put(4, 4);    // evicts key 3
        assert_eq!(cache.get(1), 1);
        assert_eq!(cache.get(3), -1);
        assert_eq!(cache.get(4), 4);
    }
    
    #[test]
    fn test_frequency_tie_breaking() {
        let mut cache = LFUCache::new(2);
        
        cache.put(1, 1);
        cache.put(2, 2);
        cache.get(1);       // freq: key 1 = 2, key 2 = 1
        cache.put(3, 3);    // evicts key 2 (lower frequency)
        
        assert_eq!(cache.get(2), -1);
        assert_eq!(cache.get(1), 1);
        assert_eq!(cache.get(3), 3);
    }
    
    #[test]
    fn test_single_capacity() {
        let mut cache = LFUCache::new(1);
        
        cache.put(1, 1);
        assert_eq!(cache.get(1), 1);
        
        cache.put(2, 2);    // evicts key 1
        assert_eq!(cache.get(1), -1);
        assert_eq!(cache.get(2), 2);
    }
    
    #[test]
    fn test_zero_capacity() {
        let mut cache = LFUCache::new(0);
        
        cache.put(1, 1);
        assert_eq!(cache.get(1), -1);
    }
    
    #[test]
    fn test_update_existing_key() {
        let mut cache = LFUCache::new(2);
        
        cache.put(1, 1);
        cache.put(2, 2);
        cache.put(1, 10);   // updates value, increases frequency
        
        assert_eq!(cache.get(1), 10);
        assert_eq!(cache.get(2), 2);
    }
    
    #[test]
    fn test_complex_scenario() {
        let mut cache = LFUCache::new(3);
        
        cache.put(2, 2);    // freq: 1
        cache.put(1, 1);    // freq: 1
        assert_eq!(cache.get(2), 2);  // freq: 2
        assert_eq!(cache.get(1), 1);  // freq: 2
        assert_eq!(cache.get(2), 2);  // freq: 3
        
        cache.put(3, 3);    // freq: 1
        cache.put(4, 4);    // evicts key 3 (lowest frequency = 1)
        
        assert_eq!(cache.get(3), -1);  // key 3 was evicted
        assert_eq!(cache.get(2), 2);   // key 2 has freq 3->4
        assert_eq!(cache.get(1), 1);   // key 1 has freq 2->3
        assert_eq!(cache.get(4), 4);   // key 4 has freq 1->2
    }
    
    #[test]
    fn test_get_nonexistent_key() {
        let mut cache = LFUCache::new(2);
        
        assert_eq!(cache.get(1), -1);
        
        cache.put(1, 1);
        assert_eq!(cache.get(1), 1);
        assert_eq!(cache.get(2), -1);
    }
    
    #[test]
    fn test_frequency_updates() {
        let mut cache = LFUCache::new(3);
        
        cache.put(1, 1);    // freq: 1
        cache.put(2, 2);    // freq: 1
        cache.put(3, 3);    // freq: 1
        
        cache.get(1);       // freq: 2
        cache.get(1);       // freq: 3
        cache.get(2);       // freq: 2
        
        cache.put(4, 4);    // evicts key 3 (lowest frequency = 1)
        
        assert_eq!(cache.get(3), -1);
        assert_eq!(cache.get(1), 1);
        assert_eq!(cache.get(2), 2);
        assert_eq!(cache.get(4), 4);
    }
    
    #[test]
    fn test_lru_within_same_frequency() {
        let mut cache = LFUCache::new(2);
        
        cache.put(1, 1);
        cache.put(2, 2);
        // Both have frequency 1, key 1 was inserted first
        
        cache.put(3, 3);    // evicts key 1 (LRU among same frequency)
        
        assert_eq!(cache.get(1), -1);
        assert_eq!(cache.get(2), 2);
        assert_eq!(cache.get(3), 3);
    }
    
    #[test]
    fn test_large_capacity() {
        let mut cache = LFUCache::new(100);
        
        // Fill cache
        for i in 1..=100 {
            cache.put(i, i * 10);
        }
        
        // Access some keys to increase frequency
        for i in 1..=50 {
            assert_eq!(cache.get(i), i * 10);
        }
        
        // Add new key, should evict LFU
        cache.put(101, 1010);
        
        // Keys 51-100 should have frequency 1, key 51 should be evicted (LRU)
        assert_eq!(cache.get(51), -1);
        assert_eq!(cache.get(101), 1010);
        assert_eq!(cache.get(1), 10);  // This has frequency 2
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let test_cases = vec![
            vec![
                ("put", 1, 1),
                ("put", 2, 2),
                ("get", 1, 1),
                ("put", 3, 3),
                ("get", 2, -1),
                ("get", 3, 3),
                ("put", 4, 4),
                ("get", 1, -1),
                ("get", 3, 3),
                ("get", 4, 4),
            ],
        ];
        
        for ops in test_cases {
            let mut cache1 = LFUCacheHashMap::new(2);
            let mut cache2 = LFUCacheBTreeMap::new(2);
            let mut cache3 = LFUCacheVector::new(2);
            let mut cache4 = LFUCacheTwoLevel::new(2);
            let mut cache5 = LFUCacheBalanced::new(2);
            
            for (op, key, expected) in ops {
                match op {
                    "put" => {
                        cache1.put(key, expected);
                        cache2.put(key, expected);
                        cache3.put(key, expected);
                        cache4.put(key, expected);
                        cache5.put(key, expected);
                    }
                    "get" => {
                        let result1 = cache1.get(key);
                        let result2 = cache2.get(key);
                        let result3 = cache3.get(key);
                        let result4 = cache4.get(key);
                        let result5 = cache5.get(key);
                        
                        assert_eq!(result1, expected, "HashMap approach failed");
                        assert_eq!(result2, expected, "BTreeMap approach failed");
                        assert_eq!(result3, expected, "Vector approach failed");
                        assert_eq!(result4, expected, "TwoLevel approach failed");
                        assert_eq!(result5, expected, "Balanced approach failed");
                    }
                    _ => {}
                }
            }
        }
    }
}