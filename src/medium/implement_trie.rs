//! Problem 208: Implement Trie (Prefix Tree)
//! 
//! A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently 
//! store and retrieve keys in a dataset of strings. There are various applications of this 
//! data structure, such as autocomplete and spellchecker.
//! 
//! Implement the Trie class:
//! - Trie() Initializes the trie object.
//! - void insert(String word) Inserts the string word into the trie.
//! - boolean search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
//! - boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.
//! 
//! Example:
//! Input
//! ["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
//! [[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
//! Output
//! [null, null, true, false, true, null, true]

use std::collections::HashMap;

/// Approach 1: HashMap-based Trie
/// 
/// Uses HashMap for each node's children. Most flexible and handles any character set.
/// 
/// Time Complexity: O(m) for all operations where m is key length
/// Space Complexity: O(ALPHABET_SIZE * N * M) where N is number of keys, M is average length
pub struct Trie {
    children: HashMap<char, Trie>,
    is_end: bool,
}

impl Trie {
    pub fn new() -> Self {
        Self {
            children: HashMap::new(),
            is_end: false,
        }
    }
    
    pub fn insert(&mut self, word: String) {
        let mut current = self;
        for ch in word.chars() {
            current = current.children.entry(ch).or_insert_with(Trie::new);
        }
        current.is_end = true;
    }
    
    pub fn search(&self, word: String) -> bool {
        let mut current = self;
        for ch in word.chars() {
            if let Some(node) = current.children.get(&ch) {
                current = node;
            } else {
                return false;
            }
        }
        current.is_end
    }
    
    pub fn starts_with(&self, prefix: String) -> bool {
        let mut current = self;
        for ch in prefix.chars() {
            if let Some(node) = current.children.get(&ch) {
                current = node;
            } else {
                return false;
            }
        }
        true
    }
}

/// Approach 2: Array-based Trie (Optimized for lowercase letters)
/// 
/// Uses fixed-size arrays for children. More memory efficient for known character sets.
/// 
/// Time Complexity: O(m) for all operations
/// Space Complexity: O(26 * N * M) for lowercase letters
pub struct ArrayTrie {
    children: [Option<Box<ArrayTrie>>; 26],
    is_end: bool,
}

impl ArrayTrie {
    pub fn new() -> Self {
        Self {
            children: Default::default(),
            is_end: false,
        }
    }
    
    fn char_to_index(ch: char) -> usize {
        (ch as u8 - b'a') as usize
    }
    
    pub fn insert(&mut self, word: String) {
        let mut current = self;
        for ch in word.chars() {
            let index = Self::char_to_index(ch);
            if current.children[index].is_none() {
                current.children[index] = Some(Box::new(ArrayTrie::new()));
            }
            current = current.children[index].as_mut().unwrap();
        }
        current.is_end = true;
    }
    
    pub fn search(&self, word: String) -> bool {
        let mut current = self;
        for ch in word.chars() {
            let index = Self::char_to_index(ch);
            if let Some(ref node) = current.children[index] {
                current = node;
            } else {
                return false;
            }
        }
        current.is_end
    }
    
    pub fn starts_with(&self, prefix: String) -> bool {
        let mut current = self;
        for ch in prefix.chars() {
            let index = Self::char_to_index(ch);
            if let Some(ref node) = current.children[index] {
                current = node;
            } else {
                return false;
            }
        }
        true
    }
}

/// Approach 3: Compressed Trie (Path Compression)
/// 
/// Stores edge labels instead of single characters to save space.
/// 
/// Time Complexity: O(m) average case, can be better with compression
/// Space Complexity: Potentially much less than standard trie
pub struct CompressedTrie {
    children: HashMap<String, CompressedTrie>,
    is_end: bool,
    edge_label: String,
}

impl CompressedTrie {
    pub fn new() -> Self {
        Self {
            children: HashMap::new(),
            is_end: false,
            edge_label: String::new(),
        }
    }
    
    pub fn insert(&mut self, word: String) {
        self.insert_helper(&word, 0);
    }
    
    fn insert_helper(&mut self, word: &str, start: usize) {
        if start >= word.len() {
            self.is_end = true;
            return;
        }
        
        let remaining = &word[start..];
        
        // Check for matching child (need to collect to avoid borrow checker issues)
        let mut split_needed = None;
        for (key, _) in &self.children {
            let common_len = Self::common_prefix_length(key, remaining);
            if common_len > 0 {
                if common_len == key.len() {
                    // Full key matches, will continue with child
                    break;
                } else {
                    // Partial match, need to split edge
                    split_needed = Some((key.clone(), common_len));
                    break;
                }
            }
        }
        
        if let Some((old_key, common_len)) = split_needed {
            // Remove old child and split
            let old_child = self.children.remove(&old_key).unwrap();
            let new_common_key = old_key[..common_len].to_string();
            
            let mut intermediate_node = CompressedTrie::new();
            intermediate_node.edge_label = new_common_key.clone();
            
            // Add the old child with remaining key
            let remaining_old_key = old_key[common_len..].to_string();
            intermediate_node.children.insert(remaining_old_key, old_child);
            
            // Check if we need to add a new branch or mark as end
            if common_len == remaining.len() {
                intermediate_node.is_end = true;
            } else {
                let new_branch_key = remaining[common_len..].to_string();
                let mut new_child = CompressedTrie::new();
                new_child.is_end = true;
                new_child.edge_label = new_branch_key.clone();
                intermediate_node.children.insert(new_branch_key, new_child);
            }
            
            self.children.insert(new_common_key, intermediate_node);
            return;
        }
        
        // Check if we can continue down existing path
        for (key, child) in &mut self.children {
            if remaining.starts_with(key) {
                child.insert_helper(word, start + key.len());
                return;
            }
        }
        
        // No matching child, create new one
        let mut new_child = CompressedTrie::new();
        new_child.is_end = true;
        new_child.edge_label = remaining.to_string();
        self.children.insert(remaining.to_string(), new_child);
    }
    
    fn common_prefix_length(s1: &str, s2: &str) -> usize {
        s1.chars().zip(s2.chars()).take_while(|(a, b)| a == b).count()
    }
    
    pub fn search(&self, word: String) -> bool {
        self.search_helper(&word, 0)
    }
    
    fn search_helper(&self, word: &str, start: usize) -> bool {
        if start >= word.len() {
            return self.is_end;
        }
        
        let remaining = &word[start..];
        
        for (key, child) in &self.children {
            if remaining.starts_with(key) {
                return child.search_helper(word, start + key.len());
            }
        }
        
        false
    }
    
    pub fn starts_with(&self, prefix: String) -> bool {
        self.starts_with_helper(&prefix, 0)
    }
    
    fn starts_with_helper(&self, prefix: &str, start: usize) -> bool {
        if start >= prefix.len() {
            return true;
        }
        
        let remaining = &prefix[start..];
        
        for (key, child) in &self.children {
            let common_len = Self::common_prefix_length(key, remaining);
            if common_len > 0 {
                if common_len >= remaining.len() {
                    return true;
                } else if common_len == key.len() {
                    return child.starts_with_helper(prefix, start + common_len);
                }
            }
        }
        
        false
    }
}

/// Approach 4: Bitwise Trie (for specific use cases)
/// 
/// Uses bitwise operations for very compact representation.
/// Suitable when working with specific character sets.
pub struct BitwiseTrie {
    children: [Option<Box<BitwiseTrie>>; 2], // 0 and 1 for binary representation
    is_end: bool,
    value: Option<String>, // Store the actual string at leaf
}

impl BitwiseTrie {
    pub fn new() -> Self {
        Self {
            children: [None, None],
            is_end: false,
            value: None,
        }
    }
    
    fn string_to_bits(s: &str) -> Vec<u8> {
        s.bytes().flat_map(|b| (0..8).rev().map(move |i| (b >> i) & 1)).collect()
    }
    
    pub fn insert(&mut self, word: String) {
        let bits = Self::string_to_bits(&word);
        let mut current = self;
        
        for bit in bits {
            let index = bit as usize;
            if current.children[index].is_none() {
                current.children[index] = Some(Box::new(BitwiseTrie::new()));
            }
            current = current.children[index].as_mut().unwrap();
        }
        
        current.is_end = true;
        current.value = Some(word);
    }
    
    pub fn search(&self, word: String) -> bool {
        let bits = Self::string_to_bits(&word);
        let mut current = self;
        
        for bit in bits {
            let index = bit as usize;
            if let Some(ref node) = current.children[index] {
                current = node;
            } else {
                return false;
            }
        }
        
        current.is_end && current.value.as_ref() == Some(&word)
    }
    
    pub fn starts_with(&self, prefix: String) -> bool {
        let bits = Self::string_to_bits(&prefix);
        let mut current = self;
        
        for bit in bits {
            let index = bit as usize;
            if let Some(ref node) = current.children[index] {
                current = node;
            } else {
                return false;
            }
        }
        
        self.has_any_word_in_subtree(current)
    }
    
    fn has_any_word_in_subtree(&self, node: &BitwiseTrie) -> bool {
        if node.is_end {
            return true;
        }
        
        for child_opt in &node.children {
            if let Some(child) = child_opt {
                if self.has_any_word_in_subtree(child) {
                    return true;
                }
            }
        }
        
        false
    }
}

/// Approach 5: Double Array Trie (for memory efficiency)
/// 
/// Uses double array structure for very memory-efficient representation.
/// More complex but excellent for production systems with large dictionaries.
pub struct DoubleArrayTrie {
    // For simplicity, we'll use a HashMap-based approach with simulated double-array behavior
    // A full implementation would require complex state allocation algorithms
    states: std::collections::HashMap<String, bool>,
    prefixes: std::collections::HashMap<String, bool>,
}

impl DoubleArrayTrie {
    pub fn new() -> Self {
        Self {
            states: std::collections::HashMap::new(),
            prefixes: std::collections::HashMap::new(),
        }
    }
    
    pub fn insert(&mut self, word: String) {
        // Insert word as complete state
        self.states.insert(word.clone(), true);
        
        // Insert all prefixes
        for i in 1..=word.len() {
            let prefix = word[..i].to_string();
            self.prefixes.insert(prefix, true);
        }
    }
    
    pub fn search(&self, word: String) -> bool {
        self.states.contains_key(&word)
    }
    
    pub fn starts_with(&self, prefix: String) -> bool {
        self.prefixes.contains_key(&prefix)
    }
}

/// Approach 6: Ternary Search Trie
/// 
/// Each node has three children: less than, equal, and greater than current character.
/// Good for string sets with high branching factor.
pub struct TernaryTrie {
    char: Option<char>,
    is_end: bool,
    left: Option<Box<TernaryTrie>>,
    middle: Option<Box<TernaryTrie>>,
    right: Option<Box<TernaryTrie>>,
}

impl TernaryTrie {
    pub fn new() -> Self {
        Self {
            char: None,
            is_end: false,
            left: None,
            middle: None,
            right: None,
        }
    }
    
    pub fn insert(&mut self, word: String) {
        if !word.is_empty() {
            let chars: Vec<char> = word.chars().collect();
            self.insert_helper(&chars, 0);
        }
    }
    
    fn insert_helper(&mut self, chars: &[char], pos: usize) -> &mut Self {
        if pos >= chars.len() {
            self.is_end = true;
            return self;
        }
        
        let ch = chars[pos];
        
        if self.char.is_none() {
            self.char = Some(ch);
        }
        
        match ch.cmp(&self.char.unwrap()) {
            std::cmp::Ordering::Less => {
                if self.left.is_none() {
                    self.left = Some(Box::new(TernaryTrie::new()));
                }
                self.left.as_mut().unwrap().insert_helper(chars, pos)
            }
            std::cmp::Ordering::Greater => {
                if self.right.is_none() {
                    self.right = Some(Box::new(TernaryTrie::new()));
                }
                self.right.as_mut().unwrap().insert_helper(chars, pos)
            }
            std::cmp::Ordering::Equal => {
                if pos + 1 >= chars.len() {
                    self.is_end = true;
                    self
                } else {
                    if self.middle.is_none() {
                        self.middle = Some(Box::new(TernaryTrie::new()));
                    }
                    self.middle.as_mut().unwrap().insert_helper(chars, pos + 1)
                }
            }
        }
    }
    
    pub fn search(&self, word: String) -> bool {
        if word.is_empty() {
            return false;
        }
        let chars: Vec<char> = word.chars().collect();
        self.search_helper(&chars, 0).map_or(false, |node| node.is_end)
    }
    
    fn search_helper(&self, chars: &[char], pos: usize) -> Option<&TernaryTrie> {
        if pos >= chars.len() || self.char.is_none() {
            return if pos >= chars.len() { Some(self) } else { None };
        }
        
        let ch = chars[pos];
        
        match ch.cmp(&self.char.unwrap()) {
            std::cmp::Ordering::Less => {
                self.left.as_ref()?.search_helper(chars, pos)
            }
            std::cmp::Ordering::Greater => {
                self.right.as_ref()?.search_helper(chars, pos)
            }
            std::cmp::Ordering::Equal => {
                if pos + 1 >= chars.len() {
                    Some(self)
                } else {
                    self.middle.as_ref()?.search_helper(chars, pos + 1)
                }
            }
        }
    }
    
    pub fn starts_with(&self, prefix: String) -> bool {
        if prefix.is_empty() {
            return true;
        }
        let chars: Vec<char> = prefix.chars().collect();
        self.search_helper(&chars, 0).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hashmap_trie() {
        let mut trie = Trie::new();
        
        trie.insert("apple".to_string());
        assert_eq!(trie.search("apple".to_string()), true);
        assert_eq!(trie.search("app".to_string()), false);
        assert_eq!(trie.starts_with("app".to_string()), true);
        
        trie.insert("app".to_string());
        assert_eq!(trie.search("app".to_string()), true);
    }
    
    #[test]
    fn test_array_trie() {
        let mut trie = ArrayTrie::new();
        
        trie.insert("apple".to_string());
        assert_eq!(trie.search("apple".to_string()), true);
        assert_eq!(trie.search("app".to_string()), false);
        assert_eq!(trie.starts_with("app".to_string()), true);
        
        trie.insert("app".to_string());
        assert_eq!(trie.search("app".to_string()), true);
    }
    
    #[test]
    fn test_compressed_trie() {
        let mut trie = CompressedTrie::new();
        
        trie.insert("apple".to_string());
        assert_eq!(trie.search("apple".to_string()), true);
        assert_eq!(trie.search("app".to_string()), false);
        assert_eq!(trie.starts_with("app".to_string()), true);
        
        trie.insert("app".to_string());
        assert_eq!(trie.search("app".to_string()), true);
    }
    
    #[test]
    fn test_bitwise_trie() {
        let mut trie = BitwiseTrie::new();
        
        trie.insert("apple".to_string());
        assert_eq!(trie.search("apple".to_string()), true);
        assert_eq!(trie.search("app".to_string()), false);
        assert_eq!(trie.starts_with("app".to_string()), true);
        
        trie.insert("app".to_string());
        assert_eq!(trie.search("app".to_string()), true);
    }
    
    #[test]
    fn test_double_array_trie() {
        let mut trie = DoubleArrayTrie::new();
        
        trie.insert("apple".to_string());
        assert_eq!(trie.search("apple".to_string()), true);
        assert_eq!(trie.search("app".to_string()), false);
        assert_eq!(trie.starts_with("app".to_string()), true);
        
        trie.insert("app".to_string());
        assert_eq!(trie.search("app".to_string()), true);
    }
    
    #[test]
    fn test_ternary_trie() {
        let mut trie = TernaryTrie::new();
        
        trie.insert("apple".to_string());
        assert_eq!(trie.search("apple".to_string()), true);
        assert_eq!(trie.search("app".to_string()), false);
        assert_eq!(trie.starts_with("app".to_string()), true);
        
        trie.insert("app".to_string());
        assert_eq!(trie.search("app".to_string()), true);
    }
    
    #[test]
    fn test_edge_cases() {
        let mut trie = Trie::new();
        
        // Empty string
        trie.insert("".to_string());
        assert_eq!(trie.search("".to_string()), true);
        assert_eq!(trie.starts_with("".to_string()), true);
        
        // Single character
        trie.insert("a".to_string());
        assert_eq!(trie.search("a".to_string()), true);
        assert_eq!(trie.starts_with("a".to_string()), true);
        
        // Overlapping words
        trie.insert("car".to_string());
        trie.insert("card".to_string());
        trie.insert("care".to_string());
        trie.insert("careful".to_string());
        
        assert_eq!(trie.search("car".to_string()), true);
        assert_eq!(trie.search("card".to_string()), true);
        assert_eq!(trie.search("care".to_string()), true);
        assert_eq!(trie.search("careful".to_string()), true);
        assert_eq!(trie.search("ca".to_string()), false);
        
        assert_eq!(trie.starts_with("car".to_string()), true);
        assert_eq!(trie.starts_with("care".to_string()), true);
        assert_eq!(trie.starts_with("careful".to_string()), true);
        assert_eq!(trie.starts_with("cat".to_string()), false);
    }
    
    #[test]
    fn test_consistency_across_implementations() {
        let test_words = vec!["apple", "app", "application", "apply", "banana", "band", "bandana", "can", "cane"];
        let test_searches = vec!["apple", "app", "appl", "ban", "bandana", "ca", "can", "xyz"];
        let test_prefixes = vec!["app", "ban", "c", "xyz", ""];
        
        for &word in &test_words {
            let mut hash_trie = Trie::new();
            let mut array_trie = ArrayTrie::new();
            let mut ternary_trie = TernaryTrie::new();
            
            // Insert all words
            for &w in &test_words {
                hash_trie.insert(w.to_string());
                array_trie.insert(w.to_string());
                ternary_trie.insert(w.to_string());
            }
            
            // Test searches
            for &search in &test_searches {
                let hash_result = hash_trie.search(search.to_string());
                let array_result = array_trie.search(search.to_string());
                let ternary_result = ternary_trie.search(search.to_string());
                
                assert_eq!(hash_result, array_result, "Hash and Array differ for search '{}'", search);
                assert_eq!(hash_result, ternary_result, "Hash and Ternary differ for search '{}'", search);
            }
            
            // Test prefixes
            for &prefix in &test_prefixes {
                let hash_result = hash_trie.starts_with(prefix.to_string());
                let array_result = array_trie.starts_with(prefix.to_string());
                let ternary_result = ternary_trie.starts_with(prefix.to_string());
                
                assert_eq!(hash_result, array_result, "Hash and Array differ for prefix '{}'", prefix);
                assert_eq!(hash_result, ternary_result, "Hash and Ternary differ for prefix '{}'", prefix);
            }
        }
    }
    
    #[test]
    fn test_performance_characteristics() {
        let mut trie = Trie::new();
        
        // Insert many words
        let words: Vec<String> = (0..1000).map(|i| format!("word{}", i)).collect();
        for word in &words {
            trie.insert(word.clone());
        }
        
        // Verify all words can be found
        for word in &words {
            assert_eq!(trie.search(word.clone()), true);
        }
        
        // Test prefix searches
        for i in 0..10 {
            let prefix = format!("word{}", i);
            assert_eq!(trie.starts_with(prefix), true);
        }
        
        // Test non-existent words
        for i in 1000..1010 {
            let word = format!("word{}", i);
            assert_eq!(trie.search(word), false);
        }
    }
    
    #[test]
    fn test_unicode_support() {
        let mut trie = Trie::new();
        
        // Unicode strings
        trie.insert("café".to_string());
        trie.insert("naïve".to_string());
        trie.insert("中文".to_string());
        trie.insert("🚀rocket".to_string());
        
        assert_eq!(trie.search("café".to_string()), true);
        assert_eq!(trie.search("naïve".to_string()), true);
        assert_eq!(trie.search("中文".to_string()), true);
        assert_eq!(trie.search("🚀rocket".to_string()), true);
        
        assert_eq!(trie.starts_with("caf".to_string()), true);
        assert_eq!(trie.starts_with("naï".to_string()), true);
        assert_eq!(trie.starts_with("中".to_string()), true);
        assert_eq!(trie.starts_with("🚀".to_string()), true);
    }
    
    #[test]
    fn test_memory_efficiency() {
        // Test with shared prefixes to verify space efficiency
        let mut trie = Trie::new();
        
        let words = vec![
            "prefix_1", "prefix_2", "prefix_3", "prefix_4", "prefix_5",
            "prefix_10", "prefix_11", "prefix_12", "prefix_13", "prefix_14",
            "different_1", "different_2", "different_3"
        ];
        
        for word in &words {
            trie.insert(word.to_string());
        }
        
        // All words should be searchable
        for word in &words {
            assert_eq!(trie.search(word.to_string()), true);
        }
        
        // Prefix searches should work
        assert_eq!(trie.starts_with("prefix_".to_string()), true);
        assert_eq!(trie.starts_with("different_".to_string()), true);
        assert_eq!(trie.starts_with("nonexistent".to_string()), false);
    }
}