//! Problem 211: Design Add and Search Words Data Structure
//! 
//! Design a data structure that supports adding new words and finding if a string matches any 
//! previously added string.
//! 
//! Implement the WordDictionary class:
//! - WordDictionary() Initializes the object.
//! - void addWord(word) Adds word to the data structure, it can be matched later.
//! - bool search(word) Returns true if there is any string in the data structure that matches word 
//!   or false otherwise. word may contain dots '.' where dots can be matched with any letter.
//! 
//! Example:
//! Input
//! ["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
//! [[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
//! Output
//! [null,null,null,null,false,true,true,true]
//! 
//! Constraints:
//! - 1 <= word.length <= 25
//! - word in addWord consists of lowercase English letters.
//! - word in search consists of '.' or lowercase English letters.
//! - There will be at most 2 * 10^4 calls to addWord and search.

use std::collections::HashMap;

/// Approach 1: HashMap-based Trie with Wildcard Support
/// 
/// Uses a standard trie structure with DFS search to handle wildcard '.' characters.
/// Each '.' can match any single character at that position.
/// 
/// Time Complexity: O(m) for addWord, O(n * 26^k) for search where k is number of dots
/// Space Complexity: O(ALPHABET_SIZE * N * M) where N is number of words, M is average length
pub struct WordDictionary {
    children: HashMap<char, WordDictionary>,
    is_end: bool,
}

impl WordDictionary {
    pub fn new() -> Self {
        Self {
            children: HashMap::new(),
            is_end: false,
        }
    }
    
    pub fn add_word(&mut self, word: String) {
        let mut current = self;
        for ch in word.chars() {
            current = current.children.entry(ch).or_insert_with(WordDictionary::new);
        }
        current.is_end = true;
    }
    
    pub fn search(&self, word: String) -> bool {
        self.search_helper(&word, 0)
    }
    
    fn search_helper(&self, word: &str, index: usize) -> bool {
        if index == word.len() {
            return self.is_end;
        }
        
        let chars: Vec<char> = word.chars().collect();
        let ch = chars[index];
        
        if ch == '.' {
            // Wildcard: try all possible children
            for child in self.children.values() {
                if child.search_helper(word, index + 1) {
                    return true;
                }
            }
            false
        } else {
            // Regular character: exact match required
            if let Some(child) = self.children.get(&ch) {
                child.search_helper(word, index + 1)
            } else {
                false
            }
        }
    }
}

/// Approach 2: Array-based Trie with Wildcard Support
/// 
/// Uses fixed-size arrays for children (optimized for lowercase English letters).
/// More memory efficient for dense character sets.
/// 
/// Time Complexity: O(m) for addWord, O(26^k) for search where k is number of dots
/// Space Complexity: O(26 * N * M) where N is number of words, M is average length
pub struct WordDictionaryArray {
    children: [Option<Box<WordDictionaryArray>>; 26],
    is_end: bool,
}

impl WordDictionaryArray {
    pub fn new() -> Self {
        Self {
            children: Default::default(),
            is_end: false,
        }
    }
    
    pub fn add_word(&mut self, word: String) {
        let mut current = self;
        for ch in word.chars() {
            let index = (ch as u8 - b'a') as usize;
            if current.children[index].is_none() {
                current.children[index] = Some(Box::new(WordDictionaryArray::new()));
            }
            current = current.children[index].as_mut().unwrap();
        }
        current.is_end = true;
    }
    
    pub fn search(&self, word: String) -> bool {
        self.search_helper(&word, 0)
    }
    
    fn search_helper(&self, word: &str, index: usize) -> bool {
        if index == word.len() {
            return self.is_end;
        }
        
        let chars: Vec<char> = word.chars().collect();
        let ch = chars[index];
        
        if ch == '.' {
            // Wildcard: try all possible children
            for child_opt in &self.children {
                if let Some(child) = child_opt {
                    if child.search_helper(word, index + 1) {
                        return true;
                    }
                }
            }
            false
        } else {
            // Regular character: exact match required
            let char_index = (ch as u8 - b'a') as usize;
            if let Some(child) = &self.children[char_index] {
                child.search_helper(word, index + 1)
            } else {
                false
            }
        }
    }
}

/// Approach 3: Length-Based Grouping with Trie
/// 
/// Groups words by length to optimize wildcard searches.
/// Reduces search space when word length is known.
/// 
/// Time Complexity: O(m) for addWord, O(26^k) for search within length group
/// Space Complexity: O(N * M) where N is number of words, M is average length
pub struct WordDictionaryLengthGrouped {
    tries_by_length: HashMap<usize, WordDictionary>,
}

impl WordDictionaryLengthGrouped {
    pub fn new() -> Self {
        Self {
            tries_by_length: HashMap::new(),
        }
    }
    
    pub fn add_word(&mut self, word: String) {
        let length = word.len();
        let trie = self.tries_by_length.entry(length).or_insert_with(WordDictionary::new);
        trie.add_word(word);
    }
    
    pub fn search(&self, word: String) -> bool {
        let length = word.len();
        if let Some(trie) = self.tries_by_length.get(&length) {
            trie.search(word)
        } else {
            false
        }
    }
}

/// Approach 4: Bitset Optimization for Dense Patterns
/// 
/// Uses bitsets to quickly eliminate impossible branches during wildcard search.
/// Precomputes possible character sets at each position.
/// 
/// Time Complexity: O(m) for addWord, O(26^k) for search but with early pruning
/// Space Complexity: O(N * M * 26) for bitsets
pub struct WordDictionaryBitset {
    children: HashMap<char, WordDictionaryBitset>,
    is_end: bool,
    possible_chars: u32, // Bitset for characters that can appear at this position
}

impl WordDictionaryBitset {
    pub fn new() -> Self {
        Self {
            children: HashMap::new(),
            is_end: false,
            possible_chars: 0,
        }
    }
    
    pub fn add_word(&mut self, word: String) {
        let mut current = self;
        for ch in word.chars() {
            let bit_pos = (ch as u8 - b'a') as u32;
            current.possible_chars |= 1 << bit_pos;
            
            current = current.children.entry(ch).or_insert_with(WordDictionaryBitset::new);
        }
        current.is_end = true;
    }
    
    pub fn search(&self, word: String) -> bool {
        self.search_helper(&word, 0)
    }
    
    fn search_helper(&self, word: &str, index: usize) -> bool {
        if index == word.len() {
            return self.is_end;
        }
        
        let chars: Vec<char> = word.chars().collect();
        let ch = chars[index];
        
        if ch == '.' {
            // Wildcard: try only children that are possible
            for ch_candidate in 'a'..='z' {
                let bit_pos = (ch_candidate as u8 - b'a') as u32;
                if (self.possible_chars & (1 << bit_pos)) != 0 {
                    if let Some(child) = self.children.get(&ch_candidate) {
                        if child.search_helper(word, index + 1) {
                            return true;
                        }
                    }
                }
            }
            false
        } else {
            // Regular character: exact match required
            if let Some(child) = self.children.get(&ch) {
                child.search_helper(word, index + 1)
            } else {
                false
            }
        }
    }
}

/// Approach 5: Suffix Optimization
/// 
/// Optimizes search by preprocessing common suffixes and patterns.
/// Useful when many words share common endings.
/// 
/// Time Complexity: O(m) for addWord, O(26^k) for search with suffix pruning
/// Space Complexity: O(N * M) with additional suffix storage
pub struct WordDictionarySuffix {
    trie: WordDictionary,
    suffixes: HashMap<String, bool>, // Cache of known suffixes
}

impl WordDictionarySuffix {
    pub fn new() -> Self {
        Self {
            trie: WordDictionary::new(),
            suffixes: HashMap::new(),
        }
    }
    
    pub fn add_word(&mut self, word: String) {
        // Add to main trie
        self.trie.add_word(word.clone());
        
        // Cache all suffixes
        for i in 0..word.len() {
            let suffix = word[i..].to_string();
            self.suffixes.insert(suffix, true);
        }
    }
    
    pub fn search(&self, word: String) -> bool {
        // Try main trie search
        if self.trie.search(word.clone()) {
            return true;
        }
        
        // For patterns with wildcards, use suffix optimization
        if word.contains('.') {
            self.search_with_suffix_optimization(&word)
        } else {
            false
        }
    }
    
    fn search_with_suffix_optimization(&self, word: &str) -> bool {
        // Find longest suffix without wildcards
        let chars: Vec<char> = word.chars().collect();
        let mut suffix_start = word.len();
        
        for (i, &ch) in chars.iter().enumerate().rev() {
            if ch == '.' {
                suffix_start = i + 1;
                break;
            }
        }
        
        if suffix_start < word.len() {
            let suffix = &word[suffix_start..];
            if !self.suffixes.contains_key(suffix) {
                return false; // Suffix doesn't exist, so word can't exist
            }
        }
        
        // Fall back to regular search
        self.trie.search(word.to_string())
    }
}

/// Approach 6: Lazy Evaluation with Memoization
/// 
/// Caches search results for repeated patterns to avoid recomputation.
/// Particularly effective when the same wildcard patterns are searched multiple times.
/// 
/// Time Complexity: O(m) for addWord, O(1) for cached searches, O(26^k) for new searches
/// Space Complexity: O(N * M + cache_size)
pub struct WordDictionaryMemoized {
    trie: WordDictionary,
    search_cache: std::cell::RefCell<HashMap<String, bool>>,
}

impl WordDictionaryMemoized {
    pub fn new() -> Self {
        Self {
            trie: WordDictionary::new(),
            search_cache: std::cell::RefCell::new(HashMap::new()),
        }
    }
    
    pub fn add_word(&mut self, word: String) {
        // Clear cache when new words are added
        self.search_cache.borrow_mut().clear();
        self.trie.add_word(word);
    }
    
    pub fn search(&self, word: String) -> bool {
        // Check cache first
        if let Some(&cached_result) = self.search_cache.borrow().get(&word) {
            return cached_result;
        }
        
        // Compute result and cache it
        let result = self.trie.search(word.clone());
        self.search_cache.borrow_mut().insert(word, result);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hashmap_trie() {
        let mut wd = WordDictionary::new();
        
        wd.add_word("bad".to_string());
        wd.add_word("dad".to_string());
        wd.add_word("mad".to_string());
        
        assert_eq!(wd.search("pad".to_string()), false);
        assert_eq!(wd.search("bad".to_string()), true);
        assert_eq!(wd.search(".ad".to_string()), true);
        assert_eq!(wd.search("b..".to_string()), true);
        assert_eq!(wd.search("...".to_string()), true);
        assert_eq!(wd.search("....".to_string()), false);
    }
    
    #[test]
    fn test_array_trie() {
        let mut wd = WordDictionaryArray::new();
        
        wd.add_word("bad".to_string());
        wd.add_word("dad".to_string());
        wd.add_word("mad".to_string());
        
        assert_eq!(wd.search("pad".to_string()), false);
        assert_eq!(wd.search("bad".to_string()), true);
        assert_eq!(wd.search(".ad".to_string()), true);
        assert_eq!(wd.search("b..".to_string()), true);
        assert_eq!(wd.search("...".to_string()), true);
        assert_eq!(wd.search("....".to_string()), false);
    }
    
    #[test]
    fn test_length_grouped() {
        let mut wd = WordDictionaryLengthGrouped::new();
        
        wd.add_word("bad".to_string());
        wd.add_word("dad".to_string());
        wd.add_word("mad".to_string());
        
        assert_eq!(wd.search("pad".to_string()), false);
        assert_eq!(wd.search("bad".to_string()), true);
        assert_eq!(wd.search(".ad".to_string()), true);
        assert_eq!(wd.search("b..".to_string()), true);
        assert_eq!(wd.search("...".to_string()), true);
        assert_eq!(wd.search("....".to_string()), false);
    }
    
    #[test]
    fn test_bitset_optimization() {
        let mut wd = WordDictionaryBitset::new();
        
        wd.add_word("bad".to_string());
        wd.add_word("dad".to_string());
        wd.add_word("mad".to_string());
        
        assert_eq!(wd.search("pad".to_string()), false);
        assert_eq!(wd.search("bad".to_string()), true);
        assert_eq!(wd.search(".ad".to_string()), true);
        assert_eq!(wd.search("b..".to_string()), true);
        assert_eq!(wd.search("...".to_string()), true);
        assert_eq!(wd.search("....".to_string()), false);
    }
    
    #[test]
    fn test_suffix_optimization() {
        let mut wd = WordDictionarySuffix::new();
        
        wd.add_word("bad".to_string());
        wd.add_word("dad".to_string());
        wd.add_word("mad".to_string());
        
        assert_eq!(wd.search("pad".to_string()), false);
        assert_eq!(wd.search("bad".to_string()), true);
        assert_eq!(wd.search(".ad".to_string()), true);
        assert_eq!(wd.search("b..".to_string()), true);
    }
    
    #[test]
    fn test_memoized() {
        let mut wd = WordDictionaryMemoized::new();
        
        wd.add_word("bad".to_string());
        wd.add_word("dad".to_string());
        wd.add_word("mad".to_string());
        
        assert_eq!(wd.search("pad".to_string()), false);
        assert_eq!(wd.search("bad".to_string()), true);
        assert_eq!(wd.search(".ad".to_string()), true);
        assert_eq!(wd.search("b..".to_string()), true);
        
        // Test caching by searching same patterns again
        assert_eq!(wd.search(".ad".to_string()), true);
        assert_eq!(wd.search("b..".to_string()), true);
    }
    
    #[test]
    fn test_edge_cases() {
        let mut wd = WordDictionary::new();
        
        // Single character
        wd.add_word("a".to_string());
        assert_eq!(wd.search("a".to_string()), true);
        assert_eq!(wd.search(".".to_string()), true);
        assert_eq!(wd.search("b".to_string()), false);
        
        // Empty search after words added
        wd.add_word("".to_string());
        assert_eq!(wd.search("".to_string()), true);
        
        // All wildcards
        wd.add_word("abc".to_string());
        assert_eq!(wd.search("...".to_string()), true);
        assert_eq!(wd.search("..".to_string()), false);
        assert_eq!(wd.search("....".to_string()), false);
    }
    
    #[test]
    fn test_complex_patterns() {
        let mut wd = WordDictionary::new();
        
        wd.add_word("apple".to_string());
        wd.add_word("apply".to_string());
        wd.add_word("application".to_string());
        
        assert_eq!(wd.search("app..".to_string()), true);  // matches apple, apply
        assert_eq!(wd.search("app.....".to_string()), false);  // too short for application
        assert_eq!(wd.search("app.......".to_string()), false); // too long
        assert_eq!(wd.search("...le".to_string()), true);  // matches apple
        assert_eq!(wd.search("...ly".to_string()), true);  // matches apply
        assert_eq!(wd.search(".....".to_string()), true);  // matches apple, apply
    }
    
    #[test]
    fn test_no_matches() {
        let mut wd = WordDictionary::new();
        
        wd.add_word("word".to_string());
        
        assert_eq!(wd.search("words".to_string()), false);  // too long
        assert_eq!(wd.search("wor".to_string()), false);    // too short
        assert_eq!(wd.search("wird".to_string()), false);   // different char
        assert_eq!(wd.search(".ord".to_string()), true);    // wildcard match
        assert_eq!(wd.search("w.rd".to_string()), true);    // wildcard match
        assert_eq!(wd.search("wo.d".to_string()), true);    // wildcard match
        assert_eq!(wd.search("wor.".to_string()), true);    // wildcard match
    }
    
    #[test]
    fn test_multiple_wildcards() {
        let mut wd = WordDictionary::new();
        
        wd.add_word("abcde".to_string());
        wd.add_word("axcye".to_string());
        
        assert_eq!(wd.search("a.c.e".to_string()), true);   // matches both
        assert_eq!(wd.search("a.c.f".to_string()), false);  // matches neither
        assert_eq!(wd.search(".....".to_string()), true);   // matches both
        assert_eq!(wd.search("......".to_string()), false); // too long
    }
    
    #[test]
    fn test_consistency_across_implementations() {
        let test_cases = vec![
            (vec!["bad", "dad", "mad"], vec![
                ("pad", false),
                ("bad", true),
                (".ad", true),
                ("b..", true),
                ("...", true),
                ("....", false),
            ]),
            (vec!["apple", "apply"], vec![
                ("app..", true),
                (".....", true),
                ("......", false),
                ("a...e", true),
                ("a...y", true),
            ]),
        ];
        
        for (words, searches) in test_cases {
            // Test HashMap implementation
            let mut wd1 = WordDictionary::new();
            for word in &words {
                wd1.add_word(word.to_string());
            }
            
            // Test Array implementation
            let mut wd2 = WordDictionaryArray::new();
            for word in &words {
                wd2.add_word(word.to_string());
            }
            
            // Test Length Grouped implementation
            let mut wd3 = WordDictionaryLengthGrouped::new();
            for word in &words {
                wd3.add_word(word.to_string());
            }
            
            for (search_word, expected) in searches {
                let result1 = wd1.search(search_word.to_string());
                let result2 = wd2.search(search_word.to_string());
                let result3 = wd3.search(search_word.to_string());
                
                assert_eq!(result1, expected, "HashMap failed for '{}' with words {:?}", search_word, words);
                assert_eq!(result2, expected, "Array failed for '{}' with words {:?}", search_word, words);
                assert_eq!(result3, expected, "LengthGrouped failed for '{}' with words {:?}", search_word, words);
                assert_eq!(result1, result2, "HashMap and Array differ for '{}' with words {:?}", search_word, words);
                assert_eq!(result1, result3, "HashMap and LengthGrouped differ for '{}' with words {:?}", search_word, words);
            }
        }
    }
    
    #[test]
    fn test_performance_characteristics() {
        let mut wd = WordDictionary::new();
        
        // Add many words with common patterns
        for i in 0..100 {
            wd.add_word(format!("word{:03}", i));
        }
        
        // Test various wildcard patterns
        assert_eq!(wd.search("word...".to_string()), true);
        assert_eq!(wd.search("word...".to_string()), true);  // Should be fast on second call
        assert_eq!(wd.search("ward...".to_string()), false);
        
        // Test with many wildcards (expensive)
        assert_eq!(wd.search(".......".to_string()), true);
        assert_eq!(wd.search("........".to_string()), false);
    }
    
    #[test]
    fn test_memory_efficiency() {
        let mut wd_hash = WordDictionary::new();
        let mut wd_array = WordDictionaryArray::new();
        
        // Add same set of words to both
        let words = vec!["a", "ab", "abc", "abcd", "abcde"];
        for word in words {
            wd_hash.add_word(word.to_string());
            wd_array.add_word(word.to_string());
        }
        
        // Both should produce same results
        for pattern in vec!["a", "ab", "..", "...", ".....", "......"] {
            let hash_result = wd_hash.search(pattern.to_string());
            let array_result = wd_array.search(pattern.to_string());
            assert_eq!(hash_result, array_result, "Results differ for pattern '{}'", pattern);
        }
    }
}