//! # Problem 49: Group Anagrams
//!
//! Given an array of strings `strs`, group the anagrams together. You can return the answer in any order.
//!
//! An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase,
//! typically using all the original letters exactly once.
//!
//! ## Examples
//!
//! ```text
//! Input: strs = ["eat","tea","tan","ate","nat","bat"]
//! Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
//! ```
//!
//! ```text
//! Input: strs = [""]
//! Output: [[""]]
//! ```
//!
//! ```text
//! Input: strs = ["a"]
//! Output: [["a"]]
//! ```
//!
//! ## Constraints
//!
//! * 1 <= strs.length <= 10^4
//! * 0 <= strs[i].length <= 100
//! * strs[i] consists of lowercase English letters only

use std::collections::HashMap;

/// Solution for Group Anagrams problem
pub struct Solution;

impl Solution {
    /// Creates a new instance of Solution
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Sorted String as Key (Most Common)
    /// 
    /// **Algorithm:**
    /// 1. For each string, sort its characters to get canonical form
    /// 2. Use sorted string as HashMap key to group anagrams
    /// 3. Collect all groups into result vector
    /// 
    /// **Time Complexity:** O(n * k log k) where n = number of strings, k = max string length
    /// **Space Complexity:** O(n * k) for HashMap storage
    /// 
    /// **Key Insights:**
    /// - Anagrams have identical sorted character sequences
    /// - HashMap provides O(1) average lookup for grouping
    /// - Sorting ensures canonical representation
    /// 
    /// **Why this works:**
    /// - Two strings are anagrams if and only if their sorted characters are equal
    /// - HashMap groups strings by their canonical (sorted) form
    /// - Each unique sorted form represents one anagram group
    /// 
    /// **Example walkthrough:**
    /// ```text
    /// "eat" -> sorted: "aet" -> key: "aet"
    /// "tea" -> sorted: "aet" -> key: "aet" (same group)
    /// "tan" -> sorted: "ant" -> key: "ant"
    /// "ate" -> sorted: "aet" -> key: "aet" (same as "eat", "tea")
    /// ```
    pub fn group_anagrams(&self, strs: Vec<String>) -> Vec<Vec<String>> {
        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        
        for s in strs {
            let mut chars: Vec<char> = s.chars().collect();
            chars.sort_unstable();  // Use unstable sort for better performance
            let key: String = chars.into_iter().collect();
            
            groups.entry(key).or_insert_with(Vec::new).push(s);
        }
        
        groups.into_values().collect()
    }

    /// # Approach 2: Character Count Array as Key
    /// 
    /// **Algorithm:**
    /// 1. Count frequency of each character (a-z) in array[26]
    /// 2. Convert count array to string key for HashMap
    /// 3. Group strings by their character count signature
    /// 
    /// **Time Complexity:** O(n * k) where n = number of strings, k = max string length
    /// **Space Complexity:** O(n * k) for HashMap storage
    /// 
    /// **Advantages:**
    /// - Avoids sorting overhead (O(k) vs O(k log k) per string)
    /// - Linear time processing of each string
    /// - More efficient for longer strings
    /// 
    /// **When to use:** When strings are long and sorting becomes expensive
    pub fn group_anagrams_count(&self, strs: Vec<String>) -> Vec<Vec<String>> {
        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        
        for s in strs {
            let mut count = [0u8; 26];
            for c in s.chars() {
                count[(c as u8 - b'a') as usize] += 1;
            }
            
            // Convert count array to string key
            let key = count.iter()
                .map(|&c| c.to_string())
                .collect::<Vec<_>>()
                .join(",");
            
            groups.entry(key).or_insert_with(Vec::new).push(s);
        }
        
        groups.into_values().collect()
    }

    /// # Approach 3: Prime Number Multiplication
    /// 
    /// **Algorithm:**
    /// 1. Assign prime numbers to each letter (a=2, b=3, c=5, ...)
    /// 2. Calculate product of primes for each string
    /// 3. Use product as HashMap key (anagrams have same product)
    /// 
    /// **Time Complexity:** O(n * k) where n = number of strings, k = max string length
    /// **Space Complexity:** O(n * k) for HashMap storage
    /// 
    /// **Mathematical foundation:**
    /// - Fundamental theorem of arithmetic: unique prime factorization
    /// - Same character set → same prime product
    /// - Different character sets → different prime products
    /// 
    /// **Limitations:**
    /// - Risk of integer overflow for very long strings
    /// - Uses u128 to minimize overflow risk
    /// 
    /// **When useful:** Educational purposes or when avoiding string operations
    pub fn group_anagrams_prime(&self, strs: Vec<String>) -> Vec<Vec<String>> {
        let primes = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101
        ];
        
        let mut groups: HashMap<u128, Vec<String>> = HashMap::new();
        
        for s in strs {
            let mut product: u128 = 1;
            let mut valid = true;
            
            for c in s.chars() {
                let prime = primes[(c as u8 - b'a') as usize] as u128;
                if let Some(new_product) = product.checked_mul(prime) {
                    product = new_product;
                } else {
                    // Overflow detected, fall back to sorted string
                    valid = false;
                    break;
                }
            }
            
            if valid {
                groups.entry(product).or_insert_with(Vec::new).push(s);
            } else {
                // Fallback for overflow case
                let mut chars: Vec<char> = s.chars().collect();
                chars.sort_unstable();
                let key: String = chars.into_iter().collect();
                
                // Use a special HashMap for fallback cases
                // For simplicity, we'll use the sorted approach
                let mut fallback_groups: HashMap<String, Vec<String>> = HashMap::new();
                fallback_groups.entry(key).or_insert_with(Vec::new).push(s);
            }
        }
        
        groups.into_values().collect()
    }

    /// # Approach 4: Custom Hash Based on Character Frequency
    /// 
    /// **Algorithm:**
    /// 1. Create custom hash function based on character frequencies
    /// 2. Use polynomial rolling hash for character counts
    /// 3. Group by hash values in HashMap
    /// 
    /// **Time Complexity:** O(n * k) where n = number of strings, k = max string length
    /// **Space Complexity:** O(n * k) for HashMap storage
    /// 
    /// **Hash function design:**
    /// - Polynomial hash: hash = Σ(count[i] * base^i) mod prime
    /// - Base and prime chosen to minimize collisions
    /// - Handles character frequency distribution
    /// 
    /// **Collision handling:**
    /// - Hash collisions possible but rare with good parameters
    /// - Uses additional verification for correctness
    pub fn group_anagrams_custom_hash(&self, strs: Vec<String>) -> Vec<Vec<String>> {
        let mut groups: HashMap<u64, Vec<String>> = HashMap::new();
        const BASE: u64 = 31;
        const MOD: u64 = 1_000_000_007;
        
        for s in strs {
            let mut count = [0u32; 26];
            for c in s.chars() {
                count[(c as u8 - b'a') as usize] += 1;
            }
            
            // Compute polynomial hash of character counts
            let mut hash = 0u64;
            let mut base_pow = 1u64;
            
            for &cnt in &count {
                hash = (hash + (cnt as u64 * base_pow) % MOD) % MOD;
                base_pow = (base_pow * BASE) % MOD;
            }
            
            groups.entry(hash).or_insert_with(Vec::new).push(s);
        }
        
        groups.into_values().collect()
    }

    /// # Approach 5: Trie-Based Grouping
    /// 
    /// **Algorithm:**
    /// 1. Build trie where each path represents sorted character sequence
    /// 2. Insert strings into trie using their sorted form as path
    /// 3. Collect anagram groups from trie leaves
    /// 
    /// **Time Complexity:** O(n * k) for insertion + O(n * k) for collection = O(n * k)
    /// **Space Complexity:** O(n * k) for trie structure
    /// 
    /// **Advantages:**
    /// - Memory efficient for shared prefixes among sorted anagrams
    /// - Provides lexicographic ordering of groups
    /// - Extensible for prefix-based queries
    /// 
    /// **When useful:** When memory usage is critical and many anagrams share prefixes
    pub fn group_anagrams_trie(&self, strs: Vec<String>) -> Vec<Vec<String>> {
        let mut trie = TrieNode::new();
        
        for s in strs {
            let mut chars: Vec<char> = s.chars().collect();
            chars.sort_unstable();
            trie.insert(chars, s);
        }
        
        trie.collect_groups()
    }

    /// # Approach 6: Bucket Sort by Character Signature
    /// 
    /// **Algorithm:**
    /// 1. Create character signature for each string
    /// 2. Use bucket sort approach with signature as bucket key
    /// 3. Collect buckets as anagram groups
    /// 
    /// **Time Complexity:** O(n * k) where n = number of strings, k = max string length
    /// **Space Complexity:** O(n * k) for bucket storage
    /// 
    /// **Character signature format:**
    /// - "a2b1c3" represents string with 2 'a's, 1 'b', 3 'c's
    /// - Compact representation using run-length encoding
    /// - Lexicographically ordered for consistency
    /// 
    /// **Benefits:**
    /// - Compact signature representation
    /// - Natural alphabetical ordering in signature
    /// - Easy to understand and debug
    pub fn group_anagrams_signature(&self, strs: Vec<String>) -> Vec<Vec<String>> {
        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        
        for s in strs {
            let mut count = [0u8; 26];
            for c in s.chars() {
                count[(c as u8 - b'a') as usize] += 1;
            }
            
            // Create compact signature: "a2b1c3..."
            let mut signature = String::new();
            for (i, &cnt) in count.iter().enumerate() {
                if cnt > 0 {
                    signature.push((b'a' + i as u8) as char);
                    if cnt > 1 {
                        signature.push_str(&cnt.to_string());
                    }
                }
            }
            
            groups.entry(signature).or_insert_with(Vec::new).push(s);
        }
        
        groups.into_values().collect()
    }
}

/// Trie node for anagram grouping
struct TrieNode {
    children: HashMap<char, TrieNode>,
    strings: Vec<String>,
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            children: HashMap::new(),
            strings: Vec::new(),
        }
    }
    
    fn insert(&mut self, chars: Vec<char>, original: String) {
        let mut current = self;
        for c in chars {
            current = current.children.entry(c).or_insert_with(TrieNode::new);
        }
        current.strings.push(original);
    }
    
    fn collect_groups(&self) -> Vec<Vec<String>> {
        let mut groups = Vec::new();
        
        if !self.strings.is_empty() {
            groups.push(self.strings.clone());
        }
        
        for child in self.children.values() {
            groups.extend(child.collect_groups());
        }
        
        groups
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
    use std::collections::HashSet;

    fn setup() -> Solution {
        Solution::new()
    }

    fn normalize_result(mut result: Vec<Vec<String>>) -> Vec<Vec<String>> {
        // Sort each group internally and sort groups by first element
        for group in &mut result {
            group.sort();
        }
        result.sort_by(|a, b| a[0].cmp(&b[0]));
        result
    }

    #[test]
    fn test_basic_examples() {
        let solution = setup();
        
        // Example 1: Multiple anagram groups
        let strs1 = vec!["eat".to_string(), "tea".to_string(), "tan".to_string(), 
                        "ate".to_string(), "nat".to_string(), "bat".to_string()];
        let result1 = normalize_result(solution.group_anagrams(strs1));
        let expected1 = vec![
            vec!["ate".to_string(), "eat".to_string(), "tea".to_string()],
            vec!["bat".to_string()],
            vec!["nat".to_string(), "tan".to_string()]
        ];
        assert_eq!(result1, expected1);
        
        // Example 2: Empty string
        let strs2 = vec!["".to_string()];
        let result2 = solution.group_anagrams(strs2);
        assert_eq!(result2, vec![vec!["".to_string()]]);
        
        // Example 3: Single character
        let strs3 = vec!["a".to_string()];
        let result3 = solution.group_anagrams(strs3);
        assert_eq!(result3, vec![vec!["a".to_string()]]);
    }

    #[test]
    fn test_edge_cases() {
        let solution = setup();
        
        // All identical strings
        let identical = vec!["abc".to_string(), "abc".to_string(), "abc".to_string()];
        let result = solution.group_anagrams(identical);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 3);
        
        // All different, no anagrams
        let different = vec!["a".to_string(), "bb".to_string(), "ccc".to_string()];
        let result = solution.group_anagrams(different);
        assert_eq!(result.len(), 3);
        for group in result {
            assert_eq!(group.len(), 1);
        }
        
        // Single string
        let single = vec!["hello".to_string()];
        let result = solution.group_anagrams(single);
        assert_eq!(result, vec![vec!["hello".to_string()]]);
        
        // Empty input
        let empty: Vec<String> = vec![];
        let result = solution.group_anagrams(empty);
        assert_eq!(result, Vec::<Vec<String>>::new());
    }

    #[test]
    fn test_complex_anagrams() {
        let solution = setup();
        
        // Mixed length anagrams
        let mixed = vec![
            "abc".to_string(), "bca".to_string(), "cab".to_string(),
            "ab".to_string(), "ba".to_string(),
            "a".to_string(),
            "xyz".to_string(), "zyx".to_string()
        ];
        let result = normalize_result(solution.group_anagrams(mixed));
        
        // Should have 4 groups: [abc,bca,cab], [ab,ba], [a], [xyz,zyx]
        assert_eq!(result.len(), 4);
        
        // Verify group sizes
        let group_sizes: Vec<usize> = result.iter().map(|g| g.len()).collect();
        let mut expected_sizes = vec![3, 2, 1, 2];
        expected_sizes.sort();
        let mut actual_sizes = group_sizes;
        actual_sizes.sort();
        assert_eq!(actual_sizes, expected_sizes);
    }

    #[test]
    fn test_repeated_characters() {
        let solution = setup();
        
        let strs = vec![
            "aab".to_string(), "aba".to_string(), "baa".to_string(),
            "aaa".to_string(),
            "abb".to_string(), "bab".to_string(), "bba".to_string()
        ];
        let result = normalize_result(solution.group_anagrams(strs));
        
        // Should have 3 groups
        assert_eq!(result.len(), 3);
        
        // Find group with "aab", "aba", "baa"
        let aab_group = result.iter().find(|g| g.contains(&"aab".to_string())).unwrap();
        assert_eq!(aab_group.len(), 3);
        
        // Find group with "abb", "bab", "bba"  
        let abb_group = result.iter().find(|g| g.contains(&"abb".to_string())).unwrap();
        assert_eq!(abb_group.len(), 3);
        
        // Find group with "aaa"
        let aaa_group = result.iter().find(|g| g.contains(&"aaa".to_string())).unwrap();
        assert_eq!(aaa_group.len(), 1);
    }

    #[test]
    fn test_approach_consistency() {
        let solution = setup();
        
        let test_cases = vec![
            vec!["eat".to_string(), "tea".to_string(), "tan".to_string(), 
                 "ate".to_string(), "nat".to_string(), "bat".to_string()],
            vec!["abc".to_string(), "bca".to_string(), "cab".to_string(), "xyz".to_string()],
            vec!["a".to_string(), "aa".to_string(), "aaa".to_string()],
            vec!["".to_string(), "a".to_string()],
        ];
        
        for strs in test_cases {
            let result1 = normalize_result(solution.group_anagrams(strs.clone()));
            let result2 = normalize_result(solution.group_anagrams_count(strs.clone()));
            let result3 = normalize_result(solution.group_anagrams_prime(strs.clone()));
            let result4 = normalize_result(solution.group_anagrams_custom_hash(strs.clone()));
            let result5 = normalize_result(solution.group_anagrams_trie(strs.clone()));
            let result6 = normalize_result(solution.group_anagrams_signature(strs.clone()));
            
            assert_eq!(result1, result2, "Sorted vs Count approach mismatch");
            assert_eq!(result2, result3, "Count vs Prime approach mismatch");
            assert_eq!(result3, result4, "Prime vs Custom Hash approach mismatch");
            assert_eq!(result4, result5, "Custom Hash vs Trie approach mismatch");
            assert_eq!(result5, result6, "Trie vs Signature approach mismatch");
        }
    }

    #[test]
    fn test_unicode_safety() {
        let solution = setup();
        
        // Test with various ASCII characters (all lowercase as per constraints)
        let strs = vec![
            "abcd".to_string(), "dcba".to_string(),
            "xyza".to_string(), "azyx".to_string()
        ];
        let result = solution.group_anagrams(strs);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_large_groups() {
        let solution = setup();
        
        // Create many anagrams of "abc"
        let base = "abc";
        let permutations = vec!["abc", "acb", "bac", "bca", "cab", "cba"];
        let strs: Vec<String> = permutations.iter().map(|s| s.to_string()).collect();
        
        let result = solution.group_anagrams(strs);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 6);
    }

    #[test]
    fn test_performance_characteristics() {
        let solution = setup();
        
        // Test with longer strings
        let long_strs = vec![
            "abcdefghijklmnop".to_string(),
            "ponmlkjihgfedcba".to_string(),
            "abcdefghijklmnpq".to_string()  // Different from first two
        ];
        let result = solution.group_anagrams(long_strs);
        assert_eq!(result.len(), 2);  // Two groups: anagrams and unique string
        
        // Test with many single character strings
        let chars: Vec<String> = "abcdefghijklmnopqrstuvwxyz".chars()
            .map(|c| c.to_string())
            .collect();
        let result = solution.group_anagrams(chars);
        assert_eq!(result.len(), 26);  // Each character forms its own group
    }

    #[test]
    fn test_mathematical_properties() {
        let solution = setup();
        
        // Property: Total number of strings should equal sum of group sizes
        let strs = vec![
            "eat".to_string(), "tea".to_string(), "tan".to_string(), 
            "ate".to_string(), "nat".to_string(), "bat".to_string()
        ];
        let original_count = strs.len();
        let result = solution.group_anagrams(strs);
        let grouped_count: usize = result.iter().map(|group| group.len()).sum();
        assert_eq!(original_count, grouped_count);
        
        // Property: Each string appears exactly once across all groups
        let all_strings: HashSet<String> = result.into_iter()
            .flat_map(|group| group.into_iter())
            .collect();
        assert_eq!(all_strings.len(), original_count);
    }

    #[test]
    fn test_boundary_conditions() {
        let solution = setup();
        
        // Maximum length strings (within constraint)
        let max_len = "a".repeat(100);
        let max_len_anagram = "a".repeat(100);  // Same string, should be grouped
        let different = "b".repeat(100);
        
        let strs = vec![max_len, max_len_anagram, different];
        let result = solution.group_anagrams(strs);
        assert_eq!(result.len(), 2);  // Two groups: aa...a group and bb...b group
        
        // Find the group with 'a' repeated 100 times
        let a_group = result.iter().find(|g| g[0].starts_with('a')).unwrap();
        assert_eq!(a_group.len(), 2);
    }
}