#!/usr/bin/env cargo-script

//! Interactive Problem Selector for LeetCode Solutions
//! 
//! This tool helps you navigate the collection of LeetCode problems
//! by difficulty, topic, or complexity requirements.

use std::collections::HashMap;
use std::io::{self, Write};

#[derive(Debug, Clone)]
struct Problem {
    id: u32,
    title: String,
    difficulty: Difficulty,
    topics: Vec<String>,
    time_complexity: String,
    space_complexity: String,
    file_path: String,
    approaches: u8,
}

#[derive(Debug, Clone, PartialEq)]
enum Difficulty {
    Easy,
    Medium,
    Hard,
}

impl std::fmt::Display for Difficulty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Difficulty::Easy => write!(f, "Easy"),
            Difficulty::Medium => write!(f, "Medium"),
            Difficulty::Hard => write!(f, "Hard"),
        }
    }
}

fn create_problem_database() -> Vec<Problem> {
    vec![
        // Easy Problems
        Problem {
            id: 1,
            title: "Two Sum".to_string(),
            difficulty: Difficulty::Easy,
            topics: vec!["Array".to_string(), "Hash Table".to_string()],
            time_complexity: "O(n)".to_string(),
            space_complexity: "O(n)".to_string(),
            file_path: "src/easy/two_sum.rs".to_string(),
            approaches: 3,
        },
        Problem {
            id: 20,
            title: "Valid Parentheses".to_string(),
            difficulty: Difficulty::Easy,
            topics: vec!["String".to_string(), "Stack".to_string()],
            time_complexity: "O(n)".to_string(),
            space_complexity: "O(n)".to_string(),
            file_path: "src/easy/valid_parentheses.rs".to_string(),
            approaches: 2,
        },
        Problem {
            id: 21,
            title: "Merge Two Sorted Lists".to_string(),
            difficulty: Difficulty::Easy,
            topics: vec!["Linked List".to_string(), "Recursion".to_string()],
            time_complexity: "O(m+n)".to_string(),
            space_complexity: "O(1)".to_string(),
            file_path: "src/easy/merge_two_sorted_lists.rs".to_string(),
            approaches: 2,
        },
        Problem {
            id: 121,
            title: "Best Time to Buy and Sell Stock".to_string(),
            difficulty: Difficulty::Easy,
            topics: vec!["Array".to_string(), "Dynamic Programming".to_string()],
            time_complexity: "O(n)".to_string(),
            space_complexity: "O(1)".to_string(),
            file_path: "src/easy/best_time_to_buy_and_sell_stock.rs".to_string(),
            approaches: 2,
        },
        
        // Medium Problems  
        Problem {
            id: 3,
            title: "Longest Substring Without Repeating Characters".to_string(),
            difficulty: Difficulty::Medium,
            topics: vec!["String".to_string(), "Sliding Window".to_string(), "Hash Table".to_string()],
            time_complexity: "O(n)".to_string(),
            space_complexity: "O(min(m,n))".to_string(),
            file_path: "src/medium/longest_substring_without_repeating_characters.rs".to_string(),
            approaches: 4,
        },
        Problem {
            id: 15,
            title: "3Sum".to_string(),
            difficulty: Difficulty::Medium,
            topics: vec!["Array".to_string(), "Two Pointers".to_string(), "Sorting".to_string()],
            time_complexity: "O(n²)".to_string(),
            space_complexity: "O(1)".to_string(),
            file_path: "src/medium/three_sum.rs".to_string(),
            approaches: 3,
        },
        Problem {
            id: 53,
            title: "Maximum Subarray".to_string(),
            difficulty: Difficulty::Medium,
            topics: vec!["Array".to_string(), "Dynamic Programming".to_string(), "Divide and Conquer".to_string()],
            time_complexity: "O(n)".to_string(),
            space_complexity: "O(1)".to_string(),
            file_path: "src/medium/maximum_subarray.rs".to_string(),
            approaches: 3,
        },
        Problem {
            id: 146,
            title: "LRU Cache".to_string(),
            difficulty: Difficulty::Medium,
            topics: vec!["Hash Table".to_string(), "Linked List".to_string(), "Design".to_string()],
            time_complexity: "O(1)".to_string(),
            space_complexity: "O(capacity)".to_string(),
            file_path: "src/medium/lru_cache.rs".to_string(),
            approaches: 1,
        },
        Problem {
            id: 200,
            title: "Number of Islands".to_string(),
            difficulty: Difficulty::Medium,
            topics: vec!["Array".to_string(), "DFS".to_string(), "BFS".to_string(), "Union Find".to_string()],
            time_complexity: "O(M×N)".to_string(),
            space_complexity: "O(M×N)".to_string(),
            file_path: "src/medium/number_of_islands.rs".to_string(),
            approaches: 3,
        },
        
        // Hard Problems
        Problem {
            id: 4,
            title: "Median of Two Sorted Arrays".to_string(),
            difficulty: Difficulty::Hard,
            topics: vec!["Array".to_string(), "Binary Search".to_string(), "Divide and Conquer".to_string()],
            time_complexity: "O(log(min(m,n)))".to_string(),
            space_complexity: "O(1)".to_string(),
            file_path: "src/hard/median_of_two_sorted_arrays.rs".to_string(),
            approaches: 3,
        },
        Problem {
            id: 23,
            title: "Merge k Sorted Lists".to_string(),
            difficulty: Difficulty::Hard,
            topics: vec!["Linked List".to_string(), "Divide and Conquer".to_string(), "Heap".to_string()],
            time_complexity: "O(N×log(k))".to_string(),
            space_complexity: "O(log k)".to_string(),
            file_path: "src/hard/merge_k_sorted_lists.rs".to_string(),
            approaches: 4,
        },
        Problem {
            id: 42,
            title: "Trapping Rain Water".to_string(),
            difficulty: Difficulty::Hard,
            topics: vec!["Array".to_string(), "Two Pointers".to_string(), "Dynamic Programming".to_string(), "Stack".to_string()],
            time_complexity: "O(n)".to_string(),
            space_complexity: "O(1)".to_string(),
            file_path: "src/hard/trapping_rain_water.rs".to_string(),
            approaches: 4,
        },
        Problem {
            id: 72,
            title: "Edit Distance".to_string(),
            difficulty: Difficulty::Hard,
            topics: vec!["String".to_string(), "Dynamic Programming".to_string()],
            time_complexity: "O(m×n)".to_string(),
            space_complexity: "O(m×n)".to_string(),
            file_path: "src/hard/edit_distance.rs".to_string(),
            approaches: 3,
        },
        Problem {
            id: 146,
            title: "LRU Cache".to_string(),
            difficulty: Difficulty::Hard,
            topics: vec!["Hash Table".to_string(), "Linked List".to_string(), "Design".to_string()],
            time_complexity: "O(1)".to_string(),
            space_complexity: "O(capacity)".to_string(),
            file_path: "src/hard/lru_cache.rs".to_string(),
            approaches: 1,
        },
    ]
}

fn display_menu() {
    println!("\n🚀 LeetCode Problem Selector");
    println!("============================");
    println!("1. Browse by Difficulty");
    println!("2. Search by Topic");
    println!("3. Filter by Complexity");
    println!("4. View Problem Details");
    println!("5. Random Problem");
    println!("6. Learning Path Recommendations");
    println!("7. Show Statistics");
    println!("8. Exit");
    print!("\nSelect an option (1-8): ");
    io::stdout().flush().unwrap();
}

fn browse_by_difficulty(problems: &[Problem]) {
    println!("\n📊 Problems by Difficulty:");
    
    let difficulties = [Difficulty::Easy, Difficulty::Medium, Difficulty::Hard];
    
    for difficulty in &difficulties {
        let filtered: Vec<_> = problems.iter().filter(|p| p.difficulty == *difficulty).collect();
        println!("\n{} ({} problems):", difficulty, filtered.len());
        
        for problem in filtered.iter().take(5) {
            println!("  {}. {} [{}]", problem.id, problem.title, 
                     problem.topics.join(", "));
        }
        
        if filtered.len() > 5 {
            println!("  ... and {} more", filtered.len() - 5);
        }
    }
}

fn search_by_topic(problems: &[Problem]) {
    let mut topic_map: HashMap<String, Vec<&Problem>> = HashMap::new();
    
    for problem in problems {
        for topic in &problem.topics {
            topic_map.entry(topic.clone()).or_insert_with(Vec::new).push(problem);
        }
    }
    
    println!("\n🏷️  Available Topics:");
    let mut topics: Vec<_> = topic_map.keys().collect();
    topics.sort();
    
    for (i, topic) in topics.iter().enumerate() {
        print!("{:<20}", topic);
        if (i + 1) % 3 == 0 {
            println!();
        }
    }
    println!();
    
    print!("\nEnter topic to explore: ");
    io::stdout().flush().unwrap();
    
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let topic = input.trim();
    
    if let Some(topic_problems) = topic_map.get(topic) {
        println!("\n{} problems found for '{}':", topic_problems.len(), topic);
        
        for problem in topic_problems.iter().take(10) {
            println!("  {}. {} [{}] - {} time, {} space", 
                     problem.id, problem.title, problem.difficulty,
                     problem.time_complexity, problem.space_complexity);
        }
        
        if topic_problems.len() > 10 {
            println!("  ... and {} more", topic_problems.len() - 10);
        }
    } else {
        println!("❌ Topic '{}' not found. Try one from the list above.", topic);
    }
}

fn filter_by_complexity(problems: &[Problem]) {
    println!("\n⚡ Filter by Complexity:");
    println!("1. Time Complexity");  
    println!("2. Space Complexity");
    print!("Choose filter type (1-2): ");
    io::stdout().flush().unwrap();
    
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    
    match input.trim() {
        "1" => filter_by_time_complexity(problems),
        "2" => filter_by_space_complexity(problems),
        _ => println!("❌ Invalid option"),
    }
}

fn filter_by_time_complexity(problems: &[Problem]) {
    let complexities = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n²)", "O(2^n)"];
    
    println!("\nAvailable Time Complexities:");
    for (i, complexity) in complexities.iter().enumerate() {
        println!("{}. {}", i + 1, complexity);
    }
    
    print!("Select complexity (1-{}): ", complexities.len());
    io::stdout().flush().unwrap();
    
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    
    if let Ok(choice) = input.trim().parse::<usize>() {
        if choice > 0 && choice <= complexities.len() {
            let target_complexity = complexities[choice - 1];
            let filtered: Vec<_> = problems.iter()
                .filter(|p| p.time_complexity.contains(target_complexity))
                .collect();
            
            println!("\n{} problems with {} time complexity:", filtered.len(), target_complexity);
            for problem in filtered.iter().take(10) {
                println!("  {}. {} [{}] - {}", 
                         problem.id, problem.title, problem.difficulty, problem.topics.join(", "));
            }
        }
    }
}

fn filter_by_space_complexity(problems: &[Problem]) {
    let complexities = ["O(1)", "O(log n)", "O(n)", "O(n²)"];
    
    println!("\nAvailable Space Complexities:");
    for (i, complexity) in complexities.iter().enumerate() {
        println!("{}. {}", i + 1, complexity);
    }
    
    print!("Select complexity (1-{}): ", complexities.len());
    io::stdout().flush().unwrap();
    
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    
    if let Ok(choice) = input.trim().parse::<usize>() {
        if choice > 0 && choice <= complexities.len() {
            let target_complexity = complexities[choice - 1];
            let filtered: Vec<_> = problems.iter()
                .filter(|p| p.space_complexity.contains(target_complexity))
                .collect();
            
            println!("\n{} problems with {} space complexity:", filtered.len(), target_complexity);
            for problem in filtered.iter().take(10) {
                println!("  {}. {} [{}] - {}", 
                         problem.id, problem.title, problem.difficulty, problem.topics.join(", "));
            }
        }
    }
}

fn view_problem_details(problems: &[Problem]) {
    print!("\nEnter problem ID or title: ");
    io::stdout().flush().unwrap();
    
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let query = input.trim();
    
    let found_problem = if let Ok(id) = query.parse::<u32>() {
        problems.iter().find(|p| p.id == id)
    } else {
        problems.iter().find(|p| p.title.to_lowercase().contains(&query.to_lowercase()))
    };
    
    if let Some(problem) = found_problem {
        println!("\n📋 Problem Details:");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("ID: {}", problem.id);
        println!("Title: {}", problem.title);
        println!("Difficulty: {}", problem.difficulty);
        println!("Topics: {}", problem.topics.join(", "));
        println!("Time Complexity: {}", problem.time_complexity);
        println!("Space Complexity: {}", problem.space_complexity);
        println!("File Path: {}", problem.file_path);
        println!("Approaches Implemented: {}", problem.approaches);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        println!("\n💡 Quick Access Commands:");
        println!("  View implementation: code {}", problem.file_path);
        println!("  Run tests: cargo test {}::", problem.file_path.split('/').last().unwrap().replace(".rs", ""));
        println!("  Run benchmarks: cargo bench {}", problem.title.to_lowercase().replace(' ', "_"));
    } else {
        println!("❌ Problem not found. Try a different ID or title.");
    }
}

fn random_problem(problems: &[Problem]) {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    let seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let index = (seed as usize) % problems.len();
    let problem = &problems[index];
    
    println!("\n🎲 Random Problem Selected:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("{} - {} [{}]", problem.id, problem.title, problem.difficulty);
    println!("Topics: {}", problem.topics.join(", "));
    println!("Complexity: {} time, {} space", problem.time_complexity, problem.space_complexity);
    println!("File: {}", problem.file_path);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}

fn learning_path_recommendations() {
    println!("\n🎯 Recommended Learning Paths:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    println!("\n👶 Beginner Path (Start Here!):");
    println!("  1. Two Sum (Hash Table basics)");
    println!("  2. Valid Parentheses (Stack fundamentals)");
    println!("  3. Merge Two Sorted Lists (Linked list operations)");
    println!("  4. Maximum Depth of Binary Tree (Tree traversal)");
    println!("  5. Climbing Stairs (Basic DP)");
    
    println!("\n🏃 Intermediate Path:");
    println!("  1. 3Sum (Two pointers technique)");
    println!("  2. Longest Substring Without Repeating Characters (Sliding window)");
    println!("  3. Maximum Subarray (Kadane's algorithm)");
    println!("  4. Number of Islands (DFS/BFS)");
    println!("  5. LRU Cache (System design)");
    
    println!("\n🚀 Advanced Path:");
    println!("  1. Median of Two Sorted Arrays (Binary search mastery)");
    println!("  2. Edit Distance (Complex DP)");
    println!("  3. Trapping Rain Water (Multiple approaches)");
    println!("  4. Merge k Sorted Lists (Divide and conquer)");
    println!("  5. Regular Expression Matching (Advanced DP)");
    
    println!("\n📚 Topic-Focused Paths:");
    println!("  • Dynamic Programming: 70 → 198 → 53 → 152 → 72");
    println!("  • Tree Algorithms: 104 → 226 → 98 → 230 → 236");
    println!("  • Graph Algorithms: 200 → 207 → 133 → 127 → 126");
    println!("  • String Processing: 3 → 5 → 76 → 10 → 214");
}

fn show_statistics(problems: &[Problem]) {
    println!("\n📈 Repository Statistics:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let easy_count = problems.iter().filter(|p| p.difficulty == Difficulty::Easy).count();
    let medium_count = problems.iter().filter(|p| p.difficulty == Difficulty::Medium).count();
    let hard_count = problems.iter().filter(|p| p.difficulty == Difficulty::Hard).count();
    
    println!("Total Problems: {}", problems.len());
    println!("Easy: {} ({:.1}%)", easy_count, (easy_count as f64 / problems.len() as f64) * 100.0);
    println!("Medium: {} ({:.1}%)", medium_count, (medium_count as f64 / problems.len() as f64) * 100.0);
    println!("Hard: {} ({:.1}%)", hard_count, (hard_count as f64 / problems.len() as f64) * 100.0);
    
    // Topic distribution
    let mut topic_count: HashMap<String, usize> = HashMap::new();
    for problem in problems {
        for topic in &problem.topics {
            *topic_count.entry(topic.clone()).or_insert(0) += 1;
        }
    }
    
    println!("\nTop Topics:");
    let mut topics: Vec<_> = topic_count.iter().collect();
    topics.sort_by(|a, b| b.1.cmp(a.1));
    
    for (topic, count) in topics.iter().take(5) {
        println!("  {}: {}", topic, count);
    }
    
    let total_approaches: usize = problems.iter().map(|p| p.approaches as usize).sum();
    println!("\nTotal Solution Approaches: {}", total_approaches);
    println!("Average Approaches per Problem: {:.1}", total_approaches as f64 / problems.len() as f64);
    
    println!("\n💻 Quick Commands:");
    println!("  Run all tests: cargo test");
    println!("  Run benchmarks: cargo bench");
    println!("  Check coverage: cargo tarpaulin --out Html");
}

fn main() {
    let problems = create_problem_database();
    
    loop {
        display_menu();
        
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        
        match input.trim() {
            "1" => browse_by_difficulty(&problems),
            "2" => search_by_topic(&problems),
            "3" => filter_by_complexity(&problems),
            "4" => view_problem_details(&problems),
            "5" => random_problem(&problems),
            "6" => learning_path_recommendations(),
            "7" => show_statistics(&problems),
            "8" => {
                println!("\n👋 Happy coding! Keep practicing those algorithms!");
                break;
            },
            _ => println!("❌ Invalid option. Please select 1-8."),
        }
        
        println!("\nPress Enter to continue...");
        let mut _input = String::new();
        io::stdin().read_line(&mut _input).unwrap();
    }
}