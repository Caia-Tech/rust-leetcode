#!/usr/bin/env cargo-script

//! Progress Tracking System for LeetCode Problems
//! 
//! This tool helps you track your completion status across all 105 implemented problems,
//! providing analytics on your progress and helping identify areas for improvement.

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};

#[derive(Debug, Clone)]
struct Problem {
    id: u32,
    title: String,
    difficulty: String,
    topics: Vec<String>,
    status: CompletionStatus,
    attempts: u32,
    best_time_complexity: String,
    best_space_complexity: String,
    notes: String,
    last_attempted: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CompletionStatus {
    NotStarted,
    InProgress,
    Completed,
    Mastered,
    NeedsReview,
}

impl CompletionStatus {
    fn as_str(&self) -> &str {
        match self {
            CompletionStatus::NotStarted => "Not Started",
            CompletionStatus::InProgress => "In Progress",
            CompletionStatus::Completed => "Completed",
            CompletionStatus::Mastered => "Mastered",
            CompletionStatus::NeedsReview => "Needs Review",
        }
    }

    fn from_str(s: &str) -> Self {
        match s {
            "Not Started" => CompletionStatus::NotStarted,
            "In Progress" => CompletionStatus::InProgress,
            "Completed" => CompletionStatus::Completed,
            "Mastered" => CompletionStatus::Mastered,
            "Needs Review" => CompletionStatus::NeedsReview,
            _ => CompletionStatus::NotStarted,
        }
    }
}

struct ProgressTracker {
    problems: HashMap<u32, Problem>,
    progress_file: String,
}

impl ProgressTracker {
    fn new() -> Self {
        let mut tracker = Self {
            problems: HashMap::new(),
            progress_file: "progress.json".to_string(),
        };
        tracker.load_problems();
        tracker.load_progress();
        tracker
    }

    fn load_problems(&mut self) {
        // Load problems from the codebase structure
        let problems = vec![
            // Easy Problems (14)
            (1, "Two Sum", "Easy", vec!["Array", "Hash Table"]),
            (20, "Valid Parentheses", "Easy", vec!["String", "Stack"]),
            (21, "Merge Two Sorted Lists", "Easy", vec!["Linked List", "Recursion"]),
            (26, "Remove Duplicates from Sorted Array", "Easy", vec!["Array", "Two Pointers"]),
            (27, "Remove Element", "Easy", vec!["Array", "Two Pointers"]),
            (70, "Climbing Stairs", "Easy", vec!["Math", "Dynamic Programming", "Memoization"]),
            (104, "Maximum Depth of Binary Tree", "Easy", vec!["Tree", "Depth-First Search", "Breadth-First Search", "Binary Tree"]),
            (121, "Best Time to Buy and Sell Stock", "Easy", vec!["Array", "Dynamic Programming"]),
            (136, "Single Number", "Easy", vec!["Array", "Bit Manipulation"]),
            (169, "Majority Element", "Easy", vec!["Array", "Hash Table", "Divide and Conquer", "Sorting", "Counting"]),
            (189, "Rotate Array", "Easy", vec!["Array", "Math", "Two Pointers"]),
            (226, "Invert Binary Tree", "Easy", vec!["Tree", "Depth-First Search", "Breadth-First Search", "Binary Tree"]),
            (242, "Valid Anagram", "Easy", vec!["Hash Table", "String", "Sorting"]),
            (283, "Move Zeroes", "Easy", vec!["Array", "Two Pointers"]),

            // Medium Problems (45)
            (2, "Add Two Numbers", "Medium", vec!["Linked List", "Math", "Recursion"]),
            (3, "Longest Substring Without Repeating Characters", "Medium", vec!["Hash Table", "String", "Sliding Window"]),
            (5, "Longest Palindromic Substring", "Medium", vec!["String", "Dynamic Programming"]),
            (11, "Container With Most Water", "Medium", vec!["Array", "Two Pointers", "Greedy"]),
            (15, "3Sum", "Medium", vec!["Array", "Two Pointers", "Sorting"]),
            (19, "Remove Nth Node From End of List", "Medium", vec!["Linked List", "Two Pointers"]),
            (22, "Generate Parentheses", "Medium", vec!["String", "Dynamic Programming", "Backtracking"]),
            (33, "Search in Rotated Sorted Array", "Medium", vec!["Array", "Binary Search"]),
            (34, "Find First and Last Position of Element in Sorted Array", "Medium", vec!["Array", "Binary Search"]),
            (39, "Combination Sum", "Medium", vec!["Array", "Backtracking"]),
            (46, "Permutations", "Medium", vec!["Array", "Backtracking"]),
            (48, "Rotate Image", "Medium", vec!["Array", "Math", "Matrix"]),
            (49, "Group Anagrams", "Medium", vec!["Array", "Hash Table", "String", "Sorting"]),
            (53, "Maximum Subarray", "Medium", vec!["Array", "Divide and Conquer", "Dynamic Programming"]),
            (55, "Jump Game", "Medium", vec!["Array", "Dynamic Programming", "Greedy"]),
            (56, "Merge Intervals", "Medium", vec!["Array", "Sorting"]),
            (62, "Unique Paths", "Medium", vec!["Math", "Dynamic Programming", "Combinatorics"]),
            (75, "Sort Colors", "Medium", vec!["Array", "Two Pointers", "Sorting"]),
            (78, "Subsets", "Medium", vec!["Array", "Backtracking", "Bit Manipulation"]),
            (79, "Word Search", "Medium", vec!["Array", "Backtracking", "Matrix"]),
            (91, "Decode Ways", "Medium", vec!["String", "Dynamic Programming"]),
            (102, "Binary Tree Level Order Traversal", "Medium", vec!["Tree", "Breadth-First Search", "Binary Tree"]),
            (105, "Construct Binary Tree from Preorder and Inorder Traversal", "Medium", vec!["Array", "Hash Table", "Divide and Conquer", "Tree", "Binary Tree"]),
            (128, "Longest Consecutive Sequence", "Medium", vec!["Union Find", "Array", "Hash Table"]),
            (139, "Word Break", "Medium", vec!["Hash Table", "String", "Dynamic Programming", "Trie", "Memoization"]),
            (141, "Linked List Cycle", "Medium", vec!["Hash Table", "Linked List", "Two Pointers"]),
            (146, "LRU Cache", "Medium", vec!["Hash Table", "Linked List", "Design", "Doubly-Linked List"]),
            (148, "Sort List", "Medium", vec!["Linked List", "Two Pointers", "Divide and Conquer", "Sorting", "Merge Sort"]),
            (152, "Maximum Product Subarray", "Medium", vec!["Array", "Dynamic Programming"]),
            (153, "Find Minimum in Rotated Sorted Array", "Medium", vec!["Array", "Binary Search"]),
            (198, "House Robber", "Medium", vec!["Array", "Dynamic Programming"]),
            (200, "Number of Islands", "Medium", vec!["Array", "Depth-First Search", "Breadth-First Search", "Union Find", "Matrix"]),
            (207, "Course Schedule", "Medium", vec!["Depth-First Search", "Breadth-First Search", "Graph", "Topological Sort"]),
            (208, "Implement Trie (Prefix Tree)", "Medium", vec!["Hash Table", "String", "Design", "Trie"]),
            (213, "House Robber II", "Medium", vec!["Array", "Dynamic Programming"]),
            (215, "Kth Largest Element in an Array", "Medium", vec!["Array", "Divide and Conquer", "Sorting", "Heap (Priority Queue)", "Quickselect"]),
            (230, "Kth Smallest Element in a BST", "Medium", vec!["Tree", "Depth-First Search", "Binary Search Tree", "Binary Tree"]),
            (236, "Lowest Common Ancestor of a Binary Tree", "Medium", vec!["Tree", "Depth-First Search", "Binary Tree"]),
            (238, "Product of Array Except Self", "Medium", vec!["Array", "Prefix Sum"]),
            (287, "Find the Duplicate Number", "Medium", vec!["Array", "Two Pointers", "Binary Search", "Bit Manipulation"]),
            (300, "Longest Increasing Subsequence", "Medium", vec!["Array", "Binary Search", "Dynamic Programming"]),
            (322, "Coin Change", "Medium", vec!["Array", "Dynamic Programming", "Breadth-First Search"]),
            (347, "Top K Frequent Elements", "Medium", vec!["Array", "Hash Table", "Divide and Conquer", "Sorting", "Heap (Priority Queue)", "Bucket Sort", "Counting", "Quickselect"]),
            (417, "Pacific Atlantic Water Flow", "Medium", vec!["Array", "Depth-First Search", "Breadth-First Search", "Matrix"]),
            (435, "Non-overlapping Intervals", "Medium", vec!["Array", "Dynamic Programming", "Greedy", "Sorting"]),
            (647, "Palindromic Substrings", "Medium", vec!["String", "Dynamic Programming"]),

            // Hard Problems (46)
            (4, "Median of Two Sorted Arrays", "Hard", vec!["Array", "Binary Search", "Divide and Conquer"]),
            (10, "Regular Expression Matching", "Hard", vec!["String", "Dynamic Programming", "Recursion"]),
            (23, "Merge k Sorted Lists", "Hard", vec!["Linked List", "Divide and Conquer", "Heap (Priority Queue)", "Merge Sort"]),
            (25, "Reverse Nodes in k-Group", "Hard", vec!["Linked List", "Recursion"]),
            (32, "Longest Valid Parentheses", "Hard", vec!["String", "Dynamic Programming", "Stack"]),
            (37, "Sudoku Solver", "Hard", vec!["Array", "Backtracking", "Matrix"]),
            (41, "First Missing Positive", "Hard", vec!["Array", "Hash Table"]),
            (42, "Trapping Rain Water", "Hard", vec!["Array", "Two Pointers", "Dynamic Programming", "Stack", "Monotonic Stack"]),
            (44, "Wildcard Matching", "Hard", vec!["String", "Dynamic Programming", "Greedy", "Recursion"]),
            (51, "N-Queens", "Hard", vec!["Array", "Backtracking"]),
            (72, "Edit Distance", "Hard", vec!["String", "Dynamic Programming"]),
            (76, "Minimum Window Substring", "Hard", vec!["Hash Table", "String", "Sliding Window"]),
            (84, "Largest Rectangle in Histogram", "Hard", vec!["Array", "Stack", "Monotonic Stack"]),
            (85, "Maximal Rectangle", "Hard", vec!["Array", "Dynamic Programming", "Stack", "Matrix", "Monotonic Stack"]),
            (87, "Scramble String", "Hard", vec!["String", "Dynamic Programming"]),
            (99, "Recover Binary Search Tree", "Hard", vec!["Tree", "Depth-First Search", "Binary Search Tree", "Binary Tree"]),
            (115, "Distinct Subsequences", "Hard", vec!["String", "Dynamic Programming"]),
            (124, "Binary Tree Maximum Path Sum", "Hard", vec!["Dynamic Programming", "Tree", "Depth-First Search", "Binary Tree"]),
            (126, "Word Ladder II", "Hard", vec!["Hash Table", "String", "Backtracking", "Breadth-First Search"]),
            (127, "Word Ladder", "Hard", vec!["Hash Table", "String", "Breadth-First Search"]),
            (132, "Palindrome Partitioning II", "Hard", vec!["String", "Dynamic Programming"]),
            (154, "Find Minimum in Rotated Sorted Array II", "Hard", vec!["Array", "Binary Search"]),
            (188, "Best Time to Buy and Sell Stock IV", "Hard", vec!["Array", "Dynamic Programming"]),
            (212, "Word Search II", "Hard", vec!["Array", "String", "Backtracking", "Trie", "Matrix"]),
            (239, "Sliding Window Maximum", "Hard", vec!["Array", "Queue", "Sliding Window", "Heap (Priority Queue)", "Monotonic Queue"]),
            (295, "Find Median from Data Stream", "Hard", vec!["Two Pointers", "Design", "Sorting", "Heap (Priority Queue)", "Data Stream"]),
            (297, "Serialize and Deserialize Binary Tree", "Hard", vec!["String", "Tree", "Depth-First Search", "Breadth-First Search", "Design", "Binary Tree"]),
            (301, "Remove Invalid Parentheses", "Hard", vec!["String", "Backtracking", "Breadth-First Search"]),
            (312, "Burst Balloons", "Hard", vec!["Array", "Dynamic Programming"]),
            (315, "Count of Smaller Numbers After Self", "Hard", vec!["Array", "Binary Search", "Divide and Conquer", "Binary Indexed Tree", "Segment Tree", "Merge Sort", "Ordered Set"]),
            (327, "Count of Range Sum", "Hard", vec!["Array", "Binary Search", "Divide and Conquer", "Binary Indexed Tree", "Segment Tree", "Merge Sort", "Ordered Set"]),
            (336, "Palindrome Pairs", "Hard", vec!["Array", "Hash Table", "String", "Trie"]),
            (337, "House Robber III", "Hard", vec!["Dynamic Programming", "Tree", "Depth-First Search", "Binary Tree"]),
            (352, "Data Stream as Disjoint Intervals", "Hard", vec!["Binary Search", "Design", "Ordered Set"]),
            (381, "Insert Delete GetRandom O(1) - Duplicates allowed", "Hard", vec!["Array", "Hash Table", "Math", "Design", "Randomized"]),
            (460, "LFU Cache", "Hard", vec!["Hash Table", "Linked List", "Design", "Doubly-Linked List"]),
            (472, "Concatenated Words", "Hard", vec!["Array", "String", "Dynamic Programming", "Trie"]),
            (489, "Robot Room Cleaner", "Hard", vec!["Interactive", "Backtracking"]),
            (493, "Reverse Pairs", "Hard", vec!["Array", "Binary Search", "Divide and Conquer", "Binary Indexed Tree", "Segment Tree", "Merge Sort", "Ordered Set"]),
            (543, "Diameter of Binary Tree", "Hard", vec!["Tree", "Depth-First Search", "Binary Tree"]),
            (621, "Task Scheduler", "Hard", vec!["Array", "Hash Table", "Greedy", "Sorting", "Heap (Priority Queue)", "Counting"]),
            (632, "Smallest Range Covering Elements from K Lists", "Hard", vec!["Array", "Hash Table", "Greedy", "Sliding Window", "Sorting", "Heap (Priority Queue)"]),
            (642, "Design Search Autocomplete System", "Hard", vec!["Hash Table", "String", "Design", "Trie", "Data Stream"]),
            (668, "Kth Smallest Number in Multiplication Table", "Hard", vec!["Math", "Binary Search"]),
            (685, "Redundant Connection II", "Hard", vec!["Depth-First Search", "Breadth-First Search", "Union Find", "Graph", "Tree"]),
            (719, "Find K-th Smallest Pair Distance", "Hard", vec!["Array", "Two Pointers", "Binary Search", "Sorting"]),
            (743, "Network Delay Time", "Hard", vec!["Depth-First Search", "Breadth-First Search", "Graph", "Heap (Priority Queue)", "Shortest Path"]),
        ];

        for (id, title, difficulty, topics) in problems {
            let problem = Problem {
                id,
                title: title.to_string(),
                difficulty: difficulty.to_string(),
                topics: topics.iter().map(|s| s.to_string()).collect(),
                status: CompletionStatus::NotStarted,
                attempts: 0,
                best_time_complexity: "Unknown".to_string(),
                best_space_complexity: "Unknown".to_string(),
                notes: String::new(),
                last_attempted: None,
            };
            self.problems.insert(id, problem);
        }
    }

    fn load_progress(&mut self) {
        if let Ok(content) = fs::read_to_string(&self.progress_file) {
            if let Ok(progress_data) = serde_json::from_str::<HashMap<String, serde_json::Value>>(&content) {
                for (id_str, data) in progress_data {
                    if let Ok(id) = id_str.parse::<u32>() {
                        if let Some(problem) = self.problems.get_mut(&id) {
                            if let Some(status) = data.get("status").and_then(|s| s.as_str()) {
                                problem.status = CompletionStatus::from_str(status);
                            }
                            if let Some(attempts) = data.get("attempts").and_then(|a| a.as_u64()) {
                                problem.attempts = attempts as u32;
                            }
                            if let Some(time_complexity) = data.get("best_time_complexity").and_then(|t| t.as_str()) {
                                problem.best_time_complexity = time_complexity.to_string();
                            }
                            if let Some(space_complexity) = data.get("best_space_complexity").and_then(|s| s.as_str()) {
                                problem.best_space_complexity = space_complexity.to_string();
                            }
                            if let Some(notes) = data.get("notes").and_then(|n| n.as_str()) {
                                problem.notes = notes.to_string();
                            }
                            if let Some(last_attempted) = data.get("last_attempted").and_then(|l| l.as_str()) {
                                problem.last_attempted = Some(last_attempted.to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    fn save_progress(&self) {
        let mut progress_data = HashMap::new();
        for (id, problem) in &self.problems {
            let mut problem_data = std::collections::HashMap::new();
            problem_data.insert("status".to_string(), serde_json::Value::String(problem.status.as_str().to_string()));
            problem_data.insert("attempts".to_string(), serde_json::Value::Number(problem.attempts.into()));
            problem_data.insert("best_time_complexity".to_string(), serde_json::Value::String(problem.best_time_complexity.clone()));
            problem_data.insert("best_space_complexity".to_string(), serde_json::Value::String(problem.best_space_complexity.clone()));
            problem_data.insert("notes".to_string(), serde_json::Value::String(problem.notes.clone()));
            if let Some(last_attempted) = &problem.last_attempted {
                problem_data.insert("last_attempted".to_string(), serde_json::Value::String(last_attempted.clone()));
            }
            progress_data.insert(id.to_string(), serde_json::Value::Object(problem_data.into_iter().collect()));
        }
        
        if let Ok(json) = serde_json::to_string_pretty(&progress_data) {
            let _ = fs::write(&self.progress_file, json);
        }
    }

    fn run(&mut self) {
        println!("📊 LeetCode Progress Tracker");
        println!("============================");

        loop {
            self.show_main_menu();
            
            print!("\nEnter your choice: ");
            io::stdout().flush().unwrap();
            
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let choice = input.trim();

            match choice {
                "1" => self.show_overall_progress(),
                "2" => self.show_problems_by_difficulty(),
                "3" => self.show_problems_by_topic(),
                "4" => self.show_problems_by_status(),
                "5" => self.update_problem_status(),
                "6" => self.add_problem_notes(),
                "7" => self.show_recommended_problems(),
                "8" => self.show_statistics(),
                "9" => self.export_progress_report(),
                "10" => {
                    self.save_progress();
                    println!("Progress saved. Goodbye!");
                    break;
                }
                _ => println!("Invalid choice. Please try again."),
            }
        }
    }

    fn show_main_menu(&self) {
        println!("\n🎯 Main Menu:");
        println!("1.  📈 Overall Progress");
        println!("2.  📊 Problems by Difficulty");
        println!("3.  🏷️  Problems by Topic");
        println!("4.  ✅ Problems by Status");
        println!("5.  🔄 Update Problem Status");
        println!("6.  📝 Add Problem Notes");
        println!("7.  💡 Recommended Problems");
        println!("8.  📈 Statistics & Analytics");
        println!("9.  📋 Export Progress Report");
        println!("10. 💾 Save & Exit");
    }

    fn show_overall_progress(&self) {
        let total = self.problems.len();
        let mut status_counts = HashMap::new();
        
        for problem in self.problems.values() {
            *status_counts.entry(&problem.status).or_insert(0) += 1;
        }

        println!("\n📈 Overall Progress Summary");
        println!("==========================");
        println!("Total Problems: {}", total);
        
        for status in &[CompletionStatus::NotStarted, CompletionStatus::InProgress, 
                       CompletionStatus::Completed, CompletionStatus::Mastered, 
                       CompletionStatus::NeedsReview] {
            let count = status_counts.get(status).unwrap_or(&0);
            let percentage = (*count as f64 / total as f64) * 100.0;
            println!("{}: {} ({:.1}%)", status.as_str(), count, percentage);
        }

        // Progress by difficulty
        println!("\n📊 Progress by Difficulty:");
        let mut difficulty_progress = HashMap::new();
        for problem in self.problems.values() {
            let entry = difficulty_progress.entry(&problem.difficulty).or_insert((0, 0));
            entry.1 += 1; // total
            if problem.status == CompletionStatus::Completed || problem.status == CompletionStatus::Mastered {
                entry.0 += 1; // completed
            }
        }

        for (difficulty, (completed, total)) in difficulty_progress {
            let percentage = (completed as f64 / total as f64) * 100.0;
            println!("  {}: {}/{} ({:.1}%)", difficulty, completed, total, percentage);
        }
    }

    fn show_problems_by_difficulty(&self) {
        println!("\n📊 Problems by Difficulty");
        println!("=========================");

        for difficulty in &["Easy", "Medium", "Hard"] {
            println!("\n🎯 {} Problems:", difficulty);
            let mut difficulty_problems: Vec<_> = self.problems.values()
                .filter(|p| p.difficulty == *difficulty)
                .collect();
            difficulty_problems.sort_by_key(|p| p.id);

            for problem in difficulty_problems {
                let status_emoji = match problem.status {
                    CompletionStatus::NotStarted => "⭕",
                    CompletionStatus::InProgress => "🔄",
                    CompletionStatus::Completed => "✅",
                    CompletionStatus::Mastered => "🏆",
                    CompletionStatus::NeedsReview => "🔍",
                };
                println!("  {} #{}: {} ({})", status_emoji, problem.id, problem.title, problem.status.as_str());
            }
        }
    }

    fn show_problems_by_topic(&self) {
        println!("\n🏷️ Problems by Topic");
        println!("====================");

        let mut topic_problems: HashMap<String, Vec<&Problem>> = HashMap::new();
        for problem in self.problems.values() {
            for topic in &problem.topics {
                topic_problems.entry(topic.clone()).or_default().push(problem);
            }
        }

        let mut topics: Vec<_> = topic_problems.keys().collect();
        topics.sort();

        for topic in topics {
            let problems = &topic_problems[topic];
            let completed = problems.iter().filter(|p| 
                p.status == CompletionStatus::Completed || p.status == CompletionStatus::Mastered
            ).count();
            
            println!("\n📚 {} ({}/{})", topic, completed, problems.len());
            
            let mut topic_problems_sorted = problems.clone();
            topic_problems_sorted.sort_by_key(|p| p.id);
            
            for problem in topic_problems_sorted {
                let status_emoji = match problem.status {
                    CompletionStatus::NotStarted => "⭕",
                    CompletionStatus::InProgress => "🔄",
                    CompletionStatus::Completed => "✅",
                    CompletionStatus::Mastered => "🏆",
                    CompletionStatus::NeedsReview => "🔍",
                };
                println!("    {} #{}: {}", status_emoji, problem.id, problem.title);
            }
        }
    }

    fn show_problems_by_status(&self) {
        println!("\n✅ Problems by Status");
        println!("=====================");

        for status in &[CompletionStatus::NotStarted, CompletionStatus::InProgress, 
                       CompletionStatus::Completed, CompletionStatus::Mastered, 
                       CompletionStatus::NeedsReview] {
            let mut status_problems: Vec<_> = self.problems.values()
                .filter(|p| p.status == *status)
                .collect();
            status_problems.sort_by_key(|p| p.id);

            if !status_problems.is_empty() {
                println!("\n📋 {} ({} problems):", status.as_str(), status_problems.len());
                for problem in status_problems {
                    println!("  #{}: {} [{}]", problem.id, problem.title, problem.difficulty);
                }
            }
        }
    }

    fn update_problem_status(&mut self) {
        print!("Enter problem ID: ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        
        if let Ok(id) = input.trim().parse::<u32>() {
            if let Some(problem) = self.problems.get_mut(&id) {
                println!("\nCurrent status for #{}: {} - {}", id, problem.title, problem.status.as_str());
                println!("\nSelect new status:");
                println!("1. Not Started");
                println!("2. In Progress");
                println!("3. Completed");
                println!("4. Mastered");
                println!("5. Needs Review");
                
                print!("Choice: ");
                io::stdout().flush().unwrap();
                input.clear();
                io::stdin().read_line(&mut input).unwrap();
                
                let new_status = match input.trim() {
                    "1" => CompletionStatus::NotStarted,
                    "2" => CompletionStatus::InProgress,
                    "3" => CompletionStatus::Completed,
                    "4" => CompletionStatus::Mastered,
                    "5" => CompletionStatus::NeedsReview,
                    _ => {
                        println!("Invalid choice.");
                        return;
                    }
                };
                
                problem.status = new_status;
                problem.attempts += 1;
                problem.last_attempted = Some(format!("{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()));
                
                // Optionally update complexity information
                if problem.status == CompletionStatus::Completed || problem.status == CompletionStatus::Mastered {
                    print!("Enter time complexity (e.g., O(n), O(log n)): ");
                    io::stdout().flush().unwrap();
                    input.clear();
                    io::stdin().read_line(&mut input).unwrap();
                    if !input.trim().is_empty() {
                        problem.best_time_complexity = input.trim().to_string();
                    }
                    
                    print!("Enter space complexity (e.g., O(1), O(n)): ");
                    io::stdout().flush().unwrap();
                    input.clear();
                    io::stdin().read_line(&mut input).unwrap();
                    if !input.trim().is_empty() {
                        problem.best_space_complexity = input.trim().to_string();
                    }
                }
                
                println!("✅ Status updated successfully!");
                self.save_progress();
            } else {
                println!("Problem ID {} not found.", id);
            }
        } else {
            println!("Invalid problem ID.");
        }
    }

    fn add_problem_notes(&mut self) {
        print!("Enter problem ID: ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        
        if let Ok(id) = input.trim().parse::<u32>() {
            if let Some(problem) = self.problems.get_mut(&id) {
                println!("\nCurrent notes for #{}: {}", id, problem.title);
                if !problem.notes.is_empty() {
                    println!("Current: {}", problem.notes);
                }
                
                print!("Enter new notes: ");
                io::stdout().flush().unwrap();
                input.clear();
                io::stdin().read_line(&mut input).unwrap();
                
                problem.notes = input.trim().to_string();
                println!("✅ Notes updated successfully!");
                self.save_progress();
            } else {
                println!("Problem ID {} not found.", id);
            }
        } else {
            println!("Invalid problem ID.");
        }
    }

    fn show_recommended_problems(&self) {
        println!("\n💡 Recommended Problems");
        println!("=======================");

        // Recommend based on current progress
        let not_started: Vec<_> = self.problems.values()
            .filter(|p| p.status == CompletionStatus::NotStarted)
            .collect();

        let needs_review: Vec<_> = self.problems.values()
            .filter(|p| p.status == CompletionStatus::NeedsReview)
            .collect();

        if !needs_review.is_empty() {
            println!("\n🔍 Problems that need review:");
            for problem in needs_review.iter().take(5) {
                println!("  #{}: {} [{}]", problem.id, problem.title, problem.difficulty);
            }
        }

        if !not_started.is_empty() {
            println!("\n🆕 Recommended next problems (Easy first):");
            let mut easy_problems: Vec<_> = not_started.iter().filter(|p| p.difficulty == "Easy").collect();
            easy_problems.sort_by_key(|p| p.id);
            
            for problem in easy_problems.iter().take(3) {
                println!("  #{}: {} [{}]", problem.id, problem.title, problem.difficulty);
            }

            println!("\n🎯 Medium problems to try:");
            let mut medium_problems: Vec<_> = not_started.iter().filter(|p| p.difficulty == "Medium").collect();
            medium_problems.sort_by_key(|p| p.id);
            
            for problem in medium_problems.iter().take(3) {
                println!("  #{}: {} [{}]", problem.id, problem.title, problem.difficulty);
            }
        }

        // Topic-based recommendations
        println!("\n📚 By topic completion rate:");
        let mut topic_completion: Vec<_> = self.get_topic_completion_rates();
        topic_completion.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for (topic, rate) in topic_completion.iter().take(5) {
            println!("  {}: {:.1}% complete", topic, rate * 100.0);
        }
    }

    fn get_topic_completion_rates(&self) -> Vec<(String, f64)> {
        let mut topic_stats: HashMap<String, (usize, usize)> = HashMap::new();
        
        for problem in self.problems.values() {
            for topic in &problem.topics {
                let entry = topic_stats.entry(topic.clone()).or_insert((0, 0));
                entry.1 += 1; // total
                if problem.status == CompletionStatus::Completed || problem.status == CompletionStatus::Mastered {
                    entry.0 += 1; // completed
                }
            }
        }

        topic_stats.into_iter()
            .map(|(topic, (completed, total))| (topic, completed as f64 / total as f64))
            .collect()
    }

    fn show_statistics(&self) {
        println!("\n📈 Statistics & Analytics");
        println!("=========================");

        let total = self.problems.len();
        let completed = self.problems.values()
            .filter(|p| p.status == CompletionStatus::Completed || p.status == CompletionStatus::Mastered)
            .count();

        println!("📊 Overall Statistics:");
        println!("  Total Problems: {}", total);
        println!("  Completed: {} ({:.1}%)", completed, (completed as f64 / total as f64) * 100.0);
        
        let total_attempts: u32 = self.problems.values().map(|p| p.attempts).sum();
        println!("  Total Attempts: {}", total_attempts);
        
        if completed > 0 {
            let avg_attempts = total_attempts as f64 / completed as f64;
            println!("  Average Attempts per Completed Problem: {:.1}", avg_attempts);
        }

        // Most attempted problems
        println!("\n🔥 Most Attempted Problems:");
        let mut most_attempted: Vec<_> = self.problems.values().collect();
        most_attempted.sort_by(|a, b| b.attempts.cmp(&a.attempts));

        for problem in most_attempted.iter().take(5) {
            if problem.attempts > 0 {
                println!("  #{}: {} - {} attempts [{}]", 
                    problem.id, problem.title, problem.attempts, problem.status.as_str());
            }
        }

        // Topic mastery
        println!("\n🎯 Topic Mastery:");
        let topic_rates = self.get_topic_completion_rates();
        let mut sorted_rates: Vec<_> = topic_rates;
        sorted_rates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (topic, rate) in sorted_rates.iter().take(10) {
            println!("  {}: {:.1}%", topic, rate * 100.0);
        }
    }

    fn export_progress_report(&self) {
        let filename = format!("progress_report_{}.md", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs());
        let mut report = String::new();
        
        report.push_str("# LeetCode Progress Report\n\n");
        report.push_str(&format!("Generated: {}\n\n", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()));
        
        let total = self.problems.len();
        let completed = self.problems.values()
            .filter(|p| p.status == CompletionStatus::Completed || p.status == CompletionStatus::Mastered)
            .count();
        
        report.push_str("## Overall Progress\n\n");
        report.push_str(&format!("- Total Problems: {}\n", total));
        report.push_str(&format!("- Completed: {} ({:.1}%)\n\n", completed, (completed as f64 / total as f64) * 100.0));
        
        // Problems by status
        report.push_str("## Problems by Status\n\n");
        for status in &[CompletionStatus::NotStarted, CompletionStatus::InProgress, 
                       CompletionStatus::Completed, CompletionStatus::Mastered, 
                       CompletionStatus::NeedsReview] {
            let status_problems: Vec<_> = self.problems.values()
                .filter(|p| p.status == *status)
                .collect();
            
            if !status_problems.is_empty() {
                report.push_str(&format!("### {} ({} problems)\n\n", status.as_str(), status_problems.len()));
                for problem in status_problems {
                    report.push_str(&format!("- #{}: {} [{}]\n", problem.id, problem.title, problem.difficulty));
                }
                report.push_str("\n");
            }
        }
        
        if let Err(e) = fs::write(&filename, report) {
            println!("Error writing report: {}", e);
        } else {
            println!("✅ Progress report exported to: {}", filename);
        }
    }
}

fn main() {
    let mut tracker = ProgressTracker::new();
    tracker.run();
}

// Add serde dependency for JSON handling (would need to be added to Cargo.toml)
