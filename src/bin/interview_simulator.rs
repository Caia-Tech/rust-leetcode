#!/usr/bin/env cargo-script

//! Interview Simulation Mode for LeetCode Practice
//! 
//! This tool simulates real coding interview conditions with:
//! - Timed problem solving sessions
//! - Interactive interviewer prompts
//! - Real-time performance tracking
//! - Post-interview analysis and feedback

use std::collections::HashMap;
use std::io::{self, Write};
use std::time::{Duration, Instant};
use rand::seq::SliceRandom;

#[derive(Debug, Clone)]
struct InterviewProblem {
    id: u32,
    title: String,
    difficulty: String,
    topics: Vec<String>,
    description: String,
    examples: Vec<String>,
    constraints: Vec<String>,
    hints: Vec<String>,
    time_limit_minutes: u32,
    follow_up_questions: Vec<String>,
}

#[derive(Debug, Clone)]
struct InterviewSession {
    problems: Vec<InterviewProblem>,
    current_problem_index: usize,
    start_time: Instant,
    problem_start_time: Option<Instant>,
    time_spent_per_problem: Vec<Duration>,
    solutions_attempted: Vec<String>,
    interviewer_notes: Vec<String>,
    session_type: SessionType,
    total_duration: Duration,
}

#[derive(Debug, Clone)]
enum SessionType {
    TechnicalRound,    // 45-60 minutes, 2-3 problems
    PhoneScreen,       // 30 minutes, 1-2 problems
    OnSite,           // 45-60 minutes, 1-2 complex problems
    MockInterview,     // Customizable duration
}

impl SessionType {
    fn duration_minutes(&self) -> u32 {
        match self {
            SessionType::PhoneScreen => 30,
            SessionType::TechnicalRound => 45,
            SessionType::OnSite => 60,
            SessionType::MockInterview => 45, // default
        }
    }

    fn problem_count(&self) -> usize {
        match self {
            SessionType::PhoneScreen => 2,
            SessionType::TechnicalRound => 2,
            SessionType::OnSite => 2,
            SessionType::MockInterview => 2,
        }
    }
}

struct InterviewSimulator {
    problem_bank: HashMap<String, Vec<InterviewProblem>>,
    session_history: Vec<InterviewSession>,
}

impl InterviewSimulator {
    fn new() -> Self {
        let mut simulator = Self {
            problem_bank: HashMap::new(),
            session_history: Vec::new(),
        };
        simulator.initialize_problem_bank();
        simulator
    }

    fn initialize_problem_bank(&mut self) {
        // Easy problems for phone screens
        let easy_problems = vec![
            InterviewProblem {
                id: 1,
                title: "Two Sum".to_string(),
                difficulty: "Easy".to_string(),
                topics: vec!["Array".to_string(), "Hash Table".to_string()],
                description: "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.".to_string(),
                examples: vec![
                    "Input: nums = [2,7,11,15], target = 9\nOutput: [0,1]\nExplanation: Because nums[0] + nums[1] == 9, we return [0, 1].".to_string()
                ],
                constraints: vec![
                    "2 <= nums.length <= 10^4".to_string(),
                    "-10^9 <= nums[i] <= 10^9".to_string(),
                    "Only one valid answer exists.".to_string(),
                ],
                hints: vec![
                    "Try using a hash map to store complements".to_string(),
                    "What's the time complexity of your solution?".to_string(),
                ],
                time_limit_minutes: 15,
                follow_up_questions: vec![
                    "What if the array was sorted?".to_string(),
                    "How would you handle duplicate elements?".to_string(),
                    "Can you solve it with O(1) space complexity?".to_string(),
                ],
            },
            InterviewProblem {
                id: 20,
                title: "Valid Parentheses".to_string(),
                difficulty: "Easy".to_string(),
                topics: vec!["String".to_string(), "Stack".to_string()],
                description: "Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.".to_string(),
                examples: vec![
                    "Input: s = \"()\"\nOutput: true".to_string(),
                    "Input: s = \"()[]{}\"\nOutput: true".to_string(),
                    "Input: s = \"(]\"\nOutput: false".to_string(),
                ],
                constraints: vec![
                    "1 <= s.length <= 10^4".to_string(),
                    "s consists of parentheses only '()[]{}'.".to_string(),
                ],
                hints: vec![
                    "Think about using a stack data structure".to_string(),
                    "What should happen when you encounter a closing bracket?".to_string(),
                ],
                time_limit_minutes: 15,
                follow_up_questions: vec![
                    "How would you handle nested parentheses of unlimited depth?".to_string(),
                    "What if we had custom bracket types?".to_string(),
                ],
            },
        ];

        // Medium problems for technical rounds
        let medium_problems = vec![
            InterviewProblem {
                id: 3,
                title: "Longest Substring Without Repeating Characters".to_string(),
                difficulty: "Medium".to_string(),
                topics: vec!["Hash Table".to_string(), "String".to_string(), "Sliding Window".to_string()],
                description: "Given a string s, find the length of the longest substring without repeating characters.".to_string(),
                examples: vec![
                    "Input: s = \"abcabcbb\"\nOutput: 3\nExplanation: The answer is \"abc\", with the length of 3.".to_string(),
                    "Input: s = \"bbbbb\"\nOutput: 1\nExplanation: The answer is \"b\", with the length of 1.".to_string(),
                ],
                constraints: vec![
                    "0 <= s.length <= 5 * 10^4".to_string(),
                    "s consists of English letters, digits, symbols and spaces.".to_string(),
                ],
                hints: vec![
                    "Consider using a sliding window approach".to_string(),
                    "How can you efficiently track character positions?".to_string(),
                    "What happens when you find a duplicate character?".to_string(),
                ],
                time_limit_minutes: 25,
                follow_up_questions: vec![
                    "How would you handle Unicode characters?".to_string(),
                    "Can you implement this with O(1) space for ASCII characters?".to_string(),
                    "What if you needed to return the actual substring?".to_string(),
                ],
            },
            InterviewProblem {
                id: 15,
                title: "3Sum".to_string(),
                difficulty: "Medium".to_string(),
                topics: vec!["Array".to_string(), "Two Pointers".to_string(), "Sorting".to_string()],
                description: "Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.".to_string(),
                examples: vec![
                    "Input: nums = [-1,0,1,2,-1,-4]\nOutput: [[-1,-1,2],[-1,0,1]]".to_string(),
                ],
                constraints: vec![
                    "3 <= nums.length <= 3000".to_string(),
                    "-10^5 <= nums[i] <= 10^5".to_string(),
                ],
                hints: vec![
                    "Try sorting the array first".to_string(),
                    "Can you reduce this to a 2Sum problem?".to_string(),
                    "How will you handle duplicate triplets?".to_string(),
                ],
                time_limit_minutes: 30,
                follow_up_questions: vec![
                    "How would you solve 4Sum?".to_string(),
                    "What if we needed K-Sum for arbitrary K?".to_string(),
                    "Can you optimize for space complexity?".to_string(),
                ],
            },
        ];

        // Hard problems for onsite rounds
        let hard_problems = vec![
            InterviewProblem {
                id: 4,
                title: "Median of Two Sorted Arrays".to_string(),
                difficulty: "Hard".to_string(),
                topics: vec!["Array".to_string(), "Binary Search".to_string(), "Divide and Conquer".to_string()],
                description: "Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.".to_string(),
                examples: vec![
                    "Input: nums1 = [1,3], nums2 = [2]\nOutput: 2.00000\nExplanation: merged array = [1,2,3] and median is 2.".to_string(),
                    "Input: nums1 = [1,2], nums2 = [3,4]\nOutput: 2.50000\nExplanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.".to_string(),
                ],
                constraints: vec![
                    "nums1.length == m".to_string(),
                    "nums2.length == n".to_string(),
                    "0 <= m <= 1000".to_string(),
                    "0 <= n <= 1000".to_string(),
                    "1 <= m + n <= 2000".to_string(),
                ],
                hints: vec![
                    "The naive solution would be O(m+n). Can you do better?".to_string(),
                    "Think about binary search on the smaller array".to_string(),
                    "What properties must the median satisfy?".to_string(),
                ],
                time_limit_minutes: 40,
                follow_up_questions: vec![
                    "How would this work with k sorted arrays?".to_string(),
                    "What if the arrays weren't sorted?".to_string(),
                    "How would you handle streaming data?".to_string(),
                ],
            },
        ];

        self.problem_bank.insert("Easy".to_string(), easy_problems);
        self.problem_bank.insert("Medium".to_string(), medium_problems);
        self.problem_bank.insert("Hard".to_string(), hard_problems);
    }

    fn run(&mut self) {
        println!("🎯 Technical Interview Simulator");
        println!("================================");
        println!("Practice coding interviews in realistic conditions!");
        
        loop {
            self.show_main_menu();
            
            print!("\nEnter your choice: ");
            io::stdout().flush().unwrap();
            
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let choice = input.trim();

            match choice {
                "1" => self.start_phone_screen(),
                "2" => self.start_technical_round(),
                "3" => self.start_onsite_round(),
                "4" => self.start_custom_session(),
                "5" => self.view_session_history(),
                "6" => self.practice_specific_topic(),
                "7" => self.show_interview_tips(),
                "8" => {
                    println!("Good luck in your interviews! 🚀");
                    break;
                }
                _ => println!("Invalid choice. Please try again."),
            }
        }
    }

    fn show_main_menu(&self) {
        println!("\n🎯 Interview Simulation Menu:");
        println!("1. 📞 Phone Screen (30 min, 1-2 Easy problems)");
        println!("2. 💻 Technical Round (45 min, 2-3 Medium problems)");
        println!("3. 🏢 On-site Round (60 min, 1-2 Hard problems)");
        println!("4. ⚙️  Custom Session");
        println!("5. 📊 View Session History");
        println!("6. 🎯 Practice Specific Topic");
        println!("7. 💡 Interview Tips & Best Practices");
        println!("8. 🚪 Exit");
    }

    fn start_phone_screen(&mut self) {
        println!("\n📞 Starting Phone Screen Simulation");
        println!("===================================");
        println!("Duration: 30 minutes");
        println!("Problems: 2 Easy problems");
        println!("Focus: Basic problem-solving and communication\n");

        let session = self.create_session(SessionType::PhoneScreen);
        self.run_interview_session(session);
    }

    fn start_technical_round(&mut self) {
        println!("\n💻 Starting Technical Round Simulation");
        println!("======================================");
        println!("Duration: 45 minutes");
        println!("Problems: 2-3 Medium problems");
        println!("Focus: Algorithm design and optimization\n");

        let session = self.create_session(SessionType::TechnicalRound);
        self.run_interview_session(session);
    }

    fn start_onsite_round(&mut self) {
        println!("\n🏢 Starting On-site Round Simulation");
        println!("====================================");
        println!("Duration: 60 minutes");
        println!("Problems: 1-2 Hard problems");
        println!("Focus: Complex algorithms and system thinking\n");

        let session = self.create_session(SessionType::OnSite);
        self.run_interview_session(session);
    }

    fn start_custom_session(&mut self) {
        println!("\n⚙️ Custom Session Setup");
        println!("=======================");

        print!("Enter duration in minutes (default 45): ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let duration = input.trim().parse::<u32>().unwrap_or(45);

        print!("Enter number of problems (default 2): ");
        io::stdout().flush().unwrap();
        input.clear();
        io::stdin().read_line(&mut input).unwrap();
        let problem_count = input.trim().parse::<usize>().unwrap_or(2);

        print!("Choose difficulty mix (1=Easy, 2=Medium, 3=Hard, 4=Mixed): ");
        io::stdout().flush().unwrap();
        input.clear();
        io::stdin().read_line(&mut input).unwrap();
        let difficulty_choice = input.trim();

        let mut session = InterviewSession {
            problems: Vec::new(),
            current_problem_index: 0,
            start_time: Instant::now(),
            problem_start_time: None,
            time_spent_per_problem: Vec::new(),
            solutions_attempted: Vec::new(),
            interviewer_notes: Vec::new(),
            session_type: SessionType::MockInterview,
            total_duration: Duration::from_secs(duration as u64 * 60),
        };

        // Select problems based on difficulty choice
        session.problems = self.select_problems_for_custom_session(difficulty_choice, problem_count);
        
        self.run_interview_session(session);
    }

    fn create_session(&self, session_type: SessionType) -> InterviewSession {
        let mut problems = Vec::new();
        
        match session_type {
            SessionType::PhoneScreen => {
                if let Some(easy_problems) = self.problem_bank.get("Easy") {
                    let mut selected = easy_problems.clone();
                    selected.shuffle(&mut rand::thread_rng());
                    problems.extend(selected.into_iter().take(2));
                }
            },
            SessionType::TechnicalRound => {
                if let Some(medium_problems) = self.problem_bank.get("Medium") {
                    let mut selected = medium_problems.clone();
                    selected.shuffle(&mut rand::thread_rng());
                    problems.extend(selected.into_iter().take(2));
                }
            },
            SessionType::OnSite => {
                if let Some(hard_problems) = self.problem_bank.get("Hard") {
                    let mut selected = hard_problems.clone();
                    selected.shuffle(&mut rand::thread_rng());
                    problems.extend(selected.into_iter().take(1));
                }
                // Add one medium problem for variety
                if let Some(medium_problems) = self.problem_bank.get("Medium") {
                    let mut selected = medium_problems.clone();
                    selected.shuffle(&mut rand::thread_rng());
                    problems.extend(selected.into_iter().take(1));
                }
            },
            SessionType::MockInterview => {
                // Will be customized in start_custom_session
            }
        }

        InterviewSession {
            problems,
            current_problem_index: 0,
            start_time: Instant::now(),
            problem_start_time: None,
            time_spent_per_problem: Vec::new(),
            solutions_attempted: Vec::new(),
            interviewer_notes: Vec::new(),
            session_type: session_type.clone(),
            total_duration: Duration::from_secs(session_type.duration_minutes() as u64 * 60),
        }
    }

    fn select_problems_for_custom_session(&self, difficulty_choice: &str, count: usize) -> Vec<InterviewProblem> {
        let mut problems = Vec::new();
        
        match difficulty_choice {
            "1" => {
                if let Some(easy_problems) = self.problem_bank.get("Easy") {
                    let mut selected = easy_problems.clone();
                    selected.shuffle(&mut rand::thread_rng());
                    problems.extend(selected.into_iter().take(count));
                }
            },
            "2" => {
                if let Some(medium_problems) = self.problem_bank.get("Medium") {
                    let mut selected = medium_problems.clone();
                    selected.shuffle(&mut rand::thread_rng());
                    problems.extend(selected.into_iter().take(count));
                }
            },
            "3" => {
                if let Some(hard_problems) = self.problem_bank.get("Hard") {
                    let mut selected = hard_problems.clone();
                    selected.shuffle(&mut rand::thread_rng());
                    problems.extend(selected.into_iter().take(count));
                }
            },
            "4" | _ => {
                // Mixed difficulty
                let difficulties = ["Easy", "Medium", "Hard"];
                for i in 0..count {
                    let difficulty = difficulties[i % difficulties.len()];
                    if let Some(difficulty_problems) = self.problem_bank.get(difficulty) {
                        if let Some(problem) = difficulty_problems.choose(&mut rand::thread_rng()) {
                            problems.push(problem.clone());
                        }
                    }
                }
            }
        }

        problems
    }

    fn run_interview_session(&mut self, mut session: InterviewSession) {
        println!("\n🚀 Interview Session Started!");
        println!("=============================");
        println!("Total Duration: {} minutes", session.total_duration.as_secs() / 60);
        println!("Number of Problems: {}", session.problems.len());
        println!("\n⏰ Timer starts now! Stay calm and think aloud.");
        println!("Type 'hint' for hints, 'next' to move to next problem, 'quit' to end session.\n");

        // Setup timer (simplified for demo - in real implementation would use proper async timer)
        let _total_duration = session.total_duration;

        session.start_time = Instant::now();
        
        for i in 0..session.problems.len() {
            session.current_problem_index = i;
            
            if self.present_problem(&mut session, i) {
                continue; // Next problem
            } else {
                break; // Session ended early
            }
        }

        self.conduct_post_interview_analysis(&session);
        self.session_history.push(session);
    }

    fn present_problem(&self, session: &mut InterviewSession, problem_index: usize) -> bool {
        let problem = &session.problems[problem_index];
        
        println!("\n{}", "=".repeat(60));
        println!("📋 Problem {} of {}: {} ({})", 
                problem_index + 1, session.problems.len(), problem.title, problem.difficulty);
        println!("{}", "=".repeat(60));
        
        println!("\n📖 Problem Description:");
        println!("{}", problem.description);
        
        println!("\n💡 Examples:");
        for (i, example) in problem.examples.iter().enumerate() {
            println!("Example {}:\n{}\n", i + 1, example);
        }
        
        println!("📏 Constraints:");
        for constraint in &problem.constraints {
            println!("• {}", constraint);
        }
        
        println!("\n⏱️  Suggested time limit: {} minutes", problem.time_limit_minutes);
        println!("\nStart explaining your approach! Remember to:");
        println!("• Think out loud");
        println!("• Discuss time/space complexity");
        println!("• Consider edge cases");
        println!("• Ask clarifying questions");

        session.problem_start_time = Some(Instant::now());

        // Interactive problem session
        self.interactive_problem_session(session, problem_index)
    }

    fn interactive_problem_session(&self, session: &mut InterviewSession, problem_index: usize) -> bool {
        let problem = &session.problems[problem_index];
        let mut hint_count = 0;
        let mut solution = String::new();
        
        loop {
            print!("\n💬 Enter your response (or command): ");
            io::stdout().flush().unwrap();
            
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let response = input.trim().to_lowercase();
            
            match response.as_str() {
                "hint" => {
                    if hint_count < problem.hints.len() {
                        println!("\n💡 Hint {}: {}", hint_count + 1, problem.hints[hint_count]);
                        hint_count += 1;
                    } else {
                        println!("\n🚫 No more hints available for this problem.");
                    }
                },
                "solution" | "code" => {
                    println!("\n💻 Please paste your solution (type 'done' when finished):");
                    solution.clear();
                    loop {
                        let mut code_input = String::new();
                        io::stdin().read_line(&mut code_input).unwrap();
                        if code_input.trim().to_lowercase() == "done" {
                            break;
                        }
                        solution.push_str(&code_input);
                    }
                    session.solutions_attempted.push(solution.clone());
                    println!("✅ Solution recorded! Great work on explaining your approach.");
                },
                "complexity" => {
                    println!("\n📊 Let's discuss complexity:");
                    print!("What's the time complexity of your solution? ");
                    io::stdout().flush().unwrap();
                    let mut complexity_input = String::new();
                    io::stdin().read_line(&mut complexity_input).unwrap();
                    
                    print!("What's the space complexity? ");
                    io::stdout().flush().unwrap();
                    complexity_input.clear();
                    io::stdin().read_line(&mut complexity_input).unwrap();
                    
                    println!("🎯 Good analysis! Can you optimize it further?");
                },
                "next" => {
                    if let Some(start_time) = session.problem_start_time {
                        session.time_spent_per_problem.push(start_time.elapsed());
                    }
                    println!("➡️ Moving to next problem...");
                    return true;
                },
                "quit" | "exit" => {
                    println!("🛑 Interview session ended early.");
                    return false;
                },
                "time" => {
                    let elapsed = session.start_time.elapsed();
                    let remaining = session.total_duration.saturating_sub(elapsed);
                    println!("⏰ Time remaining: {} minutes", remaining.as_secs() / 60);
                },
                _ => {
                    // Simulate interviewer responses
                    self.generate_interviewer_response(&response, problem);
                }
            }
        }
    }

    fn generate_interviewer_response(&self, response: &str, _problem: &InterviewProblem) {
        let responses = vec![
            "🤔 That's an interesting approach. Can you walk me through it step by step?",
            "👍 Good thinking! How would you handle edge cases?",
            "📊 What do you think the time complexity would be?",
            "🔄 Can you think of a way to optimize this further?",
            "🎯 That sounds right. Can you code that up?",
            "💭 Hmm, let me think about that. What if the input was very large?",
            "✨ Nice! How confident are you in this solution?",
            "🧪 Great! How would you test this?",
        ];
        
        // Simple response selection based on keywords
        if response.contains("hash") || response.contains("map") {
            println!("🎯 Hash tables are often useful! What's the trade-off here?");
        } else if response.contains("sort") {
            println!("📊 Sorting could work. What's the time complexity of that approach?");
        } else if response.contains("two pointer") || response.contains("pointer") {
            println!("🎪 Two pointers is a good technique! How do you know when to move each pointer?");
        } else if response.contains("dp") || response.contains("dynamic") {
            println!("🧮 Dynamic programming! Can you identify the subproblems?");
        } else {
            let response = responses.choose(&mut rand::thread_rng()).unwrap_or(&responses[0]);
            println!("{}", response);
        }
    }

    fn conduct_post_interview_analysis(&self, session: &InterviewSession) {
        println!("\n{}", "=".repeat(60));
        println!("📊 Post-Interview Analysis");
        println!("{}", "=".repeat(60));
        
        let total_elapsed = session.start_time.elapsed();
        let total_minutes = total_elapsed.as_secs() / 60;
        let total_seconds = total_elapsed.as_secs() % 60;
        
        println!("\n⏱️ Session Summary:");
        println!("• Total time: {}:{:02}", total_minutes, total_seconds);
        println!("• Problems attempted: {}/{}", session.current_problem_index + 1, session.problems.len());
        println!("• Solutions submitted: {}", session.solutions_attempted.len());
        
        println!("\n📋 Problems Covered:");
        for (i, problem) in session.problems.iter().enumerate() {
            let status = if i <= session.current_problem_index {
                if i < session.solutions_attempted.len() { "✅ Completed" } else { "🔄 Attempted" }
            } else { "⏸️ Not reached" };
            println!("{}. {} ({}) - {}", i + 1, problem.title, problem.difficulty, status);
        }
        
        if !session.time_spent_per_problem.is_empty() {
            println!("\n⏰ Time per problem:");
            for (i, duration) in session.time_spent_per_problem.iter().enumerate() {
                let minutes = duration.as_secs() / 60;
                let seconds = duration.as_secs() % 60;
                println!("Problem {}: {}:{:02}", i + 1, minutes, seconds);
            }
        }
        
        println!("\n🎯 Performance Analysis:");
        self.provide_performance_feedback(session);
        
        println!("\n💡 Areas for Improvement:");
        self.suggest_improvements(session);
        
        println!("\n🔄 Follow-up Questions to Practice:");
        for problem in &session.problems {
            if !problem.follow_up_questions.is_empty() {
                println!("\n{}:", problem.title);
                for question in &problem.follow_up_questions {
                    println!("• {}", question);
                }
            }
        }
    }

    fn provide_performance_feedback(&self, session: &InterviewSession) {
        let completion_rate = session.solutions_attempted.len() as f64 / session.problems.len() as f64;
        let time_efficiency = if session.start_time.elapsed() <= session.total_duration {
            "Good time management"
        } else {
            "Went over time limit"
        };
        
        match completion_rate {
            rate if rate >= 0.8 => println!("🌟 Excellent! You completed most problems successfully."),
            rate if rate >= 0.5 => println!("👍 Good job! You made solid progress on the problems."),
            rate if rate >= 0.3 => println!("📈 Not bad! Focus on improving problem-solving speed."),
            _ => println!("💪 Keep practicing! Consider starting with easier problems."),
        }
        
        println!("⏰ Time management: {}", time_efficiency);
        
        // Analyze difficulty progression
        let attempted_difficulties: Vec<String> = session.problems.iter()
            .take(session.current_problem_index + 1)
            .map(|p| p.difficulty.clone())
            .collect();
        
        if attempted_difficulties.len() > 1 {
            println!("🎯 Problem difficulty handled: {}", attempted_difficulties.join(" → "));
        }
    }

    fn suggest_improvements(&self, session: &InterviewSession) {
        let _suggestions = vec![
            "Practice explaining your thought process more clearly",
            "Spend more time on complexity analysis",
            "Consider more edge cases before coding",
            "Practice coding faster and more accurately",
            "Work on pattern recognition skills",
            "Review common data structures and algorithms",
            "Practice time management during problem solving",
            "Focus on communicating assumptions and trade-offs",
        ];
        
        // Select relevant suggestions based on session performance
        let relevant_suggestions = if session.solutions_attempted.len() < session.problems.len() {
            vec![
                "Practice coding faster and more accurately",
                "Work on pattern recognition skills",
                "Practice time management during problem solving",
            ]
        } else {
            vec![
                "Practice explaining your thought process more clearly",
                "Spend more time on complexity analysis",
                "Focus on communicating assumptions and trade-offs",
            ]
        };
        
        for suggestion in relevant_suggestions {
            println!("• {}", suggestion);
        }
    }

    fn view_session_history(&self) {
        if self.session_history.is_empty() {
            println!("\n📊 No interview sessions completed yet.");
            println!("Complete some practice sessions to see your progress!");
            return;
        }
        
        println!("\n📊 Interview Session History");
        println!("============================");
        
        for (i, session) in self.session_history.iter().enumerate() {
            let duration = session.start_time.elapsed();
            let completion_rate = session.solutions_attempted.len() as f64 / session.problems.len() as f64;
            
            println!("\nSession {} ({:?}):", i + 1, session.session_type);
            println!("• Duration: {} minutes", duration.as_secs() / 60);
            println!("• Completion rate: {:.1}%", completion_rate * 100.0);
            println!("• Problems: {}", session.problems.iter()
                .map(|p| format!("{} ({})", p.title, p.difficulty))
                .collect::<Vec<_>>().join(", "));
        }
        
        // Overall statistics
        let total_sessions = self.session_history.len();
        let avg_completion = self.session_history.iter()
            .map(|s| s.solutions_attempted.len() as f64 / s.problems.len() as f64)
            .sum::<f64>() / total_sessions as f64;
        
        println!("\n📈 Overall Statistics:");
        println!("• Total sessions: {}", total_sessions);
        println!("• Average completion rate: {:.1}%", avg_completion * 100.0);
    }

    fn practice_specific_topic(&self) {
        println!("\n🎯 Topic-Focused Practice");
        println!("=========================");
        println!("Choose a topic to focus on:");
        println!("1. Arrays & Hash Tables");
        println!("2. Strings & Pattern Matching");
        println!("3. Trees & Graphs");
        println!("4. Dynamic Programming");
        println!("5. Sorting & Searching");
        
        print!("Enter choice (1-5): ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        
        match input.trim() {
            "1" => println!("🎯 Focus on: Two Sum, 3Sum, Group Anagrams, LRU Cache"),
            "2" => println!("🎯 Focus on: Longest Substring, Valid Parentheses, Pattern Matching"),
            "3" => println!("🎯 Focus on: Tree Traversals, Graph DFS/BFS, Path Finding"),
            "4" => println!("🎯 Focus on: Climbing Stairs, House Robber, Edit Distance"),
            "5" => println!("🎯 Focus on: Binary Search variants, Merge Sort, Quick Select"),
            _ => println!("Invalid choice."),
        }
    }

    fn show_interview_tips(&self) {
        println!("\n💡 Technical Interview Best Practices");
        println!("=====================================");
        
        println!("\n🗣️ Communication Tips:");
        println!("• Think out loud - explain your thought process");
        println!("• Ask clarifying questions about the problem");
        println!("• Discuss trade-offs between different approaches");
        println!("• Explain time and space complexity");
        
        println!("\n🧠 Problem-Solving Strategy:");
        println!("• Understand the problem completely first");
        println!("• Start with a brute force solution");
        println!("• Identify patterns and optimize step by step");
        println!("• Consider edge cases and constraints");
        
        println!("\n💻 Coding Tips:");
        println!("• Write clean, readable code");
        println!("• Use meaningful variable names");
        println!("• Add comments for complex logic");
        println!("• Test your solution with examples");
        
        println!("\n⏰ Time Management:");
        println!("• Spend 25% time understanding/planning");
        println!("• Spend 50% time coding the solution");
        println!("• Spend 25% time testing and optimization");
        
        println!("\n🎯 Common Patterns to Master:");
        println!("• Two Pointers / Sliding Window");
        println!("• Hash Maps for O(1) lookups");
        println!("• Binary Search on sorted data");
        println!("• DFS/BFS for trees and graphs");
        println!("• Dynamic Programming for optimization");
        
        println!("\n🚫 What to Avoid:");
        println!("• Don't jump into coding immediately");
        println!("• Don't stay silent during problem solving");
        println!("• Don't ignore edge cases");
        println!("• Don't panic if you get stuck - ask for hints");
    }
}

fn main() {
    let mut simulator = InterviewSimulator::new();
    simulator.run();
}

