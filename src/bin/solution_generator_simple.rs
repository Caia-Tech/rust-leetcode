#!/usr/bin/env cargo-script

//! Simple Solution Template Generator for LeetCode Problems
//! 
//! This simplified tool generates basic boilerplate code for new LeetCode problems.

use std::fs;
use std::io::{self, Write};

#[derive(Debug)]
struct ProblemInfo {
    id: u32,
    title: String,
    difficulty: String,
    description: String,
    function_signature: String,
    return_type: String,
    topics: Vec<String>,
}

fn main() {
    println!("🏗️  LeetCode Solution Template Generator");
    println!("========================================");
    
    let problem = collect_problem_info();
    let difficulty_folder = problem.difficulty.to_lowercase();
    
    // Generate solution file
    generate_solution_file(&problem, &difficulty_folder);
    
    // Update module declarations
    update_mod_file(&problem, &difficulty_folder);
    
    println!("\n✅ Successfully generated templates for Problem {}: {}", problem.id, problem.title);
    println!("\n📁 Generated files:");
    println!("  • src/{}/{}.rs", difficulty_folder, problem.title.to_lowercase().replace(' ', "_"));
    println!("  • Updated src/{}/mod.rs", difficulty_folder);
}

fn collect_problem_info() -> ProblemInfo {
    print!("Enter problem ID: ");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let id: u32 = input.trim().parse().expect("Invalid problem ID");
    
    print!("Enter problem title: ");
    io::stdout().flush().unwrap();
    input.clear();
    io::stdin().read_line(&mut input).unwrap();
    let title = input.trim().to_string();
    
    print!("Enter difficulty (Easy/Medium/Hard): ");
    io::stdout().flush().unwrap();
    input.clear();
    io::stdin().read_line(&mut input).unwrap();
    let difficulty = input.trim().to_string();
    
    print!("Enter brief problem description: ");
    io::stdout().flush().unwrap();
    input.clear();
    io::stdin().read_line(&mut input).unwrap();
    let description = input.trim().to_string();
    
    print!("Enter function name (e.g., two_sum): ");
    io::stdout().flush().unwrap();
    input.clear();
    io::stdin().read_line(&mut input).unwrap();
    let function_signature = input.trim().to_string();
    
    print!("Enter return type (e.g., Vec<i32>, i32, bool): ");
    io::stdout().flush().unwrap();
    input.clear();
    io::stdin().read_line(&mut input).unwrap();
    let return_type = input.trim().to_string();
    
    print!("Enter topics (comma-separated): ");
    io::stdout().flush().unwrap();
    input.clear();
    io::stdin().read_line(&mut input).unwrap();
    let topics: Vec<String> = input.trim().split(',').map(|s| s.trim().to_string()).collect();
    
    ProblemInfo {
        id,
        title,
        difficulty,
        description,
        function_signature,
        return_type,
        topics,
    }
}

fn generate_solution_file(problem: &ProblemInfo, difficulty: &str) {
    let filename = format!("{}.rs", problem.title.to_lowercase().replace(' ', "_"));
    let filepath = format!("src/{}/{}", difficulty, filename);
    
    let template = generate_solution_template(problem);
    
    fs::write(&filepath, template).expect("Failed to write solution file");
    println!("✅ Generated: {}", filepath);
}

fn generate_solution_template(problem: &ProblemInfo) -> String {
    let func_name = &problem.function_signature;
    let return_type = &problem.return_type;
    let topics = problem.topics.join(", ");
    
    format!(r#"//! # Problem {}: {}
//!
//! {}
//!
//! ## Topics
//! {}

/// Solution struct following LeetCode format
pub struct Solution;

impl Solution {{
    pub fn new() -> Self {{
        Solution
    }}

    /// Main solution approach
    pub fn {}(&self, input: Vec<i32>) -> {} {{
        todo!("Implement solution")
    }}

    /// Alternative approach
    pub fn {}_alternative(&self, input: Vec<i32>) -> {} {{
        todo!("Implement alternative solution")
    }}
}}

impl Default for Solution {{
    fn default() -> Self {{
        Self::new()
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_basic_cases() {{
        let solution = Solution::new();
        
        // Add test cases here
        assert_eq!(solution.{}(vec![]), /* expected */);
    }}
}}
"#,
        problem.id,
        problem.title,
        problem.description,
        topics,
        func_name,
        return_type,
        func_name,
        return_type,
        func_name
    )
}

fn update_mod_file(problem: &ProblemInfo, difficulty: &str) {
    let mod_file = format!("src/{}/mod.rs", difficulty);
    let module_name = problem.title.to_lowercase().replace(' ', "_");
    let new_line = format!("pub mod {};", module_name);
    
    if let Ok(content) = fs::read_to_string(&mod_file) {
        if !content.contains(&new_line) {
            let updated_content = format!("{}\n{}", content, new_line);
            fs::write(&mod_file, updated_content).expect("Failed to update mod.rs");
            println!("✅ Updated: {}", mod_file);
        }
    }
}