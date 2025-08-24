#!/usr/bin/env cargo-script

//! # LeetCode API Demo
//! 
//! Demonstration of the LeetCode API integration capabilities.
//! Shows fetching problems, statistics, and automated template generation.

use rust_leetcode::api::{LeetCodeAPIClient, Company, AlgorithmPattern, Difficulty};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 LeetCode API Integration Demo");
    println!("==================================");
    println!();

    let mut client = LeetCodeAPIClient::new();
    
    // Test 1: Get problem statistics
    println!("📊 Fetching LeetCode problem statistics...");
    match client.get_problem_stats().await {
        Ok(stats) => {
            println!("✅ Total problems: {}", stats.total_problems);
            println!("   - Easy: {}", stats.easy_count);
            println!("   - Medium: {}", stats.medium_count);
            println!("   - Hard: {}", stats.hard_count);
            println!("   - Average acceptance: {:.1}%", stats.acceptance_rate);
        }
        Err(e) => println!("❌ Failed to get stats: {}", e),
    }
    println!();

    // Test 2: Fetch problems by company
    println!("🏢 Fetching Amazon interview problems (top 10)...");
    match client.fetch_problems_by_company(Company::amazon(), Some(10)).await {
        Ok(problems) => {
            println!("✅ Found {} Amazon problems:", problems.len());
            for (i, problem) in problems.iter().enumerate().take(5) {
                println!("   {}. {} ({})", i + 1, problem.title, problem.difficulty);
            }
            if problems.len() > 5 {
                println!("   ... and {} more", problems.len() - 5);
            }
        }
        Err(e) => println!("❌ Failed to fetch Amazon problems: {}", e),
    }
    println!();

    // Test 3: Fetch problems by pattern
    println!("🧩 Fetching Two Pointers problems (top 5)...");
    match client.fetch_problems_by_pattern(AlgorithmPattern::two_pointers(), Some(5)).await {
        Ok(problems) => {
            println!("✅ Found {} Two Pointers problems:", problems.len());
            for (i, problem) in problems.iter().enumerate() {
                let tags: Vec<String> = problem.topic_tags.iter().map(|t| t.name.clone()).collect();
                println!("   {}. {} ({}) - Tags: {}", 
                    i + 1, 
                    problem.title, 
                    problem.difficulty,
                    tags.join(", ")
                );
            }
        }
        Err(e) => println!("❌ Failed to fetch Two Pointers problems: {}", e),
    }
    println!();

    // Test 4: Search specific problems
    println!("🔍 Searching for 'Two Sum' problems...");
    match client.search_problems("Two Sum").await {
        Ok(problems) => {
            println!("✅ Found {} matching problems:", problems.len());
            for (i, problem) in problems.iter().enumerate().take(3) {
                println!("   {}. {} ({})", i + 1, problem.title, problem.difficulty);
            }
        }
        Err(e) => println!("❌ Failed to search problems: {}", e),
    }
    println!();

    // Test 5: Get top interview problems
    println!("🎯 Fetching top interview problems (top 5)...");
    match client.get_top_interview_problems(5).await {
        Ok(problems) => {
            println!("✅ Top interview problems:");
            for (i, problem) in problems.iter().enumerate() {
                println!("   {}. {} ({}) - {}% acceptance", 
                    i + 1, 
                    problem.title, 
                    problem.difficulty,
                    problem.acceptance_rate.unwrap_or(0.0) as i32
                );
            }
        }
        Err(e) => println!("❌ Failed to fetch top interview problems: {}", e),
    }

    println!();
    println!("🎉 API Demo completed successfully!");
    println!("💡 The LeetCode API integration is working and ready for automated problem fetching.");

    Ok(())
}