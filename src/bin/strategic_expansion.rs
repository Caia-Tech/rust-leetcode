#!/usr/bin/env cargo-script

//! # Strategic Expansion Tool
//! 
//! Automated execution of the strategic expansion plan to scale from 105 to 1000+ problems.
//! This tool implements the phased approach outlined in the expansion roadmap.

use rust_leetcode::api::{ProblemFetcher, Company, AlgorithmPattern};
use std::collections::HashSet;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Strategic LeetCode Expansion Tool");
    println!("====================================");
    println!();

    let mut fetcher = ProblemFetcher::new();
    
    // Phase 1: Essential Interview Problems
    execute_phase_1(&mut fetcher).await?;
    
    // Phase 2: Algorithm Pattern Completion
    execute_phase_2(&mut fetcher).await?;
    
    // Phase 3: Company Focus
    execute_phase_3(&mut fetcher).await?;
    
    // Final Status Report
    print_final_status(&mut fetcher).await?;
    
    println!("🎉 Strategic expansion completed successfully!");
    println!("🚀 Repository transformed from curated collection to comprehensive learning platform!");

    Ok(())
}

async fn execute_phase_1(fetcher: &mut ProblemFetcher) -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 PHASE 1: Essential Interview Problems");
    println!("==========================================");
    println!();

    // Get current statistics
    println!("📊 Analyzing current repository status...");
    match fetcher.get_repository_stats().await {
        Ok(stats) => {
            println!("✅ Current status:");
            println!("   - Local problems: {}", stats.local_stats.total_problems);
            println!("   - Total LeetCode problems: {}", stats.api_stats.total_problems);
            println!("   - Coverage: {:.2}%", stats.coverage_percentage);
            println!();
        }
        Err(e) => println!("⚠️  Could not fetch stats: {}", e),
    }

    // Add top interview problems
    println!("🎯 Adding top 25 interview problems...");
    match fetcher.add_top_interview_problems(25).await {
        Ok(added_problems) => {
            println!("✅ Successfully added {} problems:", added_problems.len());
            for (i, problem) in added_problems.iter().enumerate().take(10) {
                println!("   {}. {}", i + 1, problem.split('/').last().unwrap_or(problem));
            }
            if added_problems.len() > 10 {
                println!("   ... and {} more", added_problems.len() - 10);
            }
            println!();
        }
        Err(e) => println!("❌ Failed to add interview problems: {}", e),
    }

    println!("✅ Phase 1 completed!\n");
    Ok(())
}

async fn execute_phase_2(fetcher: &mut ProblemFetcher) -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 PHASE 2: Algorithm Pattern Completion");
    println!("=========================================");
    println!();

    let patterns = vec![
        ("Two Pointers", AlgorithmPattern::two_pointers(), 8),
        ("Sliding Window", AlgorithmPattern::sliding_window(), 8),
        ("Tree Traversal", AlgorithmPattern::tree_traversal(), 10),
        ("Graph Traversal", AlgorithmPattern::graph_traversal(), 8),
        ("Dynamic Programming", AlgorithmPattern::dynamic_programming(), 12),
        ("Fast & Slow Pointers", AlgorithmPattern::fast_slow_pointers(), 6),
        ("Merge Intervals", AlgorithmPattern::merge_intervals(), 6),
        ("Cyclic Sort", AlgorithmPattern::cyclic_sort(), 5),
    ];

    for (name, pattern, count) in patterns {
        println!("🧩 Adding {} problems for {} pattern...", count, name);
        match fetcher.add_problems_by_pattern(pattern, count).await {
            Ok(added_problems) => {
                println!("   ✅ Added {} {} problems", added_problems.len(), name);
            }
            Err(e) => println!("   ⚠️  Could not add {} problems: {}", name, e),
        }
    }

    println!("\n✅ Phase 2 completed!\n");
    Ok(())
}

async fn execute_phase_3(fetcher: &mut ProblemFetcher) -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 PHASE 3: Company Focus");
    println!("==========================");
    println!();

    let companies = vec![
        ("Amazon", Company::amazon(), 15),
        ("Google", Company::google(), 15),
        ("Microsoft", Company::microsoft(), 12),
        ("Meta", Company::facebook(), 12),
        ("Apple", Company::apple(), 10),
    ];

    for (name, company, count) in companies {
        println!("🏢 Adding {} problems from {}...", count, name);
        match fetcher.add_problems_by_company(company, count).await {
            Ok(added_problems) => {
                println!("   ✅ Added {} {} problems", added_problems.len(), name);
            }
            Err(e) => println!("   ⚠️  Could not add {} problems: {}", name, e),
        }
    }

    println!("\n✅ Phase 3 completed!\n");
    Ok(())
}

async fn print_final_status(fetcher: &mut ProblemFetcher) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 FINAL STATUS REPORT");
    println!("=======================");
    println!();

    match fetcher.get_repository_stats().await {
        Ok(stats) => {
            println!("🏆 Expansion Results:");
            println!("   - Final problem count: {}", stats.local_stats.total_problems);
            println!("   - Easy: {}", stats.local_stats.easy_count);
            println!("   - Medium: {}", stats.local_stats.medium_count);
            println!("   - Hard: {}", stats.local_stats.hard_count);
            println!("   - Coverage: {:.2}%", stats.coverage_percentage);
            println!();

            let expansion_factor = stats.local_stats.total_problems as f64 / 105.0;
            println!("📈 Expansion Metrics:");
            println!("   - Starting problems: 105");
            println!("   - Final problems: {}", stats.local_stats.total_problems);
            println!("   - Growth factor: {:.1}x", expansion_factor);
            
            if stats.coverage_percentage >= 10.0 {
                println!("   - ✅ Phase 1 target (10% coverage) achieved!");
            }
            
            println!();
            
            if stats.local_stats.total_problems >= 300 {
                println!("🎯 SUCCESS: Reached Phase 1 target of 300+ problems!");
            } else {
                println!("📋 STATUS: {} problems added, continue expansion for Phase 1 target", 
                    stats.local_stats.total_problems - 105);
            }
        }
        Err(e) => println!("❌ Could not generate final status: {}", e),
    }

    Ok(())
}

// Helper function to identify most valuable problems to add next
async fn identify_priority_problems() -> Vec<&'static str> {
    // These are consistently ranked as top interview problems across multiple sources
    vec![
        "Valid Palindrome II",
        "Meeting Rooms II", 
        "Design Hit Counter",
        "Add Two Numbers II",
        "Flatten Binary Tree to Linked List",
        "Binary Tree Right Side View",
        "Kth Largest Element in Stream",
        "Top K Frequent Elements",
        "Design Twitter",
        "LFU Cache",
        "Design Search Autocomplete System",
        "Alien Dictionary",
        "Course Schedule II",
        "Graph Valid Tree",
        "Number of Connected Components",
        "Palindrome Partitioning",
        "Word Break II",
        "Combination Sum II",
        "Permutations II",
        "N-Queens",
        "Sudoku Solver",
        "Word Ladder II",
        "Minimum Window Substring",
        "Sliding Window Maximum",
        "Find Median from Data Stream",
    ]
}