#!/usr/bin/env cargo-script

//! # LeetCode Problem Fetcher
//! 
//! Automated tool for fetching and adding LeetCode problems to the repository.
//! Provides various strategies for expanding the problem collection systematically.

use rust_leetcode::api::{ProblemFetcher, Company, AlgorithmPattern, Difficulty};
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 LeetCode Problem Fetcher");
    println!("============================");
    println!();

    let mut fetcher = ProblemFetcher::new();
    
    loop {
        print_menu();
        
        let choice = get_user_input("Enter your choice: ")?;
        
        match choice.as_str() {
            "1" => show_repository_stats(&mut fetcher).await?,
            "2" => fetch_top_interview_problems(&mut fetcher).await?,
            "3" => fetch_problems_by_company(&mut fetcher).await?,
            "4" => fetch_problems_by_difficulty(&mut fetcher).await?,
            "5" => fetch_problems_by_pattern(&mut fetcher).await?,
            "6" => find_missing_problems(&mut fetcher).await?,
            "7" => batch_add_problems(&mut fetcher).await?,
            "8" => {
                println!("👋 Goodbye!");
                break;
            }
            _ => println!("❌ Invalid choice. Please try again."),
        }
        
        println!();
        println!("Press Enter to continue...");
        let _ = io::stdin().read_line(&mut String::new());
        println!();
    }

    Ok(())
}

fn print_menu() {
    println!("📋 Available Operations:");
    println!("1. 📊 Show repository statistics");
    println!("2. 🎯 Fetch top interview problems");
    println!("3. 🏢 Fetch problems by company");
    println!("4. ⚖️  Fetch problems by difficulty");
    println!("5. 🧩 Fetch problems by algorithm pattern");
    println!("6. 🔍 Find missing problems");
    println!("7. 📦 Batch add problems");
    println!("8. 🚪 Exit");
    println!();
}

async fn show_repository_stats(fetcher: &mut ProblemFetcher) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Fetching repository statistics...");
    
    match fetcher.get_repository_stats().await {
        Ok(stats) => {
            println!("{}", stats);
            
            if stats.coverage_percentage < 10.0 {
                println!("💡 Tip: Your coverage is below 10%. Consider adding top interview problems first!");
            } else if stats.coverage_percentage < 25.0 {
                println!("💡 Tip: Good progress! Consider focusing on specific companies or patterns.");
            } else {
                println!("🎉 Excellent coverage! You're well on your way to comprehensive mastery.");
            }
        }
        Err(e) => println!("❌ Failed to fetch statistics: {}", e),
    }
    
    Ok(())
}

async fn fetch_top_interview_problems(fetcher: &mut ProblemFetcher) -> Result<(), Box<dyn std::error::Error>> {
    let count_str = get_user_input("Enter number of top problems to fetch (default: 50): ")?;
    let count: i32 = if count_str.trim().is_empty() {
        50
    } else {
        count_str.trim().parse().unwrap_or(50)
    };
    
    println!("🎯 Fetching top {} interview problems...", count);
    
    match fetcher.add_top_interview_problems(count).await {
        Ok(added_problems) => {
            println!("✅ Successfully added {} problems:", added_problems.len());
            for (i, problem) in added_problems.iter().enumerate().take(10) {
                println!("  {}. {}", i + 1, problem);
            }
            if added_problems.len() > 10 {
                println!("  ... and {} more", added_problems.len() - 10);
            }
        }
        Err(e) => println!("❌ Failed to fetch problems: {}", e),
    }
    
    Ok(())
}

async fn fetch_problems_by_company(fetcher: &mut ProblemFetcher) -> Result<(), Box<dyn std::error::Error>> {
    println!("🏢 Available companies:");
    println!("1. Amazon");
    println!("2. Google");
    println!("3. Microsoft");
    println!("4. Meta (Facebook)");
    println!("5. Apple");
    
    let choice = get_user_input("Select company (1-5): ")?;
    let company = match choice.as_str() {
        "1" => Company::amazon(),
        "2" => Company::google(),
        "3" => Company::microsoft(),
        "4" => Company::facebook(),
        "5" => Company::apple(),
        _ => {
            println!("❌ Invalid choice");
            return Ok(());
        }
    };
    
    let count_str = get_user_input("Enter number of problems to fetch (default: 25): ")?;
    let count: i32 = if count_str.trim().is_empty() {
        25
    } else {
        count_str.trim().parse().unwrap_or(25)
    };
    
    println!("🏢 Fetching {} problems from {}...", count, company.name);
    
    match fetcher.add_problems_by_company(company, count).await {
        Ok(added_problems) => {
            println!("✅ Successfully added {} problems:", added_problems.len());
            for (i, problem) in added_problems.iter().enumerate().take(5) {
                println!("  {}. {}", i + 1, problem);
            }
            if added_problems.len() > 5 {
                println!("  ... and {} more", added_problems.len() - 5);
            }
        }
        Err(e) => println!("❌ Failed to fetch problems: {}", e),
    }
    
    Ok(())
}

async fn fetch_problems_by_difficulty(fetcher: &mut ProblemFetcher) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚖️ Available difficulties:");
    println!("1. Easy");
    println!("2. Medium");
    println!("3. Hard");
    
    let choice = get_user_input("Select difficulty (1-3): ")?;
    let difficulty = match choice.as_str() {
        "1" => Difficulty::Easy,
        "2" => Difficulty::Medium,
        "3" => Difficulty::Hard,
        _ => {
            println!("❌ Invalid choice");
            return Ok(());
        }
    };
    
    let count_str = get_user_input("Enter number of problems to fetch (default: 20): ")?;
    let count: i32 = if count_str.trim().is_empty() {
        20
    } else {
        count_str.trim().parse().unwrap_or(20)
    };
    
    println!("⚖️ Fetching {} {} problems...", count, difficulty);
    
    // For now, we'll use the company fetcher as a placeholder since we need API implementation
    println!("⚠️ Difficulty-based fetching not yet implemented. Use company or pattern-based fetching instead.");
    
    Ok(())
}

async fn fetch_problems_by_pattern(fetcher: &mut ProblemFetcher) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧩 Available algorithm patterns:");
    println!("1. Two Pointers");
    println!("2. Sliding Window");
    println!("3. Fast & Slow Pointers");
    println!("4. Merge Intervals");
    println!("5. Cyclic Sort");
    println!("6. Tree Traversal");
    println!("7. Graph Traversal");
    println!("8. Dynamic Programming");
    
    let choice = get_user_input("Select pattern (1-8): ")?;
    let pattern = match choice.as_str() {
        "1" => AlgorithmPattern::two_pointers(),
        "2" => AlgorithmPattern::sliding_window(),
        "3" => AlgorithmPattern::fast_slow_pointers(),
        "4" => AlgorithmPattern::merge_intervals(),
        "5" => AlgorithmPattern::cyclic_sort(),
        "6" => AlgorithmPattern::tree_traversal(),
        "7" => AlgorithmPattern::graph_traversal(),
        "8" => AlgorithmPattern::dynamic_programming(),
        _ => {
            println!("❌ Invalid choice");
            return Ok(());
        }
    };
    
    let count_str = get_user_input("Enter number of problems to fetch (default: 15): ")?;
    let count: i32 = if count_str.trim().is_empty() {
        15
    } else {
        count_str.trim().parse().unwrap_or(15)
    };
    
    println!("🧩 Fetching {} problems for {} pattern...", count, pattern.name);
    
    match fetcher.add_problems_by_pattern(pattern, count).await {
        Ok(added_problems) => {
            println!("✅ Successfully added {} problems:", added_problems.len());
            for (i, problem) in added_problems.iter().enumerate().take(5) {
                println!("  {}. {}", i + 1, problem);
            }
            if added_problems.len() > 5 {
                println!("  ... and {} more", added_problems.len() - 5);
            }
        }
        Err(e) => println!("❌ Failed to fetch problems: {}", e),
    }
    
    Ok(())
}

async fn find_missing_problems(fetcher: &mut ProblemFetcher) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Finding missing problems...");
    
    match fetcher.find_missing_problems(None).await {
        Ok(missing_problems) => {
            println!("📋 Found {} missing problems", missing_problems.len());
            
            if missing_problems.is_empty() {
                println!("🎉 No missing problems found! Your collection is complete.");
            } else {
                println!("📊 Missing problems breakdown:");
                let easy_count = missing_problems.iter().filter(|p| p.difficulty == "Easy").count();
                let medium_count = missing_problems.iter().filter(|p| p.difficulty == "Medium").count();
                let hard_count = missing_problems.iter().filter(|p| p.difficulty == "Hard").count();
                
                println!("  - Easy: {}", easy_count);
                println!("  - Medium: {}", medium_count);
                println!("  - Hard: {}", hard_count);
                
                println!("\n📝 Sample missing problems:");
                for (i, problem) in missing_problems.iter().enumerate().take(10) {
                    println!("  {}. {} ({})", i + 1, problem.title, problem.difficulty);
                }
                
                if missing_problems.len() > 10 {
                    println!("  ... and {} more", missing_problems.len() - 10);
                }
            }
        }
        Err(e) => println!("❌ Failed to find missing problems: {}", e),
    }
    
    Ok(())
}

async fn batch_add_problems(fetcher: &mut ProblemFetcher) -> Result<(), Box<dyn std::error::Error>> {
    println!("📦 Batch Add Problems");
    println!("This will add problems in a strategic order for optimal learning progression.");
    println!();
    
    let confirm = get_user_input("Continue with batch addition? (y/n): ")?;
    if confirm.to_lowercase() != "y" && confirm.to_lowercase() != "yes" {
        println!("❌ Batch addition cancelled.");
        return Ok(());
    }
    
    println!("🚀 Starting strategic batch addition...");
    
    // Phase 1: Top interview problems
    println!("\n📋 Phase 1: Adding top interview problems...");
    if let Ok(problems) = fetcher.add_top_interview_problems(25).await {
        println!("✅ Added {} top interview problems", problems.len());
    }
    
    // Phase 2: Algorithm patterns
    println!("\n📋 Phase 2: Adding algorithm pattern problems...");
    let patterns = vec![
        AlgorithmPattern::two_pointers(),
        AlgorithmPattern::sliding_window(),
        AlgorithmPattern::tree_traversal(),
    ];
    
    for pattern in patterns {
        if let Ok(problems) = fetcher.add_problems_by_pattern(pattern.clone(), 10).await {
            println!("✅ Added {} problems for {} pattern", problems.len(), pattern.name);
        }
    }
    
    // Phase 3: Company problems
    println!("\n📋 Phase 3: Adding company-specific problems...");
    let companies = vec![Company::amazon(), Company::google()];
    
    for company in companies {
        if let Ok(problems) = fetcher.add_problems_by_company(company.clone(), 15).await {
            println!("✅ Added {} problems from {}", problems.len(), company.name);
        }
    }
    
    println!("\n🎉 Batch addition completed!");
    println!("📊 Run statistics to see your updated coverage.");
    
    Ok(())
}

fn get_user_input(prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
    print!("{}", prompt);
    io::stdout().flush()?;
    
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    
    Ok(input.trim().to_string())
}