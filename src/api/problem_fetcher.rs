//! Problem fetcher for automated problem addition

use crate::api::types::*;
use crate::api::leetcode_client::LeetCodeAPIClient;
use std::fs;
use std::path::Path;

/// Problem fetcher with automated solution template generation
pub struct ProblemFetcher {
    client: LeetCodeAPIClient,
    template_generator: SolutionTemplateGenerator,
}

impl ProblemFetcher {
    /// Create a new problem fetcher
    pub fn new() -> Self {
        Self {
            client: LeetCodeAPIClient::new(),
            template_generator: SolutionTemplateGenerator::new(),
        }
    }

    /// Fetch and add top interview problems to repository
    pub async fn add_top_interview_problems(&mut self, count: i32) -> Result<Vec<String>, APIError> {
        let problems = self.client.get_top_interview_problems(count).await?;
        let mut added_problems = Vec::new();

        for problem in problems {
            if let Ok(file_path) = self.add_problem_to_repository(&problem).await {
                added_problems.push(file_path);
            }
        }

        Ok(added_problems)
    }

    /// Fetch and add problems by company
    pub async fn add_problems_by_company(
        &mut self, 
        company: Company, 
        count: i32
    ) -> Result<Vec<String>, APIError> {
        let problems = self.client.fetch_problems_by_company(company, Some(count)).await?;
        let mut added_problems = Vec::new();

        for problem in problems {
            if let Ok(file_path) = self.add_problem_to_repository(&problem).await {
                added_problems.push(file_path);
            }
        }

        Ok(added_problems)
    }

    /// Add problems by algorithm pattern
    pub async fn add_problems_by_pattern(
        &mut self,
        pattern: AlgorithmPattern,
        count: i32
    ) -> Result<Vec<String>, APIError> {
        let problems = self.client.fetch_problems_by_pattern(pattern, Some(count)).await?;
        let mut added_problems = Vec::new();

        for problem in problems {
            if let Ok(file_path) = self.add_problem_to_repository(&problem).await {
                added_problems.push(file_path);
            }
        }

        Ok(added_problems)
    }

    /// Add a single problem to the repository
    async fn add_problem_to_repository(&self, problem: &LeetCodeProblem) -> Result<String, Box<dyn std::error::Error>> {
        // Check if problem already exists
        let difficulty_folder = problem.difficulty.to_lowercase();
        let filename = self.generate_filename(&problem.title);
        let file_path = format!("src/{}/{}.rs", difficulty_folder, filename);

        if Path::new(&file_path).exists() {
            println!("Problem {} already exists, skipping", problem.title);
            return Ok(file_path);
        }

        // Generate solution template
        let template = self.template_generator.generate_template(problem);

        // Create directory if it doesn't exist
        fs::create_dir_all(format!("src/{}", difficulty_folder))?;

        // Write solution file
        fs::write(&file_path, template)?;

        // Update mod.rs file
        self.update_mod_file(&difficulty_folder, &filename)?;

        println!("✅ Added problem {}: {} -> {}", problem.frontend_id, problem.title, file_path);
        Ok(file_path)
    }

    /// Generate filename from problem title
    fn generate_filename(&self, title: &str) -> String {
        title
            .to_lowercase()
            .replace(' ', "_")
            .replace('-', "_")
            .replace('(', "")
            .replace(')', "")
            .replace('/', "_")
            .replace("'", "")
            .replace('"', "")
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect()
    }

    /// Update mod.rs file with new module
    fn update_mod_file(&self, difficulty: &str, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mod_file = format!("src/{}/mod.rs", difficulty);
        let module_line = format!("pub mod {};", filename);

        if let Ok(content) = fs::read_to_string(&mod_file) {
            if !content.contains(&module_line) {
                let updated_content = format!("{}\n{}", content.trim(), module_line);
                fs::write(&mod_file, updated_content)?;
            }
        } else {
            // Create mod.rs if it doesn't exist
            fs::write(&mod_file, &module_line)?;
        }

        Ok(())
    }

    /// Get repository statistics
    pub async fn get_repository_stats(&mut self) -> Result<RepositoryStats, APIError> {
        let api_stats = self.client.get_problem_stats().await?;
        let local_stats = self.count_local_problems()?;

        let coverage_percentage = (local_stats.total_problems as f64 / api_stats.total_problems as f64) * 100.0;
        
        Ok(RepositoryStats {
            api_stats,
            local_stats,
            coverage_percentage,
        })
    }

    /// Count problems in local repository
    fn count_local_problems(&self) -> Result<LocalStats, Box<dyn std::error::Error>> {
        let mut easy_count = 0;
        let mut medium_count = 0;
        let mut hard_count = 0;

        for difficulty in &["easy", "medium", "hard"] {
            let dir_path = format!("src/{}", difficulty);
            if let Ok(entries) = fs::read_dir(&dir_path) {
                let count = entries
                    .filter_map(|entry| entry.ok())
                    .filter(|entry| {
                        entry.path().extension()
                            .and_then(|s| s.to_str())
                            .map(|s| s == "rs")
                            .unwrap_or(false)
                    })
                    .filter(|entry| {
                        entry.file_name().to_str()
                            .map(|name| name != "mod.rs")
                            .unwrap_or(false)
                    })
                    .count() as i32;

                match *difficulty {
                    "easy" => easy_count = count,
                    "medium" => medium_count = count,
                    "hard" => hard_count = count,
                    _ => {}
                }
            }
        }

        Ok(LocalStats {
            total_problems: easy_count + medium_count + hard_count,
            easy_count,
            medium_count,
            hard_count,
        })
    }

    /// Search for missing problems in current implementation
    pub async fn find_missing_problems(&mut self, pattern: Option<AlgorithmPattern>) -> Result<Vec<LeetCodeProblem>, APIError> {
        let all_problems = if let Some(pattern) = pattern {
            self.client.fetch_problems_by_pattern(pattern, None).await?
        } else {
            self.client.fetch_all_problems().await?
        };

        let local_stats = self.count_local_problems().map_err(|e| APIError::ParseError(e.to_string()))?;
        
        // For now, return all problems since we don't have a mapping of implemented problems
        // In the future, this could be enhanced to check against a database or file manifest
        let missing_count = all_problems.len().saturating_sub(local_stats.total_problems as usize);
        
        Ok(all_problems.into_iter().take(missing_count).collect())
    }
}

/// Solution template generator
struct SolutionTemplateGenerator;

impl SolutionTemplateGenerator {
    fn new() -> Self {
        Self
    }

    fn generate_template(&self, problem: &LeetCodeProblem) -> String {
        let function_name = self.generate_function_name(&problem.title);
        let topics = problem.topic_tags.iter().map(|tag| tag.name.clone()).collect::<Vec<_>>().join(", ");
        
        format!(r#"//! # Problem {}: {}
//!
//! **Difficulty**: {}
//! **Topics**: {}
//! **Acceptance Rate**: {:.1}%

/// Solution struct following LeetCode format
pub struct Solution;

impl Solution {{
    /// Create a new solution instance
    pub fn new() -> Self {{
        Solution
    }}

    /// Main solution approach
    /// 
    /// Time Complexity: O(?) - TODO: Analyze and update
    /// Space Complexity: O(?) - TODO: Analyze and update
    pub fn {}(&self, input: Vec<i32>) -> Vec<i32> {{
        // TODO: Implement solution
        todo!("Implement {} solution")
    }}

    /// Alternative solution approach
    /// 
    /// Time Complexity: O(?) - TODO: Analyze and update  
    /// Space Complexity: O(?) - TODO: Analyze and update
    pub fn {}_alternative(&self, input: Vec<i32>) -> Vec<i32> {{
        // TODO: Implement alternative solution
        todo!("Implement alternative {} solution")
    }}

    /// Brute force solution (for comparison)
    /// 
    /// Time Complexity: O(?) - TODO: Analyze and update
    /// Space Complexity: O(?) - TODO: Analyze and update  
    pub fn {}_brute_force(&self, input: Vec<i32>) -> Vec<i32> {{
        // TODO: Implement brute force solution
        todo!("Implement brute force {} solution")
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
        
        // TODO: Add test cases based on problem examples
        // Example:
        // assert_eq!(solution.{}(vec![1, 2, 3]), vec![expected]);
    }}

    #[test]
    fn test_edge_cases() {{
        let solution = Solution::new();
        
        // TODO: Add edge case tests
        // - Empty input
        // - Single element
        // - Minimum constraints
        // - Maximum constraints
    }}

    #[test]
    fn test_approach_consistency() {{
        let solution = Solution::new();
        
        // TODO: Test that all approaches return the same results
        let test_cases = vec![
            // Add test cases here
        ];

        for case in test_cases {{
            // Uncomment when solutions are implemented
            // assert_eq!(solution.{}(case.clone()), solution.{}_alternative(case.clone()));
            // assert_eq!(solution.{}(case.clone()), solution.{}_brute_force(case));
        }}
    }}
}}

#[cfg(test)]
mod benchmarks {{
    use super::*;
    // Note: Benchmarks will be added to benches/ directory for Criterion integration
}}
"#,
            problem.frontend_id,
            problem.title,
            problem.difficulty,
            topics,
            problem.acceptance_rate.unwrap_or(0.0),
            function_name,
            problem.title,
            function_name,
            problem.title,
            function_name,
            problem.title,
            function_name,
            function_name,
            function_name,
            function_name,
            function_name,
        )
    }

    fn generate_function_name(&self, title: &str) -> String {
        title
            .to_lowercase()
            .replace(' ', "_")
            .replace('-', "_")
            .replace('(', "")
            .replace(')', "")
            .replace('/', "_")
            .replace("'", "")
            .replace('"', "")
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect()
    }
}

#[derive(Debug)]
pub struct RepositoryStats {
    pub api_stats: ProblemStats,
    pub local_stats: LocalStats,
    pub coverage_percentage: f64,
}

#[derive(Debug, Clone)]
pub struct LocalStats {
    pub total_problems: i32,
    pub easy_count: i32,
    pub medium_count: i32,
    pub hard_count: i32,
}

impl Default for ProblemFetcher {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RepositoryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "📊 Repository Statistics")?;
        writeln!(f, "========================")?;
        writeln!(f, "Local Problems: {}", self.local_stats.total_problems)?;
        writeln!(f, "  - Easy: {}", self.local_stats.easy_count)?;
        writeln!(f, "  - Medium: {}", self.local_stats.medium_count)?;
        writeln!(f, "  - Hard: {}", self.local_stats.hard_count)?;
        writeln!(f)?;
        writeln!(f, "Total LeetCode Problems: {}", self.api_stats.total_problems)?;
        writeln!(f, "  - Easy: {}", self.api_stats.easy_count)?;
        writeln!(f, "  - Medium: {}", self.api_stats.medium_count)?;
        writeln!(f, "  - Hard: {}", self.api_stats.hard_count)?;
        writeln!(f)?;
        writeln!(f, "Coverage: {:.2}%", self.coverage_percentage)?;
        Ok(())
    }
}