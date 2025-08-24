//! LeetCode API client implementation

use crate::api::types::*;
use std::time::{Duration, Instant};

/// LeetCode API client with rate limiting and caching
pub struct LeetCodeAPIClient {
    base_url: String,
    alpha_api_url: String,
    client: reqwest::Client,
    rate_limiter: RateLimiter,
    cache: ProblemCache,
}

impl LeetCodeAPIClient {
    /// Create a new LeetCode API client
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("rust-leetcode/1.0")
            .build()
            .expect("Failed to create HTTP client");

        Self {
            base_url: "https://leetcode.com".to_string(),
            alpha_api_url: "https://alfa-leetcode-api.onrender.com".to_string(),
            client,
            rate_limiter: RateLimiter::new(100, Duration::from_secs(60)), // 100 requests per minute
            cache: ProblemCache::new(),
        }
    }

    /// Fetch all problems using the alpha API (more reliable)
    pub async fn fetch_all_problems(&mut self) -> Result<Vec<LeetCodeProblem>, APIError> {
        if let Some(cached) = self.cache.get_all_problems() {
            return Ok(cached);
        }

        self.rate_limiter.wait_if_needed().await;

        let url = format!("{}/problems?limit=3000", self.alpha_api_url);
        let response = self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| APIError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(APIError::ServerError(response.status().as_u16()));
        }

        let problems: Vec<LeetCodeProblem> = response
            .json()
            .await
            .map_err(|e| APIError::ParseError(e.to_string()))?;

        self.cache.set_all_problems(problems.clone());
        Ok(problems)
    }

    /// Fetch problems with filters
    pub async fn fetch_problems_with_filters(
        &mut self, 
        filters: ProblemFilters,
        limit: Option<i32>
    ) -> Result<Vec<LeetCodeProblem>, APIError> {
        self.rate_limiter.wait_if_needed().await;

        let mut url = format!("{}/problems", self.alpha_api_url);
        let mut query_params = Vec::new();

        if let Some(limit) = limit {
            query_params.push(format!("limit={}", limit));
        }

        if let Some(difficulty) = filters.difficulty {
            query_params.push(format!("difficulty={}", difficulty.to_uppercase()));
        }

        if let Some(tags) = filters.tags {
            if !tags.is_empty() {
                let tags_str = tags.join("+");
                query_params.push(format!("tags={}", tags_str));
            }
        }

        if !query_params.is_empty() {
            url.push('?');
            url.push_str(&query_params.join("&"));
        }

        let response = self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| APIError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(APIError::ServerError(response.status().as_u16()));
        }

        let problems: Vec<LeetCodeProblem> = response
            .json()
            .await
            .map_err(|e| APIError::ParseError(e.to_string()))?;

        Ok(problems)
    }

    /// Fetch problems by company
    pub async fn fetch_problems_by_company(
        &mut self,
        company: Company,
        limit: Option<i32>
    ) -> Result<Vec<LeetCodeProblem>, APIError> {
        let filters = ProblemFilters {
            difficulty: None,
            tags: Some(vec![company.slug]),
            status: None,
            company_tags: None,
            list_id: None,
        };

        self.fetch_problems_with_filters(filters, limit).await
    }

    /// Fetch problems by difficulty
    pub async fn fetch_problems_by_difficulty(
        &mut self,
        difficulty: Difficulty,
        limit: Option<i32>
    ) -> Result<Vec<LeetCodeProblem>, APIError> {
        let filters = ProblemFilters {
            difficulty: Some(difficulty.to_string()),
            tags: None,
            status: None,
            company_tags: None,
            list_id: None,
        };

        self.fetch_problems_with_filters(filters, limit).await
    }

    /// Fetch problems by algorithm pattern
    pub async fn fetch_problems_by_pattern(
        &mut self,
        pattern: AlgorithmPattern,
        limit: Option<i32>
    ) -> Result<Vec<LeetCodeProblem>, APIError> {
        let filters = ProblemFilters {
            difficulty: None,
            tags: Some(pattern.tags),
            status: None,
            company_tags: None,
            list_id: None,
        };

        self.fetch_problems_with_filters(filters, limit).await
    }

    /// Get problem statistics
    pub async fn get_problem_stats(&mut self) -> Result<ProblemStats, APIError> {
        if let Some(cached) = self.cache.get_stats() {
            return Ok(cached);
        }

        let problems = self.fetch_all_problems().await?;
        
        let total_problems = problems.len() as i32;
        let easy_count = problems.iter().filter(|p| p.difficulty == "Easy").count() as i32;
        let medium_count = problems.iter().filter(|p| p.difficulty == "Medium").count() as i32;
        let hard_count = problems.iter().filter(|p| p.difficulty == "Hard").count() as i32;
        
        let total_acceptance: f64 = problems
            .iter()
            .filter_map(|p| p.acceptance_rate)
            .sum();
        let acceptance_rate = total_acceptance / problems.len() as f64;

        let stats = ProblemStats {
            total_problems,
            easy_count,
            medium_count,
            hard_count,
            solved_count: 0, // Would need user data for this
            acceptance_rate,
        };

        self.cache.set_stats(stats.clone());
        Ok(stats)
    }

    /// Get top problems for interview preparation
    pub async fn get_top_interview_problems(
        &mut self,
        count: i32
    ) -> Result<Vec<LeetCodeProblem>, APIError> {
        // Fetch high-frequency problems from major companies
        let mut top_problems = Vec::new();

        // Get problems from top companies
        let companies = vec![
            Company::amazon(),
            Company::google(),
            Company::microsoft(),
            Company::facebook(),
            Company::apple(),
        ];

        for company in companies {
            let company_problems = self.fetch_problems_by_company(company, Some(count / 5)).await?;
            top_problems.extend(company_problems);
        }

        // Remove duplicates and sort by acceptance rate
        top_problems.sort_by(|a, b| {
            b.acceptance_rate.unwrap_or(0.0)
                .partial_cmp(&a.acceptance_rate.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        top_problems.dedup_by(|a, b| a.frontend_id == b.frontend_id);
        top_problems.truncate(count as usize);

        Ok(top_problems)
    }

    /// Search problems by title
    pub async fn search_problems(&mut self, query: &str) -> Result<Vec<LeetCodeProblem>, APIError> {
        let all_problems = self.fetch_all_problems().await?;
        
        let query_lower = query.to_lowercase();
        let matching_problems: Vec<LeetCodeProblem> = all_problems
            .into_iter()
            .filter(|p| {
                p.title.to_lowercase().contains(&query_lower) ||
                p.title_slug.contains(&query_lower)
            })
            .collect();

        Ok(matching_problems)
    }
}

/// Rate limiter to prevent API abuse
struct RateLimiter {
    max_requests: u32,
    window: Duration,
    requests: Vec<Instant>,
}

impl RateLimiter {
    fn new(max_requests: u32, window: Duration) -> Self {
        Self {
            max_requests,
            window,
            requests: Vec::new(),
        }
    }

    async fn wait_if_needed(&mut self) {
        let now = Instant::now();
        
        // Remove old requests outside the window
        self.requests.retain(|&time| now.duration_since(time) < self.window);
        
        // If we've hit the limit, wait
        if self.requests.len() >= self.max_requests as usize {
            if let Some(&oldest) = self.requests.first() {
                let wait_time = self.window - now.duration_since(oldest);
                tokio::time::sleep(wait_time).await;
            }
        }
        
        self.requests.push(now);
    }
}

/// Cache for API responses to reduce network calls
struct ProblemCache {
    all_problems: Option<(Vec<LeetCodeProblem>, Instant)>,
    stats: Option<(ProblemStats, Instant)>,
    cache_duration: Duration,
}

impl ProblemCache {
    fn new() -> Self {
        Self {
            all_problems: None,
            stats: None,
            cache_duration: Duration::from_secs(3600), // 1 hour cache
        }
    }

    fn get_all_problems(&self) -> Option<Vec<LeetCodeProblem>> {
        if let Some((problems, timestamp)) = &self.all_problems {
            if timestamp.elapsed() < self.cache_duration {
                return Some(problems.clone());
            }
        }
        None
    }

    fn set_all_problems(&mut self, problems: Vec<LeetCodeProblem>) {
        self.all_problems = Some((problems, Instant::now()));
    }

    fn get_stats(&self) -> Option<ProblemStats> {
        if let Some((stats, timestamp)) = &self.stats {
            if timestamp.elapsed() < self.cache_duration {
                return Some(stats.clone());
            }
        }
        None
    }

    fn set_stats(&mut self, stats: ProblemStats) {
        self.stats = Some((stats, Instant::now()));
    }
}

impl Default for LeetCodeAPIClient {
    fn default() -> Self {
        Self::new()
    }
}