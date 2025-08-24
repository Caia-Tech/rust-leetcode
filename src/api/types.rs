//! Types for LeetCode API integration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeetCodeProblem {
    #[serde(rename = "questionId")]
    pub question_id: String,
    
    #[serde(rename = "questionFrontendId")]
    pub frontend_id: String,
    
    pub title: String,
    
    #[serde(rename = "titleSlug")]
    pub title_slug: String,
    
    pub difficulty: String,
    
    #[serde(rename = "isPaidOnly")]
    pub is_paid_only: bool,
    
    #[serde(rename = "acRate")]
    pub acceptance_rate: Option<f64>,
    
    #[serde(rename = "topicTags")]
    pub topic_tags: Vec<TopicTag>,
    
    pub status: Option<String>,
    
    #[serde(rename = "hasSolution")]
    pub has_solution: Option<bool>,
    
    #[serde(rename = "hasVideoSolution")]
    pub has_video_solution: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicTag {
    pub id: String,
    pub name: String,
    pub slug: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemList {
    #[serde(rename = "totalNum")]
    pub total_num: i32,
    
    pub data: Vec<LeetCodeProblem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemListResponse {
    pub data: ProblemListData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemListData {
    #[serde(rename = "problemsetQuestionList")]
    pub problemset_question_list: ProblemList,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemFilters {
    pub difficulty: Option<String>,
    pub tags: Option<Vec<String>>,
    pub status: Option<String>,
    pub company_tags: Option<Vec<String>>,
    pub list_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLQuery {
    pub query: String,
    pub variables: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemDetail {
    pub question_id: String,
    pub title: String,
    pub content: String,
    pub code_snippets: Vec<CodeSnippet>,
    pub sample_test_case: String,
    pub constraints: String,
    pub hints: Vec<String>,
    pub similar_questions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSnippet {
    pub lang: String,
    pub lang_slug: String,
    pub code: String,
}

#[derive(Debug)]
pub enum APIError {
    NetworkError(String),
    ParseError(String),
    RateLimitExceeded,
    Unauthorized,
    NotFound,
    ServerError(u16),
    Other(String),
}

impl From<Box<dyn std::error::Error>> for APIError {
    fn from(error: Box<dyn std::error::Error>) -> Self {
        APIError::Other(error.to_string())
    }
}

impl std::fmt::Display for APIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            APIError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            APIError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            APIError::RateLimitExceeded => write!(f, "Rate limit exceeded"),
            APIError::Unauthorized => write!(f, "Unauthorized access"),
            APIError::NotFound => write!(f, "Resource not found"),
            APIError::ServerError(code) => write!(f, "Server error: {}", code),
            APIError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for APIError {}

#[derive(Debug, Clone)]
pub struct Company {
    pub name: String,
    pub slug: String,
    pub tag_id: String,
}

impl Company {
    pub fn amazon() -> Self {
        Self {
            name: "Amazon".to_string(),
            slug: "amazon".to_string(),
            tag_id: "1".to_string(),
        }
    }
    
    pub fn google() -> Self {
        Self {
            name: "Google".to_string(),
            slug: "google".to_string(),
            tag_id: "2".to_string(),
        }
    }
    
    pub fn microsoft() -> Self {
        Self {
            name: "Microsoft".to_string(),
            slug: "microsoft".to_string(),
            tag_id: "3".to_string(),
        }
    }
    
    pub fn facebook() -> Self {
        Self {
            name: "Meta".to_string(),
            slug: "facebook".to_string(),
            tag_id: "4".to_string(),
        }
    }
    
    pub fn apple() -> Self {
        Self {
            name: "Apple".to_string(),
            slug: "apple".to_string(),
            tag_id: "5".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Difficulty {
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

#[derive(Debug, Clone)]
pub struct AlgorithmPattern {
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
}

impl AlgorithmPattern {
    pub fn two_pointers() -> Self {
        Self {
            name: "Two Pointers".to_string(),
            description: "Use two pointers to traverse data structure".to_string(),
            tags: vec!["two-pointers".to_string(), "array".to_string()],
        }
    }
    
    pub fn sliding_window() -> Self {
        Self {
            name: "Sliding Window".to_string(),
            description: "Maintain a window of elements for optimization".to_string(),
            tags: vec!["sliding-window".to_string(), "array".to_string(), "string".to_string()],
        }
    }
    
    pub fn fast_slow_pointers() -> Self {
        Self {
            name: "Fast & Slow Pointers".to_string(),
            description: "Use pointers moving at different speeds".to_string(),
            tags: vec!["linked-list".to_string(), "two-pointers".to_string()],
        }
    }
    
    pub fn merge_intervals() -> Self {
        Self {
            name: "Merge Intervals".to_string(),
            description: "Deal with overlapping intervals".to_string(),
            tags: vec!["intervals".to_string(), "array".to_string(), "sorting".to_string()],
        }
    }
    
    pub fn cyclic_sort() -> Self {
        Self {
            name: "Cyclic Sort".to_string(),
            description: "Sort array with numbers in given range".to_string(),
            tags: vec!["sorting".to_string(), "array".to_string()],
        }
    }
    
    pub fn tree_traversal() -> Self {
        Self {
            name: "Tree Traversal".to_string(),
            description: "Traverse binary trees using various methods".to_string(),
            tags: vec!["tree".to_string(), "binary-tree".to_string(), "dfs".to_string(), "bfs".to_string()],
        }
    }
    
    pub fn graph_traversal() -> Self {
        Self {
            name: "Graph Traversal".to_string(),
            description: "Traverse graphs using DFS/BFS".to_string(),
            tags: vec!["graph".to_string(), "dfs".to_string(), "bfs".to_string()],
        }
    }
    
    pub fn dynamic_programming() -> Self {
        Self {
            name: "Dynamic Programming".to_string(),
            description: "Solve problems using optimal substructure".to_string(),
            tags: vec!["dynamic-programming".to_string(), "memoization".to_string()],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemStats {
    pub total_problems: i32,
    pub easy_count: i32,
    pub medium_count: i32,
    pub hard_count: i32,
    pub solved_count: i32,
    pub acceptance_rate: f64,
}