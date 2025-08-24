//! # Rust LeetCode Solutions
//!
//! A comprehensive collection of optimal LeetCode solutions implemented in Rust.
//! Each solution includes detailed explanations, complexity analysis, and comprehensive tests.
//!
//! Created by Marvin Tutt, Caia Tech

/// Easy difficulty problems (1-300 range typically)
pub mod easy;

/// Medium difficulty problems (301-600 range typically)  
pub mod medium;

/// Hard difficulty problems (601+ range typically)
pub mod hard;

/// Common utilities, data structures, and helper functions
pub mod utils;

/// LeetCode API integration for automated problem fetching
pub mod api;

/// Re-export commonly used types and traits
pub use utils::*;
pub use api::*;