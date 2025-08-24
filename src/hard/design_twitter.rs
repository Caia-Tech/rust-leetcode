//! # Problem 355: Design Twitter
//!
//! **Difficulty**: Hard
//! **Topics**: Hash Table, Linked List, Design, Heap (Priority Queue)
//! **Acceptance Rate**: 35.8%

use std::collections::{HashMap, HashSet, BinaryHeap};

/// Twitter design supporting post tweets, follow/unfollow users, and get news feed
pub struct Twitter {
    tweets: Vec<Tweet>,
    following: HashMap<i32, HashSet<i32>>,
    tweet_counter: i32,
}

#[derive(Debug, Clone)]
struct Tweet {
    id: i32,
    user_id: i32,
    timestamp: i32,
}

impl Twitter {
    /// Create a new Twitter instance
    pub fn new() -> Self {
        Twitter {
            tweets: Vec::new(),
            following: HashMap::new(),
            tweet_counter: 0,
        }
    }

    /// Post a new tweet
    /// 
    /// Time Complexity: O(1) - constant time insertion
    /// Space Complexity: O(1) - per tweet
    pub fn post_tweet(&mut self, user_id: i32, tweet_id: i32) {
        self.tweets.push(Tweet {
            id: tweet_id,
            user_id,
            timestamp: self.tweet_counter,
        });
        self.tweet_counter += 1;
    }

    /// Get the 10 most recent tweets in the user's news feed
    /// 
    /// Time Complexity: O(n log k) where n is total tweets, k=10
    /// Space Complexity: O(k) - for the result
    pub fn get_news_feed(&self, user_id: i32) -> Vec<i32> {
        let mut relevant_users = HashSet::new();
        relevant_users.insert(user_id); // User sees their own tweets
        
        // Add all followed users
        if let Some(followed) = self.following.get(&user_id) {
            relevant_users.extend(followed);
        }
        
        // Collect tweets from relevant users
        let mut relevant_tweets: Vec<&Tweet> = self.tweets
            .iter()
            .filter(|tweet| relevant_users.contains(&tweet.user_id))
            .collect();
        
        // Sort by timestamp (most recent first) and take top 10
        relevant_tweets.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        relevant_tweets
            .iter()
            .take(10)
            .map(|tweet| tweet.id)
            .collect()
    }

    /// Follow a user
    /// 
    /// Time Complexity: O(1) - hash set insertion
    /// Space Complexity: O(1) - per follow relationship
    pub fn follow(&mut self, follower_id: i32, followee_id: i32) {
        if follower_id != followee_id {
            self.following
                .entry(follower_id)
                .or_insert_with(HashSet::new)
                .insert(followee_id);
        }
    }

    /// Unfollow a user
    /// 
    /// Time Complexity: O(1) - hash set removal
    /// Space Complexity: O(1) - constant space
    pub fn unfollow(&mut self, follower_id: i32, followee_id: i32) {
        if let Some(followed) = self.following.get_mut(&follower_id) {
            followed.remove(&followee_id);
        }
    }
}

/// Alternative implementation using priority queue for better performance
pub struct TwitterOptimized {
    tweets: HashMap<i32, Vec<Tweet>>, // tweets by user
    following: HashMap<i32, HashSet<i32>>,
    global_time: i32,
}

impl TwitterOptimized {
    /// Create a new optimized Twitter instance
    pub fn new() -> Self {
        TwitterOptimized {
            tweets: HashMap::new(),
            following: HashMap::new(),
            global_time: 0,
        }
    }

    /// Post a new tweet
    pub fn post_tweet(&mut self, user_id: i32, tweet_id: i32) {
        let tweet = Tweet {
            id: tweet_id,
            user_id,
            timestamp: self.global_time,
        };
        
        self.tweets
            .entry(user_id)
            .or_insert_with(Vec::new)
            .push(tweet);
        
        self.global_time += 1;
    }

    /// Get news feed using merge approach
    /// 
    /// Time Complexity: O(k log u) where k=10, u is number of followed users
    /// Space Complexity: O(u) - for the heap
    pub fn get_news_feed(&self, user_id: i32) -> Vec<i32> {
        let mut heap = BinaryHeap::new();
        
        // Add user's own tweets
        if let Some(user_tweets) = self.tweets.get(&user_id) {
            if let Some(latest_tweet) = user_tweets.last() {
                heap.push((latest_tweet.timestamp, user_id, user_tweets.len() - 1));
            }
        }
        
        // Add followed users' tweets
        if let Some(followed_users) = self.following.get(&user_id) {
            for &followee_id in followed_users {
                if let Some(followee_tweets) = self.tweets.get(&followee_id) {
                    if let Some(latest_tweet) = followee_tweets.last() {
                        heap.push((latest_tweet.timestamp, followee_id, followee_tweets.len() - 1));
                    }
                }
            }
        }
        
        let mut result = Vec::new();
        
        // Extract top 10 tweets
        while result.len() < 10 && !heap.is_empty() {
            if let Some((_, user_id_val, tweet_index)) = heap.pop() {
                if let Some(user_tweets) = self.tweets.get(&user_id_val) {
                    result.push(user_tweets[tweet_index].id);
                    
                    // Add next tweet from same user if exists
                    if tweet_index > 0 {
                        let next_tweet = &user_tweets[tweet_index - 1];
                        heap.push((next_tweet.timestamp, user_id_val, tweet_index - 1));
                    }
                }
            }
        }
        
        result
    }

    /// Follow a user
    pub fn follow(&mut self, follower_id: i32, followee_id: i32) {
        if follower_id != followee_id {
            self.following
                .entry(follower_id)
                .or_insert_with(HashSet::new)
                .insert(followee_id);
        }
    }

    /// Unfollow a user
    pub fn unfollow(&mut self, follower_id: i32, followee_id: i32) {
        if let Some(followed) = self.following.get_mut(&follower_id) {
            followed.remove(&followee_id);
        }
    }
}

impl Default for Twitter {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TwitterOptimized {
    fn default() -> Self {
        Self::new()
    }
}

/// Solution struct for LeetCode interface compatibility
pub struct Solution;

impl Solution {
    pub fn new() -> Self {
        Solution
    }
}

impl Default for Solution {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        let mut twitter = Twitter::new();
        
        // User 1 posts tweet 5
        twitter.post_tweet(1, 5);
        
        // User 1's news feed should contain tweet 5
        assert_eq!(twitter.get_news_feed(1), vec![5]);
        
        // User 1 follows user 2
        twitter.follow(1, 2);
        
        // User 2 posts tweet 6
        twitter.post_tweet(2, 6);
        
        // User 1's news feed should contain tweets from user 2
        let feed = twitter.get_news_feed(1);
        assert!(feed.contains(&5) && feed.contains(&6));
        
        // Most recent tweet should be first
        assert_eq!(feed[0], 6); // User 2's tweet is more recent
        assert_eq!(feed[1], 5); // User 1's tweet
    }

    #[test]
    fn test_unfollow() {
        let mut twitter = Twitter::new();
        
        twitter.post_tweet(1, 5);
        twitter.follow(1, 2);
        twitter.post_tweet(2, 6);
        
        // Before unfollow
        let feed = twitter.get_news_feed(1);
        assert!(feed.contains(&6));
        
        // Unfollow user 2
        twitter.unfollow(1, 2);
        
        // After unfollow, should not see user 2's tweets
        let feed = twitter.get_news_feed(1);
        assert!(!feed.contains(&6));
        assert!(feed.contains(&5)); // Still see own tweets
    }

    #[test]
    fn test_news_feed_limit() {
        let mut twitter = Twitter::new();
        
        // Post more than 10 tweets
        for i in 1..=15 {
            twitter.post_tweet(1, i);
        }
        
        let feed = twitter.get_news_feed(1);
        
        // Should return at most 10 tweets
        assert_eq!(feed.len(), 10);
        
        // Should be the 10 most recent (6-15)
        for i in 6..=15 {
            assert!(feed.contains(&i));
        }
        
        // Should not contain old tweets (1-5)
        for i in 1..=5 {
            assert!(!feed.contains(&i));
        }
    }

    #[test]
    fn test_self_follow() {
        let mut twitter = Twitter::new();
        
        twitter.post_tweet(1, 5);
        twitter.follow(1, 1); // Try to follow self
        
        let feed = twitter.get_news_feed(1);
        // Should still work normally (not create duplicate entries)
        assert_eq!(feed, vec![5]);
    }

    #[test]
    fn test_multiple_users() {
        let mut twitter = Twitter::new();
        
        // User 1 follows users 2 and 3
        twitter.follow(1, 2);
        twitter.follow(1, 3);
        
        // Users post tweets in different order
        twitter.post_tweet(2, 10);
        twitter.post_tweet(1, 11);
        twitter.post_tweet(3, 12);
        twitter.post_tweet(2, 13);
        
        let feed = twitter.get_news_feed(1);
        
        // Should see all tweets in chronological order (most recent first)
        assert_eq!(feed, vec![13, 12, 11, 10]);
    }

    #[test]
    fn test_optimized_implementation() {
        let mut twitter_basic = Twitter::new();
        let mut twitter_opt = TwitterOptimized::new();
        
        // Test same operations on both implementations
        let operations = vec![
            (1, 10), (2, 20), (1, 30), (3, 40), (2, 50)
        ];
        
        for (user, tweet) in operations {
            twitter_basic.post_tweet(user, tweet);
            twitter_opt.post_tweet(user, tweet);
        }
        
        twitter_basic.follow(1, 2);
        twitter_basic.follow(1, 3);
        twitter_opt.follow(1, 2);
        twitter_opt.follow(1, 3);
        
        // Both should produce same result
        let feed_basic = twitter_basic.get_news_feed(1);
        let feed_opt = twitter_opt.get_news_feed(1);
        
        assert_eq!(feed_basic, feed_opt);
    }

    #[test]
    fn test_empty_feed() {
        let twitter = Twitter::new();
        
        // User with no tweets and not following anyone
        assert_eq!(twitter.get_news_feed(1), Vec::<i32>::new());
    }
}

#[cfg(test)]
mod benchmarks {
    // Note: Benchmarks will be added to benches/ directory for Criterion integration
}