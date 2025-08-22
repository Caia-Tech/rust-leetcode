//! # Problem 283: Move Zeroes
//!
//! Given an integer array `nums`, move all zeros to the end while keeping
//! the relative order of the non-zero elements. This must be done in-place
//! with minimal operations.
//!
//! ## Examples
//!
//! ```
//! use rust_leetcode::easy::move_zeroes::Solution;
//!
//! let mut nums = vec![0, 1, 0, 3, 12];
//! Solution::new().move_zeroes_two_pointers(&mut nums);
//! assert_eq!(nums, vec![1, 3, 12, 0, 0]);
//!
//! let mut nums2 = vec![0, 1, 0, 3, 12];
//! Solution::new().move_zeroes_retain(&mut nums2);
//! assert_eq!(nums2, vec![1, 3, 12, 0, 0]);
//! ```
//!
//! ## Constraints
//!
//! - `1 <= nums.len() <= 10^4`
//! - `-2^{31} <= nums[i] <= 2^{31} - 1`
//!
/// Solution struct following LeetCode conventions
#[derive(Default)]
pub struct Solution;

impl Solution {
    /// Create a new instance of `Solution`
    pub fn new() -> Self {
        Solution
    }

    /// # Approach 1: Two-pointer swap
    ///
    /// Iterate through the array, maintaining the position of the next
    /// non-zero element. Swap non-zero elements forward as they are found.
    ///
    /// **Time Complexity:** `O(n)`
    /// **Space Complexity:** `O(1)`
    pub fn move_zeroes_two_pointers(&self, nums: &mut [i32]) {
        let mut insert = 0;
        for i in 0..nums.len() {
            if nums[i] != 0 {
                nums.swap(insert, i);
                insert += 1;
            }
        }
    }

    /// # Approach 2: Retain non-zero elements then pad
    ///
    /// Retain all non-zero values, count how many were removed, then append
    /// that many zeros. Requires extra allocation but showcases a concise
    /// alternative using standard library utilities.
    ///
    /// **Time Complexity:** `O(n)`
    /// **Space Complexity:** `O(n)`
    pub fn move_zeroes_retain(&self, nums: &mut Vec<i32>) {
        let original_len = nums.len();
        nums.retain(|&x| x != 0);
        let zeros = original_len - nums.len();
        nums.extend(std::iter::repeat(0).take(zeros));
    }
}

#[cfg(test)]
mod tests {
    use super::Solution;
    use rstest::rstest;

    fn setup() -> Solution {
        Solution::new()
    }

    #[rstest]
    #[case(vec![0,1,0,3,12], vec![1,3,12,0,0])]
    #[case(vec![0], vec![0])]
    #[case(vec![2,1], vec![2,1])]
    fn test_move_zeroes_two_pointers(#[case] mut input: Vec<i32>, #[case] expected: Vec<i32>) {
        let solution = setup();
        solution.move_zeroes_two_pointers(&mut input);
        assert_eq!(input, expected);
    }

    #[rstest]
    #[case(vec![0,1,0,3,12], vec![1,3,12,0,0])]
    #[case(vec![0], vec![0])]
    #[case(vec![2,1], vec![2,1])]
    fn test_move_zeroes_retain(#[case] mut input: Vec<i32>, #[case] expected: Vec<i32>) {
        let solution = setup();
        solution.move_zeroes_retain(&mut input);
        assert_eq!(input, expected);
    }

    #[test]
    fn test_consistency_between_methods() {
        let solution = setup();
        let cases = vec![
            vec![0,1,0,3,12],
            vec![1,0,0,2,3],
            vec![0,0,1],
            vec![1,2,3],
        ];
        for mut nums in cases {
            let mut nums2 = nums.clone();
            solution.move_zeroes_two_pointers(&mut nums);
            solution.move_zeroes_retain(&mut nums2);
            assert_eq!(nums, nums2);
        }
    }
}

