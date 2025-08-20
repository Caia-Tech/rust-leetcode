use rust_leetcode::easy::two_sum::Solution;
use proptest::prelude::*;

#[test]
fn test_solution_instantiation() {
    let solution = Solution::new();
    let result = solution.two_sum(vec![2, 7, 11, 15], 9);
    assert_eq!(result, vec![0, 1]);
}

#[test] 
fn test_default_trait() {
    let solution: Solution = Default::default();
    let result = solution.two_sum(vec![3, 2, 4], 6);
    assert_eq!(result, vec![1, 2]);
}

// Property-based testing for Two Sum
proptest! {
    #[test]
    fn test_two_sum_property_based(
        nums in prop::collection::vec(-1000..1000i32, 2..100),
        idx1 in 0usize..2,
        idx2 in 0usize..2
    ) {
        // Ensure we have at least 2 elements and different indices
        if nums.len() >= 2 && idx1 < nums.len() && idx2 < nums.len() && idx1 != idx2 {
            let target = nums[idx1] + nums[idx2];
            let solution = Solution::new();
            let result = solution.two_sum(nums.clone(), target);
            
            // Verify the result is correct
            prop_assert_eq!(result.len(), 2);
            let i = result[0] as usize;
            let j = result[1] as usize;
            prop_assert!(i < nums.len());
            prop_assert!(j < nums.len()); 
            prop_assert_ne!(i, j);
            prop_assert_eq!(nums[i] + nums[j], target);
        }
    }
    
    #[test]
    fn test_approaches_consistency(
        a in -100..100i32,
        b in -100..100i32,
        c in -100..100i32,
        d in -100..100i32
    ) {
        let nums = vec![a, b, c, d];
        let target = a + c; // We know indices 0 and 2 should work
        
        let solution = Solution::new();
        let result1 = solution.two_sum_brute_force(nums.clone(), target);
        let result2 = solution.two_sum(nums.clone(), target);
        let result3 = solution.two_sum_two_pass(nums.clone(), target);
        
        // All should find valid solutions
        prop_assert_eq!(nums[result1[0] as usize] + nums[result1[1] as usize], target);
        prop_assert_eq!(nums[result2[0] as usize] + nums[result2[1] as usize], target);
        prop_assert_eq!(nums[result3[0] as usize] + nums[result3[1] as usize], target);
    }
}