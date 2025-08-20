use rust_leetcode::easy::two_sum::Solution;

fn main() {
    println!("Two Sum Problem Examples\n");
    
    let solution = Solution::new();
    
    // Example 1: Basic case
    println!("Example 1:");
    let nums1 = vec![2, 7, 11, 15];
    let target1 = 9;
    let result1 = solution.two_sum(nums1.clone(), target1);
    println!("Input: nums = {:?}, target = {}", nums1, target1);
    println!("Output: {:?}", result1);
    println!("Explanation: nums[{}] + nums[{}] = {} + {} = {}\n", 
             result1[0], result1[1], nums1[result1[0] as usize], nums1[result1[1] as usize], target1);
    
    // Example 2: Different order
    println!("Example 2:");
    let nums2 = vec![3, 2, 4];
    let target2 = 6;
    let result2 = solution.two_sum(nums2.clone(), target2);
    println!("Input: nums = {:?}, target = {}", nums2, target2);
    println!("Output: {:?}", result2);
    println!("Explanation: nums[{}] + nums[{}] = {} + {} = {}\n", 
             result2[0], result2[1], nums2[result2[0] as usize], nums2[result2[1] as usize], target2);
    
    // Example 3: Duplicate values
    println!("Example 3:");
    let nums3 = vec![3, 3];
    let target3 = 6;
    let result3 = solution.two_sum(nums3.clone(), target3);
    println!("Input: nums = {:?}, target = {}", nums3, target3);
    println!("Output: {:?}", result3);
    println!("Explanation: nums[{}] + nums[{}] = {} + {} = {}\n", 
             result3[0], result3[1], nums3[result3[0] as usize], nums3[result3[1] as usize], target3);
    
    // Performance comparison
    println!("Performance Comparison:");
    let large_nums: Vec<i32> = (0..1000).collect();
    let large_target = 999;
    
    let start = std::time::Instant::now();
    let _ = solution.two_sum_brute_force(large_nums.clone(), large_target);
    let brute_force_time = start.elapsed();
    
    let start = std::time::Instant::now();
    let _ = solution.two_sum(large_nums.clone(), large_target);
    let hash_map_time = start.elapsed();
    
    println!("For array of size 1000:");
    println!("Brute force approach: {:?}", brute_force_time);
    println!("Hash map approach: {:?}", hash_map_time);
    println!("Speedup: {:.2}x", brute_force_time.as_nanos() as f64 / hash_map_time.as_nanos() as f64);
}