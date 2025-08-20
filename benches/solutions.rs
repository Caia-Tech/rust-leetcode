use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rust_leetcode::easy::{
    two_sum::Solution as TwoSumSolution,
    valid_parentheses::Solution as ParenthesesSolution,
};
use rust_leetcode::medium::{
    longest_substring_without_repeating_characters::Solution as LongestSubstringSolution,
    longest_palindromic_substring::Solution as PalindromeSolution,
    three_sum::Solution as ThreeSumSolution,
};
use rust_leetcode::hard::{
    median_of_two_sorted_arrays::Solution as MedianSolution,
    merge_k_sorted_lists::Solution as MergeKSolution,
    trapping_rain_water::Solution as TrappingRainWaterSolution,
};
use rust_leetcode::utils::data_structures::ListNode;

fn benchmark_two_sum_approaches(c: &mut Criterion) {
    let solution = TwoSumSolution::new();
    
    // Test different input sizes
    let sizes = [10, 100, 1000];
    
    for size in sizes.iter() {
        let nums: Vec<i32> = (0..*size).collect();
        let target = size - 1;
        
        let mut group = c.benchmark_group("two_sum");
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(BenchmarkId::new("brute_force", size), size, |b, _| {
            b.iter(|| solution.two_sum_brute_force(black_box(nums.clone()), black_box(target)))
        });
        
        group.bench_with_input(BenchmarkId::new("hash_map", size), size, |b, _| {
            b.iter(|| solution.two_sum(black_box(nums.clone()), black_box(target)))
        });
        
        group.bench_with_input(BenchmarkId::new("two_pass", size), size, |b, _| {
            b.iter(|| solution.two_sum_two_pass(black_box(nums.clone()), black_box(target)))
        });
        
        group.finish();
    }
}

fn benchmark_valid_parentheses(c: &mut Criterion) {
    let solution = ParenthesesSolution::new();
    
    // Generate test strings of different sizes
    let small = "()".repeat(50);    // 100 chars
    let medium = "()".repeat(500);  // 1000 chars
    let large = "()".repeat(5000);  // 10000 chars
    
    let mut group = c.benchmark_group("valid_parentheses");
    
    group.bench_function("small_valid", |b| {
        b.iter(|| solution.is_valid(black_box(small.clone())))
    });
    
    group.bench_function("medium_valid", |b| {
        b.iter(|| solution.is_valid(black_box(medium.clone())))
    });
    
    group.bench_function("large_valid", |b| {
        b.iter(|| solution.is_valid(black_box(large.clone())))
    });
    
    group.finish();
}

fn benchmark_longest_substring(c: &mut Criterion) {
    let solution = LongestSubstringSolution::new();
    
    // Different character patterns
    let test_cases = vec![
        ("small_unique", "abcdefg".to_string()),
        ("medium_repeating", "abcabcbb".repeat(50)),
        ("large_mixed", "pwwkew".repeat(1000)),
    ];
    
    let mut group = c.benchmark_group("longest_substring");
    
    for (name, input) in test_cases {
        group.bench_function(&format!("sliding_window_{}", name), |b| {
            b.iter(|| solution.length_of_longest_substring(black_box(input.clone())))
        });
        
        group.bench_function(&format!("hash_set_{}", name), |b| {
            b.iter(|| solution.length_of_longest_substring_hashset(black_box(input.clone())))
        });
        
        group.bench_function(&format!("brute_force_{}", name), |b| {
            b.iter(|| solution.length_of_longest_substring_brute_force(black_box(input.clone())))
        });
    }
    
    group.finish();
}

fn benchmark_longest_palindrome(c: &mut Criterion) {
    let solution = PalindromeSolution::new();
    
    let test_cases = vec![
        ("short", "babad".to_string()),
        ("medium", "abcdef".repeat(50)),
        ("palindromic", "aaaaaa".repeat(100)),
    ];
    
    let mut group = c.benchmark_group("longest_palindrome");
    
    for (name, input) in test_cases {
        group.bench_function(&format!("expand_center_{}", name), |b| {
            b.iter(|| solution.longest_palindrome(black_box(input.clone())))
        });
        
        group.bench_function(&format!("manacher_{}", name), |b| {
            b.iter(|| solution.longest_palindrome_manacher(black_box(input.clone())))
        });
        
        group.bench_function(&format!("dp_{}", name), |b| {
            b.iter(|| solution.longest_palindrome_dp(black_box(input.clone())))
        });
    }
    
    group.finish();
}

fn benchmark_three_sum(c: &mut Criterion) {
    let solution = ThreeSumSolution::new();
    
    // Different input sizes
    let sizes = [50, 100, 200];
    
    for size in sizes.iter() {
        let mut nums: Vec<i32> = (-(*size as i32 / 2)..(*size as i32 / 2)).collect();
        nums.extend(vec![0; 10]); // Add some zeros for triplets
        
        let mut group = c.benchmark_group("three_sum");
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(BenchmarkId::new("two_pointer", size), size, |b, _| {
            b.iter(|| solution.three_sum(black_box(nums.clone())))
        });
        
        group.bench_with_input(BenchmarkId::new("hash_set", size), size, |b, _| {
            b.iter(|| solution.three_sum_hashset(black_box(nums.clone())))
        });
        
        group.bench_with_input(BenchmarkId::new("brute_force", size), size, |b, _| {
            b.iter(|| solution.three_sum_brute_force(black_box(nums.clone())))
        });
        
        group.finish();
    }
}

fn benchmark_median_arrays(c: &mut Criterion) {
    let solution = MedianSolution::new();
    
    let sizes = [10, 100, 1000];
    
    for size in sizes.iter() {
        let nums1: Vec<i32> = (0..*size).step_by(2).collect();
        let nums2: Vec<i32> = (1..*size).step_by(2).collect();
        
        let mut group = c.benchmark_group("median_arrays");
        group.throughput(Throughput::Elements((*size * 2) as u64));
        
        group.bench_with_input(BenchmarkId::new("binary_search", size), size, |b, _| {
            b.iter(|| solution.find_median_sorted_arrays(black_box(nums1.clone()), black_box(nums2.clone())))
        });
        
        group.bench_with_input(BenchmarkId::new("merge", size), size, |b, _| {
            b.iter(|| solution.find_median_sorted_arrays_merge(black_box(nums1.clone()), black_box(nums2.clone())))
        });
        
        group.finish();
    }
}

fn benchmark_trapping_rain_water(c: &mut Criterion) {
    let solution = TrappingRainWaterSolution::new();
    
    // Generate height arrays of different sizes and patterns
    let heights = vec![
        vec![0,1,0,2,1,0,1,3,2,1,2,1], // Classic example
        (0..100).map(|i| i % 5).collect::<Vec<i32>>(), // Medium sawtooth
        (0..1000).map(|i| (i % 10).min(10 - (i % 10))).collect::<Vec<i32>>(), // Large mountain
    ];
    
    let mut group = c.benchmark_group("trapping_rain_water");
    
    for (_i, height) in heights.iter().enumerate() {
        let name = format!("size_{}", height.len());
        
        group.bench_function(&format!("two_pointer_{}", name), |b| {
            b.iter(|| solution.trap(black_box(height.clone())))
        });
        
        group.bench_function(&format!("dp_{}", name), |b| {
            b.iter(|| solution.trap_dp(black_box(height.clone())))
        });
        
        group.bench_function(&format!("stack_{}", name), |b| {
            b.iter(|| solution.trap_stack(black_box(height.clone())))
        });
        
        group.bench_function(&format!("brute_force_{}", name), |b| {
            b.iter(|| solution.trap_brute_force(black_box(height.clone())))
        });
    }
    
    group.finish();
}

fn benchmark_merge_k_lists(c: &mut Criterion) {
    let solution = MergeKSolution::new();
    
    // Helper function to create linked list
    fn create_list(vals: Vec<i32>) -> Option<Box<ListNode>> {
        let mut head = None;
        for val in vals.into_iter().rev() {
            let mut node = Box::new(ListNode::new(val));
            node.next = head;
            head = Some(node);
        }
        head
    }
    
    // Create test cases with different numbers of lists
    let test_cases = vec![
        (5, 10),   // 5 lists, 10 nodes each
        (10, 50),  // 10 lists, 50 nodes each  
        (20, 100), // 20 lists, 100 nodes each
    ];
    
    for (num_lists, list_size) in test_cases {
        let mut lists = Vec::new();
        for i in 0..num_lists {
            let vals: Vec<i32> = (i * list_size..(i + 1) * list_size).collect();
            lists.push(create_list(vals));
        }
        
        let mut group = c.benchmark_group("merge_k_lists");
        let name = format!("{}x{}", num_lists, list_size);
        
        group.bench_function(&format!("divide_conquer_{}", name), |b| {
            b.iter(|| solution.merge_k_lists(black_box(lists.clone())))
        });
        
        group.bench_function(&format!("heap_{}", name), |b| {
            b.iter(|| solution.merge_k_lists_heap(black_box(lists.clone())))
        });
        
        group.bench_function(&format!("sequential_{}", name), |b| {
            b.iter(|| solution.merge_k_lists_sequential(black_box(lists.clone())))
        });
        
        group.finish();
    }
}

criterion_group!(
    benches,
    benchmark_two_sum_approaches,
    benchmark_valid_parentheses,
    benchmark_longest_substring,
    benchmark_longest_palindrome,
    benchmark_three_sum,
    benchmark_median_arrays,
    benchmark_trapping_rain_water,
    benchmark_merge_k_lists
);
criterion_main!(benches);