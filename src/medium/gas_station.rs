//! Problem 134: Gas Station
//! 
//! There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i].
//! You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station 
//! to its next (i + 1)th station. You begin the journey with an empty tank at one of the gas stations.
//! 
//! Given two integer arrays gas and cost, return the starting gas station's index if you can travel 
//! around the circuit once in the clockwise direction, otherwise return -1.
//! 
//! If there exists a solution, it is guaranteed to be unique.
//! 
//! Example 1:
//! Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
//! Output: 3
//! Explanation:
//! Start at station 3 (index 3) and fill up with 4 units of gas. Your tank = 0 + 4 = 4
//! Travel to station 4. Your tank = 4 - 1 + 5 = 8
//! Travel to station 0. Your tank = 8 - 2 + 1 = 7
//! Travel to station 1. Your tank = 7 - 3 + 2 = 6
//! Travel to station 2. Your tank = 6 - 4 + 3 = 5
//! Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
//! Therefore, return 3 as the starting index.
//! 
//! Example 2:
//! Input: gas = [2,3,4], cost = [3,4,3]
//! Output: -1
//! Explanation:
//! You can't start at station 0 or 1, as there is not enough gas to travel to the next station.
//! Let's start at station 2 and fill up with 4 units of gas. Your tank = 0 + 4 = 4
//! Travel to station 0. Your tank = 4 - 3 + 2 = 3
//! Travel to station 1. Your tank = 3 - 3 + 3 = 3
//! Travel to station 2. Your tank = 3 - 4 + 4 = 3
//! You cannot travel back to station 2, as it requires 4 units of gas but you only have 3.
//! Therefore, you can't travel around the circuit once no matter where you start.

pub struct Solution;

impl Solution {
    /// Approach 1: Greedy Single Pass
    /// 
    /// Key insight: If the total gas >= total cost, there must be a solution.
    /// If we fail to reach station j from station i, then any station between i and j
    /// cannot be the starting point either. So we can start checking from j+1.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn can_complete_circuit_greedy(&self, gas: Vec<i32>, cost: Vec<i32>) -> i32 {
        let n = gas.len();
        let mut total_tank = 0;
        let mut current_tank = 0;
        let mut start = 0;
        
        for i in 0..n {
            let net_gas = gas[i] - cost[i];
            total_tank += net_gas;
            current_tank += net_gas;
            
            // If current tank becomes negative, we can't reach next station
            // from current starting point. Try starting from next station.
            if current_tank < 0 {
                start = i + 1;
                current_tank = 0;
            }
        }
        
        // If total tank is non-negative, there exists a solution
        if total_tank >= 0 { start as i32 } else { -1 }
    }
    
    /// Approach 2: Brute Force Simulation
    /// 
    /// Try starting from each station and simulate the entire journey.
    /// This is straightforward but less efficient.
    /// 
    /// Time Complexity: O(n²)
    /// Space Complexity: O(1)
    pub fn can_complete_circuit_brute_force(&self, gas: Vec<i32>, cost: Vec<i32>) -> i32 {
        let n = gas.len();
        
        for start in 0..n {
            let mut tank = 0;
            let mut can_complete = true;
            
            for i in 0..n {
                let station = (start + i) % n;
                tank += gas[station] - cost[station];
                
                if tank < 0 {
                    can_complete = false;
                    break;
                }
            }
            
            if can_complete {
                return start as i32;
            }
        }
        
        -1
    }
    
    /// Approach 3: Two Pointers Technique
    /// 
    /// Uses two pointers to expand the current window of stations we can visit.
    /// If we can't proceed forward, we try adding stations from behind.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn can_complete_circuit_two_pointers(&self, gas: Vec<i32>, cost: Vec<i32>) -> i32 {
        let n = gas.len();
        let mut tank = 0;
        let mut start = 0;
        let mut end = 0;
        let mut stations_covered = 0;
        
        while stations_covered < n {
            // Try to extend forward
            while tank >= 0 && stations_covered < n {
                tank += gas[end] - cost[end];
                end = (end + 1) % n;
                stations_covered += 1;
            }
            
            // If we covered all stations and tank is non-negative, we found solution
            if stations_covered == n && tank >= 0 {
                return start as i32;
            }
            
            // If tank is negative, try to add stations from behind
            while tank < 0 && start != end {
                start = (start + n - 1) % n;
                tank += gas[start] - cost[start];
                stations_covered += 1;
            }
            
            // If we can't make progress, no solution exists
            if tank < 0 {
                return -1;
            }
        }
        
        if tank >= 0 { start as i32 } else { -1 }
    }
    
    /// Approach 4: Prefix Sum Analysis
    /// 
    /// Analyzes the problem using prefix sums of net gas (gas[i] - cost[i]).
    /// This approach is equivalent to the greedy approach but uses different reasoning.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn can_complete_circuit_prefix_sum(&self, gas: Vec<i32>, cost: Vec<i32>) -> i32 {
        let n = gas.len();
        let mut total_gas = 0;
        let mut current_gas = 0;
        let mut start = 0;
        
        for i in 0..n {
            let net_gas = gas[i] - cost[i];
            total_gas += net_gas;
            current_gas += net_gas;
            
            // If current gas becomes negative, reset starting point
            if current_gas < 0 {
                start = i + 1;
                current_gas = 0;
            }
        }
        
        // If total gas is non-negative, return starting point
        if total_gas >= 0 { start as i32 } else { -1 }
    }
    
    /// Approach 5: Segment Analysis
    /// 
    /// Divides the circular array into segments where each segment
    /// has a net positive gas contribution, then finds the optimal starting point.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn can_complete_circuit_segment(&self, gas: Vec<i32>, cost: Vec<i32>) -> i32 {
        let n = gas.len();
        let mut total_gas = 0;
        let mut current_gas = 0;
        let mut start = 0;
        let mut deficit = 0;
        
        for i in 0..n {
            let net = gas[i] - cost[i];
            total_gas += net;
            current_gas += net;
            
            if current_gas < 0 {
                // Record the deficit and reset for next segment
                deficit += current_gas;
                current_gas = 0;
                start = i + 1;
            }
        }
        
        // Check if we have enough surplus to cover the deficit
        if current_gas + deficit >= 0 && total_gas >= 0 {
            start as i32
        } else {
            -1
        }
    }
    
    /// Approach 6: Mathematical Optimization
    /// 
    /// Uses mathematical properties to determine the starting point more directly.
    /// Based on the fact that if a solution exists, there's exactly one valid starting point.
    /// 
    /// Time Complexity: O(n)
    /// Space Complexity: O(1)
    pub fn can_complete_circuit_math(&self, gas: Vec<i32>, cost: Vec<i32>) -> i32 {
        let n = gas.len();
        
        // First check if solution is possible
        let total_gas: i32 = gas.iter().sum();
        let total_cost: i32 = cost.iter().sum();
        
        if total_gas < total_cost {
            return -1;
        }
        
        // Find the starting point using cumulative balance
        let mut balance = 0;
        let mut start = 0;
        
        for i in 0..n {
            balance += gas[i] - cost[i];
            
            // If balance becomes negative, we cannot start from any previous station
            // The next station becomes our new candidate starting point
            if balance < 0 {
                balance = 0;
                start = i + 1;
            }
        }
        
        // If we reach here, start is the answer (guaranteed since total_gas >= total_cost)
        start as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_greedy() {
        let solution = Solution;
        
        assert_eq!(solution.can_complete_circuit_greedy(vec![1,2,3,4,5], vec![3,4,5,1,2]), 3);
        assert_eq!(solution.can_complete_circuit_greedy(vec![2,3,4], vec![3,4,3]), -1);
        assert_eq!(solution.can_complete_circuit_greedy(vec![5], vec![4]), 0);
        assert_eq!(solution.can_complete_circuit_greedy(vec![3], vec![4]), -1);
        assert_eq!(solution.can_complete_circuit_greedy(vec![1,2], vec![2,1]), 1);
    }
    
    #[test]
    fn test_brute_force() {
        let solution = Solution;
        
        assert_eq!(solution.can_complete_circuit_brute_force(vec![1,2,3,4,5], vec![3,4,5,1,2]), 3);
        assert_eq!(solution.can_complete_circuit_brute_force(vec![2,3,4], vec![3,4,3]), -1);
        assert_eq!(solution.can_complete_circuit_brute_force(vec![5], vec![4]), 0);
        assert_eq!(solution.can_complete_circuit_brute_force(vec![3], vec![4]), -1);
        assert_eq!(solution.can_complete_circuit_brute_force(vec![1,2], vec![2,1]), 1);
    }
    
    #[test]
    fn test_two_pointers() {
        let solution = Solution;
        
        assert_eq!(solution.can_complete_circuit_two_pointers(vec![1,2,3,4,5], vec![3,4,5,1,2]), 3);
        assert_eq!(solution.can_complete_circuit_two_pointers(vec![2,3,4], vec![3,4,3]), -1);
        assert_eq!(solution.can_complete_circuit_two_pointers(vec![5], vec![4]), 0);
        assert_eq!(solution.can_complete_circuit_two_pointers(vec![3], vec![4]), -1);
        assert_eq!(solution.can_complete_circuit_two_pointers(vec![1,2], vec![2,1]), 1);
    }
    
    #[test]
    fn test_prefix_sum() {
        let solution = Solution;
        
        assert_eq!(solution.can_complete_circuit_prefix_sum(vec![1,2,3,4,5], vec![3,4,5,1,2]), 3);
        assert_eq!(solution.can_complete_circuit_prefix_sum(vec![2,3,4], vec![3,4,3]), -1);
        assert_eq!(solution.can_complete_circuit_prefix_sum(vec![5], vec![4]), 0);
        assert_eq!(solution.can_complete_circuit_prefix_sum(vec![3], vec![4]), -1);
        assert_eq!(solution.can_complete_circuit_prefix_sum(vec![1,2], vec![2,1]), 1);
    }
    
    #[test]
    fn test_segment() {
        let solution = Solution;
        
        assert_eq!(solution.can_complete_circuit_segment(vec![1,2,3,4,5], vec![3,4,5,1,2]), 3);
        assert_eq!(solution.can_complete_circuit_segment(vec![2,3,4], vec![3,4,3]), -1);
        assert_eq!(solution.can_complete_circuit_segment(vec![5], vec![4]), 0);
        assert_eq!(solution.can_complete_circuit_segment(vec![3], vec![4]), -1);
        assert_eq!(solution.can_complete_circuit_segment(vec![1,2], vec![2,1]), 1);
    }
    
    #[test]
    fn test_math() {
        let solution = Solution;
        
        assert_eq!(solution.can_complete_circuit_math(vec![1,2,3,4,5], vec![3,4,5,1,2]), 3);
        assert_eq!(solution.can_complete_circuit_math(vec![2,3,4], vec![3,4,3]), -1);
        assert_eq!(solution.can_complete_circuit_math(vec![5], vec![4]), 0);
        assert_eq!(solution.can_complete_circuit_math(vec![3], vec![4]), -1);
        assert_eq!(solution.can_complete_circuit_math(vec![1,2], vec![2,1]), 1);
    }
    
    #[test]
    fn test_edge_cases() {
        let solution = Solution;
        
        // Single station with exact gas
        assert_eq!(solution.can_complete_circuit_greedy(vec![1], vec![1]), 0);
        
        // Multiple stations with same net gas
        assert_eq!(solution.can_complete_circuit_greedy(vec![2,2,2], vec![1,1,1]), 0);
        
        // Decreasing gas, increasing cost
        assert_eq!(solution.can_complete_circuit_greedy(vec![5,4,3,2,1], vec![1,2,3,4,5]), 0);
        
        // All stations have negative net gas except one
        assert_eq!(solution.can_complete_circuit_greedy(vec![1,1,1,10], vec![2,2,2,1]), 3);
        
        // Large numbers
        assert_eq!(solution.can_complete_circuit_greedy(vec![1000,1000], vec![999,1001]), 0);
    }
    
    #[test]
    fn test_complex_scenarios() {
        let solution = Solution;
        
        // Multiple possible starting points, but only one works
        let gas = vec![2,4,3,1,5,6,1,2];
        let cost = vec![3,3,4,2,4,5,2,3];
        let result = solution.can_complete_circuit_greedy(gas.clone(), cost.clone());
        
        // Verify that the result is valid (if not -1)
        if result != -1 {
            let start = result as usize;
            let mut tank = 0;
            let n = gas.len();
            
            for i in 0..n {
                let station = (start + i) % n;
                tank += gas[station] - cost[station];
                assert!(tank >= 0, "Invalid solution: tank becomes negative at station {}", station);
            }
        }
        
        // Tight constraints where total gas exactly equals total cost
        let tight_result = solution.can_complete_circuit_greedy(vec![3,1,1], vec![1,2,2]);
        assert_ne!(tight_result, -1); // Should have a valid solution
        
        // Alternating high/low values
        let alternating_result = solution.can_complete_circuit_greedy(vec![1,5,1,5], vec![2,3,4,2]);
        assert_ne!(alternating_result, -1); // Should have a valid solution
    }
    
    #[test]
    fn test_consistency_across_approaches() {
        let solution = Solution;
        
        let test_cases = vec![
            (vec![1,2,3,4,5], vec![3,4,5,1,2]),
            (vec![2,3,4], vec![3,4,3]),
            (vec![5], vec![4]),
            (vec![3], vec![4]),
            (vec![1,2], vec![2,1]),
            (vec![1], vec![1]),
            (vec![2,2,2], vec![1,1,1]),
            (vec![5,4,3,2,1], vec![1,2,3,4,5]),
            (vec![1,1,1,10], vec![2,2,2,1]),
            (vec![3,1,1], vec![1,2,2]),
            (vec![1,5,1,5], vec![2,3,4,2]),
        ];
        
        for (gas, cost) in test_cases {
            let greedy = solution.can_complete_circuit_greedy(gas.clone(), cost.clone());
            let brute_force = solution.can_complete_circuit_brute_force(gas.clone(), cost.clone());
            let two_pointers = solution.can_complete_circuit_two_pointers(gas.clone(), cost.clone());
            let prefix_sum = solution.can_complete_circuit_prefix_sum(gas.clone(), cost.clone());
            let segment = solution.can_complete_circuit_segment(gas.clone(), cost.clone());
            let math = solution.can_complete_circuit_math(gas.clone(), cost.clone());
            
            assert_eq!(greedy, brute_force, "Greedy and brute force differ for gas={:?}, cost={:?}", gas, cost);
            assert_eq!(greedy, two_pointers, "Greedy and two pointers differ for gas={:?}, cost={:?}", gas, cost);
            assert_eq!(greedy, prefix_sum, "Greedy and prefix sum differ for gas={:?}, cost={:?}", gas, cost);
            assert_eq!(greedy, segment, "Greedy and segment differ for gas={:?}, cost={:?}", gas, cost);
            assert_eq!(greedy, math, "Greedy and math differ for gas={:?}, cost={:?}", gas, cost);
        }
    }
    
    #[test]
    fn test_circular_property() {
        let solution = Solution;
        
        // Test that the solution works for circular routes
        let gas = vec![1,2,3,4,5];
        let cost = vec![3,4,5,1,2];
        let start = solution.can_complete_circuit_greedy(gas.clone(), cost.clone());
        
        if start != -1 {
            let start_idx = start as usize;
            let mut tank = 0;
            
            // Simulate the complete circular journey
            for i in 0..gas.len() {
                let station = (start_idx + i) % gas.len();
                tank += gas[station];
                
                // Can we travel to the next station?
                if i < gas.len() - 1 || station != start_idx {
                    assert!(tank >= cost[station], "Cannot travel from station {}", station);
                    tank -= cost[station];
                }
            }
            
            // Should end up back at start with non-negative gas
            assert!(tank >= 0, "Should end with non-negative gas");
        }
    }
    
    #[test]
    fn test_performance_characteristics() {
        let solution = Solution;
        
        // Large array test
        let n = 1000;
        let mut gas = vec![2; n];
        let mut cost = vec![1; n];
        
        // Make one station require more gas to ensure unique solution
        gas[500] = 1000;
        cost[499] = 999;
        
        let result = solution.can_complete_circuit_greedy(gas.clone(), cost.clone());
        assert_ne!(result, -1);
        
        // Verify the same result across efficient methods
        assert_eq!(result, solution.can_complete_circuit_math(gas.clone(), cost.clone()));
        assert_eq!(result, solution.can_complete_circuit_prefix_sum(gas, cost));
    }
    
    #[test]
    fn test_impossible_scenarios() {
        let solution = Solution;
        
        // Total gas less than total cost
        assert_eq!(solution.can_complete_circuit_greedy(vec![1,2,3], vec![4,5,6]), -1);
        
        // Equal total but impossible distribution
        assert_eq!(solution.can_complete_circuit_greedy(vec![1,1,8], vec![2,9,1]), -1);
        
        // Single station insufficient
        assert_eq!(solution.can_complete_circuit_greedy(vec![1], vec![2]), -1);
    }
}