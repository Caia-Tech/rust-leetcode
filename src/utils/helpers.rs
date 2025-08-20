//! Helper functions and utilities

/// Test helper macros and functions
#[cfg(test)]
pub mod test_helpers {
    /// Macro to create test cases with multiple inputs and expected outputs
    #[macro_export]
    macro_rules! test_cases {
        ($($name:ident: $inputs:expr => $expected:expr,)*) => {
            $(
                #[test]
                fn $name() {
                    let solution = Solution::new();
                    let result = solution.solve($inputs);
                    assert_eq!(result, $expected);
                }
            )*
        };
    }
}

/// Common mathematical utilities
pub mod math {
    /// Check if a number is prime
    pub fn is_prime(n: i32) -> bool {
        if n < 2 {
            return false;
        }
        for i in 2..=((n as f64).sqrt() as i32) {
            if n % i == 0 {
                return false;
            }
        }
        true
    }
    
    /// Greatest common divisor
    pub fn gcd(a: i32, b: i32) -> i32 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
}