//! Helper functions and utilities

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
