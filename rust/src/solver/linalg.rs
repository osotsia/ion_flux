/// Solves a dense square linear system J * dx = b in-place using Gaussian elimination.
/// O(N^3) time complexity. J is expected in Column-Major layout.
pub fn solve_dense_system(n: usize, jac: &mut [f64], b: &mut [f64]) -> Result<(), String> {
    for k in 0..n {
        let mut max_val = 0.0;
        let mut pivot_row = k;
        for i in k..n {
            let val = jac[k * n + i].abs();
            if val > max_val {
                max_val = val;
                pivot_row = i;
            }
        }
        if max_val < 1e-25 { return Err("Singular Jacobian matrix.".to_string()); }
        
        if pivot_row != k {
            b.swap(k, pivot_row);
            for col in 0..n {
                let tmp = jac[col * n + k];
                jac[col * n + k] = jac[col * n + pivot_row];
                jac[col * n + pivot_row] = tmp;
            }
        }
        
        let pivot = jac[k * n + k];
        for i in (k + 1)..n {
            let factor = jac[k * n + i] / pivot;
            b[i] -= factor * b[k];
            for col in k..n {
                jac[col * n + i] -= factor * jac[col * n + k];
            }
        }
    }
    // Back-substitution
    for i in (0..n).rev() {
        let mut sum = b[i];
        for col in (i + 1)..n { sum -= jac[col * n + i] * b[col]; }
        b[i] = sum / jac[i * n + i];
    }
    Ok(())
}

/// Solves a Banded linear system in-place unlocking massive Data Parallelism.
/// O(N * bw^2) time complexity. Operates on the full dense array but only strictly within bandwidth.
pub fn solve_banded_system(n: usize, bw: usize, jac: &mut [f64], b: &mut [f64]) -> Result<(), String> {
    for k in 0..n {
        let pivot = jac[k * n + k];
        if pivot.abs() < 1e-25 { return Err("Singular or ill-conditioned Banded Jacobian.".to_string()); }

        let end_row = std::cmp::min(n, k + bw + 1);
        for i in (k + 1)..end_row {
            let factor = jac[k * n + i] / pivot;
            b[i] -= factor * b[k];

            let end_col = std::cmp::min(n, k + bw + 1);
            for col in k..end_col {
                jac[col * n + i] -= factor * jac[col * n + k];
            }
        }
    }
    for i in (0..n).rev() {
        let mut sum = b[i];
        let end_col = std::cmp::min(n, i + bw + 1);
        for col in (i + 1)..end_col { sum -= jac[col * n + i] * b[col]; }
        b[i] = sum / jac[i * n + i];
    }
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_gaussian_elimination() {
        let n = 2;
        // J = [ 2.0, -1.0 ]
        //     [ 1.0,  3.0 ] (Column-Major)
        let mut jac = vec![2.0, 1.0, -1.0, 3.0];
        let mut b = vec![1.0, 4.0];
        
        // Expected solution: x = [1.0, 1.0]
        solve_dense_system(n, &mut jac, &mut b).unwrap();
        
        assert!((b[0] - 1.0).abs() < 1e-10);
        assert!((b[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_banded_system_solver() {
        let n = 3;
        let bw = 1; // Tridiagonal
        
        // J = [ 4.0, -1.0,  0.0 ]
        //     [-1.0,  4.0, -1.0 ]
        //     [ 0.0, -1.0,  4.0 ] (Column-Major)
        let mut jac = vec![
             4.0, -1.0,  0.0,
            -1.0,  4.0, -1.0,
             0.0, -1.0,  4.0
        ];
        let mut b = vec![3.0, 2.0, 3.0];
        
        // Expected solution: x = [1.0, 1.0, 1.0]
        solve_banded_system(n, bw, &mut jac, &mut b).unwrap();
        
        assert!((b[0] - 1.0).abs() < 1e-10);
        assert!((b[1] - 1.0).abs() < 1e-10);
        assert!((b[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_singular_matrix_panic() {
        let n = 2;
        let mut jac = vec![1.0, 2.0, 2.0, 4.0]; // Linearly dependent columns
        let mut b = vec![1.0, 2.0];
        
        let result = solve_dense_system(n, &mut jac, &mut b);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Singular Jacobian matrix.");
    }
}