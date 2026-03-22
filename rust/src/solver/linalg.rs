/// Solves a dense square linear system J * dx = b in-place using Gaussian elimination.
pub fn solve_dense_system(n: usize, jac: &mut [f64], b: &mut [f64]) -> Result<(), String> {
    for k in 0..n {
        let mut max_val = 0.0;
        let mut pivot_row = k;
        for i in k..n {
            let val = jac[k * n + i].abs();
            if val > max_val { max_val = val; pivot_row = i; }
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
            for col in k..n { jac[col * n + i] -= factor * jac[col * n + k]; }
        }
    }
    for i in (0..n).rev() {
        let mut sum = b[i];
        for col in (i + 1)..n { sum -= jac[col * n + i] * b[col]; }
        b[i] = sum / jac[i * n + i];
    }
    Ok(())
}

/// Solves a Banded linear system in-place unlocking massive Data Parallelism.
pub fn solve_banded_system(n: usize, bw: usize, jac: &mut [f64], b: &mut [f64]) -> Result<(), String> {
    for k in 0..n {
        let pivot = jac[k * n + k];
        if pivot.abs() < 1e-25 { return Err("Singular or ill-conditioned Banded Jacobian.".to_string()); }

        let end_row = std::cmp::min(n, k + bw + 1);
        for i in (k + 1)..end_row {
            let factor = jac[k * n + i] / pivot;
            b[i] -= factor * b[k];

            let end_col = std::cmp::min(n, k + bw + 1);
            for col in k..end_col { jac[col * n + i] -= factor * jac[col * n + k]; }
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

/// Matrix-Free Restarted GMRES Solver.
/// Evaluates J*v dynamically using Forward-Mode AD without explicitly storing the Jacobian.
pub fn solve_gmres<F>(n: usize, b: &mut [f64], mut jvp: F) -> Result<(), String>
where
    F: FnMut(&[f64], &mut [f64]),
{
    let m = std::cmp::min(n, 30);
    let mut v = vec![vec![0.0; n]; m + 1];
    let mut h = vec![vec![0.0; m]; m + 1];
    let mut cs = vec![0.0; m];
    let mut sn = vec![0.0; m];
    let mut g = vec![0.0; m + 1];

    let mut b_norm = 0.0;
    for i in 0..n { b_norm += b[i] * b[i]; }
    b_norm = b_norm.sqrt();

    if b_norm < 1e-12 {
        for i in 0..n { b[i] = 0.0; }
        return Ok(());
    }

    for i in 0..n { v[0][i] = b[i] / b_norm; }
    g[0] = b_norm;

    let mut k = 0;
    while k < m {
        let (left, right) = v.split_at_mut(k + 1);
        let v_k = &left[k];
        let v_kp1 = &mut right[0];
        
        jvp(v_k, v_kp1);

        // Modified Gram-Schmidt Orthogonalization
        for i in 0..=k {
            let v_i = &left[i];
            let mut dot = 0.0;
            for j in 0..n { dot += v_i[j] * v_kp1[j]; }
            h[i][k] = dot;
            for j in 0..n { v_kp1[j] -= dot * v_i[j]; }
        }

        let mut w_norm = 0.0;
        for j in 0..n { w_norm += v_kp1[j] * v_kp1[j]; }
        w_norm = w_norm.sqrt();

        h[k + 1][k] = w_norm;
        if w_norm > 1e-14 {
            for j in 0..n { v_kp1[j] /= w_norm; }
        }

        // Apply previous Givens rotations to H
        for i in 0..k {
            let temp = cs[i] * h[i][k] + sn[i] * h[i + 1][k];
            h[i + 1][k] = -sn[i] * h[i][k] + cs[i] * h[i + 1][k];
            h[i][k] = temp;
        }

        // Compute and apply new Givens rotation
        let beta = (h[k][k] * h[k][k] + h[k + 1][k] * h[k + 1][k]).sqrt();
        if beta > 1e-14 {
            cs[k] = h[k][k] / beta;
            sn[k] = h[k + 1][k] / beta;
        } else {
            cs[k] = 1.0;
            sn[k] = 0.0;
        }

        h[k][k] = cs[k] * h[k][k] + sn[k] * h[k + 1][k];
        h[k + 1][k] = 0.0;

        g[k + 1] = -sn[k] * g[k];
        g[k] = cs[k] * g[k];

        if g[k + 1].abs() < 1e-6 * b_norm {
            k += 1;
            break;
        }
        k += 1;
    }

    // Solve upper triangular system H y = g
    let mut y = vec![0.0; k];
    for i in (0..k).rev() {
        y[i] = g[i];
        for j in (i + 1)..k {
            y[i] -= h[i][j] * y[j];
        }
        y[i] /= h[i][i];
    }

    // Construct final solution x = V_k * y
    for i in 0..n { b[i] = 0.0; }
    for j in 0..k {
        for i in 0..n {
            b[i] += v[j][i] * y[j];
        }
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
        
        solve_dense_system(n, &mut jac, &mut b).unwrap();
        
        // Expected solution: x = [1.0, 1.0]
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
        
        solve_banded_system(n, bw, &mut jac, &mut b).unwrap();
        
        // Expected solution: x = [1.0, 1.0, 1.0]
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
    
    #[test]
    fn test_gmres_matrix_free_solver() {
        let n = 2;
        
        // System A x = b
        // A = [ 4.0, 1.0 ]
        //     [ 1.0, 3.0 ]
        // b = [ 1.0, 2.0 ]
        let mut b = vec![1.0, 2.0];
        
        // Matrix-Free Jacobian-Vector Product closure
        let jvp = |v: &[f64], out: &mut [f64]| {
            out[0] = 4.0 * v[0] + 1.0 * v[1];
            out[1] = 1.0 * v[0] + 3.0 * v[1];
        };
        
        solve_gmres(n, &mut b, jvp).unwrap();
        
        // Expected solution: x = [1/11, 7/11]
        assert!((b[0] - (1.0 / 11.0)).abs() < 1e-10);
        assert!((b[1] - (7.0 / 11.0)).abs() < 1e-10);
    }
}