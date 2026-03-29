use std::time::Instant;
use super::Diagnostics;
use faer::sparse::linalg::solvers::{SymbolicLu, Lu};
use faer::sparse::SparseColMat;
use faer::col::from_slice_mut;
use faer::prelude::SpSolver;

/// A robust wrapper around `faer`'s Sparse LU solver.
/// Replaces legacy O(N * bw^2) Banded routines, converting the dense C++ AD output 
/// into an optimal CSC structure on the fly, eliminating the DFN bandwidth penalty.
pub struct NativeSparseLuSolver {
    pub is_stale: bool,
    pub n: usize,
    pub bw: isize,
    symbolic: Option<SymbolicLu<usize>>, 
    numeric: Option<Lu<usize, f64>>,     
    pub triplets: Vec<(usize, usize, f64)>,
    pub row_scales: Vec<f64>,
}

impl NativeSparseLuSolver {
    pub fn new(n: usize, bw: isize) -> Self {
        let estimated_nnz = n * (2 * bw.max(0) as usize + 1).min(n);
        
        Self {
            is_stale: true,
            n,
            bw,
            symbolic: None,
            numeric: None,
            triplets: Vec::with_capacity(estimated_nnz),
            row_scales: vec![1.0; n],
        }
    }

    pub fn factorize(&mut self, jac_dense: &[f64], diag: &mut Diagnostics) -> Result<(), String> {
        let start_time = Instant::now();
        let n = self.n;
        
        self.triplets.clear();

        // Compute row equilibration scales
        for r in 0..n {
            let mut max_val = 0.0_f64;
            for c in 0..n {
                let val = jac_dense[c * n + r].abs();
                if val > max_val { max_val = val; }
            }
            self.row_scales[r] = if max_val > 1e-15 { 1.0 / max_val } else { 1.0 };
        }

        // Assemble scaled triplets
        for c in 0..n {
            for r in 0..n {
                let val = jac_dense[c * n + r] * self.row_scales[r];
                if val.abs() > 1e-14 {
                    self.triplets.push((r, c, val));
                }
            }
        }

        let jac_sparse = SparseColMat::try_new_from_triplets(
            n, n, &self.triplets
        ).map_err(|_| "Failed to assemble sparse matrix from triplets.".to_string())?;

        let sym = SymbolicLu::try_new(jac_sparse.symbolic()).map_err(|_| "Symbolic LU failed".to_string())?;
        self.numeric = Some(
            Lu::try_new_with_symbolic(sym.clone(), jac_sparse.as_ref()).map_err(|_| "Numeric LU failed".to_string())?
        );
        self.symbolic = Some(sym);

        self.is_stale = false;
        diag.numeric_factorizations += 1;
        diag.linear_solve_time_us += start_time.elapsed().as_micros();
        Ok(())
    }

    pub fn solve(&self, b: &mut [f64], diag: &mut Diagnostics) -> Result<(), String> {
        let start_time = Instant::now();
        if let Some(lu) = &self.numeric {
            // Apply equilibration row scales to RHS residual
            for i in 0..self.n {
                b[i] *= self.row_scales[i];
            }
            lu.solve_in_place(from_slice_mut(b));
            diag.linear_solve_time_us += start_time.elapsed().as_micros();
            Ok(())
        } else {
            Err("Attempted to solve before factorization.".to_string())
        }
    }

    pub fn mark_stale(&mut self) {
        self.is_stale = true;
    }
}

/// Left-Preconditioned Matrix-Free Restarted GMRES Solver (Used for 3D meshes where bw == -1)
pub fn solve_gmres<F, P>(n: usize, b: &mut [f64], mut jvp: F, mut precond: P) -> Result<(), String>
where F: FnMut(&[f64], &mut [f64]), P: FnMut(&[f64], &mut[f64]) {
    let m = std::cmp::min(n, 30);
    let mut v = vec![vec![0.0; n]; m + 1];
    let mut h = vec![vec![0.0; m]; m + 1];
    let mut cs = vec![0.0; m];
    let mut sn = vec![0.0; m];
    let mut g = vec![0.0; m + 1];

    let mut b_pre = vec![0.0; n];
    precond(b, &mut b_pre);

    let mut b_norm = 0.0;
    for i in 0..n { b_norm += b_pre[i] * b_pre[i]; }
    b_norm = b_norm.sqrt();

    if b_norm < 1e-12 {
        for i in 0..n { b[i] = 0.0; }
        return Ok(());
    }

    for i in 0..n { v[0][i] = b_pre[i] / b_norm; }
    g[0] = b_norm;

    let mut k = 0;
    let mut temp_jvp = vec![0.0; n];

    while k < m {
        let (left, right) = v.split_at_mut(k + 1);
        let v_k = &left[k];
        let v_kp1 = &mut right[0];
        
        jvp(v_k, &mut temp_jvp);
        precond(&temp_jvp, v_kp1);

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

        for i in 0..k {
            let temp = cs[i] * h[i][k] + sn[i] * h[i + 1][k];
            h[i + 1][k] = -sn[i] * h[i][k] + cs[i] * h[i + 1][k];
            h[i][k] = temp;
        }

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

    let mut y = vec![0.0; k];
    for i in (0..k).rev() {
        y[i] = g[i];
        for j in (i + 1)..k { y[i] -= h[i][j] * y[j]; }
        if h[i][i].abs() < 1e-14 {
            return Err("GMRES Singular H matrix".to_string());
        }
        y[i] /= h[i][i];
    }

    for i in 0..n { b[i] = 0.0; }
    for j in 0..k {
        for i in 0..n { b[i] += v[j][i] * y[j]; }
    }

    Ok(())
}