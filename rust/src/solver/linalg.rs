use std::time::Instant;
use std::panic::{catch_unwind, AssertUnwindSafe};
use super::Diagnostics;
use faer::sparse::linalg::solvers::{SymbolicLu, Lu};
use faer::sparse::SparseColMat;
use faer::col::from_slice_mut;
use faer::prelude::SpSolver;

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

        // 1. Compute row equilibration scales
        for r in 0..n {
            let mut max_val = 0.0_f64;
            for c in 0..n {
                let val = jac_dense[c * n + r].abs();
                if val.is_nan() { return Err("NaN detected in Jacobian".to_string()); }
                if val > max_val { max_val = val; }
            }
            self.row_scales[r] = if max_val > 0.0 { 1.0 / max_val } else { 1.0 };
        }

        // 2. Assemble scaled sparse triplets
        for c in 0..n {
            let mut has_diag = false;
            for r in 0..n {
                let unscaled = jac_dense[c * n + r];
                if unscaled != 0.0 {
                    let val = unscaled * self.row_scales[r];
                    self.triplets.push((r, c, val));
                    if r == c { has_diag = true; }
                }
            }
            if !has_diag {
                self.triplets.push((c, c, 1e-14)); 
            }
        }

        let jac_sparse_res = catch_unwind(AssertUnwindSafe(|| {
            SparseColMat::try_new_from_triplets(n, n, &self.triplets)
        }));
        let jac_sparse = match jac_sparse_res {
            Ok(Ok(mat)) => mat,
            _ => return Err("Sparse matrix assembly failed or panicked".to_string()),
        };

        // 3. Symbolic Cache
        if self.symbolic.is_none() {
            let sym_res = catch_unwind(AssertUnwindSafe(|| {
                SymbolicLu::try_new(jac_sparse.symbolic())
            }));
            self.symbolic = match sym_res {
                Ok(Ok(s)) => Some(s),
                _ => return Err("Symbolic LU failed or panicked".to_string()),
            };
        }

        // 4. Numeric Factorization
        let num_res = catch_unwind(AssertUnwindSafe(|| {
            Lu::try_new_with_symbolic(self.symbolic.as_ref().unwrap().clone(), jac_sparse.as_ref())
        }));
        
        match num_res {
            Ok(Ok(n_lu)) => self.numeric = Some(n_lu),
            _ => {
                // Fallback: Rebuild Symbolic safely inside a catch_unwind boundary
                // to prevent pyo3_runtime.PanicException on perfectly singular matrices.
                // Explicitly define the Result type to resolve type inference errors.
                let fallback_res = catch_unwind(AssertUnwindSafe(|| -> Result<(SymbolicLu<usize>, Lu<usize, f64>), String> {
                    let sym = SymbolicLu::try_new(jac_sparse.symbolic()).map_err(|_| "Symbolic Fallback failed".to_string())?;
                    let num = Lu::try_new_with_symbolic(sym.clone(), jac_sparse.as_ref()).map_err(|_| "Numeric Fallback failed".to_string())?;
                    Ok((sym, num))
                }));
                
                match fallback_res {
                    Ok(Ok((sym, num))) => {
                        self.symbolic = Some(sym);
                        self.numeric = Some(num);
                    },
                    _ => return Err("LU Factorization failed or panicked".to_string()),
                }
            }
        };

        self.is_stale = false;
        diag.numeric_factorizations += 1;
        diag.linear_solve_time_us += start_time.elapsed().as_micros();
        Ok(())
    }

    pub fn solve(&self, b: &mut [f64], diag: &mut Diagnostics) -> Result<(), String> {
        let start_time = Instant::now();
        if let Some(lu) = &self.numeric {
            for i in 0..self.n { b[i] *= self.row_scales[i]; }
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