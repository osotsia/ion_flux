use std::time::Instant;
use std::panic::{catch_unwind, AssertUnwindSafe};
use crate::solver::shared::diagnostics::Diagnostics;
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
    cached_pattern: Vec<(usize, usize)>,
    pub row_scales: Vec<f64>,
}

impl NativeSparseLuSolver {
    pub fn new(n: usize, bw: isize) -> Self {
        let estimated_nnz = n * (2 * bw.max(0) as usize + 1).min(n);
        Self {
            is_stale: true, n, bw,
            symbolic: None, numeric: None,
            triplets: Vec::with_capacity(estimated_nnz),
            cached_pattern: Vec::with_capacity(estimated_nnz),
            row_scales: vec![1.0; n],
        }
    }

    pub fn factorize_from_triplets(&mut self, diag: &mut Diagnostics) -> Result<(), String> {
        let n = self.n;
        for r in 0..n { self.row_scales[r] = 1.0; }
        for &(r, _, val) in &self.triplets {
            let abs_val = val.abs();
            if abs_val.is_nan() { return Err("NaN detected in Jacobian".to_string()); }
            if abs_val > self.row_scales[r] { self.row_scales[r] = abs_val; }
        }
        for r in 0..n { self.row_scales[r] = if self.row_scales[r] > 0.0 { 1.0 / self.row_scales[r] } else { 1.0 }; }
        
        for i in 0..self.triplets.len() {
            let (r, c, val) = self.triplets[i];
            let mut scaled = val * self.row_scales[r];
            if r == c && scaled.abs() < 1e-14 { scaled = 1e-14; }
            self.triplets[i].2 = scaled;
        }
        self.factorize_internal(diag)
    }

    fn factorize_internal(&mut self, diag: &mut Diagnostics) -> Result<(), String> {
        let start_time = Instant::now();
        let mut j_max = 0.0_f64;
        let mut j_min = std::f64::MAX;
        
        for &(_, _, val) in &self.triplets {
            let abs_val = val.abs();
            if abs_val > j_max { j_max = abs_val; }
            if abs_val > 0.0 && abs_val < j_min { j_min = abs_val; }
        }
        diag.jac_max = j_max; diag.jac_min = j_min;

        let mut pattern_changed = self.triplets.len() != self.cached_pattern.len();
        if !pattern_changed {
            for i in 0..self.triplets.len() {
                if self.triplets[i].0 != self.cached_pattern[i].0 || self.triplets[i].1 != self.cached_pattern[i].1 {
                    pattern_changed = true; break;
                }
            }
        }

        if pattern_changed {
            self.symbolic = None;
            self.cached_pattern.clear();
            for &(r, c, _) in &self.triplets { self.cached_pattern.push((r, c)); }
        }

        let jac_sparse_res = catch_unwind(AssertUnwindSafe(|| { SparseColMat::try_new_from_triplets(self.n, self.n, &self.triplets) }));
        let jac_sparse = match jac_sparse_res {
            Ok(Ok(mat)) => mat,
            _ => return Err("Sparse matrix assembly panicked.".to_string()),
        };

        if self.symbolic.is_none() {
            let sym_res = catch_unwind(AssertUnwindSafe(|| { SymbolicLu::try_new(jac_sparse.symbolic()) }));
            self.symbolic = match sym_res {
                Ok(Ok(s)) => Some(s),
                _ => return Err("Symbolic LU panicked.".to_string()),
            };
        }

        let num_res = catch_unwind(AssertUnwindSafe(|| { Lu::try_new_with_symbolic(self.symbolic.as_ref().unwrap().clone(), jac_sparse.as_ref()) }));
        match num_res {
            Ok(Ok(n_lu)) => self.numeric = Some(n_lu),
            _ => {
                let fallback = catch_unwind(AssertUnwindSafe(|| -> Result<(SymbolicLu<usize>, Lu<usize, f64>), String> {
                    let sym = SymbolicLu::try_new(jac_sparse.symbolic()).map_err(|_| "Sym fail".to_string())?;
                    let num = Lu::try_new_with_symbolic(sym.clone(), jac_sparse.as_ref()).map_err(|_| "Num fail".to_string())?;
                    Ok((sym, num))
                }));
                match fallback {
                    Ok(Ok((sym, num))) => { self.symbolic = Some(sym); self.numeric = Some(num); },
                    _ => return Err("LU Factorization panicked.".to_string())
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
        } else { Err("Attempted to solve before factorization.".to_string()) }
    }

    pub fn mark_stale(&mut self) { self.is_stale = true; }
}