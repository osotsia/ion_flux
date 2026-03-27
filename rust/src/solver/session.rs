use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, ToPyArray};
use std::fmt::Write;
use super::{NativeResFn, NativeJacFn, NativeJvpFn};
use super::integrator::BdfIntegrator;
use super::linalg::{solve_dense_system, solve_banded_system, solve_gmres};

#[pyclass(unsendable)]
pub struct SolverHandle {
    _lib: libloading::Library,
    res_fn: NativeResFn,
    jac_fn: NativeJacFn,
    jvp_fn: Option<NativeJvpFn>,
    pub n: usize,
    pub bw: isize,
    pub t: f64,
    pub y: Vec<f64>,
    pub ydot: Vec<f64>,
    pub id: Vec<f64>,
    pub p: Vec<f64>,
    pub spatial_diag: Vec<f64>,
    pub y_prev: Vec<f64>,
    pub y_prev2: Vec<f64>,
    pub dt_prev: f64,
    pub order: usize,
    pub debug: bool,
}

#[pymethods]
impl SolverHandle {
    #[new]
    pub fn new(lib_path: String, n: usize, bw: isize, y0: Vec<f64>, ydot0: Vec<f64>, id: Vec<f64>, p: Vec<f64>, spatial_diag: Vec<f64>, debug: bool) -> PyResult<Self> {
        let lib = unsafe { libloading::Library::new(&lib_path).expect("Failed to load JIT shared library.") };
        let res_fn: NativeResFn = unsafe { *lib.get::<NativeResFn>(b"evaluate_residual\0").unwrap() };
        let jac_fn: NativeJacFn = unsafe { *lib.get::<NativeJacFn>(b"evaluate_jacobian\0").unwrap() };
        
        let jvp_fn: Option<NativeJvpFn> = unsafe {
            lib.get::<NativeJvpFn>(b"evaluate_jvp\0").map(|sym| *sym).ok()
        };
        
        let y_prev = y0.clone();
        let y_prev2 = y0.clone();

        let mut handle = SolverHandle { 
            _lib: lib, res_fn, jac_fn, jvp_fn, n, bw, 
            t: 0.0, y: y0, ydot: ydot0, id, p, spatial_diag,
            y_prev, y_prev2, dt_prev: 0.0, order: 1, debug
        };
        
        handle.calc_algebraic_roots()?;
        Ok(handle)
    }

    pub fn clone_state(&self) -> (f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64, usize) {
        (self.t, self.y.clone(), self.ydot.clone(), self.y_prev.clone(), self.y_prev2.clone(), self.dt_prev, self.order)
    }

    pub fn restore_state(&mut self, t: f64, y: Vec<f64>, ydot: Vec<f64>, y_prev: Vec<f64>, y_prev2: Vec<f64>, dt_prev: f64, order: usize) {
        self.t = t; self.y = y; self.ydot = ydot; self.y_prev = y_prev; self.y_prev2 = y_prev2; self.dt_prev = dt_prev; self.order = order;
    }

    pub fn step(&mut self, dt: f64) -> PyResult<()> {
        self.step_with_history(dt, None)
    }

    pub fn step_history<'py>(&mut self, py: Python<'py>, dt: f64) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let mut hist = vec![];
        self.step_with_history(dt, Some(&mut hist))?;
        
        let h_len = hist.len();
        let mut micro_t = vec![0.0; h_len];
        let mut micro_y = vec![0.0; h_len * self.n];
        let mut micro_ydot = vec![0.0; h_len * self.n];
        for (i, (t, y, ydot)) in hist.into_iter().enumerate() {
            micro_t[i] = t;
            for j in 0..self.n {
                micro_y[i * self.n + j] = y[j];
                micro_ydot[i * self.n + j] = ydot[j];
            }
        }
        Ok((
            numpy::ndarray::Array1::from_vec(micro_t).to_pyarray_bound(py),
            numpy::ndarray::Array2::from_shape_vec((h_len, self.n), micro_y).unwrap().to_pyarray_bound(py),
            numpy::ndarray::Array2::from_shape_vec((h_len, self.n), micro_ydot).unwrap().to_pyarray_bound(py)
        ))
    }

    /// Evaluates a zero-dt solve over the system to find consistent initial conditions (ICs). 
    /// Freezes differential states and evaluates the algebraic equilibrium manifold.
    pub fn calc_algebraic_roots(&mut self) -> PyResult<()> {
        let mut res = vec![0.0; self.n];
        let mut jac = vec![0.0; self.n * self.n];
        let mut dy = vec![0.0; self.n];
        let mut weights = vec![0.0; self.n];
        
        let mut n_alg = 0;
        for i in 0..self.n {
            if self.id[i] == 0.0 { n_alg += 1; }
        }
        let n_alg_f64 = if n_alg > 0 { n_alg as f64 } else { 1.0 };
        
        let mut history: Vec<(usize, f64, f64, f64)> = Vec::new();
        let max_iters = 100;

        for iter in 0..max_iters {
            unsafe { (self.res_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), res.as_mut_ptr()) };
            
            for i in 0..self.n {
                weights[i] = 1.0 / (1e-6 * self.y[i].abs() + 1e-8);
            }
            
            let wrms_norm = |v: &[f64], w: &[f64]| -> f64 {
                let mut sum = 0.0;
                for i in 0..self.n {
                    if self.id[i] == 0.0 { // Only norm algebraic states
                        let scaled = v[i] * w[i];
                        if !scaled.is_finite() { return f64::INFINITY; }
                        sum += scaled * scaled;
                    }
                }
                (sum / n_alg_f64).sqrt()
            };

            let f_norm = wrms_norm(&res, &weights);
            if !f_norm.is_finite() {
                let report = self.build_crash_report("Algebraic Initialization", "Residual evaluated to NaN or Inf.", &res, &dy, &history);
                return Err(pyo3::exceptions::PyRuntimeError::new_err(report));
            }
            
            let max_res = res.iter().enumerate().filter(|(i, _)| self.id[*i] == 0.0).map(|(_, v)| v.abs()).fold(0.0f64, |a: f64, b: f64| a.max(b));
            if max_res < 1e-8 { break; } 

            let mut lin_success = true;
            if self.bw == -1 {
                let mut rhs = vec![0.0; self.n];
                for i in 0..self.n { rhs[i] = if self.id[i] == 1.0 { 0.0 } else { -res[i] }; }
                
                let jvp = self.jvp_fn.expect("evaluate_jvp not found. Recompile the JIT model.");
                let y_ptr = self.y.as_ptr();
                let ydot_ptr = self.ydot.as_ptr();
                let p_ptr = self.p.as_ptr();
                let id_ptr = self.id.as_ptr();
                
                let jvp_closure = |v: &[f64], out: &mut [f64]| {
                    unsafe { 
                        jvp(y_ptr, ydot_ptr, p_ptr, 0.0, v.as_ptr(), out.as_mut_ptr()); 
                        for i in 0..self.n { if *id_ptr.add(i) == 1.0 { out[i] = v[i]; } }
                    }
                };
                
                let precond = |v: &[f64], out: &mut [f64]| {
                    for i in 0..self.n { out[i] = v[i] / (self.spatial_diag[i] + 1.0); }
                };
                
                if solve_gmres(self.n, &mut rhs, jvp_closure, precond).is_err() { lin_success = false; }
                dy.copy_from_slice(&rhs);
            } else {
                unsafe { (self.jac_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), 0.0, jac.as_mut_ptr()) };
                for i in 0..self.n {
                    dy[i] = -res[i];
                    if self.id[i] == 1.0 {
                        dy[i] = 0.0; // Freeze differential states
                        for col in 0..self.n { jac[col * self.n + i] = if col == i { 1.0 } else { 0.0 }; }
                    }
                }
                if self.bw > 0 { 
                    if solve_banded_system(self.n, self.bw as usize, &mut jac, &mut dy).is_err() { lin_success = false; }
                } else { 
                    if solve_dense_system(self.n, &mut jac, &mut dy).is_err() { lin_success = false; }
                }
            }

            if !lin_success {
                let report = self.build_crash_report("Algebraic Initialization", "Linear solver failed (Singular Jacobian).", &res, &dy, &history);
                return Err(pyo3::exceptions::PyRuntimeError::new_err(report));
            }

            let max_dy = dy.iter().map(|v| v.abs()).fold(0.0f64, |a: f64, b: f64| if !a.is_finite() { a } else if !b.is_finite() { b } else { a.max(b) });
            if !max_dy.is_finite() {
                let report = self.build_crash_report("Algebraic Initialization", "Newton step (dy) generated NaN or Inf.", &res, &dy, &history);
                return Err(pyo3::exceptions::PyRuntimeError::new_err(report));
            }

            // SUNDIALS Technique: Bounded Step Limiter (Trust Region)
            // Prevents massive Newton overshoots from throwing variables into unphysical domains
            // (e.g. updating a 4.2V potential by 1,000,000 Volts in a single step).
            if max_dy > 50.0 {
                let scale = 50.0 / max_dy;
                for i in 0..self.n { dy[i] *= scale; }
            }

            let dy_norm = wrms_norm(&dy, &weights); // Calculate AFTER applying the step limiter

            let mut alpha = 1.0;
            let mut step_accepted = false;
            let mut y_trial = vec![0.0; self.n];
            let mut res_trial = vec![0.0; self.n];
            
            // Deepened Armijo Line Search
            for _ in 0..15 {
                for i in 0..self.n { y_trial[i] = self.y[i] + alpha * dy[i]; }
                unsafe { (self.res_fn)(y_trial.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), res_trial.as_mut_ptr()) };
                
                let f_norm_trial = wrms_norm(&res_trial, &weights);
                let max_res_trial = res_trial.iter().enumerate().filter(|(i, _)| self.id[*i] == 0.0).map(|(_, v)| v.abs()).fold(0.0f64, |a: f64, b: f64| if !a.is_finite() { a } else if !b.is_finite() { b } else { a.max(b) });

                if f_norm_trial.is_finite() && (f_norm_trial <= f_norm * (1.0 - 1e-4 * alpha) || max_res_trial < 1e-6) {
                    self.y.copy_from_slice(&y_trial);
                    step_accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            
            history.push((iter, f_norm, dy_norm, alpha));

            if !step_accepted {
                let report = self.build_crash_report("Algebraic Initialization", "Line Search exhausted (step rejected).", &res, &dy, &history);
                return Err(pyo3::exceptions::PyRuntimeError::new_err(report));
            }

            if iter == max_iters - 1 {
                let report = self.build_crash_report("Algebraic Initialization", "Failed to converge within maximum iterations.", &res, &dy, &history);
                return Err(pyo3::exceptions::PyRuntimeError::new_err(report));
            }
        }

        self.y_prev.copy_from_slice(&self.y);
        self.y_prev2.copy_from_slice(&self.y);
        self.dt_prev = 0.0;
        self.order = 1;
        Ok(())
    }

    /// Evaluates a zero-dt solve over the entire system assuming equilibrium (`ydot` = 0). 
    pub fn reach_steady_state(&mut self) -> PyResult<()> {
        for i in 0..self.n { self.ydot[i] = 0.0; }
        
        let mut res = vec![0.0; self.n];
        let mut jac = vec![0.0; self.n * self.n];
        let mut dy = vec![0.0; self.n];
        let mut weights = vec![0.0; self.n];
        
        let mut history: Vec<(usize, f64, f64, f64)> = Vec::new();
        let max_iters = 100;

        for iter in 0..max_iters {
            unsafe { (self.res_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), res.as_mut_ptr()) };
            
            for i in 0..self.n {
                weights[i] = 1.0 / (1e-6 * self.y[i].abs() + 1e-8);
            }
            
            let wrms_norm = |v: &[f64], w: &[f64]| -> f64 {
                let mut sum = 0.0;
                for i in 0..self.n {
                    let scaled = v[i] * w[i];
                    if !scaled.is_finite() { return f64::INFINITY; }
                    sum += scaled * scaled;
                }
                (sum / (self.n as f64)).sqrt()
            };

            let f_norm = wrms_norm(&res, &weights);
            if !f_norm.is_finite() {
                let report = self.build_crash_report("Steady State Initialization", "Residual evaluated to NaN or Inf.", &res, &dy, &history);
                return Err(pyo3::exceptions::PyRuntimeError::new_err(report));
            }
            
            let max_res = res.iter().map(|v| v.abs()).fold(0.0f64, |a: f64, b: f64| a.max(b));
            if max_res < 1e-12 { break; } 

            let mut lin_success = true;
            if self.bw == -1 {
                let mut rhs = vec![0.0; self.n];
                for i in 0..self.n { rhs[i] = -res[i]; }

                let jvp = self.jvp_fn.expect("evaluate_jvp not found. Recompile the JIT model.");
                let y_ptr = self.y.as_ptr();
                let ydot_ptr = self.ydot.as_ptr();
                let p_ptr = self.p.as_ptr();
                
                let jvp_closure = |v: &[f64], out: &mut [f64]| {
                    unsafe { jvp(y_ptr, ydot_ptr, p_ptr, 0.0, v.as_ptr(), out.as_mut_ptr()) };
                };
                let precond = |v: &[f64], out: &mut [f64]| {
                    for i in 0..self.n { out[i] = v[i] / (self.spatial_diag[i] + 1.0); }
                };
                
                if solve_gmres(self.n, &mut rhs, jvp_closure, precond).is_err() { lin_success = false; }
                dy.copy_from_slice(&rhs);
            } else {
                unsafe { (self.jac_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), 0.0, jac.as_mut_ptr()) };
                for i in 0..self.n { dy[i] = -res[i]; }
                if self.bw > 0 { 
                    if solve_banded_system(self.n, self.bw as usize, &mut jac, &mut dy).is_err() { lin_success = false; }
                } else { 
                    if solve_dense_system(self.n, &mut jac, &mut dy).is_err() { lin_success = false; }
                }
            }

            if !lin_success {
                let report = self.build_crash_report("Steady State Initialization", "Linear solver failed (Singular Jacobian).", &res, &dy, &history);
                return Err(pyo3::exceptions::PyRuntimeError::new_err(report));
            }

            let max_dy = dy.iter().map(|v| v.abs()).fold(0.0f64, |a: f64, b: f64| if !a.is_finite() { a } else if !b.is_finite() { b } else { a.max(b) });
            if !max_dy.is_finite() {
                let report = self.build_crash_report("Steady State Initialization", "Newton step (dy) generated NaN or Inf.", &res, &dy, &history);
                return Err(pyo3::exceptions::PyRuntimeError::new_err(report));
            }

            // Trust Region Step Limiter
            if max_dy > 50.0 {
                let scale = 50.0 / max_dy;
                for i in 0..self.n { dy[i] *= scale; }
            }

            let dy_norm = wrms_norm(&dy, &weights); 

            let mut alpha = 1.0;
            let mut step_accepted = false;
            let mut y_trial = vec![0.0; self.n];
            let mut res_trial = vec![0.0; self.n];
            
            for _ in 0..15 { 
                for i in 0..self.n { y_trial[i] = self.y[i] + alpha * dy[i]; }
                unsafe { (self.res_fn)(y_trial.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), res_trial.as_mut_ptr()) };
                
                let f_norm_trial = wrms_norm(&res_trial, &weights);
                let max_res_trial = res_trial.iter().map(|v| v.abs()).fold(0.0f64, |a: f64, b: f64| if !a.is_finite() { a } else if !b.is_finite() { b } else { a.max(b) });
                
                if f_norm_trial.is_finite() && (f_norm_trial <= f_norm * (1.0 - 1e-4 * alpha) || max_res_trial < 1e-6) {
                    self.y.copy_from_slice(&y_trial);
                    step_accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            
            history.push((iter, f_norm, dy_norm, alpha));

            if !step_accepted {
                let report = self.build_crash_report("Steady State Initialization", "Line Search exhausted (step rejected).", &res, &dy, &history);
                return Err(pyo3::exceptions::PyRuntimeError::new_err(report));
            }

            if dy_norm < 1.0 { break; } 
            
            if iter == max_iters - 1 {
                let report = self.build_crash_report("Steady State Initialization", "Failed to converge within maximum iterations.", &res, &dy, &history);
                return Err(pyo3::exceptions::PyRuntimeError::new_err(report));
            }
        }
        
        self.y_prev.copy_from_slice(&self.y);
        self.y_prev2.copy_from_slice(&self.y);
        self.dt_prev = 0.0;
        self.order = 1;
        Ok(())
    }

    pub fn set_parameter(&mut self, idx: usize, val: f64) { if idx < self.p.len() { self.p[idx] = val; } }
    
    pub fn get_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> { numpy::ndarray::Array1::from_vec(self.y.clone()).to_pyarray_bound(py) }
    
    pub fn get_jacobian<'py>(&self, py: Python<'py>, c_j: f64) -> Bound<'py, PyArray2<f64>> {
        let mut jac = vec![0.0; self.n * self.n];
        unsafe { (self.jac_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), c_j, jac.as_mut_ptr()) };
        let mut jac_rm = vec![0.0; self.n * self.n];
        for col in 0..self.n {
            for row in 0..self.n { jac_rm[row * self.n + col] = jac[col * self.n + row]; }
        }
        numpy::ndarray::Array2::from_shape_vec((self.n, self.n), jac_rm).unwrap().to_pyarray_bound(py)
    }

    pub fn set_threads(&self, threads: i32) {
        if let Ok(func) = unsafe { self._lib.get::<unsafe extern "C" fn(i32)>(b"set_spatial_threads\0") } {
            unsafe { func(threads) };
        }
    }
}

// -----------------------------------------------------------------------------
// Internal Rust Methods 
// -----------------------------------------------------------------------------
impl SolverHandle {
    pub fn step_with_history(&mut self, dt: f64, history: Option<&mut Vec<(f64, Vec<f64>, Vec<f64>)>>) -> PyResult<()> {
        let mut integrator = BdfIntegrator::default();
        integrator.debug = self.debug;
        
        integrator.step(
            self.n, self.bw, &mut self.y, &mut self.ydot, &self.p, &self.id, &self.spatial_diag, dt, 
            self.res_fn, self.jac_fn, self.jvp_fn,
            &mut self.y_prev, &mut self.y_prev2, &mut self.dt_prev, &mut self.order,
            history, self.t
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        self.t += dt;
        Ok(())
    }

    fn build_crash_report(&self, stage: &str, error_msg: &str, res: &[f64], dy: &[f64], history: &[(usize, f64, f64, f64)]) -> String {
        let mut s = String::new();
        
        let nan_y = self.y.iter().filter(|v| v.is_nan()).count();
        let nan_res = res.iter().filter(|v| v.is_nan()).count();
        let nan_dy = dy.iter().filter(|v| v.is_nan()).count();

        writeln!(&mut s, "\n============================================================").unwrap();
        writeln!(&mut s, "ION FLUX SOLVER CRASH SUMMARY").unwrap();
        writeln!(&mut s, "============================================================").unwrap();
        writeln!(&mut s, "Stage: {}", stage).unwrap();
        writeln!(&mut s, "Error: {}", error_msg).unwrap();
        writeln!(&mut s, "System Size (N): {}", self.n).unwrap();
        writeln!(&mut s, "------------------------------------------------------------").unwrap();
        writeln!(&mut s, "NaN Count -> States (y): {}, Residuals (res): {}, Steps (dy): {}", nan_y, nan_res, nan_dy).unwrap();
        writeln!(&mut s, "------------------------------------------------------------").unwrap();
        
        writeln!(&mut s, "Recent Iteration History (Convergence Profile):").unwrap();
        if history.is_empty() {
            writeln!(&mut s, "  No iterations completed.").unwrap();
        } else {
            let recent: Vec<_> = history.iter().rev().take(5).collect();
            for &(iter, r_norm, d_norm, alpha) in recent.into_iter().rev() {
                writeln!(&mut s, "  Iter: {:>3} | Res Norm: {:12.5e} | Step Norm: {:12.5e} | Alpha: {:12.5e}", iter, r_norm, d_norm, alpha).unwrap();
            }
        }
        writeln!(&mut s, "------------------------------------------------------------").unwrap();
        
        writeln!(&mut s, "Top 5 Worst Offenders (Largest Unsolved Equations):").unwrap();
        let mut res_idx: Vec<usize> = (0..self.n).collect();
        res_idx.sort_by(|&a, &b| {
            let va = if res[a].is_nan() { f64::INFINITY } else { res[a].abs() };
            let vb = if res[b].is_nan() { f64::INFINITY } else { res[b].abs() };
            vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        for &i in res_idx.iter().take(5) {
            let t = if self.id[i] == 1.0 { "Differential" } else { "Algebraic" };
            writeln!(&mut s, "  Index {:>4} | Res: {:12.5e} | State (y): {:12.5e} | Type: {}", i, res[i], self.y[i], t).unwrap();
        }
        writeln!(&mut s, "------------------------------------------------------------").unwrap();
        
        writeln!(&mut s, "Top 5 Most Volatile States (Largest Attempted Updates):").unwrap();
        let mut dy_idx: Vec<usize> = (0..self.n).collect();
        dy_idx.sort_by(|&a, &b| {
            let va = if dy[a].is_nan() { f64::INFINITY } else { dy[a].abs() };
            let vb = if dy[b].is_nan() { f64::INFINITY } else { dy[b].abs() };
            vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        for &i in dy_idx.iter().take(5) {
            let t = if self.id[i] == 1.0 { "Differential" } else { "Algebraic" };
            writeln!(&mut s, "  Index {:>4} | Step (dy): {:12.5e} | State (y): {:12.5e} | Type: {}", i, dy[i], self.y[i], t).unwrap();
        }
        write!(&mut s, "============================================================").unwrap();

        s
    }
}