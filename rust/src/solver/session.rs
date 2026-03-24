use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, ToPyArray};
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
    pub debug: bool, // Track internally
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

    /// Evaluates a zero-dt solve over the system. Freezes differential states (id == 1.0) 
    /// while snapping algebraic variables to the new equilibrium manifold.
    pub fn calc_algebraic_roots(&mut self) -> PyResult<()> {
        let mut res = vec![0.0; self.n];
        let mut jac = vec![0.0; self.n * self.n];
        let mut dy = vec![0.0; self.n];

        for _ in 0..20 {
            unsafe { (self.res_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), res.as_mut_ptr()) };
            
            let max_res = res.iter().enumerate()
                .filter(|(i, _)| self.id[*i] == 0.0)
                .map(|(_, v)| v.abs())
                .fold(0.0, f64::max);
                
            if max_res < 1e-8 { break; }

            if self.bw == -1 {
                let mut rhs = vec![0.0; self.n];
                for i in 0..self.n { rhs[i] = if self.id[i] == 1.0 { 0.0 } else { -res[i] }; }
                
                let jvp = self.jvp_fn.expect("evaluate_jvp not found. Clear cache and recompile the JIT model.");
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
                    for i in 0..self.n { out[i] = v[i] / (0.0 * self.id[i] + self.spatial_diag[i] + 1.0); }
                };
                
                solve_gmres(self.n, &mut rhs, jvp_closure, precond).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                dy.copy_from_slice(&rhs);
            } else {
                unsafe { (self.jac_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), 0.0, jac.as_mut_ptr()) };
                for i in 0..self.n {
                    dy[i] = -res[i];
                    if self.id[i] == 1.0 {
                        dy[i] = 0.0; // Force differential variables to freeze
                        for col in 0..self.n { jac[col * self.n + i] = if col == i { 1.0 } else { 0.0 }; }
                    }
                }
                if self.bw > 0 { solve_banded_system(self.n, self.bw as usize, &mut jac, &mut dy).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?; } 
                else { solve_dense_system(self.n, &mut jac, &mut dy).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?; }
            }
            
            // Apply the Newton update with globalization (Line Search)
            let mut alpha = 1.0;
            let mut step_accepted = false;
            let mut y_trial = vec![0.0; self.n];
            let mut res_trial = vec![0.0; self.n];
            
            for _ in 0..5 {
                for i in 0..self.n { y_trial[i] = self.y[i] + alpha * dy[i]; }
                unsafe { (self.res_fn)(y_trial.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), res_trial.as_mut_ptr()) };
                
                let max_res_trial = res_trial.iter().enumerate()
                    .filter(|(i, _)| self.id[*i] == 0.0)
                    .map(|(_, v)| v.abs())
                    .fold(0.0, f64::max);
                    
                if max_res_trial <= max_res * (1.0 - 1e-4 * alpha) || max_res_trial.is_nan() == false && max_res < 1e-6 {
                    self.y.copy_from_slice(&y_trial);
                    step_accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            
            if !step_accepted {
                for i in 0..self.n { self.y[i] += alpha * dy[i]; }
            }
        }

        // Restore integration history context based on the new equilibrium manifold
        self.y_prev.copy_from_slice(&self.y);
        self.y_prev2.copy_from_slice(&self.y);
        self.dt_prev = 0.0;
        self.order = 1; // Explicitly drop to BDF1 across the discontinuity shock
        Ok(())
    }

    pub fn reach_steady_state(&mut self) -> PyResult<()> {
        for i in 0..self.n { self.ydot[i] = 0.0; }
        
        let mut res = vec![0.0; self.n];
        let mut jac = vec![0.0; self.n * self.n];
        let mut dy = vec![0.0; self.n];
        let mut weights = vec![0.0; self.n];
        
        for i in 0..self.n {
            weights[i] = 1.0 / (1e-6 * self.y[i].abs() + 1e-8);
        }
        
        let wrms_norm = |v: &[f64], w: &[f64]| -> f64 {
            let mut sum = 0.0;
            for i in 0..self.n {
                let scaled = v[i] * w[i];
                if scaled.is_nan() || scaled.is_infinite() { return f64::INFINITY; }
                sum += scaled * scaled;
            }
            (sum / (self.n as f64)).sqrt()
        };

        let mut converged = false;
        
        for _ in 0..100 {
            unsafe { (self.res_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), res.as_mut_ptr()) };
            
            let mut max_abs_res = 0.0;
            for i in 0..self.n {
                if res[i].abs() > max_abs_res { max_abs_res = res[i].abs(); }
                dy[i] = -res[i];
            }
            if max_abs_res < 1e-12 { converged = true; break; }

            if self.bw == -1 {
                let jvp = self.jvp_fn.expect("evaluate_jvp not found. Clear cache and recompile the JIT model.");
                let y_ptr = self.y.as_ptr();
                let ydot_ptr = self.ydot.as_ptr();
                let p_ptr = self.p.as_ptr();
                
                let jvp_closure = |v: &[f64], out: &mut [f64]| {
                    unsafe { jvp(y_ptr, ydot_ptr, p_ptr, 0.0, v.as_ptr(), out.as_mut_ptr()) };
                };
                let precond = |v: &[f64], out: &mut [f64]| {
                    for i in 0..self.n { out[i] = v[i] / (0.0 * self.id[i] + self.spatial_diag[i] + 1.0); }
                };
                solve_gmres(self.n, &mut dy, jvp_closure, precond).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            } else {
                unsafe { (self.jac_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), 0.0, jac.as_mut_ptr()) };
                if self.bw > 0 { solve_banded_system(self.n, self.bw as usize, &mut jac, &mut dy).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?; } 
                else { solve_dense_system(self.n, &mut jac, &mut dy).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?; }
            }
            
            let dy_norm = wrms_norm(&dy, &weights);
            let f_norm = wrms_norm(&res, &weights);
            
            let mut alpha = 1.0;
            let mut step_accepted = false;
            let mut y_trial = vec![0.0; self.n];
            let mut res_trial = vec![0.0; self.n];
            
            for _ in 0..5 {
                for i in 0..self.n { y_trial[i] = self.y[i] + alpha * dy[i]; }
                unsafe { (self.res_fn)(y_trial.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), res_trial.as_mut_ptr()) };
                
                let f_norm_trial = wrms_norm(&res_trial, &weights);
                if f_norm_trial <= f_norm * (1.0 - 1e-4 * alpha) || dy_norm < 0.1 {
                    self.y.copy_from_slice(&y_trial);
                    step_accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            
            if !step_accepted {
                for i in 0..self.n { self.y[i] += alpha * dy[i]; }
            }
            
            if dy_norm < 1.0 { converged = true; break; }
        }
        
        if !converged { return Err(pyo3::exceptions::PyRuntimeError::new_err("Steady state Newton failed to converge.")); }
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

impl SolverHandle {
    pub fn step_with_history(&mut self, dt: f64, history: Option<&mut Vec<(f64, Vec<f64>, Vec<f64>)>>) -> PyResult<()> {
        let mut integrator = BdfIntegrator::default();
        integrator.debug = self.debug; // Inject observability flag
        
        integrator.step(
            self.n, self.bw, &mut self.y, &mut self.ydot, &self.p, &self.id, &self.spatial_diag, dt, 
            self.res_fn, self.jac_fn, self.jvp_fn,
            &mut self.y_prev, &mut self.y_prev2, &mut self.dt_prev, &mut self.order,
            history, self.t
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        self.t += dt;
        Ok(())
    }
}