use pyo3::prelude::*;
use numpy::{PyArray1, ToPyArray};
use super::{NativeResFn, NativeObsFn, NativeJacSparseFn, NativeJvpFn, NativeVjpFn, NativeSetThreadsFn, SolverConfig, Diagnostics};
use super::integrator::{step_bdf_vsvo, BdfHistory};
use super::linalg::NativeSparseLuSolver;
use std::os::raw::{c_double, c_int};

#[pyclass(unsendable)]
pub struct SolverHandle {
    _lib: libloading::Library,
    res_fn: NativeResFn,
    obs_fn: Option<NativeObsFn>,
    jac_sparse_fn: NativeJacSparseFn,
    jvp_fn: Option<NativeJvpFn>,
    vjp_fn: Option<NativeVjpFn>,
    set_threads_fn: Option<NativeSetThreadsFn>,
    pub n: usize,
    pub bw: isize,
    pub n_obs: usize,
    pub t: f64,
    pub y: Vec<f64>,
    pub ydot: Vec<f64>,
    pub id: Vec<f64>,
    pub constraints: Vec<f64>,
    pub p: Vec<f64>,
    pub m: Vec<f64>,
    pub spatial_diag: Vec<f64>,
    pub max_steps: Vec<f64>,
    pub cpr: super::CprData,
    
    pub jac_rows_buf: Vec<i32>,
    pub jac_cols_buf: Vec<i32>,
    pub jac_vals_buf: Vec<f64>,
    
    history: BdfHistory,
    lu_solver: NativeSparseLuSolver,
    config: SolverConfig,
    diag: Diagnostics,
}

#[pymethods]
impl SolverHandle {
    #[new]
    pub fn new(
        lib_path: String, n: usize, bw: isize, y0: Vec<f64>, ydot0: Vec<f64>, id: Vec<f64>, 
        constraints: Vec<f64>, p: Vec<f64>, m: Vec<f64>, spatial_diag: Vec<f64>, max_steps: Vec<f64>, 
        n_obs: usize, _debug: bool, 
        cpr_seeds: Vec<Vec<f64>>, cpr_ptrs: Vec<usize>, cpr_rows: Vec<usize>, cpr_cols: Vec<usize>, cpr_dense: Vec<usize>
    ) -> PyResult<Self> {
        let lib = unsafe { libloading::Library::new(&lib_path).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))? };
        let res_fn: NativeResFn = unsafe { *lib.get(b"evaluate_residual\0").map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))? };
        let obs_fn: Option<NativeObsFn> = unsafe { lib.get(b"evaluate_observables\0").map(|s| *s).ok() };
        let jac_sparse_fn: NativeJacSparseFn = unsafe { *lib.get(b"evaluate_jacobian_sparse\0").map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))? };
        let jvp_fn: Option<NativeJvpFn> = unsafe { lib.get(b"evaluate_jvp\0").map(|s| *s).ok() };
        let vjp_fn: Option<NativeVjpFn> = unsafe { lib.get(b"evaluate_vjp\0").map(|s| *s).ok() };
        let set_threads_fn: Option<NativeSetThreadsFn> = unsafe { lib.get(b"set_spatial_threads\0").map(|s| *s).ok() };
        
        let history = BdfHistory::new(n);

        let diag = Diagnostics::default();
        let lu_solver = NativeSparseLuSolver::new(n, bw);
        
        let cpr = super::CprData { 
            color_seeds: cpr_seeds, 
            color_ptrs: cpr_ptrs, 
            color_rows: cpr_rows, 
            color_cols: cpr_cols, 
            dense_rows: cpr_dense 
        };
        
        let mut handle = SolverHandle { 
            _lib: lib, res_fn, obs_fn, jac_sparse_fn, jvp_fn, vjp_fn, set_threads_fn, n, bw, n_obs,
            t: 0.0, y: y0, ydot: ydot0, id, constraints, p, m, spatial_diag, max_steps, cpr,
            jac_rows_buf: vec![0; n * 50], jac_cols_buf: vec![0; n * 50], jac_vals_buf: vec![0.0; n * 50],
            history, lu_solver,
            config: SolverConfig::default(), diag,
        };
        
        handle.calc_algebraic_roots()?;
        Ok(handle)
    }

    pub fn step(&mut self, dt: f64) -> PyResult<()> {
        self.step_with_history(dt, None)
    }

    pub fn calc_algebraic_roots(&mut self) -> PyResult<()> {
        let max_iters = 50;
        for _iter in 0..max_iters {
            let mut res = vec![0.0; self.n];
            unsafe { (self.res_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), self.m.as_ptr(), res.as_mut_ptr()); }

            let mut max_res = 0.0_f64;
            for i in 0..self.n {
                if self.id[i] < 0.5 && res[i].abs() > max_res { max_res = res[i].abs(); }
            }
            if max_res < 1e-8 { break; }

            crate::solver::newton::assemble_jacobian_triplets(
                self.n, &self.y, &self.ydot, &self.p, &self.m, 0.0,
                self.jac_sparse_fn, self.jvp_fn, self.vjp_fn,
                &mut self.lu_solver, &mut self.jac_rows_buf, &mut self.jac_cols_buf, &mut self.jac_vals_buf,
                &self.cpr
            );

            if self.bw == -1 {
                for &(r, c, val) in &self.lu_solver.triplets {
                    if r == c && self.id[r] < 0.5 { 
                        if val.abs() > 1e-12 {
                            let mut step = -res[r] / val;
                            if self.max_steps[r] > 0.0 && step.abs() > self.max_steps[r] {
                                step = step.signum() * self.max_steps[r];
                            }
                            self.y[r] += step * 0.8; 
                        }
                    }
                }
            } else {
                for i in 0..self.n {
                    if self.id[i] > 0.5 {
                        res[i] = 0.0;
                        self.lu_solver.triplets.push((i, i, 1.0));
                    }
                }

                let mut dy = res.clone();
                for i in 0..self.n { dy[i] = -dy[i]; }

                if let Err(_) = self.lu_solver.factorize_from_triplets(&mut self.diag) { break; }
                if let Err(_) = self.lu_solver.solve(&mut dy, &mut self.diag) { break; }

                let mut max_step = 0.0;
                for i in 0..self.n {
                    if self.id[i] < 0.5 {
                        let mut step = dy[i];
                        if self.max_steps[i] > 0.0 && step.abs() > self.max_steps[i] {
                            step = step.signum() * self.max_steps[i];
                        }
                        self.y[i] += step * 0.8; 
                        if step.abs() > max_step { max_step = step.abs(); }
                    }
                }
                if max_step < 1e-14 { break; } 
            }
        }
        
        self.history.order = 1;
        self.history.k_used = 0;
        self.history.ns = 0;
        self.lu_solver.mark_stale();
        self.diag.accepted_steps = 0; 
        
        Ok(())
    }

    pub fn get_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> { 
        numpy::ndarray::Array1::from_vec(self.y.clone()).to_pyarray(py) 
    }

    pub fn get_observables_py<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut obs = vec![0.0; self.n_obs];
        self.get_observables(&mut obs)?;
        Ok(numpy::ndarray::Array1::from_vec(obs).to_pyarray(py))
    }
    
    pub fn set_parameter(&mut self, idx: usize, val: f64) { 
        if idx < self.p.len() { self.p[idx] = val; } 
    }
    
    pub fn reach_steady_state(&mut self) -> PyResult<()> { 
        self.step(1000.0)
    }

    pub fn clone_state(&self) -> PyResult<(f64, Vec<f64>, Vec<f64>)> {
        Ok((self.t, self.y.clone(), self.ydot.clone()))
    }

    pub fn restore_state(&mut self, t: f64, y: Vec<f64>, ydot: Vec<f64>) -> PyResult<()> {
        self.t = t;
        self.y = y;
        self.ydot = ydot;
        self.lu_solver.mark_stale();
        
        self.diag.accepted_steps = 0; 
        self.history.order = 1;       
        self.history.k_used = 0;
        self.history.ns = 0;
        
        self.history.h = 0.0;
        self.history.h_used = 0.0;
        self.history.c_j = 0.0;
        self.history.c_j_old = 0.0;
        for j in 0..6 {
            self.history.phi[j].fill(0.0);
        }
        
        Ok(())
    }
}

impl SolverHandle {
    pub fn set_spatial_threads(&self, num_threads: i32) {
        if let Some(f) = self.set_threads_fn {
            unsafe { f(num_threads) };
        }
    }    
    pub fn step_with_history(&mut self, dt: f64, hist: Option<&mut Vec<(f64, Vec<f64>, Vec<f64>)>>) -> PyResult<()> {
        step_bdf_vsvo(
            self.n, self.bw, &mut self.y, &mut self.ydot, &self.p, &self.m, &self.id, &self.constraints, &self.spatial_diag, &self.max_steps,
            dt, &mut self.history,
            self.res_fn, self.jac_sparse_fn, self.jvp_fn, self.vjp_fn,
            &mut self.lu_solver, &mut self.jac_rows_buf, &mut self.jac_cols_buf, &mut self.jac_vals_buf, &self.config, &mut self.diag, &self.cpr,
            hist, self.t
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        
        self.t += dt;
        Ok(())
    }

    pub fn get_observables(&mut self, obs: &mut [f64]) -> PyResult<()> {
        if let Some(f) = self.obs_fn {
            unsafe { f(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), self.m.as_ptr(), obs.as_mut_ptr()); }
        }
        Ok(())
    }
}

impl Drop for SolverHandle {
    fn drop(&mut self) {
        super::integrator::dump_diagnostics(&self.diag);
    }
}