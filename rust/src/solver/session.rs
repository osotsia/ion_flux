use pyo3::prelude::*;
use numpy::{PyArray1, ToPyArray};
use super::{NativeResFn, NativeObsFn, NativeJacFn, NativeJvpFn, SolverConfig, Diagnostics};
use super::integrator::{step_bdf_vsvo, BdfHistory};
use super::linalg::NativeSparseLuSolver;

#[pyclass(unsendable)]
pub struct SolverHandle {
    _lib: libloading::Library,
    res_fn: NativeResFn,
    obs_fn: Option<NativeObsFn>,
    jac_fn: NativeJacFn,
    jvp_fn: Option<NativeJvpFn>,
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
    
    history: BdfHistory,
    lu_solver: NativeSparseLuSolver,
    jac_buffer: Vec<f64>,
    config: SolverConfig,
    diag: Diagnostics,
}

#[pymethods]
impl SolverHandle {
    #[new]
    pub fn new(lib_path: String, n: usize, bw: isize, y0: Vec<f64>, ydot0: Vec<f64>, id: Vec<f64>, constraints: Vec<f64>, p: Vec<f64>, m: Vec<f64>, spatial_diag: Vec<f64>, max_steps: Vec<f64>, n_obs: usize, _debug: bool) -> PyResult<Self> {
        let lib = unsafe { libloading::Library::new(&lib_path).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))? };
        let res_fn: NativeResFn = unsafe { *lib.get(b"evaluate_residual\0").map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))? };
        let obs_fn: Option<NativeObsFn> = unsafe { lib.get(b"evaluate_observables\0").map(|s| *s).ok() };
        let jac_fn: NativeJacFn = unsafe { *lib.get(b"evaluate_jacobian\0").map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))? };
        let jvp_fn: Option<NativeJvpFn> = unsafe { lib.get(b"evaluate_jvp\0").map(|s| *s).ok() };
        
        let history = BdfHistory::new(n);

        let mut handle = SolverHandle { 
            _lib: lib, res_fn, obs_fn, jac_fn, jvp_fn, n, bw, n_obs,
            t: 0.0, y: y0, ydot: ydot0, id, constraints, p, m, spatial_diag, max_steps,
            history, lu_solver: NativeSparseLuSolver::new(n, bw), jac_buffer: vec![0.0; n * n],
            config: SolverConfig::default(), diag: Diagnostics::default(),
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

            let mut jac = vec![0.0; self.n * self.n];
            unsafe { (self.jac_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), self.m.as_ptr(), 0.0, jac.as_mut_ptr()); }

            if self.bw == -1 {
                // Fallback under-relaxation for Matrix-Free GMRES initialization
                for i in 0..self.n {
                    if self.id[i] < 0.5 { 
                        let diag = jac[i * self.n + i];
                        if diag.abs() > 1e-12 {
                            let mut step = -res[i] / diag;
                            if self.max_steps[i] > 0.0 && step.abs() > self.max_steps[i] {
                                step = step.signum() * self.max_steps[i];
                            }
                            self.y[i] += step * 0.5; 
                        }
                    }
                }
            } else {
                // Exact Sparse LU fully coupled resolution
                for i in 0..self.n {
                    if self.id[i] > 0.5 {
                        res[i] = 0.0;
                        for j in 0..self.n {
                            jac[j * self.n + i] = if i == j { 1.0 } else { 0.0 };
                        }
                    }
                }

                let mut dy = res.clone();
                for i in 0..self.n { dy[i] = -dy[i]; }

                if let Err(_) = self.lu_solver.factorize(&jac, &mut self.diag) { break; }
                if let Err(_) = self.lu_solver.solve(&mut dy, &mut self.diag) { break; }

                let mut max_step = 0.0;
                for i in 0..self.n {
                    if self.id[i] < 0.5 {
                        let mut step = dy[i];
                        if self.max_steps[i] > 0.0 && step.abs() > self.max_steps[i] {
                            step = step.signum() * self.max_steps[i];
                        }
                        self.y[i] += step;
                        if step.abs() > max_step { max_step = step.abs(); }
                    }
                }
                if max_step < 1e-14 { break; } // Stalled
            }
        }
        
        self.history.order = 1;
        self.history.k_used = 0;
        self.history.ns = 0;
        self.lu_solver.mark_stale();
        self.diag.accepted_steps = 0; // Force safe BDF cold start
        
        Ok(())
    }

    pub fn get_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> { 
        numpy::ndarray::Array1::from_vec(self.y.clone()).to_pyarray_bound(py) 
    }

    pub fn get_observables_py<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut obs = vec![0.0; self.n_obs];
        self.get_observables(&mut obs)?;
        Ok(numpy::ndarray::Array1::from_vec(obs).to_pyarray_bound(py))
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
        self.diag.accepted_steps = 0; // Force a safe BDF cold start to rebuild history phi polynomials!
        self.history.order = 1;       // Flush corrupted VSVO polynomial predictions
        self.history.k_used = 0;
        self.history.ns = 0;
        Ok(())
    }

    pub fn get_jacobian<'py>(&mut self, py: Python<'py>, c_j: f64) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        unsafe { (self.jac_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), self.m.as_ptr(), c_j, self.jac_buffer.as_mut_ptr()) };
        let jac_2d = numpy::ndarray::Array2::from_shape_vec((self.n, self.n), self.jac_buffer.clone())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(jac_2d.to_pyarray_bound(py))
    }
}

impl SolverHandle {
    pub fn step_with_history(&mut self, dt: f64, hist: Option<&mut Vec<(f64, Vec<f64>, Vec<f64>)>>) -> PyResult<()> {
        step_bdf_vsvo(
            self.n, self.bw, &mut self.y, &mut self.ydot, &self.p, &self.m, &self.id, &self.constraints, &self.spatial_diag, &self.max_steps,
            dt, &mut self.history,
            self.res_fn, self.jac_fn, self.jvp_fn,
            &mut self.lu_solver, &mut self.jac_buffer, &self.config, &mut self.diag, hist, self.t
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