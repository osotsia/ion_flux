use pyo3::prelude::*;
use numpy::{PyArray1, ToPyArray};
use super::{NativeResFn, NativeJacFn, NativeJvpFn, SolverConfig, Diagnostics};
use super::integrator::{step_bdf_vsvo, BdfHistory};
use super::linalg::NativeSparseLuSolver;

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
    pub constraints: Vec<f64>,
    pub p: Vec<f64>,
    pub spatial_diag: Vec<f64>,
    
    // Architectural State
    history: BdfHistory,
    lu_solver: NativeSparseLuSolver,
    jac_buffer: Vec<f64>,
    config: SolverConfig,
    diag: Diagnostics,
}

#[pymethods]
impl SolverHandle {
    #[new]
    pub fn new(lib_path: String, n: usize, bw: isize, y0: Vec<f64>, ydot0: Vec<f64>, id: Vec<f64>, constraints: Vec<f64>, p: Vec<f64>, spatial_diag: Vec<f64>, _debug: bool) -> PyResult<Self> {
        let lib = unsafe { libloading::Library::new(&lib_path).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))? };
        let res_fn: NativeResFn = unsafe { *lib.get(b"evaluate_residual\0").map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))? };
        let jac_fn: NativeJacFn = unsafe { *lib.get(b"evaluate_jacobian\0").map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))? };
        let jvp_fn: Option<NativeJvpFn> = unsafe { lib.get(b"evaluate_jvp\0").map(|s| *s).ok() };
        
        let history = BdfHistory::new(n);

        let handle = SolverHandle { 
            _lib: lib, res_fn, jac_fn, jvp_fn, n, bw, 
            t: 0.0, y: y0, ydot: ydot0, id, constraints, p, spatial_diag,
            history, lu_solver: NativeSparseLuSolver::new(n, bw), jac_buffer: vec![0.0; n * n],
            config: SolverConfig::default(), diag: Diagnostics::default(),
        };
        
        Ok(handle)
    }

    pub fn step(&mut self, dt: f64) -> PyResult<()> {
        self.step_with_history(dt, None)
    }

    pub fn calc_algebraic_roots(&mut self) -> PyResult<()> {
        let dt = 1e-11;
        self.step_with_history(dt, None)?;
        self.t -= dt; 
        
        // [CHANGED] Reset the BDF history to order 1 to prevent high-order extrapolation 
        // across a discontinuous parameter jump (e.g. Current to Voltage control).
        self.history.order = 1;
        self.history.k_used = 0;
        self.history.ns = 0;
        self.lu_solver.mark_stale();
        
        Ok(())
    }

    pub fn get_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> { 
        numpy::ndarray::Array1::from_vec(self.y.clone()).to_pyarray_bound(py) 
    }
    
    pub fn set_parameter(&mut self, idx: usize, val: f64) { 
        if idx < self.p.len() { self.p[idx] = val; } 
    }
    
    pub fn reach_steady_state(&mut self) -> PyResult<()> { 
        self.step(1000.0)
    }

    pub fn clone_state(&self) -> PyResult<(f64, Vec<f64>)> {
        Ok((self.t, self.y.clone()))
    }

    pub fn restore_state(&mut self, t: f64, y: Vec<f64>) -> PyResult<()> {
        self.t = t;
        self.y = y;
        self.lu_solver.mark_stale();
        Ok(())
    }

    pub fn get_jacobian<'py>(&mut self, py: Python<'py>, c_j: f64) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        unsafe { (self.jac_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), c_j, self.jac_buffer.as_mut_ptr()) };
        let jac_2d = numpy::ndarray::Array2::from_shape_vec((self.n, self.n), self.jac_buffer.clone())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(jac_2d.to_pyarray_bound(py))
    }
}

impl SolverHandle {
    pub fn step_with_history(&mut self, dt: f64, hist: Option<&mut Vec<(f64, Vec<f64>, Vec<f64>)>>) -> PyResult<()> {
        step_bdf_vsvo(
            self.n, self.bw, &mut self.y, &mut self.ydot, &self.p, &self.id, &self.constraints, &self.spatial_diag,
            dt, &mut self.history,
            self.res_fn, self.jac_fn, self.jvp_fn,
            &mut self.lu_solver, &mut self.jac_buffer, &self.config, &mut self.diag, hist, self.t
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        
        self.t += dt;
        Ok(())
    }
}

impl Drop for SolverHandle {
    fn drop(&mut self) {
        super::integrator::dump_diagnostics(&self.diag);
    }
}