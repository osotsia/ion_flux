use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyArray2, PyReadwriteArray1, ToPyArray};
use crate::ffi::SafeEvaluator;
use crate::stepper::problem::{Problem, Callbacks, CprData, SolverConfig};
use crate::stepper::state::{SolverInstance, Checkpoint};
use crate::api::{NativeSolverCrash, crash_report_to_pydict, diagnostics_to_pydict};

#[pyclass(unsendable)]
pub struct SolverHandle {
    _lib: libloading::Library,
    pub prob: Problem,
    pub instance: SolverInstance,
}

#[pymethods]
impl SolverHandle {
    #[new]
    #[pyo3(signature = (
        lib_path, n, bw, y0, ydot0, id, constraints, p, m, spatial_diag, max_steps, n_obs,
        cpr_seeds, cpr_ptrs, cpr_rows, cpr_cols, cpr_dense, is_realtime=false,
        rel_tol=None, abs_tol=None, max_newton_iters=None, record_trace=false
    ))]
    pub fn new(
        lib_path: String, n: usize, bw: isize, y0: Vec<f64>, ydot0: Vec<f64>, id: Vec<f64>, 
        constraints: Vec<f64>, p: Vec<f64>, m: Vec<f64>, spatial_diag: Vec<f64>, max_steps: Vec<f64>, 
        n_obs: usize,
        cpr_seeds: Vec<Vec<f64>>, cpr_ptrs: Vec<usize>, cpr_rows: Vec<usize>, cpr_cols: Vec<usize>, cpr_dense: Vec<usize>,
        is_realtime: bool, rel_tol: Option<f64>, abs_tol: Option<f64>, max_newton_iters: Option<usize>, record_trace: bool
    ) -> PyResult<Self> {
        let lib = unsafe { libloading::Library::new(&lib_path).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))? };
        let fns = Callbacks {
            res_fn: unsafe {
                *lib.get(b"evaluate_residual\0")
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Missing evaluate_residual: {e}")))?
            },
            obs_fn: unsafe { lib.get(b"evaluate_observables\0").map(|s| *s).ok() },
            jvp_fn: unsafe { lib.get(b"evaluate_jvp\0").map(|s| *s).ok() },
            vjp_fn: unsafe { lib.get(b"evaluate_vjp\0").map(|s| *s).ok() },
            set_threads_fn: unsafe { lib.get(b"set_spatial_threads\0").map(|s| *s).ok() },
        };
        
        let cpr = CprData { color_seeds: cpr_seeds, color_ptrs: cpr_ptrs, color_rows: cpr_rows, color_cols: cpr_cols, dense_rows: cpr_dense };
        let mut config = if is_realtime { SolverConfig::realtime() } else { SolverConfig::default() };
        if let Some(rt) = rel_tol { config.rel_tol = rt; }
        if let Some(at) = abs_tol { config.abs_tol = at; }
        if let Some(mi) = max_newton_iters { config.max_newton_iters = mi; }
        config.record_trace = record_trace && !is_realtime;
        let prob = Problem { n, bw, n_obs, id, constraints, m, spatial_diag, max_steps, cpr, config, fns };

        let mut instance = SolverInstance::new(n, bw, y0, ydot0, p);
        instance.calc_algebraic_roots(&prob)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            
        Ok(SolverHandle { _lib: lib, prob, instance })
    }

    pub fn step(&mut self, py: Python<'_>, dt: f64) -> PyResult<()> {
        self.instance.step(&self.prob, dt, None).map_err(|e| {
            NativeSolverCrash::new_err(crash_report_to_pydict(py, &e).unbind())
        })
    }

    pub fn step_history<'py>(&mut self, py: Python<'py>, dt: f64) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let mut history = vec![];
        self.instance.step(&self.prob, dt, Some(&mut history)).map_err(|e| {
            NativeSolverCrash::new_err(crash_report_to_pydict(py, &e).unbind())
        })?;
        
        let h_len = history.len();
        let mut micro_t = vec![0.0; h_len];
        let mut micro_y = vec![0.0; h_len * self.prob.n];
        let mut micro_ydot = vec![0.0; h_len * self.prob.n];
        
        for (i, (t, y, ydot)) in history.into_iter().enumerate() {
            micro_t[i] = t;
            for j in 0..self.prob.n {
                micro_y[i * self.prob.n + j] = y[j];
                micro_ydot[i * self.prob.n + j] = ydot[j];
            }
        }
        
        let arr_y = numpy::ndarray::Array2::from_shape_vec((h_len, self.prob.n), micro_y)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let arr_ydot = numpy::ndarray::Array2::from_shape_vec((h_len, self.prob.n), micro_ydot)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok((
            numpy::ndarray::Array1::from_vec(micro_t).to_pyarray(py),
            arr_y.to_pyarray(py),
            arr_ydot.to_pyarray(py),
        ))
    }

    pub fn get_time(&self) -> f64 { 
        self.instance.state.t 
    }

    pub fn calc_algebraic_roots(&mut self) -> PyResult<()> {
        self.instance.calc_algebraic_roots(&self.prob)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    pub fn get_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> { 
        numpy::ndarray::Array1::from_vec(self.instance.state.y.clone()).to_pyarray(py) 
    }

    pub fn get_diagnostics<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.instance.diag.peak_rss_bytes = crate::diagnostics::get_peak_rss_bytes();
        Ok(diagnostics_to_pydict(py, &self.instance.diag))
    }

    pub fn read_state_into(&self, mut target: PyReadwriteArray1<f64>) -> PyResult<()> {
        let slice = target.as_slice_mut().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Target buffer must be contiguous: {}", e))
        })?;
        if slice.len() != self.prob.n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Target buffer length ({}) must match state size ({})",
                slice.len(), self.prob.n
            )));
        }
        slice.copy_from_slice(&self.instance.state.y);
        Ok(())
    }

    pub fn get_observables_py<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut obs = vec![0.0; self.prob.n_obs];
        let evaluator = SafeEvaluator::new(self.prob.fns, self.prob.n, self.instance.state.p.len(), self.prob.n_obs);
        evaluator.evaluate_observables(&self.instance.state.y, &self.instance.state.ydot, &self.instance.state.p, &self.prob.m, &mut obs);
        Ok(numpy::ndarray::Array1::from_vec(obs).to_pyarray(py))
    }

    pub fn read_observables_into(&mut self, mut target: PyReadwriteArray1<f64>) -> PyResult<()> {
        let slice = target.as_slice_mut().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Target buffer must be contiguous: {}", e))
        })?;
        if slice.len() != self.prob.n_obs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Target buffer length ({}) must match observables size ({})",
                slice.len(), self.prob.n_obs
            )));
        }
        let evaluator = SafeEvaluator::new(self.prob.fns, self.prob.n, self.instance.state.p.len(), self.prob.n_obs);
        if self.prob.fns.obs_fn.is_some() {
            evaluator.evaluate_observables(&self.instance.state.y, &self.instance.state.ydot, &self.instance.state.p, &self.prob.m, slice);
        } else {
            slice.fill(0.0);
        }
        Ok(())
    }
    
    pub fn set_parameter(&mut self, idx: usize, val: f64) { 
        if idx < self.instance.state.p.len() { self.instance.state.p[idx] = val; } 
    }
    
    pub fn reach_steady_state(&mut self, py: Python<'_>) -> PyResult<()> { self.step(py, 1000.0) }
    pub fn clone_state(&self) -> PyResult<(f64, Vec<f64>, Vec<f64>)> { 
        let ckpt = self.instance.checkpoint();
        Ok((ckpt.t, ckpt.y, ckpt.ydot))
    }
    pub fn restore_state(&mut self, t: f64, y: Vec<f64>, ydot: Vec<f64>) -> PyResult<()> {
        self.instance.rollback(&Checkpoint { t, y, ydot });
        Ok(())
    }
}
