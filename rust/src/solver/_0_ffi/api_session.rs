use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadwriteArray1, ToPyArray};
use crate::solver::shared::problem::{Problem, Callbacks, CprData, SolverConfig};
use crate::stepper::state::{SystemState, StepScratchpad};
use crate::stepper::history::BdfHistory;
use crate::linear::solver::LinearSolver;
use crate::solver::shared::diagnostics::Diagnostics;
use crate::stepper::bdf;
use crate::nonlinear::algebraic_roots;
use crate::NativeSolverCrash;
use crate::solver::_0_ffi::crash_report_to_pydict;

#[pyclass(unsendable)]
pub struct SolverHandle {
    _lib: libloading::Library,
    pub prob: Problem,
    pub state: SystemState,
    pub history: BdfHistory,
    pub scratch: StepScratchpad,
    pub linear_solver: LinearSolver,
    pub diag: Diagnostics,
}

#[pymethods]
impl SolverHandle {
    #[new]
    #[pyo3(signature = (
        lib_path, n, bw, y0, ydot0, id, constraints, p, m, spatial_diag, max_steps, n_obs, debug, 
        cpr_seeds, cpr_ptrs, cpr_rows, cpr_cols, cpr_dense, is_realtime=false
    ))]
    pub fn new(
        lib_path: String, n: usize, bw: isize, y0: Vec<f64>, ydot0: Vec<f64>, id: Vec<f64>, 
        constraints: Vec<f64>, p: Vec<f64>, m: Vec<f64>, spatial_diag: Vec<f64>, max_steps: Vec<f64>, 
        n_obs: usize, debug: bool, 
        cpr_seeds: Vec<Vec<f64>>, cpr_ptrs: Vec<usize>, cpr_rows: Vec<usize>, cpr_cols: Vec<usize>, cpr_dense: Vec<usize>,
        is_realtime: bool
    ) -> PyResult<Self> {
        let _ = debug;
        let lib = unsafe { libloading::Library::new(&lib_path).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))? };
        let fns = Callbacks {
            res_fn: unsafe { *lib.get(b"evaluate_residual\0").unwrap() },
            obs_fn: unsafe { lib.get(b"evaluate_observables\0").map(|s| *s).ok() },
            jvp_fn: unsafe { lib.get(b"evaluate_jvp\0").map(|s| *s).ok() },
            vjp_fn: unsafe { lib.get(b"evaluate_vjp\0").map(|s| *s).ok() },
            set_threads_fn: unsafe { lib.get(b"set_spatial_threads\0").map(|s| *s).ok() },
        };
        
        let cpr = CprData { color_seeds: cpr_seeds, color_ptrs: cpr_ptrs, color_rows: cpr_rows, color_cols: cpr_cols, dense_rows: cpr_dense };
        let config = if is_realtime { SolverConfig::realtime() } else { SolverConfig::default() };
        let prob = Problem { n, bw, n_obs, id, constraints, m, spatial_diag, max_steps, cpr, config, fns };

        let mut state = SystemState::new(y0, ydot0, p);
        let mut history = BdfHistory::new(n);
        let mut scratch = StepScratchpad::new(n);
        let mut linear_solver = LinearSolver::new(n, bw);
        let mut diag = Diagnostics::new(n);
        
        algebraic_roots::calc_algebraic_roots(&prob, &mut state, &mut history, &mut scratch, &mut linear_solver, &mut diag)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            
        Ok(SolverHandle { _lib: lib, prob, state, history, scratch, linear_solver, diag })
    }

    pub fn step(&mut self, py: Python<'_>, dt: f64) -> PyResult<()> {
        bdf::step(&self.prob, &mut self.state, &mut self.history, &mut self.scratch, &mut self.linear_solver, &mut self.diag, dt, None).map_err(|e| {
            NativeSolverCrash::new_err(crash_report_to_pydict(py, &e).unbind())
        })
    }

    pub fn step_history<'py>(&mut self, py: Python<'py>, dt: f64) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let mut history = vec![];
        bdf::step(&self.prob, &mut self.state, &mut self.history, &mut self.scratch, &mut self.linear_solver, &mut self.diag, dt, Some(&mut history)).map_err(|e| {
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
        
        Ok((
            numpy::ndarray::Array1::from_vec(micro_t).to_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec((h_len, self.prob.n), micro_y).unwrap().to_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec((h_len, self.prob.n), micro_ydot).unwrap().to_pyarray(py)
        ))
    }

    pub fn get_time(&self) -> f64 { 
        self.state.t 
    }

    pub fn calc_algebraic_roots(&mut self) -> PyResult<()> {
        algebraic_roots::calc_algebraic_roots(&self.prob, &mut self.state, &mut self.history, &mut self.scratch, &mut self.linear_solver, &mut self.diag)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    pub fn get_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> { 
        numpy::ndarray::Array1::from_vec(self.state.y.clone()).to_pyarray(py) 
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
        slice.copy_from_slice(&self.state.y);
        Ok(())
    }

    pub fn get_observables_py<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut obs = vec![0.0; self.prob.n_obs];
        if let Some(obs_fn) = self.prob.fns.obs_fn {
            unsafe { obs_fn(self.state.y.as_ptr(), self.state.ydot.as_ptr(), self.state.p.as_ptr(), self.prob.m.as_ptr(), obs.as_mut_ptr()); }
        }
        Ok(numpy::ndarray::Array1::from_vec(obs).to_pyarray(py))
    }

    pub fn read_observables_into(&mut self, mut target: PyReadwriteArray1<f64>) -> PyResult<()> {
        let slice = target.as_slice_mut().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Target buffer must be contiguous: {}", e))
        })?;
        if slice.len() != self.prob.n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Target buffer length ({}) must match observables size ({})",
                slice.len(), self.prob.n
            )));
        }
        if let Some(obs_fn) = self.prob.fns.obs_fn {
            unsafe {
                obs_fn(self.state.y.as_ptr(), self.state.ydot.as_ptr(), self.state.p.as_ptr(), self.prob.m.as_ptr(), slice.as_mut_ptr());
            }
        } else {
            slice.fill(0.0);
        }
        Ok(())
    }
    
    pub fn set_parameter(&mut self, idx: usize, val: f64) { 
        if idx < self.state.p.len() { self.state.p[idx] = val; } 
    }
    
    pub fn reach_steady_state(&mut self, py: Python<'_>) -> PyResult<()> { self.step(py, 1000.0) }
    pub fn clone_state(&self) -> PyResult<(f64, Vec<f64>, Vec<f64>)> { Ok(self.state.clone_tuple()) }
    pub fn restore_state(&mut self, t: f64, y: Vec<f64>, ydot: Vec<f64>) -> PyResult<()> {
        self.state.restore(t, y, ydot);
        self.linear_solver.mark_stale();
        self.diag.accepted_steps = 0;
        self.history.restore_state();
        Ok(())
    }
}
