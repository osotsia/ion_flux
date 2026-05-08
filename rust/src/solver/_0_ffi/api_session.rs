use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, ToPyArray};
use crate::solver::shared::problem::{Problem, Callbacks, CprData, SolverConfig};
use crate::solver::shared::workspace::Workspace;
use crate::solver::_2_stepper::bdf;
use crate::solver::_3_nonlinear::newton;

#[pyclass(unsendable)]
pub struct SolverHandle {
    _lib: libloading::Library,
    pub prob: Problem,
    pub wk: Workspace,
}

#[pymethods]
impl SolverHandle {
    #[new]
    pub fn new(
        lib_path: String, n: usize, bw: isize, y0: Vec<f64>, ydot0: Vec<f64>, id: Vec<f64>, 
        constraints: Vec<f64>, p: Vec<f64>, m: Vec<f64>, spatial_diag: Vec<f64>, max_steps: Vec<f64>, 
        n_obs: usize, debug: bool, 
        cpr_seeds: Vec<Vec<f64>>, cpr_ptrs: Vec<usize>, cpr_rows: Vec<usize>, cpr_cols: Vec<usize>, cpr_dense: Vec<usize>
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
        let prob = Problem { n, bw, n_obs, id, constraints, m, spatial_diag, max_steps, cpr, config: SolverConfig::default(), fns };
        let mut wk = Workspace::new(n, bw, y0, ydot0, p);
        
        newton::calc_algebraic_roots(&prob, &mut wk).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(SolverHandle { _lib: lib, prob, wk })
    }

    pub fn step(&mut self, dt: f64) -> PyResult<()> {
        bdf::step(&self.prob, &mut self.wk, dt, None).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    pub fn step_history<'py>(&mut self, py: Python<'py>, dt: f64) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let mut history = vec![];
        bdf::step(&self.prob, &mut self.wk, dt, Some(&mut history)).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        
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
        self.wk.t 
    }

    pub fn calc_algebraic_roots(&mut self) -> PyResult<()> {
        newton::calc_algebraic_roots(&self.prob, &mut self.wk).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    pub fn get_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> { 
        numpy::ndarray::Array1::from_vec(self.wk.y.clone()).to_pyarray(py) 
    }

    pub fn get_observables_py<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut obs = vec![0.0; self.prob.n_obs];
        if let Some(obs_fn) = self.prob.fns.obs_fn {
            unsafe { obs_fn(self.wk.y.as_ptr(), self.wk.ydot.as_ptr(), self.wk.p.as_ptr(), self.prob.m.as_ptr(), obs.as_mut_ptr()); }
        }
        Ok(numpy::ndarray::Array1::from_vec(obs).to_pyarray(py))
    }
    
    pub fn set_parameter(&mut self, idx: usize, val: f64) { 
        if idx < self.wk.p.len() { self.wk.p[idx] = val; } 
    }
    
    pub fn reach_steady_state(&mut self) -> PyResult<()> { self.step(1000.0) }
    pub fn clone_state(&self) -> PyResult<(f64, Vec<f64>, Vec<f64>)> { Ok(self.wk.clone_state()) }
    pub fn restore_state(&mut self, t: f64, y: Vec<f64>, ydot: Vec<f64>) -> PyResult<()> { self.wk.restore_state(t, y, ydot); Ok(()) }
}