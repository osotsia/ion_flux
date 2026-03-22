use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, ToPyArray};
use super::{NativeResFn, NativeJacFn, NativeVjpFn};
use super::integrator::Bdf1Integrator;
use super::linalg::{solve_dense_system, solve_banded_system};

#[pyclass(unsendable)]
pub struct SolverHandle {
    _lib: libloading::Library, // Holds the compiled SO open in memory
    res_fn: NativeResFn,
    jac_fn: NativeJacFn,
    pub n: usize,
    pub bw: usize,
    pub t: f64,
    pub y: Vec<f64>,
    pub ydot: Vec<f64>,
    pub id: Vec<f64>,
    pub p: Vec<f64>,
}

#[pymethods]
impl SolverHandle {
    #[new]
    pub fn new(lib_path: String, n: usize, bw: usize, y0: Vec<f64>, ydot0: Vec<f64>, id: Vec<f64>, p: Vec<f64>) -> PyResult<Self> {
        let lib = unsafe { libloading::Library::new(&lib_path).expect("Failed to load JIT shared library.") };
        let res_fn: NativeResFn = unsafe { *lib.get::<NativeResFn>(b"evaluate_residual\0").unwrap() };
        let jac_fn: NativeJacFn = unsafe { *lib.get::<NativeJacFn>(b"evaluate_jacobian\0").unwrap() };
        
        let mut handle = SolverHandle { _lib: lib, res_fn, jac_fn, n, bw, t: 0.0, y: y0, ydot: ydot0, id, p };
        handle.initialize_ic()?;
        Ok(handle)
    }

    pub fn clone_state(&self) -> (f64, Vec<f64>, Vec<f64>) {
        (self.t, self.y.clone(), self.ydot.clone())
    }

    pub fn restore_state(&mut self, t: f64, y: Vec<f64>, ydot: Vec<f64>) {
        self.t = t;
        self.y = y;
        self.ydot = ydot;
    }

    pub fn step(&mut self, dt: f64) -> PyResult<()> {
        let integrator = Bdf1Integrator::default();
        integrator.step(
            self.n, self.bw, &mut self.y, &mut self.ydot, &self.p, &self.id, 
            dt, self.res_fn, self.jac_fn
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        
        self.t += dt;
        Ok(())
    }

    pub fn reach_steady_state(&mut self) -> PyResult<()> {
        for i in 0..self.n { self.ydot[i] = 0.0; } // BUG 10 FIX: True steady-state forces ydot=0
        
        let mut res = vec![0.0; self.n];
        let mut jac = vec![0.0; self.n * self.n];
        let mut dy = vec![0.0; self.n];
        
        let mut converged = false;
        for _ in 0..100 {
            unsafe { (self.res_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), res.as_mut_ptr()) };
            let mut max_res = 0.0;
            for i in 0..self.n {
                if res[i].abs() > max_res { max_res = res[i].abs(); }
                dy[i] = -res[i];
            }
            if max_res < 1e-8 { converged = true; break; }

            unsafe { (self.jac_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), 0.0, jac.as_mut_ptr()) };
            
            if self.bw > 0 { 
                solve_banded_system(self.n, self.bw, &mut jac, &mut dy).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?; 
            } else { 
                solve_dense_system(self.n, &mut jac, &mut dy).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?; 
            }
            for i in 0..self.n { self.y[i] += dy[i]; }
        }
        
        if !converged { return Err(pyo3::exceptions::PyRuntimeError::new_err("Steady state Newton failed to converge.")); }
        Ok(())
    }

    pub fn set_parameter(&mut self, idx: usize, val: f64) {
        if idx < self.p.len() { self.p[idx] = val; }
    }

    pub fn get_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        numpy::ndarray::Array1::from_vec(self.y.clone()).to_pyarray_bound(py)
    }

    pub fn get_jacobian<'py>(&self, py: Python<'py>, c_j: f64) -> Bound<'py, PyArray2<f64>> {
        let mut jac = vec![0.0; self.n * self.n];
        unsafe { (self.jac_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), c_j, jac.as_mut_ptr()) };
        
        let mut jac_rm = vec![0.0; self.n * self.n];
        for col in 0..self.n {
            for row in 0..self.n { jac_rm[row * self.n + col] = jac[col * self.n + row]; }
        }
        numpy::ndarray::Array2::from_shape_vec((self.n, self.n), jac_rm).unwrap().to_pyarray_bound(py)
    }
}

impl SolverHandle {
    fn initialize_ic(&mut self) -> PyResult<()> {
        let mut res = vec![0.0; self.n];
        let mut jac = vec![0.0; self.n * self.n];
        let mut dy = vec![0.0; self.n];

        for _ in 0..20 {
            unsafe { (self.res_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), res.as_mut_ptr()) };
            let max_res = res.iter().enumerate().filter(|(i, _)| self.id[*i] == 0.0).map(|(_, v)| v.abs()).fold(0.0, f64::max);
            if max_res < 1e-8 { break; }

            unsafe { (self.jac_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), 0.0, jac.as_mut_ptr()) };

            for i in 0..self.n {
                dy[i] = -res[i];
                if self.id[i] == 1.0 {
                    dy[i] = 0.0;
                    for col in 0..self.n { jac[col * self.n + i] = if col == i { 1.0 } else { 0.0 }; }
                }
            }
            if self.bw > 0 { solve_banded_system(self.n, self.bw, &mut jac, &mut dy).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?; } 
            else { solve_dense_system(self.n, &mut jac, &mut dy).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?; }
            
            for i in 0..self.n { self.y[i] += dy[i]; }
        }
        Ok(())
    }
}