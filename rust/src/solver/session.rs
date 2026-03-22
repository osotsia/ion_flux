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
    pub y_prev: Vec<f64>,
    pub y_prev2: Vec<f64>,
    pub dt_prev: f64,
    pub order: usize,
}

#[pymethods]
impl SolverHandle {
    #[new]
    pub fn new(lib_path: String, n: usize, bw: isize, y0: Vec<f64>, ydot0: Vec<f64>, id: Vec<f64>, p: Vec<f64>) -> PyResult<Self> {
        let lib = unsafe { libloading::Library::new(&lib_path).expect("Failed to load JIT shared library.") };
        let res_fn: NativeResFn = unsafe { *lib.get::<NativeResFn>(b"evaluate_residual\0").unwrap() };
        let jac_fn: NativeJacFn = unsafe { *lib.get::<NativeJacFn>(b"evaluate_jacobian\0").unwrap() };
        
        // Correctly maps the FFI wrapper Symbol without dereferencing a mismatched fn pointer fallback
        let jvp_fn: Option<NativeJvpFn> = unsafe {
            lib.get::<NativeJvpFn>(b"evaluate_jvp\0").map(|sym| *sym).ok()
        };
        
        let y_prev = y0.clone();
        let y_prev2 = y0.clone();

        let mut handle = SolverHandle { 
            _lib: lib, res_fn, jac_fn, jvp_fn, n, bw, 
            t: 0.0, y: y0, ydot: ydot0, id, p,
            y_prev, y_prev2, dt_prev: 0.0, order: 1,
        };
        handle.initialize_ic()?;
        Ok(handle)
    }

    pub fn clone_state(&self) -> (f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64, usize) {
        (self.t, self.y.clone(), self.ydot.clone(), self.y_prev.clone(), self.y_prev2.clone(), self.dt_prev, self.order)
    }

    pub fn restore_state(&mut self, t: f64, y: Vec<f64>, ydot: Vec<f64>, y_prev: Vec<f64>, y_prev2: Vec<f64>, dt_prev: f64, order: usize) {
        self.t = t; self.y = y; self.ydot = ydot; self.y_prev = y_prev; self.y_prev2 = y_prev2; self.dt_prev = dt_prev; self.order = order;
    }

    pub fn step(&mut self, dt: f64) -> PyResult<()> {
        let integrator = BdfIntegrator::default();
        integrator.step(
            self.n, self.bw, &mut self.y, &mut self.ydot, &self.p, &self.id, dt, 
            self.res_fn, self.jac_fn, self.jvp_fn,
            &mut self.y_prev, &mut self.y_prev2, &mut self.dt_prev, &mut self.order
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        self.t += dt;
        Ok(())
    }

    pub fn reach_steady_state(&mut self) -> PyResult<()> {
        for i in 0..self.n { self.ydot[i] = 0.0; }
        
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

            if self.bw == -1 {
                let jvp = self.jvp_fn.expect("evaluate_jvp not found. Clear cache and recompile the JIT model.");
                let y_ptr = self.y.as_ptr();
                let ydot_ptr = self.ydot.as_ptr();
                let p_ptr = self.p.as_ptr();
                
                let jvp_closure = |v: &[f64], out: &mut [f64]| {
                    unsafe { jvp(y_ptr, ydot_ptr, p_ptr, 0.0, v.as_ptr(), out.as_mut_ptr()) };
                };
                let precond = |v: &[f64], out: &mut [f64]| {
                    for i in 0..self.n { out[i] = v[i] / (0.0 * self.id[i] + 1.0); }
                };
                solve_gmres(self.n, &mut dy, jvp_closure, precond).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            } else {
                unsafe { (self.jac_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), 0.0, jac.as_mut_ptr()) };
                if self.bw > 0 { solve_banded_system(self.n, self.bw as usize, &mut jac, &mut dy).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?; } 
                else { solve_dense_system(self.n, &mut jac, &mut dy).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?; }
            }
            for i in 0..self.n { self.y[i] += dy[i]; }
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
                        // Force structural identity row for differential states during IC solve
                        for i in 0..self.n { if *id_ptr.add(i) == 1.0 { out[i] = v[i]; } }
                    }
                };
                
                let precond = |v: &[f64], out: &mut [f64]| {
                    for i in 0..self.n { out[i] = v[i] / (0.0 * self.id[i] + 1.0); }
                };
                
                solve_gmres(self.n, &mut rhs, jvp_closure, precond).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                dy.copy_from_slice(&rhs);
            } else {
                unsafe { (self.jac_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), 0.0, jac.as_mut_ptr()) };
                for i in 0..self.n {
                    dy[i] = -res[i];
                    if self.id[i] == 1.0 {
                        dy[i] = 0.0;
                        for col in 0..self.n { jac[col * self.n + i] = if col == i { 1.0 } else { 0.0 }; }
                    }
                }
                if self.bw > 0 { solve_banded_system(self.n, self.bw as usize, &mut jac, &mut dy).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?; } 
                else { solve_dense_system(self.n, &mut jac, &mut dy).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?; }
            }
            for i in 0..self.n { self.y[i] += dy[i]; }
        }

        self.y_prev.copy_from_slice(&self.y);
        self.y_prev2.copy_from_slice(&self.y);
        self.dt_prev = 0.0;
        self.order = 1;
        Ok(())
    }
}