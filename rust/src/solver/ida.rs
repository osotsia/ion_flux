use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, ToPyArray};
use std::os::raw::c_double;
use rayon::prelude::*;

type NativeResFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *mut c_double);
type NativeJacFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, c_double, *mut c_double);
type NativeVjpFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *const c_double, *mut c_double, *mut c_double);

/// Solves a dense square linear system J * dx = b in-place using Gaussian elimination.
/// O(N^3) time complexity. J is expected in Column-Major layout.
fn solve_dense_system(n: usize, jac: &mut [f64], b: &mut [f64]) -> Result<(), String> {
    for k in 0..n {
        let mut max_val = 0.0;
        let mut pivot_row = k;
        for i in k..n {
            let val = jac[k * n + i].abs();
            if val > max_val {
                max_val = val;
                pivot_row = i;
            }
        }
        if max_val < 1e-25 { return Err("Singular Jacobian matrix.".to_string()); }
        
        if pivot_row != k {
            b.swap(k, pivot_row);
            for col in 0..n {
                let tmp = jac[col * n + k];
                jac[col * n + k] = jac[col * n + pivot_row];
                jac[col * n + pivot_row] = tmp;
            }
        }
        
        let pivot = jac[k * n + k];
        for i in (k + 1)..n {
            let factor = jac[k * n + i] / pivot;
            b[i] -= factor * b[k];
            for col in k..n {
                jac[col * n + i] -= factor * jac[col * n + k];
            }
        }
    }
    for i in (0..n).rev() {
        let mut sum = b[i];
        for col in (i + 1)..n { sum -= jac[col * n + i] * b[col]; }
        b[i] = sum / jac[i * n + i];
    }
    Ok(())
}

/// Solves a Banded linear system in-place to unlock massive Data Parallelism.
/// O(N * bw^2) time complexity. Operates on the full dense array but only within bounds.
fn solve_banded_system(n: usize, bw: usize, jac: &mut [f64], b: &mut [f64]) -> Result<(), String> {
    for k in 0..n {
        let pivot = jac[k * n + k];
        if pivot.abs() < 1e-25 { return Err("Singular or ill-conditioned Banded Jacobian.".to_string()); }

        let end_row = std::cmp::min(n, k + bw + 1);
        for i in (k + 1)..end_row {
            let factor = jac[k * n + i] / pivot;
            b[i] -= factor * b[k];

            let end_col = std::cmp::min(n, k + bw + 1);
            for col in k..end_col {
                jac[col * n + i] -= factor * jac[col * n + k];
            }
        }
    }
    for i in (0..n).rev() {
        let mut sum = b[i];
        let end_col = std::cmp::min(n, i + bw + 1);
        for col in (i + 1)..end_col { sum -= jac[col * n + i] * b[col]; }
        b[i] = sum / jac[i * n + i];
    }
    Ok(())
}

#[pyclass(unsendable)]
pub struct SolverHandle {
    _lib: libloading::Library,
    res_fn: NativeResFn,
    jac_fn: NativeJacFn,
    pub n: usize,
    pub bw: usize,
    pub t: f64,
    pub y: Vec<f64>,
    pub ydot: Vec<f64>,
    id: Vec<f64>,
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
        let mut t_local = 0.0;
        let mut sub_dt = dt.min(1e-3);
        
        // --- SCALAR FAST PATH ---
        // Bypasses dense vector allocations, bound-checked iteration, and matrix inversions for N=1 models
        if self.n == 1 {
            while t_local < dt {
                if t_local + sub_dt > dt { sub_dt = dt - t_local; }
                let c_j = 1.0 / sub_dt;
                
                let mut res = 0.0;
                let mut jac = 0.0;
                let y_prev = self.y[0];
                let mut converged = false;
                let mut iters = 0;
                
                for iter in 0..10 {
                    iters = iter;
                    self.ydot[0] = if self.id[0] == 1.0 { (self.y[0] - y_prev) / sub_dt } else { 0.0 };
                    unsafe { (self.res_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), &mut res) };
                    
                    if res.abs() < 1e-6 { 
                        converged = true; 
                        break; 
                    }
                    
                    unsafe { (self.jac_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), c_j, &mut jac) };
                    
                    if jac.abs() < 1e-25 { return Err(pyo3::exceptions::PyRuntimeError::new_err("Singular Jacobian matrix.")); }
                    
                    self.y[0] -= res / jac;
                }
                
                if converged {
                    t_local += sub_dt;
                    if iters < 4 { sub_dt *= 1.5; }
                } else {
                    self.y[0] = y_prev;
                    sub_dt *= 0.25;
                    if sub_dt < 1e-10 {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Newton method failed to converge at t={}", self.t + t_local)));
                    }
                }
            }
            self.t += dt;
            return Ok(());
        }

        // --- STANDARD VECTORIZED PATH ---
        while t_local < dt {
            if t_local + sub_dt > dt { sub_dt = dt - t_local; }
            let c_j = 1.0 / sub_dt;
            
            let mut res = vec![0.0; self.n];
            let mut jac = vec![0.0; self.n * self.n];
            let mut dy = vec![0.0; self.n];
            
            let y_prev = self.y.clone();
            let mut converged = false;
            let mut iters = 0;
            
            for iter in 0..10 {
                iters = iter;
                for i in 0..self.n { 
                    self.ydot[i] = if self.id[i] == 1.0 { (self.y[i] - y_prev[i]) / sub_dt } else { 0.0 }; 
                }
                unsafe { (self.res_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), res.as_mut_ptr()) };

                let mut max_res = 0.0;
                for i in 0..self.n {
                    if res[i].abs() > max_res { max_res = res[i].abs(); }
                    dy[i] = -res[i];
                }
                
                if max_res < 1e-6 { 
                    converged = true; 
                    break; 
                }

                unsafe { (self.jac_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), c_j, jac.as_mut_ptr()) };
                
                if self.bw > 0 { 
                    solve_banded_system(self.n, self.bw, &mut jac, &mut dy).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?; 
                } else { 
                    solve_dense_system(self.n, &mut jac, &mut dy).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?; 
                }

                for i in 0..self.n { self.y[i] += dy[i]; }
            }
            
            if converged {
                t_local += sub_dt;
                if iters < 4 { sub_dt *= 1.5; }
            } else {
                self.y = y_prev;
                sub_dt *= 0.25;
                if sub_dt < 1e-10 {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Newton method failed to converge at t={}", self.t + t_local)));
                }
            }
        }
        self.t += dt;
        Ok(())
    }

    pub fn reach_steady_state(&mut self) -> PyResult<()> {
        if self.n == 1 {
            self.ydot[0] = 0.0;
            let mut res = 0.0;
            let mut jac = 0.0;
            let mut converged = false;
            
            for _ in 0..100 {
                unsafe { (self.res_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), &mut res) };
                if res.abs() < 1e-8 {
                    converged = true;
                    break;
                }
                unsafe { (self.jac_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), 0.0, &mut jac) };
                if jac.abs() < 1e-25 { return Err(pyo3::exceptions::PyRuntimeError::new_err("Singular Jacobian matrix.")); }
                self.y[0] -= res / jac;
            }
            if !converged { return Err(pyo3::exceptions::PyRuntimeError::new_err("Steady state Newton failed to converge.")); }
            return Ok(());
        }

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
            if max_res < 1e-8 {
                converged = true;
                break;
            }
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
        if self.n == 1 {
            let mut res = 0.0;
            let mut jac = 0.0;
            for _ in 0..20 {
                unsafe { (self.res_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), &mut res) };
                if self.id[0] == 0.0 && res.abs() < 1e-8 { break; }
                
                unsafe { (self.jac_fn)(self.y.as_ptr(), self.ydot.as_ptr(), self.p.as_ptr(), 0.0, &mut jac) };
                
                if self.id[0] == 1.0 {
                    break; // Differential variable ICs remain unchanged.
                } else {
                    if jac.abs() < 1e-25 { return Err(pyo3::exceptions::PyRuntimeError::new_err("Singular Jacobian matrix during IC init.")); }
                    self.y[0] -= res / jac;
                }
            }
            return Ok(());
        }

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
            if self.bw > 0 { 
                solve_banded_system(self.n, self.bw, &mut jac, &mut dy).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?; 
            } else { 
                solve_dense_system(self.n, &mut jac, &mut dy).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?; 
            }
            for i in 0..self.n { self.y[i] += dy[i]; }
        }
        Ok(())
    }
}

#[pyfunction]
pub fn solve_ida_native<'py>(
    py: Python<'py>,
    lib_path: String,
    y0_py: Vec<f64>,
    ydot0_py: Vec<f64>,
    id_py: Vec<f64>,
    p_list: Vec<f64>,
    t_eval: Vec<f64>,
    bandwidth: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut handle = SolverHandle::new(lib_path, y0_py.len(), bandwidth, y0_py, ydot0_py, id_py, p_list)?;
    let mut out_traj = vec![0.0; t_eval.len() * handle.n];
    
    for i in 0..handle.n { out_traj[i] = handle.y[i]; }
    for step in 1..t_eval.len() {
        let dt = t_eval[step] - t_eval[step - 1];
        handle.step(dt)?;
        for i in 0..handle.n { out_traj[step * handle.n + i] = handle.y[i]; }
    }
    Ok(numpy::ndarray::Array2::from_shape_vec((t_eval.len(), handle.n), out_traj).unwrap().to_pyarray_bound(py))
}

#[pyfunction]
pub fn solve_batch_native<'py>(
    py: Python<'py>,
    lib_path: String,
    y0: Vec<f64>,
    ydot0: Vec<f64>,
    id: Vec<f64>,
    p_batch: Vec<Vec<f64>>,
    t_eval: Vec<f64>,
    bandwidth: usize,
) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
    let results: Result<Vec<Vec<f64>>, String> = p_batch.par_iter().map(|p| {
        let mut handle = SolverHandle::new(lib_path.clone(), y0.len(), bandwidth, y0.clone(), ydot0.clone(), id.clone(), p.clone())
            .map_err(|e| e.to_string())?;
            
        let mut out_traj = vec![0.0; t_eval.len() * handle.n];
        for i in 0..handle.n { out_traj[i] = handle.y[i]; }
        
        for step in 1..t_eval.len() {
            let dt = t_eval[step] - t_eval[step - 1];
            handle.step(dt).map_err(|e| e.to_string())?;
            for i in 0..handle.n { out_traj[step * handle.n + i] = handle.y[i]; }
        }
        Ok(out_traj)
    }).collect();

    let unwrapped_results = results.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let mut py_results = Vec::new();
    for res in unwrapped_results {
        py_results.push(numpy::ndarray::Array2::from_shape_vec((t_eval.len(), y0.len()), res).unwrap().to_pyarray_bound(py));
    }
    Ok(py_results)
}

#[pyfunction]
pub fn discrete_adjoint_native<'py>(
    py: Python<'py>,
    lib_path: String,
    y_traj: Vec<Vec<f64>>,
    t_eval: Vec<f64>,
    id_arr: Vec<f64>,
    p_list: Vec<f64>,
    dl_dy: Vec<Vec<f64>>
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let n_steps = y_traj.len();
    let n = y_traj[0].len();
    let n_params = p_list.len();
    let mut p_grad = vec![0.0; n_params];

    let lib = unsafe { libloading::Library::new(&lib_path).expect("Failed to load JIT library") };
    let jac_fn: NativeJacFn = unsafe { *lib.get::<NativeJacFn>(b"evaluate_jacobian\0").unwrap() };
    let vjp_fn: NativeVjpFn = unsafe { *lib.get::<NativeVjpFn>(b"evaluate_vjp\0").unwrap() };

    let mut lambda = vec![0.0; n];
    
    for step in (1..n_steps).rev() {
        let dt = t_eval[step] - t_eval[step - 1];
        let c_j = 1.0 / dt;
        let y = &y_traj[step];
        
        let mut ydot = vec![0.0; n];
        for i in 0..n {
            if id_arr[i] == 1.0 { ydot[i] = (y_traj[step][i] - y_traj[step - 1][i]) / dt; }
        }
        
        // 1. Evaluate Forward Jacobian J = dF/dy + c_j * dF/dydot
        let mut jac = vec![0.0; n * n];
        unsafe { jac_fn(y.as_ptr(), ydot.as_ptr(), p_list.as_ptr(), c_j, jac.as_mut_ptr()) };
        
        // 2. Transpose Jacobian (J^T) since SUNDIALS emits Column-Major (jac[col * n + row])
        let mut jac_t = vec![0.0; n * n];
        for row in 0..n {
            for col in 0..n { jac_t[row * n + col] = jac[col * n + row]; }
        }
        
        // 3. Form Adjoint RHS = -dL/dy + c_j * diag(id_arr) * lambda_prev
        let mut rhs = vec![0.0; n];
        for i in 0..n { rhs[i] = -dl_dy[step][i] + lambda[i] * id_arr[i] * c_j; }
        
        // 4. Implicitly solve J^T * lambda_next = RHS
        solve_dense_system(n, &mut jac_t, &mut rhs).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        lambda = rhs;
        
        // 5. Native Reverse-Mode Enzyme Vector-Jacobian Product (VJP) mapping
        let mut dp_out = vec![0.0; n_params];
        let mut dy_out = vec![0.0; n];
        unsafe { vjp_fn(y.as_ptr(), ydot.as_ptr(), p_list.as_ptr(), lambda.as_ptr(), dp_out.as_mut_ptr(), dy_out.as_mut_ptr()) };
        
        // 6. Accumulate exact continuous parameter sensitivities
        for p_idx in 0..n_params { p_grad[p_idx] += dp_out[p_idx]; }
    }
    
    Ok(numpy::ndarray::Array1::from_vec(p_grad).to_pyarray_bound(py))
}