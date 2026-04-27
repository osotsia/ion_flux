use pyo3::prelude::*;
use numpy::{PyArray1, ToPyArray};
use std::os::raw::c_int;
use super::{NativeJacSparseFn, NativeVjpFn, Diagnostics};
use super::linalg::{NativeSparseLuSolver, solve_gmres};

#[pyfunction]
pub fn discrete_adjoint_native<'py>(
    py: Python<'py>, lib_path: String, y_traj: Vec<Vec<f64>>, ydot_traj: Vec<Vec<f64>>,
    t_eval: Vec<f64>, id_arr: Vec<f64>, p_traj: Vec<Vec<f64>>, m_list: Vec<f64>, dl_dy: Vec<Vec<f64>>, bandwidth: isize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let n_steps = y_traj.len();
    let n = y_traj[0].len();
    let n_params = p_traj[0].len();
    let mut p_grad = vec![0.0; n_params];

    let lib = unsafe { libloading::Library::new(&lib_path).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))? };
    let jac_sparse_fn: Option<NativeJacSparseFn> = unsafe { lib.get::<NativeJacSparseFn>(b"evaluate_jacobian_sparse\0").map(|sym| *sym).ok() };
    let vjp_fn: NativeVjpFn = unsafe { *lib.get::<NativeVjpFn>(b"evaluate_vjp\0").map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))? };

    let mut lambda = vec![0.0; n];
    let mut prev_dydot_vjp = vec![0.0; n];
    let mut prev_c_j = 0.0;
    let mut diag = Diagnostics::default();
    let mut solver = NativeSparseLuSolver::new(n, bandwidth);
    
    let mut jac_rows_buf = vec![0; n * 50];
    let mut jac_cols_buf = vec![0; n * 50];
    let mut jac_vals_buf = vec![0.0; n * 50];
    
    for step in (1..n_steps).rev() {
        let dt = t_eval[step] - t_eval[step - 1];
        if dt <= 1e-12 { continue; }
        
        let c_j = 1.0 / dt; 
        let y = &y_traj[step];
        let ydot = &ydot_traj[step];
        let p_list = &p_traj[step];
        
        let mut rhs = vec![0.0; n];
        
        for i in 0..n { rhs[i] = -dl_dy[step][i] + prev_dydot_vjp[i] * prev_c_j; }
        
        if bandwidth == -1 {
            let y_ptr = y.as_ptr(); let ydot_ptr = ydot.as_ptr(); let p_ptr = p_list.as_ptr(); let m_ptr = m_list.as_ptr();
            let jvp_t = |v: &[f64], out: &mut [f64]| {
                let mut dp_dummy = vec![0.0; n_params];
                let mut dy_out = vec![0.0; n];
                let mut dydot_out = vec![0.0; n];
                unsafe { vjp_fn(y_ptr, ydot_ptr, p_ptr, m_ptr, v.as_ptr(), dp_dummy.as_mut_ptr(), dy_out.as_mut_ptr(), dydot_out.as_mut_ptr()) };
                for i in 0..n { out[i] = dy_out[i] + c_j * dydot_out[i]; }
            };
            let precond = |v: &[f64], out: &mut[f64]| { for i in 0..n { out[i] = v[i] / (c_j * id_arr[i] + 1.0); } };
            solve_gmres(n, &mut rhs, jvp_t, precond).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        } else {
            let jac_fn_ptr = jac_sparse_fn.expect("evaluate_jacobian_sparse required for Dense/Banded adjoints.");
            
            let mut nnz: c_int = 0;
            unsafe {
                jac_fn_ptr(
                    y.as_ptr(), ydot.as_ptr(), p_list.as_ptr(), m_list.as_ptr(), c_j,
                    jac_rows_buf.as_mut_ptr(), jac_cols_buf.as_mut_ptr(), jac_vals_buf.as_mut_ptr(), &mut nnz
                );
            }
            
            solver.triplets.clear();
            for i in 0..(nnz as usize) {
                // To transpose the sparse matrix implicitly for the adjoint solve,
                // we push (col, row, val) instead of (row, col, val)
                solver.triplets.push((jac_cols_buf[i] as usize, jac_rows_buf[i] as usize, jac_vals_buf[i]));
            }
            
            solver.factorize_from_triplets(&mut diag).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            solver.solve(&mut rhs, &mut diag).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        }
        
        lambda = rhs;
        
        let mut dp_out = vec![0.0; n_params];
        let mut dy_out = vec![0.0; n];
        let mut dydot_out = vec![0.0; n];
        unsafe { vjp_fn(y.as_ptr(), ydot.as_ptr(), p_list.as_ptr(), m_list.as_ptr(), lambda.as_ptr(), dp_out.as_mut_ptr(), dy_out.as_mut_ptr(), dydot_out.as_mut_ptr()) };
        for p_idx in 0..n_params { p_grad[p_idx] += dp_out[p_idx]; }
        
        prev_dydot_vjp = dydot_out;
        prev_c_j = c_j;
    }
    Ok(numpy::ndarray::Array1::from_vec(p_grad).to_pyarray(py))
}