use pyo3::prelude::*;
use numpy::{PyArray1, ToPyArray};
use super::{NativeJacFn, NativeVjpFn};
use super::linalg::{solve_dense_system, solve_banded_system, solve_gmres};

#[pyfunction]
pub fn discrete_adjoint_native<'py>(
    py: Python<'py>,
    lib_path: String,
    y_traj: Vec<Vec<f64>>,
    t_eval: Vec<f64>,
    id_arr: Vec<f64>,
    p_list: Vec<f64>,
    dl_dy: Vec<Vec<f64>>,
    bandwidth: isize, // Upgraded to isize to allow -1 (Matrix-Free Adjoint)
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let n_steps = y_traj.len();
    let n = y_traj[0].len();
    let n_params = p_list.len();
    let mut p_grad = vec![0.0; n_params];

    let lib = unsafe { libloading::Library::new(&lib_path).expect("Failed to load JIT library") };
    let jac_fn: Option<NativeJacFn> = unsafe { lib.get::<NativeJacFn>(b"evaluate_jacobian\0").map(|sym| *sym).ok() };
    let vjp_fn: NativeVjpFn = unsafe { *lib.get::<NativeVjpFn>(b"evaluate_vjp\0").expect("evaluate_vjp missing from binary.") };

    let mut lambda = vec![0.0; n];
    
    // Implicit Euler Backward Adjoint Step
    for step in (1..n_steps).rev() {
        let dt = t_eval[step] - t_eval[step - 1];
        let c_j = 1.0 / dt;
        let y = &y_traj[step];
        
        let mut ydot = vec![0.0; n];
        for i in 0..n {
            if id_arr[i] == 1.0 { ydot[i] = (y_traj[step][i] - y_traj[step - 1][i]) / dt; }
        }
        
        let mut rhs = vec![0.0; n];
        for i in 0..n { rhs[i] = -dl_dy[step][i] + lambda[i] * id_arr[i] * c_j; }
        
        // BUG 5 FINALLY FIXED: Matrix-Free Adjoint Solve bypassing O(N^2) memory entirely
        if bandwidth == -1 {
            let y_ptr = y.as_ptr();
            let ydot_ptr = ydot.as_ptr();
            let p_ptr = p_list.as_ptr();
            
            // J^T v = (dF/dy)^T v + c_j * (dF/dydot)^T v 
            let jvp_T = |v: &[f64], out: &mut [f64]| {
                let mut dp_dummy = vec![0.0; n_params];
                let mut dy_out = vec![0.0; n];
                let mut dydot_out = vec![0.0; n];
                unsafe { vjp_fn(y_ptr, ydot_ptr, p_ptr, v.as_ptr(), dp_dummy.as_mut_ptr(), dy_out.as_mut_ptr(), dydot_out.as_mut_ptr()) };
                for i in 0..n { out[i] = dy_out[i] + c_j * dydot_out[i]; }
            };
            
            let precond = |v: &[f64], out: &mut [f64]| {
                for i in 0..n { out[i] = v[i] / (c_j * id_arr[i] + 1.0); }
            };
            
            solve_gmres(n, &mut rhs, jvp_T, precond).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            
        } else {
            let mut jac = vec![0.0; n * n];
            let jac_fn_ptr = jac_fn.expect("Dense/Banded adjoints require evaluate_jacobian. Recompile model.");
            unsafe { jac_fn_ptr(y.as_ptr(), ydot.as_ptr(), p_list.as_ptr(), c_j, jac.as_mut_ptr()) };
            
            let mut jac_t = vec![0.0; n * n];
            for row in 0..n {
                for col in 0..n { jac_t[row * n + col] = jac[col * n + row]; }
            }
            
            if bandwidth > 0 { solve_banded_system(n, bandwidth as usize, &mut jac_t, &mut rhs).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?; } 
            else { solve_dense_system(n, &mut jac_t, &mut rhs).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?; }
        }
        
        lambda = rhs;
        
        // Native Reverse-Mode Enzyme Vector-Jacobian Product applied globally
        let mut dp_out = vec![0.0; n_params];
        let mut dy_out = vec![0.0; n];
        let mut dydot_out = vec![0.0; n];
        unsafe { vjp_fn(y.as_ptr(), ydot.as_ptr(), p_list.as_ptr(), lambda.as_ptr(), dp_out.as_mut_ptr(), dy_out.as_mut_ptr(), dydot_out.as_mut_ptr()) };
        
        for p_idx in 0..n_params { p_grad[p_idx] += dp_out[p_idx]; }
    }
    
    Ok(numpy::ndarray::Array1::from_vec(p_grad).to_pyarray_bound(py))
}