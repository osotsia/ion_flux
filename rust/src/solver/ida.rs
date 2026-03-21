use pyo3::prelude::*;
use numpy::{PyArray2, ToPyArray};
use std::os::raw::c_double;

type NativeResFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *mut c_double);
type NativeJacFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, c_double, *mut c_double);

/// Solves a dense square linear system J * dx = b in-place using Gaussian elimination.
/// O(N^3) time complexity. J is expected in Column-Major layout.
fn solve_dense_system(n: usize, jac: &mut [f64], b: &mut [f64]) -> Result<(), String> {
    for k in 0..n {
        let pivot = jac[k * n + k];
        if pivot.abs() < 1e-14 { return Err("Singular Jacobian matrix.".to_string()); }

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
        if pivot.abs() < 1e-14 { return Err("Singular or ill-conditioned Banded Jacobian.".to_string()); }

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
    
    let n = y0_py.len();
    let n_steps = t_eval.len();
    let mut out_traj = vec![0.0; n_steps * n];

    let lib = unsafe { libloading::Library::new(lib_path).expect("Failed to load JIT shared library.") };
    let res_fn: libloading::Symbol<NativeResFn> = unsafe { lib.get(b"evaluate_residual\0").unwrap() };
    let jac_fn: libloading::Symbol<NativeJacFn> = unsafe { lib.get(b"evaluate_jacobian\0").unwrap() };

    let mut y = y0_py.clone();
    let mut ydot = ydot0_py.clone();
    let mut res = vec![0.0; n];
    let mut jac = vec![0.0; n * n];
    let mut dy = vec![0.0; n];

    // --- Consistent Initial Condition Evaluation ---
    for _iter in 0..20 {
        unsafe { res_fn(y.as_ptr(), ydot.as_ptr(), p_list.as_ptr(), res.as_mut_ptr()) };
        let max_res = res.iter().enumerate().filter(|(i, _)| id_py[*i] == 0.0).map(|(_, v)| v.abs()).fold(0.0, f64::max);
        if max_res < 1e-8 { break; }

        unsafe { jac_fn(y.as_ptr(), ydot.as_ptr(), p_list.as_ptr(), 0.0, jac.as_mut_ptr()) };

        for i in 0..n {
            dy[i] = -res[i];
            if id_py[i] == 1.0 {
                dy[i] = 0.0;
                for col in 0..n { jac[col * n + i] = if col == i { 1.0 } else { 0.0 }; }
            }
        }

        if bandwidth > 0 { solve_banded_system(n, bandwidth, &mut jac, &mut dy).expect("IC Init Failed"); }
        else { solve_dense_system(n, &mut jac, &mut dy).expect("IC Init Failed"); }

        for i in 0..n { y[i] += dy[i]; }
    }
    for i in 0..n { out_traj[i] = y[i]; }

    // --- Implicit Time Integration ---
    let micro_steps = 100;
    for step in 1..n_steps {
        let dt = (t_eval[step] - t_eval[step - 1]) / (micro_steps as f64);
        let c_j = 1.0 / dt;

        for _substep in 0..micro_steps {
            let y_prev = y.clone();
            let mut converged = false;

            for _iter in 0..20 {
                for i in 0..n { ydot[i] = if id_py[i] == 1.0 { (y[i] - y_prev[i]) / dt } else { 0.0 }; }
                unsafe { res_fn(y.as_ptr(), ydot.as_ptr(), p_list.as_ptr(), res.as_mut_ptr()) };

                let mut max_res = 0.0;
                for i in 0..n {
                    if res[i].abs() > max_res { max_res = res[i].abs(); }
                    dy[i] = -res[i];
                }

                if max_res < 1e-7 { converged = true; break; }

                unsafe { jac_fn(y.as_ptr(), ydot.as_ptr(), p_list.as_ptr(), c_j, jac.as_mut_ptr()) };
                
                if bandwidth > 0 { solve_banded_system(n, bandwidth, &mut jac, &mut dy).expect("Newton Failed"); }
                else { solve_dense_system(n, &mut jac, &mut dy).expect("Newton Failed"); }

                for i in 0..n { y[i] += dy[i]; }
            }
            if !converged { panic!("Newton method failed at t={}.", t_eval[step]); }
        }
        for i in 0..n { out_traj[step * n + i] = y[i]; }
    }

    Ok(numpy::ndarray::Array2::from_shape_vec((n_steps, n), out_traj).unwrap().to_pyarray_bound(py))
}