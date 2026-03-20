use pyo3::prelude::*;
use numpy::{PyArray2, ToPyArray};
use std::os::raw::c_double;

type NativeResFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *mut c_double);
type NativeJacFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, c_double, *mut c_double);

/// Solves a dense square linear system J * dx = b in-place using Gaussian elimination with partial pivoting.
/// J is expected in Column-Major layout: J[col * n + row].
fn solve_linear_system(n: usize, jac: &mut [f64], b: &mut [f64]) -> Result<(), String> {
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
        
        if max_val < 1e-14 {
            return Err("Jacobian matrix is singular or ill-conditioned.".to_string());
        }

        b.swap(k, pivot_row);
        for col in 0..n {
            let tmp = jac[col * n + k];
            jac[col * n + k] = jac[col * n + pivot_row];
            jac[col * n + pivot_row] = tmp;
        }

        for i in (k + 1)..n {
            let factor = jac[k * n + i] / jac[k * n + k];
            b[i] -= factor * b[k];
            for col in k..n {
                jac[col * n + i] -= factor * jac[col * n + k];
            }
        }
    }

    for i in (0..n).rev() {
        let mut sum = b[i];
        for col in (i + 1)..n {
            sum -= jac[col * n + i] * b[col];
        }
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

    // 1. Consistent Initial Condition Loop
    // Solves for exact algebraic roots at t=0 before taking a time step.
    for _iter in 0..20 {
        unsafe { res_fn(y.as_ptr(), ydot.as_ptr(), p_list.as_ptr(), res.as_mut_ptr()) };
        
        let mut max_res = 0.0;
        for i in 0..n {
            // Only algebraic residuals matter for this correction step
            if id_py[i] == 0.0 && res[i].abs() > max_res {
                max_res = res[i].abs();
            }
        }
        if max_res < 1e-8 { break; }

        unsafe { jac_fn(y.as_ptr(), ydot.as_ptr(), p_list.as_ptr(), 0.0, jac.as_mut_ptr()) };

        // Mask differential equations: force dy = 0 so they do not update
        for i in 0..n {
            dy[i] = -res[i];
            if id_py[i] == 1.0 {
                dy[i] = 0.0;
                for col in 0..n { jac[col * n + i] = if col == i { 1.0 } else { 0.0 }; }
            }
        }

        solve_linear_system(n, &mut jac, &mut dy).expect("IC Initialization Failed");
        for i in 0..n { y[i] += dy[i]; }
    }

    // Record t=0
    for i in 0..n { out_traj[i] = y[i]; }

    // 2. Integration Loop (BDF1 / Implicit Euler)
    // Uses internal micro-stepping to maintain accuracy without complex adaptive logic.
    let micro_steps = 100;

    for step in 1..n_steps {
        let dt_macro = t_eval[step] - t_eval[step - 1];
        let dt = dt_macro / (micro_steps as f64);
        let c_j = 1.0 / dt;

        for _substep in 0..micro_steps {
            let y_prev = y.clone();
            let mut converged = false;

            for _iter in 0..20 {
                for i in 0..n {
                    ydot[i] = if id_py[i] == 1.0 { (y[i] - y_prev[i]) / dt } else { 0.0 };
                }

                unsafe { res_fn(y.as_ptr(), ydot.as_ptr(), p_list.as_ptr(), res.as_mut_ptr()) };

                let mut max_res = 0.0;
                for i in 0..n {
                    if res[i].abs() > max_res { max_res = res[i].abs(); }
                    dy[i] = -res[i];
                }

                if max_res < 1e-7 {
                    converged = true;
                    break;
                }

                unsafe { jac_fn(y.as_ptr(), ydot.as_ptr(), p_list.as_ptr(), c_j, jac.as_mut_ptr()) };
                solve_linear_system(n, &mut jac, &mut dy).expect("Newton iteration failed");

                for i in 0..n { y[i] += dy[i]; }
            }

            if !converged {
                panic!("Newton method failed to converge at t={}.", t_eval[step]);
            }
        }

        for i in 0..n {
            out_traj[step * n + i] = y[i];
        }
    }

    let ndarray = numpy::ndarray::Array2::from_shape_vec((n_steps, n), out_traj).unwrap();
    Ok(ndarray.to_pyarray_bound(py))
}
