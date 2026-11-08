use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use crate::solver::shared::callbacks::{NativeJvpFn, NativeVjpFn};
use crate::solver::shared::problem::CprData;
use crate::solver::_4_linear::sparse_lu::NativeSparseLuSolver;
use crate::solver::_4_linear::gmres::solve_gmres;
use crate::solver::shared::diagnostics::Diagnostics;

#[pyfunction]
#[pyo3(signature = (lib_path, y_traj, ydot_traj, t_eval, id_arr, p_traj, m_list, dl_dy, bandwidth, cpr_seeds, cpr_ptrs, cpr_rows, cpr_cols, cpr_dense, event_trigger_idx=-1))]
pub fn discrete_adjoint_native<'py>(
    py: Python<'py>, lib_path: String, 
    y_traj: PyReadonlyArray2<f64>, ydot_traj: PyReadonlyArray2<f64>,
    t_eval: PyReadonlyArray1<f64>, id_arr: PyReadonlyArray1<f64>, 
    p_traj: PyReadonlyArray2<f64>, m_list: PyReadonlyArray1<f64>, 
    dl_dy: PyReadonlyArray2<f64>, bandwidth: isize,
    cpr_seeds: Vec<Vec<f64>>, cpr_ptrs: Vec<usize>, cpr_rows: Vec<usize>, cpr_cols: Vec<usize>, cpr_dense: Vec<usize>,
    event_trigger_idx: isize
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    
    let y_arr = y_traj.as_array();
    let ydot_arr = ydot_traj.as_array();
    let t_arr = t_eval.as_array();
    let id_slice = id_arr.as_slice().expect("id_arr must be contiguous");
    let p_arr = p_traj.as_array();
    let m_slice = m_list.as_slice().expect("m_list must be contiguous");
    let dl_dy_arr = dl_dy.as_array();

    let n_steps = y_arr.nrows();
    let n = y_arr.ncols();
    let n_params = p_arr.ncols();
    let mut p_grad = vec![0.0; n_params];

    let lib = unsafe { libloading::Library::new(&lib_path).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))? };
    let jvp_fn: NativeJvpFn = unsafe { *lib.get(b"evaluate_jvp\0").unwrap() };
    let vjp_fn: NativeVjpFn = unsafe { *lib.get(b"evaluate_vjp\0").unwrap() };

    let cpr = CprData { color_seeds: cpr_seeds, color_ptrs: cpr_ptrs, color_rows: cpr_rows, color_cols: cpr_cols, dense_rows: cpr_dense };
    let mut prev_dydot_vjp = vec![0.0; n];
    let mut prev_c_j = 0.0;
    let mut prev_factorized_cj = 0.0;
    let mut diag = Diagnostics::new(n);
    let mut solver = NativeSparseLuSolver::new(n, bandwidth);

    // Pre-allocated scratchpad buffers reused across all adjoint steps (zero hot-loop allocations)
    let mut rhs = vec![0.0; n];
    let mut rhs_orig = vec![0.0; n];
    let mut delta_lambda = vec![0.0; n];
    let mut jvp_out = vec![0.0; n];
    let mut dp_out = vec![0.0; n_params];
    let mut dy_out = vec![0.0; n];
    let mut dydot_out = vec![0.0; n];
    let mut lambda_vjp = vec![0.0; n];
    let mut vjp_dp = vec![0.0; n_params];
    let mut vjp_dy = vec![0.0; n];
    let mut vjp_dydot = vec![0.0; n];
    let mut gmres_dp = vec![0.0; n_params];
    let mut gmres_dy = vec![0.0; n];
    let mut gmres_dydot = vec![0.0; n];

    for step in (1..n_steps).rev() {
        let dt = t_arr[step] - t_arr[step - 1];
        if dt <= 1e-12 { continue; }

        let c_j = 1.0 / dt;
        let y = y_arr.row(step).to_slice().expect("y_traj must be contiguous");
        let ydot = ydot_arr.row(step).to_slice().expect("ydot_traj must be contiguous");
        let p_list = p_arr.row(step).to_slice().expect("p_traj must be contiguous");
        let dl_dy_step = dl_dy_arr.row(step).to_slice().expect("dl_dy must be contiguous");

        for i in 0..n {
            rhs[i] = -dl_dy_step[i] + prev_dydot_vjp[i] * prev_c_j;
        }

        // Inject Exact Continuous Adjoint Terminal Condition for Boundary Event Sensitivities
        if event_trigger_idx >= 0 && step == n_steps - 1 {
            let idx = event_trigger_idx as usize;
            let mut ydot_val = ydot[idx];
            if ydot_val.abs() < 1e-12 {
                ydot_val = 1e-12 * ydot_val.signum().max(1.0);
            }
            rhs[idx] += 1.0 / ydot_val;
        }

        if bandwidth == -1 {
            let y_ptr = y.as_ptr();
            let ydot_ptr = ydot.as_ptr();
            let p_ptr = p_list.as_ptr();
            let m_ptr = m_slice.as_ptr();
            let mut jvp_t = |v: &[f64], out: &mut [f64]| {
                unsafe {
                    vjp_fn(
                        y_ptr,
                        ydot_ptr,
                        p_ptr,
                        m_ptr,
                        v.as_ptr(),
                        gmres_dp.as_mut_ptr(),
                        gmres_dy.as_mut_ptr(),
                        gmres_dydot.as_mut_ptr(),
                    );
                };
                for i in 0..n {
                    out[i] = gmres_dy[i] + c_j * gmres_dydot[i];
                }
            };
            let precond = |v: &[f64], out: &mut [f64]| {
                for i in 0..n {
                    out[i] = v[i] / (c_j * id_slice[i] + 1.0);
                }
            };
            solve_gmres(n, &mut rhs, &mut jvp_t, precond)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            unsafe {
                vjp_fn(
                    y_ptr,
                    ydot_ptr,
                    p_ptr,
                    m_ptr,
                    rhs.as_ptr(),
                    dp_out.as_mut_ptr(),
                    dy_out.as_mut_ptr(),
                    dydot_out.as_mut_ptr(),
                );
            }
        } else {
            rhs_orig.copy_from_slice(&rhs);
            let mut solved_by_cache = false;
            let cj_ratio = if prev_factorized_cj == 0.0 { 0.0 } else { c_j / prev_factorized_cj };

            // Attempt fast solution using cached LU factorization if time step scale is comparable
            if !solver.is_stale && cj_ratio >= 0.7 && cj_ratio <= 1.4 {
                if solver.solve(&mut rhs, &mut diag).is_ok() {
                    unsafe {
                        vjp_fn(
                            y.as_ptr(),
                            ydot.as_ptr(),
                            p_list.as_ptr(),
                            m_slice.as_ptr(),
                            rhs.as_ptr(),
                            dp_out.as_mut_ptr(),
                            dy_out.as_mut_ptr(),
                            dydot_out.as_mut_ptr(),
                        );
                    }
                    let mut max_res = 0.0_f64;
                    let mut max_b = 0.0_f64;
                    for i in 0..n {
                        let r_i = rhs_orig[i] - (dy_out[i] + c_j * dydot_out[i]);
                        delta_lambda[i] = r_i;
                        if !r_i.is_finite() {
                            max_res = f64::INFINITY;
                            break;
                        }
                        let abs_r = r_i.abs();
                        if abs_r > max_res { max_res = abs_r; }
                        let abs_b = rhs_orig[i].abs();
                        if abs_b > max_b { max_b = abs_b; }
                    }
                    let tol = (1e-7 * max_b).max(1e-10);
                    if max_res <= tol {
                        solved_by_cache = true;
                    } else if max_res.is_finite() && max_res <= 0.05 * (max_b + 1.0) {
                        // Iterative refinement step
                        if solver.solve(&mut delta_lambda, &mut diag).is_ok() {
                            for i in 0..n {
                                rhs[i] += delta_lambda[i];
                            }
                            unsafe {
                                vjp_fn(
                                    y.as_ptr(),
                                    ydot.as_ptr(),
                                    p_list.as_ptr(),
                                    m_slice.as_ptr(),
                                    rhs.as_ptr(),
                                    dp_out.as_mut_ptr(),
                                    dy_out.as_mut_ptr(),
                                    dydot_out.as_mut_ptr(),
                                );
                            }
                            let mut max_res1 = 0.0_f64;
                            for i in 0..n {
                                let r_i = rhs_orig[i] - (dy_out[i] + c_j * dydot_out[i]);
                                if !r_i.is_finite() {
                                    max_res1 = f64::INFINITY;
                                    break;
                                }
                                let abs_r = r_i.abs();
                                if abs_r > max_res1 { max_res1 = abs_r; }
                            }
                            if max_res1 <= tol {
                                solved_by_cache = true;
                            }
                        }
                    }
                }
            }

            if !solved_by_cache {
                solver.triplets.clear();
                if !cpr.color_seeds.is_empty() {
                    for (c_idx, seed) in cpr.color_seeds.iter().enumerate() {
                        unsafe {
                            jvp_fn(
                                y.as_ptr(),
                                ydot.as_ptr(),
                                p_list.as_ptr(),
                                m_slice.as_ptr(),
                                c_j,
                                seed.as_ptr(),
                                jvp_out.as_mut_ptr(),
                            );
                        }
                        for i in cpr.color_ptrs[c_idx]..cpr.color_ptrs[c_idx + 1] {
                            solver.triplets.push((
                                cpr.color_cols[i],
                                cpr.color_rows[i],
                                jvp_out[cpr.color_rows[i]],
                            ));
                        }
                    }
                    if !cpr.dense_rows.is_empty() {
                        for &r in &cpr.dense_rows {
                            lambda_vjp[r] = 1.0;
                            unsafe {
                                vjp_fn(
                                    y.as_ptr(),
                                    ydot.as_ptr(),
                                    p_list.as_ptr(),
                                    m_slice.as_ptr(),
                                    lambda_vjp.as_ptr(),
                                    vjp_dp.as_mut_ptr(),
                                    vjp_dy.as_mut_ptr(),
                                    vjp_dydot.as_mut_ptr(),
                                );
                            }
                            lambda_vjp[r] = 0.0;
                            for col in 0..n {
                                let val = vjp_dy[col] + c_j * vjp_dydot[col];
                                if val.abs() > 1e-16 || val.is_nan() {
                                    solver.triplets.push((col, r, val));
                                }
                            }
                        }
                    }
                }
                solver.factorize_from_triplets(&mut diag).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                prev_factorized_cj = c_j;
                rhs.copy_from_slice(&rhs_orig);
                solver.solve(&mut rhs, &mut diag).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

                unsafe {
                    vjp_fn(
                        y.as_ptr(),
                        ydot.as_ptr(),
                        p_list.as_ptr(),
                        m_slice.as_ptr(),
                        rhs.as_ptr(),
                        dp_out.as_mut_ptr(),
                        dy_out.as_mut_ptr(),
                        dydot_out.as_mut_ptr(),
                    );
                }
            }
        }

        for p_idx in 0..n_params {
            p_grad[p_idx] += dp_out[p_idx];
        }
        prev_dydot_vjp.copy_from_slice(&dydot_out);
        prev_c_j = c_j;
    }
    Ok(numpy::ndarray::Array1::from_vec(p_grad).to_pyarray(py))
}