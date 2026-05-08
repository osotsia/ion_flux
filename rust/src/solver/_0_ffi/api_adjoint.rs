use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use crate::solver::shared::callbacks::{NativeJvpFn, NativeVjpFn};
use crate::solver::shared::problem::CprData;
use crate::solver::_4_linear::sparse_lu::NativeSparseLuSolver;
use crate::solver::_4_linear::gmres::solve_gmres;
use crate::solver::shared::diagnostics::Diagnostics;

#[pyfunction]
pub fn discrete_adjoint_native<'py>(
    py: Python<'py>, lib_path: String, 
    y_traj: PyReadonlyArray2<f64>, ydot_traj: PyReadonlyArray2<f64>,
    t_eval: PyReadonlyArray1<f64>, id_arr: PyReadonlyArray1<f64>, 
    p_traj: PyReadonlyArray2<f64>, m_list: PyReadonlyArray1<f64>, 
    dl_dy: PyReadonlyArray2<f64>, bandwidth: isize,
    cpr_seeds: Vec<Vec<f64>>, cpr_ptrs: Vec<usize>, cpr_rows: Vec<usize>, cpr_cols: Vec<usize>, cpr_dense: Vec<usize>
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
    let mut lambda = vec![0.0; n];
    let mut prev_dydot_vjp = vec![0.0; n];
    let mut prev_c_j = 0.0;
    let mut diag = Diagnostics::new(n);
    let mut solver = NativeSparseLuSolver::new(n, bandwidth);
    
    for step in (1..n_steps).rev() {
        let dt = t_arr[step] - t_arr[step - 1];
        if dt <= 1e-12 { continue; }
        
        let c_j = 1.0 / dt; 
        let y = y_arr.row(step).to_slice().expect("y_traj must be contiguous");
        let ydot = ydot_arr.row(step).to_slice().expect("ydot_traj must be contiguous");
        let p_list = p_arr.row(step).to_slice().expect("p_traj must be contiguous");
        let dl_dy_step = dl_dy_arr.row(step).to_slice().expect("dl_dy must be contiguous");
        
        let mut rhs = vec![0.0; n];
        for i in 0..n { rhs[i] = -dl_dy_step[i] + prev_dydot_vjp[i] * prev_c_j; }
        
        if bandwidth == -1 {
            let y_ptr = y.as_ptr(); let ydot_ptr = ydot.as_ptr(); let p_ptr = p_list.as_ptr(); let m_ptr = m_slice.as_ptr();
            let jvp_t = |v: &[f64], out: &mut [f64]| {
                let mut dp_dummy = vec![0.0; n_params]; let mut dy_out = vec![0.0; n]; let mut dydot_out = vec![0.0; n];
                unsafe { vjp_fn(y_ptr, ydot_ptr, p_ptr, m_ptr, v.as_ptr(), dp_dummy.as_mut_ptr(), dy_out.as_mut_ptr(), dydot_out.as_mut_ptr()) };
                for i in 0..n { out[i] = dy_out[i] + c_j * dydot_out[i]; }
            };
            let precond = |v: &[f64], out: &mut[f64]| { for i in 0..n { out[i] = v[i] / (c_j * id_slice[i] + 1.0); } };
            solve_gmres(n, &mut rhs, jvp_t, precond).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        } else {
            solver.triplets.clear();
            if !cpr.color_seeds.is_empty() {
                for (c_idx, seed) in cpr.color_seeds.iter().enumerate() {
                    let mut jvp_out = vec![0.0; n];
                    unsafe { jvp_fn(y.as_ptr(), ydot.as_ptr(), p_list.as_ptr(), m_slice.as_ptr(), c_j, seed.as_ptr(), jvp_out.as_mut_ptr()); }
                    for i in cpr.color_ptrs[c_idx]..cpr.color_ptrs[c_idx + 1] {
                        solver.triplets.push((cpr.color_cols[i], cpr.color_rows[i], jvp_out[cpr.color_rows[i]]));
                    }
                }
                if !cpr.dense_rows.is_empty() {
                    let mut dp_out = vec![0.0; n_params]; let mut dy_out = vec![0.0; n]; let mut dydot_out = vec![0.0; n]; let mut lambda_vjp = vec![0.0; n];
                    for &r in &cpr.dense_rows {
                        lambda_vjp[r] = 1.0;
                        unsafe { vjp_fn(y.as_ptr(), ydot.as_ptr(), p_list.as_ptr(), m_slice.as_ptr(), lambda_vjp.as_ptr(), dp_out.as_mut_ptr(), dy_out.as_mut_ptr(), dydot_out.as_mut_ptr()); }
                        lambda_vjp[r] = 0.0;
                        for col in 0..n {
                            let val = dy_out[col] + c_j * dydot_out[col];
                            if val.abs() > 1e-16 || val.is_nan() { solver.triplets.push((col, r, val)); }
                        }
                    }
                }
            }
            solver.factorize_from_triplets(&mut diag).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            solver.solve(&mut rhs, &mut diag).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        }
        
        lambda = rhs;
        let mut dp_out = vec![0.0; n_params]; let mut dy_out = vec![0.0; n]; let mut dydot_out = vec![0.0; n];
        unsafe { vjp_fn(y.as_ptr(), ydot.as_ptr(), p_list.as_ptr(), m_slice.as_ptr(), lambda.as_ptr(), dp_out.as_mut_ptr(), dy_out.as_mut_ptr(), dydot_out.as_mut_ptr()) };
        for p_idx in 0..n_params { p_grad[p_idx] += dp_out[p_idx]; }
        prev_dydot_vjp = dydot_out; prev_c_j = c_j;
    }
    Ok(numpy::ndarray::Array1::from_vec(p_grad).to_pyarray(py))
}