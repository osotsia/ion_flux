use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use crate::ffi::callbacks::{NativeJvpFn, NativeVjpFn};
use crate::stepper::problem::CprData;
use crate::control::backward_pass::{AdjointParams, solve_adjoint_sensitivities};

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
    let lib = unsafe { libloading::Library::new(&lib_path).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))? };
    let jvp_fn: NativeJvpFn = unsafe { *lib.get(b"evaluate_jvp\0").unwrap() };
    let vjp_fn: NativeVjpFn = unsafe { *lib.get(b"evaluate_vjp\0").unwrap() };

    let cpr = CprData { color_seeds: cpr_seeds, color_ptrs: cpr_ptrs, color_rows: cpr_rows, color_cols: cpr_cols, dense_rows: cpr_dense };
    let id_slice = id_arr.as_slice().expect("id_arr must be contiguous");
    let m_slice = m_list.as_slice().expect("m_list must be contiguous");

    let params = AdjointParams {
        bandwidth,
        id_slice,
        m_slice,
        cpr: &cpr,
        jvp_fn,
        vjp_fn,
        event_trigger_idx,
    };

    let p_grad = solve_adjoint_sensitivities(
        &params,
        y_traj.as_array(),
        ydot_traj.as_array(),
        t_eval.as_array(),
        p_traj.as_array(),
        dl_dy.as_array(),
    ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    Ok(numpy::ndarray::Array1::from_vec(p_grad).to_pyarray(py))
}
