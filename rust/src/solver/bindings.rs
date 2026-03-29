use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, ToPyArray};
use rayon::prelude::*;
use super::session::SolverHandle;
use super::sundials::SundialsHandle;

#[pyfunction]
#[pyo3(signature = (lib_path, y0_py, ydot0_py, id_py, p_list, t_eval, bandwidth, spatial_diag, record_history=false, debug=false))]
pub fn solve_ida_native<'py>(
    py: Python<'py>, lib_path: String, y0_py: Vec<f64>, ydot0_py: Vec<f64>, id_py: Vec<f64>, p_list: Vec<f64>,
    t_eval: Vec<f64>, bandwidth: isize, spatial_diag: Vec<f64>, record_history: bool, debug: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    
    let mut handle = SolverHandle::new(lib_path, y0_py.len(), bandwidth, y0_py.clone(), ydot0_py.clone(), id_py, p_list, spatial_diag, debug)?;
    let mut out_traj = vec![0.0; t_eval.len() * handle.n];
    
    let mut history = if record_history { Some(vec![(t_eval[0], y0_py.clone(), ydot0_py.clone())]) } else { None };

    for i in 0..handle.n { out_traj[i] = handle.y[i]; }
    for step in 1..t_eval.len() {
        let dt = t_eval[step] - t_eval[step - 1];
        handle.step_with_history(dt, history.as_mut())?;
        for i in 0..handle.n { out_traj[step * handle.n + i] = handle.y[i]; }
    }
    
    let res_y = numpy::ndarray::Array2::from_shape_vec((t_eval.len(), handle.n), out_traj).unwrap().to_pyarray_bound(py);
    
    if let Some(hist) = history {
        let h_len = hist.len();
        let mut micro_t = vec![0.0; h_len];
        let mut micro_y = vec![0.0; h_len * handle.n];
        let mut micro_ydot = vec![0.0; h_len * handle.n];
        for (i, (t, y, ydot)) in hist.into_iter().enumerate() {
            micro_t[i] = t;
            for j in 0..handle.n {
                micro_y[i * handle.n + j] = y[j];
                micro_ydot[i * handle.n + j] = ydot[j];
            }
        }
        Ok((res_y, numpy::ndarray::Array1::from_vec(micro_t).to_pyarray_bound(py), numpy::ndarray::Array2::from_shape_vec((h_len, handle.n), micro_y).unwrap().to_pyarray_bound(py), numpy::ndarray::Array2::from_shape_vec((h_len, handle.n), micro_ydot).unwrap().to_pyarray_bound(py)))
    } else {
        let empty_t = numpy::ndarray::Array1::<f64>::zeros(0).to_pyarray_bound(py);
        let empty_y = numpy::ndarray::Array2::<f64>::zeros((0, handle.n)).to_pyarray_bound(py);
        Ok((res_y, empty_t, empty_y.clone(), empty_y))
    }
}

#[pyfunction]
#[pyo3(signature = (lib_path, y0_py, ydot0_py, id_py, p_list, t_eval))]
pub fn solve_ida_sundials<'py>(
    py: Python<'py>, lib_path: String, y0_py: Vec<f64>, ydot0_py: Vec<f64>, id_py: Vec<f64>, p_list: Vec<f64>, t_eval: Vec<f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let mut handle = SundialsHandle::new(lib_path, y0_py.len(), y0_py, ydot0_py, id_py, p_list)?;
    let mut out_traj = vec![0.0; t_eval.len() * handle.n];
    for i in 0..handle.n { out_traj[i] = handle._y_data[i]; }
    for step in 1..t_eval.len() {
        let dt = t_eval[step] - t_eval[step - 1];
        handle.step(dt)?;
        for i in 0..handle.n { out_traj[step * handle.n + i] = handle._y_data[i]; }
    }
    let res_y = numpy::ndarray::Array2::from_shape_vec((t_eval.len(), handle.n), out_traj).unwrap().to_pyarray_bound(py);
    let empty_t = numpy::ndarray::Array1::<f64>::zeros(0).to_pyarray_bound(py);
    let empty_y = numpy::ndarray::Array2::<f64>::zeros((0, handle.n)).to_pyarray_bound(py);
    Ok((res_y, empty_t, empty_y.clone(), empty_y))
}

#[pyfunction]
pub fn solve_batch_native<'py>(
    py: Python<'py>, lib_path: String, y0: Vec<f64>, ydot0: Vec<f64>, id: Vec<f64>, p_batch: Vec<Vec<f64>>, t_eval: Vec<f64>, bandwidth: isize, spatial_diag: Vec<f64>, debug: bool,
) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
    let results: Result<Vec<Vec<f64>>, String> = p_batch.par_iter().map(|p| {
        let mut handle = SolverHandle::new(lib_path.clone(), y0.len(), bandwidth, y0.clone(), ydot0.clone(), id.clone(), p.clone(), spatial_diag.clone(), debug.clone())
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

    let unwrapped = results.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let mut py_results = Vec::new();
    for res in unwrapped { 
        py_results.push(numpy::ndarray::Array2::from_shape_vec((t_eval.len(), y0.len()), res).unwrap().to_pyarray_bound(py)); 
    }
    Ok(py_results)
}