use pyo3::prelude::*;
use numpy::{PyArray2, ToPyArray};
use rayon::prelude::*;
use super::session::SolverHandle;

#[pyfunction]
pub fn solve_ida_native<'py>(
    py: Python<'py>,
    lib_path: String,
    y0_py: Vec<f64>,
    ydot0_py: Vec<f64>,
    id_py: Vec<f64>,
    p_list: Vec<f64>,
    m_list: Vec<f64>,
    precond_diag: Vec<f64>,
    t_eval: Vec<f64>,
    bandwidth: isize, 
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut handle = SolverHandle::new(lib_path, y0_py.len(), bandwidth, y0_py, ydot0_py, id_py, p_list, m_list, precond_diag)?;
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
    m_list: Vec<f64>,
    precond_diag: Vec<f64>,
    t_eval: Vec<f64>,
    bandwidth: isize,
) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
    // Bug 5 Fix: Defends strictly against exponentially catastrophic N*N thread spawning
    std::env::set_var("OMP_NUM_THREADS", "1");

    let results: Result<Vec<Vec<f64>>, String> = p_batch.par_iter().map(|p| {
        let mut handle = SolverHandle::new(lib_path.clone(), y0.len(), bandwidth, y0.clone(), ydot0.clone(), id.clone(), p.clone(), m_list.clone(), precond_diag.clone())
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
