use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, ToPyArray};
use rayon::prelude::*;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use super::session::SolverHandle;
use super::sundials::SundialsHandle;

#[pyfunction]
#[pyo3(signature = (lib_path, y0_py, ydot0_py, id_py, p_list, m_list, t_eval, bandwidth, spatial_diag, max_steps, record_history=false, debug=false, show_progress=true))]
pub fn solve_ida_native<'py>(
    py: Python<'py>, lib_path: String, y0_py: Vec<f64>, ydot0_py: Vec<f64>, id_py: Vec<f64>, p_list: Vec<f64>, m_list: Vec<f64>,
    t_eval: Vec<f64>, bandwidth: isize, spatial_diag: Vec<f64>, max_steps: Vec<f64>, record_history: bool, debug: bool, show_progress: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    
    let constraints = vec![0.0; y0_py.len()];
    
    let mut handle = SolverHandle::new(lib_path, y0_py.len(), bandwidth, y0_py.clone(), ydot0_py.clone(), id_py, constraints, p_list, m_list, spatial_diag, max_steps, debug)?;
    let mut out_traj = vec![0.0; t_eval.len() * handle.n];
    
    let mut history = if record_history { Some(vec![(t_eval[0], y0_py.clone(), ydot0_py.clone())]) } else { None };

    for i in 0..handle.n { out_traj[i] = handle.y[i]; }
    
    let total_steps = t_eval.len().saturating_sub(1);
    
    for step in 1..t_eval.len() {
        let dt = t_eval[step] - t_eval[step - 1];
        handle.step_with_history(dt, history.as_mut())?;
        for i in 0..handle.n { out_traj[step * handle.n + i] = handle.y[i]; }
        
        if show_progress && total_steps > 0 {
            let pct = (step as f64 / total_steps as f64) * 100.0;
            let filled = ((step as f64 / total_steps as f64) * 30.0) as usize;
            let bar: String = std::iter::repeat('█').take(filled).chain(std::iter::repeat('-').take(30 - filled)).collect();
            print!("\r▶ Native [{}] {:.1}% | t: {:.1}s   ", bar, pct, t_eval[step]);
            std::io::stdout().flush().unwrap();
        }
    }
    if show_progress && total_steps > 0 { println!(); }
    
    let res_y = numpy::ndarray::Array2::from_shape_vec((t_eval.len(), handle.n), out_traj).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?.to_pyarray_bound(py);
    
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
        Ok((res_y, numpy::ndarray::Array1::from_vec(micro_t).to_pyarray_bound(py), 
            numpy::ndarray::Array2::from_shape_vec((h_len, handle.n), micro_y).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?.to_pyarray_bound(py), 
            numpy::ndarray::Array2::from_shape_vec((h_len, handle.n), micro_ydot).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?.to_pyarray_bound(py)))
    } else {
        let empty_t = numpy::ndarray::Array1::<f64>::zeros(0).to_pyarray_bound(py);
        let empty_y = numpy::ndarray::Array2::<f64>::zeros((0, handle.n)).to_pyarray_bound(py);
        Ok((res_y, empty_t, empty_y.clone(), empty_y))
    }
}

#[pyfunction]
#[pyo3(signature = (lib_path, y0_py, ydot0_py, id_py, p_list, m_list, t_eval, show_progress=true))]
pub fn solve_ida_sundials<'py>(
    py: Python<'py>, lib_path: String, y0_py: Vec<f64>, ydot0_py: Vec<f64>, id_py: Vec<f64>, p_list: Vec<f64>, m_list: Vec<f64>, t_eval: Vec<f64>, show_progress: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let mut handle = SundialsHandle::new(lib_path, y0_py.len(), y0_py, ydot0_py, id_py, p_list, m_list)?;
    let mut out_traj = vec![0.0; t_eval.len() * handle.n];
    for i in 0..handle.n { out_traj[i] = handle._y_data[i]; }
    
    let total_steps = t_eval.len().saturating_sub(1);
    
    for step in 1..t_eval.len() {
        let dt = t_eval[step] - t_eval[step - 1];
        handle.step(dt)?;
        for i in 0..handle.n { out_traj[step * handle.n + i] = handle._y_data[i]; }
        
        if show_progress && total_steps > 0 {
            let pct = (step as f64 / total_steps as f64) * 100.0;
            let filled = ((step as f64 / total_steps as f64) * 30.0) as usize;
            let bar: String = std::iter::repeat('█').take(filled).chain(std::iter::repeat('-').take(30 - filled)).collect();
            print!("\r▶ Sundials [{}] {:.1}% | t: {:.1}s   ", bar, pct, t_eval[step]);
            std::io::stdout().flush().unwrap();
        }
    }
    if show_progress && total_steps > 0 { println!(); }
    
    let res_y = numpy::ndarray::Array2::from_shape_vec((t_eval.len(), handle.n), out_traj).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?.to_pyarray_bound(py);
    let empty_t = numpy::ndarray::Array1::<f64>::zeros(0).to_pyarray_bound(py);
    let empty_y = numpy::ndarray::Array2::<f64>::zeros((0, handle.n)).to_pyarray_bound(py);
    Ok((res_y, empty_t, empty_y.clone(), empty_y))
}

#[pyfunction]
#[pyo3(signature = (lib_path, y0, ydot0, id, p_batch, m_list, t_eval, bandwidth, spatial_diag, max_steps, debug, max_workers=1, show_progress=true))]
pub fn solve_batch_native<'py>(
    py: Python<'py>, lib_path: String, y0: Vec<f64>, ydot0: Vec<f64>, id: Vec<f64>, p_batch: Vec<Vec<f64>>, m_list: Vec<f64>, t_eval: Vec<f64>, bandwidth: isize, spatial_diag: Vec<f64>, max_steps: Vec<f64>, debug: bool, max_workers: usize, show_progress: bool
) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
    
    let pool = rayon::ThreadPoolBuilder::new().num_threads(max_workers).build().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    
    let completed = AtomicUsize::new(0);
    let total = p_batch.len();

    let results: Result<Vec<Vec<f64>>, String> = py.allow_threads(|| {
        pool.install(|| {
            p_batch.par_iter().map(|p| {
                let constraints = vec![0.0; y0.len()]; 
                
                let mut handle = SolverHandle::new(lib_path.clone(), y0.len(), bandwidth, y0.clone(), ydot0.clone(), id.clone(), constraints, p.clone(), m_list.clone(), spatial_diag.clone(), max_steps.clone(), debug.clone())
                    .map_err(|e| e.to_string())?;
                    
                let mut out_traj = vec![0.0; t_eval.len() * handle.n];
                for i in 0..handle.n { out_traj[i] = handle.y[i]; }
                
                for step in 1..t_eval.len() {
                    let dt = t_eval[step] - t_eval[step - 1];
                    handle.step(dt).map_err(|e| e.to_string())?;
                    for i in 0..handle.n { out_traj[step * handle.n + i] = handle.y[i]; }
                }
                
                let c = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if show_progress {
                    let pct = (c as f64 / total as f64) * 100.0;
                    let filled = ((c as f64 / total as f64) * 30.0) as usize;
                    let bar: String = std::iter::repeat('█').take(filled).chain(std::iter::repeat('-').take(30 - filled)).collect();
                    print!("\r▶ Batch  [{}] {:.1}% | {}/{} models   ", bar, pct, c, total);
                    std::io::stdout().flush().unwrap();
                }
                
                Ok(out_traj)
            }).collect()
        })
    });
    
    if show_progress { println!(); }

    let unwrapped = results.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let mut py_results = Vec::new();
    for res in unwrapped { 
        py_results.push(numpy::ndarray::Array2::from_shape_vec((t_eval.len(), y0.len()), res).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?.to_pyarray_bound(py)); 
    }
    Ok(py_results)
}