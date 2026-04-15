use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, ToPyArray};
use rayon::prelude::*;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use super::session::SolverHandle;
use super::sundials::SundialsHandle;

#[pyfunction]
#[pyo3(signature = (lib_path, y0_py, ydot0_py, id_py, p_list, m_list, t_eval, bandwidth, spatial_diag, max_steps, n_obs, record_history=false, debug=false, show_progress=true, v_idx=-1))]
pub fn solve_ida_native<'py>(
    py: Python<'py>, lib_path: String, y0_py: Vec<f64>, ydot0_py: Vec<f64>, id_py: Vec<f64>, p_list: Vec<f64>, m_list: Vec<f64>,
    t_eval: Vec<f64>, bandwidth: isize, spatial_diag: Vec<f64>, max_steps: Vec<f64>, n_obs: usize, record_history: bool, debug: bool, show_progress: bool, v_idx: i32
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    
    let constraints = vec![0.0; y0_py.len()];
    
    let mut handle = SolverHandle::new(lib_path, y0_py.len(), bandwidth, y0_py.clone(), ydot0_py.clone(), id_py, constraints, p_list, m_list, spatial_diag, max_steps, n_obs, debug)?;
    let mut out_traj = vec![0.0; t_eval.len() * handle.n];
    let mut out_obs = vec![0.0; t_eval.len() * n_obs];
    
    let mut history = if record_history { Some(vec![(t_eval[0], y0_py.clone(), ydot0_py.clone())]) } else { None };

    for i in 0..handle.n { out_traj[i] = handle.y[i]; }
    
    let mut step_obs = vec![0.0; n_obs];
    handle.get_observables(&mut step_obs)?;
    for i in 0..n_obs { out_obs[i] = step_obs[i]; }
    
    let total_steps = t_eval.len().saturating_sub(1);
    
    for step in 1..t_eval.len() {
        let dt = t_eval[step] - t_eval[step - 1];
        handle.step_with_history(dt, history.as_mut())?;
        for i in 0..handle.n { out_traj[step * handle.n + i] = handle.y[i]; }
        
        handle.get_observables(&mut step_obs)?;
        for i in 0..n_obs { out_obs[step * n_obs + i] = step_obs[i]; }
        
        if show_progress && total_steps > 0 {
            let pct = (step as f64 / total_steps as f64) * 100.0;
            let filled = ((step as f64 / total_steps as f64) * 30.0) as usize;
            let bar: String = std::iter::repeat('█').take(filled).chain(std::iter::repeat('-').take(30 - filled)).collect();
            let v_str = if v_idx >= 0 { format!(" | V: {:.3}V", handle.y[v_idx as usize]) } else { String::new() };
            print!("\r▶ Native [{}] {:.1}% | t: {:.1}s{}   ", bar, pct, t_eval[step], v_str);
            std::io::stdout().flush().unwrap();
        }
    }
    if show_progress && total_steps > 0 { println!(); }
    
    let res_y = numpy::ndarray::Array2::from_shape_vec((t_eval.len(), handle.n), out_traj).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?.to_pyarray_bound(py);
    let res_obs = numpy::ndarray::Array2::from_shape_vec((t_eval.len(), n_obs), out_obs).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?.to_pyarray_bound(py);
    
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
        Ok((res_y, res_obs, numpy::ndarray::Array1::from_vec(micro_t).to_pyarray_bound(py), 
            numpy::ndarray::Array2::from_shape_vec((h_len, handle.n), micro_y).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?.to_pyarray_bound(py), 
            numpy::ndarray::Array2::from_shape_vec((h_len, handle.n), micro_ydot).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?.to_pyarray_bound(py)))
    } else {
        let empty_t = numpy::ndarray::Array1::<f64>::zeros(0).to_pyarray_bound(py);
        let empty_y = numpy::ndarray::Array2::<f64>::zeros((0, handle.n)).to_pyarray_bound(py);
        Ok((res_y, res_obs, empty_t, empty_y.clone(), empty_y))
    }
}

#[pyfunction]
#[pyo3(signature = (lib_path, y0_py, ydot0_py, id_py, p_list, m_list, t_eval, n_obs, show_progress=true, v_idx=-1))]
pub fn solve_ida_sundials<'py>(
    py: Python<'py>, lib_path: String, y0_py: Vec<f64>, ydot0_py: Vec<f64>, id_py: Vec<f64>, p_list: Vec<f64>, m_list: Vec<f64>, t_eval: Vec<f64>, n_obs: usize, show_progress: bool, v_idx: i32
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let mut handle = SundialsHandle::new(lib_path, y0_py.len(), y0_py, ydot0_py, id_py, p_list, m_list, n_obs)?;
    let mut out_traj = vec![0.0; t_eval.len() * handle.n];
    let mut out_obs = vec![0.0; t_eval.len() * n_obs];
    
    for i in 0..handle.n { out_traj[i] = handle._y_data[i]; }
    
    let mut step_obs = vec![0.0; n_obs];
    handle.get_observables(&mut step_obs)?;
    for i in 0..n_obs { out_obs[i] = step_obs[i]; }
    
    let total_steps = t_eval.len().saturating_sub(1);
    
    for step in 1..t_eval.len() {
        let dt = t_eval[step] - t_eval[step - 1];
        handle.step(dt)?;
        for i in 0..handle.n { out_traj[step * handle.n + i] = handle._y_data[i]; }
        
        handle.get_observables(&mut step_obs)?;
        for i in 0..n_obs { out_obs[step * n_obs + i] = step_obs[i]; }
        
        if show_progress && total_steps > 0 {
            let pct = (step as f64 / total_steps as f64) * 100.0;
            let filled = ((step as f64 / total_steps as f64) * 30.0) as usize;
            let bar: String = std::iter::repeat('█').take(filled).chain(std::iter::repeat('-').take(30 - filled)).collect();
            let v_str = if v_idx >= 0 { format!(" | V: {:.3}V", handle._y_data[v_idx as usize]) } else { String::new() };
            print!("\r▶ Sundials [{}] {:.1}% | t: {:.1}s{}   ", bar, pct, t_eval[step], v_str);
            std::io::stdout().flush().unwrap();
        }
    }
    if show_progress && total_steps > 0 { println!(); }
    
    let res_y = numpy::ndarray::Array2::from_shape_vec((t_eval.len(), handle.n), out_traj).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?.to_pyarray_bound(py);
    let res_obs = numpy::ndarray::Array2::from_shape_vec((t_eval.len(), n_obs), out_obs).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?.to_pyarray_bound(py);
    let empty_t = numpy::ndarray::Array1::<f64>::zeros(0).to_pyarray_bound(py);
    let empty_y = numpy::ndarray::Array2::<f64>::zeros((0, handle.n)).to_pyarray_bound(py);
    Ok((res_y, res_obs, empty_t, empty_y.clone(), empty_y))
}

#[pyfunction]
#[pyo3(signature = (lib_path, y0, ydot0, id, p_batch, m_list, t_eval, bandwidth, spatial_diag, max_steps, n_obs, debug, max_workers=1, show_progress=true, protocol_steps=None, v_idx=-1))]
pub fn solve_batch_native<'py>(
    py: Python<'py>, lib_path: String, y0: Vec<f64>, ydot0: Vec<f64>, id: Vec<f64>, p_batch: Vec<Vec<f64>>, m_list: Vec<f64>, 
    t_eval: Vec<f64>, bandwidth: isize, spatial_diag: Vec<f64>, max_steps: Vec<f64>, n_obs: usize, debug: bool, max_workers: usize, show_progress: bool,
    protocol_steps: Option<Vec<Vec<(i32, f64, f64, (bool, usize, usize, bool, i32, f64), usize, usize, usize)>>>,
    v_idx: i32
) -> PyResult<Vec<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)>> {
    
    let pool = rayon::ThreadPoolBuilder::new().num_threads(max_workers).build().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    
    let completed = AtomicUsize::new(0);
    let total = p_batch.len();

    let results: Result<Vec<(Vec<f64>, Vec<f64>, Vec<f64>)>, String> = py.allow_threads(|| {
        pool.install(|| {
            p_batch.par_iter().enumerate().map(|(b_idx, p)| {
                let constraints = vec![0.0; y0.len()]; 
                
                let mut handle = SolverHandle::new(lib_path.clone(), y0.len(), bandwidth, y0.clone(), ydot0.clone(), id.clone(), constraints, p.clone(), m_list.clone(), spatial_diag.clone(), max_steps.clone(), n_obs, debug.clone())
                    .map_err(|e| e.to_string())?;
                    
                // CRITICAL FIX: Prevent Rayon + OpenMP thread oversubscription.
                // Explicitly clamp OpenMP to 1 thread strictly within this Rayon worker context.
                handle.set_spatial_threads(1);

                let step_list = if let Some(ref protos) = protocol_steps {
                    protos.get(b_idx).cloned().unwrap_or_default()
                } else {
                    Vec::new()
                };
                
                let has_protocol = !step_list.is_empty();
                let mut step_obs = vec![0.0; n_obs];
                
                if has_protocol {
                    // --- NATIVE STATE MACHINE EXECUTION ---
                    let mut out_t = vec![0.0];
                    let mut out_traj = handle.y.clone();
                    if n_obs > 0 { handle.get_observables(&mut step_obs).unwrap_or(()); }
                    let mut out_obs = step_obs.clone();
                    
                    for step in &step_list {
                        let (s_type, target_val, t_limit, (has_trig, t_idx, t_size, t_is_obs, t_op, t_val), p_mode, p_i, p_v) = *step;
                        
                        if s_type == 0 { 
                            handle.set_parameter(p_mode, 1.0); handle.set_parameter(p_i, target_val);
                        } else if s_type == 1 { 
                            handle.set_parameter(p_mode, 0.0); handle.set_parameter(p_v, target_val);
                        } else if s_type == 2 { 
                            handle.set_parameter(p_mode, 1.0); handle.set_parameter(p_i, 0.0);
                        }
                        handle.calc_algebraic_roots().unwrap_or(());
                        
                        let mut t_elapsed = 0.0;
                        
                        while t_elapsed < t_limit {
                            if t_limit == std::f64::INFINITY && !has_trig { break; } 
                            let dt_step = 1.0_f64.min(t_limit - t_elapsed);
                            
                            let ckpt = handle.clone_state().unwrap();
                            let step_res = handle.step(dt_step);
                            
                            let mut triggered = false;
                            if has_trig && step_res.is_ok() {
                                for i in 0..t_size {
                                    let val = if t_is_obs {
                                        handle.get_observables(&mut step_obs).unwrap_or(());
                                        step_obs[t_idx + i]
                                    } else {
                                        handle.y[t_idx + i]
                                    };
                                    let trig = match t_op {
                                        1 => val > t_val, 2 => val < t_val, 3 => val >= t_val,
                                        4 => val <= t_val, 5 => val == t_val, 6 => val != t_val,
                                        _ => false,
                                    };
                                    if trig { triggered = true; break; }
                                }
                            }
                            
                            if triggered {
                                handle.restore_state(ckpt.0, ckpt.1.clone(), ckpt.2.clone()).unwrap_or(());
                                let mut low = 0.0;
                                let mut high = dt_step;
                                for _ in 0..15 {
                                    let mid = (low + high) / 2.0;
                                    if handle.step(mid).is_err() { break; }
                                    
                                    let mut trig_inner = false;
                                    for i in 0..t_size {
                                        let val = if t_is_obs {
                                            handle.get_observables(&mut step_obs).unwrap_or(());
                                            step_obs[t_idx + i]
                                        } else {
                                            handle.y[t_idx + i]
                                        };
                                        let trig = match t_op {
                                            1 => val > t_val, 2 => val < t_val, 3 => val >= t_val,
                                            4 => val <= t_val, 5 => val == t_val, 6 => val != t_val,
                                            _ => false,
                                        };
                                        if trig { trig_inner = true; break; }
                                    }
                                    if trig_inner { high = mid; } else { low = mid; }
                                    handle.restore_state(ckpt.0, ckpt.1.clone(), ckpt.2.clone()).unwrap_or(());
                                }
                                handle.step(low).unwrap_or(());
                                t_elapsed += low;
                                
                                out_t.push(handle.t);
                                out_traj.extend_from_slice(&handle.y);
                                if n_obs > 0 { handle.get_observables(&mut step_obs).unwrap_or(()); }
                                out_obs.extend_from_slice(&step_obs);
                                
                                // Cap off the step with a finalized 100% bar and a clean newline
                                if show_progress {
                                    let step_name = match s_type { 0 => "CC  ", 1 => "CV  ", 2 => "Rest", _ => "Step" };
                                    let v_str = if v_idx >= 0 { format!(" | V: {:.3}V", handle.y[v_idx as usize]) } else { String::new() };
                                    print!("\r▶ {} [██████████████████████████████] 100.0% | t: {:.1}s{}   \n", step_name, handle.t, v_str);
                                    std::io::stdout().flush().unwrap();
                                }
                                break; // Trigger met, advance
                            }
                            
                            if step_res.is_err() { break; }
                            
                            t_elapsed += dt_step;
                            out_t.push(handle.t);
                            out_traj.extend_from_slice(&handle.y);
                            if n_obs > 0 { handle.get_observables(&mut step_obs).unwrap_or(()); }
                            out_obs.extend_from_slice(&step_obs);
                            
                            // Render the live, ticking progress bar 
                            if show_progress {
                                let step_name = match s_type { 0 => "CC  ", 1 => "CV  ", 2 => "Rest", _ => "Step" };
                                let v_str = if v_idx >= 0 { format!(" | V: {:.3}V", handle.y[v_idx as usize]) } else { String::new() };
                                if t_limit == std::f64::INFINITY {
                                    print!("\r▶ {} ⏳ t: {:.1}s{}   ", step_name, handle.t, v_str);
                                } else {
                                    let pct = (t_elapsed / t_limit).min(1.0);
                                    let filled = (pct * 30.0) as usize;
                                    let bar: String = std::iter::repeat('█').take(filled).chain(std::iter::repeat('-').take(30 - filled)).collect();
                                    print!("\r▶ {} [{}] {:.1}% | t: {:.1}s{}   ", step_name, bar, pct * 100.0, handle.t, v_str);
                                }
                                std::io::stdout().flush().unwrap();
                            }
                        }
                        
                        // Finalize step if it completed via max time rather than trigger asymptote
                        if show_progress && t_elapsed >= t_limit && t_limit != std::f64::INFINITY {
                            let step_name = match s_type { 0 => "CC  ", 1 => "CV  ", 2 => "Rest", _ => "Step" };
                            let v_str = if v_idx >= 0 { format!(" | V: {:.3}V", handle.y[v_idx as usize]) } else { String::new() };
                            print!("\r▶ {} [██████████████████████████████] 100.0% | t: {:.1}s{}   \n", step_name, handle.t, v_str);
                            std::io::stdout().flush().unwrap();
                        }
                    }
                    
                    let c = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    if show_progress {
                        let pct = (c as f64 / total as f64) * 100.0;
                        let filled = ((c as f64 / total as f64) * 30.0) as usize;
                        let bar: String = std::iter::repeat('█').take(filled).chain(std::iter::repeat('-').take(30 - filled)).collect();
                        print!("\r▶ Batch  [{}] {:.1}% | {}/{} models   ", bar, pct, c, total);
                        std::io::stdout().flush().unwrap();
                    }
                    
                    Ok((out_t, out_traj, out_obs))
                } else {
                    // --- CONTINUOUS EVALUATION BLOCK (No Protocol) ---
                    let out_t = t_eval.clone();
                    let mut out_traj = vec![0.0; t_eval.len() * handle.n];
                    let mut out_obs = vec![0.0; t_eval.len() * n_obs];
                    for i in 0..handle.n { out_traj[i] = handle.y[i]; }
                    
                    handle.get_observables(&mut step_obs).map_err(|e| e.to_string())?;
                    for i in 0..n_obs { out_obs[i] = step_obs[i]; }
                    
                    for step in 1..t_eval.len() {
                        let dt = t_eval[step] - t_eval[step - 1];
                        handle.step(dt).map_err(|e| e.to_string())?;
                        for i in 0..handle.n { out_traj[step * handle.n + i] = handle.y[i]; }
                        
                        handle.get_observables(&mut step_obs).map_err(|e| e.to_string())?;
                        for i in 0..n_obs { out_obs[step * n_obs + i] = step_obs[i]; }
                    }
                    
                    let c = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    if show_progress {
                        let pct = (c as f64 / total as f64) * 100.0;
                        let filled = ((c as f64 / total as f64) * 30.0) as usize;
                        let bar: String = std::iter::repeat('█').take(filled).chain(std::iter::repeat('-').take(30 - filled)).collect();
                        print!("\r▶ Batch  [{}] {:.1}% | {}/{} models   ", bar, pct, c, total);
                        std::io::stdout().flush().unwrap();
                    }
                    
                    Ok((out_t, out_traj, out_obs))
                }
            }).collect()
        })
    });
    
    if show_progress { println!(); }

    let unwrapped = results.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let mut py_results = Vec::new();
    for (res_t, res_y, res_obs) in unwrapped { 
        let steps = res_t.len();
        let t_arr = numpy::ndarray::Array1::from_vec(res_t).to_pyarray_bound(py);
        let y_arr = numpy::ndarray::Array2::from_shape_vec((steps, y0.len()), res_y).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?.to_pyarray_bound(py);
        let obs_arr = numpy::ndarray::Array2::from_shape_vec((steps, n_obs), res_obs).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?.to_pyarray_bound(py);
        py_results.push((t_arr, y_arr, obs_arr)); 
    }
    Ok(py_results)
}