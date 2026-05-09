use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, ToPyArray};
use rayon::prelude::*;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::solver::shared::problem::{Problem, Callbacks, CprData, SolverConfig};
use crate::solver::shared::workspace::Workspace;
use crate::solver::_2_stepper::bdf;
use crate::solver::_1_orchestrator::protocol::{ProtocolStep, run_sequence};
use crate::solver::_1_orchestrator::bisection::TrigInfo;

#[pyfunction]
#[pyo3(signature = (lib_path, y0_py, ydot0_py, id_py, p_list, m_list, t_eval, bandwidth, spatial_diag, max_steps, n_obs, cpr_seeds, cpr_ptrs, cpr_rows, cpr_cols, cpr_dense, record_history=false, debug=false, show_progress=true, v_idx=-1))]
pub fn solve_ida_native<'py>(
    py: Python<'py>, lib_path: String, y0_py: Vec<f64>, ydot0_py: Vec<f64>, id_py: Vec<f64>, p_list: Vec<f64>, m_list: Vec<f64>,
    t_eval: Vec<f64>, bandwidth: isize, spatial_diag: Vec<f64>, max_steps: Vec<f64>, n_obs: usize, 
    cpr_seeds: Vec<Vec<f64>>, cpr_ptrs: Vec<usize>, cpr_rows: Vec<usize>, cpr_cols: Vec<usize>, cpr_dense: Vec<usize>,
    record_history: bool, debug: bool, show_progress: bool, v_idx: i32
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let _ = debug;
    
    let lib = unsafe { libloading::Library::new(&lib_path).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))? };
    let fns = Callbacks {
        res_fn: unsafe { *lib.get(b"evaluate_residual\0").unwrap() },
        obs_fn: unsafe { lib.get(b"evaluate_observables\0").map(|s| *s).ok() },
        jvp_fn: unsafe { lib.get(b"evaluate_jvp\0").map(|s| *s).ok() },
        vjp_fn: unsafe { lib.get(b"evaluate_vjp\0").map(|s| *s).ok() },
        set_threads_fn: unsafe { lib.get(b"set_spatial_threads\0").map(|s| *s).ok() },
    };
    
    let cpr = CprData { color_seeds: cpr_seeds, color_ptrs: cpr_ptrs, color_rows: cpr_rows, color_cols: cpr_cols, dense_rows: cpr_dense };
    let n = y0_py.len();
    let prob = Problem { n, bw: bandwidth, n_obs, id: id_py, constraints: vec![0.0; n], m: m_list, spatial_diag, max_steps, cpr, config: SolverConfig::default(), fns };
    let mut wk = Workspace::new(n, bandwidth, y0_py, ydot0_py, p_list);
    
    crate::solver::_3_nonlinear::newton::calc_algebraic_roots(&prob, &mut wk).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    
    let mut out_traj = vec![0.0; t_eval.len() * n];
    let mut out_obs = vec![0.0; t_eval.len() * n_obs];
    let mut history = if record_history { Some(vec![(t_eval[0], wk.y.clone(), wk.ydot.clone())]) } else { None };

    for i in 0..n { out_traj[i] = wk.y[i]; }
    
    let mut step_obs = vec![0.0; n_obs];
    if let Some(obs_fn) = prob.fns.obs_fn {
        unsafe { obs_fn(wk.y.as_ptr(), wk.ydot.as_ptr(), wk.p.as_ptr(), prob.m.as_ptr(), step_obs.as_mut_ptr()); }
        for i in 0..n_obs { out_obs[i] = step_obs[i]; }
    }
    
    let total_steps = t_eval.len().saturating_sub(1);
    for step in 1..t_eval.len() {
        let dt = t_eval[step] - t_eval[step - 1];
        bdf::step(&prob, &mut wk, dt, history.as_mut()).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        for i in 0..n { out_traj[step * n + i] = wk.y[i]; }
        
        if let Some(obs_fn) = prob.fns.obs_fn {
            unsafe { obs_fn(wk.y.as_ptr(), wk.ydot.as_ptr(), wk.p.as_ptr(), prob.m.as_ptr(), step_obs.as_mut_ptr()); }
            for i in 0..n_obs { out_obs[step * n_obs + i] = step_obs[i]; }
        }
        
        if show_progress && total_steps > 0 {
            let is_final = step == total_steps;
            let pct = (step as f64 / total_steps as f64).clamp(0.0, 1.0);
            let v_str = if v_idx >= 0 { format!(" | V: {:.3}V", wk.y[v_idx as usize]) } else { String::new() };
            if is_final {
                print!("\r▶ {:<4} [██████████████████████████████] 100.0% | t: {:.1}s{}   \n", "Natv", t_eval[step], v_str);
            } else {
                let filled = (pct * 30.0) as usize;
                let bar: String = std::iter::repeat('█').take(filled).chain(std::iter::repeat('-').take(30 - filled)).collect();
                print!("\r▶ {:<4} [{}] {:.1}% | t: {:.1}s{}   ", "Natv", bar, pct * 100.0, t_eval[step], v_str);
            }
            std::io::stdout().flush().unwrap();
        }
    }
    
    let res_y = numpy::ndarray::Array2::from_shape_vec((t_eval.len(), n), out_traj).unwrap().to_pyarray(py);
    let res_obs = numpy::ndarray::Array2::from_shape_vec((t_eval.len(), n_obs), out_obs).unwrap().to_pyarray(py);
    
    if let Some(hist) = history {
        let h_len = hist.len();
        let mut micro_t = vec![0.0; h_len];
        let mut micro_y = vec![0.0; h_len * n];
        let mut micro_ydot = vec![0.0; h_len * n];
        for (i, (t, y, ydot)) in hist.into_iter().enumerate() {
            micro_t[i] = t;
            for j in 0..n { micro_y[i * n + j] = y[j]; micro_ydot[i * n + j] = ydot[j]; }
        }
        Ok((res_y, res_obs, numpy::ndarray::Array1::from_vec(micro_t).to_pyarray(py), 
            numpy::ndarray::Array2::from_shape_vec((h_len, n), micro_y).unwrap().to_pyarray(py), 
            numpy::ndarray::Array2::from_shape_vec((h_len, n), micro_ydot).unwrap().to_pyarray(py)))
    } else {
        let empty_t = numpy::ndarray::Array1::<f64>::zeros(0).to_pyarray(py);
        let empty_y = numpy::ndarray::Array2::<f64>::zeros((0, n)).to_pyarray(py);
        Ok((res_y, res_obs, empty_t, empty_y.clone(), empty_y))
    }
}

#[pyfunction]
#[allow(deprecated)]
#[pyo3(signature = (lib_path, y0, ydot0, id, p_batch, m_list, t_eval, bandwidth, spatial_diag, max_steps, n_obs, cpr_seeds, cpr_ptrs, cpr_rows, cpr_cols, cpr_dense, debug, max_workers=1, show_progress=true, protocol_steps=None, v_idx=-1))]
pub fn solve_batch_native<'py>(
    py: Python<'py>, lib_path: String, y0: Vec<f64>, ydot0: Vec<f64>, id: Vec<f64>, p_batch: Vec<Vec<f64>>, m_list: Vec<f64>, 
    t_eval: Vec<f64>, bandwidth: isize, spatial_diag: Vec<f64>, max_steps: Vec<f64>, n_obs: usize, 
    cpr_seeds: Vec<Vec<f64>>, cpr_ptrs: Vec<usize>, cpr_rows: Vec<usize>, cpr_cols: Vec<usize>, cpr_dense: Vec<usize>,
    debug: bool, max_workers: usize, show_progress: bool,
    protocol_steps: Option<Vec<Vec<(i32, f64, f64, (bool, usize, usize, bool, i32, f64), usize, usize, usize)>>>,
    v_idx: i32
) -> PyResult<Vec<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)>> {
    let _ = debug;
    
    let pool = rayon::ThreadPoolBuilder::new().num_threads(max_workers).build().unwrap();
    let completed = AtomicUsize::new(0);
    let total = p_batch.len();
    
    let lib = unsafe { libloading::Library::new(&lib_path).unwrap() };
    let fns = Callbacks {
        res_fn: unsafe { *lib.get(b"evaluate_residual\0").unwrap() },
        obs_fn: unsafe { lib.get(b"evaluate_observables\0").map(|s| *s).ok() },
        jvp_fn: unsafe { lib.get(b"evaluate_jvp\0").map(|s| *s).ok() },
        vjp_fn: unsafe { lib.get(b"evaluate_vjp\0").map(|s| *s).ok() },
        set_threads_fn: unsafe { lib.get(b"set_spatial_threads\0").map(|s| *s).ok() },
    };
    let cpr = CprData { color_seeds: cpr_seeds, color_ptrs: cpr_ptrs, color_rows: cpr_rows, color_cols: cpr_cols, dense_rows: cpr_dense };
    let prob_base = Problem { n: y0.len(), bw: bandwidth, n_obs, id, constraints: vec![0.0; y0.len()], m: m_list, spatial_diag, max_steps, cpr, config: SolverConfig::default(), fns };

    let results: Result<Vec<(Vec<f64>, Vec<f64>, Vec<f64>)>, String> = py.allow_threads(|| {
        pool.install(|| {
            p_batch.par_iter().enumerate().map(|(b_idx, p)| {
                let prob = prob_base.clone();
                let mut wk = Workspace::new(prob.n, prob.bw, y0.clone(), ydot0.clone(), p.clone());
                if let Some(f) = prob.fns.set_threads_fn { unsafe { f(1) }; }
                
                let step_list = if let Some(ref protos) = protocol_steps { protos.get(b_idx).cloned().unwrap_or_default() } else { Vec::new() };
                let has_protocol = !step_list.is_empty();
                
                let mut out_t = vec![0.0];
                let mut out_traj = wk.y.clone();
                let mut out_obs = vec![];
                
                if has_protocol {
                    let mut mapped_steps = Vec::new();
                    for s in step_list {
                        mapped_steps.push(ProtocolStep {
                            s_type: s.0, target_val: s.1, t_limit: s.2,
                            trig: TrigInfo { has_trig: s.3.0, idx: s.3.1, size: s.3.2, is_obs: s.3.3, op: s.3.4, val: s.3.5 },
                            p_mode: s.4, p_i: s.5, p_v: s.6
                        });
                    }
                    run_sequence(&prob, &mut wk, &mapped_steps, &mut out_t, &mut out_traj, &mut out_obs, show_progress, v_idx)?;
                } else {
                    out_t = t_eval.clone();
                    out_traj = vec![0.0; t_eval.len() * prob.n];
                    out_obs = vec![0.0; t_eval.len() * prob.n_obs];
                    crate::solver::_3_nonlinear::newton::calc_algebraic_roots(&prob, &mut wk)?;
                    for i in 0..prob.n { out_traj[i] = wk.y[i]; }
                    if let Some(obs_fn) = prob.fns.obs_fn {
                        let mut step_obs = vec![0.0; prob.n_obs];
                        unsafe { obs_fn(wk.y.as_ptr(), wk.ydot.as_ptr(), wk.p.as_ptr(), prob.m.as_ptr(), step_obs.as_mut_ptr()); }
                        for i in 0..prob.n_obs { out_obs[i] = step_obs[i]; }
                    }
                    for step in 1..t_eval.len() {
                        bdf::step(&prob, &mut wk, t_eval[step] - t_eval[step - 1], None)?;
                        for i in 0..prob.n { out_traj[step * prob.n + i] = wk.y[i]; }
                        if let Some(obs_fn) = prob.fns.obs_fn {
                            let mut step_obs = vec![0.0; prob.n_obs];
                            unsafe { obs_fn(wk.y.as_ptr(), wk.ydot.as_ptr(), wk.p.as_ptr(), prob.m.as_ptr(), step_obs.as_mut_ptr()); }
                            for i in 0..prob.n_obs { out_obs[step * prob.n_obs + i] = step_obs[i]; }
                        }
                    }
                }
                
                let c = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if show_progress {
                    let is_final = c == total;
                    let pct = (c as f64 / total as f64).clamp(0.0, 1.0);
                    if is_final {
                        print!("\r▶ {:<4} [██████████████████████████████] 100.0% | {}/{} models   \n", "Btch", c, total);
                    } else {
                        let filled = (pct * 30.0) as usize;
                        let bar: String = std::iter::repeat('█').take(filled).chain(std::iter::repeat('-').take(30 - filled)).collect();
                        print!("\r▶ {:<4} [{}] {:.1}% | {}/{} models   ", "Btch", bar, pct * 100.0, c, total);
                    }
                    std::io::stdout().flush().unwrap();
                }
                Ok((out_t, out_traj, out_obs))
            }).collect()
        })
    });

    let unwrapped = results.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let mut py_results = Vec::new();
    for (res_t, res_y, res_obs) in unwrapped { 
        let steps = res_t.len();
        let t_arr = numpy::ndarray::Array1::from_vec(res_t).to_pyarray(py);
        let y_arr = numpy::ndarray::Array2::from_shape_vec((steps, y0.len()), res_y).unwrap().to_pyarray(py);
        let obs_arr = numpy::ndarray::Array2::from_shape_vec((steps, n_obs), res_obs).unwrap().to_pyarray(py);
        py_results.push((t_arr, y_arr, obs_arr)); 
    }
    Ok(py_results)
}