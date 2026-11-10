use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, ToPyArray};
use rayon::prelude::*;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::stepper::problem::{Problem, Callbacks, CprData, SolverConfig};
use crate::stepper::state::SolverInstance;
use crate::control::sequence::{ProtocolStep, run_sequence};
use crate::control::bisection::TrigInfo;
use crate::api::{NativeSolverCrash, SolverError, crash_report_to_pydict};

pub type RawProtocolStep = (i32, f64, f64, (bool, usize, usize, bool, i32, f64), usize, usize, usize);

/// Encapsulates the execution output vectors for a single model run.
pub struct ModelTrajectory {
    pub t: Vec<f64>,
    pub y: Vec<f64>,
    pub obs: Vec<f64>,
    pub micro_t: Vec<f64>,
    pub micro_y: Vec<f64>,
    pub micro_ydot: Vec<f64>,
}

// =============================================================================
// Helper Functions: Symbol Loading, Conversion, and Formatting
// =============================================================================

fn load_callbacks(lib: &libloading::Library) -> Result<Callbacks, String> {
    unsafe {
        let res_fn = *lib
            .get(b"evaluate_residual\0")
            .map_err(|e| format!("Symbol 'evaluate_residual' missing: {e}"))?;
        let obs_fn = lib.get(b"evaluate_observables\0").map(|s| *s).ok();
        let jvp_fn = lib.get(b"evaluate_jvp\0").map(|s| *s).ok();
        let vjp_fn = lib.get(b"evaluate_vjp\0").map(|s| *s).ok();
        let set_threads_fn = lib.get(b"set_spatial_threads\0").map(|s| *s).ok();

        Ok(Callbacks {
            res_fn,
            obs_fn,
            jvp_fn,
            vjp_fn,
            set_threads_fn,
        })
    }
}

fn parse_protocol_steps(raw_steps: &[RawProtocolStep]) -> Vec<ProtocolStep> {
    raw_steps
        .iter()
        .map(|s| ProtocolStep {
            s_type: s.0,
            target_val: s.1,
            t_limit: s.2,
            trig: TrigInfo {
                has_trig: s.3.0,
                idx: s.3.1,
                size: s.3.2,
                is_obs: s.3.3,
                op: s.3.4,
                val: s.3.5,
            },
            p_mode: s.4,
            p_i: s.5,
            p_v: s.6,
        })
        .collect()
}

fn flatten_tape(
    history: Option<Vec<(f64, Vec<f64>, Vec<f64>)>>,
    n_states: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if let Some(hist) = history {
        let h_len = hist.len();
        let mut mt = vec![0.0; h_len];
        let mut my = vec![0.0; h_len * n_states];
        let mut mydot = vec![0.0; h_len * n_states];
        for (i, (t, y_step, ydot_step)) in hist.into_iter().enumerate() {
            mt[i] = t;
            for j in 0..n_states {
                my[i * n_states + j] = y_step[j];
                mydot[i * n_states + j] = ydot_step[j];
            }
        }
        (mt, my, mydot)
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    }
}

fn print_step_progress(tag: &str, t: f64, pct: f64, v_idx: i32, y: &[f64], is_final: bool) {
    let p = pct.clamp(0.0, 1.0);
    let v_str = if v_idx >= 0 && (v_idx as usize) < y.len() {
        format!(" | V: {:.3}V", y[v_idx as usize])
    } else {
        String::new()
    };

    if is_final {
        print!("\r▶ {:<4} [██████████████████████████████] 100.0% | t: {:.1}s{}   \n", tag, t, v_str);
    } else {
        let filled = (p * 30.0) as usize;
        let bar: String = std::iter::repeat('█').take(filled).chain(std::iter::repeat('-').take(30 - filled)).collect();
        print!("\r▶ {:<4} [{}] {:.1}% | t: {:.1}s{}   ", tag, bar, p * 100.0, t, v_str);
    }
    let _ = std::io::stdout().flush();
}

fn print_batch_progress(completed: usize, total: usize) {
    let is_final = completed == total;
    let pct = (completed as f64 / total as f64).clamp(0.0, 1.0);
    if is_final {
        print!("\r▶ {:<4} [██████████████████████████████] 100.0% | {}/{} models   \n", "Btch", completed, total);
    } else {
        let filled = (pct * 30.0) as usize;
        let bar: String = std::iter::repeat('█').take(filled).chain(std::iter::repeat('-').take(30 - filled)).collect();
        print!("\r▶ {:<4} [{}] {:.1}% | {}/{} models   ", "Btch", bar, pct * 100.0, completed, total);
    }
    let _ = std::io::stdout().flush();
}

// =============================================================================
// Trajectory Solvers: Sequence Protocol vs Fixed Grid
// =============================================================================

fn execute_sequence_trajectory(
    prob: &Problem,
    instance: &mut SolverInstance,
    raw_steps: &[RawProtocolStep],
    record_tape: bool,
    show_progress: bool,
    v_idx: i32,
) -> Result<ModelTrajectory, SolverError> {
    let mut out_t = Vec::new();
    let mut out_traj = Vec::new();
    let mut out_obs = Vec::new();
    let mut history_record = if record_tape { Some(Vec::new()) } else { None };

    let mapped_steps = parse_protocol_steps(raw_steps);

    run_sequence(
        prob,
        instance,
        &mapped_steps,
        &mut out_t,
        &mut out_traj,
        &mut out_obs,
        history_record.as_mut(),
        show_progress,
        v_idx,
    )?;

    let (micro_t, micro_y, micro_ydot) = flatten_tape(history_record, prob.n);

    Ok(ModelTrajectory {
        t: out_t,
        y: out_traj,
        obs: out_obs,
        micro_t,
        micro_y,
        micro_ydot,
    })
}

fn execute_grid_trajectory(
    prob: &Problem,
    instance: &mut SolverInstance,
    t_eval: &[f64],
    record_tape: bool,
    show_progress: bool,
    v_idx: i32,
) -> Result<ModelTrajectory, SolverError> {
    let n = prob.n;
    let n_obs = prob.n_obs;
    let n_steps = t_eval.len();

    let mut out_traj = vec![0.0; n_steps * n];
    let mut out_obs = vec![0.0; n_steps * n_obs];
    let mut history_record = if record_tape {
        Some(Vec::with_capacity(n_steps))
    } else {
        None
    };

    instance.calc_algebraic_roots(prob).map_err(SolverError::Message)?;

    for i in 0..n {
        out_traj[i] = instance.state.y[i];
    }

    if let Some(obs_fn) = prob.fns.obs_fn {
        let mut step_obs = vec![0.0; n_obs];
        unsafe {
            obs_fn(
                instance.state.y.as_ptr(),
                instance.state.ydot.as_ptr(),
                instance.state.p.as_ptr(),
                prob.m.as_ptr(),
                step_obs.as_mut_ptr(),
            );
        }
        for i in 0..n_obs {
            out_obs[i] = step_obs[i];
        }
    }

    if let Some(ref mut hist) = history_record {
        hist.push((t_eval[0], instance.state.y.clone(), instance.state.ydot.clone()));
    }

    let total_steps = n_steps.saturating_sub(1);
    for step in 1..n_steps {
        let dt = t_eval[step] - t_eval[step - 1];
        instance.step(prob, dt, history_record.as_mut()).map_err(SolverError::Crash)?;

        for i in 0..n {
            out_traj[step * n + i] = instance.state.y[i];
        }

        if let Some(obs_fn) = prob.fns.obs_fn {
            let mut step_obs = vec![0.0; n_obs];
            unsafe {
                obs_fn(
                    instance.state.y.as_ptr(),
                    instance.state.ydot.as_ptr(),
                    instance.state.p.as_ptr(),
                    prob.m.as_ptr(),
                    step_obs.as_mut_ptr(),
                );
            }
            for i in 0..n_obs {
                out_obs[step * n_obs + i] = step_obs[i];
            }
        }

        if show_progress && total_steps > 0 {
            print_step_progress(
                "Natv",
                t_eval[step],
                step as f64 / total_steps as f64,
                v_idx,
                &instance.state.y,
                step == total_steps,
            );
        }
    }

    let (micro_t, micro_y, micro_ydot) = flatten_tape(history_record, n);

    Ok(ModelTrajectory {
        t: t_eval.to_vec(),
        y: out_traj,
        obs: out_obs,
        micro_t,
        micro_y,
        micro_ydot,
    })
}

fn execute_single_model(
    prob: &Problem,
    p: &[f64],
    y0: &[f64],
    ydot0: &[f64],
    t_eval: &[f64],
    raw_steps: Option<&[RawProtocolStep]>,
    record_tape: bool,
    show_progress: bool,
    v_idx: i32,
) -> Result<ModelTrajectory, SolverError> {
    let mut instance = SolverInstance::new(prob.n, prob.bw, y0.to_vec(), ydot0.to_vec(), p.to_vec());

    if let Some(steps) = raw_steps {
        if !steps.is_empty() {
            return execute_sequence_trajectory(prob, &mut instance, steps, record_tape, show_progress, v_idx);
        }
    }

    execute_grid_trajectory(prob, &mut instance, t_eval, record_tape, show_progress, v_idx)
}

// =============================================================================
// Concurrency Router: Calling Thread vs Rayon Pool
// =============================================================================

fn dispatch_batch(
    prob_base: &Problem,
    p_batch: &[Vec<f64>],
    y0: &[f64],
    ydot0: &[f64],
    t_eval: &[f64],
    protocol_steps: Option<&[Vec<RawProtocolStep>]>,
    record_tape: bool,
    max_workers: usize,
    spatial_threads: usize,
    show_progress: bool,
    v_idx: i32,
) -> Result<Vec<ModelTrajectory>, SolverError> {
    let total = p_batch.len();
    let completed = AtomicUsize::new(0);

    if total == 1 {
        if let Some(set_threads) = prob_base.fns.set_threads_fn {
            unsafe { set_threads(spatial_threads as i32) };
        }
        let single_protocol = protocol_steps.and_then(|p| p.get(0)).map(|v| v.as_slice());
        let res = execute_single_model(
            prob_base,
            &p_batch[0],
            y0,
            ydot0,
            t_eval,
            single_protocol,
            record_tape,
            show_progress,
            v_idx,
        )?;
        Ok(vec![res])
    } else {
        if let Some(set_threads) = prob_base.fns.set_threads_fn {
            unsafe { set_threads(1) };
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(max_workers)
            .build()
            .map_err(|e| SolverError::Message(format!("Rayon pool init failed: {e}")))?;

        pool.install(|| {
            p_batch
                .par_iter()
                .enumerate()
                .map(|(b_idx, p)| {
                    let model_protocol = protocol_steps
                        .and_then(|protos| protos.get(b_idx))
                        .map(|v| v.as_slice());
                    let res = execute_single_model(
                        prob_base,
                        p,
                        y0,
                        ydot0,
                        t_eval,
                        model_protocol,
                        record_tape,
                        show_progress,
                        v_idx,
                    )?;
                    let c = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    if show_progress {
                        print_batch_progress(c, total);
                    }
                    Ok(res)
                })
                .collect()
        })
    }
}

// =============================================================================
// PyO3 Result Marshaler
// =============================================================================

fn marshal_trajectory_to_py<'py>(
    py: Python<'py>,
    traj: ModelTrajectory,
    n_states: usize,
    n_obs: usize,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
) {
    let steps = traj.t.len();
    let h_len = traj.micro_t.len();

    let t_arr = numpy::ndarray::Array1::from_vec(traj.t).to_pyarray(py);
    let y_arr = numpy::ndarray::Array2::from_shape_vec((steps, n_states), traj.y).unwrap().to_pyarray(py);
    let obs_arr = numpy::ndarray::Array2::from_shape_vec((steps, n_obs), traj.obs).unwrap().to_pyarray(py);
    let mt_arr = numpy::ndarray::Array1::from_vec(traj.micro_t).to_pyarray(py);
    let my_arr = numpy::ndarray::Array2::from_shape_vec((h_len, n_states), traj.micro_y).unwrap().to_pyarray(py);
    let mydot_arr = numpy::ndarray::Array2::from_shape_vec((h_len, n_states), traj.micro_ydot).unwrap().to_pyarray(py);

    (t_arr, y_arr, obs_arr, mt_arr, my_arr, mydot_arr)
}

// =============================================================================
// Public PyO3 Entrypoint
// =============================================================================

#[pyfunction]
#[pyo3(signature = (
    lib_path, y0, ydot0, id, p_batch, m_list, t_eval,
    bandwidth, spatial_diag, max_steps, n_obs,
    cpr_seeds, cpr_ptrs, cpr_rows, cpr_cols, cpr_dense,
    protocol_steps=None, record_tape=false,
    max_workers=1, spatial_threads=1, show_progress=true, v_idx=-1
))]
pub fn solve_native<'py>(
    py: Python<'py>,
    lib_path: String,
    y0: Vec<f64>,
    ydot0: Vec<f64>,
    id: Vec<f64>,
    p_batch: Vec<Vec<f64>>,
    m_list: Vec<f64>,
    t_eval: Vec<f64>,
    bandwidth: isize,
    spatial_diag: Vec<f64>,
    max_steps: Vec<f64>,
    n_obs: usize,
    cpr_seeds: Vec<Vec<f64>>,
    cpr_ptrs: Vec<usize>,
    cpr_rows: Vec<usize>,
    cpr_cols: Vec<usize>,
    cpr_dense: Vec<usize>,
    protocol_steps: Option<Vec<Vec<RawProtocolStep>>>,
    record_tape: bool,
    max_workers: usize,
    spatial_threads: usize,
    show_progress: bool,
    v_idx: i32,
) -> PyResult<Vec<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)>> {
    let lib = unsafe {
        libloading::Library::new(&lib_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
    };
    let fns = load_callbacks(&lib).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let cpr = CprData {
        color_seeds: cpr_seeds,
        color_ptrs: cpr_ptrs,
        color_rows: cpr_rows,
        color_cols: cpr_cols,
        dense_rows: cpr_dense,
    };
    let n = y0.len();
    let prob_base = Problem {
        n,
        bw: bandwidth,
        n_obs,
        id,
        constraints: vec![0.0; n],
        m: m_list,
        spatial_diag,
        max_steps,
        cpr,
        config: SolverConfig::default(),
        fns,
    };

    let proto_slice = protocol_steps.as_deref();

    let results = py.detach(|| {
        dispatch_batch(
            &prob_base,
            &p_batch,
            &y0,
            &ydot0,
            &t_eval,
            proto_slice,
            record_tape,
            max_workers,
            spatial_threads,
            show_progress,
            v_idx,
        )
    });

    let trajectories = results.map_err(|e| match e {
        SolverError::Crash(report) => NativeSolverCrash::new_err(crash_report_to_pydict(py, &report).unbind()),
        SolverError::Message(msg) => pyo3::exceptions::PyRuntimeError::new_err(msg),
    })?;

    Ok(trajectories
        .into_iter()
        .map(|traj| marshal_trajectory_to_py(py, traj, n, n_obs))
        .collect())
}

