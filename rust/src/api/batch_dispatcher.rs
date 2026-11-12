use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyArray2, ToPyArray};
use rayon::prelude::*;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::stepper::problem::{Problem, Callbacks, CprData, SolverConfig};
use crate::stepper::state::SolverInstance;
use crate::control::sequence::{ProtocolStep, run_sequence};
use crate::control::bisection::TrigInfo;
use crate::api::{NativeSolverCrash, SolverError, crash_report_to_pydict, diagnostics_to_pydict};

pub type RawProtocolStep = (i32, f64, f64, (bool, usize, usize, bool, i32, f64), usize, usize, usize);

pub struct ModelTrajectory {
    pub t: Vec<f64>,
    pub y: Vec<f64>,
    pub obs: Vec<f64>,
    pub micro_t: Vec<f64>,
    pub micro_y: Vec<f64>,
    pub micro_ydot: Vec<f64>,
    pub diag: crate::diagnostics::Diagnostics,
}

pub struct SolvePayload<'a> {
    pub y0: &'a [f64],
    pub ydot0: &'a [f64],
    pub t_eval: &'a [f64],
    pub protocol_steps: Option<&'a [Vec<RawProtocolStep>]>,
    pub record_tape: bool,
    pub max_workers: usize,
    pub spatial_threads: usize,
    pub show_progress: bool,
    pub v_idx: i32,
}

struct TrajectoryBuilder;
impl TrajectoryBuilder {
    fn flatten_tape(history: Option<Vec<(f64, Vec<f64>, Vec<f64>)>>, n_states: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
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

    fn marshal_to_py<'py>(
        py: Python<'py>, traj: ModelTrajectory, n_states: usize, n_obs: usize
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>) {
        let steps = traj.t.len();
        let h_len = traj.micro_t.len();
        let diag_dict = diagnostics_to_pydict(py, &traj.diag);
        (
            numpy::ndarray::Array1::from_vec(traj.t).to_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec((steps, n_states), traj.y).unwrap().to_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec((steps, n_obs), traj.obs).unwrap().to_pyarray(py),
            numpy::ndarray::Array1::from_vec(traj.micro_t).to_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec((h_len, n_states), traj.micro_y).unwrap().to_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec((h_len, n_states), traj.micro_ydot).unwrap().to_pyarray(py),
            diag_dict,
        )
    }
}

fn load_callbacks(lib: &libloading::Library) -> Result<Callbacks, String> {
    unsafe {
        Ok(Callbacks {
            res_fn: *lib.get(b"evaluate_residual\0").map_err(|e| format!("Missing evaluate_residual: {e}"))?,
            obs_fn: lib.get(b"evaluate_observables\0").map(|s| *s).ok(),
            jvp_fn: lib.get(b"evaluate_jvp\0").map(|s| *s).ok(),
            vjp_fn: lib.get(b"evaluate_vjp\0").map(|s| *s).ok(),
            set_threads_fn: lib.get(b"set_spatial_threads\0").map(|s| *s).ok(),
        })
    }
}

fn parse_protocol_steps(raw_steps: &[RawProtocolStep]) -> Vec<ProtocolStep> {
    raw_steps.iter().map(|s| ProtocolStep {
        s_type: s.0, target_val: s.1, t_limit: s.2,
        trig: TrigInfo { has_trig: s.3.0, idx: s.3.1, size: s.3.2, is_obs: s.3.3, op: s.3.4, val: s.3.5 },
        p_mode: s.4, p_i: s.5, p_v: s.6,
    }).collect()
}

fn print_progress(tag: &str, t: f64, pct: f64, v_idx: i32, y: &[f64], is_final: bool, completed: Option<(usize, usize)>) {
    let p = pct.clamp(0.0, 1.0);
    let v_str = if v_idx >= 0 && (v_idx as usize) < y.len() { format!(" | V: {:.3}V", y[v_idx as usize]) } else { String::new() };
    let batch_str = if let Some((c, tot)) = completed { format!(" | {}/{} models", c, tot) } else { String::new() };

    if is_final {
        print!("\r▶ {:<4} [██████████████████████████████] 100.0% | t: {:.1}s{}{}   \n", tag, t, v_str, batch_str);
    } else {
        let filled = (p * 30.0) as usize;
        let bar: String = std::iter::repeat('█').take(filled).chain(std::iter::repeat('-').take(30 - filled)).collect();
        print!("\r▶ {:<4} [{}] {:.1}% | t: {:.1}s{}{}   ", tag, bar, p * 100.0, t, v_str, batch_str);
    }
    let _ = std::io::stdout().flush();
}

struct BatchExecutor<'a> {
    prob: &'a Problem,
    payload: &'a SolvePayload<'a>,
}

impl<'a> BatchExecutor<'a> {
    fn execute_model(&self, p: &[f64], steps: Option<&[RawProtocolStep]>) -> Result<ModelTrajectory, SolverError> {
        let pmu = crate::diagnostics::PmuGuard::new();
        let mut instance = SolverInstance::new(self.prob.n, self.prob.bw, self.payload.y0.to_vec(), self.payload.ydot0.to_vec(), p.to_vec());
        let mut out_t = Vec::new();
        let mut out_traj = Vec::new();
        let mut out_obs = Vec::new();
        let mut history = if self.payload.record_tape { Some(Vec::new()) } else { None };

        if let Some(raw_steps) = steps {
            if !raw_steps.is_empty() {
                run_sequence(self.prob, &mut instance, &parse_protocol_steps(raw_steps), &mut out_t, &mut out_traj, &mut out_obs, history.as_mut(), self.payload.show_progress, self.payload.v_idx)?;
                let (cycles, instr) = pmu.finish();
                instance.diag.record_pmu(cycles, instr);
                let (micro_t, micro_y, micro_ydot) = TrajectoryBuilder::flatten_tape(history, self.prob.n);
                return Ok(ModelTrajectory { t: out_t, y: out_traj, obs: out_obs, micro_t, micro_y, micro_ydot, diag: instance.diag });
            }
        }

        instance.calc_algebraic_roots(self.prob).map_err(SolverError::Message)?;
        let n = self.prob.n;
        let n_obs = self.prob.n_obs;
        let t_eval = self.payload.t_eval;
        let n_steps = t_eval.len();
        
        out_traj.resize(n_steps * n, 0.0);
        out_obs.resize(n_steps * n_obs, 0.0);
        
        let extract_obs = |inst: &SolverInstance, offset: usize, out: &mut [f64]| {
            if let Some(obs_fn) = self.prob.fns.obs_fn {
                unsafe { obs_fn(inst.state.y.as_ptr(), inst.state.ydot.as_ptr(), inst.state.p.as_ptr(), self.prob.m.as_ptr(), out[offset..].as_mut_ptr()); }
            }
        };

        out_traj[..n].copy_from_slice(&instance.state.y);
        extract_obs(&instance, 0, &mut out_obs);
        if let Some(ref mut hist) = history { hist.push((t_eval[0], instance.state.y.clone(), instance.state.ydot.clone())); }

        let total_steps = n_steps.saturating_sub(1);
        for step in 1..n_steps {
            instance.step(self.prob, t_eval[step] - t_eval[step - 1], history.as_mut()).map_err(SolverError::Crash)?;
            out_traj[step * n..(step + 1) * n].copy_from_slice(&instance.state.y);
            extract_obs(&instance, step * n_obs, &mut out_obs);

            if self.payload.show_progress && total_steps > 0 {
                print_progress("Natv", t_eval[step], step as f64 / total_steps as f64, self.payload.v_idx, &instance.state.y, step == total_steps, None);
            }
        }

        let (cycles, instr) = pmu.finish();
        instance.diag.record_pmu(cycles, instr);
        let (micro_t, micro_y, micro_ydot) = TrajectoryBuilder::flatten_tape(history, n);
        Ok(ModelTrajectory { t: t_eval.to_vec(), y: out_traj, obs: out_obs, micro_t, micro_y, micro_ydot, diag: instance.diag })
    }

    fn run(&self, p_batch: &[Vec<f64>]) -> Result<Vec<ModelTrajectory>, SolverError> {
        let total = p_batch.len();
        if total == 1 {
            if let Some(set_threads) = self.prob.fns.set_threads_fn { unsafe { set_threads(self.payload.spatial_threads as i32) }; }
            let steps = self.payload.protocol_steps.and_then(|p| p.get(0)).map(|v| v.as_slice());
            Ok(vec![self.execute_model(&p_batch[0], steps)?])
        } else {
            if let Some(set_threads) = self.prob.fns.set_threads_fn { unsafe { set_threads(1) }; }
            let completed = AtomicUsize::new(0);
            let pool = rayon::ThreadPoolBuilder::new().num_threads(self.payload.max_workers).build().map_err(|e| SolverError::Message(e.to_string()))?;
            
            pool.install(|| {
                p_batch.par_iter().enumerate().map(|(idx, p)| {
                    let steps = self.payload.protocol_steps.and_then(|protos| protos.get(idx)).map(|v| v.as_slice());
                    let res = self.execute_model(p, steps)?;
                    let c = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    if self.payload.show_progress { print_progress("Btch", 0.0, c as f64 / total as f64, -1, &[], c == total, Some((c, total))); }
                    Ok(res)
                }).collect()
            })
        }
    }
}

#[pyfunction]
#[pyo3(signature = (
    lib_path, y0, ydot0, id, p_batch, m_list, t_eval, bandwidth, spatial_diag, max_steps, n_obs,
    cpr_seeds, cpr_ptrs, cpr_rows, cpr_cols, cpr_dense, protocol_steps=None, record_tape=false,
    max_workers=1, spatial_threads=1, show_progress=true, v_idx=-1
))]
pub fn solve_native<'py>(
    py: Python<'py>, lib_path: String, y0: Vec<f64>, ydot0: Vec<f64>, id: Vec<f64>, p_batch: Vec<Vec<f64>>, 
    m_list: Vec<f64>, t_eval: Vec<f64>, bandwidth: isize, spatial_diag: Vec<f64>, max_steps: Vec<f64>, 
    n_obs: usize, cpr_seeds: Vec<Vec<f64>>, cpr_ptrs: Vec<usize>, cpr_rows: Vec<usize>, cpr_cols: Vec<usize>, cpr_dense: Vec<usize>,
    protocol_steps: Option<Vec<Vec<RawProtocolStep>>>, record_tape: bool, max_workers: usize, spatial_threads: usize, 
    show_progress: bool, v_idx: i32,
) -> PyResult<Vec<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)>> {
    
    let lib = unsafe { libloading::Library::new(&lib_path).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))? };
    let fns = load_callbacks(&lib).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let cpr = CprData { color_seeds: cpr_seeds, color_ptrs: cpr_ptrs, color_rows: cpr_rows, color_cols: cpr_cols, dense_rows: cpr_dense };
    
    let prob_base = Problem {
        n: y0.len(), bw: bandwidth, n_obs, id, constraints: vec![0.0; y0.len()], m: m_list, spatial_diag, max_steps, cpr, config: SolverConfig::default(), fns,
    };

    let payload = SolvePayload {
        y0: &y0, ydot0: &ydot0, t_eval: &t_eval, protocol_steps: protocol_steps.as_deref(),
        record_tape, max_workers, spatial_threads, show_progress, v_idx,
    };

    let results = py.detach(|| BatchExecutor { prob: &prob_base, payload: &payload }.run(&p_batch));

    let trajectories = results.map_err(|e| match e {
        SolverError::Crash(report) => NativeSolverCrash::new_err(crash_report_to_pydict(py, &report).unbind()),
        SolverError::Message(msg) => pyo3::exceptions::PyRuntimeError::new_err(msg),
    })?;

    Ok(trajectories.into_iter().map(|traj| TrajectoryBuilder::marshal_to_py(py, traj, prob_base.n, prob_base.n_obs)).collect())
}