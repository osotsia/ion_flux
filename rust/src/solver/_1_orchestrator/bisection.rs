use crate::solver::shared::problem::Problem;
use crate::stepper::state::{SystemState, StepScratchpad};
use crate::stepper::history::BdfHistory;
use crate::linear::solver::LinearSolver;
use crate::solver::shared::diagnostics::Diagnostics;
use crate::stepper::bdf;

pub struct TrigInfo {
    pub has_trig: bool,
    pub idx: usize,
    pub size: usize,
    pub is_obs: bool,
    pub op: i32,
    pub val: f64,
}

pub fn check_trigger(prob: &Problem, state: &SystemState, info: &TrigInfo, step_obs: &mut [f64]) -> bool {
    if !info.has_trig { return false; }
    if info.is_obs {
        if let Some(obs_fn) = prob.fns.obs_fn {
            unsafe { obs_fn(state.y.as_ptr(), state.ydot.as_ptr(), state.p.as_ptr(), prob.m.as_ptr(), step_obs.as_mut_ptr()); }
        }
    }
    for i in 0..info.size {
        let v = if info.is_obs { step_obs[info.idx + i] } else { state.y[info.idx + i] };
        let trig = match info.op {
            1 => v > info.val, 2 => v < info.val, 3 => v >= info.val,
            4 => v <= info.val, 5 => v == info.val, 6 => v != info.val, _ => false,
        };
        if trig { return true; }
    }
    false
}

pub fn execute_bisection(
    prob: &Problem,
    state: &mut SystemState,
    history: &mut BdfHistory,
    scratch: &mut StepScratchpad,
    linear_solver: &mut LinearSolver,
    diag: &mut Diagnostics,
    dt_step: f64,
    info: &TrigInfo,
    ckpt: (f64, Vec<f64>, Vec<f64>)
) -> f64 {
    state.restore(ckpt.0, ckpt.1.clone(), ckpt.2.clone());
    history.restore_state();
    linear_solver.mark_stale();
    let mut step_obs = vec![0.0; prob.n_obs];
    let (mut low, mut high) = (0.0, dt_step);

    for _ in 0..15 {
        let mid = (low + high) / 2.0;
        if bdf::step(prob, state, history, scratch, linear_solver, diag, mid, None).is_err() { break; }
        if check_trigger(prob, state, info, &mut step_obs) { high = mid; } else { low = mid; }
        state.restore(ckpt.0, ckpt.1.clone(), ckpt.2.clone());
        history.restore_state();
        linear_solver.mark_stale();
    }
    bdf::step(prob, state, history, scratch, linear_solver, diag, low, None).unwrap_or(());
    low
}