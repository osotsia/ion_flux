use crate::solver::shared::problem::Problem;
use crate::solver::shared::workspace::Workspace;
use crate::solver::_2_stepper::bdf;

pub struct TrigInfo {
    pub has_trig: bool,
    pub idx: usize,
    pub size: usize,
    pub is_obs: bool,
    pub op: i32,
    pub val: f64,
}

pub fn check_trigger(prob: &Problem, wk: &mut Workspace, info: &TrigInfo, step_obs: &mut [f64]) -> bool {
    if !info.has_trig { return false; }
    if info.is_obs {
        if let Some(obs_fn) = prob.fns.obs_fn {
            unsafe { obs_fn(wk.y.as_ptr(), wk.ydot.as_ptr(), wk.p.as_ptr(), prob.m.as_ptr(), step_obs.as_mut_ptr()); }
        }
    }
    for i in 0..info.size {
        let v = if info.is_obs { step_obs[info.idx + i] } else { wk.y[info.idx + i] };
        let trig = match info.op {
            1 => v > info.val, 2 => v < info.val, 3 => v >= info.val,
            4 => v <= info.val, 5 => v == info.val, 6 => v != info.val, _ => false,
        };
        if trig { return true; }
    }
    false
}

pub fn execute_bisection(prob: &Problem, wk: &mut Workspace, dt_step: f64, info: &TrigInfo, ckpt: (f64, Vec<f64>, Vec<f64>)) -> f64 {
    wk.restore_state(ckpt.0, ckpt.1.clone(), ckpt.2.clone());
    let mut step_obs = vec![0.0; prob.n_obs];
    let (mut low, mut high) = (0.0, dt_step);

    for _ in 0..15 {
        let mid = (low + high) / 2.0;
        if bdf::step(prob, wk, mid, None).is_err() { break; }
        if check_trigger(prob, wk, info, &mut step_obs) { high = mid; } else { low = mid; }
        wk.restore_state(ckpt.0, ckpt.1.clone(), ckpt.2.clone());
    }
    bdf::step(prob, wk, low, None).unwrap_or(());
    low
}