use crate::stepper::problem::Problem;
use crate::stepper::state::{TimeStepper, Checkpoint};

#[derive(Clone, Debug)]
pub struct TrigInfo {
    pub has_trig: bool,
    pub idx: usize,
    pub size: usize,
    pub is_obs: bool,
    pub op: i32,
    pub val: f64,
}

pub fn check_trigger<S: TimeStepper>(prob: &Problem, stepper: &S, info: &TrigInfo, step_obs: &mut [f64]) -> bool {
    if !info.has_trig { return false; }
    let y = stepper.y();
    let ydot = stepper.ydot();
    let p = stepper.p();
    if info.is_obs {
        if let Some(obs_fn) = prob.fns.obs_fn {
            unsafe { obs_fn(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), prob.m.as_ptr(), step_obs.as_mut_ptr()); }
        }
    }
    for i in 0..info.size {
        let v = if info.is_obs { step_obs[info.idx + i] } else { y[info.idx + i] };
        let trig = match info.op {
            1 => v > info.val, 2 => v < info.val, 3 => v >= info.val,
            4 => v <= info.val, 5 => v == info.val, 6 => v != info.val, _ => false,
        };
        if trig { return true; }
    }
    false
}

pub fn execute_bisection<S: TimeStepper>(
    prob: &Problem,
    stepper: &mut S,
    dt_step: f64,
    info: &TrigInfo,
    ckpt: &Checkpoint,
    mut history_cache: Option<&mut Vec<(f64, Vec<f64>, Vec<f64>)>>,
) -> f64 {
    let mut step_obs = vec![0.0; prob.n_obs];
    let (mut low, mut high) = (0.0, dt_step);

    for _ in 0..15 {
        stepper.rollback(ckpt);

        let mid = (low + high) / 2.0;
        if stepper.step(prob, mid, None).is_err() {
            high = mid;
            continue;
        }
        if check_trigger(prob, stepper, info, &mut step_obs) {
            high = mid;
        } else {
            low = mid;
        }
    }

    stepper.rollback(ckpt);
    if low > 1e-12 {
        let hist_step = match history_cache {
            Some(ref mut h) => Some(&mut **h),
            None => None,
        };
        if stepper.step(prob, low, hist_step).is_err() {
            stepper.rollback(ckpt);
            return 0.0;
        }
    }
    low
}
