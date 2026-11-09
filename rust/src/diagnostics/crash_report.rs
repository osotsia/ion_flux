#[derive(Clone, Debug)]
pub struct Offender {
    pub index: usize,
    pub is_diff: bool,
    pub y_val: f64,
    pub ydot_val: f64,
    pub residual: f64,
    pub proposed_step_dy: f64,
    pub solver_weight: f64,
    pub weighted_error: f64,
}

#[derive(Clone, Debug)]
pub struct CrashReport {
    pub reason: String,
    pub accepted_steps: usize,
    pub t0_max_res: f64,
    pub t0_max_res_idx: usize,
    pub jac_max: f64,
    pub jac_min: f64,
    pub cond_warning: bool,
    pub trace: Vec<(usize, f64, f64)>,
    pub offenders: Vec<Offender>,
}
