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

#[derive(Clone)]
pub struct Diagnostics {
    pub total_steps: usize,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub newton_iterations: usize,
    pub jacobian_evaluations: usize,
    pub numeric_factorizations: usize,
    
    pub jacobian_assembly_time_us: u128,
    pub linear_solve_time_us: u128,
    pub residual_time_us: u128,
    
    pub trace_t: Vec<f64>,
    pub trace_dt: Vec<f64>,
    pub trace_order: Vec<usize>,
    pub trace_iters: Vec<usize>,
    pub trace_err: Vec<f64>,
    
    pub last_res: Vec<f64>,
    pub last_dy: Vec<f64>,
    pub last_weights: Vec<f64>,
    pub last_rho: f64,
    
    pub jac_max: f64,
    pub jac_min: f64,
    pub t0_max_res: f64,
    pub t0_max_res_idx: usize,
    pub recent_newton_norms: std::collections::VecDeque<(usize, f64, f64)>,
}

impl Diagnostics {
    pub fn new(n: usize) -> Self {
        Self {
            total_steps: 0, accepted_steps: 0, rejected_steps: 0, newton_iterations: 0,
            jacobian_evaluations: 0, numeric_factorizations: 0,
            jacobian_assembly_time_us: 0, linear_solve_time_us: 0, residual_time_us: 0,
            trace_t: Vec::new(), trace_dt: Vec::new(), trace_order: Vec::new(), trace_iters: Vec::new(), trace_err: Vec::new(),
            last_res: vec![0.0; n], last_dy: vec![0.0; n], last_weights: vec![0.0; n], last_rho: 0.0,
            jac_max: 0.0, jac_min: 0.0, t0_max_res: 0.0, t0_max_res_idx: 0, recent_newton_norms: std::collections::VecDeque::new(),
        }
    }

    pub fn build_crash_report(&self, y: &[f64], ydot: &[f64], id: &[f64], reason: String) -> CrashReport {
        let mut raw_offenders: Vec<(usize, f64, f64, f64, f64, f64, f64, bool)> = self.last_res.iter().enumerate()
            .map(|(i, &res)| {
                let weight = self.last_weights.get(i).copied().unwrap_or(0.0);
                let dy = self.last_dy.get(i).copied().unwrap_or(0.0);
                let err = dy * weight;
                let is_diff = id.get(i).unwrap_or(&0.0) > &0.5;
                let y_v = y.get(i).copied().unwrap_or(0.0);
                let ydot_v = ydot.get(i).copied().unwrap_or(0.0);
                (i, res, err.abs(), y_v, ydot_v, dy, weight, is_diff)
            }).collect();
        
        raw_offenders.sort_by(|a, b| {
            let a_nan = !a.1.is_finite() || !a.3.is_finite() || !a.4.is_finite();
            let b_nan = !b.1.is_finite() || !b.3.is_finite() || !b.4.is_finite();
            if a_nan && !b_nan { return std::cmp::Ordering::Less; }
            if !a_nan && b_nan { return std::cmp::Ordering::Greater; }
            b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        let offenders: Vec<Offender> = raw_offenders.into_iter().take(15).map(|(i, res, _err, y_v, ydot_v, dy, weight, is_diff)| {
            Offender {
                index: i, is_diff, y_val: y_v, ydot_val: ydot_v,
                residual: res, proposed_step_dy: dy, solver_weight: weight, weighted_error: _err,
            }
        }).collect();

        let trace = self.recent_newton_norms.iter().map(|&(iter, fnorm, dynorm)| (iter, fnorm, dynorm)).collect();
        let cond_warning = self.jac_max > 0.0 && self.jac_min > 0.0 && (self.jac_max / self.jac_min) > 1e12;

        CrashReport {
            reason,
            accepted_steps: self.accepted_steps,
            t0_max_res: self.t0_max_res,
            t0_max_res_idx: self.t0_max_res_idx,
            jac_max: self.jac_max,
            jac_min: self.jac_min,
            cond_warning,
            trace,
            offenders,
        }
    }
}