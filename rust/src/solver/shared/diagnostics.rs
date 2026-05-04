use std::time::SystemTime;
use std::fs::File;
use std::io::Write;

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

    pub fn generate_timestamp() -> u64 {
        SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()
    }
}

pub fn dump_crash_report(diag: &Diagnostics, y: &[f64], ydot: &[f64], id: &[f64], reason: &str) {
    std::fs::create_dir_all("ion_flux_diagnostics").ok();
    
    let mut offenders: Vec<(usize, f64, f64, f64, f64, f64, f64, bool)> = diag.last_res.iter().enumerate()
        .map(|(i, &res)| {
            let weight = diag.last_weights.get(i).copied().unwrap_or(0.0);
            let dy = diag.last_dy.get(i).copied().unwrap_or(0.0);
            let err = dy * weight;
            let is_diff = id.get(i).unwrap_or(&0.0) > &0.5;
            let y_v = y.get(i).copied().unwrap_or(0.0);
            let ydot_v = ydot.get(i).copied().unwrap_or(0.0);
            (i, res, err.abs(), y_v, ydot_v, dy, weight, is_diff)
        }).collect();
    
    offenders.sort_by(|a, b| {
        let a_nan = !a.1.is_finite() || !a.3.is_finite() || !a.4.is_finite();
        let b_nan = !b.1.is_finite() || !b.3.is_finite() || !b.4.is_finite();
        if a_nan && !b_nan { return std::cmp::Ordering::Less; }
        if !a_nan && b_nan { return std::cmp::Ordering::Greater; }
        b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    let top_offenders: Vec<String> = offenders.into_iter().take(15).map(|(i, res, err, y_v, ydot_v, dy, weight, is_diff)| {
        let eq_type = if is_diff { "ODE/PDE" } else { "Algebraic" };
        format!(
            "{{\n      \"index\": {},\n      \"type\": \"{}\",\n      \"y_val\": {:.3e},\n      \"ydot_val\": {:.3e},\n      \"residual\": {:.3e},\n      \"proposed_step_dy\": {:.3e},\n      \"solver_weight\": {:.3e},\n      \"weighted_error\": {:.3e}\n    }}",
            i, eq_type, y_v, ydot_v, res, dy, weight, err
        )
    }).collect();

    let mut trace_str = String::new();
    let trace_len = diag.recent_newton_norms.len();
    for (i, &(iter, fnorm, dynorm)) in diag.recent_newton_norms.iter().enumerate() {
        trace_str.push_str(&format!("{{\"iter\": {}, \"residual_norm\": {:.3e}, \"step_norm\": {:.3e}}}", iter, fnorm, dynorm));
        if i < trace_len - 1 { trace_str.push_str(",\n    "); }
    }

    let cond_warning = diag.jac_max > 0.0 && diag.jac_min > 0.0 && (diag.jac_max / diag.jac_min) > 1e12;
    let ts = Diagnostics::generate_timestamp();
    
    if let Ok(mut file) = File::create(format!("ion_flux_diagnostics/crash_{}.json", ts)) {
        let json = format!(
            "{{\n  \"status\": \"CRASH\",\n  \"reason\": \"{}\",\n  \"accepted_steps\": {},\n  \"initialization_health\": {{\n    \"t0_max_residual\": {:.3e},\n    \"t0_max_residual_index\": {}\n  }},\n  \"jacobian_health\": {{\n    \"max_element\": {:.3e},\n    \"min_nonzero_element\": {:.3e},\n    \"condition_warning\": {}\n  }},\n  \"newton_thrashing_trace\": [\n    {}\n  ],\n  \"top_offenders\": [\n    {}\n  ]\n}}",
            reason, diag.accepted_steps, diag.t0_max_res, diag.t0_max_res_idx,
            diag.jac_max, diag.jac_min, cond_warning, trace_str, top_offenders.join(",\n    ")
        );
        file.write_all(json.as_bytes()).ok();
    }
}

pub fn dump_diagnostics(diag: &Diagnostics) {
    std::fs::create_dir_all("ion_flux_diagnostics").ok();
    let ts = Diagnostics::generate_timestamp();
    if let Ok(mut file) = File::create(format!("ion_flux_diagnostics/profile_{}.json", ts)) {
        let json = format!(
            "{{\n  \"status\": \"SUCCESS\",\n  \"accepted_steps\": {},\n  \"rejected_steps\": {},\n  \"newton_iterations\": {},\n  \"jacobian_evals\": {},\n  \"numeric_lus\": {},\n  \"timers_us\": {{\n    \"residual\": {},\n    \"jac_assembly\": {},\n    \"lu_solve\": {}\n  }}\n}}",
            diag.accepted_steps, diag.rejected_steps, diag.newton_iterations,
            diag.jacobian_evaluations, diag.numeric_factorizations,
            diag.residual_time_us, diag.jacobian_assembly_time_us, diag.linear_solve_time_us
        );
        file.write_all(json.as_bytes()).ok();
    }
}