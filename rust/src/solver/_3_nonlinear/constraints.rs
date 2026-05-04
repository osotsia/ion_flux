use crate::solver::_3_nonlinear::newton::{NewtonFailure, NewtonResult};

pub fn evaluate(n: usize, y: &[f64], ee: &[f64], phi_0: &[f64], constraints: &[f64], iter: usize) -> NewtonResult {
    let mut min_eta = 1.0;
    let mut violated = false;

    for i in 0..n {
        let c = constraints[i];
        if c == 0.0 { continue; }
        
        if (c > 0.0 && y[i] <= 0.0) || (c < 0.0 && y[i] >= 0.0) {
            violated = true;
            let num = phi_0[i]; 
            let den = phi_0[i] - y[i]; 
            
            if den.abs() > 1e-14 {
                let eta = 0.9 * (num / den);
                if eta > 0.0 && eta < min_eta { min_eta = eta; }
            }
        }
    }

    if violated {
        NewtonResult::DivergedFatal(NewtonFailure::ConstraintsViolated(min_eta.clamp(0.1, 0.9)))
    } else {
        NewtonResult::Converged(iter + 1)
    }
}

#[inline(always)]
pub fn wrms_norm_all(v: &[f64], w: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..v.len() { sum += (v[i] * w[i]).powi(2); }
    (sum / v.len() as f64).sqrt()
}

#[inline(always)]
pub fn wrms_norm_mask(v: &[f64], w: &[f64], id: &[f64], suppress_alg: bool) -> f64 {
    let mut sum = 0.0;
    for i in 0..v.len() {
        if !suppress_alg || id[i] > 0.5 { sum += (v[i] * w[i]).powi(2); }
    }
    (sum / (v.len() as f64)).sqrt()
}