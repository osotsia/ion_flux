use crate::solver::shared::problem::Problem;
use crate::solver::shared::workspace::Workspace;
use crate::solver::_3_nonlinear::newton;
use crate::solver::_3_nonlinear::constraints;
use crate::solver::shared::diagnostics::build_crash_report_json;

pub fn step(
    prob: &Problem,
    wk: &mut Workspace,
    target_dt: f64,
    mut history_cache: Option<&mut Vec<(f64, Vec<f64>, Vec<f64>)>>
) -> Result<(), String> {
    let mut t_local = 0.0;
    let mut error_fails = 0;
    let abs_t = wk.t;

    if wk.diag.accepted_steps == 0 {
        for i in 0..prob.n { wk.weights[i] = 1.0 / (prob.config.rel_tol * wk.y[i].abs() + prob.config.abs_tol); }
        let ypnorm = constraints::wrms_norm_mask(&wk.ydot, &wk.weights, &prob.id, prob.config.suppress_alg);
        let mut h0 = 0.001 * target_dt.abs();
        if ypnorm > 0.5 / h0 { h0 = 0.5 / ypnorm; }

        wk.history.h = h0.max(prob.config.min_dt);
        wk.history.phi[0].copy_from_slice(&wk.y);
        for i in 0..prob.n { wk.history.phi[1][i] = wk.history.h * wk.ydot[i]; }
        wk.history.psi[0] = wk.history.h;
        wk.history.c_j = 1.0 / wk.history.h;
        wk.history.order = 1;
        wk.history.k_used = 0;
        wk.lu_solver.mark_stale();
    }

    while target_dt - t_local > 1e-10 * target_dt.abs() {
        wk.history.h = wk.history.h.min(target_dt - t_local).clamp(prob.config.min_dt, prob.config.max_dt);
        wk.diag.total_steps += 1;

        let ck = wk.history.set_coeffs();
        wk.history.predict(&mut wk.y_pred, &mut wk.ydot_pred);
        for i in 0..prob.n { wk.weights[i] = 1.0 / (prob.config.rel_tol * wk.history.phi[0][i].abs() + prob.config.abs_tol); }

        match newton::solve(prob, wk) {
            newton::NewtonResult::Converged(iters) => {
                let enorm_k = constraints::wrms_norm_mask(&wk.ee, &wk.weights, &prob.id, prob.config.suppress_alg);
                let err_k = wk.history.sigma[wk.history.order] * enorm_k;
                let (mut err_km1, mut err_km2) = (0.0, 0.0);

                if wk.history.order > 1 {
                    let mut delta = vec![0.0; prob.n];
                    for i in 0..prob.n { delta[i] = wk.history.phi[wk.history.order][i] + wk.ee[i]; }
                    err_km1 = wk.history.sigma[wk.history.order - 1] * constraints::wrms_norm_mask(&delta, &wk.weights, &prob.id, prob.config.suppress_alg);
                    if wk.history.order > 2 {
                        for i in 0..prob.n { delta[i] += wk.history.phi[wk.history.order - 1][i]; }
                        err_km2 = wk.history.sigma[wk.history.order - 2] * constraints::wrms_norm_mask(&delta, &wk.weights, &prob.id, prob.config.suppress_alg);
                    }
                }

                if ck * enorm_k > 1.0 {
                    error_fails += 1; wk.diag.rejected_steps += 1;
                    wk.history.restore(); wk.lu_solver.mark_stale();
                    
                    if error_fails == 1 {
                        let mut knew = wk.history.order;
                        let mut err_knew = err_k;
                        if wk.history.order > 1 && (wk.history.order as f64) * err_km1 <= 0.5 * (wk.history.order as f64 + 1.0) * err_k {
                            knew = wk.history.order - 1; err_knew = err_km1;
                        }
                        wk.history.order = knew;
                        wk.history.h *= (0.9 * (2.0 * err_knew + 0.0001).powf(-1.0 / (wk.history.order as f64 + 1.0))).clamp(0.25, 0.9);
                    } else { wk.history.h *= 0.25; wk.history.order = 1; }
                    
                    if wk.history.h <= prob.config.min_dt {
                        let reason = format!("Tolerance Starvation (Step collapsed below min_dt). t={}", abs_t + t_local);
                        let crash_json = build_crash_report_json(&wk.diag, &wk.y, &wk.ydot, &prob.id, &reason);
                        return Err(crash_json);
                    }
                    continue;
                }

                error_fails = 0;
                let mut err_kp1 = 0.0;
                if wk.history.order < wk.history.max_order {
                    let mut delta = vec![0.0; prob.n];
                    for i in 0..prob.n { delta[i] = wk.ee[i] - wk.history.phi[wk.history.order + 1][i]; }
                    err_kp1 = constraints::wrms_norm_mask(&delta, &wk.weights, &prob.id, prob.config.suppress_alg) / ((wk.history.order + 2) as f64);
                }

                let ee_clone = wk.ee.clone();
                wk.history.complete_step(err_k, err_km1, err_km2, err_kp1, &ee_clone, prob.config.min_dt, prob.config.max_dt);
                
                wk.diag.accepted_steps += 1;
                wk.diag.trace_t.push(abs_t + t_local); wk.diag.trace_dt.push(wk.history.h_used);
                wk.diag.trace_order.push(wk.history.order); wk.diag.trace_iters.push(iters); wk.diag.trace_err.push(err_k);
                
                t_local += wk.history.h_used;
                if let Some(ref mut hist) = history_cache { hist.push((abs_t + t_local, wk.y.clone(), wk.ydot.clone())); }
            },
            newton::NewtonResult::DivergedStaleJac(_) | newton::NewtonResult::DivergedFatal(_) => {
                wk.diag.rejected_steps += 1; wk.history.restore(); wk.history.h *= 0.25; wk.lu_solver.mark_stale();
                if wk.history.h <= prob.config.min_dt {
                    let reason = format!("Nonlinear Divergence / Constraint Starvation (Step collapsed below min_dt). t={}", abs_t + t_local);
                    let crash_json = build_crash_report_json(&wk.diag, &wk.y, &wk.ydot, &prob.id, &reason);
                    return Err(crash_json);
                }
            }
        }
    }
    wk.t = abs_t + t_local;
    Ok(())
}