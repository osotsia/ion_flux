use crate::solver::shared::problem::Problem;
use crate::stepper::state::{SystemState, StepScratchpad};
use crate::stepper::history::BdfHistory;
use crate::linear::solver::LinearSolver;
use crate::solver::shared::diagnostics::{Diagnostics, CrashReport};
use crate::nonlinear::newton;
use crate::nonlinear::constraints;

pub fn step(
    prob: &Problem,
    state: &mut SystemState,
    history: &mut BdfHistory,
    scratch: &mut StepScratchpad,
    linear_solver: &mut LinearSolver,
    diag: &mut Diagnostics,
    target_dt: f64,
    mut history_cache: Option<&mut Vec<(f64, Vec<f64>, Vec<f64>)>>
) -> Result<(), CrashReport> {
    let mut t_local = 0.0;
    let mut error_fails = 0;
    let abs_t = state.t;

    if diag.accepted_steps == 0 {
        for i in 0..prob.n { scratch.weights[i] = 1.0 / (prob.config.rel_tol * state.y[i].abs() + prob.config.abs_tol); }
        let ypnorm = constraints::wrms_norm_mask(&state.ydot, &scratch.weights, &prob.id, prob.config.suppress_alg);
        let mut h0 = 0.001 * target_dt.abs();
        if ypnorm > 0.5 / h0 { h0 = 0.5 / ypnorm; }

        history.h = h0.max(prob.config.min_dt);
        history.phi[0].copy_from_slice(&state.y);
        for i in 0..prob.n { history.phi[1][i] = history.h * state.ydot[i]; }
        history.psi[0] = history.h;
        history.c_j = 1.0 / history.h;
        history.order = 1;
        if prob.config.is_realtime {
            history.max_order = 1;
        }
        history.k_used = 0;
        linear_solver.mark_stale();
    }

    while target_dt - t_local > 1e-10 * target_dt.abs() {
        history.h = history.h.min(target_dt - t_local).clamp(prob.config.min_dt, prob.config.max_dt);
        diag.total_steps += 1;

        let ck = history.set_coeffs();
        history.predict(&mut scratch.y_pred, &mut scratch.ydot_pred);
        for i in 0..prob.n { scratch.weights[i] = 1.0 / (prob.config.rel_tol * history.phi[0][i].abs() + prob.config.abs_tol); }

        match newton::solve(prob, state, history, scratch, linear_solver, diag) {
            newton::NewtonResult::Converged(iters) => {
                let enorm_k = constraints::wrms_norm_mask(&scratch.ee, &scratch.weights, &prob.id, prob.config.suppress_alg);
                let err_k = history.sigma[history.order] * enorm_k;
                let (mut err_km1, mut err_km2) = (0.0, 0.0);

                if !prob.config.is_realtime && history.order > 1 {
                    for i in 0..prob.n { scratch.err_delta[i] = history.phi[history.order][i] + scratch.ee[i]; }
                    err_km1 = history.sigma[history.order - 1] * constraints::wrms_norm_mask(&scratch.err_delta, &scratch.weights, &prob.id, prob.config.suppress_alg);
                    if history.order > 2 {
                        for i in 0..prob.n { scratch.err_delta[i] += history.phi[history.order - 1][i]; }
                        err_km2 = history.sigma[history.order - 2] * constraints::wrms_norm_mask(&scratch.err_delta, &scratch.weights, &prob.id, prob.config.suppress_alg);
                    }
                }

                if !prob.config.is_realtime && ck * enorm_k > 1.0 {
                    error_fails += 1; diag.rejected_steps += 1;
                    history.restore(); linear_solver.mark_stale();
                    
                    if error_fails == 1 {
                        let mut knew = history.order;
                        let mut err_knew = err_k;
                        if history.order > 1 && (history.order as f64) * err_km1 <= 0.5 * (history.order as f64 + 1.0) * err_k {
                            knew = history.order - 1; err_knew = err_km1;
                        }
                        history.order = knew;
                        history.h *= (0.9 * (2.0 * err_knew + 0.0001).powf(-1.0 / (history.order as f64 + 1.0))).clamp(0.25, 0.9);
                    } else { history.h *= 0.25; history.order = 1; }
                    
                    if history.h <= prob.config.min_dt {
                        let reason = format!("Tolerance Starvation (Step collapsed below min_dt). t={}", abs_t + t_local);
                        return Err(diag.build_crash_report(&state.y, &state.ydot, &prob.id, reason));
                    }
                    continue;
                }

                error_fails = 0;
                let mut err_kp1 = 0.0;
                if !prob.config.is_realtime && history.order < history.max_order {
                    for i in 0..prob.n { scratch.err_delta[i] = scratch.ee[i] - history.phi[history.order + 1][i]; }
                    err_kp1 = constraints::wrms_norm_mask(&scratch.err_delta, &scratch.weights, &prob.id, prob.config.suppress_alg) / ((history.order + 2) as f64);
                }

                if prob.config.is_realtime {
                    history.k_used = 1;
                    history.h_used = history.h;
                    for i in 0..prob.n { history.phi[1][i] += scratch.ee[i]; }
                    for i in 0..prob.n { history.phi[0][i] += history.phi[1][i]; }
                } else {
                    history.complete_step(err_k, err_km1, err_km2, err_kp1, &scratch.ee, prob.config.min_dt, prob.config.max_dt);
                }
                
                diag.accepted_steps += 1;
                diag.trace_t.push(abs_t + t_local); diag.trace_dt.push(history.h_used);
                diag.trace_order.push(history.order); diag.trace_iters.push(iters); diag.trace_err.push(err_k);
                
                t_local += history.h_used;
                if let Some(ref mut hist) = history_cache { hist.push((abs_t + t_local, state.y.clone(), state.ydot.clone())); }
            },
            newton::NewtonResult::Diverged(failure) => {
                if prob.config.is_realtime {
                    for i in 0..prob.n {
                        let c = prob.constraints[i];
                        if c > 0.0 && state.y[i] < 0.0 { state.y[i] = 0.0; }
                        else if c < 0.0 && state.y[i] > 0.0 { state.y[i] = 0.0; }
                    }
                    diag.accepted_steps += 1;
                    t_local += history.h;
                    break;
                }
                diag.rejected_steps += 1; history.restore(); history.h *= 0.25; linear_solver.mark_stale();
                if history.h <= prob.config.min_dt {
                    let reason = format!("Nonlinear Divergence: {:?} (Step collapsed below min_dt). t={}", failure, abs_t + t_local);
                    return Err(diag.build_crash_report(&state.y, &state.ydot, &prob.id, reason));
                }
            }
        }
    }
    state.t = abs_t + t_local;
    Ok(())
}

pub fn advance_to(
    prob: &Problem,
    state: &mut SystemState,
    history: &mut BdfHistory,
    scratch: &mut StepScratchpad,
    linear_solver: &mut LinearSolver,
    diag: &mut Diagnostics,
    target_t: f64,
    history_cache: Option<&mut Vec<(f64, Vec<f64>, Vec<f64>)>>,
) -> Result<(), CrashReport> {
    let dt = target_t - state.t;
    if dt <= 1e-14 {
        return Ok(());
    }
    step(prob, state, history, scratch, linear_solver, diag, dt, history_cache)
}
