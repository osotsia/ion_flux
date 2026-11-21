use crate::ffi::SafeEvaluator;
use crate::stepper::problem::Problem;
use crate::stepper::state::{SystemState, StepScratchpad};
use crate::stepper::history::BdfHistory;
use crate::linear::solver::LinearSolver;
use crate::linear::jacobian;
use crate::diagnostics::Diagnostics;

const ALG_RES_TOL: f64 = 1e-8;
const ALG_DAMPING: f64 = 0.8;
const ALG_MAX_ITERS: usize = 50;

/// Evaluates `res` and returns `(max_abs_res, max_row_scaled_res)` across algebraic rows (`id < 0.5`).
fn eval_alg_residual_norms(
    prob: &Problem,
    state: &SystemState,
    evaluator: &SafeEvaluator,
    res_buf: &mut [f64],
    row_scales: Option<&[f64]>,
) -> (f64, f64) {
    evaluator.evaluate_residual(&state.y, &state.ydot, &state.p, &prob.m, res_buf);

    let mut max_res = 0.0_f64;
    let mut max_scaled_res = 0.0_f64;
    for i in 0..prob.n {
        if prob.id[i] < 0.5 {
            let r = res_buf[i].abs();
            if !r.is_finite() {
                return (f64::NAN, f64::NAN);
            }
            max_res = max_res.max(r);

            let scale = row_scales.and_then(|s| s.get(i).copied()).unwrap_or(1.0);
            let sr = r * scale;
            if !sr.is_finite() {
                return (f64::NAN, f64::NAN);
            }
            max_scaled_res = max_scaled_res.max(sr);
        }
    }
    (max_res, max_scaled_res)
}

#[inline]
fn clamp_step(raw: f64, limit: f64) -> f64 {
    if limit > 0.0 && raw.abs() > limit {
        raw.signum() * limit
    } else {
        raw
    }
}

/// Solves `F(y_diff, y_alg, ydot_diff=0, p) = 0` for consistent initial algebraic states `y_alg`.
pub fn calc_algebraic_roots(
    prob: &Problem,
    state: &mut SystemState,
    history: &mut BdfHistory,
    scratch: &mut StepScratchpad,
    linear_solver: &mut LinearSolver,
    diag: &mut Diagnostics,
) -> Result<(), String> {
    let evaluator = SafeEvaluator::new(prob.fns, prob.n, state.p.len(), prob.n_obs);
    let y_backup = state.y.clone();
    let mut prev_max_res = f64::INFINITY;
    let mut converged = false;
    let mut final_max_res = 0.0_f64;
    let mut iters_used = 0_usize;
    let mut active_row_scales: Option<Vec<f64>> = None;

    for iter in 0..ALG_MAX_ITERS {
        let (max_res, max_scaled_res) = eval_alg_residual_norms(
            prob,
            state,
            &evaluator,
            &mut scratch.res,
            active_row_scales.as_deref(),
        );
        final_max_res = max_res;
        iters_used = iter;

        if !max_res.is_finite() {
            break;
        }
        let check_metric = if active_row_scales.is_some() { max_scaled_res } else { max_res };
        if check_metric < ALG_RES_TOL {
            converged = true;
            break;
        }

        let needs_refactor = iter == 0 || max_res >= prev_max_res * 0.7;
        prev_max_res = max_res;
        if needs_refactor {
            jacobian::assemble(prob, state, linear_solver, 0.0);
        }

        iters_used = iter + 1;
        if linear_solver.is_gmres() {
            for &(r, c, val) in linear_solver.triplets() {
                if r == c && prob.id[r] < 0.5 && val.abs() > 1e-12 {
                    let step = clamp_step(-scratch.res[r] / val, prob.max_steps[r]);
                    if step.is_finite() {
                        state.y[r] += step * ALG_DAMPING;
                    }
                }
            }
            continue;
        }

        for i in 0..prob.n {
            scratch.dy[i] = if prob.id[i] > 0.5 { 0.0 } else { -scratch.res[i] };
        }
        if needs_refactor {
            if linear_solver.factorize_from_triplets(diag).is_err() {
                break;
            }
            if let Some(lu) = linear_solver.as_sparse_lu() {
                active_row_scales = Some(lu.row_scales.clone());
            }
        }
        if linear_solver.solve(&mut scratch.dy, diag).is_err() {
            break;
        }

        let mut max_step = 0.0_f64;
        for i in 0..prob.n {
            if prob.id[i] < 0.5 {
                let step = clamp_step(scratch.dy[i], prob.max_steps[i]);
                if !step.is_finite() {
                    max_step = f64::INFINITY;
                    break;
                }
                state.y[i] += step * ALG_DAMPING;
                max_step = max_step.max(step.abs());
            }
        }
        if !max_step.is_finite() || max_step < 1e-14 {
            break;
        }
    }

    if !state.y.iter().all(|v| v.is_finite()) {
        state.y.copy_from_slice(&y_backup);
        converged = false;
    }

    if !converged && state.y.iter().all(|v| v.is_finite()) {
        let (max_res, max_scaled_res) = eval_alg_residual_norms(
            prob,
            state,
            &evaluator,
            &mut scratch.res,
            active_row_scales.as_deref(),
        );
        if max_res.is_finite() {
            final_max_res = max_res;
            let check_metric = if active_row_scales.is_some() { max_scaled_res } else { max_res };
            converged = check_metric.is_finite() && check_metric < ALG_RES_TOL;
        } else {
            state.y.copy_from_slice(&y_backup);
            final_max_res = f64::NAN;
            converged = false;
        }
    }

    diag.ic_converged = converged;
    diag.ic_final_max_res = final_max_res;
    diag.ic_iters = iters_used;

    history.restore_state();
    linear_solver.mark_stale();
    diag.accepted_steps = 0;
    Ok(())
}
