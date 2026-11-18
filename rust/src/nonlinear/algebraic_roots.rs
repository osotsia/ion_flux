use crate::ffi::SafeEvaluator;
use crate::stepper::problem::Problem;
use crate::stepper::state::{SystemState, StepScratchpad};
use crate::stepper::history::BdfHistory;
use crate::linear::solver::LinearSolver;
use crate::linear::jacobian;
use crate::diagnostics::Diagnostics;

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

    for iter in 0..50 {
        evaluator.evaluate_residual(&state.y, &state.ydot, &state.p, &prob.m, &mut scratch.res);

        let mut max_res = 0.0_f64;
        for i in 0..prob.n {
            if prob.id[i] < 0.5 {
                let r = scratch.res[i].abs();
                if !r.is_finite() || r > max_res {
                    max_res = r;
                }
            }
        }
        final_max_res = max_res;
        iters_used = iter;
        if max_res < 1e-8 {
            converged = true;
            break;
        }

        let needs_refactor = iter == 0 || max_res >= prev_max_res * 0.7;
        prev_max_res = max_res;

        if needs_refactor {
            jacobian::assemble(prob, state, linear_solver, 0.0);
        }

        if linear_solver.is_gmres() {
            for &(r, c, val) in linear_solver.triplets() {
                if r == c && prob.id[r] < 0.5 && val.abs() > 1e-12 {
                    let mut step = -scratch.res[r] / val;
                    if prob.max_steps[r] > 0.0 && step.abs() > prob.max_steps[r] { step = step.signum() * prob.max_steps[r]; }
                    if step.is_finite() { state.y[r] += step * 0.8; }
                }
            }
            iters_used = iter + 1;
        } else {
            for i in 0..prob.n {
                if prob.id[i] > 0.5 { scratch.res[i] = 0.0; }
                scratch.dy[i] = -scratch.res[i];
            }
            if needs_refactor {
                if linear_solver.factorize_from_triplets(diag).is_err() {
                    iters_used = iter + 1;
                    break;
                }
            }
            if linear_solver.solve(&mut scratch.dy, diag).is_err() {
                iters_used = iter + 1;
                break;
            }

            let mut max_step = 0.0;
            for i in 0..prob.n {
                if prob.id[i] < 0.5 {
                    let mut step = scratch.dy[i];
                    if prob.max_steps[i] > 0.0 && step.abs() > prob.max_steps[i] { step = step.signum() * prob.max_steps[i]; }
                    if !step.is_finite() { max_step = f64::INFINITY; break; }
                    state.y[i] += step * 0.8;
                    if step.abs() > max_step { max_step = step.abs(); }
                }
            }
            iters_used = iter + 1;
            if !max_step.is_finite() { break; }
            if max_step < 1e-14 { break; }
        }
    }

    if !state.y.iter().all(|v| v.is_finite()) {
        state.y.copy_from_slice(&y_backup);
        converged = false;
    }

    if !converged && state.y.iter().all(|v| v.is_finite()) {
        evaluator.evaluate_residual(&state.y, &state.ydot, &state.p, &prob.m, &mut scratch.res);
        let mut max_res = 0.0_f64;
        for i in 0..prob.n {
            if prob.id[i] < 0.5 {
                let r = scratch.res[i].abs();
                if !r.is_finite() || r > max_res {
                    max_res = r;
                }
            }
        }
        final_max_res = max_res;
        converged = max_res.is_finite() && max_res < 1e-8;
    }

    diag.ic_converged = converged;
    diag.ic_final_max_res = final_max_res;
    diag.ic_iters = iters_used;

    history.restore_state();
    linear_solver.mark_stale();
    diag.accepted_steps = 0;
    Ok(())
}
