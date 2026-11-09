use std::time::Instant;
use crate::solver::shared::problem::Problem;
use crate::stepper::state::{SystemState, StepScratchpad};
use crate::stepper::history::BdfHistory;
use crate::linear::solver::LinearSolver;
use crate::linear::jacobian;
use crate::solver::shared::diagnostics::Diagnostics;
use crate::nonlinear::constraints;

#[allow(dead_code)]
#[derive(Debug)]
pub enum NewtonFailure {
    NonFiniteResidual,
    SingularJacobian(String),
    ContractionThrashing(f64),
    ConstraintsViolated(f64), 
    MaxItersReached,
}

#[derive(Debug)]
pub enum NewtonResult {
    Converged(usize), 
    Diverged(NewtonFailure),    
}

pub fn solve(
    prob: &Problem,
    state: &mut SystemState,
    history: &mut BdfHistory,
    scratch: &mut StepScratchpad,
    linear_solver: &mut LinearSolver,
    diag: &mut Diagnostics,
) -> NewtonResult {
    let c_j = history.c_j;
    let mut cj_ratio = if history.c_j_old == 0.0 { 1.0 } else { c_j / history.c_j_old };
    if !prob.config.is_realtime && (cj_ratio < 0.6 || cj_ratio > 1.6666666666666667) { 
        linear_solver.mark_stale(); 
    }

    scratch.ee.fill(0.0);
    diag.recent_newton_norms.clear();
    let mut old_fnorm = 0.0;

    for iter in 0..prob.config.max_newton_iters {
        diag.newton_iterations += 1;
        for i in 0..prob.n {
            state.y[i] = scratch.y_pred[i] + scratch.ee[i];
            state.ydot[i] = scratch.ydot_pred[i] + c_j * scratch.ee[i];
        }

        let t_res = Instant::now();
        unsafe { (prob.fns.res_fn)(state.y.as_ptr(), state.ydot.as_ptr(), state.p.as_ptr(), prob.m.as_ptr(), scratch.res.as_mut_ptr()) };
        diag.residual_time_us += t_res.elapsed().as_micros();
        diag.last_res.copy_from_slice(&scratch.res);

        let f_norm = constraints::wrms_norm_all(&scratch.res, &scratch.weights);
        if !f_norm.is_finite() { return NewtonResult::Diverged(NewtonFailure::NonFiniteResidual); }

        for i in 0..prob.n { scratch.dy[i] = -scratch.res[i]; }

        linear_solver.set_gmres_context(
            state.y.as_ptr(),
            state.ydot.as_ptr(),
            state.p.as_ptr(),
            prob.m.as_ptr(),
            c_j,
            prob.id.as_ptr(),
            prob.spatial_diag.as_ptr(),
            prob.fns.jvp_fn,
        );

        if linear_solver.is_stale() {
            if prob.config.is_realtime && diag.accepted_steps > 0 {
                linear_solver.set_stale(false);
            } else {
                let start = Instant::now();
                jacobian::assemble(prob, state, linear_solver, c_j);
                diag.jacobian_assembly_time_us += start.elapsed().as_micros();
                diag.jacobian_evaluations += 1;
                
                if let Err(e) = linear_solver.factorize_from_triplets(diag) {
                    return NewtonResult::Diverged(NewtonFailure::SingularJacobian(e));
                }
                
                history.c_j_old = c_j;
                cj_ratio = 1.0; 
                linear_solver.set_stale(false);
            }
        }

        if let Err(e) = linear_solver.solve(&mut scratch.dy, diag) {
            return NewtonResult::Diverged(NewtonFailure::SingularJacobian(e));
        }

        if !linear_solver.is_gmres() && cj_ratio != 1.0 {
            let scale = 2.0 / (1.0 + cj_ratio);
            for i in 0..prob.n { scratch.dy[i] *= scale; }
        }

        let mut is_clamped = false;
        for i in 0..prob.n {
            if prob.max_steps[i] > 0.0 && scratch.dy[i].abs() > prob.max_steps[i] {
                scratch.dy[i] = scratch.dy[i].signum() * prob.max_steps[i]; is_clamped = true;
            }
            scratch.ee[i] += scratch.dy[i];
        }

        let dy_norm = constraints::wrms_norm_all(&scratch.dy, &scratch.weights);
        diag.last_dy.copy_from_slice(&scratch.dy);
        diag.last_weights.copy_from_slice(&scratch.weights);

        if diag.recent_newton_norms.len() >= 5 { diag.recent_newton_norms.pop_front(); }
        diag.recent_newton_norms.push_back((iter + 1, f_norm, dy_norm));

        if iter == 0 {
            if diag.accepted_steps == 0 {
                let (mut max_r, mut max_idx) = (0.0, 0);
                for i in 0..prob.n { if scratch.res[i].abs() > max_r { max_r = scratch.res[i].abs(); max_idx = i; } }
                diag.t0_max_res = max_r; diag.t0_max_res_idx = max_idx;
            }
            old_fnorm = dy_norm;
            if dy_norm <= 1e-8 * prob.config.eps_newt { return constraints::evaluate(prob.n, &state.y, &scratch.ee, &history.phi[0], &prob.constraints, iter); }
        } else {
            let rate = (dy_norm / old_fnorm).powf(1.0 / iter as f64);
            diag.last_rho = rate;
            
            if !prob.config.is_realtime && rate > prob.config.max_rho && !is_clamped {
                if !linear_solver.is_stale() { linear_solver.mark_stale(); continue; }
                return NewtonResult::Diverged(NewtonFailure::ContractionThrashing(rate));
            }
            let ss = rate / (1.0 - rate);
            if ss * dy_norm <= prob.config.eps_newt { return constraints::evaluate(prob.n, &state.y, &scratch.ee, &history.phi[0], &prob.constraints, iter); }
        }
    }
    NewtonResult::Diverged(NewtonFailure::MaxItersReached)
}
