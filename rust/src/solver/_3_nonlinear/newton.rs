use std::time::Instant;
use crate::solver::shared::problem::Problem;
use crate::solver::shared::workspace::Workspace;
use crate::solver::_4_linear::{jacobian, gmres};
use crate::solver::_3_nonlinear::constraints;

pub enum NewtonFailure {
    NonFiniteResidual,
    SingularJacobian(String),
    ContractionThrashing(f64),
    ConstraintsViolated(f64), 
    MaxItersReached,
}

pub enum NewtonResult {
    Converged(usize), 
    DivergedStaleJac(NewtonFailure), 
    DivergedFatal(NewtonFailure),    
}

pub fn solve(prob: &Problem, wk: &mut Workspace) -> NewtonResult {
    let c_j = wk.history.c_j;
    let mut cj_ratio = if wk.history.c_j_old == 0.0 { 1.0 } else { c_j / wk.history.c_j_old };
    if cj_ratio < 0.6 || cj_ratio > 1.6666666666666667 { wk.lu_solver.mark_stale(); }

    wk.ee.fill(0.0);
    wk.diag.recent_newton_norms.clear();
    let mut old_fnorm = 0.0;

    for iter in 0..prob.config.max_newton_iters {
        wk.diag.newton_iterations += 1;
        for i in 0..prob.n {
            wk.y[i] = wk.y_pred[i] + wk.ee[i];
            wk.ydot[i] = wk.ydot_pred[i] + c_j * wk.ee[i];
        }

        let t_res = Instant::now();
        unsafe { (prob.fns.res_fn)(wk.y.as_ptr(), wk.ydot.as_ptr(), wk.p.as_ptr(), prob.m.as_ptr(), wk.res.as_mut_ptr()) };
        wk.diag.residual_time_us += t_res.elapsed().as_micros();
        wk.diag.last_res.copy_from_slice(&wk.res);

        let f_norm = constraints::wrms_norm_all(&wk.res, &wk.weights);
        if !f_norm.is_finite() { return NewtonResult::DivergedFatal(NewtonFailure::NonFiniteResidual); }

        for i in 0..prob.n { wk.dy[i] = -wk.res[i]; }

        if prob.bw == -1 {
            let jvp = prob.fns.jvp_fn.expect("evaluate_jvp missing.");
            let y_ptr = wk.y.as_ptr(); let ydot_ptr = wk.ydot.as_ptr(); let p_ptr = wk.p.as_ptr(); let m_ptr = prob.m.as_ptr();
            let jvp_closure = |v: &[f64], out: &mut [f64]| { unsafe { jvp(y_ptr, ydot_ptr, p_ptr, m_ptr, c_j, v.as_ptr(), out.as_mut_ptr()) }; };
            let precond = |v: &[f64], out: &mut [f64]| { for i in 0..prob.n { out[i] = v[i] / (c_j * prob.id[i] + prob.spatial_diag[i] + 1.0); } };
            if let Err(e) = gmres::solve_gmres(prob.n, &mut wk.dy, jvp_closure, precond) { return NewtonResult::DivergedFatal(NewtonFailure::SingularJacobian(e)); }
            cj_ratio = 1.0;
        } else {
            if wk.lu_solver.is_stale {
                let start = Instant::now();
                jacobian::assemble(prob, wk, c_j);
                wk.diag.jacobian_assembly_time_us += start.elapsed().as_micros();
                wk.diag.jacobian_evaluations += 1;
                
                if let Err(e) = wk.lu_solver.factorize_from_triplets(&mut wk.diag) { return NewtonResult::DivergedFatal(NewtonFailure::SingularJacobian(e)); }
                
                wk.history.c_j_old = c_j; 
                cj_ratio = 1.0; 
                wk.lu_solver.is_stale = false;
            }
            if let Err(e) = wk.lu_solver.solve(&mut wk.dy, &mut wk.diag) { return NewtonResult::DivergedFatal(NewtonFailure::SingularJacobian(e)); }
        }

        if prob.bw != -1 && cj_ratio != 1.0 {
            let scale = 2.0 / (1.0 + cj_ratio);
            for i in 0..prob.n { wk.dy[i] *= scale; }
        }

        let mut is_clamped = false;
        for i in 0..prob.n {
            if prob.max_steps[i] > 0.0 && wk.dy[i].abs() > prob.max_steps[i] {
                wk.dy[i] = wk.dy[i].signum() * prob.max_steps[i]; is_clamped = true;
            }
            wk.ee[i] += wk.dy[i];
        }

        let dy_norm = constraints::wrms_norm_all(&wk.dy, &wk.weights);
        wk.diag.last_dy.copy_from_slice(&wk.dy);
        wk.diag.last_weights.copy_from_slice(&wk.weights);

        if wk.diag.recent_newton_norms.len() >= 5 { wk.diag.recent_newton_norms.pop_front(); }
        wk.diag.recent_newton_norms.push_back((iter + 1, f_norm, dy_norm));

        if iter == 0 {
            if wk.diag.accepted_steps == 0 {
                let (mut max_r, mut max_idx) = (0.0, 0);
                for i in 0..prob.n { if wk.res[i].abs() > max_r { max_r = wk.res[i].abs(); max_idx = i; } }
                wk.diag.t0_max_res = max_r; wk.diag.t0_max_res_idx = max_idx;
            }
            old_fnorm = dy_norm;
            if dy_norm <= 1e-8 * prob.config.eps_newt { return constraints::evaluate(prob.n, &wk.y, &wk.ee, &wk.history.phi[0], &prob.constraints, iter); }
        } else {
            let rate = (dy_norm / old_fnorm).powf(1.0 / iter as f64);
            wk.diag.last_rho = rate;
            
            if rate > prob.config.max_rho && !is_clamped {
                if !wk.lu_solver.is_stale { wk.lu_solver.mark_stale(); continue; }
                return NewtonResult::DivergedFatal(NewtonFailure::ContractionThrashing(rate));
            }
            let ss = rate / (1.0 - rate);
            if ss * dy_norm <= prob.config.eps_newt { return constraints::evaluate(prob.n, &wk.y, &wk.ee, &wk.history.phi[0], &prob.constraints, iter); }
        }
    }
    NewtonResult::DivergedFatal(NewtonFailure::MaxItersReached)
}

pub fn calc_algebraic_roots(prob: &Problem, wk: &mut Workspace) -> Result<(), String> {
    for _ in 0..50 {
        unsafe { (prob.fns.res_fn)(wk.y.as_ptr(), wk.ydot.as_ptr(), wk.p.as_ptr(), prob.m.as_ptr(), wk.res.as_mut_ptr()); }

        let mut max_res = 0.0_f64;
        for i in 0..prob.n { if prob.id[i] < 0.5 && wk.res[i].abs() > max_res { max_res = wk.res[i].abs(); } }
        if max_res < 1e-8 { break; }

        jacobian::assemble(prob, wk, 0.0);

        if prob.bw == -1 {
            for &(r, c, val) in &wk.lu_solver.triplets {
                if r == c && prob.id[r] < 0.5 && val.abs() > 1e-12 {
                    let mut step = -wk.res[r] / val;
                    if prob.max_steps[r] > 0.0 && step.abs() > prob.max_steps[r] { step = step.signum() * prob.max_steps[r]; }
                    wk.y[r] += step * 0.8; 
                }
            }
        } else {
            for i in 0..prob.n {
                if prob.id[i] > 0.5 { wk.res[i] = 0.0; wk.lu_solver.triplets.push((i, i, 1.0)); }
            }
            for i in 0..prob.n { wk.dy[i] = -wk.res[i]; }
            if wk.lu_solver.factorize_from_triplets(&mut wk.diag).is_err() { break; }
            if wk.lu_solver.solve(&mut wk.dy, &mut wk.diag).is_err() { break; }

            let mut max_step = 0.0;
            for i in 0..prob.n {
                if prob.id[i] < 0.5 {
                    let mut step = wk.dy[i];
                    if prob.max_steps[i] > 0.0 && step.abs() > prob.max_steps[i] { step = step.signum() * prob.max_steps[i]; }
                    wk.y[i] += step * 0.8; 
                    if step.abs() > max_step { max_step = step.abs(); }
                }
            }
            if max_step < 1e-14 { break; } 
        }
    }
    wk.history.restore_state();
    wk.lu_solver.mark_stale();
    wk.diag.accepted_steps = 0; 
    Ok(())
}