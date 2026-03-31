/* -----------------------------------------------------------------
 * This file is a Rust port of the IDAS solver from the SUNDIALS library.
 * 
 * Original SUNDIALS Copyright Start
 * Copyright (c) 2002-2026, Lawrence Livermore National Security, 
 * University of Maryland Baltimore County, Southern Methodist University, 
 * and the SUNDIALS contributors. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * -----------------------------------------------------------------*/
 
use std::time::Instant;
use super::{NativeResFn, NativeJacFn, NativeJvpFn, SolverConfig, Diagnostics};
use super::linalg::{NativeSparseLuSolver, solve_gmres};

pub enum NewtonFailure {
    NonFiniteResidual,
    SingularJacobian(String),
    ContractionThrashing(f64),
    ConstraintsViolated(f64), 
    MaxItersReached,
}

pub enum NewtonResult {
    Converged(usize, Vec<f64>), 
    DivergedStaleJac(NewtonFailure), 
    DivergedFatal(NewtonFailure),    
}

/// SUNDIALS Inexact Newton-Raphson implementation
pub fn solve_nonlinear_system(
    n: usize, bw: isize,
    y: &mut[f64], ydot: &mut [f64], p: &[f64], m: &[f64], id: &[f64], constraints: &[f64], spatial_diag: &[f64],
    c_j: f64, c_j_last_setup: &mut f64, phi_0: &[f64],
    y_pred: &[f64], ydot_pred: &[f64], weights: &[f64],
    res_fn: NativeResFn, jac_fn: NativeJacFn, jvp_fn: Option<NativeJvpFn>,
    lu_solver: &mut NativeSparseLuSolver, jac_buffer: &mut [f64],
    config: &SolverConfig, diag: &mut Diagnostics,
) -> NewtonResult {
    
    // 1. The 25% Staleness Rule
    let mut cj_ratio = if *c_j_last_setup == 0.0 { 1.0 } else { c_j / *c_j_last_setup };
    let cj_changed = cj_ratio < 0.6 || cj_ratio > 1.6666666666666667;
    
    let mut is_stale_at_start = lu_solver.is_stale;
    if cj_changed {
        lu_solver.mark_stale();
        is_stale_at_start = true;
    }

    let mut ee = vec![0.0; n];  // Accumulated Newton Correction
    let mut res = vec![0.0; n];
    let mut dy = vec![0.0; n];
    
    // Inner loop allows the solver to flush a stale Jacobian and try again 
    // WITHOUT collapsing the integration step size `dt`.
    loop {
        let mut old_fnorm = 0.0;
        let mut retry = false;

        for iter in 0..config.max_newton_iters {
            diag.newton_iterations += 1;

            // Evaluate Residual: F(y_pred + ee, ydot_pred + c_j*ee)
            for i in 0..n {
                y[i] = y_pred[i] + ee[i];
                ydot[i] = ydot_pred[i] + c_j * ee[i];
            }

            let t_res = Instant::now();
            unsafe { res_fn(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), m.as_ptr(), res.as_mut_ptr()) };
            diag.residual_time_us += t_res.elapsed().as_micros();
            
            diag.last_res = res.clone();

            // Newton convergence is measured against ALL variables (no algebraic suppression)
            let f_norm = wrms_norm_all(&res, weights);
            if !f_norm.is_finite() {
                let fail = NewtonFailure::NonFiniteResidual;
                if is_stale_at_start && bw != -1 {
                    retry = true;
                    break;
                }
                return NewtonResult::DivergedFatal(fail);
            }

            for i in 0..n { dy[i] = -res[i]; }

            // Linear Solve
            if bw == -1 {
                let jvp = jvp_fn.expect("evaluate_jvp missing.");
                let y_ptr = y.as_ptr(); let ydot_ptr = ydot.as_ptr(); let p_ptr = p.as_ptr(); let m_ptr = m.as_ptr();
                let jvp_closure = |v: &[f64], out: &mut [f64]| {
                    unsafe { jvp(y_ptr, ydot_ptr, p_ptr, m_ptr, c_j, v.as_ptr(), out.as_mut_ptr()) };
                };
                let precond = |v: &[f64], out: &mut [f64]| {
                    for i in 0..n { out[i] = v[i] / (c_j * id[i] + spatial_diag[i] + 1.0); }
                };
                if let Err(e) = solve_gmres(n, &mut dy, jvp_closure, precond) {
                    return NewtonResult::DivergedFatal(NewtonFailure::SingularJacobian(e));
                }
                cj_ratio = 1.0;
            } else {
                if lu_solver.is_stale {
                    let start = Instant::now();
                    unsafe { jac_fn(y_pred.as_ptr(), ydot_pred.as_ptr(), p.as_ptr(), m.as_ptr(), c_j, jac_buffer.as_mut_ptr()) };
                    diag.jacobian_assembly_time_us += start.elapsed().as_micros();
                    diag.jacobian_evaluations += 1;
                    
                    if let Err(e) = lu_solver.factorize(jac_buffer, diag) {
                        return NewtonResult::DivergedFatal(NewtonFailure::SingularJacobian(e)); 
                    }
                    *c_j_last_setup = c_j;
                    cj_ratio = 1.0;
                }
                if let Err(e) = lu_solver.solve(&mut dy, diag) {
                    return NewtonResult::DivergedFatal(NewtonFailure::SingularJacobian(e));
                }
            }

            // 2. Linear Solution Scaling (Ensures step is valid when c_j changes but Jacobian is retained)
            if bw != -1 && cj_ratio != 1.0 {
                let scale = 2.0 / (1.0 + cj_ratio);
                for i in 0..n { dy[i] *= scale; }
            }

            for i in 0..n { ee[i] += dy[i]; }
            let dy_norm = wrms_norm_all(&dy, weights);
            diag.last_dy = dy.clone();
            diag.last_weights = weights.to_vec();

            // 3. Stringent SUNDIALS Convergence Criteria
            if iter == 0 {
                old_fnorm = dy_norm;
                if dy_norm <= 1e-8 * config.eps_newt {
                    return evaluate_constraints_and_return(n, y, &ee, phi_0, constraints, iter);
                }
            } else {
                let rate = (dy_norm / old_fnorm).powf(1.0 / iter as f64);
                diag.last_rho = rate;
                
                // 4. Early Divergence Detection (Thrashing)
                if rate > config.max_rho {
                    let fail = NewtonFailure::ContractionThrashing(rate);
                    if is_stale_at_start && bw != -1 {
                        retry = true;
                        break;
                    }
                    return NewtonResult::DivergedFatal(fail);
                }
                
                let ss = rate / (1.0 - rate);
                if ss * dy_norm <= config.eps_newt {
                    return evaluate_constraints_and_return(n, y, &ee, phi_0, constraints, iter);
                }
            }
        }
        
        // If we reach here, we either hit max iterations or broke out to retry.
        if retry || (is_stale_at_start && bw != -1) {
            is_stale_at_start = false;
            lu_solver.mark_stale();
            ee.fill(0.0);
            continue; // Retry outer loop with a fresh Jacobian
        }
        
        return NewtonResult::DivergedFatal(NewtonFailure::MaxItersReached);
    }
}

fn evaluate_constraints_and_return(n: usize, y: &[f64], ee: &[f64], phi_0: &[f64], constraints: &[f64], iter: usize) -> NewtonResult {
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
        NewtonResult::Converged(iter + 1, ee.to_vec())
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
        if !suppress_alg || id[i] > 0.5 {
            sum += (v[i] * w[i]).powi(2);
        }
    }
    (sum / (v.len() as f64)).sqrt()
}