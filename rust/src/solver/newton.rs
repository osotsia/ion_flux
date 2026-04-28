use std::time::Instant;
use std::os::raw::c_int;
use super::{NativeResFn, NativeJacSparseFn, NativeJvpFn, NativeVjpFn, SolverConfig, Diagnostics};
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

pub fn assemble_jacobian_triplets(
    n: usize, y: &[f64], ydot: &[f64], p: &[f64], m: &[f64], c_j: f64,
    jac_sparse_fn: NativeJacSparseFn, jvp_fn: Option<NativeJvpFn>, vjp_fn: Option<NativeVjpFn>,
    lu_solver: &mut NativeSparseLuSolver, jac_rows_buf: &mut [i32], jac_cols_buf: &mut [i32], jac_vals_buf: &mut [f64],
    cpr: &super::CprData
) {
    lu_solver.triplets.clear();
    
    if !cpr.color_seeds.is_empty() {
        if let Some(jvp) = jvp_fn {
            for (c_idx, seed) in cpr.color_seeds.iter().enumerate() {
                let mut jvp_out = vec![0.0; n];
                unsafe { jvp(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), m.as_ptr(), c_j, seed.as_ptr(), jvp_out.as_mut_ptr()); }
                let start = cpr.color_ptrs[c_idx];
                let end = cpr.color_ptrs[c_idx + 1];
                for i in start..end {
                    let r = cpr.color_rows[i];
                    let c = cpr.color_cols[i];
                    lu_solver.triplets.push((r, c, jvp_out[r]));
                }
            }
        }
        
        if !cpr.dense_rows.is_empty() {
            if let Some(vjp) = vjp_fn {
                let mut dp_out = vec![0.0; p.len()];
                let mut dy_out = vec![0.0; n];
                let mut dydot_out = vec![0.0; n];
                let mut lambda = vec![0.0; n];
                for &r in &cpr.dense_rows {
                    lambda[r] = 1.0;
                    unsafe { vjp(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), m.as_ptr(), lambda.as_ptr(), dp_out.as_mut_ptr(), dy_out.as_mut_ptr(), dydot_out.as_mut_ptr()); }
                    lambda[r] = 0.0;
                    for c in 0..n {
                        let val = dy_out[c] + c_j * dydot_out[c];
                        if val.abs() > 1e-16 || val.is_nan() {
                            lu_solver.triplets.push((r, c, val));
                        }
                    }
                }
            }
        }
    } else {
        let mut nnz: c_int = 0;
        unsafe {
            jac_sparse_fn(
                y.as_ptr(), ydot.as_ptr(), p.as_ptr(), m.as_ptr(), c_j,
                jac_rows_buf.as_mut_ptr(), jac_cols_buf.as_mut_ptr(), jac_vals_buf.as_mut_ptr(),
                &mut nnz
            );
        }
        for i in 0..(nnz as usize) {
            lu_solver.triplets.push((jac_rows_buf[i] as usize, jac_cols_buf[i] as usize, jac_vals_buf[i]));
        }
    }
}

pub fn solve_nonlinear_system(
    n: usize, bw: isize,
    y: &mut[f64], ydot: &mut [f64], p: &[f64], m: &[f64], id: &[f64], constraints: &[f64], spatial_diag: &[f64], max_steps: &[f64],
    c_j: f64, c_j_last_setup: &mut f64, phi_0: &[f64],
    y_pred: &[f64], ydot_pred: &[f64], weights: &[f64],
    res_fn: NativeResFn, jac_sparse_fn: NativeJacSparseFn, jvp_fn: Option<NativeJvpFn>, vjp_fn: Option<NativeVjpFn>,
    lu_solver: &mut NativeSparseLuSolver, jac_rows_buf: &mut [i32], jac_cols_buf: &mut [i32], jac_vals_buf: &mut [f64],
    config: &SolverConfig, diag: &mut Diagnostics, cpr: &super::CprData,
) -> NewtonResult {
    
    let mut cj_ratio = if *c_j_last_setup == 0.0 { 1.0 } else { c_j / *c_j_last_setup };
    let cj_changed = cj_ratio < 0.6 || cj_ratio > 1.6666666666666667;
    
    if cj_changed {
        lu_solver.mark_stale();
    }

    let mut ee = vec![0.0; n];
    let mut res = vec![0.0; n];
    let mut dy = vec![0.0; n];
    
    diag.recent_newton_norms.clear();

    let mut old_fnorm = 0.0;

    for iter in 0..config.max_newton_iters {
        diag.newton_iterations += 1;

        for i in 0..n {
            y[i] = y_pred[i] + ee[i];
            ydot[i] = ydot_pred[i] + c_j * ee[i];
        }

        let t_res = Instant::now();
        unsafe { res_fn(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), m.as_ptr(), res.as_mut_ptr()) };
        diag.residual_time_us += t_res.elapsed().as_micros();
        
        diag.last_res = res.clone();

        let f_norm = wrms_norm_all(&res, weights);
        if !f_norm.is_finite() {
            return NewtonResult::DivergedFatal(NewtonFailure::NonFiniteResidual);
        }

        for i in 0..n { dy[i] = -res[i]; }

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
                
                assemble_jacobian_triplets(
                    n, y, ydot, p, m, c_j, 
                    jac_sparse_fn, jvp_fn, vjp_fn, 
                    lu_solver, jac_rows_buf, jac_cols_buf, jac_vals_buf, cpr
                );
                
                diag.jacobian_assembly_time_us += start.elapsed().as_micros();
                diag.jacobian_evaluations += 1;
                
                if let Err(e) = lu_solver.factorize_from_triplets(diag) {
                    return NewtonResult::DivergedFatal(NewtonFailure::SingularJacobian(e)); 
                }
                
                *c_j_last_setup = c_j;
                cj_ratio = 1.0;
                lu_solver.is_stale = false;
            }
            if let Err(e) = lu_solver.solve(&mut dy, diag) {
                return NewtonResult::DivergedFatal(NewtonFailure::SingularJacobian(e));
            }
        }

        if bw != -1 && cj_ratio != 1.0 {
            let scale = 2.0 / (1.0 + cj_ratio);
            for i in 0..n { dy[i] *= scale; }
        }

        let mut is_clamped = false;
        for i in 0..n {
            if max_steps[i] > 0.0 && dy[i].abs() > max_steps[i] {
                dy[i] = dy[i].signum() * max_steps[i];
                is_clamped = true;
            }
            ee[i] += dy[i];
        }

        let dy_norm = wrms_norm_all(&dy, weights);
        diag.last_dy = dy.clone();
        diag.last_weights = weights.to_vec();

        if diag.recent_newton_norms.len() >= 5 { diag.recent_newton_norms.pop_front(); }
        diag.recent_newton_norms.push_back((iter + 1, f_norm, dy_norm));

        if iter == 0 {
            if diag.accepted_steps == 0 {
                let mut max_r = 0.0;
                let mut max_idx = 0;
                for i in 0..n {
                    if res[i].abs() > max_r { max_r = res[i].abs(); max_idx = i; }
                }
                diag.t0_max_res = max_r;
                diag.t0_max_res_idx = max_idx;
            }
            old_fnorm = dy_norm;
            if dy_norm <= 1e-8 * config.eps_newt {
                return evaluate_constraints_and_return(n, y, &ee, phi_0, constraints, iter);
            }
        } else {
            let rate = (dy_norm / old_fnorm).powf(1.0 / iter as f64);
            diag.last_rho = rate;
            
            if rate > config.max_rho && !is_clamped {
                if !lu_solver.is_stale {
                    lu_solver.mark_stale();
                    continue;
                }
                return NewtonResult::DivergedFatal(NewtonFailure::ContractionThrashing(rate));
            }
            
            let ss = rate / (1.0 - rate);
            if ss * dy_norm <= config.eps_newt {
                return evaluate_constraints_and_return(n, y, &ee, phi_0, constraints, iter);
            }
        }
    }
    
    NewtonResult::DivergedFatal(NewtonFailure::MaxItersReached)
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