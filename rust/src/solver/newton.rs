use std::time::Instant;
use super::{NativeResFn, NativeJacFn, NativeJvpFn, SolverConfig, Diagnostics};
use super::linalg::{NativeSparseLuSolver, solve_gmres};

pub enum NewtonResult {
    Converged(usize, Vec<f64>), // Iterations taken, e_n = y - y_pred
    DivergedStaleJac, 
    DivergedFatal,    
}

/// Inexact Newton-Raphson leveraging Jacobian staleness heuristics to minimize AD sweeps.
pub fn solve_nonlinear_system(
    n: usize,
    bw: isize,
    y: &mut[f64], ydot: &mut [f64], p: &[f64], id: &[f64], spatial_diag: &[f64],
    c_j: f64,
    y_pred: &[f64], ydot_pred: &[f64],
    res_fn: NativeResFn, jac_fn: NativeJacFn, jvp_fn: Option<NativeJvpFn>,
    lu_solver: &mut NativeSparseLuSolver,
    jac_buffer: &mut [f64],
    config: &SolverConfig,
    diag: &mut Diagnostics,
) -> NewtonResult {
    
    // Capture staleness state at entry to prevent infinite loops 
    // when the Jacobian is rebuilt but the system still diverges.
    let is_stale_at_start = lu_solver.is_stale;
    
    let mut res = vec![0.0; n];
    let mut dy = vec![0.0; n];
    let mut weights = vec![0.0; n];

    y.copy_from_slice(y_pred);
    ydot.copy_from_slice(ydot_pred);

    let mut prev_dy_norm = std::f64::MAX;

    for iter in 0..config.max_newton_iters {
        unsafe { res_fn(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), res.as_mut_ptr()) };

        for i in 0..n {
            weights[i] = 1.0 / (config.rel_tol * y[i].abs() + config.abs_tol);
            dy[i] = -res[i];
        }

        // Includes algebraic elements: necessary to measure nonlinear root convergence mathematically
        let f_norm = wrms_norm_all(&res, &weights);
        if !f_norm.is_finite() {
            return if is_stale_at_start { NewtonResult::DivergedFatal } else { NewtonResult::DivergedStaleJac };
        }

        // --- Linear Solve Dispatch ---
        if bw == -1 {
            // GMRES Matrix-Free (Massive 3D)
            let start = Instant::now();
            let jvp = jvp_fn.expect("evaluate_jvp missing.");
            let y_ptr = y.as_ptr(); let ydot_ptr = ydot.as_ptr(); let p_ptr = p.as_ptr();
            let jvp_closure = |v: &[f64], out: &mut [f64]| {
                unsafe { jvp(y_ptr, ydot_ptr, p_ptr, c_j, v.as_ptr(), out.as_mut_ptr()) };
            };
            let precond = |v: &[f64], out: &mut [f64]| {
                for i in 0..n { out[i] = v[i] / (c_j * id[i] + spatial_diag[i] + 1.0); }
            };
            if solve_gmres(n, &mut dy, jvp_closure, precond).is_err() { break; }
            diag.linear_solve_time_us += start.elapsed().as_micros();
        } else {
            // Faer Sparse LU (1D / 2D)
            if lu_solver.is_stale {
                let start = Instant::now();
                unsafe { jac_fn(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), c_j, jac_buffer.as_mut_ptr()) };
                diag.jacobian_assembly_time_us += start.elapsed().as_micros();
                diag.jacobian_evaluations += 1;
                
                if lu_solver.factorize(jac_buffer, diag).is_err() { break; }
            }
            if lu_solver.solve(&mut dy, diag).is_err() { break; }
        }

        let dy_norm = wrms_norm_all(&dy, &weights);

        // Staleness Heuristic: Contraction Rate (Rho)
        if iter > 0 {
            let rho = dy_norm / prev_dy_norm;
            if rho > config.max_rho {
                return if is_stale_at_start { NewtonResult::DivergedFatal } else { NewtonResult::DivergedStaleJac };
            }
        }
        prev_dy_norm = dy_norm;

        for i in 0..n {
            y[i] += dy[i];
            if id[i] > 0.5 { ydot[i] += c_j * dy[i]; }
        }

        // Affine-invariant convergence check
        if dy_norm < 0.33 {
            diag.newton_iterations += iter + 1;
            let mut e_n = vec![0.0; n];
            for i in 0..n { e_n[i] = y[i] - y_pred[i]; }
            return NewtonResult::Converged(iter + 1, e_n);
        }
    }

    if is_stale_at_start { NewtonResult::DivergedFatal } else { NewtonResult::DivergedStaleJac }
}

/// Evaluates root convergence including instantaneously tracking algebraic DAEs
pub fn wrms_norm_all(v: &[f64], w: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..v.len() { sum += (v[i] * w[i]).powi(2); }
    (sum / v.len() as f64).sqrt()
}

/// Evaluates Local Truncation Error (LTE), exclusively punishing differential states
pub fn wrms_norm_diff(v: &[f64], w: &[f64], id: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut active = 0.0;
    for i in 0..v.len() { 
        if id[i] > 0.5 {
            sum += (v[i] * w[i]).powi(2); 
            active += 1.0;
        }
    }
    if active > 0.0 { (sum / active).sqrt() } else { 0.0 }
}