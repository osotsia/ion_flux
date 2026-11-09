use crate::solver::shared::problem::{Problem, CprData};
use crate::ffi::callbacks::{NativeJvpFn, NativeVjpFn};
use crate::stepper::state::SystemState;
use crate::linear::solver::LinearSolver;

pub fn assemble_triplets_with_scratch(
    n: usize, y: &[f64], ydot: &[f64], p: &[f64], m: &[f64], c_j: f64,
    jvp_fn: Option<NativeJvpFn>, vjp_fn: Option<NativeVjpFn>,
    triplets: &mut Vec<(usize, usize, f64)>,
    cpr: &CprData,
    id: &[f64],
    jvp_scratch: &mut [f64],
    jvp_dy_scratch: &mut [f64],
    jvp_dydot_scratch: &mut [f64],
    res_scratch: &mut [f64],
    dp_scratch: &mut [f64],
    dy_scratch: &mut [f64],
    dydot_scratch: &mut [f64],
    lambda_scratch: &mut [f64],
) {
    triplets.clear();
    let is_alg_init = c_j == 0.0;
    
    if !cpr.color_seeds.is_empty() {
        if let Some(jvp) = jvp_fn {
            for (c_idx, seed) in cpr.color_seeds.iter().enumerate() {
                unsafe {
                    jvp(
                        y.as_ptr(),
                        ydot.as_ptr(),
                        p.as_ptr(),
                        m.as_ptr(),
                        c_j,
                        seed.as_ptr(),
                        jvp_dy_scratch.as_mut_ptr(),
                        jvp_dydot_scratch.as_mut_ptr(),
                        res_scratch.as_mut_ptr(),
                        jvp_scratch.as_mut_ptr(),
                    );
                }
                let start = cpr.color_ptrs[c_idx];
                let end = cpr.color_ptrs[c_idx + 1];
                for i in start..end {
                    let r = cpr.color_rows[i];
                    let c = cpr.color_cols[i];
                    if is_alg_init && id[r] > 0.5 && r != c { continue; }
                    triplets.push((r, c, jvp_scratch[r]));
                }
            }
        }
        
        if !cpr.dense_rows.is_empty() {
            if let Some(vjp) = vjp_fn {
                for &r in &cpr.dense_rows {
                    if is_alg_init && id[r] > 0.5 { continue; }
                    lambda_scratch[r] = 1.0;
                    unsafe {
                        vjp(
                            y.as_ptr(),
                            ydot.as_ptr(),
                            p.as_ptr(),
                            m.as_ptr(),
                            lambda_scratch.as_ptr(),
                            res_scratch.as_mut_ptr(),
                            dp_scratch.as_mut_ptr(),
                            dy_scratch.as_mut_ptr(),
                            dydot_scratch.as_mut_ptr(),
                        );
                    }
                    lambda_scratch[r] = 0.0;
                    for c_idx in 0..n {
                        let val = dy_scratch[c_idx] + c_j * dydot_scratch[c_idx];
                        if val.abs() > 1e-16 || val.is_nan() {
                            triplets.push((r, c_idx, val));
                        }
                    }
                }
            }
        }
    }
}

pub fn assemble_triplets(
    n: usize, y: &[f64], ydot: &[f64], p: &[f64], m: &[f64], c_j: f64,
    jvp_fn: Option<NativeJvpFn>, vjp_fn: Option<NativeVjpFn>,
    triplets: &mut Vec<(usize, usize, f64)>,
    cpr: &CprData,
    id: &[f64]
) {
    let mut jvp_scratch = vec![0.0; n];
    let mut jvp_dy_scratch = vec![0.0; n];
    let mut jvp_dydot_scratch = vec![0.0; n];
    let mut res_scratch = vec![0.0; n];
    let mut dp_scratch = vec![0.0; p.len()];
    let mut dy_scratch = vec![0.0; n];
    let mut dydot_scratch = vec![0.0; n];
    let mut lambda_scratch = vec![0.0; n];
    assemble_triplets_with_scratch(
        n, y, ydot, p, m, c_j, jvp_fn, vjp_fn, triplets, cpr, id,
        &mut jvp_scratch, &mut jvp_dy_scratch, &mut jvp_dydot_scratch, &mut res_scratch,
        &mut dp_scratch, &mut dy_scratch, &mut dydot_scratch, &mut lambda_scratch
    );
}

pub fn assemble(prob: &Problem, state: &SystemState, linear_solver: &mut LinearSolver, c_j: f64) {
    match linear_solver {
        LinearSolver::SparseLu(lu) => {
            if lu.dp_scratch.len() < state.p.len() {
                lu.dp_scratch.resize(state.p.len(), 0.0);
            }
            assemble_triplets_with_scratch(
                prob.n, &state.y, &state.ydot, &state.p, &prob.m, c_j,
                prob.fns.jvp_fn, prob.fns.vjp_fn,
                &mut lu.triplets,
                &prob.cpr,
                &prob.id,
                &mut lu.jvp_scratch,
                &mut lu.jvp_dy_scratch,
                &mut lu.jvp_dydot_scratch,
                &mut lu.res_scratch,
                &mut lu.dp_scratch,
                &mut lu.dy_scratch,
                &mut lu.dydot_scratch,
                &mut lu.lambda_scratch,
            );
        }
        LinearSolver::Gmres(gmres) => {
            let mut jvp_scratch = vec![0.0; prob.n];
            let mut jvp_dy_scratch = vec![0.0; prob.n];
            let mut jvp_dydot_scratch = vec![0.0; prob.n];
            let mut res_scratch = vec![0.0; prob.n];
            let mut dp_scratch = vec![0.0; state.p.len()];
            let mut dy_scratch = vec![0.0; prob.n];
            let mut dydot_scratch = vec![0.0; prob.n];
            let mut lambda_scratch = vec![0.0; prob.n];
            assemble_triplets_with_scratch(
                prob.n, &state.y, &state.ydot, &state.p, &prob.m, c_j,
                prob.fns.jvp_fn, prob.fns.vjp_fn,
                &mut gmres.triplets,
                &prob.cpr,
                &prob.id,
                &mut jvp_scratch,
                &mut jvp_dy_scratch,
                &mut jvp_dydot_scratch,
                &mut res_scratch,
                &mut dp_scratch,
                &mut dy_scratch,
                &mut dydot_scratch,
                &mut lambda_scratch,
            );
        }
    }
}
