use crate::solver::shared::problem::{Problem, CprData};
use crate::solver::shared::callbacks::{NativeJvpFn, NativeVjpFn};
use crate::solver::shared::workspace::Workspace;

pub fn assemble_triplets(
    n: usize, y: &[f64], ydot: &[f64], p: &[f64], m: &[f64], c_j: f64,
    jvp_fn: Option<NativeJvpFn>, vjp_fn: Option<NativeVjpFn>,
    triplets: &mut Vec<(usize, usize, f64)>,
    cpr: &CprData,
    id: &[f64]
) {
    triplets.clear();
    let is_alg_init = c_j == 0.0;
    
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
                    if is_alg_init && id[r] > 0.5 && r != c { continue; }
                    triplets.push((r, c, jvp_out[r]));
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
                    if is_alg_init && id[r] > 0.5 { continue; }
                    lambda[r] = 1.0;
                    unsafe { vjp(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), m.as_ptr(), lambda.as_ptr(), dp_out.as_mut_ptr(), dy_out.as_mut_ptr(), dydot_out.as_mut_ptr()); }
                    lambda[r] = 0.0;
                    for c_idx in 0..n {
                        let val = dy_out[c_idx] + c_j * dydot_out[c_idx];
                        if val.abs() > 1e-16 || val.is_nan() { triplets.push((r, c_idx, val)); }
                    }
                }
            }
        }
    }
}

pub fn assemble(prob: &Problem, wk: &mut Workspace, c_j: f64) {
    assemble_triplets(
        prob.n, &wk.y, &wk.ydot, &wk.p, &prob.m, c_j,
        prob.fns.jvp_fn, prob.fns.vjp_fn,
        &mut wk.lu_solver.triplets,
        &prob.cpr,
        &prob.id
    );
}