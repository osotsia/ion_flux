use numpy::ndarray::{ArrayView1, ArrayView2};
use crate::ffi::callbacks::{NativeJvpFn, NativeVjpFn};
use crate::solver::shared::problem::CprData;
use crate::linear::sparse_lu::NativeSparseLuSolver;
use crate::linear::gmres::solve_gmres;
use crate::solver::shared::diagnostics::Diagnostics;

pub struct AdjointParams<'a> {
    pub bandwidth: isize,
    pub id_slice: &'a [f64],
    pub m_slice: &'a [f64],
    pub cpr: &'a CprData,
    pub jvp_fn: NativeJvpFn,
    pub vjp_fn: NativeVjpFn,
    pub event_trigger_idx: isize,
}

struct AdjointWorkspace {
    rhs: Vec<f64>, rhs_orig: Vec<f64>, delta_lambda: Vec<f64>,
    jvp_out: Vec<f64>, jvp_dy: Vec<f64>, jvp_dydot: Vec<f64>, res_dummy: Vec<f64>,
    dp_out: Vec<f64>, dy_out: Vec<f64>, dydot_out: Vec<f64>,
    lambda_vjp: Vec<f64>, vjp_dp: Vec<f64>, vjp_dy: Vec<f64>, vjp_dydot: Vec<f64>,
    gmres_dp: Vec<f64>, gmres_dy: Vec<f64>, gmres_dydot: Vec<f64>, gmres_res_dummy: Vec<f64>,
}

impl AdjointWorkspace {
    fn new(n: usize, n_params: usize) -> Self {
        Self {
            rhs: vec![0.0; n], rhs_orig: vec![0.0; n], delta_lambda: vec![0.0; n],
            jvp_out: vec![0.0; n], jvp_dy: vec![0.0; n], jvp_dydot: vec![0.0; n], res_dummy: vec![0.0; n],
            dp_out: vec![0.0; n_params], dy_out: vec![0.0; n], dydot_out: vec![0.0; n],
            lambda_vjp: vec![0.0; n], vjp_dp: vec![0.0; n_params], vjp_dy: vec![0.0; n], vjp_dydot: vec![0.0; n],
            gmres_dp: vec![0.0; n_params], gmres_dy: vec![0.0; n], gmres_dydot: vec![0.0; n], gmres_res_dummy: vec![0.0; n],
        }
    }
}

pub fn solve_adjoint_sensitivities(
    params: &AdjointParams,
    y_arr: ArrayView2<f64>, ydot_arr: ArrayView2<f64>, t_arr: ArrayView1<f64>,
    p_arr: ArrayView2<f64>, dl_dy_arr: ArrayView2<f64>,
) -> Result<Vec<f64>, String> {
    
    let n = y_arr.ncols();
    let n_params = p_arr.ncols();
    let mut p_grad = vec![0.0; n_params];
    
    let mut prev_dydot_vjp = vec![0.0; n];
    let mut prev_c_j = 0.0;
    let mut prev_factorized_cj = 0.0;
    
    let mut diag = Diagnostics::new(n);
    let mut solver = NativeSparseLuSolver::new(n, params.bandwidth);
    let mut ws = AdjointWorkspace::new(n, n_params);

    for step in (1..y_arr.nrows()).rev() {
        let dt = t_arr[step] - t_arr[step - 1];
        if dt <= 1e-12 { continue; }

        let c_j = 1.0 / dt;
        let y = y_arr.row(step).to_slice().unwrap();
        let ydot = ydot_arr.row(step).to_slice().unwrap();
        let p_list = p_arr.row(step).to_slice().unwrap();
        let dl_dy_step = dl_dy_arr.row(step).to_slice().unwrap();

        for i in 0..n {
            ws.rhs[i] = -dl_dy_step[i] + prev_dydot_vjp[i] * prev_c_j;
        }

        if params.event_trigger_idx >= 0 && step == y_arr.nrows() - 1 {
            let idx = params.event_trigger_idx as usize;
            let ydot_val = if ydot[idx].abs() < 1e-12 { 1e-12 * ydot[idx].signum().max(1.0) } else { ydot[idx] };
            ws.rhs[idx] += 1.0 / ydot_val;
        }

        if params.bandwidth == -1 {
            solve_gmres_step(params, &mut ws, c_j, y, ydot, p_list)?;
        } else {
            solve_lu_step(params, &mut ws, &mut solver, &mut diag, c_j, &mut prev_factorized_cj, y, ydot, p_list)?;
        }

        for p_idx in 0..n_params { p_grad[p_idx] += ws.dp_out[p_idx]; }
        prev_dydot_vjp.copy_from_slice(&ws.dydot_out);
        prev_c_j = c_j;
    }
    Ok(p_grad)
}

fn solve_gmres_step(params: &AdjointParams, ws: &mut AdjointWorkspace, c_j: f64, y: &[f64], ydot: &[f64], p: &[f64]) -> Result<(), String> {
    let n = y.len();
    let mut jvp_t = |v: &[f64], out: &mut [f64]| {
        unsafe { (params.vjp_fn)(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), params.m_slice.as_ptr(), v.as_ptr(), ws.gmres_res_dummy.as_mut_ptr(), ws.gmres_dp.as_mut_ptr(), ws.gmres_dy.as_mut_ptr(), ws.gmres_dydot.as_mut_ptr()); }
        for i in 0..n { out[i] = ws.gmres_dy[i] + c_j * ws.gmres_dydot[i]; }
    };
    let precond = |v: &[f64], out: &mut [f64]| {
        for i in 0..n { out[i] = v[i] / (c_j * params.id_slice[i] + 1.0); }
    };
    solve_gmres(n, &mut ws.rhs, &mut jvp_t, precond)?;
    unsafe { (params.vjp_fn)(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), params.m_slice.as_ptr(), ws.rhs.as_ptr(), ws.res_dummy.as_mut_ptr(), ws.dp_out.as_mut_ptr(), ws.dy_out.as_mut_ptr(), ws.dydot_out.as_mut_ptr()); }
    Ok(())
}

fn solve_lu_step(
    params: &AdjointParams, ws: &mut AdjointWorkspace, solver: &mut NativeSparseLuSolver, diag: &mut Diagnostics, 
    c_j: f64, prev_cj: &mut f64, y: &[f64], ydot: &[f64], p: &[f64]
) -> Result<(), String> {
    ws.rhs_orig.copy_from_slice(&ws.rhs);
    let cj_ratio = if *prev_cj == 0.0 { 0.0 } else { c_j / *prev_cj };
    let n = y.len();

    let execute_vjp = |w: &mut AdjointWorkspace| unsafe {
        (params.vjp_fn)(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), params.m_slice.as_ptr(), w.rhs.as_ptr(), w.res_dummy.as_mut_ptr(), w.dp_out.as_mut_ptr(), w.dy_out.as_mut_ptr(), w.dydot_out.as_mut_ptr());
    };

    if !solver.is_stale && cj_ratio >= 0.7 && cj_ratio <= 1.4 {
        if solver.solve(&mut ws.rhs, diag).is_ok() {
            execute_vjp(ws);
            let mut max_res = 0.0_f64;
            let mut max_b = 0.0_f64;
            for i in 0..n {
                let r_i = ws.rhs_orig[i] - (ws.dy_out[i] + c_j * ws.dydot_out[i]);
                ws.delta_lambda[i] = r_i;
                max_res = max_res.max(r_i.abs());
                max_b = max_b.max(ws.rhs_orig[i].abs());
                if !r_i.is_finite() { max_res = f64::INFINITY; break; }
            }
            let tol = (1e-7 * max_b).max(1e-10_f64);
            if max_res <= tol { return Ok(()); }
            
            // Iterative Refinement
            if max_res.is_finite() && max_res <= 0.05 * (max_b + 1.0) {
                if solver.solve(&mut ws.delta_lambda, diag).is_ok() {
                    for i in 0..n { ws.rhs[i] += ws.delta_lambda[i]; }
                    execute_vjp(ws);
                    let max_res1 = (0..n).map(|i| (ws.rhs_orig[i] - (ws.dy_out[i] + c_j * ws.dydot_out[i])).abs()).max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0.0);
                    if max_res1 <= tol { return Ok(()); }
                }
            }
        }
    }

    // Full Assembly Fallback
    solver.triplets.clear();
    let cpr_nnz = params.cpr.color_rows.len() + params.cpr.dense_rows.len() * n;
    if solver.triplets.capacity() < cpr_nnz {
        solver.triplets.reserve(cpr_nnz - solver.triplets.capacity());
    }
    for (c_idx, seed) in params.cpr.color_seeds.iter().enumerate() {
        unsafe { (params.jvp_fn)(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), params.m_slice.as_ptr(), c_j, seed.as_ptr(), ws.jvp_dy.as_mut_ptr(), ws.jvp_dydot.as_mut_ptr(), ws.res_dummy.as_mut_ptr(), ws.jvp_out.as_mut_ptr()); }
        for i in params.cpr.color_ptrs[c_idx]..params.cpr.color_ptrs[c_idx + 1] {
            solver.triplets.push((params.cpr.color_cols[i], params.cpr.color_rows[i], ws.jvp_out[params.cpr.color_rows[i]]));
        }
    }
    for &r in &params.cpr.dense_rows {
        ws.lambda_vjp[r] = 1.0;
        unsafe { (params.vjp_fn)(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), params.m_slice.as_ptr(), ws.lambda_vjp.as_ptr(), ws.res_dummy.as_mut_ptr(), ws.vjp_dp.as_mut_ptr(), ws.vjp_dy.as_mut_ptr(), ws.vjp_dydot.as_mut_ptr()); }
        ws.lambda_vjp[r] = 0.0;
        for col in 0..n {
            let val = ws.vjp_dy[col] + c_j * ws.vjp_dydot[col];
            if val.abs() > 1e-16 || val.is_nan() { solver.triplets.push((col, r, val)); }
        }
    }
    
    solver.factorize_from_triplets(diag)?;
    *prev_cj = c_j;
    ws.rhs.copy_from_slice(&ws.rhs_orig);
    solver.solve(&mut ws.rhs, diag)?;
    execute_vjp(ws);
    
    Ok(())
}