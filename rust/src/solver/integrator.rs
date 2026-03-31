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
 
use std::fs::File;
use std::io::Write;
use super::{NativeResFn, NativeJacFn, NativeJvpFn, SolverConfig, Diagnostics};
use super::linalg::NativeSparseLuSolver;
use super::newton::{solve_nonlinear_system, NewtonResult, NewtonFailure, wrms_norm_mask};

/// SUNDIALS Modified Divided Difference Array Structure (VSVO BDF)
#[derive(Clone)]
pub struct BdfHistory {
    pub order: usize,
    pub max_order: usize,
    pub k_used: usize,
    pub phase: usize, 
    pub ns: usize,    
    
    pub phi: Vec<Vec<f64>>, 
    pub psi: [f64; 6],
    pub alpha: [f64; 6],
    pub beta: [f64; 6],
    pub sigma: [f64; 6],
    pub gamma: [f64; 6],
    
    pub c_j: f64,
    pub c_j_old: f64,
    pub h_used: f64,
    pub h: f64,
}

impl BdfHistory {
    pub fn new(n: usize) -> Self {
        Self {
            order: 1, max_order: 5, k_used: 0, phase: 0, ns: 0,
            phi: vec![vec![0.0; n]; 6],
            psi: [0.0; 6], alpha: [0.0; 6], beta: [0.0; 6], sigma: [0.0; 6], gamma: [0.0; 6],
            c_j: 0.0, c_j_old: 0.0, h_used: 0.0, h: 0.0,
        }
    }

    pub fn set_coeffs(&mut self) -> f64 {
        if self.h != self.h_used || self.order != self.k_used { self.ns = 0; }
        self.ns = std::cmp::min(self.ns + 1, self.k_used + 2);
        
        if self.order + 1 >= self.ns {
            self.beta[0] = 1.0;
            self.alpha[0] = 1.0;
            let mut temp1 = self.h;
            self.gamma[0] = 0.0;
            self.sigma[0] = 1.0;
            for i in 1..=self.order {
                let temp2 = self.psi[i-1];
                self.psi[i-1] = temp1;
                self.beta[i] = self.beta[i-1] * self.psi[i-1] / temp2;
                temp1 = temp2 + self.h;
                self.alpha[i] = self.h / temp1;
                self.sigma[i] = (i as f64) * self.sigma[i-1] * self.alpha[i];
                self.gamma[i] = self.gamma[i-1] + self.alpha[i-1] / self.h;
            }
            self.psi[self.order] = temp1;
        }

        let mut alphas = 0.0;
        let mut alpha0 = 0.0;
        for i in 0..self.order {
            alphas -= 1.0 / ((i + 1) as f64);
            alpha0 -= self.alpha[i];
        }
        
        self.c_j = -alphas / self.h;
        
        let mut ck = (self.alpha[self.order] + alphas - alpha0).abs();
        ck = ck.max(self.alpha[self.order]);
        
        if self.ns <= self.order {
            for i in self.ns..=self.order {
                let scale = self.beta[i];
                for j in 0..self.phi[i].len() { self.phi[i][j] *= scale; }
            }
        }
        ck
    }

    pub fn predict(&self, y_pred: &mut [f64], ydot_pred: &mut [f64]) {
        let n = y_pred.len();
        y_pred.fill(0.0);
        ydot_pred.fill(0.0);
        for j in 0..=self.order {
            for i in 0..n { y_pred[i] += self.phi[j][i]; }
        }
        for j in 0..self.order {
            let g = self.gamma[j+1];
            for i in 0..n { ydot_pred[i] += g * self.phi[j+1][i]; }
        }
    }

    pub fn restore(&mut self) {
        for i in 1..=self.order {
            self.psi[i-1] = self.psi[i] - self.h;
        }
        if self.ns <= self.order {
            for i in self.ns..=self.order {
                let inv_beta = 1.0 / self.beta[i];
                for j in 0..self.phi[i].len() { self.phi[i][j] *= inv_beta; }
            }
        }
    }

    /// Cascading Phi Update & Order Evaluation (Matches IDACompleteStep)
    pub fn complete_step(&mut self, err_k: f64, err_km1: f64, err_km2: f64, err_kp1: f64, ee: &[f64], min_dt: f64, max_dt: f64) {
        let n = ee.len();
        let kdiff = self.order as isize - self.k_used as isize;
        self.k_used = self.order;
        self.h_used = self.h;

        let next_order = self.order;

        if (next_order == self.order - 1) || self.order == self.max_order { self.phase = 1; }

        if self.phase == 0 {
            if self.order < self.max_order { self.order += 1; }
            self.h = (self.h * 2.0).clamp(min_dt, max_dt); 
        } else {
            let action; // 0=Maintain, 1=Raise, -1=Lower
            let mut err_knew = err_k;
            let terr_k = (self.order as f64 + 1.0) * err_k;

            let terr_kp1 = (self.order as f64 + 2.0) * err_kp1;

            if self.order + 1 >= self.ns || kdiff == 1 { action = 0; }
            else if self.order == self.max_order { action = 0; } 
            else {
                let terr_km1 = (self.order as f64) * err_km1;
                let terr_km2 = (self.order as f64 - 1.0) * err_km2;

                if self.order == 1 {
                    if terr_kp1 >= 0.5 * terr_k { action = 0; } else { action = 1; }
                } else {
                    if terr_km1.max(terr_km2) <= terr_k { action = -1; }
                    else if terr_kp1 >= terr_k { action = 0; }
                    else { action = 1; }
                }
            }

            if action == 1 { self.order += 1; err_knew = err_kp1; }
            else if action == -1 { self.order -= 1; err_knew = err_km1; }

            let tmp = (2.0 * err_knew + 0.0001).powf(-1.0 / (self.order as f64 + 1.0));
            let mut eta = 1.0;
            if tmp >= 2.0 { eta = tmp.min(10.0); }
            else if tmp <= 1.0 { eta = tmp.min(0.9).max(0.5); }
            
            self.h = (self.h * eta).clamp(min_dt, max_dt);
        }

        // Save ee for possible order increase on next step
        if self.k_used < self.max_order {
            self.phi[self.k_used + 1].copy_from_slice(ee);
        }

        // Descending loop order ensures we cascade the NEW values of phi downwards, 
        // exactly matching SUNDIALS simultaneous X += Z array behavior.
        for i in 0..n {
            self.phi[self.k_used][i] += ee[i];
        }
        for j in (0..self.k_used).rev() {
            for i in 0..n {
                self.phi[j][i] += self.phi[j + 1][i];
            }
        }
    }
}

pub fn step_bdf_vsvo(
    n: usize, bw: isize,
    y: &mut[f64], ydot: &mut[f64], p: &[f64], id: &[f64], constraints: &[f64], spatial_diag: &[f64],
    target_dt: f64,
    history: &mut BdfHistory,
    res_fn: NativeResFn, jac_fn: NativeJacFn, jvp_fn: Option<NativeJvpFn>,
    lu_solver: &mut NativeSparseLuSolver, jac_buffer: &mut[f64],
    config: &SolverConfig, diag: &mut Diagnostics,
    mut history_cache: Option<&mut Vec<(f64, Vec<f64>, Vec<f64>)>>,
    abs_t: f64,
) -> Result<(), String> {
    
    let mut t_local = 0.0;
    let mut error_fails = 0; // State machine for consecutive LTE failures

    if diag.accepted_steps == 0 {
        // Initial Step Size Calculation (Matches IDASolve logic)
        let mut weights = vec![0.0; n];
        for i in 0..n { weights[i] = 1.0 / (config.rel_tol * y[i].abs() + config.abs_tol); }
        let ypnorm = wrms_norm_mask(ydot, &weights, id, config.suppress_alg);
        let h0 = if ypnorm > 0.0 { (0.5 / ypnorm).min(target_dt) } else { target_dt.min(1e-4) };

        history.h = h0.max(config.min_dt);
        history.phi[0].copy_from_slice(y);
        for i in 0..n { history.phi[1][i] = history.h * ydot[i]; }
        history.psi[0] = history.h;
        history.c_j = 1.0 / history.h;
        history.order = 1;
        history.k_used = 0;
        lu_solver.mark_stale();
    }

    while target_dt - t_local > 1e-10 * target_dt.abs() {
        history.h = history.h.min(target_dt - t_local).clamp(config.min_dt, config.max_dt);
        diag.total_steps += 1;

        let ck = history.set_coeffs();

        let mut y_pred = vec![0.0; n];
        let mut ydot_pred = vec![0.0; n];
        history.predict(&mut y_pred, &mut ydot_pred);

        // Calculate weights using history.phi[0] (accepted state), 
        // eliminating predictor feedback loops that caused false-positive success.
        let mut weights = vec![0.0; n];
        for i in 0..n { weights[i] = 1.0 / (config.rel_tol * history.phi[0][i].abs() + config.abs_tol); }

        let newton_res = solve_nonlinear_system(
            n, bw, y, ydot, p, id, constraints, spatial_diag,
            history.c_j, &mut history.c_j_old, &history.phi[0],
            &y_pred, &ydot_pred, &weights, res_fn, jac_fn, jvp_fn,
            lu_solver, jac_buffer, config, diag
        );

        match newton_res {
            NewtonResult::Converged(iters, ee) => {
                
                // 1. SUNDIALS Local Truncation Error Test (IDATestError)
                let enorm_k = wrms_norm_mask(&ee, &weights, id, config.suppress_alg);
                let err_k = ck * enorm_k / history.sigma[history.order];

                let mut err_km1 = 0.0;
                let mut err_km2 = 0.0;

                if history.order > 1 {
                    let mut delta = vec![0.0; n];
                    for i in 0..n { delta[i] = history.phi[history.order][i] + ee[i]; }
                    let enorm_km1 = wrms_norm_mask(&delta, &weights, id, config.suppress_alg);
                    err_km1 = history.sigma[history.order - 1] * enorm_km1;

                    if history.order > 2 {
                        for i in 0..n { delta[i] += history.phi[history.order - 1][i]; }
                        let enorm_km2 = wrms_norm_mask(&delta, &weights, id, config.suppress_alg);
                        err_km2 = history.sigma[history.order - 2] * enorm_km2;
                    }
                }

                if err_k * ck > 1.0 { // Error test failed
                    error_fails += 1;
                    diag.rejected_steps += 1;
                    history.restore();
                    lu_solver.mark_stale();
                    
                    if error_fails == 1 {
                        let mut knew = history.order;
                        let mut err_knew = err_k;
                        
                        if history.order > 1 {
                            let terr_k = (history.order as f64 + 1.0) * err_k;
                            let terr_km1 = (history.order as f64) * err_km1;
                            
                            if terr_km1 <= 0.5 * terr_k {
                                knew = history.order - 1;
                                err_knew = err_km1;
                            }
                        }
                        
                        history.order = knew;
                        let eta = (0.9 * (2.0 * err_knew + 0.0001).powf(-1.0 / (history.order as f64 + 1.0))).clamp(0.25, 0.9);
                        history.h *= eta;
                    } else {
                        // Hard backoff on consecutive LTE failures
                        history.h *= 0.25;
                        history.order = 1;
                    }
                    
                    if history.h <= config.min_dt {
                        dump_crash_report(diag, y, ydot, id, "Tolerance Starvation: Truncation error > 1.0");
                        return Err(format!("Step collapsed below min_dt. t={}", abs_t + t_local));
                    }
                    continue;
                }

                // Success! Reset error_fails counter.
                error_fails = 0;
                
                // Evaluate Truncation Error at Order K+1
                let mut err_kp1 = 0.0;
                if history.order < history.max_order {
                    let mut delta = vec![0.0; n];
                    for i in 0..n { delta[i] = ee[i] - history.phi[history.order + 1][i]; }
                    let enorm_kp1 = wrms_norm_mask(&delta, &weights, id, config.suppress_alg);
                    err_kp1 = enorm_kp1 / ((history.order + 2) as f64);
                }

                history.complete_step(err_k, err_km1, err_km2, err_kp1, &ee, config.min_dt, config.max_dt);
                
                diag.accepted_steps += 1;
                diag.trace_t.push(abs_t + t_local); 
                diag.trace_dt.push(history.h_used);
                diag.trace_order.push(history.order);
                diag.trace_iters.push(iters);
                diag.trace_err.push(err_k);
                
                t_local += history.h_used;
                if let Some(ref mut hist) = history_cache {
                    hist.push((abs_t + t_local, y.to_vec(), ydot.to_vec()));
                }
            },
            NewtonResult::DivergedStaleJac(_) | NewtonResult::DivergedFatal(NewtonFailure::NonFiniteResidual) | NewtonResult::DivergedFatal(NewtonFailure::SingularJacobian(_)) | NewtonResult::DivergedFatal(NewtonFailure::ContractionThrashing(_)) | NewtonResult::DivergedFatal(NewtonFailure::MaxItersReached) => {
                diag.rejected_steps += 1;
                history.restore();
                history.h *= 0.25; // Standard SUNDIALS back-off ratio for Newton Failures
                lu_solver.mark_stale();

                if history.h <= config.min_dt {
                    dump_crash_report(diag, y, ydot, id, "Nonlinear Divergence");
                    return Err(format!("Step collapsed below min_dt. t={}", abs_t + t_local));
                }
            },
            NewtonResult::DivergedFatal(NewtonFailure::ConstraintsViolated(eta)) => {
                diag.rejected_steps += 1;
                history.restore();
                history.h *= eta; 
                lu_solver.mark_stale();

                if history.h <= config.min_dt {
                    dump_crash_report(diag, y, ydot, id, "Constraint Violation Starvation");
                    return Err(format!("Step collapsed below min_dt. t={}", abs_t + t_local));
                }
            }
        }
    }
    
    Ok(())
}

pub fn dump_crash_report(diag: &Diagnostics, y: &[f64], ydot: &[f64], id: &[f64], reason: &str) {
    std::fs::create_dir_all("ion_flux_diagnostics").ok();
    
    let mut offenders: Vec<(usize, f64, f64, f64, f64, bool)> = diag.last_res.iter().enumerate()
        .map(|(i, &res)| {
            let err = diag.last_dy.get(i).unwrap_or(&0.0) * diag.last_weights.get(i).unwrap_or(&0.0);
            let is_diff = id.get(i).unwrap_or(&0.0) > &0.5;
            let y_v = y.get(i).copied().unwrap_or(0.0);
            let ydot_v = ydot.get(i).copied().unwrap_or(0.0);
            (i, res, err.abs(), y_v, ydot_v, is_diff)
        }).collect();
    
    offenders.sort_by(|a, b| {
        let a_nan = !a.1.is_finite() || !a.3.is_finite() || !a.4.is_finite();
        let b_nan = !b.1.is_finite() || !b.3.is_finite() || !b.4.is_finite();
        if a_nan && !b_nan { return std::cmp::Ordering::Less; }
        if !a_nan && b_nan { return std::cmp::Ordering::Greater; }
        b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    let top_offenders: Vec<String> = offenders.into_iter().take(15).map(|(i, res, err, y_v, ydot_v, is_diff)| {
        let eq_type = if is_diff { "ODE/PDE" } else { "Algebraic" };
        format!("{{\"index\": {}, \"type\": \"{}\", \"y_val\": {:.3e}, \"ydot_val\": {:.3e}, \"residual\": {:.3e}, \"weighted_error\": {:.3e}}}", i, eq_type, y_v, ydot_v, res, err)
    }).collect();

    let mut volatile: Vec<(usize, f64)> = diag.last_dy.iter().enumerate()
        .map(|(i, &dy)| (i, dy.abs()))
        .collect();
    volatile.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    let top_volatile: Vec<String> = volatile.into_iter().take(5).map(|(i, dy)| {
        format!("{{\"index\": {}, \"dy_abs\": {:.3e}}}", i, dy)
    }).collect();

    let mut history_str = String::new();
    let hist_len = diag.trace_t.len();
    let start_idx = hist_len.saturating_sub(5);
    for i in start_idx..hist_len {
        history_str.push_str(&format!("{{\"step\": {}, \"t\": {:.3e}, \"dt\": {:.3e}, \"order\": {}, \"err_norm\": {:.3e}}}",
            i, diag.trace_t[i], diag.trace_dt[i], diag.trace_order[i], diag.trace_err[i]));
        if i < hist_len - 1 { history_str.push_str(",\n    "); }
    }

    let ts = Diagnostics::generate_timestamp();
    let filename = format!("ion_flux_diagnostics/crash_{}.json", ts);
    
    if let Ok(mut file) = File::create(&filename) {
        let json = format!(
            "{{\n  \"status\": \"CRASH\",\n  \"reason\": \"{}\",\n  \"accepted_steps\": {},\n  \"recent_history\": [\n    {}\n  ],\n  \"top_volatile_states\": [\n    {}\n  ],\n  \"top_offenders\": [\n    {}\n  ]\n}}",
            reason, diag.accepted_steps, history_str, top_volatile.join(",\n    "), top_offenders.join(",\n    ")
        );
        file.write_all(json.as_bytes()).ok();
    }
}

pub fn dump_diagnostics(diag: &Diagnostics) {
    std::fs::create_dir_all("ion_flux_diagnostics").ok();
    let ts = Diagnostics::generate_timestamp();
    if let Ok(mut file) = File::create(format!("ion_flux_diagnostics/profile_{}.json", ts)) {
        let json = format!(
            "{{\n  \"status\": \"SUCCESS\",\n  \"accepted_steps\": {},\n  \"rejected_steps\": {},\n  \"newton_iterations\": {},\n  \"jacobian_evals\": {},\n  \"numeric_lus\": {},\n  \"timers_us\": {{\n    \"residual\": {},\n    \"jac_assembly\": {},\n    \"lu_solve\": {}\n  }}\n}}",
            diag.accepted_steps, diag.rejected_steps, diag.newton_iterations,
            diag.jacobian_evaluations, diag.numeric_factorizations,
            diag.residual_time_us, diag.jacobian_assembly_time_us, diag.linear_solve_time_us
        );
        file.write_all(json.as_bytes()).ok();
    }
}