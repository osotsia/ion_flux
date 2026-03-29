use std::fs::File;
use std::io::Write;
use super::{NativeResFn, NativeJacFn, NativeJvpFn, SolverConfig, Diagnostics};
use super::linalg::NativeSparseLuSolver;
use super::newton::{solve_nonlinear_system, NewtonResult, NewtonFailure, wrms_norm_diff};

pub struct NordsieckHistory {
    pub order: usize,
    pub z: Vec<Vec<f64>>, 
    pub l: Vec<f64>, 
}

impl NordsieckHistory {
    pub fn new(n: usize) -> Self {
        let mut hist = Self {
            order: 1,
            z: vec![vec![0.0; n]; 6], 
            l: vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0], 
        };
        hist.set_order(1);
        hist
    }

    pub fn set_order(&mut self, order: usize) {
        self.order = order.clamp(1, 5);
        self.l = match self.order {
            1 => vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            2 => vec![1.0, 3.0/2.0, 1.0/2.0, 0.0, 0.0, 0.0],
            3 => vec![1.0, 11.0/6.0, 1.0, 1.0/6.0, 0.0, 0.0],
            4 => vec![1.0, 25.0/12.0, 35.0/24.0, 5.0/12.0, 1.0/24.0, 0.0],
            5 => vec![1.0, 137.0/60.0, 15.0/8.0, 17.0/24.0, 5.0/24.0, 1.0/120.0],
            _ => vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        };
    }

    pub fn predict(&mut self) {
        for j in 1..=self.order {
            for i in (1..=self.order - j).rev() {
                for k in 0..self.z[0].len() {
                    self.z[i][k] += self.z[i + 1][k];
                }
            }
        }
    }

    pub fn correct(&mut self, e_n: &[f64]) {
        for j in 0..=self.order {
            let l_j = self.l[j];
            for k in 0..self.z[0].len() {
                self.z[j][k] += l_j * e_n[k];
            }
        }
    }

    pub fn rescale(&mut self, ratio: f64) {
        let mut factor = 1.0;
        for j in 1..=self.order {
            factor *= ratio;
            for k in 0..self.z[0].len() {
                self.z[j][k] *= factor;
            }
        }
    }
}

pub fn step_bdf_vsvo(
    n: usize, bw: isize,
    y: &mut[f64], ydot: &mut[f64], p: &[f64], id: &[f64], spatial_diag: &[f64],
    target_dt: f64, current_dt: &mut f64,
    history: &mut NordsieckHistory,
    res_fn: NativeResFn, jac_fn: NativeJacFn, jvp_fn: Option<NativeJvpFn>,
    lu_solver: &mut NativeSparseLuSolver, jac_buffer: &mut[f64],
    config: &SolverConfig, diag: &mut Diagnostics,
    mut history_cache: Option<&mut Vec<(f64, Vec<f64>, Vec<f64>)>>,
    abs_t: f64,
) -> Result<(), String> {
    
    let mut t_local = 0.0;
    let mut sub_dt = if *current_dt > 0.0 { target_dt.min(*current_dt * 1.5).max(config.min_dt) } else { target_dt.min(1e-4) };

    if diag.accepted_steps == 0 {
        history.set_order(1);
    }

    while target_dt - t_local > 1e-8 {
        if t_local + sub_dt > target_dt { sub_dt = target_dt - t_local; }

        diag.total_steps += 1;

        if *current_dt == 0.0 {
            *current_dt = sub_dt;
            lu_solver.mark_stale();
        } else if (sub_dt - *current_dt).abs() > 1e-12 {
            history.rescale(sub_dt / *current_dt);
            lu_solver.mark_stale(); 
            *current_dt = sub_dt;
        }

        let c_j = history.l[1] / *current_dt;

        history.predict();
        let y_pred = history.z[0].clone();
        let mut ydot_pred = vec![0.0; n];
        for i in 0..n { ydot_pred[i] = history.z[1][i] / *current_dt; }

        let newton_res = solve_nonlinear_system(
            n, bw, y, ydot, p, id, spatial_diag, c_j,
            &y_pred, &ydot_pred, res_fn, jac_fn, jvp_fn,
            lu_solver, jac_buffer, config, diag
        );

        match newton_res {
            NewtonResult::Converged(iters, e_n) => {
                let mut weights = vec![0.0; n];
                for i in 0..n { weights[i] = 1.0 / (config.rel_tol * y[i].abs() + config.abs_tol); }
                let err_norm = wrms_norm_diff(&e_n, &weights, id);
                
                let safety = 0.9;
                let mut err_factor = if err_norm > 1e-10 { safety * (1.0 / err_norm).powf(1.0 / (history.order as f64 + 1.0)) } else { 2.0 };
                err_factor = err_factor.clamp(0.2, 2.0);

                if err_norm > 1.0 { 
                    diag.rejected_steps += 1;
                    let mut e_rev = vec![0.0; n];
                    for i in 0..n { e_rev[i] = y_pred[i] - history.z[0][i]; }
                    history.correct(&e_rev); 
                    lu_solver.mark_stale();
                    
                    let shrink_ratio = err_factor.max(0.25);
                    history.rescale(shrink_ratio);
                    *current_dt *= shrink_ratio;
                    sub_dt *= shrink_ratio;

                    if sub_dt < config.min_dt {
                        dump_crash_report(diag, id, "Tolerance Starvation: Error estimator rejected step repeatedly.");
                        return Err(format!("Step collapsed at t={} below min_dt ({}). Cause: Truncation error > 1.0", abs_t + t_local, config.min_dt));
                    }
                    continue;
                }

                history.correct(&e_n);
                
                diag.accepted_steps += 1;
                diag.trace_t.push(abs_t + t_local); 
                diag.trace_dt.push(sub_dt);
                diag.trace_order.push(history.order);
                diag.trace_iters.push(iters);
                diag.trace_err.push(err_norm);
                
                t_local += sub_dt;
                if let Some(ref mut hist) = history_cache {
                    hist.push((abs_t + t_local, y.to_vec(), ydot.to_vec()));
                }
                
                if diag.accepted_steps % 5 == 0 && history.order < 5 {
                    history.set_order(history.order + 1);
                }
                
                sub_dt *= err_factor;
                *current_dt = sub_dt;
            },
            NewtonResult::DivergedStaleJac(_) => {
                let mut e_rev = vec![0.0; n];
                for i in 0..n { e_rev[i] = y_pred[i] - history.z[0][i]; }
                history.correct(&e_rev); 
                lu_solver.mark_stale();
                continue;
            },
            NewtonResult::DivergedFatal(fail_reason) => {
                diag.rejected_steps += 1;
                let mut e_rev = vec![0.0; n];
                for i in 0..n { e_rev[i] = y_pred[i] - history.z[0][i]; }
                history.correct(&e_rev); 

                let shrink_ratio = 0.25;
                history.rescale(shrink_ratio);
                *current_dt *= shrink_ratio;
                sub_dt *= shrink_ratio;
                lu_solver.mark_stale();

                if sub_dt < config.min_dt {
                    let reason_str = match fail_reason {
                        NewtonFailure::NonFiniteResidual => "Mathematical Singularity (NaN/Inf in residual)".to_string(),
                        NewtonFailure::SingularJacobian(ref e) => format!("Structural Singularity: {}", e),
                        NewtonFailure::ContractionThrashing(rho) => format!("Nonlinear Thrashing (rho = {:.2})", rho),
                        NewtonFailure::F64Stagnation => "f64 Precision Stagnation (y + dy == y)".to_string(),
                        NewtonFailure::MaxItersReached => "Newton iterations stalled.".to_string(),
                    };
                    dump_crash_report(diag, id, &reason_str);
                    return Err(format!("Step collapsed at t={} below min_dt ({}). Cause: {}", abs_t + t_local, config.min_dt, reason_str));
                }
            }
        }
    }
    Ok(())
}

pub fn dump_crash_report(diag: &Diagnostics, id: &[f64], reason: &str) {
    std::fs::create_dir_all("ion_flux_diagnostics").ok();
    
    let mut offenders: Vec<(usize, f64, f64, f64, bool)> = diag.last_res.iter().enumerate()
        .map(|(i, &res)| {
            let err = diag.last_dy.get(i).unwrap_or(&0.0) * diag.last_weights.get(i).unwrap_or(&0.0);
            let is_diff = id.get(i).unwrap_or(&0.0) > &0.5;
            (i, res, err.abs(), *diag.last_y_pred.get(i).unwrap_or(&0.0), is_diff)
        }).collect();
    
    offenders.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
    
    let top_offenders: Vec<String> = offenders.into_iter().take(10).map(|(i, res, err, y, is_diff)| {
        let eq_type = if is_diff { "ODE/PDE" } else { "Algebraic" };
        format!("{{\"index\": {}, \"type\": \"{}\", \"y_val\": {:.3e}, \"residual\": {:.3e}, \"weighted_error\": {:.3e}}}", i, eq_type, y, res, err)
    }).collect();

    let ts = Diagnostics::generate_timestamp();
    let filename = format!("ion_flux_diagnostics/crash_{}.json", ts);
    
    if let Ok(mut file) = File::create(&filename) {
        let json = format!(
            "{{\n  \"status\": \"CRASH\",\n  \"reason\": \"{}\",\n  \"accepted_steps\": {},\n  \"top_offenders\": [\n    {}\n  ]\n}}",
            reason, diag.accepted_steps, top_offenders.join(",\n    ")
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