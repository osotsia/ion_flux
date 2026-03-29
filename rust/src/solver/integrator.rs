use std::fs::File;
use std::io::Write;
use super::{NativeResFn, NativeJacFn, NativeJvpFn, SolverConfig, Diagnostics};
use super::linalg::NativeSparseLuSolver;
use super::newton::{solve_nonlinear_system, NewtonResult, wrms_norm_diff};

/// Variable-Step, Variable-Order Nordsieck History Array.
/// Generates and scales integration states to seamlessly implement implicit BDF Orders 1 to 5.
pub struct NordsieckHistory {
    pub order: usize,
    pub z: Vec<Vec<f64>>, 
    pub l: Vec<f64>, // BDF Corrector coefficients
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
        // Fixed-Leading Coefficient (FLC) BDF values directly derived from interpolating polynomials
        self.l = match self.order {
            1 => vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            2 => vec![1.0, 3.0/2.0, 1.0/2.0, 0.0, 0.0, 0.0],
            3 => vec![1.0, 11.0/6.0, 1.0, 1.0/6.0, 0.0, 0.0],
            4 => vec![1.0, 25.0/12.0, 35.0/24.0, 5.0/12.0, 1.0/24.0, 0.0],
            5 => vec![1.0, 137.0/60.0, 15.0/8.0, 17.0/24.0, 5.0/24.0, 1.0/120.0],
            _ => vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        };
    }

    /// Extrapolates history forward using Pascal's triangle matrix equivalent.
    pub fn predict(&mut self) {
        for j in 1..=self.order {
            for i in (1..=self.order - j).rev() {
                for k in 0..self.z[0].len() {
                    self.z[i][k] += self.z[i + 1][k];
                }
            }
        }
    }

    /// Incorporates the Newton error update into the entire derivative history.
    pub fn correct(&mut self, e_n: &[f64]) {
        for j in 0..=self.order {
            let l_j = self.l[j];
            for k in 0..self.z[0].len() {
                self.z[j][k] += l_j * e_n[k];
            }
        }
    }

    /// Rescales history seamlessly without polynomial re-interpolation.
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

        let c_j = history.l[1] / *current_dt; // Dynamic scaling correctly shifts for BDF2-5

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
                // Evaluate Local Truncation Error (LTE). Critically, algebraic variables (id == 0.0) 
                // are explicitly ignored here to prevent Index-1 DAEs from crushing step scaling.
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
                        dump_diagnostics(diag);
                        return Err(format!("Step collapsed below min_dt ({})", config.min_dt));
                    }
                    continue;
                }

                history.correct(&e_n);
                
                diag.accepted_steps += 1;
                diag.trace_t.push(abs_t + t_local); 
                diag.trace_dt.push(sub_dt);
                diag.trace_order.push(history.order);
                diag.trace_iters.push(iters);
                
                t_local += sub_dt;
                if let Some(ref mut hist) = history_cache {
                    hist.push((abs_t + t_local, y.to_vec(), ydot.to_vec()));
                }
                
                // Aggressive Order Heuristic: Ramp order up rapidly during smooth segments
                // BDF2 provides immense immediate stability & step advantages over BDF1.
                if diag.accepted_steps % 5 == 0 && history.order < 5 {
                    history.set_order(history.order + 1);
                }
                
                sub_dt *= err_factor;
                *current_dt = sub_dt;
            },
            NewtonResult::DivergedStaleJac => {
                let mut e_rev = vec![0.0; n];
                for i in 0..n { e_rev[i] = y_pred[i] - history.z[0][i]; }
                history.correct(&e_rev); 
                lu_solver.mark_stale();
                continue;
            },
            NewtonResult::DivergedFatal => {
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
                    dump_diagnostics(diag);
                    return Err(format!("Step collapsed below min_dt ({})", config.min_dt));
                }
            }
        }
    }
    Ok(())
}

pub fn dump_diagnostics(diag: &Diagnostics) {
    std::fs::create_dir_all("ion_flux_diagnostics").ok();
    if let Ok(mut file) = File::create("ion_flux_diagnostics/solver_stats.json") {
        let json = format!(
            "{{\n  \"accepted_steps\": {},\n  \"rejected_steps\": {},\n  \"newton_iterations\": {},\n  \"jacobian_evals\": {},\n  \"numeric_lus\": {},\n  \"jac_time_us\": {},\n  \"solve_time_us\": {}\n}}",
            diag.accepted_steps, diag.rejected_steps, diag.newton_iterations,
            diag.jacobian_evaluations, diag.numeric_factorizations,
            diag.jacobian_assembly_time_us, diag.linear_solve_time_us
        );
        file.write_all(json.as_bytes()).ok();
    }
}