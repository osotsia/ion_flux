use super::{NativeResFn, NativeJacFn, NativeJvpFn};
use super::linalg::{solve_dense_system, solve_banded_system, solve_gmres};
use std::fs::{File, OpenOptions};
use std::io::Write;

pub struct BdfIntegrator {
    pub max_newton_iters: usize,
    pub min_dt: f64,
    pub rel_tol: f64,
    pub atol: f64,
    pub debug: bool, // Added observability toggle
}

impl Default for BdfIntegrator {
    fn default() -> Self { Self { max_newton_iters: 10, min_dt: 1e-10, rel_tol: 1e-6, atol: 1e-8, debug: false } }
}

impl BdfIntegrator {
    pub fn step(
        &self,
        n: usize,
        bw: isize,
        y: &mut [f64],
        ydot: &mut [f64],
        p: &[f64],
        id: &[f64],
        spatial_diag: &[f64],
        target_dt: f64,
        res_fn: NativeResFn,
        jac_fn: NativeJacFn,
        jvp_fn: Option<NativeJvpFn>,
        y_prev: &mut [f64],
        y_prev2: &mut [f64],
        dt_prev: &mut f64,
        order: &mut usize,
        mut history: Option<&mut Vec<(f64, Vec<f64>, Vec<f64>)>>,
        abs_t: f64,
    ) -> Result<(), String> {
        let mut cur_y_prev = y_prev.to_vec();
        let mut cur_y_prev2 = y_prev2.to_vec();
        let mut cur_dt_prev = *dt_prev;
        let mut cur_order = *order;

        let mut t_local = 0.0;
        let mut sub_dt = if cur_dt_prev > 0.0 { target_dt.min(cur_dt_prev * 1.5).max(self.min_dt) } else { target_dt.min(1e-3) };
        
        let mut res = vec![0.0; n];
        let mut jac = vec![0.0; if bw == -1 { 0 } else { n * n }];
        let mut dy = vec![0.0; n];
        let mut y_revert = vec![0.0; n]; 
        let mut weights = vec![0.0; n];
        
        let mut y_trial = vec![0.0; n];
        let mut ydot_trial = vec![0.0; n];
        let mut res_trial = vec![0.0; n];
        
        // Setup Diagnostic File
        let mut trace_file = if self.debug {
            std::fs::create_dir_all("ion_flux_diagnostics").ok();
            Some(OpenOptions::new().create(true).append(true).open("ion_flux_diagnostics/newton_trace.csv").unwrap())
        } else {
            None
        };

        if let Some(ref mut file) = trace_file {
            writeln!(file, "abs_t,sub_dt,iter,max_res_val,max_res_idx,f_norm,dy_norm,alpha").ok();
        }

        let wrms_norm = |v: &[f64], w: &[f64]| -> f64 {
            let mut sum = 0.0;
            for i in 0..n {
                let scaled = v[i] * w[i];
                if scaled.is_nan() || scaled.is_infinite() { return f64::INFINITY; }
                sum += scaled * scaled;
            }
            (sum / (n as f64)).sqrt()
        };
        
        while target_dt - t_local > 1e-8 {
            if t_local + sub_dt > target_dt { sub_dt = target_dt - t_local; }
            
            let r = if cur_dt_prev > 0.0 { sub_dt / cur_dt_prev } else { 1.0 };
            let (c_j, c_1, c_2) = if cur_order == 2 {
                ((1.0 + 2.0 * r) / (sub_dt * (1.0 + r)), (1.0 + r) / sub_dt, (r * r) / (sub_dt * (1.0 + r)))
            } else {
                (1.0 / sub_dt, 1.0 / sub_dt, 0.0)
            };
            
            y_revert.copy_from_slice(y);

            if cur_order == 2 {
                for i in 0..n {
                    if id[i] == 1.0 { y[i] = cur_y_prev[i] + r * (cur_y_prev[i] - cur_y_prev2[i]); }
                }
            }
            
            for i in 0..n {
                weights[i] = 1.0 / (self.rel_tol * y[i].abs() + self.atol);
            }

            let mut converged = false;
            let mut iters = 0;
            
            for iter in 0..self.max_newton_iters {
                iters = iter;
                for i in 0..n { 
                    ydot[i] = if id[i] == 1.0 { c_j * y[i] - c_1 * cur_y_prev[i] + c_2 * cur_y_prev2[i] } else { 0.0 }; 
                }
                
                unsafe { res_fn(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), res.as_mut_ptr()) };

                let mut max_abs_res = 0.0;
                let mut max_res_idx = 0;
                for i in 0..n {
                    let r_abs = res[i].abs();
                    if r_abs > max_abs_res || r_abs.is_nan() { 
                        max_abs_res = r_abs; 
                        max_res_idx = i;
                    }
                    dy[i] = -res[i];
                }

                if max_abs_res < 1e-12 && !max_abs_res.is_nan() { converged = true; break; }

                if bw == -1 {
                    let jvp = jvp_fn.expect("evaluate_jvp missing.");
                    let y_ptr = y.as_ptr();
                    let ydot_ptr = ydot.as_ptr();
                    let p_ptr = p.as_ptr();
                    
                    let jvp_closure = |v: &[f64], out: &mut [f64]| {
                        unsafe { jvp(y_ptr, ydot_ptr, p_ptr, c_j, v.as_ptr(), out.as_mut_ptr()) };
                    };
                    let precond = |v: &[f64], out: &mut [f64]| {
                        for i in 0..n { out[i] = v[i] / (c_j * id[i] + spatial_diag[i] + 1.0); }
                    };
                    // Pass linear algebra failures upward to trigger crash dumps
                    if let Err(e) = solve_gmres(n, &mut dy, jvp_closure, precond) {
                        return self.trigger_crash_dump(n, y, &res, &weights, format!("GMRES Failed: {}", e));
                    }
                } else {
                    unsafe { jac_fn(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), c_j, jac.as_mut_ptr()) };
                    let lin_res = if bw > 0 { solve_banded_system(n, bw as usize, &mut jac, &mut dy) } 
                                  else { solve_dense_system(n, &mut jac, &mut dy) };
                    if let Err(e) = lin_res {
                        return self.trigger_crash_dump(n, y, &res, &weights, format!("Linear Solve Failed: {}", e));
                    }
                }

                let dy_norm = wrms_norm(&dy, &weights);
                let f_norm = wrms_norm(&res, &weights);
                
                // Backtracking Line Search (Armijo Rule)
                let mut alpha = 1.0;
                let mut step_accepted = false;
                
                for _ in 0..5 {
                    for i in 0..n { 
                        y_trial[i] = y[i] + alpha * dy[i]; 
                        ydot_trial[i] = if id[i] == 1.0 { c_j * y_trial[i] - c_1 * cur_y_prev[i] + c_2 * cur_y_prev2[i] } else { 0.0 };
                    }
                    
                    unsafe { res_fn(y_trial.as_ptr(), ydot_trial.as_ptr(), p.as_ptr(), res_trial.as_mut_ptr()) };
                    let f_norm_trial = wrms_norm(&res_trial, &weights);
                    
                    // Safely bounds exponential terms (NaN tracking) and ensures monotonic descent
                    if f_norm_trial <= f_norm * (1.0 - 1e-4 * alpha) || dy_norm < 0.1 {
                        y.copy_from_slice(&y_trial);
                        step_accepted = true;
                        break;
                    }
                    alpha *= 0.5; 
                }

                if let Some(ref mut file) = trace_file {
                    writeln!(file, "{},{},{},{},{},{},{},{}", abs_t + t_local, sub_dt, iter, max_abs_res, max_res_idx, f_norm, dy_norm, alpha).ok();
                    file.flush().ok();
                }

                if !step_accepted { break; }
                if dy_norm < 1.0 { converged = true; break; }
            }
            
            if converged {
                t_local += sub_dt;
                if let Some(ref mut hist) = history {
                    hist.push((abs_t + t_local, y.to_vec(), ydot.to_vec()));
                }
                cur_y_prev2.copy_from_slice(&cur_y_prev);
                cur_y_prev.copy_from_slice(y);
                cur_dt_prev = sub_dt;
                cur_order = 2; 

                if iters < 4 { sub_dt *= 1.5; } 
            } else {
                y.copy_from_slice(&y_revert);
                sub_dt *= 0.25; 
                if sub_dt < self.min_dt { 
                    return self.trigger_crash_dump(n, y, &res, &weights, format!("Timestep collapsed below {}.", self.min_dt));
                }
            }
        }

        y_prev.copy_from_slice(&cur_y_prev);
        y_prev2.copy_from_slice(&cur_y_prev2);
        *dt_prev = cur_dt_prev;
        *order = cur_order;
        Ok(())
    }

    /// Pre-mortem JSON dump to persist the exact array shapes that killed the solver.
    fn trigger_crash_dump(&self, n: usize, y: &[f64], res: &[f64], weights: &[f64], error_msg: String) -> Result<(), String> {
        if self.debug {
            std::fs::create_dir_all("ion_flux_diagnostics").ok();
            if let Ok(mut file) = File::create("ion_flux_diagnostics/crash_dump.json") {
                let y_str = y.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
                let res_str = res.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
                let w_str = weights.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
                
                writeln!(file, "{{\n  \"error\": \"{}\",\n  \"n\": {},\n  \"y\": [{}],\n  \"res\": [{}],\n  \"weights\": [{}]\n}}", 
                         error_msg, n, y_str, res_str, w_str).ok();
                file.flush().ok();
            }
        }
        Err(error_msg)
    }
}