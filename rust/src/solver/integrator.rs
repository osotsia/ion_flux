use super::{NativeResFn, NativeJacFn};
use super::linalg::{solve_dense_system, solve_banded_system};

/// A highly modular Variable-Step BDF2 Integrator with Adaptive Step-Sizing.
/// Isolates numerical integration from state management and minimizes allocations.
pub struct BdfIntegrator {
    pub max_newton_iters: usize,
    pub min_dt: f64,
    pub rel_tol: f64,
}

impl Default for BdfIntegrator {
    fn default() -> Self {
        Self { max_newton_iters: 10, min_dt: 1e-10, rel_tol: 1e-6 }
    }
}

impl BdfIntegrator {
    /// Advances the system using an implicit variable-step BDF2 formulation.
    pub fn step(
        &self,
        n: usize,
        bw: usize,
        y: &mut [f64],
        ydot: &mut [f64],
        p: &[f64],
        id: &[f64],
        target_dt: f64,
        res_fn: NativeResFn,
        jac_fn: NativeJacFn,
        y_prev: &mut [f64],
        y_prev2: &mut [f64],
        dt_prev: &mut f64,
        order: &mut usize,
    ) -> Result<(), String> {
        let mut cur_y_prev = y_prev.to_vec();
        let mut cur_y_prev2 = y_prev2.to_vec();
        let mut cur_dt_prev = *dt_prev;
        let mut cur_order = *order;

        let mut t_local = 0.0;
        
        // Seed sub_dt seamlessly from the previous integration history to preserve 
        // acceleration across Python API boundaries.
        let mut sub_dt = if cur_dt_prev > 0.0 {
            target_dt.min(cur_dt_prev * 1.5).max(self.min_dt)
        } else {
            target_dt.min(1e-3)
        };
        
        let mut res = vec![0.0; n];
        let mut jac = vec![0.0; n * n];
        let mut dy = vec![0.0; n];
        let mut y_revert = vec![0.0; n]; 
        
        // Epsilon guard (1e-8) prevents microscopic remainder steps from causing Jacobian collapse
        while target_dt - t_local > 1e-8 {
            if t_local + sub_dt > target_dt { 
                sub_dt = target_dt - t_local; 
            }
            
            // Dynamic BDF coefficients adapted for shifting time-steps (r = h_n / h_{n-1})
            let r = if cur_dt_prev > 0.0 { sub_dt / cur_dt_prev } else { 1.0 };
            let (c_j, c_1, c_2) = if cur_order == 2 {
                (
                    (1.0 + 2.0 * r) / (sub_dt * (1.0 + r)),
                    (1.0 + r) / sub_dt,
                    (r * r) / (sub_dt * (1.0 + r))
                )
            } else {
                (1.0 / sub_dt, 1.0 / sub_dt, 0.0)
            };
            
            y_revert.copy_from_slice(y);

            // BDF2 Predictor: Variable-step linear extrapolation for a highly accurate 
            // Newton initial guess. Prevents overshooting across stiff non-linear kinks.
            if cur_order == 2 {
                for i in 0..n {
                    if id[i] == 1.0 {
                        y[i] = cur_y_prev[i] + r * (cur_y_prev[i] - cur_y_prev2[i]);
                    }
                }
            }

            let mut converged = false;
            let mut iters = 0;
            
            // Implicit Newton-Raphson Loop
            for iter in 0..self.max_newton_iters {
                iters = iter;
                for i in 0..n { 
                    ydot[i] = if id[i] == 1.0 { 
                        c_j * y[i] - c_1 * cur_y_prev[i] + c_2 * cur_y_prev2[i]
                    } else { 0.0 }; 
                }
                
                unsafe { res_fn(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), res.as_mut_ptr()) };

                let mut max_res = 0.0;
                for i in 0..n {
                    if res[i].abs() > max_res { max_res = res[i].abs(); }
                    dy[i] = -res[i];
                }
                
                if max_res < self.rel_tol { 
                    converged = true; break; 
                }

                unsafe { jac_fn(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), c_j, jac.as_mut_ptr()) };
                
                if bw > 0 { solve_banded_system(n, bw, &mut jac, &mut dy)?; } 
                else { solve_dense_system(n, &mut jac, &mut dy)?; }

                for i in 0..n { y[i] += dy[i]; }
            }
            
            // Step-Size Controller & History Propagation
            if converged {
                t_local += sub_dt;
                cur_y_prev2.copy_from_slice(&cur_y_prev);
                cur_y_prev.copy_from_slice(y);
                cur_dt_prev = sub_dt;
                cur_order = 2; // Safely ramp up to BDF2 after the first initial BDF1 step

                if iters < 4 { sub_dt *= 1.5; } 
            } else {
                y.copy_from_slice(&y_revert);
                sub_dt *= 0.25; 
                if sub_dt < self.min_dt {
                    return Err(format!("Newton method failed to converge. Timestep collapsed below {}.", self.min_dt));
                }
            }
        }

        // Commit trajectory state back out to the API wrapper
        y_prev.copy_from_slice(&cur_y_prev);
        y_prev2.copy_from_slice(&cur_y_prev2);
        *dt_prev = cur_dt_prev;
        *order = cur_order;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn mock_res(y: *const f64, ydot: *const f64, _p: *const f64, res: *mut f64) {
        // Simple ODE: dy/dt = -y  => res = ydot + y = 0
        *res = *ydot + *y;
    }

    unsafe extern "C" fn mock_jac(_y: *const f64, _ydot: *const f64, _p: *const f64, c_j: f64, jac: *mut f64) {
        // d(res)/dy = c_j * d(ydot)/d(ydot) + d(y)/dy = c_j + 1.0
        *jac = c_j + 1.0;
    }

    #[test]
    fn test_bdf2_variable_step_correctness() {
        let integrator = BdfIntegrator::default();
        let mut y = vec![1.0];
        let mut ydot = vec![0.0];
        let id = vec![1.0];
        let p = vec![];

        let mut y_prev = vec![1.0];
        let mut y_prev2 = vec![1.0];
        let mut dt_prev = 0.0;
        let mut order = 1;

        // Advance BDF2 history trajectory seamlessly
        // Use 10 explicit integer loops to cleanly bypass float accumulation errors.
        for _ in 0..10 {
            integrator.step(1, 0, &mut y, &mut ydot, &p, &id, 0.1, mock_res, mock_jac, 
                &mut y_prev, &mut y_prev2, &mut dt_prev, &mut order).unwrap();
        }
        
        let analytical_expected = std::f64::consts::E.powf(-1.0);
        
        // A tolerance of 1e-2 safely accounts for the mathematical global 
        // truncation error of BDF2 taking rapid target steps of 0.1.
        assert!((y[0] - analytical_expected).abs() < 1e-2); 
    }
}