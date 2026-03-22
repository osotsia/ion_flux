use super::{NativeResFn, NativeJacFn};
use super::linalg::{solve_dense_system, solve_banded_system};

/// A highly modular BDF1 (Backward Euler) Integrator with Adaptive Step-Sizing.
/// Isolates numerical integration from state management.
pub struct Bdf1Integrator {
    pub max_newton_iters: usize,
    pub min_dt: f64,
    pub rel_tol: f64,
}

impl Default for Bdf1Integrator {
    fn default() -> Self {
        Self { max_newton_iters: 10, min_dt: 1e-10, rel_tol: 1e-6 }
    }
}

impl Bdf1Integrator {
    /// Advances the system from current time to `target_dt` using adaptive sub-stepping.
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
    ) -> Result<(), String> {
        let mut t_local = 0.0;
        let mut sub_dt = target_dt.min(1e-3); // Seed adaptive stepper
        
        let mut res = vec![0.0; n];
        let mut jac = vec![0.0; n * n];
        let mut dy = vec![0.0; n];
        
        while t_local < target_dt {
            if t_local + sub_dt > target_dt { sub_dt = target_dt - t_local; }
            let c_j = 1.0 / sub_dt;
            
            let y_prev = y.to_vec();
            let mut converged = false;
            let mut iters = 0;
            
            // Implicit Newton-Raphson Loop
            for iter in 0..self.max_newton_iters {
                iters = iter;
                for i in 0..n { 
                    ydot[i] = if id[i] == 1.0 { (y[i] - y_prev[i]) / sub_dt } else { 0.0 }; 
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
            
            // Step-Size Controller
            if converged {
                t_local += sub_dt;
                if iters < 4 { sub_dt *= 1.5; } // Accelerate if convergence was easy
            } else {
                y.copy_from_slice(&y_prev);
                sub_dt *= 0.25; // Decelerate aggressively on failure
                if sub_dt < self.min_dt {
                    return Err(format!("Newton method failed to converge. Timestep collapsed below {}.", self.min_dt));
                }
            }
        }
        Ok(())
    }
}