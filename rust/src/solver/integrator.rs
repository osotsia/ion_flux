use super::{NativeResFn, NativeJacFn, NativeJvpFn};
use super::linalg::{solve_dense_system, solve_banded_system, solve_gmres};

pub struct BdfIntegrator {
    pub max_newton_iters: usize,
    pub min_dt: f64,
    pub rel_tol: f64,
}

impl Default for BdfIntegrator {
    fn default() -> Self { Self { max_newton_iters: 10, min_dt: 1e-10, rel_tol: 1e-6 } }
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
        target_dt: f64,
        res_fn: NativeResFn,
        jac_fn: NativeJacFn,
        jvp_fn: Option<NativeJvpFn>,
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
        let mut sub_dt = if cur_dt_prev > 0.0 { target_dt.min(cur_dt_prev * 1.5).max(self.min_dt) } else { target_dt.min(1e-3) };
        
        let mut res = vec![0.0; n];
        let mut jac = vec![0.0; if bw == -1 { 0 } else { n * n }];
        let mut dy = vec![0.0; n];
        let mut y_revert = vec![0.0; n]; 
        
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

            let mut converged = false;
            let mut iters = 0;
            
            for iter in 0..self.max_newton_iters {
                iters = iter;
                for i in 0..n { 
                    ydot[i] = if id[i] == 1.0 { c_j * y[i] - c_1 * cur_y_prev[i] + c_2 * cur_y_prev2[i] } else { 0.0 }; 
                }
                
                unsafe { res_fn(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), res.as_mut_ptr()) };

                let mut max_res = 0.0;
                for i in 0..n {
                    if res[i].abs() > max_res { max_res = res[i].abs(); }
                    dy[i] = -res[i];
                }
                
                if max_res < self.rel_tol { converged = true; break; }

                if bw == -1 {
                    let jvp = jvp_fn.expect("evaluate_jvp missing for Matrix-Free solver. Clear cache and recompile model.");
                    let y_ptr = y.as_ptr();
                    let ydot_ptr = ydot.as_ptr();
                    let p_ptr = p.as_ptr();
                    
                    let jvp_closure = |v: &[f64], out: &mut [f64]| {
                        unsafe { jvp(y_ptr, ydot_ptr, p_ptr, c_j, v.as_ptr(), out.as_mut_ptr()) };
                    };
                    
                    // Lumped Mass Matrix Preconditioner stabilizes highly stiff GMRES convergences automatically.
                    let precond = |v: &[f64], out: &mut [f64]| {
                        for i in 0..n { out[i] = v[i] / (c_j * id[i] + 1.0); }
                    };
                    
                    solve_gmres(n, &mut dy, jvp_closure, precond)?;
                } else {
                    unsafe { jac_fn(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), c_j, jac.as_mut_ptr()) };
                    if bw > 0 { solve_banded_system(n, bw as usize, &mut jac, &mut dy)?; } 
                    else { solve_dense_system(n, &mut jac, &mut dy)?; }
                }

                for i in 0..n { y[i] += dy[i]; }
            }
            
            if converged {
                t_local += sub_dt;
                cur_y_prev2.copy_from_slice(&cur_y_prev);
                cur_y_prev.copy_from_slice(y);
                cur_dt_prev = sub_dt;
                cur_order = 2; 

                if iters < 4 { sub_dt *= 1.5; } 
            } else {
                y.copy_from_slice(&y_revert);
                sub_dt *= 0.25; 
                if sub_dt < self.min_dt { return Err(format!("Newton method failed to converge. Timestep collapsed below {}.", self.min_dt)); }
            }
        }

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
    use std::os::raw::c_double;

    unsafe extern "C" fn mock_res(y: *const c_double, ydot: *const c_double, _p: *const c_double, res: *mut c_double) {
        *res = *ydot + *y;
    }

    unsafe extern "C" fn mock_jac(_y: *const c_double, _ydot: *const c_double, _p: *const c_double, c_j: c_double, jac: *mut c_double) {
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

        for _ in 0..10 {
            integrator.step(1, 0, &mut y, &mut ydot, &p, &id, 0.1, mock_res, mock_jac, None, 
                &mut y_prev, &mut y_prev2, &mut dt_prev, &mut order).unwrap();
        }
        
        let analytical_expected = std::f64::consts::E.powf(-1.0);
        assert!((y[0] - analytical_expected).abs() < 1e-2); 
    }
}