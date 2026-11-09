use crate::ffi::callbacks::Callbacks;

/// Encapsulates raw C-ABI function pointers, enforcing slice boundary invariants
/// before dispatching into compiled native library binaries.
pub struct SafeEvaluator {
    fns: Callbacks,
    pub n_states: usize,
    pub n_params: usize,
    pub n_obs: usize,
}

impl SafeEvaluator {
    pub fn new(fns: Callbacks, n_states: usize, n_params: usize, n_obs: usize) -> Self {
        Self {
            fns,
            n_states,
            n_params,
            n_obs,
        }
    }

    #[inline]
    pub fn evaluate_residual(&self, y: &[f64], ydot: &[f64], p: &[f64], m: &[f64], res: &mut [f64]) {
        assert_eq!(y.len(), self.n_states, "y slice length must equal n_states");
        assert_eq!(ydot.len(), self.n_states, "ydot slice length must equal n_states");
        assert_eq!(res.len(), self.n_states, "res slice length must equal n_states");
        assert_eq!(p.len(), self.n_params, "p slice length must equal n_params");

        unsafe {
            (self.fns.res_fn)(
                y.as_ptr(),
                ydot.as_ptr(),
                p.as_ptr(),
                m.as_ptr(),
                res.as_mut_ptr(),
            );
        }
    }

    #[inline]
    pub fn evaluate_observables(&self, y: &[f64], ydot: &[f64], p: &[f64], m: &[f64], obs: &mut [f64]) {
        assert_eq!(y.len(), self.n_states, "y slice length must equal n_states");
        assert_eq!(ydot.len(), self.n_states, "ydot slice length must equal n_states");
        assert_eq!(obs.len(), self.n_obs, "obs slice length must equal n_obs");
        assert_eq!(p.len(), self.n_params, "p slice length must equal n_params");

        if let Some(obs_fn) = self.fns.obs_fn {
            unsafe {
                obs_fn(
                    y.as_ptr(),
                    ydot.as_ptr(),
                    p.as_ptr(),
                    m.as_ptr(),
                    obs.as_mut_ptr(),
                );
            }
        }
    }

    #[inline]
    pub fn evaluate_jvp(
        &self,
        y: &[f64],
        ydot: &[f64],
        p: &[f64],
        m: &[f64],
        c_j: f64,
        v: &[f64],
        out: &mut [f64],
    ) {
        assert_eq!(y.len(), self.n_states, "y slice length must equal n_states");
        assert_eq!(ydot.len(), self.n_states, "ydot slice length must equal n_states");
        assert_eq!(v.len(), self.n_states, "v slice length must equal n_states");
        assert_eq!(out.len(), self.n_states, "out slice length must equal n_states");
        assert_eq!(p.len(), self.n_params, "p slice length must equal n_params");

        let jvp = self.fns.jvp_fn.expect("evaluate_jvp function pointer missing");
        unsafe {
            jvp(
                y.as_ptr(),
                ydot.as_ptr(),
                p.as_ptr(),
                m.as_ptr(),
                c_j,
                v.as_ptr(),
                out.as_mut_ptr(),
            );
        }
    }

    #[inline]
    pub fn evaluate_vjp(
        &self,
        y: &[f64],
        ydot: &[f64],
        p: &[f64],
        m: &[f64],
        lambda_vec: &[f64],
        dp_out: &mut [f64],
        dy_out: &mut [f64],
        dydot_out: &mut [f64],
    ) {
        assert_eq!(y.len(), self.n_states, "y slice length must equal n_states");
        assert_eq!(ydot.len(), self.n_states, "ydot slice length must equal n_states");
        assert_eq!(lambda_vec.len(), self.n_states, "lambda_vec slice length must equal n_states");
        assert_eq!(dp_out.len(), self.n_params, "dp_out slice length must equal n_params");
        assert_eq!(dy_out.len(), self.n_states, "dy_out slice length must equal n_states");
        assert_eq!(dydot_out.len(), self.n_states, "dydot_out slice length must equal n_states");
        assert_eq!(p.len(), self.n_params, "p slice length must equal n_params");

        let vjp = self.fns.vjp_fn.expect("evaluate_vjp function pointer missing");
        unsafe {
            vjp(
                y.as_ptr(),
                ydot.as_ptr(),
                p.as_ptr(),
                m.as_ptr(),
                lambda_vec.as_ptr(),
                dp_out.as_mut_ptr(),
                dy_out.as_mut_ptr(),
                dydot_out.as_mut_ptr(),
            );
        }
    }

    #[inline]
    pub fn set_spatial_threads(&self, num_threads: i32) {
        if let Some(f) = self.fns.set_threads_fn {
            unsafe { f(num_threads); }
        }
    }

    #[inline]
    pub fn raw_callbacks(&self) -> &Callbacks {
        &self.fns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::mock_c_abi::*;

    #[test]
    fn test_safe_evaluator_valid_execution() {
        let (prob, ..) = create_mock_fixture();
        let evaluator = SafeEvaluator::new(prob.fns, MOCK_N, MOCK_N_PARAMS, MOCK_N_OBS);

        let y = [1.0, 1.0, 1.0, 1.0, 1.0, 6.0];
        let ydot = [0.0; MOCK_N];
        let p = [0.5, 1.0];
        let m = [1.0; 10];
        let mut res = [0.0; MOCK_N];

        evaluator.evaluate_residual(&y, &ydot, &p, &m, &mut res);

        assert!((res[0] - (2.5 * 1.0 - 1.0)).abs() < 1e-12);
        assert!((res[5] - (6.0 - 5.0 - 1.0)).abs() < 1e-12);

        let mut obs = [0.0; MOCK_N_OBS];
        evaluator.evaluate_observables(&y, &ydot, &p, &m, &mut obs);
        assert_eq!(obs[0], 1.0);
        assert_eq!(obs[1], 6.0);
    }

    #[test]
    fn test_safe_evaluator_slice_mismatch_panics() {
        let (prob, ..) = create_mock_fixture();
        let evaluator = SafeEvaluator::new(prob.fns, MOCK_N, MOCK_N_PARAMS, MOCK_N_OBS);

        let y_short = [1.0; 5]; // Mismatched: 5 vs 6
        let ydot = [0.0; MOCK_N];
        let p = [0.5, 1.0];
        let m = [1.0; 10];
        let mut res = [0.0; MOCK_N];

        let result_y = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            evaluator.evaluate_residual(&y_short, &ydot, &p, &m, &mut res);
        }));
        assert!(result_y.is_err(), "Expected panic on mismatched y slice length");

        let y = [1.0; MOCK_N];
        let mut res_short = [0.0; 4]; // Mismatched: 4 vs 6
        let result_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            evaluator.evaluate_residual(&y, &ydot, &p, &m, &mut res_short);
        }));
        assert!(result_res.is_err(), "Expected panic on mismatched res slice length");

        let p_short = [0.5]; // Mismatched: 1 vs 2
        let result_p = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            evaluator.evaluate_residual(&y, &ydot, &p_short, &m, &mut res);
        }));
        assert!(result_p.is_err(), "Expected panic on mismatched p slice length");
    }
}
