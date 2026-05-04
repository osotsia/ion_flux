use super::callbacks::*;

#[derive(Clone, Default)]
pub struct CprData {
    pub color_seeds: Vec<Vec<f64>>,
    pub color_ptrs: Vec<usize>,
    pub color_rows: Vec<usize>,
    pub color_cols: Vec<usize>,
    pub dense_rows: Vec<usize>,
}

#[derive(Clone, Copy)]
pub struct SolverConfig {
    pub rel_tol: f64,
    pub abs_tol: f64,
    pub max_newton_iters: usize,
    pub min_dt: f64,
    pub max_dt: f64,
    pub max_rho: f64,
    pub eps_newt: f64,
    pub suppress_alg: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            rel_tol: 1e-6, abs_tol: 1e-8, max_newton_iters: 20, min_dt: 1e-12, max_dt: std::f64::INFINITY,
            max_rho: 0.9, eps_newt: 0.33, suppress_alg: true,
        }
    }
}

#[derive(Clone)]
pub struct Callbacks {
    pub res_fn: NativeResFn,
    pub obs_fn: Option<NativeObsFn>,
    pub jvp_fn: Option<NativeJvpFn>,
    pub vjp_fn: Option<NativeVjpFn>,
    pub set_threads_fn: Option<NativeSetThreadsFn>,
}

#[derive(Clone)]
pub struct Problem {
    pub n: usize,
    pub bw: isize,
    pub n_obs: usize,
    pub id: Vec<f64>,
    pub constraints: Vec<f64>,
    pub m: Vec<f64>,
    pub spatial_diag: Vec<f64>,
    pub max_steps: Vec<f64>,
    pub cpr: CprData,
    pub config: SolverConfig,
    pub fns: Callbacks,
}