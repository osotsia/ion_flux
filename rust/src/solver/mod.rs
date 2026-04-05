pub mod linalg;
pub mod newton;
pub mod integrator;
pub mod session;
pub mod adjoint;
pub mod bindings;
pub mod sundials;

use std::os::raw::c_double;
use std::time::SystemTime;

pub type NativeResFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *const c_double, *mut c_double);
pub type NativeJacFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *const c_double, c_double, *mut c_double);
pub type NativeJvpFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *const c_double, c_double, *const c_double, *mut c_double);
pub type NativeVjpFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *const c_double, *const c_double, *mut c_double, *mut c_double, *mut c_double);

#[derive(Clone, Copy)]
pub struct SolverConfig {
    pub rel_tol: f64,
    pub abs_tol: f64,
    pub max_newton_iters: usize,
    pub min_dt: f64,
    pub max_dt: f64,
    
    // SUNDIALS-Specific Internal Heuristics
    pub max_rho: f64,             // Early divergence rate threshold (0.9)
    pub eps_newt: f64,            // Newton convergence constant (0.33)
    pub max_cj_ratio_change: f64, // Jacobian staleness threshold (0.25)
    pub suppress_alg: bool,       // Exclude algebraic variables from truncation error tests
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            rel_tol: 1e-6,
            abs_tol: 1e-8,
            max_newton_iters: 20, // Increased to 20 to allow step-clamping to walk down steep cliffs
            min_dt: 1e-12,
            max_dt: std::f64::INFINITY,
            max_rho: 0.9, 
            eps_newt: 0.33,
            max_cj_ratio_change: 0.25,
            suppress_alg: true,   // Crucial for DAE robustness!
        }
    }
}

/// Zero-overhead telemetry recorded during the solve. 
/// Dumped to JSON post-execution for observability.
#[derive(Default, Clone)]
pub struct Diagnostics {
    // --- Global Counters ---
    pub total_steps: usize,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub newton_iterations: usize,
    pub jacobian_evaluations: usize,
    pub numeric_factorizations: usize,
    pub max_chromatic_number: usize,
    
    // --- Microsecond Timers ---
    pub jacobian_assembly_time_us: u128,
    pub linear_solve_time_us: u128,
    pub residual_time_us: u128,
    
    // --- Trajectory Trace (Performance Profiling) ---
    pub trace_t: Vec<f64>,
    pub trace_dt: Vec<f64>,
    pub trace_order: Vec<usize>,
    pub trace_iters: Vec<usize>,
    pub trace_err: Vec<f64>,
    
    // --- Crash Evidence Cache ---
    pub last_y_pred: Vec<f64>,
    pub last_ydot_pred: Vec<f64>,
    pub last_res: Vec<f64>,
    pub last_dy: Vec<f64>,
    pub last_weights: Vec<f64>,
    pub last_rho: f64,
}

impl Diagnostics {
    pub fn generate_timestamp() -> u64 {
        SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()
    }
}