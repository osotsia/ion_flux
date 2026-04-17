// File: rust/src/solver/mod.rs

pub mod linalg;
pub mod newton;
pub mod integrator;
pub mod session;
pub mod adjoint;
pub mod bindings;
pub mod sundials;

use std::os::raw::{c_double, c_int};
use std::time::SystemTime;

pub type NativeResFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *const c_double, *mut c_double);
pub type NativeObsFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *const c_double, *mut c_double);
pub type NativeJacFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *const c_double, c_double, *mut c_double);
pub type NativeJvpFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *const c_double, c_double, *const c_double, *mut c_double);
pub type NativeVjpFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *const c_double, *const c_double, *mut c_double, *mut c_double, *mut c_double);
pub type NativeSetThreadsFn = unsafe extern "C" fn(c_int);

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
#[derive(Clone)]
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
    
    // --- Enhanced Diagnostics ---
    pub jac_max: f64,
    pub jac_min: f64,
    pub t0_max_res: f64,
    pub t0_max_res_idx: usize,
    pub recent_newton_norms: std::collections::VecDeque<(usize, f64, f64)>, // (Iter, F_norm, dy_norm)
}

impl Default for Diagnostics {
    fn default() -> Self {
        Self {
            total_steps: 0, accepted_steps: 0, rejected_steps: 0, newton_iterations: 0,
            jacobian_evaluations: 0, numeric_factorizations: 0, max_chromatic_number: 0,
            jacobian_assembly_time_us: 0, linear_solve_time_us: 0, residual_time_us: 0,
            trace_t: Vec::new(), trace_dt: Vec::new(), trace_order: Vec::new(), trace_iters: Vec::new(), trace_err: Vec::new(),
            last_y_pred: Vec::new(), last_ydot_pred: Vec::new(), last_res: Vec::new(), last_dy: Vec::new(), last_weights: Vec::new(), last_rho: 0.0,
            jac_max: 0.0, jac_min: 0.0, t0_max_res: 0.0, t0_max_res_idx: 0, recent_newton_norms: std::collections::VecDeque::new(),
        }
    }
}

impl Diagnostics {
    pub fn generate_timestamp() -> u64 {
        SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()
    }
}