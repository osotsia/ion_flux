pub mod session_handle;
pub mod batch_dispatcher;
pub mod adjoint_dispatcher;

pub use session_handle::*;
pub use batch_dispatcher::*;
pub use adjoint_dispatcher::*;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use crate::diagnostics::CrashReport;
use crate::diagnostics::Diagnostics;

pyo3::create_exception!(_core, NativeSolverCrash, pyo3::exceptions::PyRuntimeError);

pub fn diagnostics_to_pydict<'py>(py: Python<'py>, diag: &Diagnostics) -> Bound<'py, PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("total_steps", diag.total_steps).unwrap();
    dict.set_item("accepted_steps", diag.accepted_steps).unwrap();
    dict.set_item("rejected_steps", diag.rejected_steps).unwrap();
    let rej_ratio = if diag.total_steps > 0 {
        diag.rejected_steps as f64 / diag.total_steps as f64
    } else {
        0.0
    };
    dict.set_item("step_rejection_ratio", rej_ratio).unwrap();
    dict.set_item("newton_iterations", diag.newton_iterations).unwrap();
    let newton_per_step = if diag.total_steps > 0 {
        diag.newton_iterations as f64 / diag.total_steps as f64
    } else {
        0.0
    };
    dict.set_item("newton_iterations_per_step", newton_per_step).unwrap();
    dict.set_item("jacobian_evaluations", diag.jacobian_evaluations).unwrap();
    dict.set_item("numeric_factorizations", diag.numeric_factorizations).unwrap();
    dict.set_item("symbolic_factorizations", diag.symbolic_factorizations).unwrap();
    dict.set_item("jacobian_assembly_time_us", diag.jacobian_assembly_time_us as u64).unwrap();
    dict.set_item("linear_solve_time_us", diag.linear_solve_time_us as u64).unwrap();
    dict.set_item("residual_time_us", diag.residual_time_us as u64).unwrap();
    dict.set_item("last_rho", diag.last_rho).unwrap();
    dict.set_item("peak_rss_bytes", diag.peak_rss_bytes).unwrap();
    dict.set_item("cpu_cycles", diag.cpu_cycles).unwrap();
    dict.set_item("instructions", diag.instructions).unwrap();
    dict
}

#[derive(Debug)]
pub enum SolverError {
    Crash(CrashReport),
    Message(String),
}

impl From<String> for SolverError {
    fn from(s: String) -> Self { SolverError::Message(s) }
}

impl From<CrashReport> for SolverError {
    fn from(r: CrashReport) -> Self { SolverError::Crash(r) }
}

pub fn crash_report_to_pydict<'py>(py: Python<'py>, report: &CrashReport) -> Bound<'py, PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("status", "CRASH").unwrap();
    dict.set_item("reason", &report.reason).unwrap();
    dict.set_item("accepted_steps", report.accepted_steps).unwrap();
    
    let init_health = PyDict::new(py);
    init_health.set_item("t0_max_residual", report.t0_max_res).unwrap();
    init_health.set_item("t0_max_residual_index", report.t0_max_res_idx).unwrap();
    dict.set_item("initialization_health", init_health).unwrap();
    
    let jac_health = PyDict::new(py);
    jac_health.set_item("max_element", report.jac_max).unwrap();
    jac_health.set_item("min_nonzero_element", report.jac_min).unwrap();
    jac_health.set_item("condition_warning", report.cond_warning).unwrap();
    dict.set_item("jacobian_health", jac_health).unwrap();
    
    let trace_list = PyList::empty(py);
    for &(iter, fnorm, dynorm) in &report.trace {
        let t_dict = PyDict::new(py);
        t_dict.set_item("iter", iter).unwrap();
        t_dict.set_item("residual_norm", fnorm).unwrap();
        t_dict.set_item("step_norm", dynorm).unwrap();
        trace_list.append(&t_dict).unwrap();
    }
    dict.set_item("newton_thrashing_trace", trace_list).unwrap();
    
    let off_list = PyList::empty(py);
    for off in &report.offenders {
        let o_dict = PyDict::new(py);
        o_dict.set_item("index", off.index).unwrap();
        o_dict.set_item("type", if off.is_diff { "ODE/PDE" } else { "Algebraic" }).unwrap();
        o_dict.set_item("y_val", off.y_val).unwrap();
        o_dict.set_item("ydot_val", off.ydot_val).unwrap();
        o_dict.set_item("residual", off.residual).unwrap();
        o_dict.set_item("proposed_step_dy", off.proposed_step_dy).unwrap();
        o_dict.set_item("solver_weight", off.solver_weight).unwrap();
        o_dict.set_item("weighted_error", off.weighted_error).unwrap();
        off_list.append(&o_dict).unwrap();
    }
    dict.set_item("top_offenders", off_list).unwrap();
    
    dict
}
