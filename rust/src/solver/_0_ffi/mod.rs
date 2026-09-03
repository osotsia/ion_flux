// rust/src/solver/_0_ffi/mod.rs
pub mod api_batch;
pub mod api_adjoint;
pub mod api_session;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use crate::solver::shared::diagnostics::CrashReport;

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