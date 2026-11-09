pub mod ffi;
pub mod linear;
pub mod stepper;
pub mod nonlinear;
pub mod control;
pub mod diagnostics;
pub mod api;
pub mod vendor;
pub mod solver;

#[cfg(test)]
pub mod tests;

use pyo3::prelude::*;
use api::batch_dispatcher::{solve_ida_native, solve_batch_native};
use api::adjoint_dispatcher::discrete_adjoint_native;
use api::session_handle::SolverHandle;
use api::NativeSolverCrash;
use vendor::sundials::wrapper::{SundialsHandle, solve_ida_sundials};

#[pymodule]
fn _core(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("NativeSolverCrash", py.get_type::<NativeSolverCrash>())?;
    m.add_function(wrap_pyfunction!(solve_ida_native, m)?)?;
    m.add_function(wrap_pyfunction!(solve_ida_sundials, m)?)?;
    m.add_function(wrap_pyfunction!(solve_batch_native, m)?)?;
    m.add_function(wrap_pyfunction!(discrete_adjoint_native, m)?)?;
    m.add_class::<SolverHandle>()?;
    m.add_class::<SundialsHandle>()?;
    Ok(())
}