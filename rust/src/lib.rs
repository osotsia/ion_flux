mod solver;

use pyo3::prelude::*;
use solver::_0_ffi::api_batch::{solve_ida_native, solve_batch_native};
use solver::_0_ffi::api_adjoint::discrete_adjoint_native;
use solver::_0_ffi::api_session::SolverHandle;
use solver::sundials::wrapper::{SundialsHandle, solve_ida_sundials};

pyo3::create_exception!(_core, NativeSolverCrash, pyo3::exceptions::PyRuntimeError);

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