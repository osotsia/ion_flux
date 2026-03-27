mod solver;

use pyo3::prelude::*;
use solver::bindings::{solve_ida_native, solve_ida_sundials, solve_batch_native};
use solver::adjoint::discrete_adjoint_native;
use solver::session::SolverHandle;
use solver::sundials::SundialsHandle;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_ida_native, m)?)?;
    m.add_function(wrap_pyfunction!(solve_ida_sundials, m)?)?;
    m.add_function(wrap_pyfunction!(solve_batch_native, m)?)?;
    m.add_function(wrap_pyfunction!(discrete_adjoint_native, m)?)?;
    m.add_class::<SolverHandle>()?;
    m.add_class::<SundialsHandle>()?;

    Ok(())
}