mod solver;

use pyo3::prelude::*;
use solver::ida::{solve_ida_native, solve_batch_native, discrete_adjoint_native, SolverHandle};

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_ida_native, m)?)?;
    m.add_function(wrap_pyfunction!(solve_batch_native, m)?)?;
    m.add_function(wrap_pyfunction!(discrete_adjoint_native, m)?)?;
    m.add_class::<SolverHandle>()?;

    Ok(())
}