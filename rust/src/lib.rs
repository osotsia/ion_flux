mod solver_sundials;
mod solver_native;

use pyo3::prelude::*;
use solver_sundials::ida::solve_ida_sundials;
use solver_native::ida::solve_ida_native;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_ida_sundials, m)?)?;
    m.add_function(wrap_pyfunction!(solve_ida_native, m)?)?;

    Ok(())
}