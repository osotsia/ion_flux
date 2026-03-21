mod solver;

use pyo3::prelude::*;
use solver::ida::solve_ida_native;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_ida_native, m)?)?;

    Ok(())
}