mod compiler {
    pub mod ast;
    pub mod codegen_cpu;
}

use pyo3::prelude::*;
use compiler::ast::Equation;
use compiler::codegen_cpu::generate_enzyme_cpp;

#[pyfunction]
fn compile_to_cpp(ast_json: String) -> PyResult<String> {
    // 1. Deserialize Python AST
    let equations: Vec<Equation> = serde_json::from_str(&ast_json)
        .expect("Failed to parse AST JSON");
    
    // 2. Lower to Enzyme C++
    let cpp_code = generate_enzyme_cpp(&equations);
    
    Ok(cpp_code)
}

// In PyO3 >= 0.21, modules are passed as a Bound reference to manage GIL lifetimes safely.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile_to_cpp, m)?)?;
    Ok(())
}